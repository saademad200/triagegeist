# Triagegeist: Acuity Prediction & Demographic Bias Audit
### Laitinen-Fredriksson Foundation — AI in Emergency Triage Competition

**Author:** [Your Name] | **Date:** May 2026

---

## Summary
This notebook builds an XGBoost classifier predicting emergency triage acuity (ESI 1–5) 
from patient intake data, then conducts a systematic demographic bias audit identifying 
which patient groups are systematically undertriaged.

**Key Results:**
- 85.9% validation accuracy, 0.86 weighted F1
- 96% recall on ESI 1 (critical) patients
- Overall undertriage rate: 5.27%
- Highest risk group: Pediatric / Somali-speaking patients at 7.2% undertriage rate
- Elderly / Russian-speaking patients at 6.6% — 25% above overall rate

## Dataset
Synthetic ED dataset provided by the Laitinen-Fredriksson Foundation. 
80,000 training patients, 20,000 test patients across a simulated Finnish 
multi-site hospital network. Features include vital signs, demographics, 
chief complaint system, and 25 binary comorbidity flags.

## Approach
1. **EDA** — clinical validity check, demographic analysis, missingness patterns
2. **Modeling** — XGBoost multiclass classifier with early stopping
3. **Interpretability** — SHAP values connecting model behavior to clinical signals
4. **Bias Audit** — undertriage rate analysis across language, insurance, sex, and age group

---

## 1. Data Loading


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)

# Data paths
PATH = '/kaggle/input/competitions/triagegeist/'

# Load all datasets
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')
complaints = pd.read_csv(PATH + 'chief_complaints.csv')
history = pd.read_csv(PATH + 'patient_history.csv')

print("=== DATA LOADED ===")
print(f"Train:      {train.shape}")
print(f"Test:       {test.shape}")
print(f"Complaints: {complaints.shape}")
print(f"History:    {history.shape}")
```

All four competition files loaded successfully. Patient history (25 comorbidity flags) 
will be merged onto train and test as additional features.

## 2. Feature Engineering & Preprocessing
Merge comorbidities, drop leakage columns, fix pain score encoding, encode categoricals, split train/validation.


```python
# Save demographics before encoding
demo_cols = ['patient_id', 'sex', 'language', 'insurance_type', 
             'age_group', 'triage_acuity']
train_demo = train[demo_cols].copy()

# Merge comorbidity history
train = train.merge(history, on='patient_id', how='left')
test = test.merge(history, on='patient_id', how='left')

# Drop leakage and identifier columns
drop_cols = ['patient_id', 'site_id', 'triage_nurse_id',
             'disposition', 'ed_los_hours', 'age_group']
drop_cols = [c for c in drop_cols if c in train.columns]
train = train.drop(columns=drop_cols)
test = test.drop(columns=[c for c in drop_cols if c in test.columns])

# Fix pain score
train['pain_score'] = train['pain_score'].replace(-1, np.nan)
test['pain_score'] = test['pain_score'].replace(-1, np.nan)

# Encode categoricals
cat_cols = train.select_dtypes(include='object').columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    le_dict[col] = le

# Split features and target
TARGET = 'triage_acuity'
FEATURES = [c for c in train.columns if c != TARGET]
X = train[FEATURES]
y = train[TARGET] - 1

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples:   {X_train.shape[0]:,}")
print(f"Validation samples: {X_val.shape[0]:,}")
print(f"Features:           {X_train.shape[1]}")
```

## 3. Model Training — XGBoost Classifier
500 tree maximum with early stopping monitoring validation log-loss. Uses all 16 CPU cores.


```python
# Train XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=5,
    eval_metric='mlogloss',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print(f"\nBest iteration: {model.best_iteration}")
```

## 4. Model Evaluation


```python
# Evaluate on validation set
y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='weighted')

print(f"Validation Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
print(f"Weighted F1 Score:    {f1:.4f}")

print("\n=== Classification Report ===")
print(classification_report(
    y_val, y_pred,
    target_names=['ESI 1','ESI 2','ESI 3','ESI 4','ESI 5']
))

# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['ESI 1','ESI 2','ESI 3','ESI 4','ESI 5'],
            yticklabels=['ESI 1','ESI 2','ESI 3','ESI 4','ESI 5'])
ax.set_title('Confusion Matrix — XGBoost Triage Classifier')
ax.set_ylabel('Actual ESI Level')
ax.set_xlabel('Predicted ESI Level')
plt.tight_layout()
plt.show()
```

## 5. Model Interpretability — SHAP Values
SHAP (SHapley Additive exPlanations) quantifies each feature's contribution to individual predictions, 
connecting model behavior to clinical reasoning.


```python
# SHAP Values — Model Interpretability
print("Calculating SHAP values... (this may take a minute)")

# Use a sample of 2000 for speed
X_sample = X_val.sample(2000, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

# SHAP summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values[:, :, 1],  # ESI 2 class
    X_sample,
    feature_names=FEATURES,
    max_display=15,
    show=False
)
plt.title('SHAP Values — Feature Impact on ESI 2 Prediction\n(Emergent Acuity)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print("SHAP plot complete.")
```

## 6. Demographic Bias Audit
We compare model-predicted acuity against assigned acuity to identify systematic undertriage patterns 
across language, insurance, sex, and age group.


```python
# ── Bias Analysis ────────────────────────────────────────────────────

# Retrain on full dataset
X_full = train[FEATURES]
y_full = train[TARGET] - 1

model_full = xgb.XGBClassifier(
    n_estimators=336,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=5,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
model_full.fit(X_full, y_full)

# Generate predictions on full training set
train_preds = model_full.predict(X_full) + 1

# Add predictions to demographics
train_demo['predicted_acuity'] = train_preds
train_demo['undertriaged'] = (train_demo['predicted_acuity'] < 
                               train_demo['triage_acuity']).astype(int)

overall_rate = train_demo['undertriaged'].mean() * 100
print(f"Overall undertriage rate: {overall_rate:.2f}%")
print(f"Total undertriaged:       {train_demo['undertriaged'].sum():,}")
```


```python
# Undertriage rates by demographic group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Undertriage Rates by Demographic Group', 
             fontsize=15, fontweight='bold')

def plot_undertriage(ax, col, title):
    rates = (train_demo.groupby(col)['undertriaged']
             .agg(['mean', 'count'])
             .reset_index())
    rates['rate'] = rates['mean'] * 100
    rates = rates.sort_values('rate', ascending=False)
    colors = ['#d73027' if r > overall_rate else '#4575b4' 
              for r in rates['rate']]
    bars = ax.bar(rates[col], rates['rate'], color=colors, edgecolor='white')
    ax.axhline(y=overall_rate, color='black', linestyle='--',
               linewidth=1.5, label=f'Overall: {overall_rate:.1f}%')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Undertriage Rate (%)')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, rate in zip(bars, rates['rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    return rates

rates_language = plot_undertriage(axes[0,0], 'language', 'By Language')
rates_insurance = plot_undertriage(axes[0,1], 'insurance_type', 'By Insurance Type')
rates_sex = plot_undertriage(axes[1,0], 'sex', 'By Sex')
rates_age = plot_undertriage(axes[1,1], 'age_group', 'By Age Group')

plt.tight_layout()
plt.show()

# Compound risk
train_demo['risk_group'] = (train_demo['age_group'] + ' / ' + 
                             train_demo['language'])
compound = (train_demo.groupby('risk_group')['undertriaged']
            .agg(['mean', 'count'])
            .reset_index())
compound['rate'] = compound['mean'] * 100
compound = compound[compound['count'] >= 100]
compound = compound.sort_values('rate', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#d73027' if r > overall_rate else '#4575b4' 
          for r in compound['rate']]
ax.barh(compound['risk_group'][::-1], compound['rate'][::-1],
        color=colors[::-1], edgecolor='white')
ax.axvline(x=overall_rate, color='black', linestyle='--',
           linewidth=1.5, label=f'Overall: {overall_rate:.1f}%')
ax.set_title('Top 15 Highest Undertriage Risk Groups\n(Age + Language Combination)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Undertriage Rate (%)')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

print("\nTop 10 highest risk combinations:")
print(compound[['risk_group', 'count', 'rate']].head(10).to_string(index=False))
```

### Key Findings
- **Pediatric / Somali:** 7.2% undertriage rate — highest risk group, 37% above overall
- **Elderly / Russian:** 6.6% — language barrier compounding age-related vulnerability  
- **Sex:** No meaningful difference across F, M, and Other — sex is not a bias dimension here
- **Language minority status + vulnerable age = compounding risk**

These findings suggest targeted interventions for non-Finnish speaking pediatric 
and elderly patients in Finnish emergency departments.

## 7. Generate Submission


```python
# Generate final submission
X_test = test[FEATURES]
test_preds = model_full.predict(X_test) + 1

test_original = pd.read_csv(PATH + 'test.csv')
submission = pd.DataFrame({
    'patient_id': test_original['patient_id'],
    'triage_acuity': test_preds
})

print("Prediction distribution:")
print(submission['triage_acuity'].value_counts().sort_index())
print(f"\nMissing values: {submission.isnull().sum().sum()}")
print(f"Shape: {submission.shape}")

submission.to_csv('submission.csv', index=False)
print("\nSubmission saved.")
```

## 8. Conclusions & Clinical Implications

### Model Performance
Our XGBoost classifier achieves 85.9% accuracy on unseen patients with 96% recall 
on ESI 1 (critical) cases — demonstrating that AI can reliably support triage 
acuity prediction from structured intake data alone.

### Bias Findings
This analysis reveals systematic undertriage patterns affecting specific demographic 
groups in simulated Finnish ED data:

1. **Language minority patients** — Russian, Arabic, Estonian, and Other language 
speakers are consistently undertriaged above the overall rate
2. **Elderly patients** — face elevated undertriage risk, most dangerous given 
reduced physiological reserve
3. **Compounding vulnerability** — pediatric and elderly patients speaking minority 
languages face the highest undertriage rates, up to 7.2%

### Clinical Recommendations
- Structured interpreter access protocols for non-Finnish speaking patients
- Mandatory complete vital sign assessment for elderly patients regardless of 
apparent acuity
- Targeted triage training addressing atypical presentations in pediatric and 
elderly minority language patients

### Limitations
- Analysis conducted on synthetic data — real-world patterns may differ
- Undertriage defined relative to model predictions, which may carry their own biases
- Small sample sizes for some subgroups limit statistical confidence
- Causality cannot be established from observational patterns alone

### Reproducibility
All code is available at: [GitHub link — to be added]
Dataset: Triagegeist synthetic ED dataset, Laitinen-Fredriksson Foundation

## 8. Real-World Validation — NHAMCS 2018-2022

To validate findings from the synthetic dataset, we apply the same bias audit 
methodology to the NHAMCS (National Hospital Ambulatory Medical Care Survey) 
dataset — 58,124 real US emergency department visits from 2018-2022.

**Why this matters:** Findings from synthetic data alone are suggestive but not 
conclusive. Consistent patterns across both synthetic and real-world data 
strengthen the clinical argument significantly.


```python
# ── NHAMCS Real-World Validation ─────────────────────────────────────
import os

# Find NHAMCS file path
for dirname, _, filenames in os.walk('/kaggle/input/datasets/reaper0ai/nhamcs-2018-22'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```


```python
# Load NHAMCS data
nhamcs = pd.read_csv('/kaggle/input/datasets/reaper0ai/nhamcs-2018-22/nhamcs_data_2018_22.csv')

# Save demographics
nhamcs_demo = nhamcs[['sex', 'race', 'insurance', 'target_triage_acuity']].copy()

# Feature preparation
nhamcs['temp_c'] = (nhamcs['temp'] - 32) * 5/9
nhamcs['pain_score'] = nhamcs['pain_score'].replace(-1, np.nan)

features_nhamcs = [
    'age', 'sys_bp', 'dias_bp', 'heart_rate', 'resp_rate',
    'spo2', 'temp_c', 'pain_score', 'sex', 'insurance', 'race',
    'visit_month', 'arrival_time', 'ems_arrival', 'seen_last_72h',
    'episode', 'hist_alzheimers', 'hist_asthma', 'hist_cancer',
    'hist_stroke', 'hist_ckd', 'hist_copd', 'hist_chf', 'hist_cad',
    'hist_depression', 'hist_diabetes_t1', 'hist_diabetes_t2',
    'hist_hypertension', 'hist_obesity', 'hist_substance_abuse'
]

nhamcs = nhamcs.dropna(subset=['target_triage_acuity'])
X_nhamcs = nhamcs[features_nhamcs].copy()
y_nhamcs = nhamcs['target_triage_acuity'].astype(int) - 1

# Encode categoricals
for col in X_nhamcs.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_nhamcs[col] = le.fit_transform(X_nhamcs[col].astype(str))

# Train model
model_nhamcs = xgb.XGBClassifier(
    n_estimators=209, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5,
    random_state=42, n_jobs=-1, verbosity=0
)
model_nhamcs.fit(X_nhamcs, y_nhamcs)

# Bias analysis
nhamcs_demo = nhamcs_demo.loc[X_nhamcs.index].copy()
nhamcs_demo['predicted_acuity'] = model_nhamcs.predict(X_nhamcs) + 1
nhamcs_demo['undertriaged'] = (nhamcs_demo['predicted_acuity'] < nhamcs_demo['target_triage_acuity']).astype(int)

overall_nhamcs = nhamcs_demo['undertriaged'].mean() * 100
print(f"NHAMCS shape: {nhamcs.shape}")
print(f"NHAMCS overall undertriage rate: {overall_nhamcs:.2f}%")
```


```python
# NHAMCS Bias Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('NHAMCS Undertriage Rates by Demographic Group\n(Real US ED Data 2018-2022)',
             fontsize=13, fontweight='bold')

def plot_nhamcs(ax, col, title):
    rates = (nhamcs_demo.groupby(col)['undertriaged']
             .agg(['mean', 'count'])
             .reset_index())
    rates['rate'] = rates['mean'] * 100
    rates = rates.sort_values('rate', ascending=False)
    colors = ['#d73027' if r > overall_nhamcs else '#4575b4' for r in rates['rate']]
    ax.bar(rates[col], rates['rate'], color=colors, edgecolor='white')
    ax.axhline(y=overall_nhamcs, color='black', linestyle='--',
               linewidth=1.5, label=f'Overall: {overall_nhamcs:.1f}%')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Undertriage Rate (%)')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plot_nhamcs(axes[0], 'race', 'By Race/Ethnicity')
plot_nhamcs(axes[1], 'insurance', 'By Insurance Type')
plot_nhamcs(axes[2], 'sex', 'By Sex')

plt.tight_layout()
plt.show()

# Cross-dataset summary
print("\n=== CROSS-DATASET BIAS COMPARISON ===")
print(f"\nSynthetic Finnish ED — Overall undertriage: {overall_rate:.2f}%")
print(f"NHAMCS Real US ED  — Overall undertriage: {overall_nhamcs:.2f}%")
print(f"\nConsistent finding: Uninsured patients face elevated undertriage")
print(f"risk in both synthetic and real-world ED datasets.")
print(f"\nNew finding in real data: Female patients undertriaged at")
print(f"higher rates than males — not observed in synthetic data.")
```

### Cross-Dataset Findings

Both datasets consistently show elevated undertriage risk for uninsured/self-pay 
patients — a robust signal across synthetic Finnish ED and real US ED data.

The real US data reveals an additional finding not present in the synthetic data: 
**female patients are undertriaged at higher rates than males** (24.1% vs 21.6%). 
This is consistent with published literature on gender bias in emergency medicine.

The higher overall undertriage rate in NHAMCS (23.0% vs 5.3%) reflects the 
inherent complexity of real clinical data compared to synthetic data calibrated 
for modeling purposes.
