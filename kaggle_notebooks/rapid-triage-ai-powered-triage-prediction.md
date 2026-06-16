# RAPID-TRIAGE: AI-Powered Emergency Triage Acuity Prediction
**Laitinen-Fredriksson Foundation Hackathon · April 2026**

---

### Overview
Emergency department triage nurses must assign an ESI acuity level (1–5) to every arriving patient — often within two minutes, under time pressure, with incomplete information. Errors lead directly to undertriage, delayed care, and adverse outcomes.

This notebook builds a **LightGBM multiclass classifier** that predicts ESI acuity from structured intake data available at the moment of triage: vital signs, demographics, chief complaint system, comorbidities, and prior utilisation. The model achieves a **Quadratic Weighted Kappa (QWK) of 0.931** on 5-fold out-of-fold validation.

### Approach 
| Step | Detail |
|---|---|
| Data | 80k labeled ED encounters + comorbidity history + chief complaints |
| Target | ESI acuity level 1–5 (ordinal, 5-class) |
| Features | 61 (vitals, demographics, comorbidities, triage context) |
| Model | LightGBM multiclass with early stopping |
| Validation | 5-fold stratified cross-validation |
| Metric | Quadratic Weighted Kappa (QWK) |
| Result | **OOF QWK = 0.931 ± 0.001** |

---

### Data Disclosure
All data originates exclusively from the Triagegeist competition dataset provided via Kaggle. No external datasets were used.

| File | Rows | Description |
|---|---|---|
| `train.csv` | 80,000 | Labeled patient encounters with ESI acuity (1–5) |
| `test.csv` | 20,000 | Unlabeled encounters for final prediction |
| `patient_history.csv` | 100,000 | Binary comorbidity flags (25 conditions per patient) |
| `chief_complaints.csv` | 100,000 | Chief complaint text and organ system classification |

**Citation:** Laitinen-Fredriksson Foundation. *Triagegeist Competition Dataset.* Kaggle, 2026.




```python
# ── Dependencies ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.dpi'] = 120

print("LightGBM:", lgb.__version__)
print("pandas:  ", pd.__version__)
print("numpy:   ", np.__version__)

```


```python
# ── 1. Load Data ──────────────────────────────────────────────────────────────
BASE = '/kaggle/input/competitions/triagegeist'

train      = pd.read_csv(f'{BASE}/train.csv')
test       = pd.read_csv(f'{BASE}/test.csv')
history    = pd.read_csv(f'{BASE}/patient_history.csv')
complaints = pd.read_csv(f'{BASE}/chief_complaints.csv')

print(f"train:      {train.shape}")
print(f"test:       {test.shape}")
print(f"history:    {history.shape}")
print(f"complaints: {complaints.shape}")

# Target distribution
print("\nESI acuity distribution (train):")
dist = train['triage_acuity'].value_counts().sort_index()
for lvl, n in dist.items():
    bar = '█' * int(n / 500)
    print(f"  ESI-{lvl}: {n:6,}  ({n/len(train)*100:.1f}%)  {bar}")

```


```python
# ── 2. Exploratory Data Analysis ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Acuity distribution
ax = axes[0]
counts = train['triage_acuity'].value_counts().sort_index()
colors = ['#DC2626','#F97316','#3B82F6','#10B981','#6B7280']
ax.bar([f'ESI-{i}' for i in counts.index], counts.values, color=colors, edgecolor='none')
ax.set_title('Acuity Distribution (Train)', fontweight='bold')
ax.set_ylabel('Patients')
ax.spines[['top','right']].set_visible(False)
for i, v in enumerate(counts.values):
    ax.text(i, v + 200, f'{v/len(train)*100:.1f}%', ha='center', fontsize=9, color='#374151')

# NEWS2 by acuity
ax = axes[1]
for i, lvl in enumerate([1,2,3,4,5]):
    data = train[train['triage_acuity']==lvl]['news2_score'].dropna()
    ax.boxplot(data, positions=[i], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=colors[i], alpha=0.7),
               medianprops=dict(color='white', linewidth=2),
               flierprops=dict(marker='.', markersize=2, alpha=0.3),
               whiskerprops=dict(color='#9CA3AF'),
               capprops=dict(color='#9CA3AF'))
ax.set_xticks(range(5))
ax.set_xticklabels([f'ESI-{i}' for i in [1,2,3,4,5]])
ax.set_title('NEWS2 Score by Acuity', fontweight='bold')
ax.set_ylabel('NEWS2 Score')
ax.spines[['top','right']].set_visible(False)

# Pain score by acuity
ax = axes[2]
for i, lvl in enumerate([1,2,3,4,5]):
    data = train[train['triage_acuity']==lvl]['pain_score'].dropna()
    ax.boxplot(data, positions=[i], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=colors[i], alpha=0.7),
               medianprops=dict(color='white', linewidth=2),
               flierprops=dict(marker='.', markersize=2, alpha=0.3),
               whiskerprops=dict(color='#9CA3AF'),
               capprops=dict(color='#9CA3AF'))
ax.set_xticks(range(5))
ax.set_xticklabels([f'ESI-{i}' for i in [1,2,3,4,5]])
ax.set_title('Pain Score by Acuity', fontweight='bold')
ax.set_ylabel('Pain Score (0–10)')
ax.spines[['top','right']].set_visible(False)

plt.suptitle('Key Clinical Features vs. ESI Acuity Level', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda.png', bbox_inches='tight')
plt.show()

# Missing value summary
print("\nMissing values (train):")
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing):
    print(missing.to_string())
else:
    print("  None")

```


```python
# ── 3. Feature Engineering ────────────────────────────────────────────────────

# Merge auxiliary tables
train = train.merge(history, on='patient_id', how='left')
test  = test.merge(history, on='patient_id', how='left')
train = train.merge(complaints[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
test  = test.merge(complaints[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')

# Derived features
for df in [train, test]:
    df['shock_index']   = df['heart_rate'] / df['systolic_bp'].replace(0, np.nan)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['bmi']           = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# Columns excluded from model
# - disposition, ed_los_hours: post-triage leakage
# - chief_complaint_raw: free-text NLP not implemented (future work)
# - patient_id: identifier only
DROP = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_raw']

# Categorical encoding — combined vocab prevents unseen-category errors at inference
CAT_COLS = [
    'site_id', 'triage_nurse_id', 'arrival_mode', 'arrival_day', 'arrival_season',
    'shift', 'age_group', 'sex', 'language', 'insurance_type', 'transport_origin',
    'pain_location', 'mental_status_triage', 'chief_complaint_system', 'arrival_month'
]

le_dict = {}
for col in CAT_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    le_dict[col] = le

FEAT_COLS = [c for c in train.columns if c not in DROP]
X      = train[FEAT_COLS]
y      = train['triage_acuity'].astype(int)
X_test = test[FEAT_COLS]

print(f"Training features: {len(FEAT_COLS)}")
print(f"Training samples:  {len(X):,}")
print(f"Test samples:      {len(X_test):,}")

```


```python
# ── 4. Model: LightGBM with 5-Fold Stratified CV ─────────────────────────────
#
# LightGBM is chosen for its strong tabular performance, native handling of
# categorical features, and fast training — enabling full CV within notebook limits.
#
# QWK is the primary metric because ESI levels are ordinal: a model that confuses
# ESI-1 with ESI-2 should be penalised less than one confusing ESI-1 with ESI-5.

PARAMS = {
    'objective':         'multiclass',
    'num_class':         6,          # 0-indexed; class 0 unused (ESI runs 1–5)
    'metric':            'multi_logloss',
    'learning_rate':     0.05,
    'num_leaves':        127,
    'min_child_samples': 20,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'reg_alpha':         0.1,
    'reg_lambda':        1.0,
    'n_jobs':           -1,
    'verbose':          -1,
    'seed':              42,
}

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds  = np.zeros((len(X), 6))
test_preds = np.zeros((len(X_test), 6))
kappa_scores = []
models = []

print(f"Training {N_FOLDS}-fold LightGBM...")
print(f"{'─'*45}")

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr - 1)
    dval   = lgb.Dataset(X_val, label=y_val - 1)

    model = lgb.train(
        PARAMS, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(False)]
    )

    val_prob  = model.predict(X_val)
    oof_preds[val_idx] = val_prob
    test_preds += model.predict(X_test) / N_FOLDS

    val_pred = np.argmax(val_prob, axis=1) + 1
    kappa    = cohen_kappa_score(y_val, val_pred, weights='quadratic')
    kappa_scores.append(kappa)
    models.append(model)
    print(f"  Fold {fold+1}: QWK = {kappa:.4f}  |  trees = {model.best_iteration:4d}")

print(f"{'─'*45}")
print(f"  Mean QWK : {np.mean(kappa_scores):.4f}")
print(f"  Std QWK  : {np.std(kappa_scores):.4f}")

```


```python
# ── 5. Evaluation ─────────────────────────────────────────────────────────────
oof_class = np.argmax(oof_preds, axis=1) + 1
overall_kappa = cohen_kappa_score(y, oof_class, weights='quadratic')
print(f"Overall OOF QWK: {overall_kappa:.4f}\n")
print("Per-class classification report:")
print(classification_report(y, oof_class, target_names=[f'ESI-{i}' for i in range(1, 6)]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y, oof_class)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
disp = ConfusionMatrixDisplay(confusion_matrix=cm_pct,
                               display_labels=[f'ESI-{i}' for i in range(1,6)])
disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.1f')
ax.set_title(f'OOF Confusion Matrix (row-normalised %)\nQWK = {overall_kappa:.4f}',
             fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()

```


```python
# ── 6. Feature Importance ─────────────────────────────────────────────────────
# Average gain across all 5 fold models for a more stable ranking

importance_df = pd.DataFrame(
    {f"fold_{i+1}": m.feature_importance(importance_type='gain')
     for i, m in enumerate(models)},
    index=FEAT_COLS
)
importance_df['mean_gain'] = importance_df.mean(axis=1)
importance_df = importance_df.sort_values('mean_gain', ascending=False)

top20 = importance_df['mean_gain'].head(20)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#DC2626' if i == 0 else '#3B82F6' if i < 5 else '#93C5FD'
          for i in range(len(top20))]
ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1], edgecolor='none')
ax.set_xlabel('Mean Gain (5-fold average)', fontsize=11)
ax.set_title('Top 20 Feature Importances\nLightGBM · Mean Gain across 5 Folds',
             fontsize=12, fontweight='bold')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='x', alpha=0.25)
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight')
plt.show()

print("Top 10 features:")
for feat, gain in top20.head(10).items():
    print(f"  {feat:<35} {gain:>10,.0f}")

```


```python
# ── 7. Generate Submission ────────────────────────────────────────────────────
final_preds = np.argmax(test_preds, axis=1) + 1

submission = pd.DataFrame({
    'patient_id':    test['patient_id'],
    'triage_acuity': final_preds
})

submission.to_csv('submission.csv', index=False)
print("Saved: submission.csv")
print(f"Shape: {submission.shape}")
print("\nPredicted acuity distribution:")
dist = submission['triage_acuity'].value_counts().sort_index()
for lvl, n in dist.items():
    bar = '█' * int(n / 100)
    print(f"  ESI-{lvl}: {n:6,}  ({n/len(submission)*100:.1f}%)  {bar}")

submission.head()

```

---

## Results

| Metric | Value |
|---|---|
| OOF Quadratic Weighted Kappa | **0.931 ± 0.001** |
| Model | LightGBM (multiclass) |
| Validation | 5-fold stratified CV |
| Training samples | 80,000 |
| Features | 61 |

### Top predictive features
The model's most important features are clinically coherent — they map directly to the physiological parameters used in established triage protocols (NEWS2, ESI):

1. `news2_score` — validated composite early warning score
2. `pain_score` — subjective severity (0–10 NRS)
3. `gcs_total` — Glasgow Coma Scale (neurological status)
4. `spo2` — peripheral oxygen saturation
5. `respiratory_rate` — respiratory distress marker

The alignment between model-derived feature importance and established clinical knowledge provides confidence that the model is learning genuine clinical signal, not dataset artifacts.

### Limitations 
- **No NLP on `chief_complaint_raw`** — the highest-value next step; clinical NLP (e.g., ClinicalBERT) on free-text complaints could significantly improve performance
- **Synthetic data** — real-world noise, transcription errors, and missing-data patterns may differ
- **No calibration** — probability outputs should be calibrated before clinical use
- **No fairness audit** — bias by age, sex, language, and insurance type requires explicit evaluation before any deployment



```python

```
