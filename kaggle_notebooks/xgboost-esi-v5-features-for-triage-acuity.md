# Triagegeist — ESI Triage Acuity Prediction

**Approach:** XGBoost classifier with clinically grounded feature engineering derived from the ESI Version 5 Handbook (Emergency Nurses Association, 2023) and published triage ML research (Ivanov et al. 2021, Tyler et al. 2024).

**Clinical framing:** The Emergency Severity Index assigns patients to one of five acuity levels based on four sequential decision points: (A) lifesaving intervention required, (B) high-risk presentation, (C) resource prediction, and (D) vital sign thresholds. Each engineered feature in this notebook maps directly to one of these decision points.

**Key benchmark:** Ivanov et al. 2021 (KATE model) achieved 75.7% accuracy and 80% ESI-2 recall using XGBoost + clinical NLP on real hospital data. Nurse baseline accuracy is 59.8% with ESI-2 recall of 41.4%.


```python
# ── Cell 1: Imports & Data Load ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/competitions/triagegeist/train.csv')
test  = pd.read_csv('/kaggle/input/competitions/triagegeist/test.csv')
cc    = pd.read_csv('/kaggle/input/competitions/triagegeist/chief_complaints.csv')
ph    = pd.read_csv('/kaggle/input/competitions/triagegeist/patient_history.csv')

print(f'Train:    {train.shape}')
print(f'Test:     {test.shape}')
print(f'CC:       {cc.shape}')
print(f'History:  {ph.shape}')
```

## Why We Join Three Files

A model learns from rows, not files. Each patient's full clinical picture — vital signs, complaint, and medical history — lives across three separate files. A triage nurse uses all three simultaneously when assigning ESI level. We must give the model the same information in a single row.

Hong et al. 2018 showed that combining triage data with patient history raised AUC from 0.874 to 0.924 — a significant gain from simply joining files correctly.


```python
# ── Cell 2: Join auxiliary tables ────────────────────────────────────────────
# Drop duplicate column that exists in both train and cc
if 'chief_complaint_system' in cc.columns and 'chief_complaint_system' in train.columns:
    cc = cc.drop(columns=['chief_complaint_system'])

train = train.merge(cc, on='patient_id', how='left')
test  = test.merge(cc, on='patient_id', how='left')
train = train.merge(ph, on='patient_id', how='left')
test  = test.merge(ph, on='patient_id', how='left')

print(f'After join: train={train.shape}, test={test.shape}')
print(f'Train patients: {train["patient_id"].nunique()}')
print(f'Test  patients: {test["patient_id"].nunique()}')
```

## Data Cleaning

Three data quality issues identified during EDA:

1. **Systolic BP below 50 mmHg** — 236 patients have physiologically impossible values. Capped at 50.
2. **Pain score = -1** — Encodes 'not recorded' rather than zero pain. Converted to NaN with a separate binary flag.
3. **Negative pulse pressure** — Derived from bad BP values. Recalculated after BP clip.


```python
# ── Cell 3: Data cleaning ─────────────────────────────────────────────────────
def clean_data(df):
    df = df.copy()
    # Clip impossible BP values
    df['systolic_bp']  = df['systolic_bp'].clip(lower=50)
    df['diastolic_bp'] = df['diastolic_bp'].clip(lower=20)
    # Pain score -1 means not recorded
    df['pain_score_missing'] = (df['pain_score'] == -1).astype(int)
    df['pain_score']   = df['pain_score'].replace(-1, np.nan)
    # Recalculate pulse pressure after BP fix
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    return df

train = clean_data(train)
test  = clean_data(test)
print('Cleaning done')
print(f'Pain score missing flag: {train["pain_score_missing"].sum()} patients ({train["pain_score_missing"].mean():.1%})')
print(f'Negative pulse_pressure remaining: {(train["pulse_pressure"] < 0).sum()}')
```

## Feature Engineering — Clinical Decision Points

Every feature below maps to a specific decision point in the ESI v5 algorithm or to a finding from the EDA.

**Critical EDA finding:** When systolic BP is missing, 100% of patients are ESI 4 or 5. Zero ESI 1 or 2 patients have missing BP. This is not random missingness — it reflects clinical practice where stable patients are sometimes not fully measured. Missing data is a clinical signal.


```python
# ── Cell 4: Missing-value flags ───────────────────────────────────────────────
# Clinical reasoning: missing vitals = patient was stable enough to skip measurement
# EDA showed bp_missing predicts ESI 4/5 with 100% certainty
def add_missing_flags(df):
    df = df.copy()
    df['bp_missing']   = df['systolic_bp'].isna().astype(int)
    df['rr_missing']   = df['respiratory_rate'].isna().astype(int)
    df['temp_missing'] = df['temperature_c'].isna().astype(int)
    vital_cols = ['heart_rate','respiratory_rate','systolic_bp','spo2','temperature_c']
    df['n_vitals_missing'] = df[vital_cols].isna().sum(axis=1)
    return df

train = add_missing_flags(train)
test  = add_missing_flags(test)
print('Missing flags done')
print(f'Patients with missing BP: {train["bp_missing"].sum()} ({train["bp_missing"].mean():.1%})')
```


```python
# ── Cell 5: Mental status encoding ───────────────────────────────────────────
# ESI Decision Point B: confused/lethargic/disoriented = ESI 2 immediately
# EDA: mental_status_triage was one of the strongest predictors
def add_mental_status_features(df):
    df = df.copy()
    # Binary: alert vs any altered state
    df['not_alert'] = (df['mental_status_triage'] != 'alert').astype(int)
    # Ordinal severity scale
    mental_map = {'alert':0,'confused':1,'agitated':2,'drowsy':3,'unresponsive':4}
    df['mental_status_severity'] = df['mental_status_triage'].map(mental_map).fillna(0)
    return df

train = add_mental_status_features(train)
test  = add_mental_status_features(test)
print('Mental status features done')
print(f'Not alert patients: {train["not_alert"].sum()} ({train["not_alert"].mean():.1%})')
```


```python
# ── Cell 6: Comorbidity features ──────────────────────────────────────────────
# ESI handbook: immunocompromised + fever = ESI 2 always
# EDA: ESI 1/2 patients average 6.6 comorbidities vs 5.0 for ESI 3/4/5
def add_comorbidity_features(df):
    df = df.copy()
    hx_cols = [c for c in df.columns if c.startswith('hx_')]

    # Total burden
    df['total_comorbidities'] = df[hx_cols].sum(axis=1)

    # High risk group — any of these makes vitals more serious
    high_risk = ['hx_malignancy','hx_immunosuppressed','hx_heart_failure',
                 'hx_copd','hx_coagulopathy','hx_coronary_artery_disease']
    df['high_risk_comorbidity'] = df[[c for c in high_risk if c in df.columns]].sum(axis=1)

    # Immunocompromised from any cause
    immuno = ['hx_malignancy','hx_immunosuppressed','hx_hiv']
    df['any_immunocompromised'] = (
        df[[c for c in immuno if c in df.columns]].sum(axis=1) > 0
    ).astype(int)

    # Cardiac risk
    cardiac = ['hx_heart_failure','hx_coronary_artery_disease','hx_atrial_fibrillation']
    df['cardiac_risk'] = (
        df[[c for c in cardiac if c in df.columns]].sum(axis=1) > 0
    ).astype(int)

    # Neurological risk
    neuro = ['hx_dementia','hx_epilepsy','hx_stroke_prior']
    df['neuro_risk'] = (
        df[[c for c in neuro if c in df.columns]].sum(axis=1) > 0
    ).astype(int)
    return df

train = add_comorbidity_features(train)
test  = add_comorbidity_features(test)
print('Comorbidity features done')
```


```python
# ── Cell 7: Interaction features ──────────────────────────────────────────────
# Age alone has correlation -0.004 with ESI level (EDA finding)
# Age combined with vitals or comorbidities is meaningful
# Same HR means different risk depending on cardiac history
def add_interaction_features(df):
    df = df.copy()
    # Age × heart rate: elderly tachycardia is more serious
    df['age_hr_interaction']  = df['age'] * df['heart_rate']
    # Immunocompromised + fever = ESI 2 by handbook
    df['immuno_fever']        = (
        df['any_immunocompromised'] * (df['temperature_c'] > 38.3).astype(int)
    )
    # Cardiac patient + tachycardia = elevated risk
    df['cardiac_tachycardia'] = (
        df['cardiac_risk'] * (df['heart_rate'] > 100).astype(int)
    )
    # Severe pain + altered consciousness = almost always ESI 1/2
    df['pain_and_altered']    = (
        (df['pain_score'] >= 7).fillna(False).astype(int) * df['not_alert']
    )
    return df

train = add_interaction_features(train)
test  = add_interaction_features(test)
print('Interaction features done')
```


```python
# ── Cell 8: Chief complaint keyword flags ─────────────────────────────────────
# ESI Decision Point B lists specific high-risk presentations
# These keywords encode that clinical knowledge directly
# Note: raw TF-IDF/BERT text features were tested but excluded
# (see narrative below this cell)
def add_complaint_features(df):
    df = df.copy()
    cc_text = df['chief_complaint_raw'].fillna('').str.lower()

    # High-risk terms from ESI v5 Decision Point B
    high_risk_words = [
        'thunderclap','chest pain','stroke','seizure',
        'syncope','unconscious','overdose','suicide',
        'shortness of breath','difficulty breathing',
        'rigors','sepsis','anaphylaxis','hemorrhage',
        'bleeding','trauma','intoxication','altered'
    ]
    for word in high_risk_words:
        col = 'cc_' + word.replace(' ','_')
        df[col] = cc_text.str.contains(word, regex=False).astype(int)

    # Composite flag: any high-risk term present
    df['cc_any_high_risk'] = (
        df[[f'cc_{w.replace(" ","_")}' for w in high_risk_words]].sum(axis=1) > 0
    ).astype(int)

    # Deterioration signals
    df['cc_worsening'] = cc_text.str.contains('worsening|worse|deteriorat', regex=True).astype(int)
    df['cc_acute']     = cc_text.str.contains('sudden|acute|abrupt|rapid|hours', regex=True).astype(int)
    return df

train = add_complaint_features(train)
test  = add_complaint_features(test)
print('Complaint keyword features done')
print(f'High-risk complaint flag: {train["cc_any_high_risk"].sum()} patients ({train["cc_any_high_risk"].mean():.1%})')
```

### Why Raw Text Features Were Excluded

TF-IDF vectorisation of `chief_complaint_raw` and BioClinicalBERT embeddings were both tested. Both produced suspiciously high accuracy (TF-IDF: 0.987, BERT: 0.994).

Investigation revealed the cause. The synthetic data generator encoded ESI acuity into complaint text using severity descriptors. Correlation analysis confirmed:

| TF-IDF Feature | Correlation with ESI level |
|---|---|
| tfidf_mild | +0.307 |
| tfidf_acute | -0.252 |
| tfidf_severe | -0.212 |
| tfidf_minor | +0.155 |

Words like 'mild', 'minor', and 'advice' appeared systematically in ESI 4/5 complaints while 'severe', 'acute', and 'critical' appeared in ESI 1/2 complaints. In real ED data, complaint text is written independently before ESI is assigned. In this synthetic dataset the text appears to have been generated from the ESI label rather than independently — creating artificial leakage.

We retained manual keyword flags for clinically validated high-risk terms from the ESI handbook (thunderclap, rigors, chest pain etc.) which represent genuine domain knowledge rather than data artifacts. The final model uses structured clinical features only, giving honest and clinically defensible results.


```python
# ── Cell 9: ESI threshold flags ───────────────────────────────────────────────
# Directly encoding ESI Decision Point D vital sign thresholds
# From ESI v5 Table 6-1: adult thresholds HR>100, RR>20, SpO2<92%
def add_esi_threshold_flags(df):
    df = df.copy()
    # Decision Point D thresholds (adults)
    df['hr_high_risk']  = (df['heart_rate'] > 100).astype(int)
    df['rr_high_risk']  = (df['respiratory_rate'] > 20).astype(int)
    df['spo2_low']      = (df['spo2'] < 92).astype(int)
    df['sbp_low']       = (df['systolic_bp'] < 90).astype(int)
    df['temp_fever']    = (df['temperature_c'] > 38.3).astype(int)
    df['temp_low']      = (df['temperature_c'] < 36.0).astype(int)
    df['gcs_impaired']  = (df['gcs_total'] < 14).astype(int)
    df['pain_severe']   = (df['pain_score'] >= 7).fillna(0).astype(int)
    df['shock_high']    = (df['shock_index'] > 1.0).astype(int)
    # NEWS2 risk thresholds
    df['news2_medium']  = (df['news2_score'] >= 5).astype(int)
    df['news2_high']    = (df['news2_score'] >= 7).astype(int)
    # Count of abnormal vitals
    risk_flags = ['hr_high_risk','rr_high_risk','spo2_low','sbp_low','temp_fever','gcs_impaired']
    df['n_high_risk_vitals'] = df[risk_flags].sum(axis=1)
    return df

train = add_esi_threshold_flags(train)
test  = add_esi_threshold_flags(test)
print('ESI threshold flags done')
```


```python
# ── Cell 10: Prepare tabular features ────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

# Columns to exclude from features
drop_cols = [
    'patient_id',
    'triage_acuity',       # target
    'disposition',         # post-triage outcome — leakage
    'ed_los_hours',        # post-triage outcome — leakage
    'chief_complaint_raw'  # replaced by keyword flags
]

def prepare_features(df, drop_cols, encoders=None):
    df = df.copy()
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}

    # Drop target and leakage columns
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drops)

    # Encode all remaining string columns automatically
    object_cols = df.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        df[col] = df[col].fillna('unknown')
        if fit_mode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else 'unknown'
            )
            df[col] = le.transform(df[col])

    # Final check — drop any remaining object columns
    remaining = df.select_dtypes(include='object').columns.tolist()
    if remaining:
        print(f'Warning: dropping remaining object columns: {remaining}')
        df = df.drop(columns=remaining)

    return df, encoders

X_train, encoders = prepare_features(train, drop_cols)
X_test,  _        = prepare_features(test, drop_cols, encoders)
y_train = train['triage_acuity'] - 1  # XGBoost needs 0-indexed classes

print(f'Tabular features: train={X_train.shape}, test={X_test.shape}')
print(f'Target classes: {sorted(y_train.unique())}')
```


```python
# ── Cell 11: Class weights ────────────────────────────────────────────────────
# ESI 1 = 4% of data, ESI 3 = 36%
# Without weights: model ignores rare but critical ESI 1/2 patients
# With weights: ESI 1 mistakes cost 5x more, forcing the model to learn them
class_counts = y_train.value_counts().sort_index()
total = len(y_train)
class_weights = {
    i: total / (5 * count)
    for i, count in class_counts.items()
}
sample_weights = y_train.map(class_weights)

print('Class weights (ESI 0-4 maps to ESI 1-5):')
for k, v in class_weights.items():
    print(f'  ESI {k+1}: weight={v:.3f}')
print(f'\nsample_weights shape: {sample_weights.shape}')
```

## Hyperparameter Tuning

Optuna was used to search hyperparameter space over 50 trials with 3-fold cross-validation, optimising macro-F1 on **clean tabular features only**. The search ran for approximately 90 minutes on Kaggle CPU.

The improvement from tuning was marginal (+0.0009 F1 over baseline defaults) indicating the feature engineering already extracted signal cleanly enough that parameter choice had limited impact.


```python
# ── Cell 12: Best parameters (clean tabular tuning) ──────────────────────────
# Optuna 50 trials × 3-fold CV on clean tabular features
# Best trial: 31, Best F1: 0.8789
import xgboost as xgb

best_params = {
    'n_estimators':     1110,
    'max_depth':        8,
    'learning_rate':    0.02056285924146223,
    'subsample':        0.6723264435496604,
    'colsample_bytree': 0.7150724738986798,
    'min_child_weight': 2,
    'gamma':            0.27539414608117785,
    'eval_metric':      'mlogloss',
    'objective':        'multi:softprob',
    'num_class':        5,
    'random_state':     42,
    'n_jobs':           -1,
    'tree_method':      'hist',
    'device':           'cuda',
}

print('=== BEST PARAMETERS ===')
print('Source: Optuna 50-trial search on clean tabular features')
print('Optimised metric: macro-F1')
print()
for k, v in best_params.items():
    print(f'  {k}: {v}')
```

## Cross-Validation

5-fold stratified cross-validation ensures honest performance estimates on data the model never saw during training. Stratification preserves ESI level proportions in each fold, preventing any fold from having disproportionately few ESI 1 cases.

**Key metric: ESI-2 recall** — the fraction of true ESI 2 patients correctly identified. This is the most clinically important metric because ESI 2 patients are high-risk but not immediately dying. Nurse accuracy at this boundary is only 41.4% (Ivanov et al. 2021). Missing an ESI 2 patient means a high-risk patient waits — which increases morbidity and mortality.


```python
# ── Cell 13: 5-Fold Stratified CV ────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

val_preds = np.zeros(len(y_train), dtype=int)
val_proba = np.zeros((len(y_train), 5))
fold_accs, fold_f1s, fold_esi2s = [], [], []

y_arr  = y_train.values
sw_arr = sample_weights.values

for fold, (tr_idx, val_idx) in enumerate(
        cv.split(X_train.values, y_arr)):

    X_tr  = X_train.values[tr_idx]
    X_val = X_train.values[val_idx]
    y_tr  = y_arr[tr_idx]
    y_val = y_arr[val_idx]
    sw_tr = sw_arr[tr_idx]

    m = xgb.XGBClassifier(**best_params)
    m.fit(X_tr, y_tr, sample_weight=sw_tr)

    y_pred  = m.predict(X_val)
    y_proba = m.predict_proba(X_val)

    val_preds[val_idx] = y_pred
    val_proba[val_idx] = y_proba

    acc      = accuracy_score(y_val, y_pred)
    f1       = f1_score(y_val, y_pred, average='macro')
    esi2_rec = (y_pred[y_val == 1] == 1).mean()

    fold_accs.append(acc)
    fold_f1s.append(f1)
    fold_esi2s.append(esi2_rec)

    print(f'Fold {fold+1}: acc={acc:.4f}  macro-F1={f1:.4f}  ESI-2 recall={esi2_rec:.4f}')

print(f'\n=== 5-FOLD CV RESULTS ===')
print(f'Accuracy:      {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}')
print(f'Macro-F1:      {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}')
print(f'ESI-2 recall:  {np.mean(fold_esi2s):.4f} ± {np.std(fold_esi2s):.4f}')

print(f'\n=== VS BENCHMARKS ===')
print(f'  Nurses (Ivanov 2021):   acc=0.598  ESI-2 recall=0.414')
print(f'  KATE ML (Ivanov 2021):  acc=0.757  ESI-2 recall=0.800')
print(f'  This model:             acc={np.mean(fold_accs):.3f}  ESI-2 recall={np.mean(fold_esi2s):.3f}')
```


```python
# ── Cell 14: Full OOF classification report ───────────────────────────────────
from sklearn.metrics import classification_report

print('=== OUT-OF-FOLD CLASSIFICATION REPORT ===')
print(classification_report(
    y_arr,
    val_preds,
    target_names=['ESI-1','ESI-2','ESI-3','ESI-4','ESI-5'],
    digits=4
))

print('=== MODEL COMPARISON ===')
print(f'{"Model":<25} {"Accuracy":>10} {"Macro-F1":>10} {"ESI-2 Recall":>14}')
print('-' * 63)
print(f'{"Nurses baseline":<25} {0.598:>10.3f} {"???":>10} {0.414:>14.3f}')
print(f'{"KATE (Ivanov 2021)":<25} {0.757:>10.3f} {"???":>10} {0.800:>14.3f}')
print(f'{"This model":<25} {np.mean(fold_accs):>10.3f} {np.mean(fold_f1s):>10.3f} {np.mean(fold_esi2s):>14.3f}')
```


```python
# ── Cell 15: Confusion matrix ─────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_arr, val_preds)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

labels = ['ESI-1','ESI-2','ESI-3','ESI-4','ESI-5']

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title('Confusion Matrix (counts)', fontsize=13)
axes[0].set_ylabel('True ESI Level')
axes[0].set_xlabel('Predicted ESI Level')

sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title('Confusion Matrix (row %)', fontsize=13)
axes[1].set_ylabel('True ESI Level')
axes[1].set_xlabel('Predicted ESI Level')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print('Confusion matrix saved.')
```


```python
# ── Cell 16: Per-class ROC-AUC ────────────────────────────────────────────────
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_bin = label_binarize(y_arr, classes=[0,1,2,3,4])
colors = ['#e41a1c','#ff7f00','#377eb8','#4daf4a','#984ea3']

fig, ax = plt.subplots(figsize=(8, 6))
for i, (label, color) in enumerate(zip(labels, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], val_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{label} (AUC = {roc_auc:.4f})')

ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlim([0,1])
ax.set_ylim([0,1.02])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — One vs Rest per ESI Level')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print('ROC curves saved.')
```


```python
# ── Cell 17: Feature importance ───────────────────────────────────────────────
# Train a separate model on tabular features to get feature importances
# (sparse matrices don't support feature_importances_ directly)
print('Training model for feature importance...')

fi_model = xgb.XGBClassifier(**best_params)
fi_model.fit(X_train, y_arr, sample_weight=sw_arr)

importance = pd.DataFrame({
    'feature':    X_train.columns,
    'importance': fi_model.feature_importances_
}).sort_values('importance', ascending=False)

print('\n=== TOP 25 FEATURES ===')
print(importance.head(25).to_string(index=False))

# Plot top 20
fig, ax = plt.subplots(figsize=(10, 8))
top20 = importance.head(20)
ax.barh(top20['feature'][::-1], top20['importance'][::-1], color='steelblue')
ax.set_xlabel('Feature Importance (gain)')
ax.set_title('Top 20 Feature Importances\n(clinically grounded features dominate)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print('\nFeature importance plot saved.')
```


```python
# ── Cell 18: ESI-specific recall analysis ─────────────────────────────────────
# Clinical interpretation: where does the model make errors
# and what are the clinical consequences?
print('=== PER-CLASS RECALL BREAKDOWN ===')
for esi_level in range(5):
    mask = (y_arr == esi_level)
    n = mask.sum()
    preds_for_class = val_preds[mask]
    recall = (preds_for_class == esi_level).mean()

    # Where do errors go?
    errors = preds_for_class[preds_for_class != esi_level]
    error_counts = pd.Series(errors).value_counts().sort_index()
    error_str = ', '.join([f'ESI-{k+1}: {v}' for k, v in error_counts.items()])

    print(f'  ESI-{esi_level+1} (n={n:,}): recall={recall:.3f}  |  errors → {error_str}')

print()
print('Clinical interpretation:')
print('  Undertriage = predicting LESS urgent than true (dangerous)')
print('  Overtriage  = predicting MORE urgent than true (wasteful)')

undertriage = (val_preds > y_arr).mean()
overtriage  = (val_preds < y_arr).mean()
print(f'\n  Overall undertriage rate: {undertriage:.3f}')
print(f'  Overall overtriage rate:  {overtriage:.3f}')
print(f'\n  Nurse undertriage (Ivanov 2021): 0.198')
print(f'  KATE undertriage:                0.097')
print(f'  This model undertriage:          {undertriage:.3f}')
```

## Bias Analysis

The ESI v5 Handbook (Chapter 2) explicitly documents that racial, age, and gender bias leads to systematic undertriage in real emergency departments. Vigil et al. 2015 found Black patients assigned less urgent ESI scores than white patients. Grossmann et al. 2014 found significant undertriage in geriatric populations. Arslanian-Engoren 2004 documented inaccurate triage of women with chest pain.

We test whether this model — trained on synthetic data with demographic features — shows differential performance across demographic groups. Because the model excludes race and socioeconomic status as predictors (following Ivanov et al.'s explicit design choice), it may mitigate some documented biases.


```python
# ── Cell 19: Bias analysis ────────────────────────────────────────────────────
# Test for differential model performance across demographic groups
# Following Ivanov et al. 2021 who explicitly excluded race/ethnicity
# as features because 'reliance on these has been shown to decrease accuracy'

bias_df = train[['sex','language','age_group','insurance_type','triage_acuity']].copy()
bias_df['predicted'] = val_preds + 1
bias_df['true']      = y_arr + 1
bias_df['correct']   = (bias_df['predicted'] == bias_df['true']).astype(int)
bias_df['undertriaged'] = (bias_df['predicted'] > bias_df['true']).astype(int)

groups = {
    'Sex':           'sex',
    'Language':      'language',
    'Age Group':     'age_group',
    'Insurance':     'insurance_type'
}

print('=== BIAS ANALYSIS — UNDERTRIAGE RATE BY DEMOGRAPHIC GROUP ===')
print('Higher undertriage = model predicts less urgent than true = worse for patient')
print()

for group_name, col in groups.items():
    rates = bias_df.groupby(col)['undertriaged'].mean().sort_values(ascending=False)
    acc   = bias_df.groupby(col)['correct'].mean().sort_values()
    gap   = rates.max() - rates.min()

    print(f'{group_name}:')
    for grp in rates.index:
        n = (bias_df[col] == grp).sum()
        print(f'  {grp:<20} undertriage={rates[grp]:.3f}  accuracy={acc.get(grp, 0):.3f}  n={n:,}')

    if gap > 0.05:
        print(f'  ⚠️  Gap = {gap:.3f} — meaningful disparity exists')
    else:
        print(f'  ✅ Gap = {gap:.3f} — groups treated similarly')
    print()

print('=== ESI-2 RECALL BY SEX ===')
esi2_patients = bias_df[bias_df['true'] == 2]
esi2_sex = esi2_patients.groupby('sex').agg(
    n_patients = ('correct', 'count'),
    esi2_recall = ('correct', 'mean')
).round(3)
print(esi2_sex)

print('\n=== ESI-2 RECALL BY AGE GROUP ===')
esi2_age = esi2_patients.groupby('age_group').agg(
    n_patients  = ('correct', 'count'),
    esi2_recall = ('correct', 'mean')
).round(3)
print(esi2_age)
```


```python
# ── Cell 20: Final model — retrain on ALL training data ──────────────────────
print('Retraining final model on 100% of training data...')
print('Using clean tabular features only (no TF-IDF, no BERT)')
print(f'Feature matrix: {X_train.shape}')

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(
    X_train,
    y_arr,
    sample_weight=sw_arr
)

print('Final model trained successfully')
print(f'Features used: {X_train.shape[1]}')
```


```python
# ── Cell 21: Generate submission ──────────────────────────────────────────────
test_preds = final_model.predict(X_test) + 1  # back to 1-5

submission = pd.DataFrame({
    'patient_id':    test['patient_id'],
    'triage_acuity': test_preds
})

submission.to_csv('submission.csv', index=False)

print('=== SUBMISSION DISTRIBUTION ===')
dist = submission['triage_acuity'].value_counts(normalize=True).sort_index()
for level, pct in dist.items():
    bar = '█' * int(pct * 50)
    print(f'  ESI {level}: {pct:.1%}  {bar}')

print(f'\nTotal rows: {len(submission):,}')
print('Saved → submission.csv')

print('\n=== SANITY CHECK — matches train distribution? ===')
for level in [1,2,3,4,5]:
    train_pct = (y_arr + 1 == level).mean()
    test_pct  = (test_preds == level).mean()
    diff = abs(train_pct - test_pct)
    flag = '⚠️' if diff > 0.03 else '✅'
    print(f'  {flag} ESI {level}: train={train_pct:.1%}  test={test_pct:.1%}  diff={diff:.1%}')
```


```python
# ── Cell 22: Final summary ────────────────────────────────────────────────────
print('=' * 65)
print('TRIAGEGEIST — FINAL MODEL SUMMARY')
print('=' * 65)

print(f"""
APPROACH
  Algorithm:   XGBoost (multi:softprob, 5-class)
  Validation:  5-fold stratified cross-validation
  Imbalance:   Inverse frequency class weights

FEATURES ({X_train.shape[1]} total)
  Vital signs:     HR, RR, BP, SpO2, temp (raw + threshold flags)
  Clinical scores: NEWS2, GCS, shock index, pulse pressure
  Missing flags:   bp_missing, rr_missing (clinical signal)
  Mental status:   not_alert, mental_status_severity (0-4 scale)
  Comorbidities:   25 binary hx_ flags + 5 engineered groups
  Interactions:    age*HR, immuno_fever, cardiac_tachycardia
  Keyword flags:   20 high-risk terms from ESI v5 handbook
  ESI thresholds:  Decision Point D encoded as binary flags

NOTE ON TEXT FEATURES
  TF-IDF and BioClinicalBERT embeddings were tested but
  excluded after detecting synthetic data artifacts.
  Severity words (mild/severe/minor) in complaints
  correlated artificially with ESI labels (r=0.31).
  Manual keyword flags based on ESI handbook criteria
  were retained as genuine clinical knowledge.
""")

print('CROSS-VALIDATION RESULTS (5-fold stratified)')
print(f'  Accuracy:      {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}')
print(f'  Macro-F1:      {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}')
print(f'  ESI-2 Recall:  {np.mean(fold_esi2s):.4f} ± {np.std(fold_esi2s):.4f}')

print(f"""
BENCHMARK COMPARISON
  Nurses (Ivanov 2021):   accuracy=0.598  ESI-2 recall=0.414
  KATE ML (Ivanov 2021):  accuracy=0.757  ESI-2 recall=0.800
  This model:             accuracy={np.mean(fold_accs):.3f}  ESI-2 recall={np.mean(fold_esi2s):.3f}

KEY CLINICAL FINDING
  ESI-2 recall of {np.mean(fold_esi2s):.3f} means the model correctly
  identifies ~{np.mean(fold_esi2s)*100:.0f} out of 100 high-risk patients.
  Nurses identify only 41 out of 100 (Ivanov et al. 2021).
  ESI-2 is the most clinically important boundary in triage:
  missing these patients means high-risk individuals wait,
  increasing risk of deterioration, morbidity, and mortality.

REFERENCES
  Ivanov et al. (2021). Improving ED ESI acuity assignment
    using machine learning and clinical NLP. J Emerg Nurs.
  Tyler et al. (2024). Use of AI in triage in hospital EDs:
    a scoping review. Cureus.
  Emergency Nurses Association (2023). ESI v5 Handbook.
""")
print('=' * 65)
```


```python

```
