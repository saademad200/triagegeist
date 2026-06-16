# Triagegeist: Predicting Emergency Triage Acuity with Gradient Boosting Ensembles

**Competition:** Triagegeist — AI in Emergency Triage
**Author:** Seunghwan Kim, MD

---

## Abstract

Emergency department (ED) triage determines the priority of patient care through the Emergency Severity Index (ESI), a 5-level system that relies on clinical judgment. Inconsistencies in triage decisions — particularly undertriage — can lead to adverse outcomes [1].

This notebook develops a **stacked ensemble of LightGBM, XGBoost, and CatBoost** to predict ESI acuity from structured intake data and free-text chief complaints. The pipeline achieves strong predictive performance while maintaining clinical interpretability through SHAP-based feature analysis.

**Key findings:**
- Chief complaint text is the dominant predictor of triage acuity, consistent with ESI's emphasis on presenting complaint severity
- Vital signs and NEWS2 scores provide complementary physiological signals, particularly for distinguishing ESI-2 from ESI-3
- The stacked ensemble with threshold optimisation outperforms each individual model, achieving near-perfect cross-validated QWK
- Severe undertriage (ESI-1/2 predicted as ESI-3+) is minimised, addressing the primary safety concern in triage AI


## 1. Clinical Background

### The Emergency Severity Index (ESI)

The ESI is a 5-level triage algorithm used across emergency departments worldwide [1]:

| ESI Level | Category | Clinical Criteria |
|:---------:|:---------|:-----------------|
| 1 | Immediate | Life-threatening, requires immediate intervention |
| 2 | Emergent | High risk, confused/lethargic/disoriented, severe pain |
| 3 | Urgent | Stable but likely needs multiple resources |
| 4 | Less Urgent | Needs one resource |
| 5 | Non-Urgent | No resources needed |

### Why AI in Triage?

Machine learning models trained on structured ED data have demonstrated the ability to match or exceed human triage accuracy [2]. Natural language processing of chief complaint text offers additional predictive power [4], and combining structured features with NLP has emerged as a promising approach [5].

**Undertriage** — assigning a lower acuity than warranted — is the primary safety concern in triage systems. A model that minimises undertriage while maintaining overall accuracy has direct clinical value.


## 2. Setup and Data Loading


```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import gc
import os
from scipy import sparse
from scipy.optimize import minimize

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (cohen_kappa_score, classification_report,
                             confusion_matrix, log_loss)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    GPU_AVAILABLE = result.returncode == 0
except FileNotFoundError:
    GPU_AVAILABLE = False
print(f'GPU available: {GPU_AVAILABLE}')
if GPU_AVAILABLE:
    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())

warnings.filterwarnings('ignore')

SEED = 2025
np.random.seed(SEED)

# Paths
DATA_PATH = '/kaggle/input/competitions/triagegeist/'
if not os.path.exists(DATA_PATH):
    DATA_PATH = '/kaggle/input/triagegeist/'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/'
print(f'Data path: {DATA_PATH}')

N_FOLDS = 5
TARGET = 'triage_acuity'

print('Libraries loaded.')
print(f'LightGBM {lgb.__version__} | XGBoost {xgb.__version__} | CatBoost {cb.__version__}')

```


```
# Load all datasets
train = pd.read_csv(DATA_PATH + 'train.csv')
test  = pd.read_csv(DATA_PATH + 'test.csv')
cc    = pd.read_csv(DATA_PATH + 'chief_complaints.csv')
hist  = pd.read_csv(DATA_PATH + 'patient_history.csv')
sample_sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')

print(f'Train:   {train.shape[0]:>6,} rows x {train.shape[1]} cols')
print(f'Test:    {test.shape[0]:>6,} rows x {test.shape[1]} cols')
print(f'CC:      {cc.shape[0]:>7,} rows x {cc.shape[1]} cols')
print(f'History: {hist.shape[0]:>7,} rows x {hist.shape[1]} cols')

# Merge all tables
train = train.merge(hist, on='patient_id', how='left').merge(
    cc[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
test = test.merge(hist, on='patient_id', how='left').merge(
    cc[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')

print(f'\nAfter merge - Train: {train.shape}, Test: {test.shape}')
train.head(3)

```

## 3. Exploratory Data Analysis

### 3.1 Target Distribution and Missing Values


```
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Target distribution
acu_counts = train[TARGET].value_counts().sort_index()
labels = ['ESI-1\n(Immediate)', 'ESI-2\n(Emergent)', 'ESI-3\n(Urgent)',
          'ESI-4\n(Less Urgent)', 'ESI-5\n(Non-Urgent)']
colors = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c', '#1976d2']

axes[0].bar(labels, acu_counts.values, color=colors, edgecolor='white')
axes[0].set_title('Triage Acuity Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(acu_counts.values):
    axes[0].text(i, v + 200, f'{v:,}\n({v/len(train)*100:.1f}%)', ha='center', fontsize=9)

# Missing values
miss = train.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
miss_pct = (miss / len(train) * 100).round(1)
axes[1].barh(miss_pct.index, miss_pct.values, color='steelblue')
axes[1].set_xlabel('Missing (%)')
axes[1].set_title('Missing Values', fontsize=13, fontweight='bold')
for i, v in enumerate(miss_pct.values):
    axes[1].text(v + 0.1, i, f'{v}%', va='center', fontsize=9)

plt.tight_layout()
plt.show()

print(f'pain_score = -1 (not recorded): {(train["pain_score"] == -1).sum():,} '
      f'({(train["pain_score"] == -1).mean()*100:.1f}%)')

```

### 3.2 Vital Signs by Acuity


```
# Vital signs across acuity levels — small multiples with individual scales
vitals_info = {
    'SBP': ('systolic_bp', 'mmHg', '#e53935'),
    'Heart Rate': ('heart_rate', 'bpm', '#fb8c00'),
    'Resp. Rate': ('respiratory_rate', '/min', '#7cb342'),
    'Temperature': ('temperature_c', '°C', '#00897b'),
    'SpO2': ('spo2', '%', '#1e88e5'),
    'GCS': ('gcs_total', 'pts', '#5e35b1'),
    'NEWS2': ('news2_score', 'pts', '#d81b60'),
}

acuity_levels = [1, 2, 3, 4, 5]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

for i, (name, (col, unit, color)) in enumerate(vitals_info.items()):
    ax = axes[i]
    meds = train.groupby(TARGET)[col].median()
    q25 = train.groupby(TARGET)[col].quantile(0.25)
    q75 = train.groupby(TARGET)[col].quantile(0.75)

    ax.fill_between(acuity_levels, q25.values, q75.values, alpha=0.15, color=color)
    ax.plot(acuity_levels, meds.values, 'o-', color=color, linewidth=2.5, markersize=7)

    for x, v in zip(acuity_levels, meds.values):
        ax.text(x, v, f'{v:.0f}' if v >= 10 else f'{v:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

    ax.set_title(f'{name} ({unit})', fontsize=11, fontweight='bold', color='#333')
    ax.set_xticks(acuity_levels)
    ax.set_xticklabels(['1', '2', '3', '4', '5'])
    ax.set_xlabel('ESI')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[7].axis('off')
axes[7].text(0.5, 0.6, 'Median ± IQR\nby ESI Acuity\n\nESI-1 = Most Critical\nESI-5 = Least Critical',
             ha='center', va='center', fontsize=11, color='#555',
             transform=axes[7].transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#ccc'))

fig.suptitle('Vital Signs Across Acuity Levels', fontsize=15, fontweight='bold', y=1.0)
plt.tight_layout()
plt.show()

print('Each vital sign shown on its own scale. Higher acuity (ESI-1) correlates with')
print('physiological derangement across all parameters, consistent with clinical expectations [6].')

```

### 3.3 Chief Complaint Analysis

The chief complaint field contains semi-structured text with a base condition followed by optional modifiers (severity, trajectory, duration). This structure is key to the feature engineering approach.


```
# Parse chief complaint structure
def parse_chief_complaint(text):
    """Extract base condition, severity modifier, and trajectory from CC text."""
    if pd.isna(text):
        return "unknown", 0, "none"

    parts = [p.strip() for p in re.split(r"[,\uff0c]", text)]
    base_condition = parts[0]

    severity = 0
    sev_high = ["severe", "massive"]
    sev_mod  = ["moderate"]
    sev_low  = ["mild", "minor", "light"]

    all_text = text.lower()
    for w in sev_high:
        if w in all_text:
            severity = 3; break
    if severity == 0:
        for w in sev_mod:
            if w in all_text:
                severity = 2; break
    if severity == 0:
        for w in sev_low:
            if w in all_text:
                severity = 1; break

    trajectory = "none"
    traj_map = {"worsening": "worsening", "improving": "improving",
                "intermittent": "intermittent", "constant": "constant",
                "acute": "acute", "chronic": "chronic"}
    for kw, val in traj_map.items():
        if kw in all_text:
            trajectory = val; break

    return base_condition, severity, trajectory

# Apply to both sets
parsed = train["chief_complaint_raw"].apply(parse_chief_complaint)
train["clean_condition"] = [p[0] for p in parsed]
train["complaint_severity"] = [p[1] for p in parsed]

parsed_test = test["chief_complaint_raw"].apply(parse_chief_complaint)
test["clean_condition"] = [p[0] for p in parsed_test]
test["complaint_severity"] = [p[1] for p in parsed_test]

print(f'Unique base conditions (train): {train["clean_condition"].nunique():,}')
print(f'Unique base conditions (test):  {test["clean_condition"].nunique():,}')

# Check condition -> acuity mapping
cond_stats = train.groupby("clean_condition")[TARGET].agg(["nunique", "count", "mean"])
single_acuity = (cond_stats["nunique"] == 1).sum()
print(f'\nConditions with single acuity value: {single_acuity} / {len(cond_stats)} '
      f'({single_acuity/len(cond_stats)*100:.1f}%)')
print("Chief complaint text has a very strong association with triage acuity.")

```


```
# Severity modifier distribution and acuity relationship
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sev_labels = ['None', 'Mild/Minor/Light', 'Moderate', 'Severe/Massive']
sev_counts = train['complaint_severity'].value_counts().sort_index()
axes[0].bar(sev_labels, sev_counts.values, color=['grey', '#4caf50', '#ff9800', '#d32f2f'])
axes[0].set_title('Complaint Severity Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')

ct = pd.crosstab(train['complaint_severity'], train[TARGET], normalize='index')
ct.index = sev_labels
ct.columns = [f'ESI-{c}' for c in ct.columns]
ct.plot(kind='bar', stacked=True, ax=axes[1], color=colors)
axes[1].set_title('Acuity Distribution by Severity Modifier', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Proportion')
axes[1].set_xticklabels(sev_labels, rotation=0)
axes[1].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()

```


```
# Unseen chief complaints in test set
train_texts = set(train['chief_complaint_raw'].dropna())
test_texts = set(test['chief_complaint_raw'].dropna())
unseen_texts = test_texts - train_texts
print(f'Unseen chief complaint texts in test: {len(unseen_texts)}')

test_unseen_mask = test['chief_complaint_raw'].isin(unseen_texts)
print(f'Test patients with unseen CC: {test_unseen_mask.sum()} ({test_unseen_mask.mean()*100:.2f}%)')
print('These will be handled via TF-IDF partial matching and global mean fallback.')

```

### 3.4 Train-Only Columns: `disposition` and `ed_los_hours`

The training set contains two columns absent from the test set:

| Column | Description | Why Excluded |
|--------|-------------|-------------|
| `disposition` | Patient outcome (discharged, admitted, etc.) | **Target leakage** — determined *after* triage, not available at triage time |
| `ed_los_hours` | ED length of stay in hours | **Target leakage** — measured *after* triage, not available at triage time |

These columns represent post-triage outcomes and must never be used as input features. Including them would constitute **target leakage**, as the model would learn from information unavailable at the point of triage decision-making. They are dropped in the feature engineering step below.

### 3.5 Observations on Data Characteristics

Several patterns are consistent with synthetic data generation:
- The chief complaint text follows a highly regular structure (base condition + optional modifiers)
- Vital signs show clean monotonic trends by acuity level
- Nurse and site identifiers show negligible systematic variation in triage patterns

These observations inform our modeling choices: we prioritise chief complaint encoding as the primary feature source, with vital signs as complementary physiological signals.

## 4. Feature Engineering

Our feature engineering follows three principles:
1. **Chief complaint text is the primary signal** — encoded via fold-aware target encoding (LGB/XGB) or native categorical handling (CatBoost) [4]
2. **Vital signs are kept as raw continuous values** — GBDT models find optimal splits automatically; binary thresholds are unnecessary [7]
3. **Minimal derived features** — only where there is clear clinical motivation

### Excluded Columns

| Column | Reason for Exclusion |
|--------|---------------------|
| `disposition` | **Target leakage** — post-triage outcome (discharge/admit/death). Not in test set |
| `ed_los_hours` | **Target leakage** — post-triage measurement (ED length of stay). Not in test set |
| `triage_acuity` | Target variable (prediction target) |
| `patient_id` | Identifier, not a feature |
| `triage_nurse_id` | Used only via target encoding; raw ID excluded (negligible variation, std=0.020) |
| `site_id` | Used only via target encoding; raw ID excluded (negligible variation, std=0.007) |
| `complaint_temporality` | MI ~ 0.0004 with target — no predictive value |
| `complaint_trajectory` | MI ~ 0.0004 with target — no predictive value |



```
# ========================================================================
# 4a. Chief Complaint - Target Encoding (fold-aware, Bayesian smoothing)
# ========================================================================

def bayesian_target_encode(train_df, test_df, col, target, n_folds=5, seed=42,
                           min_samples=10):
    """Fold-aware Bayesian target encoding with smoothing and global mean fallback."""
    global_mean = train_df[target].mean()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    train_encoded = pd.Series(np.nan, index=train_df.index, dtype=float)
    y = train_df[target].values

    for tr_idx, val_idx in skf.split(train_df, y):
        fold_train = train_df.iloc[tr_idx]
        stats = fold_train.groupby(col)[target].agg(["mean", "count"])
        smoothing = 1 / (1 + np.exp(-(stats["count"] - min_samples) / 5))
        stats["smoothed"] = smoothing * stats["mean"] + (1 - smoothing) * global_mean
        mapping = stats["smoothed"].to_dict()
        train_encoded.iloc[val_idx] = train_df.iloc[val_idx][col].map(mapping)

    train_encoded = train_encoded.fillna(global_mean)

    full_stats = train_df.groupby(col)[target].agg(["mean", "count"])
    smoothing = 1 / (1 + np.exp(-(full_stats["count"] - min_samples) / 5))
    full_stats["smoothed"] = smoothing * full_stats["mean"] + (1 - smoothing) * global_mean
    test_encoded = test_df[col].map(full_stats["smoothed"].to_dict()).fillna(global_mean)

    return train_encoded, test_encoded


# Apply target encoding to clean_condition
train["condition_target_enc"], test["condition_target_enc"] = bayesian_target_encode(
    train, test, "clean_condition", TARGET, n_folds=N_FOLDS, seed=SEED
)

# Nurse and site target encoding (expected low impact)
train["nurse_target_enc"], test["nurse_target_enc"] = bayesian_target_encode(
    train, test, "triage_nurse_id", TARGET, n_folds=N_FOLDS, seed=SEED
)
train["site_target_enc"], test["site_target_enc"] = bayesian_target_encode(
    train, test, "site_id", TARGET, n_folds=N_FOLDS, seed=SEED
)

print("Target encoding applied: condition, nurse, site")
print(f'  condition_target_enc range: [{train["condition_target_enc"].min():.3f}, '
      f'{train["condition_target_enc"].max():.3f}]')

```


```
# ========================================================================
# 4b. Vital Signs - Raw + Minimal Derived
# ========================================================================

for df in [train, test]:
    # pain_score: -1 -> NaN + flag
    df['pain_not_recorded'] = (df['pain_score'] == -1).astype(np.int8)
    df.loc[df['pain_score'] == -1, 'pain_score'] = np.nan
    df['high_pain'] = (df['pain_score'] >= 7).astype(np.int8)

    # Impute pain_score by age_group median
    df['pain_score'] = df.groupby('age_group')['pain_score'].transform(
        lambda x: x.fillna(x.median()))

    # Missing flags for vitals
    for col in ['systolic_bp', 'diastolic_bp', 'mean_arterial_pressure',
                'pulse_pressure', 'respiratory_rate', 'temperature_c', 'shock_index']:
        df[f'{col}_missing'] = df[col].isnull().astype(np.int8)

    # Vitals missing count
    vital_miss_cols = ['systolic_bp', 'diastolic_bp', 'respiratory_rate', 'temperature_c']
    df['vitals_missing_count'] = df[vital_miss_cols].isnull().sum(axis=1).astype(np.int8)

    # Impute vitals: age_group x shift median, then global median
    impute_cols = ['systolic_bp', 'diastolic_bp', 'mean_arterial_pressure',
                   'pulse_pressure', 'shock_index', 'respiratory_rate', 'temperature_c']
    for col in impute_cols:
        df[col] = df.groupby(['age_group', 'shift'])[col].transform(
            lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

print('Vital sign processing complete.')
print(f'  Missing flags created: {sum(1 for c in train.columns if c.endswith("_missing"))}')

```


```
# ========================================================================
# 4c. Patient Characteristics - Minimal Derived
# ========================================================================

hx_cols = [c for c in train.columns if c.startswith('hx_')]

for df in [train, test]:
    # Frailty indicator (0-3)
    df['frailty_indicator'] = (
        (df['age'] >= 65).astype(int) +
        df['hx_heart_failure'].fillna(0).astype(int) +
        df['hx_ckd'].fillna(0).astype(int) +
        (df['num_comorbidities'] >= 5).astype(int)
    ).clip(0, 3).astype(np.int8)

    # Immunocompromised: OR combination
    immuno_cols = ['hx_hiv', 'hx_immunosuppressed', 'hx_malignancy']
    df['immunocompromised'] = df[immuno_cols].max(axis=1).astype(np.int8)

    # Arrival hour cyclical encoding
    df['arrival_hour_sin'] = np.sin(2 * np.pi * df['arrival_hour'] / 24)
    df['arrival_hour_cos'] = np.cos(2 * np.pi * df['arrival_hour'] / 24)

print('Patient characteristic features created.')

```


```
# ========================================================================
# 4d. TF-IDF on Chief Complaint Text
# ========================================================================

train['chief_complaint_raw'] = train['chief_complaint_raw'].fillna('unknown')
test['chief_complaint_raw'] = test['chief_complaint_raw'].fillna('unknown')

tfidf = TfidfVectorizer(
    max_features=500, ngram_range=(1, 3), min_df=3, sublinear_tf=True
)
tfidf_train = tfidf.fit_transform(train['chief_complaint_raw'])
tfidf_test  = tfidf.transform(test['chief_complaint_raw'])

tfidf_feat_names = [f'tfidf_{c}' for c in tfidf.get_feature_names_out()]
print(f'TF-IDF features: {len(tfidf_feat_names)}')
print('Vocabulary coverage for unseen CC: partial matching via individual n-grams')

```


```
# ========================================================================
# 4e. Categorical Encoding + Feature Set Assembly
# ========================================================================

# Label encode categoricals for LGB/XGB
cat_cols = ['arrival_mode', 'arrival_day', 'arrival_season', 'shift', 'age_group',
            'sex', 'language', 'insurance_type', 'transport_origin',
            'pain_location', 'mental_status_triage', 'chief_complaint_system']

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[f'{col}_le'] = le.transform(train[col].astype(str))
    test[f'{col}_le']  = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Columns NOT to use as features
drop_cols = ['patient_id', 'triage_nurse_id', 'site_id',
             'disposition', 'ed_los_hours', TARGET,
             'chief_complaint_raw', 'clean_condition'] + cat_cols

# Structured features (shared across models)
struct_cols = [c for c in train.columns if c not in drop_cols]
print(f'Structured features: {len(struct_cols)}')

# Build arrays
y_train = train[TARGET].values

X_struct_train = train[struct_cols].values.astype(np.float32)
X_struct_test  = test[struct_cols].values.astype(np.float32)

# Combined with TF-IDF for LGB/XGB
X_lgb_train = np.hstack([X_struct_train, tfidf_train.toarray().astype(np.float32)])
X_lgb_test  = np.hstack([X_struct_test,  tfidf_test.toarray().astype(np.float32)])
lgb_feat_names = struct_cols + tfidf_feat_names

# CatBoost: use clean_condition as native categorical (no target enc needed)
cb_cat_features = ['arrival_mode', 'arrival_day', 'arrival_season', 'shift',
                   'age_group', 'sex', 'language', 'insurance_type',
                   'transport_origin', 'pain_location', 'mental_status_triage',
                   'chief_complaint_system', 'clean_condition']

# For CatBoost, use original categoricals (not label-encoded) + remove condition_target_enc
cb_struct_cols = [c for c in struct_cols if c not in
                  [f'{cat}_le' for cat in cat_cols] + ['condition_target_enc']]

cb_train_df = train[cb_struct_cols + cb_cat_features].copy()
cb_test_df  = test[cb_struct_cols + cb_cat_features].copy()
for col in cb_cat_features:
    cb_train_df[col] = cb_train_df[col].astype(str)
    cb_test_df[col]  = cb_test_df[col].astype(str)
cb_feat_names = list(cb_train_df.columns)
cb_cat_indices = [cb_feat_names.index(c) for c in cb_cat_features]

print(f'LGB/XGB features: {X_lgb_train.shape[1]} (struct + TF-IDF)')
print(f'CatBoost features: {cb_train_df.shape[1]} (struct + native categoricals)')
print(f'CatBoost categorical indices: {len(cb_cat_indices)}')

# Verify no leakage
assert 'disposition' not in struct_cols, 'LEAKAGE: disposition in features!'
assert 'ed_los_hours' not in struct_cols, 'LEAKAGE: ed_los_hours in features!'
print('\nNo leakage detected. Feature engineering complete.')

```

## 5. Modeling

### Strategy
1. **Layer 1:** Three GBDT models (LightGBM, XGBoost, CatBoost) each with 5-fold stratified CV
2. **Layer 2:** Logistic Regression stacking on OOF probabilities [3]
3. **Threshold optimisation** for QWK on the stacked OOF predictions


### 5.1 Hyperparameters

Hyperparameters were tuned via Optuna (30/20/20 trials with 5-fold stratified CV) in a preliminary experiment. The best parameters are used directly here for reproducibility and runtime efficiency.

| Model | Key Parameters |
|-------|---------------|
| **LightGBM** | lr=0.0143, leaves=219, depth=8, subsample=0.663, colsample=0.546 |
| **XGBoost** | lr=0.0401, depth=10, min_child=11, subsample=0.714, colsample=0.734 |
| **CatBoost** | lr=0.0384, depth=8, l2_reg=2.308, bagging_temp=0.037, border=168 |



```
# ========================================================================
# QWK metric helper
# ========================================================================

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# ========================================================================
# Hyperparameters tuned via Optuna (30/20/20 trials, 5-fold CV)
# ========================================================================

best_lgb = {
    "objective": "multiclass", "num_class": 5, "metric": "multi_logloss",
    "verbosity": -1, "n_jobs": -1, "random_state": SEED,
    "n_estimators": 1500, "class_weight": "balanced",
    "device": "gpu" if GPU_AVAILABLE else "cpu", "gpu_use_dp": False,
    "learning_rate": 0.0143, "num_leaves": 219, "max_depth": 8,
    "min_child_samples": 42, "subsample": 0.663, "colsample_bytree": 0.546,
    "reg_alpha": 0.00175, "reg_lambda": 0.131, "min_split_gain": 0.168,
}

best_xgb = {
    "objective": "multi:softprob", "num_class": 5, "eval_metric": "mlogloss",
    "verbosity": 0, "nthread": -1, "random_state": SEED,
    "n_estimators": 1500, "tree_method": "hist",
    "device": "cuda" if GPU_AVAILABLE else "cpu", "early_stopping_rounds": 50,
    "learning_rate": 0.0401, "max_depth": 10, "min_child_weight": 11,
    "subsample": 0.714, "colsample_bytree": 0.734,
    "reg_alpha": 1.115, "reg_lambda": 4.903, "gamma": 1.965,
}

best_cb = {
    "loss_function": "MultiClass", "classes_count": 5, "eval_metric": "MultiClass",
    "random_seed": SEED, "verbose": 0, "iterations": 1500,
    "task_type": "GPU" if GPU_AVAILABLE else "CPU", "devices": "0",
    "auto_class_weights": "Balanced", "early_stopping_rounds": 50,
    "learning_rate": 0.0384, "depth": 8, "l2_leaf_reg": 2.308,
    "bagging_temperature": 0.037, "random_strength": 9.622, "border_count": 168,
}

print("Hyperparameters loaded (tuned via Optuna in preliminary experiment).")
print(f"  LGB: lr={best_lgb['learning_rate']}, leaves={best_lgb['num_leaves']}, depth={best_lgb['max_depth']}")
print(f"  XGB: lr={best_xgb['learning_rate']}, depth={best_xgb['max_depth']}, min_child={best_xgb['min_child_weight']}")
print(f"  CB:  lr={best_cb['learning_rate']}, depth={best_cb['depth']}, l2_reg={best_cb['l2_leaf_reg']}")

```

### 5.2 Full Training with Best Parameters (OOF Collection)


```
# ========================================================================
# Train all 3 models with best params, collect OOF predictions
# ========================================================================

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Storage
oof_lgb  = np.zeros((len(y_train), 5))
oof_xgb  = np.zeros((len(y_train), 5))
oof_cb   = np.zeros((len(y_train), 5))
test_lgb = np.zeros((len(test), 5))
test_xgb = np.zeros((len(test), 5))
test_cb  = np.zeros((len(test), 5))

lgb_models = []
xgb_models = []
cb_models  = []

qwk_lgb, qwk_xgb, qwk_cb = [], [], []

# -- LightGBM -----------------------------------------------------------
print("Training LightGBM...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_lgb_train, y_train)):
    model = lgb.LGBMClassifier(**best_lgb)
    model.fit(
        X_lgb_train[tr_idx], y_train[tr_idx],
        eval_set=[(X_lgb_train[val_idx], y_train[val_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_lgb[val_idx] = model.predict_proba(X_lgb_train[val_idx])
    test_lgb += model.predict_proba(X_lgb_test) / N_FOLDS
    lgb_models.append(model)

    pred = np.argmax(oof_lgb[val_idx], axis=1) + 1
    q = quadratic_weighted_kappa(y_train[val_idx], pred)
    qwk_lgb.append(q)
    print(f"  Fold {fold+1} QWK: {q:.4f} | Best iter: {model.best_iteration_}")

print(f"  LGB CV QWK: {np.mean(qwk_lgb):.4f} +/- {np.std(qwk_lgb):.4f}\n")

# -- XGBoost -------------------------------------------------------------
class_counts = np.bincount(y_train - 1)
total = len(y_train)
sample_weights = np.array([total / (5 * class_counts[y - 1]) for y in y_train])

print("Training XGBoost...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_lgb_train, y_train)):
    model = xgb.XGBClassifier(**best_xgb)
    model.fit(
        X_lgb_train[tr_idx], y_train[tr_idx] - 1,
        eval_set=[(X_lgb_train[val_idx], y_train[val_idx] - 1)],
        sample_weight=sample_weights[tr_idx],
        verbose=False
    )
    oof_xgb[val_idx] = model.predict_proba(X_lgb_train[val_idx])
    test_xgb += model.predict_proba(X_lgb_test) / N_FOLDS
    xgb_models.append(model)

    pred = np.argmax(oof_xgb[val_idx], axis=1) + 1
    q = quadratic_weighted_kappa(y_train[val_idx], pred)
    qwk_xgb.append(q)
    print(f"  Fold {fold+1} QWK: {q:.4f}")

print(f"  XGB CV QWK: {np.mean(qwk_xgb):.4f} +/- {np.std(qwk_xgb):.4f}\n")

# -- CatBoost ------------------------------------------------------------
print("Training CatBoost...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(cb_train_df, y_train)):
    pool_tr = cb.Pool(cb_train_df.iloc[tr_idx], y_train[tr_idx] - 1,
                      cat_features=cb_cat_indices)
    pool_val = cb.Pool(cb_train_df.iloc[val_idx], y_train[val_idx] - 1,
                       cat_features=cb_cat_indices)

    model = cb.CatBoostClassifier(**best_cb)
    model.fit(pool_tr, eval_set=pool_val, verbose=0)

    oof_cb[val_idx] = model.predict_proba(cb_train_df.iloc[val_idx])
    test_cb += model.predict_proba(cb_test_df) / N_FOLDS
    cb_models.append(model)

    pred = np.argmax(oof_cb[val_idx], axis=1) + 1
    q = quadratic_weighted_kappa(y_train[val_idx], pred)
    qwk_cb.append(q)
    print(f"  Fold {fold+1} QWK: {q:.4f}")

print(f"  CB CV QWK: {np.mean(qwk_cb):.4f} +/- {np.std(qwk_cb):.4f}")

```

### 5.3 complaint_severity Marginal Gain Verification

We verify whether the severity ordinal feature provides marginal gain over condition_target_enc alone, since 99.7% of conditions map deterministically to a single acuity.


```
# Quick ablation: LGB with vs without complaint_severity
sev_col_idx = struct_cols.index("complaint_severity")

X_no_sev = np.delete(X_lgb_train, sev_col_idx, axis=1)

scores_with, scores_without = [], []
for tr_idx, val_idx in skf.split(X_lgb_train, y_train):
    m1 = lgb.LGBMClassifier(**best_lgb)
    m1.fit(X_lgb_train[tr_idx], y_train[tr_idx],
           eval_set=[(X_lgb_train[val_idx], y_train[val_idx])],
           callbacks=[lgb.early_stopping(50, verbose=False)])
    p1 = np.argmax(m1.predict_proba(X_lgb_train[val_idx]), axis=1) + 1
    scores_with.append(quadratic_weighted_kappa(y_train[val_idx], p1))

    m2 = lgb.LGBMClassifier(**best_lgb)
    m2.fit(X_no_sev[tr_idx], y_train[tr_idx],
           eval_set=[(X_no_sev[val_idx], y_train[val_idx])],
           callbacks=[lgb.early_stopping(50, verbose=False)])
    p2 = np.argmax(m2.predict_proba(X_no_sev[val_idx]), axis=1) + 1
    scores_without.append(quadratic_weighted_kappa(y_train[val_idx], p2))

print(f"With complaint_severity:    QWK = {np.mean(scores_with):.4f} +/- {np.std(scores_with):.4f}")
print(f"Without complaint_severity: QWK = {np.mean(scores_without):.4f} +/- {np.std(scores_without):.4f}")
delta = np.mean(scores_with) - np.mean(scores_without)
print(f"Delta: {delta:+.4f}")
if abs(delta) < 0.001:
    print("Marginal gain is negligible - severity ordinal retained for interpretability.")
else:
    print("Severity ordinal provides meaningful marginal gain.")

```

### 5.4 TF-IDF Marginal Gain Verification


```
# Quick ablation: LGB with vs without TF-IDF features
n_struct = len(struct_cols)
X_no_tfidf = X_lgb_train[:, :n_struct]

scores_tfidf = []
for tr_idx, val_idx in skf.split(X_no_tfidf, y_train):
    m = lgb.LGBMClassifier(**best_lgb)
    m.fit(X_no_tfidf[tr_idx], y_train[tr_idx],
          eval_set=[(X_no_tfidf[val_idx], y_train[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    p = np.argmax(m.predict_proba(X_no_tfidf[val_idx]), axis=1) + 1
    scores_tfidf.append(quadratic_weighted_kappa(y_train[val_idx], p))

print(f"With TF-IDF:    QWK = {np.mean(scores_with):.4f}")
print(f"Without TF-IDF: QWK = {np.mean(scores_tfidf):.4f}")
delta = np.mean(scores_with) - np.mean(scores_tfidf)
print(f"Delta: {delta:+.4f}")
print("TF-IDF provides partial matching for unseen chief complaints and NLP depth.")

```

### 5.5 Stacking - Meta-Learner


```
# ========================================================================
# Layer 2: Logistic Regression on OOF probabilities (15 features)
# ========================================================================

# Stack OOF: 5 classes x 3 models = 15 features
X_stack_train = np.hstack([oof_lgb, oof_xgb, oof_cb])
X_stack_test  = np.hstack([test_lgb, test_xgb, test_cb])

print(f"Stacking features: {X_stack_train.shape[1]} (5 classes x 3 models)")
print("Meta-learner: Logistic Regression (regularised) - minimal ensemble for robustness")

# Train meta-learner with full OOF (for final test prediction)
meta_model = LogisticRegression(
    C=1.0, max_iter=1000, 
    solver="lbfgs", random_state=SEED
)
meta_model.fit(X_stack_train, y_train - 1)  # 0-indexed

# OOF predictions from meta-learner (via CV for fair evaluation)
oof_stack = np.zeros((len(y_train), 5))
for tr_idx, val_idx in skf.split(X_stack_train, y_train):
    meta_fold = LogisticRegression(
        C=1.0, max_iter=1000, 
        solver="lbfgs", random_state=SEED
    )
    meta_fold.fit(X_stack_train[tr_idx], y_train[tr_idx] - 1)
    oof_stack[val_idx] = meta_fold.predict_proba(X_stack_train[val_idx])

oof_stack_pred = np.argmax(oof_stack, axis=1) + 1
stack_qwk = quadratic_weighted_kappa(y_train, oof_stack_pred)
print(f"\nStacked OOF QWK (argmax): {stack_qwk:.4f}")

lgb_oof_qwk = quadratic_weighted_kappa(y_train, np.argmax(oof_lgb, axis=1) + 1)
xgb_oof_qwk = quadratic_weighted_kappa(y_train, np.argmax(oof_xgb, axis=1) + 1)
cb_oof_qwk  = quadratic_weighted_kappa(y_train, np.argmax(oof_cb,  axis=1) + 1)
print(f"  LGB OOF QWK: {lgb_oof_qwk:.4f}")
print(f"  XGB OOF QWK: {xgb_oof_qwk:.4f}")
print(f"  CB  OOF QWK: {cb_oof_qwk:.4f}")

```

### 5.6 Threshold Optimisation for QWK


```
# ========================================================================
# Ordinal threshold optimisation on stacked OOF probabilities
# ========================================================================

def optimise_thresholds(oof_probs, y_true, n_classes=5):
    """Find optimal ordinal thresholds to maximise QWK."""
    # Expected value (1-5)
    ev = oof_probs @ np.arange(1, n_classes + 1)

    def neg_qwk(thresholds):
        thresholds = np.sort(thresholds)
        preds = np.digitize(ev, thresholds) + 1
        return -cohen_kappa_score(y_true, preds, weights="quadratic")

    init = np.array([1.5, 2.5, 3.5, 4.5])
    best_qwk = -1
    best_thresh = init

    for offset in np.arange(-0.3, 0.31, 0.1):
        result = minimize(neg_qwk, init + offset, method="Nelder-Mead",
                         options={"maxiter": 5000, "xatol": 1e-5})
        if -result.fun > best_qwk:
            best_qwk = -result.fun
            best_thresh = np.sort(result.x)

    return best_thresh, best_qwk

thresholds, opt_qwk = optimise_thresholds(oof_stack, y_train)
print(f"Optimised thresholds: {thresholds.round(4)}")
print(f"QWK with optimal thresholds: {opt_qwk:.4f}")
print(f"QWK improvement over argmax: {opt_qwk - stack_qwk:+.4f}")

```

## 6. Results


```
# Final OOF predictions with optimised thresholds
ev_oof = oof_stack @ np.arange(1, 6)
final_oof = np.digitize(ev_oof, thresholds) + 1
final_qwk = quadratic_weighted_kappa(y_train, final_oof)

print(f"Final OOF QWK: {final_qwk:.4f}")
print(f"\nModel Summary:")
print(f"  LGB: {np.mean(qwk_lgb):.4f} -> XGB: {np.mean(qwk_xgb):.4f} -> "
      f"CB: {np.mean(qwk_cb):.4f} -> Stack: {stack_qwk:.4f} -> Thresh-opt: {final_qwk:.4f}")

```


```
# Classification report
target_names = ["ESI-1", "ESI-2", "ESI-3", "ESI-4", "ESI-5"]
print("Classification Report (OOF):")
print(classification_report(y_train, final_oof, target_names=target_names))

```


```
# Baseline comparisons
# 1) Majority class baseline
majority_class = train[TARGET].mode()[0]
maj_pred = np.full_like(y_train, majority_class)
maj_qwk = quadratic_weighted_kappa(y_train, maj_pred)

# 2) Single model baselines (from OOF)
lgb_only_qwk = quadratic_weighted_kappa(y_train, np.argmax(oof_lgb, axis=1) + 1)
xgb_only_qwk = quadratic_weighted_kappa(y_train, np.argmax(oof_xgb, axis=1) + 1)
cb_only_qwk  = quadratic_weighted_kappa(y_train, np.argmax(oof_cb,  axis=1) + 1)

# 3) Undertriage rates
baselines = [
    ("Majority Class", maj_qwk, (maj_pred > y_train).mean()*100),
    ("LightGBM only", lgb_only_qwk, (np.argmax(oof_lgb,1)+1 > y_train).mean()*100),
    ("XGBoost only", xgb_only_qwk, (np.argmax(oof_xgb,1)+1 > y_train).mean()*100),
    ("CatBoost only", cb_only_qwk, (np.argmax(oof_cb,1)+1 > y_train).mean()*100),
    ("Stacked Ensemble", stack_qwk, (oof_stack_pred > y_train).mean()*100),
    ("+ Threshold Opt", final_qwk, (final_oof > y_train).mean()*100),
]
print("Model Comparison:")
print(f"{'Model':<22} {'QWK':>8} {'Undertriage%':>14}")
print("-" * 46)
for name, qwk, ut in baselines:
    print(f"{name:<22} {qwk:>8.4f} {ut:>13.1f}%")

```


```
# Confusion matrix
cm = confusion_matrix(y_train, final_oof)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            linewidths=0.5, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')

sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            linewidths=0.5, ax=axes[1], cbar_kws={'label': 'Row %'})
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix (Row-Normalised %)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

```


```
# Undertriage analysis
print("=== Undertriage Analysis ===")
print("Undertriage = predicted acuity > true acuity (patient assigned lower priority)\n")

undertriage_mask = final_oof > y_train
overtriage_mask  = final_oof < y_train
correct_mask     = final_oof == y_train

print(f"Correct:     {correct_mask.sum():>6,} ({correct_mask.mean()*100:.1f}%)")
print(f"Undertriage: {undertriage_mask.sum():>6,} ({undertriage_mask.mean()*100:.1f}%)")
print(f"Overtriage:  {overtriage_mask.sum():>6,} ({overtriage_mask.mean()*100:.1f}%)")

print("\nUndertriage rate by true acuity:")
for acuity in range(1, 5):
    mask = y_train == acuity
    ut = (final_oof[mask] > acuity).mean() * 100
    print(f"  ESI-{acuity}: {ut:.1f}%")

severe_ut = ((y_train <= 2) & (final_oof >= 3)).sum()
print(f"\nSevere undertriage (ESI-1/2 -> ESI-3+): {severe_ut} "
      f"({severe_ut / (y_train <= 2).sum() * 100:.2f}%)")

```


```
# Error analysis: most confused acuity pairs
print("=== Error Analysis ===\n")
errors = final_oof != y_train
error_pairs = list(zip(y_train[errors], final_oof[errors]))
from collections import Counter
pair_counts = Counter(error_pairs)
print("Most Common Misclassification Pairs (True -> Predicted):")
for (true_a, pred_a), count in pair_counts.most_common(10):
    direction = "UNDERTRIAGE" if pred_a > true_a else "overtriage"
    print(f"  ESI-{true_a} -> ESI-{pred_a}: {count:>4} cases ({direction})")

# Example misclassified patients
print("\n--- Example Misclassified Patients ---")
np.random.seed(42)
for (true_a, pred_a), _ in pair_counts.most_common(3):
    mask = (y_train == true_a) & (final_oof == pred_a)
    idx = np.random.choice(np.where(mask)[0])
    row = train.iloc[idx]
    print(f'\nTrue ESI-{true_a} -> Predicted ESI-{pred_a}:')
    print(f'  CC: "{row["chief_complaint_raw"]}"')
    print(f'  Age: {row["age"]}y, NEWS2: {row["news2_score"]:.0f}, GCS: {row["gcs_total"]:.0f}')
    probs = oof_stack[idx]
    print(f'  Probs: {" | ".join([f"ESI-{i+1}:{p:.3f}" for i,p in enumerate(probs)])}')

```

### Error Analysis Interpretation

The misclassification patterns reveal that errors concentrate at **adjacent acuity boundaries** (e.g., ESI-3 vs ESI-4), where the clinical distinction depends on resource estimation rather than physiological severity. This is consistent with known inter-rater variability in ESI triage [1].

Undertriage errors are clinically more concerning than overtriage. The example patients above illustrate cases where the model's probability distribution is spread across adjacent levels — these **low-confidence predictions** would benefit from flagging for senior clinician review in a decision support system.


```
# Calibration curve
fig, ax = plt.subplots(figsize=(8, 6))

for i, name in enumerate(target_names):
    y_bin = (y_train == (i + 1)).astype(int)
    prob = oof_stack[:, i]
    fraction, mean_pred = calibration_curve(y_bin, prob, n_bins=10, strategy='quantile')
    ax.plot(mean_pred, fraction, marker='o', label=name, color=colors[i])

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curves by Acuity Class', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

```


```
# Expected calibration error
ece = 0
for i in range(5):
    y_bin = (y_train == (i + 1)).astype(int)
    prob = oof_stack[:, i]
    fraction, mean_pred = calibration_curve(y_bin, prob, n_bins=10, strategy='quantile')
    ece += np.mean(np.abs(fraction - mean_pred))
ece /= 5
print(f'\nMean Expected Calibration Error (across 5 classes): {ece:.4f}')
print('Well-calibrated probabilities support safe clinical use — predicted confidence')
print('aligns with actual class frequency, enabling reliable uncertainty communication.')

```

## 7. Clinical Insights

### 7.1 SHAP Feature Importance


```
import shap

# SHAP on LightGBM (first fold) with a sample
np.random.seed(SEED)
sample_idx = np.random.choice(len(X_lgb_train), size=3000, replace=False)
X_sample = pd.DataFrame(X_lgb_train[sample_idx], columns=lgb_feat_names)

explainer = shap.TreeExplainer(lgb_models[0])
shap_values = explainer.shap_values(X_sample)

```


```
# SHAP bar plot - mean |SHAP| across all classes
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Overall importance (mean across classes)
if isinstance(shap_values, list):
    mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
else:
    mean_shap = np.abs(shap_values).mean(axis=2) if shap_values.ndim == 3 else np.abs(shap_values)

feat_importance = pd.Series(mean_shap.mean(axis=0), index=lgb_feat_names)
top20 = feat_importance.nlargest(20)

bar_colors = ['#1976d2' if n.startswith('tfidf_') else '#d32f2f' for n in top20.index]
axes[0].barh(top20.index[::-1], top20.values[::-1], color=bar_colors[::-1])
axes[0].set_xlabel('Mean |SHAP value|')
axes[0].set_title('Top 20 Features (All Classes)', fontsize=11, fontweight='bold')

# SHAP for ESI-1 (most critical class)
if isinstance(shap_values, list):
    shap_esi1 = shap_values[0]
else:
    shap_esi1 = shap_values[:, :, 0] if shap_values.ndim == 3 else shap_values

feat_imp_esi1 = pd.Series(np.abs(shap_esi1).mean(axis=0), index=lgb_feat_names)
top20_esi1 = feat_imp_esi1.nlargest(20)
bar_colors_esi1 = ['#1976d2' if n.startswith('tfidf_') else '#d32f2f' for n in top20_esi1.index]
axes[1].barh(top20_esi1.index[::-1], top20_esi1.values[::-1], color=bar_colors_esi1[::-1])
axes[1].set_xlabel('Mean |SHAP value|')
axes[1].set_title('Top 20 Features for ESI-1 (Immediate)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print('Chief complaint features (target encoding + TF-IDF terms) dominate the SHAP rankings.')
print('Vital signs contribute meaningful but secondary signal, consistent with clinical')
print('intuition that the presenting complaint drives initial triage assessment [1].')

```

### 7.2 Fairness Audit

We examine whether the model exhibits systematic performance differences across demographic subgroups.


```
# Fairness: QWK by demographic subgroups
demo_cols = {
    'sex': train['sex'].values,
    'age_group': train['age_group'].values,
    'insurance_type': train['insurance_type'].values,
    'language': train['language'].values,
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (col_name, col_vals) in enumerate(demo_cols.items()):
    groups = np.unique(col_vals)
    qwk_vals = []
    for g in groups:
        mask = col_vals == g
        if mask.sum() >= 50:
            q = quadratic_weighted_kappa(y_train[mask], final_oof[mask])
            qwk_vals.append((g, q, mask.sum()))

    qwk_vals.sort(key=lambda x: -x[1])
    names = [f'{v[0]}\n(n={v[2]:,})' for v in qwk_vals]
    vals = [v[1] for v in qwk_vals]

    axes[idx].barh(names, vals, color='steelblue')
    axes[idx].set_xlabel('QWK')
    axes[idx].set_title(f'QWK by {col_name}', fontsize=11, fontweight='bold')
    axes[idx].set_xlim(min(vals) - 0.05, 1.0)
    for i, v in enumerate(vals):
        axes[idx].text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.show()

print('In this dataset, no systematic demographic bias is observed in model performance.')
print('However, real-world clinical data has documented triage disparities by race, ethnicity,')
print('and socioeconomic status. Any deployment would require re-evaluation')
print('with real clinical data and ongoing monitoring for fairness.')

```

### 7.3 Nurse Variability


```
nurse_stats = train.groupby('triage_nurse_id').agg(
    n_patients=('patient_id', 'count'),
    mean_acuity=(TARGET, 'mean'),
    std_acuity=(TARGET, 'std')
)

print(f'Nurse count: {len(nurse_stats)}')
print(f'Mean acuity across nurses: {nurse_stats["mean_acuity"].mean():.3f} '
      f'+/- {nurse_stats["mean_acuity"].std():.3f}')
print(f'Acuity std range: [{nurse_stats["mean_acuity"].min():.3f}, '
      f'{nurse_stats["mean_acuity"].max():.3f}]')
print('\nNo clinically significant systematic variation observed across triage nurses.')

```

### 7.4 Decision Support Demo


```
# Example patient walkthrough
np.random.seed(42)
demo_indices = []
for acuity in [1, 2, 3, 4, 5]:
    candidates = np.where(y_train == acuity)[0]
    correct = candidates[final_oof[candidates] == acuity]
    if len(correct) > 0:
        demo_indices.append(np.random.choice(correct))

print("=== Decision Support Demo: Example Patients ===\n")
for idx in demo_indices:
    true_a = y_train[idx]
    pred_a = final_oof[idx]
    probs = oof_stack[idx]
    cc_text = train.iloc[idx]["chief_complaint_raw"]
    age = train.iloc[idx]["age"]
    sex = train.iloc[idx]["sex"]
    news2 = train.iloc[idx]["news2_score"]
    gcs = train.iloc[idx]["gcs_total"]

    print(f'Patient: {age}y {sex} | CC: "{cc_text}"')
    print(f"  NEWS2: {news2:.0f} | GCS: {gcs:.0f}")
    print(f"  True: ESI-{true_a} | Predicted: ESI-{pred_a}")
    print("  Probabilities: " + " | ".join([f"ESI-{i+1}: {p:.3f}" for i, p in enumerate(probs)]))
    print()

print("In a clinical decision support system, these probability distributions would be")
print("presented alongside the recommended acuity level, allowing the triage nurse to")
print("exercise clinical judgment with additional quantitative context.")

```

## 8. Discussion

### Key Observations

**Chief complaint dominance.** The model's reliance on chief complaint text aligns with the ESI algorithm's design, where the presenting complaint determines the initial branch point for acuity assessment [1]. This finding reinforces that text-based clinical decision support systems should prioritise complaint encoding.

**Vital signs as complementary signals.** While vital signs alone are insufficient for accurate triage prediction, they provide critical physiological context — particularly for distinguishing ESI-2 from ESI-3, where the boundary depends on resource estimation and physiological stability [1][6].

**Undertriage safety.** The model's low severe undertriage rate (ESI-1/2 misclassified as ESI-3+) suggests potential as a safety net for human triage decisions. However, any deployment would require prospective validation against patient outcomes, not just agreement with retrospectively assigned ESI labels [2].

### Clinical Deployment Considerations

A clinical triage AI system would need to:
1. **Operate as decision support**, not autonomous decision-making — the final triage decision must remain with the clinician
2. **Flag low-confidence predictions** where model probability is distributed across multiple acuity levels, triggering senior review
3. **Undergo prospective validation** comparing model-assisted vs. unassisted triage against patient outcomes (admission, ICU transfer, mortality)
4. **Maintain ongoing monitoring** for demographic fairness and performance drift as patient populations change


## 9. Limitations

1. **Synthetic data:** This dataset was synthetically generated. Model performance and feature relationships may not transfer directly to real clinical data. External validation with actual ED records is essential before any clinical deployment.

2. **Text structure:** The chief complaint field follows a regular structure that may not reflect the variability of real clinical free-text documentation. Real-world NLP would need to handle misspellings, abbreviations, and unstructured narratives.

3. **No clinical validation:** The model has not been evaluated against actual clinician triage decisions or patient outcomes. QWK measures agreement with assigned labels, not clinical appropriateness.

4. **Temporal dynamics:** The model uses snapshot data and does not account for patient deterioration over time, which is a key consideration in real triage.

5. **Demographic considerations:** While no bias was observed in this synthetic dataset, real clinical data has documented triage disparities by race and socioeconomic status. Deployment would require thorough fairness evaluation.

6. **Ensemble interpretability:** A stacked ensemble of three GBDT models is inherently less interpretable than a single model. While SHAP analysis provides global feature importance, individual prediction explanations involve aggregating across 15 base models (5 folds × 3 algorithms), which may limit clinical trust and adoption.

7. **Evaluation metric limitations:** QWK measures ordinal agreement but does not distinguish between clinically consequential errors (ESI-1 → ESI-3) and minor disagreements (ESI-4 → ESI-5). A cost-sensitive evaluation metric that penalises undertriage more heavily would better capture clinical safety requirements.


## 10. Submission


```
# Generate final predictions
test_stack_probs = meta_model.predict_proba(X_stack_test)
ev_test = test_stack_probs @ np.arange(1, 6)
test_labels = np.digitize(ev_test, thresholds) + 1
test_labels = test_labels.astype(int)

submission = pd.DataFrame({
    "patient_id": test["patient_id"],
    "triage_acuity": test_labels
})

print("Submission acuity distribution:")
print(submission["triage_acuity"].value_counts().sort_index())
print(f"\nTotal predictions: {len(submission):,}")

# Validate format
assert len(submission) == len(sample_sub), "Row count mismatch!"
assert set(submission.columns) == set(sample_sub.columns), "Column mismatch!"
assert submission["triage_acuity"].dtype in [np.int32, np.int64, int], "Not integer!"
assert submission["triage_acuity"].between(1, 5).all(), "Out of range!"

submission.to_csv("submission.csv", index=False)
print("\nsubmission.csv saved successfully.")
submission.head(10)

```

## 11. References

[1] Gilboy N, Tanabe P, Travers DA, Rosenau AM, Eitel DR. *Emergency Severity Index, Version 4: Implementation Handbook*. AHRQ Publication No. 05-0046-2. Agency for Healthcare Research and Quality.

[2] Raita Y, Goto T, Faridi MK, Brown DFM, Camargo CA Jr, Hasegawa K. Emergency department triage prediction of clinical outcomes using machine learning models. *Critical Care*. 2019;23:64. DOI: [10.1186/s13054-019-2351-7](https://doi.org/10.1186/s13054-019-2351-7)

[3] Hong WS, Haimovich AD, Taylor RA. Predicting hospital admission at emergency department triage using machine learning. *PLoS ONE*. 2018;13(7):e0201016. DOI: [10.1371/journal.pone.0201016](https://doi.org/10.1371/journal.pone.0201016)

[4] Stewart J, Lu J, Goudie A, et al. Applications of natural language processing at emergency department triage: A narrative review. *PLoS ONE*. 2023;18(12):e0279953. DOI: [10.1371/journal.pone.0279953](https://doi.org/10.1371/journal.pone.0279953)

[5] Porto BM. Improving triage performance in emergency departments using machine learning and natural language processing: a systematic review. *BMC Emergency Medicine*. 2024;24:219. DOI: [10.1186/s12873-024-01135-2](https://doi.org/10.1186/s12873-024-01135-2)

[6] Vergara P, Forero D, Bastidas A, Garcia JC, Blanco J, Azocar J, Bustos RH, Liebisch H. Validation of the NEWS-2 for adults in the emergency department. *Medicine*. 2021;100(40):e27325. DOI: [10.1097/MD.0000000000027325](https://doi.org/10.1097/MD.0000000000027325)

[7] Klug M, et al. A Gradient Boosting Machine Learning Model for Predicting Early Mortality in the Emergency Department. *JGIM*. 2020. DOI: [10.1007/s11606-019-05512-7](https://doi.org/10.1007/s11606-019-05512-7)

