# Triagegeist — Solution Notebook
**CV Accuracy: 0.9995 | 3-Tier Hybrid Predictor**

This notebook walks through the complete solution to the [Triagegeist](https://www.kaggle.com/competitions/triagegeist) competition — predicting emergency triage acuity (1–5) from 80k clinical ED records.

---

### The journey in one table

| Experiment | CV Accuracy | Change |
|---|---|---|
| Baseline (TF-IDF 50 features) | 0.8910 | — |
| TF-IDF 150 features | 0.9836 | +0.0926 |
| TF-IDF 300 features | 0.9919 | +0.0083 |
| TF-IDF 500 features | 0.9948 | +0.0029 |
| TF-IDF 1000 features | 0.9980 | +0.0032 |
| TF-IDF 2000 features | 0.9989 | +0.0009 |
| **+ Glaucoma tier (final)** | **0.9995** | **+0.0006** |

Chief complaint text was the dominant signal. TF-IDF on that text alone was worth 10.8 points of accuracy. Everything else — vitals engineering, hyperparameter tuning, comorbidity features — was secondary.

---

### Architecture

```
test row
   │
   ├─ Complaint in unambiguous lookup? ──YES──► direct label   (19,885 rows, 99.4%)
   │
   ├─ Glaucoma variant? ────────────────YES──► binary LightGBM (   76 rows,  0.4%)
   │
   └─ Unseen complaint ─────────────────────► full multiclass  (   39 rows,  0.2%)
```

## 1. Setup


```python
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ── Kaggle paths ────────────────────────────────────────────────────────────
import os
BASE = '/kaggle/input/competitions/triagegeist'


TRAIN_PATH      = f'{BASE}/train.csv'
TEST_PATH       = f'{BASE}/test.csv'
COMPLAINTS_PATH = f'{BASE}/chief_complaints.csv'
HISTORY_PATH    = f'{BASE}/patient_history.csv'
SAMPLE_PATH     = f'{BASE}/sample_submission.csv'

TARGET = 'triage_acuity'
RANDOM_STATE = 42

print('Libraries loaded.')
```

## 2. Load Data


```python
train      = pd.read_csv(TRAIN_PATH)
test       = pd.read_csv(TEST_PATH)
complaints = pd.read_csv(COMPLAINTS_PATH)
history    = pd.read_csv(HISTORY_PATH)
sample_sub = pd.read_csv(SAMPLE_PATH)

print(f'Train:      {train.shape}')
print(f'Test:       {test.shape}')
print(f'Complaints: {complaints.shape}')
print(f'History:    {history.shape}')
train.head(3)
```

## 3. Exploratory Data Analysis


```python
# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#1e1e1e')

counts = train[TARGET].value_counts().sort_index()
colors = ['#C9A84C' if i == counts.idxmax() else '#3e3e42' for i in counts.index]

ax = axes[0]
ax.set_facecolor('#252526')
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='#3e3e42', linewidth=0.5)
ax.set_xlabel('Triage Acuity', color='#858585')
ax.set_ylabel('Count', color='#858585')
ax.set_title('Target Distribution', color='#d4d4d4', pad=10)
ax.tick_params(colors='#858585')
for spine in ax.spines.values(): spine.set_color('#3e3e42')
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{count:,}', ha='center', va='bottom', color='#858585', fontsize=9)

# Missing values heatmap
ax2 = axes[1]
ax2.set_facecolor('#252526')
missing = train.isnull().mean().sort_values(ascending=False).head(15)
ax2.barh(missing.index, missing.values * 100, color='#C9A84C', alpha=0.8)
ax2.set_xlabel('Missing %', color='#858585')
ax2.set_title('Top 15 Columns by Missing %', color='#d4d4d4', pad=10)
ax2.tick_params(colors='#858585', labelsize=8)
for spine in ax2.spines.values(): spine.set_color('#3e3e42')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=120, bbox_inches='tight', facecolor='#1e1e1e')
plt.show()
```


```python
# Chief complaint text samples
train_c = train.merge(complaints[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')

print('Sample complaints by acuity:')
print('=' * 60)
for acuity in [1, 2, 3, 4, 5]:
    samples = train_c[train_c[TARGET] == acuity]['chief_complaint_raw'].dropna().head(3).tolist()
    print(f'\nAcuity {acuity}:')
    for s in samples:
        print(f'  • {s}')
```


```python
# How many unique complaint texts are there?
n_unique = train_c['chief_complaint_raw'].nunique()
print(f'Unique complaint texts in training: {n_unique:,}')

# How many map unambiguously to one acuity?
per_text = train_c.groupby('chief_complaint_raw')[TARGET].nunique()
n_unamb = (per_text == 1).sum()
n_amb = (per_text > 1).sum()
print(f'Unambiguous (always same acuity): {n_unamb:,} ({100*n_unamb/len(per_text):.1f}%)')
print(f'Ambiguous   (multiple acuities):  {n_amb:,}  ({100*n_amb/len(per_text):.1f}%)')
```

## 4. Feature Engineering

All feature engineering follows the **fit_params pattern** — encoders are fit on training data only and applied to validation/test, preventing any leakage.

```python
X_train_fe, fit_params = engineer_features(X_train, is_train=True)
X_val_fe = apply_features(X_val, fit_params)  # uses train-fitted encoders
```

**Key features:**
- TF-IDF bigrams on `chief_complaint_raw` (2000 features, sublinear TF) — dominant signal
- Frequency encoding for all categoricals
- Median imputation for missing vitals (fitted on train)
- Clinical interactions: `gcs × news2`, `resp × spo2`, `pain × news2`
- 24 binary comorbidity flags + burden sum

**Dropped:**
- `ed_los_hours`, `disposition` — post-triage outcomes, not in test (leakage)
- `triage_nurse_id`, `site_id` — high cardinality, won't generalise


```python
LEAKAGE_COLS = ['ed_los_hours', 'disposition']
DROP_COLS    = ['triage_nurse_id', 'site_id']

VITAL_COLS = ['systolic_bp', 'diastolic_bp', 'mean_arterial_pressure',
              'pulse_pressure', 'respiratory_rate', 'temperature_c',
              'shock_index', 'heart_rate', 'spo2', 'weight_kg', 'height_cm', 'bmi']

def engineer_features(df, is_train=True, fit_params=None,
                      complaints_df=None, history_df=None,
                      tfidf_features=2000):
    if fit_params is None:
        fit_params = {}

    # 1. Merge auxiliary tables
    if complaints_df is not None:
        df = df.merge(complaints_df[['patient_id', 'chief_complaint_raw']],
                      on='patient_id', how='left')
    if history_df is not None:
        df = df.merge(history_df, on='patient_id', how='left')

    # 2. Drop leakage + ID columns
    drop = LEAKAGE_COLS + DROP_COLS + ['patient_id']
    df = df.drop(columns=[c for c in drop if c in df.columns])

    # 3. Impute missing vitals
    for col in VITAL_COLS:
        if col in df.columns:
            if is_train:
                fit_params[f'median_{col}'] = df[col].median()
            df[col] = df[col].fillna(fit_params.get(f'median_{col}', 0))

    # 4. Clinical interaction features
    if 'respiratory_rate' in df.columns and 'spo2' in df.columns:
        df['resp_x_spo2'] = df['respiratory_rate'] * df['spo2']
    if 'heart_rate' in df.columns and 'systolic_bp' in df.columns:
        df['hr_x_sbp'] = df['heart_rate'] * df['systolic_bp']
    if 'gcs_total' in df.columns and 'news2_score' in df.columns:
        df['gcs_x_news2'] = df['gcs_total'] * df['news2_score']
    if 'pain_score' in df.columns and 'news2_score' in df.columns:
        df['pain_x_news2'] = df['pain_score'] * df['news2_score']
    if 'num_prior_ed_visits_12m' in df.columns and 'num_comorbidities' in df.columns:
        df['ed_visits_x_comorbid'] = df['num_prior_ed_visits_12m'] * df['num_comorbidities']

    # Comorbidity burden
    hx_cols = [c for c in df.columns if c.startswith('hx_')]
    if hx_cols:
        df['total_comorbidities'] = df[hx_cols].sum(axis=1)

    # 5. TF-IDF on chief complaint text
    text_col = 'chief_complaint_raw'
    if text_col in df.columns:
        df[text_col] = df[text_col].fillna('unknown')
        if is_train:
            tfidf = TfidfVectorizer(max_features=tfidf_features, ngram_range=(1, 2),
                                    sublinear_tf=True, min_df=2)
            tfidf.fit(df[text_col])
            fit_params['tfidf'] = tfidf
        tfidf = fit_params.get('tfidf')
        if tfidf is not None:
            mat = tfidf.transform(df[text_col]).toarray()
            tdf = pd.DataFrame(mat, columns=[f'tfidf_{i}' for i in range(mat.shape[1])],
                               index=df.index)
            df = pd.concat([df, tdf], axis=1)
        df = df.drop(columns=[text_col])

    # 6. Frequency encoding for categoricals
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if is_train:
            freq = df[col].value_counts(normalize=True)
            fit_params[f'freq_{col}'] = freq.to_dict()
        freq_map = fit_params.get(f'freq_{col}', {})
        df[col + '_enc'] = df[col].map(freq_map).fillna(0).astype(float)
    df = df.drop(columns=cat_cols)

    return df, fit_params


def apply_features(df, fit_params, complaints_df=None, history_df=None):
    df, _ = engineer_features(df, is_train=False, fit_params=fit_params,
                               complaints_df=complaints_df, history_df=history_df)
    return df

print('Feature engineering functions defined.')
```

## 5. TF-IDF Scaling — Where the Points Came From

Chief complaint text almost entirely determines triage acuity.

"Thunderclap headache" and "minor skin rash" carry very different acuity signals regardless of what the vitals show. Scaling TF-IDF from 50 to 2000 features was worth **+10.8 points of CV accuracy** — more than all other changes combined.

We ran one quick fold to validate each scale point:


```python
LGBM_PARAMS = {
    'objective': 'multiclass', 'num_class': 5, 'metric': 'multi_error',
    'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 63,
    'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'class_weight': 'balanced',
    'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1,
}

def quick_fold_score(n_features):
    """Single fold score with n TF-IDF features."""
    X = train.drop(columns=[TARGET])
    y = train[TARGET].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(skf.split(X, y))

    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y[train_idx], y[val_idx]

    X_tr_fe, fp = engineer_features(X_tr, is_train=True,
                                     complaints_df=complaints, history_df=history,
                                     tfidf_features=n_features)
    X_val_fe = apply_features(X_val, fp, complaints_df=complaints, history_df=history)

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_tr_fe, y_tr - 1,
              eval_set=[(X_val_fe, y_val - 1)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    preds = model.predict(X_val_fe) + 1
    return accuracy_score(y_val, preds)


# Show pre-computed results (re-run to reproduce — takes ~15 min)
tfidf_results = [
    (50,   0.8910),
    (150,  0.9836),
    (300,  0.9919),
    (500,  0.9948),
    (1000, 0.9980),
    (2000, 0.9989),
]

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#1e1e1e')
ax.set_facecolor('#252526')

xs = [r[0] for r in tfidf_results]
ys = [r[1] for r in tfidf_results]

ax.plot(xs, ys, color='#C9A84C', linewidth=2, marker='o', markersize=7, markerfacecolor='#1e1e1e', markeredgewidth=2)
ax.fill_between(xs, 0.88, ys, alpha=0.1, color='#C9A84C')

for x, y in zip(xs, ys):
    ax.annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=9, color='#d4d4d4')

ax.set_xlabel('TF-IDF Features', color='#858585')
ax.set_ylabel('CV Accuracy (1 fold)', color='#858585')
ax.set_title('TF-IDF Scaling — CV Accuracy vs Feature Count', color='#d4d4d4', pad=12)
ax.set_ylim(0.88, 1.005)
ax.tick_params(colors='#858585')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for spine in ax.spines.values(): spine.set_color('#3e3e42')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.3f}'))

plt.tight_layout()
plt.savefig('tfidf_scaling.png', dpi=120, bbox_inches='tight', facecolor='#1e1e1e')
plt.show()
print('Going from 50 → 150 TF-IDF features alone was worth +9.26 points of accuracy.')
```

## 6. Full 5-Fold CV — 2000 TF-IDF Features

With 2000 TF-IDF bigram features on the chief complaint text, full stratified 5-fold CV:


```python
X = train.drop(columns=[TARGET])
y = train[TARGET].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold_scores = []
all_val_idx = []
all_preds   = []
all_true    = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y[train_idx], y[val_idx]

    X_tr_fe, fp = engineer_features(X_tr, is_train=True,
                                     complaints_df=complaints, history_df=history,
                                     tfidf_features=2000)
    X_val_fe = apply_features(X_val, fp, complaints_df=complaints, history_df=history)

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_tr_fe, y_tr - 1,
              eval_set=[(X_val_fe, y_val - 1)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    preds = model.predict(X_val_fe) + 1
    score = accuracy_score(y_val, preds)
    fold_scores.append(score)
    all_val_idx.extend(val_idx)
    all_preds.extend(preds)
    all_true.extend(y_val)
    print(f'  Fold {fold}: {score:.4f}')

print(f'\nCV Score: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})')
```

## 7. Error Analysis — Every Mistake Traces to One Complaint

After reaching **0.9989**, the model had plateaued. We ran error analysis across all 5 CV folds.

**Finding: every single error came from the same complaint — variants of "acute angle closure glaucoma".**

This condition sits on the clinical boundary between acuity 1 (critical) and acuity 2 (emergent). The complaint text is identical across patients but the correct label differs based on vitals. The text alone cannot resolve it — NEWS2, GCS, and pain score can.


```python
# Reconstruct error analysis on the OOF predictions
oof_df = train.iloc[all_val_idx].copy()
oof_df = oof_df.merge(complaints[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
oof_df['pred'] = all_preds
oof_df['true'] = all_true
oof_df['error'] = oof_df['pred'] != oof_df['true']

errors = oof_df[oof_df['error']]
print(f'Total OOF errors: {len(errors)}')
print(f'\nErrors by chief complaint:')
print(errors['chief_complaint_raw'].value_counts().to_string())
print(f'\nTrue vs Predicted for errors:')
print(errors.groupby(['true', 'pred']).size().to_string())
```


```python
# Show the vital sign distributions for glaucoma acuity 1 vs 2
train_c = train.merge(complaints[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')

glaucoma_mask = train_c['chief_complaint_raw'].str.contains('glaucoma', case=False, na=False)
glaucoma_df = train_c[glaucoma_mask]
print(f'Glaucoma rows in training: {len(glaucoma_df)}')
print(f'Acuity distribution:\n{glaucoma_df[TARGET].value_counts().sort_index()}')

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.patch.set_facecolor('#1e1e1e')
fig.suptitle('Glaucoma rows: Acuity 1 vs 2 — vital sign distributions', color='#d4d4d4', y=1.02)

for ax, col in zip(axes, ['news2_score', 'gcs_total', 'pain_score']):
    ax.set_facecolor('#252526')
    for acuity, color in [(1, '#C9A84C'), (2, '#3e3e42')]:
        vals = glaucoma_df[glaucoma_df[TARGET] == acuity][col].dropna()
        ax.hist(vals, bins=15, alpha=0.7, label=f'Acuity {acuity}', color=color)
    ax.set_title(col, color='#d4d4d4')
    ax.legend(facecolor='#252526', labelcolor='#d4d4d4', edgecolor='#3e3e42')
    ax.tick_params(colors='#858585')
    for spine in ax.spines.values(): spine.set_color('#3e3e42')

plt.tight_layout()
plt.savefig('glaucoma_vitals.png', dpi=120, bbox_inches='tight', facecolor='#1e1e1e')
plt.show()
print('\nConclusion: vitals (especially news2_score and gcs_total) separate acuity 1 vs 2 for glaucoma.')
```

## 8. The 3-Tier Hybrid Predictor

The error analysis gives us a precise fix:

- **Tier 1** — For complaint texts that always map to one acuity in training → use a lookup. Exact, not approximate.
- **Tier 2** — For glaucoma variants (ambiguous text, vitals distinguish 1 vs 2) → dedicated binary LightGBM trained only on those 237 rows.
- **Tier 3** — For complaint texts never seen in training → full LightGBM multiclass fallback.


```python
GLAUCOMA_PARAMS = {
    'objective': 'binary', 'metric': 'binary_error',
    'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 15,
    'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1,
}

GLAUCOMA_FEATURES = [
    'news2_score', 'gcs_total', 'pain_score', 'heart_rate',
    'systolic_bp', 'diastolic_bp', 'respiratory_rate',
    'spo2', 'temperature_c', 'shock_index',
    'arrival_hour', 'num_comorbidities', 'num_prior_ed_visits_12m',
]


def build_hybrid_predictor(train_df, complaints_df):
    """Build all lookup tables and the glaucoma binary classifier."""
    tc = train_df.merge(complaints_df[['patient_id', 'chief_complaint_raw']],
                        on='patient_id', how='left')

    # Tier 1: unambiguous complaint texts
    per_text = tc.groupby('chief_complaint_raw')[TARGET].nunique()
    unamb = per_text[per_text == 1].index
    amb   = per_text[per_text > 1].index

    tier1 = (tc[tc['chief_complaint_raw'].isin(unamb)]
               .groupby('chief_complaint_raw')[TARGET].first().to_dict())

    # Tier 2: glaucoma binary classifier
    gl_train = tc[tc['chief_complaint_raw'].isin(amb)].copy()

    fit_meds = {}
    for col in GLAUCOMA_FEATURES:
        fit_meds[col] = gl_train[col].median()
        gl_train[col] = gl_train[col].fillna(fit_meds[col])

    gl_y = (gl_train[TARGET] == 1).astype(int).values
    gl_X = gl_train[GLAUCOMA_FEATURES].values

    gl_model = lgb.LGBMClassifier(**GLAUCOMA_PARAMS)
    gl_model.fit(gl_X, gl_y, callbacks=[lgb.log_evaluation(0)])

    return tier1, amb, gl_model, fit_meds


print('Building hybrid predictor on full training data...')
tier1, amb_texts, gl_model, gl_meds = build_hybrid_predictor(train, complaints)
print(f'Tier 1 lookup: {len(tier1):,} unambiguous complaint texts')
print(f'Tier 2 glaucoma texts: {len(amb_texts):,}')
```


```python
# Train full LightGBM for Tier 3 fallback
print('Training full LightGBM for Tier 3 fallback...')
X_full = train.drop(columns=[TARGET])
y_full = train[TARGET].values

X_full_fe, full_fp = engineer_features(X_full.copy(), is_train=True,
                                        complaints_df=complaints, history_df=history,
                                        tfidf_features=2000)
full_model = lgb.LGBMClassifier(**LGBM_PARAMS)
full_model.fit(X_full_fe, y_full - 1, callbacks=[lgb.log_evaluation(0)])
print('Done.')
```


```python
# Generate predictions on test set
test_c = test.merge(complaints[['patient_id', 'chief_complaint_raw']],
                    on='patient_id', how='left')
X_test_fe = apply_features(test.copy(), full_fp, complaints_df=complaints, history_df=history)

preds   = np.full(len(test), -1, dtype=int)
sources = ['unresolved'] * len(test)

# Tier 1
for i, row in test_c.iterrows():
    idx = test_c.index.get_loc(i)
    if row['chief_complaint_raw'] in tier1:
        preds[idx]   = tier1[row['chief_complaint_raw']]
        sources[idx] = 'tier1'

# Tier 2
gl_test = test_c[test_c['chief_complaint_raw'].isin(amb_texts)].copy()
for col in GLAUCOMA_FEATURES:
    gl_test[col] = gl_test[col].fillna(gl_meds.get(col, 0))

if len(gl_test) > 0:
    gl_preds_bin = gl_model.predict(gl_test[GLAUCOMA_FEATURES].values)
    gl_preds = np.where(gl_preds_bin == 1, 1, 2)
    for j, i in enumerate(gl_test.index):
        idx = test_c.index.get_loc(i)
        preds[idx]   = gl_preds[j]
        sources[idx] = 'tier2'

# Tier 3
tier3_idx = [i for i, s in enumerate(sources) if s == 'unresolved']
if tier3_idx:
    model_raw = full_model.predict(X_test_fe) + 1
    for idx in tier3_idx:
        preds[idx]   = model_raw[idx]
        sources[idx] = 'tier3'

assert (preds != -1).all(), 'Some rows unresolved!'
assert (preds >= 1).all() and (preds <= 5).all()

print(f'Tier 1 (lookup):   {sources.count("tier1"):5,}  ({100*sources.count("tier1")/len(test):.1f}%)')
print(f'Tier 2 (glaucoma): {sources.count("tier2"):5,}  ({100*sources.count("tier2")/len(test):.1f}%)')
print(f'Tier 3 (model):    {sources.count("tier3"):5,}  ({100*sources.count("tier3")/len(test):.1f}%)')
```

## 9. Submission


```python
submission = pd.DataFrame({
    'patient_id':    test['patient_id'],
    'triage_acuity': preds.astype(int),
})

assert list(submission.columns) == list(sample_sub.columns)
assert len(submission) == len(sample_sub)

submission.to_csv('submission.csv', index=False)

print('submission.csv saved.')
print(f'\nPrediction distribution:')
print(submission['triage_acuity'].value_counts().sort_index().to_string())
submission.head()
```

## 10. Lessons

**The complaint text does most of the work.**
Going from 50 → 150 TF-IDF features was worth +9.26 points of accuracy — more than all other changes combined. When a dataset has free-text that directly describes what you're predicting, that's the primary feature. Treat it that way from the start, not after everything else has failed.

**Error analysis beats hyperparameter tuning.**
We tried tuning at 0.9980 — got 0.9980 back. Then spent 30 minutes on error analysis and found every single mistake had the same root cause. Once you know the root cause, the fix is obvious. Before that, you're just guessing.

**Sometimes the answer is already in the training data.**
For 99.4% of this dataset, the right prediction was just a lookup. A lookup can't be wrong the way a model can. The model is only needed for the 0.6% where the training data gives no definitive answer.

---

*Full source code, experiment log, and Streamlit dashboard: [github.com/Archit-Konde/triagegeist-solution](https://github.com/Archit-Konde/triagegeist-solution)*
