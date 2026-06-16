# Triagegeist — Baseline Solution
## Predicting ESI Triage Acuity with LightGBM and Chief Complaint NLP

**Competition:** Triagegeist — AI in Emergency Triage  
**Host:** Laitinen-Fredriksson Foundation  
**Author:** olaflaitinen

---

### Approach Summary

This notebook presents a structured baseline for the Triagegeist competition. The goal is to predict Emergency Severity Index (ESI) triage acuity (1–5) from patient intake data.

**Pipeline:**
1. Join `train.csv`, `patient_history.csv`, and `chief_complaints.csv` on `patient_id`
2. Handle missing values with clinically-informed imputation
3. Encode free-text chief complaints with TF-IDF
4. Train LightGBM with 5-fold stratified cross-validation
5. Evaluate with Quadratic Weighted Kappa (QWK)
6. Generate test predictions and submission file

## 1. Imports and Configuration


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import shap

SEED = 2025
np.random.seed(SEED)

# Paths — update if running locally
DATA_PATH = '/kaggle/input/competitions/triagegeist/'

print('Libraries loaded successfully.')
print(f'LightGBM version: {lgb.__version__}')
```

## 2. Load Data


```python
train      = pd.read_csv(DATA_PATH + 'train.csv')
test       = pd.read_csv(DATA_PATH + 'test.csv')
cc         = pd.read_csv(DATA_PATH + 'chief_complaints.csv')
history    = pd.read_csv(DATA_PATH + 'patient_history.csv')
sample_sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')

print(f'Train:            {train.shape[0]:,} rows x {train.shape[1]} cols')
print(f'Test:             {test.shape[0]:,} rows x {test.shape[1]} cols')
print(f'Chief complaints: {cc.shape[0]:,} rows x {cc.shape[1]} cols')
print(f'Patient history:  {history.shape[0]:,} rows x {history.shape[1]} cols')
train.head(3)
```

## 3. Exploratory Data Analysis


```python
# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

acu_counts = train['triage_acuity'].value_counts().sort_index()
acu_labels = ['ESI-1\n(Immediate)', 'ESI-2\n(Emergent)', 'ESI-3\n(Urgent)',
              'ESI-4\n(Less Urgent)', 'ESI-5\n(Non-Urgent)']
colors = ['#d32f2f','#f57c00','#fbc02d','#388e3c','#1976d2']

axes[0].bar(acu_labels, acu_counts.values, color=colors, edgecolor='white', linewidth=0.8)
axes[0].set_title('Triage Acuity Distribution (Train)', fontsize=13, fontweight='bold')

axes[0].set_ylabel('Patient Count')
for i, v in enumerate(acu_counts.values):
    axes[0].text(i, v + 100, f'{v:,}', ha='center', fontsize=9)

# NEWS2 score by Acuity
train.boxplot(column='news2_score', by='triage_acuity', ax=axes[1],
              boxprops=dict(color='steelblue'),
              medianprops=dict(color='crimson', linewidth=2))
axes[1].set_title('NEWS2 Score by Acuity Level', fontsize=13, fontweight='bold')
axes[1].set_xlabel('ESI Acuity Level')
axes[1].set_ylabel('NEWS2 Score')
plt.suptitle('')
plt.tight_layout()
plt.show()
```


```python
# Missing value analysis
miss = train.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 3))
miss_pct = (miss / len(train) * 100).round(1)
ax.barh(miss_pct.index, miss_pct.values, color='steelblue', edgecolor='white')
ax.set_xlabel('Missing (%)')
ax.set_title('Missing Values in Train Set', fontsize=12, fontweight='bold')
for i, v in enumerate(miss_pct.values):
    ax.text(v + 0.1, i, f'{v}%', va='center', fontsize=9)
plt.tight_layout()
plt.show()

print('\nNote: pain_score = -1 means not recorded (not a true NaN).')
print(f'pain_score -1 count: {(train["pain_score"] == -1).sum():,}')
```


```python
# Vital signs by acuity — heatmap of medians
vital_cols = ['systolic_bp','heart_rate','respiratory_rate','temperature_c','spo2','gcs_total','news2_score']
medians = train.groupby('triage_acuity')[vital_cols].median()

fig, ax = plt.subplots(figsize=(11, 3))
sns.heatmap(medians.T, annot=True, fmt='.1f', cmap='RdYlGn_r',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Median Value'})
ax.set_xlabel('ESI Acuity Level')
ax.set_title('Median Vital Signs by Triage Acuity', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
```

## 4. Feature Engineering


```python
def prepare_features(df, cc_df, hist_df, tfidf=None, fit_tfidf=False):
    df = df.copy()

    # ── Join history and complaints ──────────────────────────────────────────
    df = df.merge(hist_df, on='patient_id', how='left')
    df = df.merge(cc_df[['patient_id','chief_complaint_raw']], on='patient_id', how='left')

    # ── pain_score: clinical missingness signal ──────────────────────────────
    df['pain_not_recorded'] = (df['pain_score'] == -1).astype(int)
    df.loc[df['pain_score'] == -1, 'pain_score'] = np.nan
    df['pain_score'] = df.groupby('age_group')['pain_score'].transform(
        lambda x: x.fillna(x.median()))

    # ── Missingness indicators for BP and RR ─────────────────────────────────
    for col in ['systolic_bp','diastolic_bp','mean_arterial_pressure',
                'pulse_pressure','shock_index','respiratory_rate']:
        df[f'{col}_missing'] = df[col].isnull().astype(int)

    # ── Impute remaining numerics with age_group + shift medians ────────────
    num_cols = ['systolic_bp','diastolic_bp','mean_arterial_pressure',
                'pulse_pressure','shock_index','respiratory_rate','temperature_c']
    for col in num_cols:
        df[col] = df.groupby(['age_group','shift'])[col].transform(
            lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    # ── Derived features ─────────────────────────────────────────────────────
    df['elderly']     = (df['age'] >= 65).astype(int)
    df['pediatric']   = (df['age'] < 16).astype(int)
    df['night_shift'] = (df['shift'] == 'night').astype(int)
    df['weekend']     = df['arrival_day'].isin(['Saturday','Sunday']).astype(int)
    df['high_risk_arrival'] = df['arrival_mode'].isin(['ambulance','helicopter']).astype(int)
    df['altered_ms']  = df['mental_status_triage'].isin(['confused','drowsy','unresponsive','agitated']).astype(int)

    # ── TF-IDF on chief complaint text ───────────────────────────────────────
    df['chief_complaint_raw'] = df['chief_complaint_raw'].fillna('unknown')
    if fit_tfidf:
        tfidf = TfidfVectorizer(max_features=200, ngram_range=(1,2),
                                min_df=5, sublinear_tf=True)
        tfidf_matrix = tfidf.fit_transform(df['chief_complaint_raw'])
    else:
        tfidf_matrix = tfidf.transform(df['chief_complaint_raw'])

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'cc_{c}' for c in tfidf.get_feature_names_out()],
        index=df.index
    )
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    # ── Label encode categoricals ────────────────────────────────────────────
    cat_cols = ['arrival_mode','arrival_day','arrival_season','shift','age_group',
                'sex','language','insurance_type','transport_origin',
                'pain_location','mental_status_triage','chief_complaint_system','site_id']
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # ── Drop non-feature columns ──────────────────────────────────────────────
    drop_cols = ['patient_id','triage_nurse_id','chief_complaint_raw',
                 'disposition','ed_los_hours','triage_acuity']
    feat_cols = [c for c in df.columns if c not in drop_cols]

    return df[feat_cols], tfidf

print('Feature engineering function defined.')
```


```python
# Prepare features
X_train, tfidf_fitted = prepare_features(train, cc, history, fit_tfidf=True)
X_test,  _            = prepare_features(test,  cc, history, tfidf=tfidf_fitted, fit_tfidf=False)
y_train = train['triage_acuity'].values - 1  # 0-indexed for LightGBM

print(f'X_train: {X_train.shape}')
print(f'X_test:  {X_test.shape}')
print(f'y_train classes: {np.unique(y_train)}')
```

## 5. Model Training — LightGBM with 5-Fold CV


```python
lgb_params = {
    'objective':       'multiclass',
    'num_class':       5,
    'metric':          'multi_logloss',
    'n_estimators':    800,
    'learning_rate':   0.05,
    'num_leaves':      63,
    'max_depth':       -1,
    'min_child_samples': 30,
    'subsample':       0.85,
    'colsample_bytree':0.85,
    'reg_alpha':       0.1,
    'reg_lambda':      0.1,
    'class_weight':    'balanced',
    'random_state':    SEED,
    'verbose':         -1,
    'n_jobs':          -1,
}

N_FOLDS   = 5
skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros((len(X_train), 5))
test_preds= np.zeros((len(X_test),  5))
qwk_scores= []
models    = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx],      y_train[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(200)]
    )

    val_prob        = model.predict_proba(X_val)
    val_pred        = np.argmax(val_prob, axis=1)
    oof_preds[val_idx] = val_prob
    test_preds     += model.predict_proba(X_test) / N_FOLDS

    qwk = cohen_kappa_score(y_val, val_pred, weights='quadratic')
    qwk_scores.append(qwk)
    models.append(model)
    print(f'  Fold {fold+1}/{N_FOLDS}  |  QWK: {qwk:.4f}  |  Best iter: {model.best_iteration_}')

oof_labels = np.argmax(oof_preds, axis=1)
oof_qwk    = cohen_kappa_score(y_train, oof_labels, weights='quadratic')
print(f'\nCV QWK: {np.mean(qwk_scores):.4f} +/- {np.std(qwk_scores):.4f}')
print(f'OOF QWK: {oof_qwk:.4f}')
```

## 6. Evaluation


```python
# Classification report
target_names = ['ESI-1','ESI-2','ESI-3','ESI-4','ESI-5']
print('Classification Report (OOF):')
print(classification_report(y_train, oof_labels, target_names=target_names))
```


```python
# Confusion matrix
cm = confusion_matrix(y_train, oof_labels)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Row %'})
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title('Confusion Matrix — OOF Predictions (row-normalised %)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
```

## 7. Feature Importance


```python
# Aggregate feature importance across folds
feat_imp = pd.DataFrame({
    'feature':    X_train.columns,
    'importance': np.mean([m.feature_importances_ for m in models], axis=0)
}).sort_values('importance', ascending=False)

top30 = feat_imp.head(30)

fig, ax = plt.subplots(figsize=(10, 8))
colors_imp = ['#d32f2f' if not c.startswith('cc_') else '#1976d2'
              for c in top30['feature']]
ax.barh(top30['feature'][::-1], top30['importance'][::-1],
        color=colors_imp[::-1], edgecolor='white')
ax.set_xlabel('Mean Feature Importance (gain)')
ax.set_title('Top 30 Features — Red: Structured  |  Blue: NLP (TF-IDF)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
```


```python
# SHAP values on a sample (first fold model)
sample_idx = np.random.choice(len(X_train), size=2000, replace=False)
X_sample   = X_train.iloc[sample_idx]

explainer   = shap.TreeExplainer(models[0])
shap_values = explainer.shap_values(X_sample)

# SHAP for ESI-1 (most critical class)
print('SHAP Summary Plot — ESI-1 (most critical class)')
shap.summary_plot(shap_values[0], X_sample, max_display=15,
                  plot_type='bar', show=True)
```

## 8. Generate Submission


```python
test_labels = np.argmax(test_preds, axis=1) + 1  # Back to 1-indexed ESI

submission = pd.DataFrame({
    'patient_id':   test['patient_id'],
    'triage_acuity': test_labels
})

print('Submission acuity distribution:')
print(submission['triage_acuity'].value_counts().sort_index())

submission.to_csv('submission.csv', index=False)
print('\nsubmission.csv saved.')
submission.head()
```

## 9. Limitations and Next Steps

This baseline demonstrates that structured intake data combined with chief complaint text can predict ESI acuity at a reasonable level (QWK ~0.71). However, several important limitations should be considered:

**1. Class imbalance:** ESI-1 accounts for only ~4% of cases. Despite `class_weight='balanced'`, recall for ESI-1 may remain imperfect. Consider focal loss or oversampling strategies.

**2. Missingness is informative:** Vital sign missingness correlates with lower acuity. The current approach retains missingness indicators but does not model the missingness mechanism explicitly.

**3. NLP is shallow:** TF-IDF captures word frequency but not semantic meaning. Consider replacing with a clinical sentence encoder (e.g. BioClinicalBERT embeddings).

**4. No temporal features:** Crowding effects (shift, hour, day) are included but not modelled interactively. A time-aware model might capture peak-hour undertriage patterns.

**5. Comorbidity interactions:** Currently treated as independent binary flags. A learned comorbidity embedding could better capture clinical risk profiles.

---

### Reproducibility

- Random seed: `2025` used throughout  
- All cells run top-to-bottom without errors  
- Dataset: Triagegeist competition data (Laitinen-Fredriksson Foundation)  
- LightGBM: 4.3.0 | scikit-learn: 1.4.0 | pandas: 2.2.0
