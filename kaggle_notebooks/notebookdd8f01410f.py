# %% [markdown]
# # Triagegeist: Clinical Triage Severity Prediction
# ## Multi-Model Ensemble with Domain-Driven Feature Engineering
#
# **Approach:** LightGBM + XGBoost ensemble (35/65 weighted) with 137 engineered features
# spanning vital sign indices, comorbidity burden scores, chief complaint NLP, and
# contextual aggregations.
#
# **Results:** 5-fold CV Accuracy = 99.4%, QWK = 0.9967

# %% [markdown]
# ## 1. Setup & Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)
plt.style.use('seaborn-v0_8-darkgrid')

# %%
train = pd.read_csv('/kaggle/input/competitions/triagegeist/train.csv')
test = pd.read_csv('/kaggle/input/competitions/triagegeist/test.csv')
cc = pd.read_csv('/kaggle/input/competitions/triagegeist/chief_complaints.csv')
ph = pd.read_csv('/kaggle/input/competitions/triagegeist/patient_history.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Chief Complaints: {cc.shape}, Patient History: {ph.shape}")

# %%
# Merge all data sources
train = train.merge(ph, on='patient_id', how='left') \
             .merge(cc[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
test = test.merge(ph, on='patient_id', how='left') \
            .merge(cc[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')

# Fix duplicate column from merge
if 'chief_complaint_system_x' in train.columns:
    train.rename(columns={'chief_complaint_system_x': 'chief_complaint_system'}, inplace=True)
    test.rename(columns={'chief_complaint_system_x': 'chief_complaint_system'}, inplace=True)
    train.drop(columns=['chief_complaint_system_y'], inplace=True, errors='ignore')
    test.drop(columns=['chief_complaint_system_y'], inplace=True, errors='ignore')

y = train['triage_acuity']
print(f"Merged train: {train.shape}, Merged test: {test.shape}")

# %% [markdown]
# ## 2. Exploratory Data Analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Target distribution
y.value_counts().sort_index().plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.8, edgecolor='black')
axes[0].set_title('ESI Acuity Level Distribution (Training Set)', fontsize=13)
axes[0].set_xlabel('ESI Level')
axes[0].set_ylabel('Count')
for i, v in enumerate(y.value_counts().sort_index()):
    axes[0].text(i, v + 200, f'{v:,}\n({v/len(y):.1%})', ha='center', fontsize=9)

# Vital signs by acuity
vital_means = train.groupby('triage_acuity')[['news2_score', 'gcs_total', 'heart_rate', 'spo2']].mean()
vital_means.plot(kind='bar', ax=axes[1], width=0.8)
axes[1].set_title('Mean Vital Signs by ESI Level', fontsize=13)
axes[1].set_xlabel('ESI Level')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.show()

# %%
# Missing values analysis
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Missing values in training set:")
for col, cnt in missing.items():
    print(f"  {col}: {cnt:,} ({cnt/len(train):.1%})")

# %% [markdown]
# ## 3. Feature Engineering
#
# We construct 137 features organized into five categories:
# 1. **Clinical severity indices** — derived vital sign ratios and threshold flags
# 2. **Comorbidity burden scores** — aggregated disease history groups
# 3. **Chief complaint NLP** — TF-IDF + SVD and keyword detection
# 4. **Age-acuity interactions** — age modulates clinical risk
# 5. **Contextual aggregations** — per-site, per-nurse statistical features

# %%
def engineer_features(df):
    """Domain-driven feature engineering for emergency triage prediction."""
    # --- Clinical Severity Indices ---
    df['pulse_pressure_ratio'] = df['pulse_pressure'] / (df['systolic_bp'] + 1e-5)
    df['map_hr_product'] = df['mean_arterial_pressure'] * df['heart_rate']
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-5)
    df['hr_rr_ratio'] = df['heart_rate'] / (df['respiratory_rate'] + 1e-5)
    df['temp_deviation'] = abs(df['temperature_c'] - 37.0)
    df['spo2_deficit'] = 100 - df['spo2']
    df['gcs_deficit'] = 15 - df['gcs_total']

    # Binary threshold flags (clinically meaningful cutoffs)
    df['tachycardia'] = (df['heart_rate'] > 100).astype(int)
    df['hypotension'] = (df['systolic_bp'] < 90).astype(int)
    df['tachypnea'] = (df['respiratory_rate'] > 20).astype(int)
    df['fever'] = (df['temperature_c'] > 38.0).astype(int)
    df['hypoxia'] = (df['spo2'] < 92).astype(int)
    df['altered_mental'] = (df['gcs_total'] < 15).astype(int)

    # Composite: how many vitals are abnormal
    df['vital_abnormality_count'] = df[['tachycardia', 'hypotension', 'tachypnea',
                                        'fever', 'hypoxia', 'altered_mental']].sum(axis=1)

    # --- Comorbidity Burden Scores ---
    hx_cols = [c for c in df.columns if c.startswith('hx_')]
    df['total_hx'] = df[hx_cols].sum(axis=1)
    df['cardiac_burden'] = df[['hx_hypertension', 'hx_heart_failure',
                                'hx_atrial_fibrillation', 'hx_coronary_artery_disease']].sum(axis=1)
    df['respiratory_burden'] = df[['hx_asthma', 'hx_copd']].sum(axis=1)

    # --- Age Interactions ---
    df['age_news2'] = df['age'] * df['news2_score']
    df['age_gcs'] = df['age'] * df['gcs_deficit']

    # --- Utilization Features ---
    df['visit_admit_ratio'] = df['num_prior_admissions_12m'] / (df['num_prior_ed_visits_12m'] + 1)
    df['n_missing_vitals'] = df[['systolic_bp', 'diastolic_bp',
                                  'respiratory_rate', 'temperature_c']].isnull().sum(axis=1)

    # --- Chief Complaint NLP Keywords ---
    text = df['chief_complaint_raw'].fillna('')
    df['cc_words'] = text.str.split().str.len()
    df['cc_chars'] = text.str.len()
    df['cc_acute'] = text.str.contains('acute|sudden|severe|worst', case=False).astype(int)
    df['cc_chronic'] = text.str.contains('chronic|intermittent|mild', case=False).astype(int)
    df['cc_chest'] = text.str.contains('chest|cardiac|heart', case=False).astype(int)
    df['cc_breath'] = text.str.contains('breath|dyspnea|respiratory', case=False).astype(int)
    df['cc_trauma'] = text.str.contains('fractur|fall|injur|wound|lacerat', case=False).astype(int)
    df['cc_neuro'] = text.str.contains('head|dizz|syncop|seizur|confus', case=False).astype(int)

    return df

train = engineer_features(train)
test = engineer_features(test)
print("Feature engineering complete.")

# %%
# TF-IDF + SVD on chief complaints (capture semantic structure)
print("Fitting TF-IDF + SVD on chief complaints...")
tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words='english')
all_text = pd.concat([train['chief_complaint_raw'].fillna(''),
                      test['chief_complaint_raw'].fillna('')])
tfidf.fit(all_text)

svd = TruncatedSVD(n_components=15, random_state=SEED)
svd.fit(tfidf.transform(all_text))

train_svd = svd.transform(tfidf.transform(train['chief_complaint_raw'].fillna('')))
test_svd = svd.transform(tfidf.transform(test['chief_complaint_raw'].fillna('')))

for i in range(15):
    train[f'cc_svd_{i}'] = train_svd[:, i]
    test[f'cc_svd_{i}'] = test_svd[:, i]

print(f"  SVD explained variance: {svd.explained_variance_ratio_.sum():.1%}")

# %%
# Groupby aggregation features (per-site, per-nurse statistics)
print("Computing groupby aggregation features...")
for cat in ['site_id', 'triage_nurse_id', 'chief_complaint_system', 'mental_status_triage']:
    for num in ['news2_score', 'gcs_total', 'heart_rate', 'spo2']:
        grp = pd.concat([train[[cat, num]], test[[cat, num]]]).groupby(cat)[num]
        mean_map = grp.mean()
        std_map = grp.std()
        train[f'{cat}_{num}_mean'] = train[cat].map(mean_map)
        test[f'{cat}_{num}_mean'] = test[cat].map(mean_map)
        train[f'{cat}_{num}_z'] = (train[num] - train[f'{cat}_{num}_mean']) / (train[cat].map(std_map) + 1e-5)
        test[f'{cat}_{num}_z'] = (test[num] - test[f'{cat}_{num}_mean']) / (test[cat].map(std_map) + 1e-5)

print("Done.")

# %% [markdown]
# ## 4. Prepare Features

# %%
drop_cols = ['patient_id', 'triage_acuity', 'disposition', 'ed_los_hours', 'chief_complaint_raw']
feature_cols = [c for c in train.columns if c not in drop_cols]
cat_features = [c for c in feature_cols if train[c].dtype == 'object']

# Label encode categorical features
for col in cat_features:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X_train = train[feature_cols]
X_test = test[feature_cols]

print(f"Total features: {len(feature_cols)} ({len(feature_cols) - len(cat_features)} numeric, {len(cat_features)} categorical)")

# %% [markdown]
# ## 5. Model Training
#
# We train two GBDT models with 5-fold stratified CV on identical fold splits,
# then combine them with optimized weights.

# %%
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(X_train, y))

# --- LightGBM ---
print("=" * 50)
print("Training LightGBM")
print("=" * 50)

lgb_params = {
    'objective': 'multiclass', 'num_class': 5,
    'metric': 'multi_logloss', 'learning_rate': 0.03,
    'num_leaves': 95, 'min_child_samples': 15,
    'feature_fraction': 0.75, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'reg_alpha': 0.1, 'reg_lambda': 0.5,
    'verbose': -1, 'n_jobs': -1, 'seed': SEED,
}

lgb_oof = np.zeros((len(X_train), 5))
lgb_test_pred = np.zeros((len(X_test), 5))
lgb_model = None

for fold, (train_idx, val_idx) in enumerate(folds):
    dtrain = lgb.Dataset(X_train.iloc[train_idx], label=y.iloc[train_idx] - 1,
                         categorical_feature=cat_features)
    dval = lgb.Dataset(X_train.iloc[val_idx], label=y.iloc[val_idx] - 1,
                       categorical_feature=cat_features)

    model = lgb.train(lgb_params, dtrain, num_boost_round=3000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])

    lgb_oof[val_idx] = model.predict(X_train.iloc[val_idx])
    lgb_test_pred += model.predict(X_test) / N_FOLDS
    lgb_model = model

    acc = accuracy_score(y.iloc[val_idx], np.argmax(lgb_oof[val_idx], axis=1) + 1)
    qwk = cohen_kappa_score(y.iloc[val_idx], np.argmax(lgb_oof[val_idx], axis=1) + 1, weights='quadratic')
    print(f"  Fold {fold+1}: Acc={acc:.4f}, QWK={qwk:.4f}")

lgb_acc = accuracy_score(y, np.argmax(lgb_oof, axis=1) + 1)
lgb_qwk = cohen_kappa_score(y, np.argmax(lgb_oof, axis=1) + 1, weights='quadratic')
print(f"\nLightGBM CV: Acc={lgb_acc:.4f}, QWK={lgb_qwk:.4f}")

# %%
# --- XGBoost ---
print("=" * 50)
print("Training XGBoost")
print("=" * 50)

xgb_params = {
    'objective': 'multi:softprob', 'num_class': 5,
    'eval_metric': 'mlogloss', 'learning_rate': 0.03,
    'max_depth': 7, 'min_child_weight': 5,
    'subsample': 0.8, 'colsample_bytree': 0.75,
    'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'tree_method': 'hist', 'seed': SEED,
}

xgb_oof = np.zeros((len(X_train), 5))
xgb_test_pred = np.zeros((len(X_test), 5))

for fold, (train_idx, val_idx) in enumerate(folds):
    dtrain = xgb.DMatrix(X_train.iloc[train_idx], label=y.iloc[train_idx] - 1)
    dval = xgb.DMatrix(X_train.iloc[val_idx], label=y.iloc[val_idx] - 1)

    model = xgb.train(xgb_params, dtrain, num_boost_round=3000,
                      evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=500)

    xgb_oof[val_idx] = model.predict(dval)
    xgb_test_pred += model.predict(xgb.DMatrix(X_test)) / N_FOLDS

    acc = accuracy_score(y.iloc[val_idx], np.argmax(xgb_oof[val_idx], axis=1) + 1)
    qwk = cohen_kappa_score(y.iloc[val_idx], np.argmax(xgb_oof[val_idx], axis=1) + 1, weights='quadratic')
    print(f"  Fold {fold+1}: Acc={acc:.4f}, QWK={qwk:.4f}")

xgb_acc = accuracy_score(y, np.argmax(xgb_oof, axis=1) + 1)
xgb_qwk = cohen_kappa_score(y, np.argmax(xgb_oof, axis=1) + 1, weights='quadratic')
print(f"\nXGBoost CV: Acc={xgb_acc:.4f}, QWK={xgb_qwk:.4f}")

# %% [markdown]
# ## 6. Ensemble Optimization

# %%
# Find optimal blending weights via grid search on OOF QWK
best_qwk = 0
best_w = 0.5

for w in np.arange(0.2, 0.8, 0.01):
    blend = w * lgb_oof + (1 - w) * xgb_oof
    pred = np.argmax(blend, axis=1) + 1
    qwk = cohen_kappa_score(y, pred, weights='quadratic')
    if qwk > best_qwk:
        best_qwk = qwk
        best_w = w

print(f"Optimal weights: LightGBM={best_w:.2f}, XGBoost={1-best_w:.2f}")

# Final ensemble predictions
oof_blend = best_w * lgb_oof + (1 - best_w) * xgb_oof
test_blend = best_w * lgb_test_pred + (1 - best_w) * xgb_test_pred

oof_pred = np.argmax(oof_blend, axis=1) + 1
test_pred = np.argmax(test_blend, axis=1) + 1

# %%
# Summary table
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
for name, oof in [("LightGBM", lgb_oof), ("XGBoost", xgb_oof), ("Ensemble", oof_blend)]:
    p = np.argmax(oof, axis=1) + 1
    acc = accuracy_score(y, p)
    qwk = cohen_kappa_score(y, p, weights='quadratic')
    f1 = f1_score(y, p, average='weighted')
    print(f"  {name:12s}: Acc={acc:.4f}, QWK={qwk:.4f}, F1={f1:.4f}")

# %% [markdown]
# ## 7. Results Analysis

# %%
# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y, oof_pred)
cm_pct = cm / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=[f'ESI-{i}' for i in range(1, 6)],
            yticklabels=[f'ESI-{i}' for i in range(1, 6)], ax=ax)
ax.set_xlabel('Predicted ESI Level', fontsize=12)
ax.set_ylabel('True ESI Level', fontsize=12)
ens_acc = accuracy_score(y, oof_pred)
ens_qwk = cohen_kappa_score(y, oof_pred, weights='quadratic')
ax.set_title(f'Confusion Matrix (Normalized)\nAccuracy={ens_acc:.1%}, QWK={ens_qwk:.4f}', fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Feature Importance
fig, ax = plt.subplots(figsize=(10, 10))
importance = lgb_model.feature_importance(importance_type='gain')
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=True).tail(25)
ax.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue')
ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
ax.set_title('Top 25 Features — LightGBM', fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Per-class performance
print("Per-Class Classification Report:")
print(classification_report(y, oof_pred,
                            target_names=[f'ESI-{i}' for i in range(1, 6)],
                            digits=4))

# %%
# Per-class bar chart
report = classification_report(y, oof_pred, output_dict=True)
metrics_df = pd.DataFrame({
    'Precision': [report[str(i)]['precision'] for i in range(1, 6)],
    'Recall': [report[str(i)]['recall'] for i in range(1, 6)],
    'F1': [report[str(i)]['f1-score'] for i in range(1, 6)],
}, index=[f'ESI-{i}' for i in range(1, 6)])

fig, ax = plt.subplots(figsize=(8, 5))
metrics_df.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Per-Class Performance Metrics', fontsize=13)
ax.set_ylabel('Score')
ax.set_ylim(0.9, 1.005)
ax.legend(loc='lower right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Generate Submission

# %%
submission = pd.DataFrame({
    'patient_id': test['patient_id'],
    'triage_acuity': test_pred
})
submission.to_csv('submission.csv', index=False)

print(f"Submission saved: {submission.shape}")
print(f"Prediction distribution:")
print(submission['triage_acuity'].value_counts().sort_index())

# %%
# Distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
y.value_counts().sort_index().plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.8, edgecolor='black')
axes[0].set_title('Training Set — ESI Distribution')
axes[0].set_xlabel('ESI Level')
axes[0].set_ylabel('Count')

submission['triage_acuity'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='coral', alpha=0.8, edgecolor='black')
axes[1].set_title('Test Predictions — ESI Distribution')
axes[1].set_xlabel('ESI Level')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

print("\nDone! Submission file ready.")
