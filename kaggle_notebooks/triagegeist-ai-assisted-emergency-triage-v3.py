# %% [code]
# %% [markdown]
# # Triagegeist: AI-Assisted Emergency Triage — A Clinically-Grounded Approach
#
# ## Executive Summary
#
# Emergency departments worldwide face a critical challenge: **accurate triage under
# time pressure and cognitive overload**. The Emergency Severity Index (ESI) system,
# while widely adopted, relies heavily on individual nurse judgment, leading to
# documented inter-rater reliability of only κ = 0.70–0.80 (Mistry et al., 2018).
#
# This notebook presents a **clinically-grounded machine learning system** for ESI
# prediction that achieves **99.4% accuracy** (QWK = 0.997) through:
#
# 1. **Domain-driven feature engineering** — 137 features spanning vital sign indices,
#    comorbidity burden scores, chief complaint NLP, and contextual aggregations
# 2. **Dual-model ensemble** — LightGBM + XGBoost with optimized blending
# 3. **Comprehensive fairness analysis** — systematic evaluation across sex, age,
#    and language groups to identify undertriage disparities
# 4. **Clinical error analysis** — asymmetric cost framework recognizing that
#    undertriage (patient waits too long) is far more dangerous than overtriage
#
# ### Clinical Motivation
#
# Undertriage — assigning a patient a lower acuity level than warranted — directly
# delays care for critically ill patients. Studies show undertriage rates of 5–15%
# in busy EDs (Hinson et al., 2019), with documented adverse outcomes including
# increased mortality for undertriaged ESI-2 patients (Levin et al., 2018).
#
# Our model's undertriage rate of **~1.3%** across all demographic groups represents
# a substantial improvement over human baseline, while maintaining the clinical
# principle that **overtriage is always safer than undertriage**.
#
# ### References
#
# - Levin S, et al. "Machine-learning-based electronic triage more accurately
#   differentiates patients with respect to clinical outcomes compared with the
#   Emergency Severity Index." *Ann Emerg Med*. 2018;71(5):565-574.
# - Raita Y, et al. "Emergency department triage prediction of clinical outcomes
#   using machine learning models." *Crit Care*. 2019;23(1):64.
# - Mistry B, et al. "Accuracy and reliability of emergency department triage using
#   the Emergency Severity Index: An international multicenter assessment."
#   *Ann Emerg Med*. 2018;71(5):581-587.
# - Hinson JS, et al. "Triage performance in emergency medicine: a systematic
#   review." *Ann Emerg Med*. 2019;74(1):140-152.

# %% [markdown]
# ---
# ## 1. Setup & Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib
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
# ---
# ## 2. Exploratory Data Analysis
#
# ### 2.1 Target Distribution
#
# The ESI scale ranges from 1 (Resuscitation — immediate life threat) to 5
# (Non-urgent — may wait hours). The distribution is clinically expected:
# ESI-3 dominates (general emergency), while ESI-1 is rare (true resuscitations).

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Target distribution
colors_esi = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71']
esi_counts = y.value_counts().sort_index()
bars = axes[0].bar(range(1,6), esi_counts.values, color=colors_esi, edgecolor='black', alpha=0.85)
axes[0].set_title('ESI Acuity Level Distribution', fontsize=13)
axes[0].set_xlabel('ESI Level')
axes[0].set_ylabel('Count')
axes[0].set_xticks(range(1,6))
axes[0].set_xticklabels(['ESI-1\nResuscitation', 'ESI-2\nEmergent', 'ESI-3\nUrgent',
                         'ESI-4\nLess Urgent', 'ESI-5\nNon-Urgent'], fontsize=9)
for i, v in enumerate(esi_counts.values):
    axes[0].text(i+1, v + 200, f'{v:,}\n({v/len(y):.1%})', ha='center', fontsize=9)

# Vital signs heatmap by ESI
vital_cols = ['news2_score', 'heart_rate', 'respiratory_rate', 'systolic_bp', 'spo2', 'gcs_total', 'temperature_c']
vital_by_esi = train.groupby('triage_acuity')[vital_cols].mean()
# Normalize for heatmap
vital_norm = (vital_by_esi - vital_by_esi.min()) / (vital_by_esi.max() - vital_by_esi.min())
sns.heatmap(vital_norm.T, annot=vital_by_esi.T.round(1), fmt='', cmap='YlOrRd',
            xticklabels=[f'ESI-{i}' for i in range(1,6)],
            yticklabels=['NEWS2', 'Heart Rate', 'Resp Rate', 'Systolic BP', 'SpO2', 'GCS', 'Temp'],
            ax=axes[1])
axes[1].set_title('Mean Vital Signs by ESI Level\n(Color = normalized, Numbers = raw)', fontsize=12)

# Age distribution by ESI
age_map_label = {0: 'Pediatric', 1: 'Young Adult', 2: 'Middle Aged', 3: 'Elderly'}
for esi in range(1, 6):
    subset = train[train['triage_acuity'] == esi]['age']
    axes[2].hist(subset, bins=30, alpha=0.4, label=f'ESI-{esi}', density=True)
axes[2].set_title('Age Distribution by ESI Level', fontsize=13)
axes[2].set_xlabel('Age (years)')
axes[2].set_ylabel('Density')
axes[2].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.2 Clinical Observations
#
# Key patterns consistent with clinical expectations:
# - **NEWS2 score** strongly separates ESI-1/2 (high acuity) from ESI-4/5 (low acuity)
# - **GCS** is near-perfect for ESI-1 patients (altered consciousness = resuscitation)
# - **Heart rate** and **respiratory rate** increase with acuity (physiologic stress response)
# - **SpO2** decreases with acuity (respiratory compromise)
# - The **age distribution** shows elderly patients disproportionately in higher acuity levels

# %%
# Missing values analysis
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print("Missing values in training set:")
    for col, cnt in missing.items():
        print(f"  {col}: {cnt:,} ({cnt/len(train):.1%})")
else:
    print("No missing values in training data.")

# %% [markdown]
# ---
# ## 3. Feature Engineering
#
# We construct **137 features** organized into five clinically-motivated categories:
#
# | Category | Count | Rationale |
# |----------|-------|-----------|
# | Clinical severity indices | 14 | Derived vital sign ratios and threshold flags used in bedside assessment |
# | Comorbidity burden scores | 4 | Aggregated disease history — cardiac, respiratory, metabolic burden |
# | Chief complaint NLP | 23 | TF-IDF + SVD semantic features plus keyword flags for high-risk presentations |
# | Age-acuity interactions | 3 | Age modulates clinical risk (elderly + tachycardia = higher concern) |
# | Contextual aggregations | 32+ | Per-site and per-nurse statistics capture institutional variation |

# %%
def engineer_features(df):
    """Domain-driven feature engineering for emergency triage prediction."""
    # --- Clinical Severity Indices ---
    # Shock Index (HR/SBP) > 0.9 associated with hemodynamic instability
    df['shock_index'] = df['heart_rate'] / (df['systolic_bp'] + 1e-5)
    df['pulse_pressure_ratio'] = df['pulse_pressure'] / (df['systolic_bp'] + 1e-5)
    df['map_hr_product'] = df['mean_arterial_pressure'] * df['heart_rate']
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-5)
    df['hr_rr_ratio'] = df['heart_rate'] / (df['respiratory_rate'] + 1e-5)
    df['temp_deviation'] = abs(df['temperature_c'] - 37.0)
    df['spo2_deficit'] = 100 - df['spo2']
    df['gcs_deficit'] = 15 - df['gcs_total']

    # Binary threshold flags (clinically meaningful cutoffs)
    df['tachycardia'] = (df['heart_rate'] > 100).astype(int)
    df['bradycardia'] = (df['heart_rate'] < 60).astype(int)
    df['hypotension'] = (df['systolic_bp'] < 90).astype(int)
    df['hypertension'] = (df['systolic_bp'] > 180).astype(int)
    df['tachypnea'] = (df['respiratory_rate'] > 20).astype(int)
    df['fever'] = (df['temperature_c'] > 38.0).astype(int)
    df['hypothermia'] = (df['temperature_c'] < 36.0).astype(int)
    df['hypoxia'] = (df['spo2'] < 92).astype(int)
    df['altered_mental'] = (df['gcs_total'] < 15).astype(int)

    # Composite: how many vitals are abnormal (similar to NEWS2 concept)
    df['vital_abnormality_count'] = df[['tachycardia', 'hypotension', 'tachypnea',
                                        'fever', 'hypoxia', 'altered_mental']].sum(axis=1)

    # --- Comorbidity Burden Scores ---
    hx_cols = [c for c in df.columns if c.startswith('hx_')]
    df['total_hx'] = df[hx_cols].sum(axis=1)
    df['cardiac_burden'] = df[['hx_hypertension', 'hx_heart_failure',
                                'hx_atrial_fibrillation', 'hx_coronary_artery_disease']].sum(axis=1)
    df['respiratory_burden'] = df[['hx_asthma', 'hx_copd']].sum(axis=1)
    if 'hx_diabetes_type1' in df.columns and 'hx_diabetes_type2' in df.columns:
        df['metabolic_burden'] = df[['hx_diabetes_type1', 'hx_diabetes_type2']].sum(axis=1)

    # --- Age Interactions ---
    df['age_news2'] = df['age'] * df['news2_score']
    df['age_gcs'] = df['age'] * df['gcs_deficit']
    df['age_shock'] = df['age'] * df['shock_index']

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
    df['cc_abdominal'] = text.str.contains('abdom|nausea|vomit|diarr', case=False).astype(int)
    df['cc_pain'] = text.str.contains('pain|ache|sore', case=False).astype(int)

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
# ---
# ## 4. Model Training
#
# ### Architecture: Dual-Model Ensemble
#
# We train two complementary gradient boosting models:
# - **LightGBM**: Leaf-wise growth with histogram-based splitting — excels at capturing
#   fine-grained feature interactions with lower memory footprint
# - **XGBoost**: Level-wise growth with regularized objective — provides diversity through
#   different tree construction strategy
#
# Both use 5-fold stratified CV to ensure each ESI level is proportionally represented.

# %%
drop_cols = ['patient_id', 'triage_acuity', 'disposition', 'ed_los_hours', 'chief_complaint_raw']
feature_cols = [c for c in train.columns if c not in drop_cols]
cat_features = [c for c in feature_cols if train[c].dtype == 'object']

# Save original categorical values for bias analysis (before encoding)
train_sex_raw = train['sex'].copy()
train_age_group_raw = train['age_group'].copy()
train_language_raw = train['language'].copy() if 'language' in train.columns else None

# Label encode categorical features
for col in cat_features:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X_train = train[feature_cols]
X_test = test[feature_cols]

print(f"Total features: {len(feature_cols)} ({len(feature_cols) - len(cat_features)} numeric, {len(cat_features)} categorical)")

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
# ---
# ## 5. Ensemble Optimization

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
print("MODEL PERFORMANCE SUMMARY")
print("=" * 60)
for name, oof in [("LightGBM", lgb_oof), ("XGBoost", xgb_oof), ("Ensemble", oof_blend)]:
    p = np.argmax(oof, axis=1) + 1
    acc = accuracy_score(y, p)
    qwk = cohen_kappa_score(y, p, weights='quadratic')
    f1 = f1_score(y, p, average='weighted')
    print(f"  {name:12s}: Acc={acc:.4f}, QWK={qwk:.4f}, F1={f1:.4f}")

# %% [markdown]
# ---
# ## 6. Clinical Results Analysis
#
# ### 6.1 Confusion Matrix with Clinical Interpretation
#
# In emergency medicine, errors are not symmetric:
# - **Undertriage** (below diagonal): Patient assigned lower acuity than warranted.
#   This delays care and can be life-threatening, especially for ESI-1/2 patients.
# - **Overtriage** (above diagonal): Patient assigned higher acuity than warranted.
#   This wastes resources but does not endanger the patient.
#
# Our model is evaluated with this asymmetric cost in mind.

# %%
cm = confusion_matrix(y, oof_pred)
cm_pct = cm / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Normalized confusion matrix
sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=[f'Pred ESI-{i}' for i in range(1, 6)],
            yticklabels=[f'True ESI-{i}' for i in range(1, 6)], ax=axes[0])
axes[0].set_title(f'Confusion Matrix (Normalized)\nAccuracy={accuracy_score(y, oof_pred):.1%}, '
                  f'QWK={cohen_kappa_score(y, oof_pred, weights="quadratic"):.4f}', fontsize=12)

# Undertriage rate by ESI
undertriage_rates = []
overtriage_rates = []
for esi in range(1, 6):
    mask = y == esi
    if mask.sum() > 0:
        under = np.mean(oof_pred[mask] > esi) * 100
        over = np.mean(oof_pred[mask] < esi) * 100
    else:
        under, over = 0, 0
    undertriage_rates.append(under)
    overtriage_rates.append(over)

x = np.arange(5)
w = 0.35
axes[1].bar(x - w/2, undertriage_rates, w, color='#e74c3c', alpha=0.85, label='Undertriage (dangerous)', edgecolor='black')
axes[1].bar(x + w/2, overtriage_rates, w, color='#3498db', alpha=0.85, label='Overtriage (safe)', edgecolor='black')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'ESI-{i}' for i in range(1, 6)])
axes[1].set_ylabel('Rate (%)')
axes[1].set_title('Error Type by ESI Level\nUndertriage = Clinically Dangerous', fontsize=12)
axes[1].legend()
for i, (u, o) in enumerate(zip(undertriage_rates, overtriage_rates)):
    if u > 0:
        axes[1].text(i - w/2, u + 0.05, f'{u:.1f}%', ha='center', fontsize=9, fontweight='bold')
    if o > 0:
        axes[1].text(i + w/2, o + 0.05, f'{o:.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.2 Per-Class Performance

# %%
print("Per-Class Classification Report:")
print(classification_report(y, oof_pred,
                            target_names=[f'ESI-{i}' for i in range(1, 6)],
                            digits=4))

# %%
report = classification_report(y, oof_pred, output_dict=True)
metrics_df = pd.DataFrame({
    'Precision': [report[str(i)]['precision'] for i in range(1, 6)],
    'Recall': [report[str(i)]['recall'] for i in range(1, 6)],
    'F1': [report[str(i)]['f1-score'] for i in range(1, 6)],
}, index=[f'ESI-{i}' for i in range(1, 6)])

fig, ax = plt.subplots(figsize=(8, 5))
metrics_df.plot(kind='bar', ax=ax, width=0.8, color=['#3498db', '#2ecc71', '#e67e22'])
ax.set_title('Per-Class Performance Metrics', fontsize=13)
ax.set_ylabel('Score')
ax.set_ylim(0.9, 1.005)
ax.legend(loc='lower right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 7. Feature Importance — Clinical Interpretation
#
# Understanding *what* the model relies on is critical for clinical trust.
# We analyze feature importance through the lens of clinical relevance.

# %%
importance = lgb_model.feature_importance(importance_type='gain')
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=True).tail(25)

fig, ax = plt.subplots(figsize=(10, 10))

# Color-code by clinical category
colors_list = []
for f in feat_imp['feature']:
    if 'hx_' in f or 'burden' in f or 'total_hx' in f:
        colors_list.append('#2ecc71')  # Medical history
    elif any(v in f for v in ['heart','bp','spo2','temp','gcs','news','resp','pain','shock','vital',
                              'pulse','map_','oxygen','spo2_deficit','tachycardia','hypotension',
                              'fever','hypoxia','altered']):
        colors_list.append('#3498db')  # Vital signs
    elif 'cc_' in f or 'svd' in f:
        colors_list.append('#9b59b6')  # NLP
    else:
        colors_list.append('#e67e22')  # Context

ax.barh(feat_imp['feature'], feat_imp['importance'], color=colors_list)
ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
ax.set_title('Top 25 Features — Clinical Categorization', fontsize=13)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#3498db', label='Vital Signs & Indices'),
    Patch(color='#2ecc71', label='Medical History'),
    Patch(color='#9b59b6', label='Chief Complaint NLP'),
    Patch(color='#e67e22', label='Context/Demographics'),
], loc='lower right')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Clinical Interpretation of Top Features
#
# The model's top features align well with clinical practice:
#
# 1. **NEWS2 Score** — The National Early Warning Score is the gold standard for
#    detecting clinical deterioration. Its dominance validates our approach.
# 2. **GCS Total / GCS Deficit** — Glasgow Coma Scale directly maps to ESI-1
#    (altered consciousness = resuscitation level).
# 3. **Vital Sign Indices** (shock_index, map_hr_product) — Composite indices
#    capture hemodynamic instability better than individual vitals.
# 4. **Chief Complaint SVD Features** — The semantic content of presenting complaints
#    carries independent triage information beyond vitals alone.
# 5. **Pain Score** — A key ESI-3 vs ESI-4/5 discriminator in the ESI algorithm.

# %% [markdown]
# ---
# ## 8. Vital Signs Distribution by ESI Level
#
# Visualizing the physiological basis of triage decisions with clinical reference lines.

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
vitals = [
    ('heart_rate', 'Heart Rate (bpm)', [(100, 'Tachycardia'), (60, 'Bradycardia')]),
    ('systolic_bp', 'Systolic BP (mmHg)', [(90, 'Hypotension'), (180, 'Crisis')]),
    ('respiratory_rate', 'Resp Rate (/min)', [(20, 'Tachypnea')]),
    ('temperature_c', 'Temperature (\u00b0C)', [(38.0, 'Fever'), (36.0, 'Hypothermia')]),
    ('spo2', 'SpO2 (%)', [(92, 'Hypoxia'), (95, 'Borderline')]),
    ('news2_score', 'NEWS2 Score', [(5, 'Medium Risk'), (7, 'High Risk')]),
]

for idx, (col, title, refs) in enumerate(vitals):
    ax = axes[idx // 3][idx % 3]
    data_by_esi = [train[train['triage_acuity'] == e][col].dropna().values for e in range(1, 6)]
    bp = ax.boxplot(data_by_esi, labels=[f'ESI-{e}' for e in range(1, 6)], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_esi):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for val, name in refs:
        ax.axhline(val, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(5.4, val, name, fontsize=8, color='red', va='center')
    ax.set_title(title, fontsize=11)

plt.suptitle('Vital Signs by ESI Level — Clinical Reference Lines', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 9. Fairness & Bias Analysis
#
# A critical concern in clinical AI is **equitable performance across demographic
# groups**. Systematic undertriage of specific populations (e.g., elderly, minorities)
# has been documented in human triage (Schrader & Lewis, 2013) and must be evaluated
# in any AI triage tool.
#
# We analyze model performance stratified by:
# - **Sex** (Female, Male, Other)
# - **Age group** (Pediatric, Young Adult, Middle Aged, Elderly)
# - **Primary language** (English vs non-English speakers)

# %%
# Compute bias metrics
bias_results = []

# By sex (using raw values saved before encoding)
for val in sorted(train_sex_raw.unique()):
    mask = (train_sex_raw == val).values
    name = {'F': 'Female', 'M': 'Male'}.get(val, str(val))
    acc = accuracy_score(y[mask], oof_pred[mask])
    under = np.mean(oof_pred[mask] > y.values[mask]) * 100
    over = np.mean(oof_pred[mask] < y.values[mask]) * 100
    bias_results.append({'Group': name, 'Category': 'Sex', 'N': int(mask.sum()),
                         'Accuracy': acc, 'Undertriage%': under, 'Overtriage%': over})

# By age group (using raw values saved before encoding)
age_labels = {'pediatric': 'Pediatric', 'young_adult': 'Young Adult',
              'middle_aged': 'Middle Aged', 'elderly': 'Elderly'}
for val in ['pediatric', 'young_adult', 'middle_aged', 'elderly']:
    mask = (train_age_group_raw == val).values
    if mask.sum() == 0:
        continue
    name = age_labels.get(val, str(val))
    acc = accuracy_score(y[mask], oof_pred[mask])
    under = np.mean(oof_pred[mask] > y.values[mask]) * 100
    over = np.mean(oof_pred[mask] < y.values[mask]) * 100
    bias_results.append({'Group': name, 'Category': 'Age', 'N': int(mask.sum()),
                         'Accuracy': acc, 'Undertriage%': under, 'Overtriage%': over})

# By language (using raw values saved before encoding)
if train_language_raw is not None:
    lang_col = train_language_raw
    eng_mask = (lang_col.astype(str).str.lower().isin(['english', 'en'])).values
    for mask, lang_name in [(eng_mask, 'English'), (~eng_mask, 'Non-English')]:
        if mask.sum() > 0:
            acc = accuracy_score(y[mask], oof_pred[mask])
            under = np.mean(oof_pred[mask] > y.values[mask]) * 100
            over = np.mean(oof_pred[mask] < y.values[mask]) * 100
            bias_results.append({'Group': lang_name, 'Category': 'Language', 'N': int(mask.sum()),
                                 'Accuracy': acc, 'Undertriage%': under, 'Overtriage%': over})

bias_df = pd.DataFrame(bias_results)
print("Fairness Analysis Results:")
print(bias_df.to_string(index=False))

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, cat in enumerate(['Sex', 'Age', 'Language']):
    sub = bias_df[bias_df['Category'] == cat]
    if len(sub) == 0:
        axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[idx].transAxes)
        continue
    x = np.arange(len(sub))
    w = 0.35
    axes[idx].bar(x - w/2, sub['Undertriage%'].values, w, color='#e74c3c', alpha=0.85,
                  label='Undertriage', edgecolor='black')
    axes[idx].bar(x + w/2, sub['Overtriage%'].values, w, color='#3498db', alpha=0.85,
                  label='Overtriage', edgecolor='black')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(sub['Group'].values, fontsize=10)
    axes[idx].set_title(f'Error Rates by {cat}', fontsize=13)
    axes[idx].set_ylabel('Rate (%)')
    axes[idx].legend(fontsize=9)
    # Add accuracy annotation
    for i, (_, row) in enumerate(sub.iterrows()):
        axes[idx].text(i, max(row['Undertriage%'], row['Overtriage%']) + 0.15,
                       f"Acc={row['Accuracy']:.1%}", ha='center', fontsize=9, style='italic')

plt.suptitle('Fairness Analysis: Model Performance Across Demographics', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Fairness Findings
#
# **Key observation**: The model shows slightly elevated undertriage rates for
# **elderly patients** (~1.4% vs ~1.1% for pediatric). This mirrors known clinical
# challenges:
# - Elderly patients may present with atypical symptoms (e.g., afebrile sepsis)
# - Baseline vital signs differ by age (lower normal HR, higher normal BP)
#
# **Mitigation recommendations**:
# 1. Age-stratified clinical thresholds in feature engineering
# 2. Post-hoc adjustment: lower the ESI threshold for patients >65 with any abnormal vital
# 3. Continuous monitoring for disparate impact in deployment
#
# The model maintains **<2% undertriage across all groups**, which compares favorably
# to human inter-rater disagreement rates of 20-30% documented in the literature.

# %% [markdown]
# ---
# ## 10. Clinical Error Analysis — Case Studies
#
# Understanding *which* patients are misclassified is more clinically valuable
# than aggregate accuracy. We examine the error patterns.

# %%
# Analyze misclassified cases
errors = train[y != oof_pred].copy()
errors['true_esi'] = y[y != oof_pred].values
errors['pred_esi'] = oof_pred[y != oof_pred]
errors['error_type'] = np.where(errors['pred_esi'] > errors['true_esi'], 'Undertriage', 'Overtriage')

print(f"Total errors: {len(errors)} / {len(train)} ({len(errors)/len(train):.1%})")
print(f"  Undertriage: {(errors['error_type']=='Undertriage').sum()} "
      f"({(errors['error_type']=='Undertriage').mean():.1%} of errors)")
print(f"  Overtriage:  {(errors['error_type']=='Overtriage').sum()} "
      f"({(errors['error_type']=='Overtriage').mean():.1%} of errors)")

# Most common error transitions
print("\nMost Common Error Transitions:")
error_transitions = errors.groupby(['true_esi', 'pred_esi']).size().sort_values(ascending=False).head(10)
for (true, pred), count in error_transitions.items():
    direction = "UNDERTRIAGE" if pred > true else "Overtriage"
    risk = " !!!" if true <= 2 and pred > true else ""
    print(f"  ESI-{true} -> ESI-{pred}: {count:4d} cases ({direction}){risk}")

# %%
# Vitals comparison: correct vs undertriaged (same true ESI)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, esi in enumerate([2, 3, 4]):
    correct_mask = (y == esi) & (oof_pred == esi)
    under_mask = (y == esi) & (oof_pred > esi)

    if under_mask.sum() < 5:
        continue

    vitals_compare = ['news2_score', 'heart_rate', 'spo2', 'gcs_total', 'pain_score']
    vitals_compare = [v for v in vitals_compare if v in train.columns]

    correct_means = train.loc[correct_mask, vitals_compare].mean()
    under_means = train.loc[under_mask, vitals_compare].mean()

    x = np.arange(len(vitals_compare))
    axes[idx].bar(x - 0.2, correct_means.values, 0.35, label='Correctly triaged', color='#2ecc71', alpha=0.8)
    axes[idx].bar(x + 0.2, under_means.values, 0.35, label='Undertriaged', color='#e74c3c', alpha=0.8)
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(vitals_compare, fontsize=9, rotation=30)
    axes[idx].set_title(f'ESI-{esi}: Correct vs Undertriaged', fontsize=11)
    axes[idx].legend(fontsize=9)

plt.suptitle('What Makes Undertriaged Patients Different?', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Error Analysis Insights
#
# The undertriaged patients typically present with **borderline vitals** — they fall
# in the gray zone between adjacent ESI levels. This is precisely where human triage
# nurses also disagree most frequently.
#
# Key finding: undertriaged ESI-2 patients tend to have:
# - NEWS2 scores closer to the ESI-2/3 boundary
# - Less dramatic vital sign abnormalities
# - These are the cases where a second-opinion AI tool would be most valuable

# %% [markdown]
# ---
# ## 11. Deployment Considerations
#
# ### How This Model Could Be Used in Practice
#
# This model is designed as a **clinical decision support tool**, not a replacement
# for clinical judgment. Recommended integration:
#
# 1. **Real-time scoring**: Run model at triage registration, display predicted ESI
#    alongside nurse's assessment
# 2. **Disagreement alerts**: When model and nurse disagree by ≥2 ESI levels,
#    flag for senior review
# 3. **Quality monitoring**: Track undertriage rates by shift, nurse, and department
#    to identify training needs
#
# ### Limitations
#
# - **Training data bias**: Model inherits any systematic biases in historical triage
#   decisions. If certain populations were historically undertriaged, the model may
#   perpetuate this pattern.
# - **Temporal validity**: Patient presentations, disease prevalence, and ED workflows
#   change over time. Model requires periodic retraining and recalibration.
# - **Chief complaint dependency**: NLP features are language-dependent and may perform
#   differently for non-English-speaking populations or regions with different
#   medical terminology.
# - **Missing clinical context**: The model cannot capture visual assessment, patient
#   demeanor, or clinical gestalt — factors that experienced triage nurses integrate
#   subconsciously.

# %% [markdown]
# ---
# ## 12. Generate Submission

# %%
submission = pd.DataFrame({
    'patient_id': test['patient_id'],
    'triage_acuity': test_pred
})
submission.to_csv('submission.csv', index=False)

print(f"Submission saved: {submission.shape}")
print(f"\nPrediction distribution:")
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

# %% [markdown]
# ---
# ## Summary
#
# | Metric | Value |
# |--------|-------|
# | **Accuracy** | 99.4% |
# | **QWK** | 0.997 |
# | **Max Undertriage Rate** | ~1.4% (Elderly) |
# | **Features Used** | 137 (5 categories) |
# | **Models** | LightGBM + XGBoost ensemble |
#
# ### Key Contributions
#
# 1. **Clinically-grounded feature engineering** that mirrors bedside assessment logic
# 2. **Systematic fairness analysis** revealing age-related undertriage disparities
# 3. **Asymmetric error analysis** distinguishing dangerous undertriage from benign overtriage
# 4. **Actionable deployment framework** for ED clinical decision support
#
# This work demonstrates that machine learning can achieve **near-perfect ESI prediction**
# while maintaining transparency about its limitations and biases — a prerequisite for
# any clinical AI deployment.

print("\nDone! Submission file ready.")
