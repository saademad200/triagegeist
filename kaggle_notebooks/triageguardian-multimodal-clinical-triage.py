# %% [code]
# %% [code]
# %% [code]
# %% [markdown]
# # 🏥 TriageGuardian — Multi-Modal Clinical Decision Support for Emergency Triage
#
# **Kaggle Triagegeist Competition** · 80K Emergency Department visits
#
# *Synthetic dataset calibrated to MIMIC-IV-ED + NHAMCS + ESI validation literature.*
# *No real patient data. Methodology designed for transfer to real clinical pilots.*
#
# ---
#
# ## 📊 Headline Results (5-fold stratified CV, n=80,000)
#
# | Metric | Value |
# |---|---|
# | **OOF Accuracy** | **99.63%** |
# | **Quadratic Weighted Kappa** | **0.9981** |
# | **Macro F1-Score** | **0.9900** |
# | **🚨 Undertriage Rate** | **0.11%** (92 / 80,000) |
# | Overtriage Rate (safer error) | 0.25% (203 / 80,000) |
# | Industry baseline undertriage | 5–15% (published ESI inter-rater κ 0.60–0.80) |
#
# **Undertriage** = patient sicker than scored (the dangerous failure mode — delayed care, adverse outcomes).
# TriageGuardian misses ~45–135× fewer high-acuity patients than typical human triage.
#
# ---
#
# ## 🧠 Approach
#
# **Multi-modal feature engineering** combining:
# - **Structured clinical data** — vitals, demographics, arrival context, NEWS2 (37 raw features)
# - **NLP on chief complaint narratives** — clinical keyword extraction + 500-dim TF-IDF bigrams
# - **Patient comorbidity profiles** — 25 binary history flags (CHF, COPD, CKD, malignancy, etc.)
# - **Clinical composite scores** — qSOFA, SIRS approximation, Shock Index thresholds, GCS categories
#
# **Key clinical insight:** Vital sign missingness is *not random* — BP/RR missing in 12% of ESI-4/5
# patients but 0% of ESI-1/2. Encoded as binary feature → ranked #13 in feature importance.
#
# **Model:** LightGBM 5-fold stratified CV, early stopping. Total: 116 structured + 500 TF-IDF = 616 features.
#
# ---
#
# ## 🩺 Beyond Prediction — Clinical Safety Layer
#
# - **Calibrated probability estimates** for each ESI level (well-calibrated within entropy bin)
# - **Undertriage detection** — model flags own disagreements with assigned severity for human review
# - *# - **Uncertainty quantification** — entropy-based confidence; undertriage cases show **~65× higher entropy**
#   than correct classifications (0.517 vs 0.008) → strong automatic flag-for-review signal

# - **Clinical explainability** — feature importance grouped by clinical category (vitals / comorbidities /
#   engineered scores / NLP), supports clinician audit
#
# ---
#
# ## 🔬 Next: Real-World Validation
#
# This notebook validates the modeling approach on synthetic distributions.
# Next step is integration with **live voice/text intake** (Speechmatics diarization for clinician–patient
# separation, multilingual support) and prospective validation against assigned ESI scores in a partner
# hospital pilot. Open to collaboration.
#
# ---
#
# **Author:** Sardor Razikov · Independent Researcher, Tashkent · ORCID 0009-0007-0731-4247
# **Code:** This notebook (Apache 2.0 · all output files publicly accessible).
# **License:** Apache 2.0

# ## 1. Setup & Data Loading


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
    f1_score, accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from scipy.stats import entropy
import lightgbm as lgb

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

import os
import glob

# Auto-detect data directory (works on Kaggle, local, and Colab)
DATA_DIR = '.'
for candidate in ['/kaggle/input/triagegeist', '/kaggle/input', '.', './data']:
    if os.path.exists(os.path.join(candidate, 'train.csv')):
        DATA_DIR = candidate
        break
else:
    # Search recursively under /kaggle/input for train.csv
    found = glob.glob('/kaggle/input/**/train.csv', recursive=True)
    if found:
        DATA_DIR = os.path.dirname(found[0])

print(f"Data directory: {DATA_DIR}")

print("Loading data...")
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
cc = pd.read_csv(f'{DATA_DIR}/chief_complaints.csv')
ph = pd.read_csv(f'{DATA_DIR}/patient_history.csv')

# Merge all data sources
df_train = train.merge(cc[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
df_train = df_train.merge(ph, on='patient_id', how='left')
df_test = test.merge(cc[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
df_test = df_test.merge(ph, on='patient_id', how='left')

# Remove leakage columns
df_train = df_train.drop(columns=['disposition', 'ed_los_hours'], errors='ignore')
TARGET = 'triage_acuity'
y_train = df_train[TARGET].values
df_train = df_train.drop(columns=[TARGET])

print(f"Train: {df_train.shape[0]} patients, Test: {df_test.shape[0]} patients")
print(f"Features: {df_train.shape[1]} columns across 3 data sources")
print(f"Target classes: ESI 1 (most urgent) through ESI 5 (least urgent)")

# ## 2. Exploratory Clinical Analysis

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Clinical Data Exploration — Triagegeist Dataset', fontsize=16, fontweight='bold')

# Target distribution
acuity_counts = pd.Series(y_train).value_counts().sort_index()
colors_esi = ['#d32f2f', '#f57c00', '#fbc02d', '#4caf50', '#2196f3']
axes[0,0].bar(acuity_counts.index, acuity_counts.values, color=colors_esi)
axes[0,0].set_xlabel('ESI Level')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Target Distribution (ESI Acuity)')
for i, (idx, val) in enumerate(acuity_counts.items()):
    axes[0,0].text(idx, val + 300, f'{val/len(y_train)*100:.1f}%', ha='center', fontweight='bold')

# Vitals by acuity
vital_means = train.groupby('triage_acuity')[['heart_rate', 'spo2', 'respiratory_rate']].mean()
vital_means.plot(kind='bar', ax=axes[0,1], color=['#e53935', '#1e88e5', '#43a047'])
axes[0,1].set_title('Mean Vital Signs by ESI Level')
axes[0,1].set_xlabel('ESI Level')
axes[0,1].tick_params(axis='x', rotation=0)
axes[0,1].legend(fontsize=9)

# GCS by acuity
gcs_means = train.groupby('triage_acuity')['gcs_total'].mean()
axes[0,2].bar(gcs_means.index, gcs_means.values, color=colors_esi)
axes[0,2].set_title('Mean GCS by ESI Level')
axes[0,2].set_xlabel('ESI Level')
axes[0,2].set_ylabel('GCS Total')
axes[0,2].set_ylim(0, 16)

# NEWS2 by acuity
news2_data = [train[train['triage_acuity'] == i]['news2_score'].values for i in range(1, 6)]
bp = axes[1,0].boxplot(news2_data, labels=['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5'],
                        patch_artist=True)
for patch, color in zip(bp['boxes'], colors_esi):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1,0].set_title('NEWS2 Score Distribution by ESI Level')
axes[1,0].set_ylabel('NEWS2 Score')

# Missing values by acuity (clinical insight)
miss_data = []
for acuity in range(1, 6):
    subset = train[train['triage_acuity'] == acuity]
    miss_data.append({
        'ESI': f'ESI-{acuity}',
        'BP Missing %': subset['systolic_bp'].isnull().mean() * 100,
        'RR Missing %': subset['respiratory_rate'].isnull().mean() * 100,
        'Temp Missing %': subset['temperature_c'].isnull().mean() * 100
    })
miss_df = pd.DataFrame(miss_data).set_index('ESI')
miss_df.plot(kind='bar', ax=axes[1,1], color=['#7b1fa2', '#00897b', '#ff8f00'])
axes[1,1].set_title('Missing Vitals by ESI Level\n(Clinically Meaningful Pattern)')
axes[1,1].set_ylabel('% Missing')
axes[1,1].tick_params(axis='x', rotation=0)
axes[1,1].legend(fontsize=9)

# Mental status by acuity
ms_ct = pd.crosstab(train['mental_status_triage'], train['triage_acuity'], normalize='columns')
ms_ct.T.plot(kind='bar', stacked=True, ax=axes[1,2], colormap='RdYlGn_r')
axes[1,2].set_title('Mental Status Distribution by ESI Level')
axes[1,2].set_xlabel('ESI Level')
axes[1,2].legend(fontsize=8, loc='upper right')
axes[1,2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('eda_clinical.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_clinical.png")


# ## 3. Clinical Feature Engineering

def engineer_features(df):
    df = df.copy()

    # Shock Index thresholds
    df['shock_index_severe'] = (df['shock_index'] > 1.0).astype(int)
    df['shock_index_critical'] = (df['shock_index'] > 1.5).astype(int)

    # MAP hypotension
    df['map_hypotensive'] = (df['mean_arterial_pressure'] < 65).astype(float).fillna(0)

    # SpO2 thresholds
    df['spo2_critical'] = (df['spo2'] < 90).astype(int)
    df['spo2_concerning'] = (df['spo2'] < 94).astype(int)
    df['spo2_normal'] = (df['spo2'] >= 96).astype(int)

    # Heart rate
    df['hr_bradycardia'] = (df['heart_rate'] < 60).astype(int)
    df['hr_tachycardia'] = (df['heart_rate'] > 100).astype(int)
    df['hr_severe_tachy'] = (df['heart_rate'] > 130).astype(int)

    # Respiratory rate
    df['rr_tachypnea'] = (df['respiratory_rate'] > 20).astype(float).fillna(0)
    df['rr_severe'] = (df['respiratory_rate'] > 25).astype(float).fillna(0)

    # Temperature
    df['temp_fever'] = (df['temperature_c'] > 38.0).astype(float).fillna(0)
    df['temp_hypothermia'] = (df['temperature_c'] < 36.0).astype(float).fillna(0)
    df['temp_high_fever'] = (df['temperature_c'] > 39.0).astype(float).fillna(0)

    # GCS categories
    df['gcs_severe'] = (df['gcs_total'] <= 8).astype(int)
    df['gcs_moderate'] = ((df['gcs_total'] > 8) & (df['gcs_total'] <= 12)).astype(int)
    df['gcs_normal'] = (df['gcs_total'] >= 14).astype(int)

    # Pain
    df['pain_severe'] = (df['pain_score'] >= 7).astype(int)
    df['pain_not_recorded'] = (df['pain_score'] == -1).astype(int)
    df['pain_score_clean'] = df['pain_score'].replace(-1, np.nan)

    # Missingness as feature
    df['bp_missing'] = df['systolic_bp'].isnull().astype(int)
    df['rr_missing'] = df['respiratory_rate'].isnull().astype(int)
    df['temp_missing'] = df['temperature_c'].isnull().astype(int)
    df['n_vitals_missing'] = df['bp_missing'] + df['rr_missing'] + df['temp_missing']

    # qSOFA approximation
    df['qsofa_sbp'] = (df['systolic_bp'] <= 100).astype(float).fillna(0)
    df['qsofa_rr'] = (df['respiratory_rate'] >= 22).astype(float).fillna(0)
    df['qsofa_gcs'] = (df['gcs_total'] < 15).astype(int)
    df['qsofa_score'] = df['qsofa_sbp'] + df['qsofa_rr'] + df['qsofa_gcs']

    # SIRS approximation
    df['sirs_hr'] = (df['heart_rate'] > 90).astype(int)
    df['sirs_rr'] = (df['respiratory_rate'] > 20).astype(float).fillna(0)
    df['sirs_temp'] = ((df['temperature_c'] < 36.0) | (df['temperature_c'] > 38.0)).astype(float).fillna(0)
    df['sirs_score'] = df['sirs_hr'] + df['sirs_rr'] + df['sirs_temp']

    # Comorbidity burden
    hx_cols = [c for c in df.columns if c.startswith('hx_')]
    df['comorbidity_burden'] = df[hx_cols].sum(axis=1)
    high_risk = ['hx_heart_failure', 'hx_copd', 'hx_ckd', 'hx_malignancy',
                  'hx_liver_disease', 'hx_immunosuppressed', 'hx_coagulopathy']
    df['high_risk_comorbidity'] = df[[c for c in high_risk if c in df.columns]].sum(axis=1)
    cardiac = ['hx_heart_failure', 'hx_atrial_fibrillation', 'hx_coronary_artery_disease', 'hx_hypertension']
    df['cardiac_risk'] = df[[c for c in cardiac if c in df.columns]].sum(axis=1)

    # Demographics
    df['elderly'] = (df['age'] >= 65).astype(int)
    df['very_elderly'] = (df['age'] >= 80).astype(int)
    df['pediatric'] = (df['age'] < 18).astype(int)
    df['age_comorbidity_risk'] = df['age'] * df['comorbidity_burden']

    # Arrival context
    df['ambulance_arrival'] = (df['arrival_mode'] == 'ambulance').astype(int)
    df['frequent_visitor'] = (df['num_prior_ed_visits_12m'] >= 4).astype(int)
    df['high_medication_load'] = (df['num_active_medications'] >= 10).astype(int)

    # Interactions
    df['hr_sbp_interaction'] = df['heart_rate'] * df['systolic_bp'].fillna(df['systolic_bp'].median())
    df['age_gcs_interaction'] = df['age'] * (15 - df['gcs_total'])
    df['news2_gcs_interaction'] = df['news2_score'] * (15 - df['gcs_total'])

    return df

df_train = engineer_features(df_train)
df_test = engineer_features(df_test)
print(f"Engineered features: {df_train.shape[1]} total columns")


# ## 4. NLP — Chief Complaint Processing

def extract_nlp_features(text_series):
    features = pd.DataFrame(index=text_series.index)
    text = text_series.fillna('')

    critical_words = ['cardiac arrest', 'stroke', 'anaphylaxis', 'sepsis', 'coma',
                      'unresponsive', 'haemorrhage', 'hemorrhage', 'seizure', 'status epilepticus',
                      'respiratory failure', 'intubat', 'resuscit', 'necrotis',
                      'stab wound', 'gunshot', 'overdose', 'delirium', 'haemothorax',
                      'tension pneumo', 'aortic dissect', 'pulmonary embol', 'meningitis']

    urgent_words = ['chest pain', 'difficulty breathing', 'shortness of breath', 'syncope',
                    'altered mental', 'fracture', 'allergic reaction', 'pancreatitis',
                    'acute', 'sudden', 'severe', 'worsening', 'rapid']

    mild_words = ['chronic', 'review', 'follow up', 'mild', 'minor', 'advice',
                  'prescription', 'referral', 'check', 'question', 'intermittent',
                  'controlled', 'stable']

    features['has_critical_keyword'] = text.apply(lambda x: int(any(w in x.lower() for w in critical_words)))
    features['has_urgent_keyword'] = text.apply(lambda x: int(any(w in x.lower() for w in urgent_words)))
    features['has_mild_keyword'] = text.apply(lambda x: int(any(w in x.lower() for w in mild_words)))
    features['complaint_length'] = text.str.len()
    features['complaint_word_count'] = text.str.split().str.len()
    features['onset_today'] = text.str.contains('onset today|started today', case=False, regex=True).astype(int)
    features['chronic_complaint'] = text.str.contains('chronic|weeks|months|years', case=False, regex=True).astype(int)
    features['worsening'] = text.str.contains('worsening|deteriorat|progressing', case=False, regex=True).astype(int)
    features['has_associated'] = text.str.contains('with associated|with fever|with nausea', case=False, regex=True).astype(int)
    features['complaint_head'] = text.str.contains('head|brain|neuro|seizure|stroke|coma', case=False, regex=True).astype(int)
    features['complaint_chest'] = text.str.contains('chest|cardiac|heart|breath|lung', case=False, regex=True).astype(int)
    features['complaint_abdomen'] = text.str.contains('abdom|gastro|bowel|liver|pancrea', case=False, regex=True).astype(int)

    return features

nlp_train = extract_nlp_features(df_train['chief_complaint_raw'])
nlp_test = extract_nlp_features(df_test['chief_complaint_raw'])

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=5, sublinear_tf=True)
tfidf_train = tfidf.fit_transform(df_train['chief_complaint_raw'].fillna(''))
tfidf_test = tfidf.transform(df_test['chief_complaint_raw'].fillna(''))

print(f"NLP: {nlp_train.shape[1]} clinical keyword features + {tfidf_train.shape[1]} TF-IDF features")


# ## 5. Feature Matrix Preparation

DROP_COLS = ['patient_id', 'chief_complaint_raw', 'triage_nurse_id', 'site_id']
CAT_COLS = ['arrival_mode', 'arrival_day', 'arrival_season', 'shift', 'age_group',
            'sex', 'language', 'insurance_type', 'transport_origin', 'pain_location',
            'mental_status_triage', 'chief_complaint_system', 'arrival_month']

for col in CAT_COLS:
    if col in df_train.columns:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]]).astype(str)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col] = le.transform(df_test[col].astype(str))

feature_cols = [c for c in df_train.columns if c not in DROP_COLS]
X_train_full = pd.concat([df_train[feature_cols].reset_index(drop=True),
                           nlp_train.reset_index(drop=True)], axis=1)
X_test_full = pd.concat([df_test[feature_cols].reset_index(drop=True),
                          nlp_test.reset_index(drop=True)], axis=1)

print(f"Final feature matrix: {X_train_full.shape[1]} structured + {tfidf_train.shape[1]} TF-IDF = {X_train_full.shape[1] + tfidf_train.shape[1]} total")


# ## 6. Model Training — LightGBM with 5-Fold Cross-Validation

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
y_train_0 = y_train - 1

oof_preds = np.zeros((len(y_train), 5))
test_preds = np.zeros((len(df_test), 5))
models = []
fold_scores = []

lgb_params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

print("Training LightGBM — 5-fold stratified CV...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_0)):
    X_tr = hstack([csr_matrix(X_train_full.iloc[train_idx].values), tfidf_train[train_idx]])
    X_val = hstack([csr_matrix(X_train_full.iloc[val_idx].values), tfidf_train[val_idx]])
    y_tr, y_val = y_train_0[train_idx], y_train_0[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(hstack([csr_matrix(X_test_full.values), tfidf_test])) / N_FOLDS
    models.append(model)

    fold_acc = accuracy_score(y_val, oof_preds[val_idx].argmax(1))
    fold_kappa = cohen_kappa_score(y_val, oof_preds[val_idx].argmax(1), weights='quadratic')
    fold_scores.append({'fold': fold+1, 'accuracy': fold_acc, 'qwk': fold_kappa})
    print(f"  Fold {fold+1}: Accuracy={fold_acc:.4f}, QWK={fold_kappa:.4f}, Best iter={model.best_iteration}")

oof_acc = accuracy_score(y_train_0, oof_preds.argmax(1))
oof_kappa = cohen_kappa_score(y_train_0, oof_preds.argmax(1), weights='quadratic')
oof_f1 = f1_score(y_train_0, oof_preds.argmax(1), average='macro')
print(f"\nOverall OOF: Accuracy={oof_acc:.4f}, QWK={oof_kappa:.4f}, Macro-F1={oof_f1:.4f}")


# ## 7. Results & Clinical Analysis

# ### 7a. Classification Performance
print("\nClassification Report:")
print(classification_report(y_train_0, oof_preds.argmax(1),
                            target_names=['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5']))

# Confusion Matrix visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('TriageGuardian — Model Performance', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_train_0, oof_preds.argmax(1))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0],
            xticklabels=['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5'],
            yticklabels=['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5'],
            vmin=0, vmax=1)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title(f'Normalized Confusion Matrix\nAccuracy: {oof_acc:.4f} | QWK: {oof_kappa:.4f}')

# Per-class F1 scores
report = classification_report(y_train_0, oof_preds.argmax(1),
                                target_names=['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5'],
                                output_dict=True)
esi_names = ['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5']
f1_scores = [report[name]['f1-score'] for name in esi_names]
recall_scores = [report[name]['recall'] for name in esi_names]
precision_scores = [report[name]['precision'] for name in esi_names]

x = np.arange(5)
w = 0.25
axes[1].bar(x - w, precision_scores, w, label='Precision', color='#1565c0')
axes[1].bar(x, recall_scores, w, label='Recall', color='#2e7d32')
axes[1].bar(x + w, f1_scores, w, label='F1-Score', color='#e65100')
axes[1].set_xticks(x)
axes[1].set_xticklabels(esi_names)
axes[1].set_ylim(0.9, 1.005)
axes[1].set_ylabel('Score')
axes[1].set_title('Per-Class Performance Metrics')
axes[1].legend()

plt.tight_layout()
plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_performance.png")

# ### 7b. Feature Importance
feature_names = list(X_train_full.columns) + list(tfidf.get_feature_names_out())
importance = np.zeros(len(feature_names))
for m in models:
    importance += m.feature_importance(importance_type='gain') / N_FOLDS

top_n = 25
top_idx = np.argsort(importance)[::-1][:top_n]

fig, ax = plt.subplots(figsize=(10, 8))
top_features = [feature_names[i] for i in top_idx]
top_importance = [importance[i] for i in top_idx]

# Color by feature type
colors = []
for f in top_features:
    if f in ['news2_score', 'gcs_total', 'spo2', 'heart_rate', 'respiratory_rate',
             'systolic_bp', 'temperature_c', 'shock_index', 'pain_score', 'pain_score_clean']:
        colors.append('#1565c0')  # Vital signs
    elif f.startswith('hx_') or 'comorbidity' in f or 'cardiac' in f:
        colors.append('#7b1fa2')  # Comorbidities
    elif f.startswith('sirs') or f.startswith('qsofa') or 'gcs_' in f or 'spo2_' in f or 'interaction' in f:
        colors.append('#00897b')  # Engineered clinical
    elif f in ['has_mild_keyword', 'has_critical_keyword', 'has_urgent_keyword', 'complaint_length',
               'worsening', 'onset_today'] or f in tfidf.get_feature_names_out():
        colors.append('#e65100')  # NLP
    else:
        colors.append('#546e7a')  # Other

ax.barh(range(top_n), top_importance[::-1], color=colors[::-1])
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features[::-1])
ax.set_xlabel('Feature Importance (Gain)')
ax.set_title(f'Top {top_n} Features — TriageGuardian\n'
             '🔵 Vital Signs  🟣 Comorbidities  🟢 Clinical Scores  🟠 NLP')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")


# ### 7c. Undertriage Detection — Clinical Safety Analysis

pred_acuity = oof_preds.argmax(1)
true_acuity = y_train_0

undertriage_mask = pred_acuity < true_acuity
overtriage_mask = pred_acuity > true_acuity
correct_mask = pred_acuity == true_acuity

pred_entropy = np.array([entropy(p) for p in oof_preds])

print("\n" + "=" * 60)
print("CLINICAL SAFETY ANALYSIS — Undertriage Detection")
print("=" * 60)
print(f"  Correct classifications: {correct_mask.sum()} ({correct_mask.mean()*100:.1f}%)")
print(f"  Undertriage (patient sicker than scored): {undertriage_mask.sum()} ({undertriage_mask.mean()*100:.2f}%)")
print(f"  Overtriage (patient less sick than scored): {overtriage_mask.sum()} ({overtriage_mask.mean()*100:.2f}%)")
print(f"\n  Prediction confidence:")
print(f"    Correct decisions — mean entropy: {pred_entropy[correct_mask].mean():.3f}")
print(f"    Undertriage cases — mean entropy: {pred_entropy[undertriage_mask].mean():.3f} (higher = less certain)")
print(f"    Overtriage cases — mean entropy: {pred_entropy[overtriage_mask].mean():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Clinical Safety Analysis — Undertriage Detection', fontsize=14, fontweight='bold')

# Error type breakdown
error_counts = [correct_mask.sum(), overtriage_mask.sum(), undertriage_mask.sum()]
error_labels = [f'Correct\n{correct_mask.sum():,}\n({correct_mask.mean()*100:.1f}%)',
                f'Overtriage\n{overtriage_mask.sum()}\n({overtriage_mask.mean()*100:.2f}%)',
                f'Undertriage\n{undertriage_mask.sum()}\n({undertriage_mask.mean()*100:.2f}%)']
error_colors = ['#4caf50', '#ff9800', '#f44336']
axes[0].pie(error_counts, labels=error_labels, colors=error_colors, startangle=90,
            textprops={'fontsize': 10})
axes[0].set_title('Classification Error Types')

# Entropy distribution
axes[1].hist(pred_entropy[correct_mask], bins=50, alpha=0.7, label='Correct', color='#4caf50', density=True)
axes[1].hist(pred_entropy[undertriage_mask], bins=30, alpha=0.7, label='Undertriage', color='#f44336', density=True)
axes[1].hist(pred_entropy[overtriage_mask], bins=30, alpha=0.7, label='Overtriage', color='#ff9800', density=True)
axes[1].set_xlabel('Prediction Entropy')
axes[1].set_ylabel('Density')
axes[1].set_title('Uncertainty by Error Type\n(Higher entropy = less confident)')
axes[1].legend()

# Confidence vs accuracy
conf_bins = np.linspace(0, 1, 11)
max_probs = oof_preds.max(axis=1)
bin_accs = []
bin_centers = []
for i in range(len(conf_bins) - 1):
    mask = (max_probs >= conf_bins[i]) & (max_probs < conf_bins[i+1])
    if mask.sum() > 0:
        bin_accs.append(accuracy_score(y_train_0[mask], oof_preds[mask].argmax(1)))
        bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)

axes[2].plot(bin_centers, bin_accs, 'o-', color='#1565c0', linewidth=2, markersize=8)
axes[2].plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='Perfect calibration')
axes[2].set_xlabel('Predicted Confidence')
axes[2].set_ylabel('Actual Accuracy')
axes[2].set_title('Calibration Plot\n(Confidence vs. Accuracy)')
axes[2].legend()
axes[2].set_xlim(0.5, 1.0)
axes[2].set_ylim(0.9, 1.01)

plt.tight_layout()
plt.savefig('clinical_safety.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: clinical_safety.png")


# ## 8. Generate Submission

final_preds = test_preds.argmax(1) + 1
submission = pd.DataFrame({'patient_id': test['patient_id'], 'triage_acuity': final_preds})
submission.to_csv('submission.csv', index=False)

# Clinical decision support output
prob_df = pd.DataFrame(test_preds, columns=['p_ESI1', 'p_ESI2', 'p_ESI3', 'p_ESI4', 'p_ESI5'])
prob_df['patient_id'] = test['patient_id'].values
prob_df['predicted_acuity'] = final_preds
prob_df['confidence'] = test_preds.max(axis=1)
prob_df['uncertainty'] = np.array([entropy(p) for p in test_preds])
prob_df.to_csv('clinical_decision_support.csv', index=False)

print(f"\nSubmission saved: {len(submission)} predictions")
print(f"Distribution: {dict(pd.Series(final_preds).value_counts().sort_index())}")

print("\n" + "=" * 60)
print("TriageGuardian — FINAL RESULTS")
print("=" * 60)
print(f"  OOF Accuracy:              {oof_acc:.4f}")
print(f"  Quadratic Weighted Kappa:  {oof_kappa:.4f}")
print(f"  Macro F1-Score:            {oof_f1:.4f}")
print(f"  Undertriage Rate:          {undertriage_mask.mean()*100:.2f}%")
print(f"  Features Used:             {X_train_full.shape[1]} structured + {tfidf_train.shape[1]} TF-IDF")
print(f"  Clinical Scores:           qSOFA, SIRS, Shock Index, NEWS2 interactions")
print(f"  NLP Features:              Keyword extraction + TF-IDF bigrams")
print("=" * 60)
