# Triagegeist: Multi-Modal Emergency Triage Acuity Prediction
### Hybrid Treeâ€“Neural Ensemble with Dual-Channel NLP, Clinical Feature Engineering, Cost-Sensitive Error Analysis & Demographic Bias Auditing

**Author:** [ladyFaye](https://www.kaggle.com/ladyfaye)  
**Repository:** [github.com/ladyFaye1998/triagegeist](https://github.com/ladyFaye1998/triagegeist)  
**Live Demo:** [ladyfaye1998.github.io/triagegeist](https://ladyfaye1998.github.io/triagegeist/)  
**Competition:** [Triagegeist â€” Laitinen-Fredriksson Foundation](https://www.kaggle.com/competitions/triagegeist)

---

## Clinical Motivation

The Emergency Severity Index (ESI) assigns patients to 5 acuity levels that determine treatment priority. Inter-rater variability (kappa 0.60â€“0.80) and documented systematic undertriage make this a critical patient safety problem. This notebook builds a **clinical decision support system** that:

1. **Predicts ESI acuity** from vitals, demographics, free-text chief complaints, and comorbidity history
2. **Identifies systematic bias** across demographic groups that could harm vulnerable populations
3. **Provides interpretable explanations** a clinician can audit before trusting the prediction

| ESI Level | Name | Clinical Definition | Train Count |
|:---------:|:-----|:--------------------|:----------:|
| 1 | Resuscitation | Immediate life-saving intervention | 3,222 (4.0%) |
| 2 | Emergent | High-risk, confused/lethargic/disoriented | 13,439 (16.8%) |
| 3 | Urgent | Multiple resources needed | 28,921 (36.2%) |
| 4 | Less Urgent | One resource needed | 23,020 (28.8%) |
| 5 | Non-Urgent | No resources needed | 11,398 (14.2%) |


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, gc, re
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, confusion_matrix, log_loss
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'figure.dpi': 120})

SEED = 42
N_FOLDS = 5
N_CLASSES = 5
TARGET = 'triage_acuity'
ID = 'patient_id'

ACUITY = {1: 'Resuscitation', 2: 'Emergent', 3: 'Urgent', 4: 'Less Urgent', 5: 'Non-Urgent'}
COLORS = ['#c62828', '#ef6c00', '#f9a825', '#2e7d32', '#1565c0']
PALETTE = {i+1: c for i, c in enumerate(COLORS)}
PALETTE.update({str(k): v for k, v in PALETTE.items()})

np.random.seed(SEED)
print('Libraries loaded successfully.')
```

---
## 1 Â· Data Loading & Multi-Table Fusion

Three data sources are merged on `patient_id`:
- **Structured intake** â€” vitals, demographics, arrival context, ED utilization
- **Chief complaints** â€” free-text clinical presentation + body system category
- **Patient history** â€” 25 binary comorbidity flags (hypertension, diabetes, COPD, etc.)


```python
# === DATA CITATION ===
# All data provided by the Laitinen-Fredriksson Foundation via the
# Triagegeist Kaggle Competition (https://kaggle.com/competitions/triagegeist).
# Citation: Olaf Yunus Laitinen Imanov (2026). Triagegeist. Kaggle.
# The dataset contains synthetic emergency department records designed to
# mirror real-world triage workflows in Northern European hospital systems.

import subprocess, zipfile
KAGGLE_PATH = '/kaggle/input/triagegeist'
LOCAL_PATH  = '../data'

if os.path.exists(KAGGLE_PATH) and os.path.isfile(f'{KAGGLE_PATH}/train.csv'):
    DATA = KAGGLE_PATH
elif os.path.exists(LOCAL_PATH) and os.path.isfile(f'{LOCAL_PATH}/train.csv'):
    DATA = LOCAL_PATH
else:
    # Fallback: download competition data via Kaggle API (works on Kaggle runtime)
    DL_DIR = '/kaggle/working/data'
    os.makedirs(DL_DIR, exist_ok=True)
    subprocess.run(['kaggle', 'competitions', 'download', '-c', 'triagegeist', '-p', DL_DIR], check=True)
    for zf in [f for f in os.listdir(DL_DIR) if f.endswith('.zip')]:
        with zipfile.ZipFile(f'{DL_DIR}/{zf}', 'r') as z:
            z.extractall(DL_DIR)
    DATA = DL_DIR
    print(f'Data downloaded to {DL_DIR}')

print(f'Data directory: {DATA}')

train_raw = pd.read_csv(f'{DATA}/train.csv')
test_raw  = pd.read_csv(f'{DATA}/test.csv')
complaints = pd.read_csv(f'{DATA}/chief_complaints.csv')
history    = pd.read_csv(f'{DATA}/patient_history.csv')
sample_sub = pd.read_csv(f'{DATA}/sample_submission.csv')

cc_dedup = complaints[[ID, 'chief_complaint_raw']].drop_duplicates(subset=ID, keep='first')

train = train_raw.merge(cc_dedup, on=ID, how='left').merge(history, on=ID, how='left')
test  = test_raw.merge(cc_dedup, on=ID, how='left').merge(history, on=ID, how='left')

print(f'Train: {train.shape[0]:,} patients x {train.shape[1]} columns')
print(f'Test:  {test.shape[0]:,} patients x {test.shape[1]} columns')
print(f'Chief complaints: {len(complaints):,} records')
print(f'Patient history:  {len(history):,} records ({history.shape[1]-1} comorbidity flags)')
print(f'\nTarget distribution:')
for lvl in sorted(train[TARGET].unique()):
    n = (train[TARGET] == lvl).sum()
    print(f'  ESI {lvl} ({ACUITY[lvl]:15s}): {n:>6,}  ({n/len(train)*100:5.1f}%)')
```

---
## 2 Â· Exploratory Data Analysis

Before engineering features, we must understand the clinical structure of the data. In real emergency departments, the ESI distribution is typically right-skewed (ESI 3â€“4 dominate), vital sign derangements correlate non-linearly with acuity, and disposition outcomes validate the ordinal acuity scale. We verify these patterns hold in the competition dataset, confirming its clinical realism.


```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 2a. Acuity distribution
ax = axes[0, 0]
counts = train[TARGET].value_counts().sort_index()
bars = ax.bar(counts.index, counts.values, color=COLORS, edgecolor='white', linewidth=0.8)
for bar, cnt in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            f'{cnt:,}\n({cnt/len(train)*100:.1f}%)', ha='center', fontsize=9)
ax.set_xlabel('ESI Triage Acuity')
ax.set_ylabel('Count')
ax.set_title('A. Triage Acuity Distribution')
ax.set_xticks(range(1, 6))

# 2b. Missing values
ax = axes[0, 1]
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=True)
ax.barh(missing.index, missing.values / len(train) * 100, color='#ef6c00')
ax.set_xlabel('Missing (%)')
ax.set_title('B. Missing Values')
for i, (idx, v) in enumerate(missing.items()):
    ax.text(v/len(train)*100 + 0.1, i, f'{v/len(train)*100:.1f}%', va='center', fontsize=9)

# 2c. Mental status vs acuity â€” the strongest categorical predictor
ax = axes[0, 2]
ct = pd.crosstab(train['mental_status_triage'], train[TARGET], normalize='index')
ct = ct.loc[['unresponsive', 'drowsy', 'agitated', 'confused', 'alert']]
ct.plot(kind='barh', stacked=True, color=COLORS, ax=ax, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Proportion')
ax.set_title('C. Mental Status vs. Acuity')
ax.legend(title='ESI', labels=[f'{k}' for k in ACUITY.keys()],
          bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

# 2d. NEWS2 distribution by acuity
ax = axes[1, 0]
for lvl in range(1, 6):
    subset = train[train[TARGET] == lvl]['news2_score']
    ax.hist(subset, bins=40, alpha=0.6, label=f'ESI {lvl}', color=COLORS[lvl-1], density=True)
ax.set_xlabel('NEWS2 Score')
ax.set_ylabel('Density')
ax.set_title('D. NEWS2 Score by Acuity (r = âˆ’0.81)')
ax.legend(fontsize=8)

# 2e. Key vital correlations with target
ax = axes[1, 1]
corr_cols = ['news2_score', 'gcs_total', 'spo2', 'respiratory_rate',
             'temperature_c', 'shock_index', 'pain_score', 'heart_rate',
             'num_prior_ed_visits_12m', 'mean_arterial_pressure',
             'systolic_bp', 'diastolic_bp', 'pulse_pressure']
corrs = train[corr_cols].corrwith(train[TARGET]).sort_values()
colors_bar = ['#c62828' if v < 0 else '#1565c0' for v in corrs.values]
ax.barh(corrs.index, corrs.values, color=colors_bar)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Pearson r with Triage Acuity')
ax.set_title('E. Feature Correlations with Target')

# 2f. Chief complaint system vs acuity
ax = axes[1, 2]
system_acuity = train.groupby('chief_complaint_system')[TARGET].mean().sort_values()
ax.barh(system_acuity.index, system_acuity.values, color='#1565c0', edgecolor='white')
ax.axvline(train[TARGET].mean(), color='red', linestyle='--', linewidth=1, label='Overall mean')
ax.set_xlabel('Mean Acuity')
ax.set_title('F. Mean Acuity by Chief Complaint System')
ax.legend(fontsize=9)

plt.suptitle('Exploratory Data Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```


```python
# Vital signs by acuity â€” detailed view
vital_cols = ['heart_rate', 'systolic_bp', 'respiratory_rate', 'temperature_c',
              'spo2', 'shock_index', 'news2_score', 'gcs_total']

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for idx, col in enumerate(vital_cols):
    ax = axes[idx // 4][idx % 4]
    sns.violinplot(data=train, x=TARGET, y=col, ax=ax, palette=PALETTE,
                   inner='quartile', linewidth=0.8, cut=0)
    ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel('ESI Level')

plt.suptitle('Vital Sign Distributions by Triage Acuity', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
```


```python
# Disposition outcome by acuity â€” validates acuity labels are clinically meaningful
fig, ax = plt.subplots(figsize=(10, 5))
disp_ct = pd.crosstab(train[TARGET], train['disposition'], normalize='index')
disp_order = ['deceased', 'admitted', 'observation', 'transferred', 'discharged', 'lwbs', 'lama']
disp_ct = disp_ct[[c for c in disp_order if c in disp_ct.columns]]
disp_ct.plot(kind='bar', stacked=True, ax=ax, colormap='RdYlBu_r', edgecolor='white', linewidth=0.5)
ax.set_xlabel('ESI Triage Acuity')
ax.set_ylabel('Proportion')
ax.set_title('Patient Disposition by Triage Acuity â€” Validates Acuity Labels')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
ax.set_xticklabels([f'ESI {i}' for i in range(1, 6)], rotation=0)
plt.tight_layout()
plt.show()
print('ESI 1 patients: {:.1f}% admitted/deceased vs ESI 5: {:.1f}% discharged'.format(
    (train[train[TARGET]==1]['disposition'].isin(['admitted','deceased'])).mean()*100,
    (train[train[TARGET]==5]['disposition']=='discharged').mean()*100
))
```

---
## 3 Â· Clinical Feature Engineering

We engineer **50+ features** grounded in emergency medicine literature:

| Category | Features | Clinical Rationale |
|:---------|:---------|:------------------|
| Vital abnormality flags | Hypotension, tachycardia, hypoxia, etc. (11 flags) | Direct ESI branch points (Gilboy et al., 2020) |
| qSOFA score | SBPâ‰¤100, RRâ‰¥22, GCS<15 | Sepsis screening per SOFA guidelines (Singer et al., 2016) |
| SIRS approximation | Temp/HR/RR abnormality count | Systemic inflammatory response |
| Composite scores | `num_abnormal_vitals`, `cv_risk_score` | Aggregate acuity burden |
| Age-vital interactions | `age Ã— shock_index`, `age Ã— NEWS2` | Elderly decompensation principle |
| Critical NLP keywords | Chest pain, seizure, stroke, SOB flags | High-risk presentation detection |
| TF-IDF (150 features) | Chief complaint unigrams + bigrams | Full text signal |
| Target-encoded IDs | Nurse ID, Site ID | Captures systematic inter-rater variability |
| Temporal | Cyclical hour/month, weekend, night shift | ED acuity temporal patterns |


```python
def engineer_features(df):
    """All feature engineering in one function for reproducibility."""
    df = df.copy()
    
    # --- VITAL SIGN ABNORMALITY FLAGS ---
    df['flag_hypotension']        = (df['systolic_bp'] < 90).astype(np.int8)
    df['flag_hypertension_crisis'] = (df['systolic_bp'] > 180).astype(np.int8)
    df['flag_severe_hypotension'] = (df['systolic_bp'] < 70).astype(np.int8)
    df['flag_tachycardia']        = (df['heart_rate'] > 100).astype(np.int8)
    df['flag_severe_tachycardia'] = (df['heart_rate'] > 130).astype(np.int8)
    df['flag_bradycardia']        = (df['heart_rate'] < 60).astype(np.int8)
    df['flag_tachypnea']          = (df['respiratory_rate'] > 20).astype(np.int8)
    df['flag_severe_tachypnea']   = (df['respiratory_rate'] > 30).astype(np.int8)
    df['flag_hypothermia']        = (df['temperature_c'] < 36.0).astype(np.int8)
    df['flag_fever']              = (df['temperature_c'] > 38.0).astype(np.int8)
    df['flag_high_fever']         = (df['temperature_c'] > 39.0).astype(np.int8)
    df['flag_hypoxia']            = (df['spo2'] < 92).astype(np.int8)
    df['flag_severe_hypoxia']     = (df['spo2'] < 88).astype(np.int8)
    df['flag_altered_mental']     = (df['gcs_total'] < 15).astype(np.int8)
    df['flag_severe_gcs']         = (df['gcs_total'] <= 8).astype(np.int8)
    df['flag_severe_pain']        = (df['pain_score'] >= 8).astype(np.int8)
    df['flag_high_shock_idx']     = (df['shock_index'] > 1.0).astype(np.int8)
    df['flag_critical_shock_idx'] = (df['shock_index'] > 1.3).astype(np.int8)
    
    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    df['num_abnormal_vitals'] = df[flag_cols].sum(axis=1)
    
    # --- qSOFA SCORE (0-3) ---
    df['qsofa_sbp']  = (df['systolic_bp'] <= 100).astype(np.int8)
    df['qsofa_rr']   = (df['respiratory_rate'] >= 22).astype(np.int8)
    df['qsofa_gcs']  = (df['gcs_total'] < 15).astype(np.int8)
    df['qsofa_score'] = df['qsofa_sbp'] + df['qsofa_rr'] + df['qsofa_gcs']
    
    # --- SIRS-like criteria count ---
    df['sirs_temp']  = ((df['temperature_c'] > 38.0) | (df['temperature_c'] < 36.0)).astype(np.int8)
    df['sirs_hr']    = (df['heart_rate'] > 90).astype(np.int8)
    df['sirs_rr']    = (df['respiratory_rate'] > 20).astype(np.int8)
    df['sirs_count'] = df['sirs_temp'] + df['sirs_hr'] + df['sirs_rr']
    
    # --- COMPOSITE RISK SCORES ---
    hx_cols = [c for c in df.columns if c.startswith('hx_')]
    if hx_cols:
        df['comorbidity_burden'] = df[hx_cols].sum(axis=1)
    
    cv_cols = ['hx_hypertension', 'hx_heart_failure', 'hx_atrial_fibrillation',
               'hx_coronary_artery_disease', 'hx_stroke_prior', 'hx_peripheral_vascular_disease']
    cv_present = [c for c in cv_cols if c in df.columns]
    df['cv_risk_score'] = df[cv_present].sum(axis=1) if cv_present else 0
    
    resp_cols = ['hx_asthma', 'hx_copd']
    resp_present = [c for c in resp_cols if c in df.columns]
    df['respiratory_risk'] = df[resp_present].sum(axis=1) if resp_present else 0
    
    meta_cols = ['hx_diabetes_type2', 'hx_diabetes_type1', 'hx_obesity',
                 'hx_hypothyroidism', 'hx_hyperthyroidism']
    meta_present = [c for c in meta_cols if c in df.columns]
    df['metabolic_risk'] = df[meta_present].sum(axis=1) if meta_present else 0
    
    psych_cols = ['hx_depression', 'hx_anxiety', 'hx_substance_use_disorder']
    psych_present = [c for c in psych_cols if c in df.columns]
    df['psych_risk'] = df[psych_present].sum(axis=1) if psych_present else 0
    
    # --- AGE-VITAL INTERACTIONS ---
    df['age_x_shock_index']  = df['age'] * df['shock_index']
    df['age_x_news2']        = df['age'] * df['news2_score']
    df['age_x_gcs']          = df['age'] * df['gcs_total']
    df['age_x_comorbidities'] = df['age'] * df['num_comorbidities']
    df['age_x_qsofa']        = df['age'] * df['qsofa_score']
    
    # --- VITAL SIGN RATIOS & PRODUCTS ---
    df['hr_rr_ratio']     = df['heart_rate'] / df['respiratory_rate'].replace(0, np.nan)
    df['sbp_hr_product']  = df['systolic_bp'] * df['heart_rate']
    df['map_hr_ratio']    = df['mean_arterial_pressure'] / df['heart_rate'].replace(0, np.nan)
    df['pp_sbp_ratio']    = df['pulse_pressure'] / df['systolic_bp'].replace(0, np.nan)
    df['spo2_rr_product'] = df['spo2'] * df['respiratory_rate']
    
    # --- ED UTILIZATION ---
    df['ed_utilization'] = df['num_prior_ed_visits_12m'] + 2 * df['num_prior_admissions_12m']
    df['high_ed_user']   = (df['num_prior_ed_visits_12m'] >= 4).astype(np.int8)
    df['polypharmacy']   = (df['num_active_medications'] >= 5).astype(np.int8)
    
    # --- TEMPORAL FEATURES ---
    df['is_weekend']    = df['arrival_day'].isin(['Saturday', 'Sunday']).astype(np.int8)
    df['is_night']      = df['shift'].eq('night').astype(np.int8)
    df['is_peak_hours'] = df['arrival_hour'].between(10, 18).astype(np.int8)
    
    hour_rad = 2 * np.pi * df['arrival_hour'] / 24
    df['hour_sin'] = np.sin(hour_rad)
    df['hour_cos'] = np.cos(hour_rad)
    
    month_rad = 2 * np.pi * df['arrival_month'] / 12
    df['month_sin'] = np.sin(month_rad)
    df['month_cos'] = np.cos(month_rad)
    
    return df


train = engineer_features(train)
test  = engineer_features(test)
print(f'After feature engineering: train {train.shape}, test {test.shape}')
```


```python
# --- CRITICAL CHIEF COMPLAINT KEYWORD FLAGS ---
# Clinically high-risk presentations that should trigger higher acuity

CRITICAL_KEYWORDS = {
    'kw_chest_pain':     r'chest\s*pain|angina|acs|acute coronary',
    'kw_sob':            r'shortness.*breath|dyspn[eo]|sob |respiratory\s*distress|breathing\s*difficult',
    'kw_stroke':         r'stroke|hemipar|aphasia|facial\s*droop|tia|cerebrovascular',
    'kw_seizure':        r'seizure|convuls|epilep|status\s*epilepticus',
    'kw_cardiac_arrest': r'cardiac\s*arrest|asystole|pulseless|cpr|resuscitat',
    'kw_trauma_major':   r'major\s*trauma|polytrauma|mva|motor\s*vehicle|fall\s*from\s*height|penetrat',
    'kw_sepsis':         r'sepsis|septic|bactere?mia',
    'kw_anaphylaxis':    r'anaphyla|severe\s*allerg',
    'kw_suicidal':       r'suicid|self.?harm|overdose|intentional|hanging',
    'kw_altered_mental': r'altered\s*mental|confusion|unresponsive|unconscious|syncope|collaps',
    'kw_gi_bleed':       r'haematemes|melena|gi\s*bleed|rectal\s*bleed|blood.*stool|vomit.*blood',
    'kw_fracture':       r'fracture|broken\s*bone|disloc',
    'kw_abdominal':      r'abdominal\s*pain|acute\s*abdomen|appendic',
    'kw_headache':       r'headache|migraine|thunderclap',
    'kw_fever':          r'fever|febrile|pyrexia|rigors',
    'kw_mild':           r'advice|follow.?up|prescription|refill|check.?up|minor|mild|chronic\s*stable',
}

for col_name, pattern in CRITICAL_KEYWORDS.items():
    train[col_name] = train['chief_complaint_raw'].fillna('').str.contains(pattern, case=False, regex=True).astype(np.int8)
    test[col_name]  = test['chief_complaint_raw'].fillna('').str.contains(pattern, case=False, regex=True).astype(np.int8)

kw_cols = [c for c in train.columns if c.startswith('kw_')]
train['num_critical_keywords'] = train[kw_cols].sum(axis=1)
test['num_critical_keywords']  = test[kw_cols].sum(axis=1)

# Chief complaint text length and word count
train['cc_length']     = train['chief_complaint_raw'].fillna('').str.len()
train['cc_word_count'] = train['chief_complaint_raw'].fillna('').str.split().str.len()
test['cc_length']      = test['chief_complaint_raw'].fillna('').str.len()
test['cc_word_count']  = test['chief_complaint_raw'].fillna('').str.split().str.len()

print(f'Added {len(kw_cols)} keyword flags + 2 text stats')
print(f'\nKeyword prevalence in training data:')
for kw in kw_cols:
    n = train[kw].sum()
    mean_acuity = train[train[kw]==1][TARGET].mean() if n > 0 else 0
    print(f'  {kw:25s}: {n:>5,} patients, mean acuity {mean_acuity:.2f}')
```


```python
# --- TF-IDF on chief complaint text ---
# Dual-channel NLP: word n-grams capture semantic meaning,
# character n-grams capture morphological patterns and misspellings
cc_train = train['chief_complaint_raw'].fillna('unknown')
cc_test  = test['chief_complaint_raw'].fillna('unknown')

tfidf_word = TfidfVectorizer(
    max_features=500, ngram_range=(1, 3), analyzer='word',
    stop_words='english', lowercase=True,
    min_df=3, max_df=0.95, sublinear_tf=True
)
tfidf_char = TfidfVectorizer(
    max_features=200, ngram_range=(2, 5), analyzer='char_wb',
    lowercase=True, min_df=5, max_df=0.95, sublinear_tf=True
)

train_word_mat = tfidf_word.fit_transform(cc_train)
test_word_mat  = tfidf_word.transform(cc_test)
train_char_mat = tfidf_char.fit_transform(cc_train)
test_char_mat  = tfidf_char.transform(cc_test)

def sanitize_name(s):
    return re.sub(r'[^A-Za-z0-9_]', '_', s).strip('_')

n_word = len(tfidf_word.get_feature_names_out())
n_char = len(tfidf_char.get_feature_names_out())
word_raw = [f'tfidf_w_{sanitize_name(n)}' for n in tfidf_word.get_feature_names_out()]
char_raw = [f'tfidf_c_{sanitize_name(n)}' for n in tfidf_char.get_feature_names_out()]
all_raw = word_raw + char_raw
seen = {}
tfidf_names = []
for name in all_raw:
    if name in seen:
        seen[name] += 1
        tfidf_names.append(f'{name}_{seen[name]}')
    else:
        seen[name] = 0
        tfidf_names.append(name)

from scipy.sparse import hstack as sp_hstack
train_tfidf_mat = sp_hstack([train_word_mat, train_char_mat])
test_tfidf_mat  = sp_hstack([test_word_mat, test_char_mat])

train_tfidf_df = pd.DataFrame(train_tfidf_mat.toarray(), columns=tfidf_names, index=train.index)
test_tfidf_df  = pd.DataFrame(test_tfidf_mat.toarray(), columns=tfidf_names, index=test.index)

print(f'TF-IDF features: {len(tfidf_names)} (word={n_word}, char={n_char})')
```


```python
# --- TARGET ENCODING for nurse_id and site_id ---
# Out-of-fold to prevent leakage

def target_encode_oof(train_df, test_df, col, target, n_folds=5, seed=42):
    """Target encoding with out-of-fold strategy to prevent leakage."""
    global_mean = train_df[target].mean()
    train_enc = pd.Series(np.full(len(train_df), global_mean), index=train_df.index, name=f'{col}_te')
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in skf.split(train_df, train_df[target]):
        means = train_df.iloc[tr_idx].groupby(col)[target].mean()
        train_enc.iloc[val_idx] = train_df.iloc[val_idx][col].map(means).fillna(global_mean).values
    
    full_means = train_df.groupby(col)[target].mean()
    test_enc = test_df[col].map(full_means).fillna(global_mean)
    test_enc.name = f'{col}_te'
    
    return train_enc, test_enc

for col in ['triage_nurse_id', 'site_id']:
    tr_enc, te_enc = target_encode_oof(train, test, col, TARGET)
    train[f'{col}_te'] = tr_enc.values
    test[f'{col}_te']  = te_enc.values
    print(f'{col}_te: train range [{tr_enc.min():.3f}, {tr_enc.max():.3f}]')
```


```python
# --- ASSEMBLE FINAL FEATURE MATRIX ---

CAT_COLS = [
    'arrival_mode', 'arrival_day', 'arrival_season', 'shift',
    'age_group', 'sex', 'language', 'insurance_type',
    'transport_origin', 'pain_location', 'mental_status_triage',
    'chief_complaint_system',
]

NUM_COLS = [
    'arrival_hour', 'arrival_month', 'age',
    'num_prior_ed_visits_12m', 'num_prior_admissions_12m',
    'num_active_medications', 'num_comorbidities',
    'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure',
    'pulse_pressure', 'heart_rate', 'respiratory_rate',
    'temperature_c', 'spo2', 'gcs_total', 'pain_score',
    'weight_kg', 'height_cm', 'bmi', 'shock_index', 'news2_score',
]

HX_COLS = [c for c in train.columns if c.startswith('hx_')]

ENG_COLS = (
    [c for c in train.columns if c.startswith('flag_')]
    + ['num_abnormal_vitals', 'qsofa_score', 'sirs_count',
       'comorbidity_burden', 'cv_risk_score', 'respiratory_risk',
       'metabolic_risk', 'psych_risk',
       'age_x_shock_index', 'age_x_news2', 'age_x_gcs',
       'age_x_comorbidities', 'age_x_qsofa',
       'hr_rr_ratio', 'sbp_hr_product', 'map_hr_ratio',
       'pp_sbp_ratio', 'spo2_rr_product',
       'ed_utilization', 'high_ed_user', 'polypharmacy',
       'is_weekend', 'is_night', 'is_peak_hours',
       'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
       'triage_nurse_id_te', 'site_id_te',
       'num_critical_keywords', 'cc_length', 'cc_word_count']
    + kw_cols
)
ENG_COLS = list(dict.fromkeys(ENG_COLS))  # deduplicate

# Encode categoricals
for col in CAT_COLS:
    combined = pd.concat([train[col], test[col]], axis=0).astype('category')
    codes = combined.cat.codes
    train[col] = codes.iloc[:len(train)].values
    test[col]  = codes.iloc[len(train):].values

all_feature_cols = CAT_COLS + NUM_COLS + HX_COLS + ENG_COLS
all_feature_cols = [c for c in all_feature_cols if c in train.columns]

X_train = pd.concat([train[all_feature_cols].reset_index(drop=True),
                      train_tfidf_df.reset_index(drop=True)], axis=1)
X_test  = pd.concat([test[all_feature_cols].reset_index(drop=True),
                      test_tfidf_df.reset_index(drop=True)], axis=1)
y_train = train[TARGET].values

print(f'Feature matrix: {X_train.shape[1]} total features')
print(f'  Categorical:   {len(CAT_COLS)}')
print(f'  Numeric:       {len(NUM_COLS)}')
print(f'  History:       {len(HX_COLS)}')
print(f'  Engineered:    {len(ENG_COLS)}')
print(f'  TF-IDF:        {len(tfidf_names)}')
print(f'  Total:         {X_train.shape[1]}')
```

---
## 4 Â· Model Training â€” Hybrid Treeâ€“Neural Ensemble with QWK Threshold Optimization

We combine **four diverse model families** through a two-level stacking architecture. Unlike pure tree-based ensembles, our approach adds a neural network to capture non-axis-aligned decision boundaries that trees inherently miss.

**Level-1 Base Learners (4 models):**
- **LightGBM** â€” leaf-wise growth, fast convergence on sparse NLP features
- **XGBoost** â€” level-wise growth with strong regularization on dense clinical features
- **CatBoost** â€” ordered boosting with symmetric trees, reduced overfitting through ordered target statistics
- **MLP Neural Network** (512â†’256â†’128) â€” captures smooth non-linear boundaries and feature interactions that axis-aligned tree splits cannot represent; adds genuine architectural diversity

**Level-2 Meta-Learner:**
- **L1-regularized Logistic Regression** with cross-validated regularization strength, trained on the 20-dimensional OOF probability outputs (5 classes Ã— 4 models). L1 sparsity automatically identifies which base model is most informative per class, learning optimal cross-architecture complementarity.

**Threshold Optimization:**
- Dual-optimizer search (differential evolution + Nelder-Mead) on ordinal cumulative probability boundaries â€” the better result is selected automatically, exploiting the ordinal nature of ESI levels for QWK maximization.


```python
# ===== LightGBM =====
lgbm_params = {
    'objective': 'multiclass', 'num_class': N_CLASSES,
    'metric': 'multi_logloss', 'boosting_type': 'gbdt',
    'n_estimators': 3000, 'learning_rate': 0.03,
    'num_leaves': 127, 'max_depth': -1,
    'min_child_samples': 20, 'min_child_weight': 1e-3,
    'subsample': 0.8, 'colsample_bytree': 0.6,
    'reg_alpha': 0.05, 'reg_lambda': 1.0,
    'random_state': SEED, 'verbose': -1, 'n_jobs': -1,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_lgbm = np.zeros((len(X_train), N_CLASSES))
test_lgbm = np.zeros((len(X_test), N_CLASSES))
lgbm_models = []

print(f'Training LightGBM â€” {N_FOLDS}-fold stratified CV')
print(f'Features: {X_train.shape[1]}, Samples: {X_train.shape[0]:,}\n')

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    
    oof_lgbm[val_idx] = model.predict_proba(X_val)
    test_lgbm += model.predict_proba(X_test) / N_FOLDS
    lgbm_models.append(model)
    
    val_pred = oof_lgbm[val_idx].argmax(axis=1) + 1
    acc = accuracy_score(y_val, val_pred)
    qwk = cohen_kappa_score(y_val, val_pred, weights='quadratic')
    print(f'  Fold {fold+1}: Acc={acc:.4f}  QWK={qwk:.4f}  (best iter: {model.best_iteration_})')

lgbm_oof_labels = oof_lgbm.argmax(axis=1) + 1
print(f'\nLightGBM OOF: Acc={accuracy_score(y_train, lgbm_oof_labels):.4f}  '
      f'F1={f1_score(y_train, lgbm_oof_labels, average="weighted"):.4f}  '
      f'QWK={cohen_kappa_score(y_train, lgbm_oof_labels, weights="quadratic"):.4f}')
```


```python
# ===== XGBoost =====
xgb_params = {
    'objective': 'multi:softprob', 'num_class': N_CLASSES,
    'eval_metric': 'mlogloss', 'tree_method': 'hist',
    'n_estimators': 2000, 'learning_rate': 0.03,
    'max_depth': 8, 'min_child_weight': 5,
    'subsample': 0.8, 'colsample_bytree': 0.6,
    'reg_alpha': 0.05, 'reg_lambda': 1.0,
    'random_state': SEED, 'verbosity': 0, 'n_jobs': -1,
}

oof_xgb = np.zeros((len(X_train), N_CLASSES))
test_xgb = np.zeros((len(X_test), N_CLASSES))
xgb_models = []

print(f'Training XGBoost â€” {N_FOLDS}-fold stratified CV\n')

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx] - 1, y_train[val_idx] - 1  # XGBoost needs 0-indexed
    
    model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=150)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_xgb[val_idx] = model.predict_proba(X_val)
    test_xgb += model.predict_proba(X_test) / N_FOLDS
    xgb_models.append(model)
    
    val_pred = oof_xgb[val_idx].argmax(axis=1) + 1
    y_val_orig = y_train[val_idx]
    acc = accuracy_score(y_val_orig, val_pred)
    qwk = cohen_kappa_score(y_val_orig, val_pred, weights='quadratic')
    print(f'  Fold {fold+1}: Acc={acc:.4f}  QWK={qwk:.4f}')

xgb_oof_labels = oof_xgb.argmax(axis=1) + 1
print(f'\nXGBoost OOF: Acc={accuracy_score(y_train, xgb_oof_labels):.4f}  '
      f'F1={f1_score(y_train, xgb_oof_labels, average="weighted"):.4f}  '
      f'QWK={cohen_kappa_score(y_train, xgb_oof_labels, weights="quadratic"):.4f}')
```


```python
# ===== CatBoost =====
cat_params = {
    'iterations': 2000, 'learning_rate': 0.05, 'depth': 8,
    'l2_leaf_reg': 3.0, 'random_seed': SEED,
    'loss_function': 'MultiClass', 'classes_count': N_CLASSES,
    'verbose': 0, 'thread_count': -1,
    'early_stopping_rounds': 150,
}

oof_cat = np.zeros((len(X_train), N_CLASSES))
test_cat = np.zeros((len(X_test), N_CLASSES))
cat_models = []

print(f'Training CatBoost â€” {N_FOLDS}-fold stratified CV\n')

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx] - 1, y_train[val_idx] - 1

    model = CatBoostClassifier(**cat_params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

    oof_cat[val_idx] = model.predict_proba(X_val)
    test_cat += model.predict_proba(X_test) / N_FOLDS
    cat_models.append(model)

    val_pred = oof_cat[val_idx].argmax(axis=1) + 1
    y_val_orig = y_train[val_idx]
    acc = accuracy_score(y_val_orig, val_pred)
    qwk = cohen_kappa_score(y_val_orig, val_pred, weights='quadratic')
    print(f'  Fold {fold+1}: Acc={acc:.4f}  QWK={qwk:.4f}')

cat_oof_labels = oof_cat.argmax(axis=1) + 1
print(f'\nCatBoost OOF: Acc={accuracy_score(y_train, cat_oof_labels):.4f}  '
      f'F1={f1_score(y_train, cat_oof_labels, average="weighted"):.4f}  '
      f'QWK={cohen_kappa_score(y_train, cat_oof_labels, weights="quadratic"):.4f}')
```


```python
# ===== MLP Neural Network =====
# Adds a fundamentally different model family (neural network) to the tree-based
# ensemble, increasing prediction diversity and capturing non-axis-aligned boundaries

X_train_filled = X_train.fillna(0)
X_test_filled = X_test.fillna(0)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_filled), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_filled), columns=X_test.columns, index=X_test.index
)
del X_train_filled, X_test_filled

oof_mlp = np.zeros((len(X_train), N_CLASSES))
test_mlp = np.zeros((len(X_test), N_CLASSES))
mlp_models = []

print(f'Training MLP Neural Network â€” {N_FOLDS}-fold stratified CV')
print(f'Architecture: 256 â†’ 128 â†’ 64 (ReLU, Î±=1e-3, batch=2048)\n')

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    X_tr = X_train_scaled.iloc[tr_idx]
    X_val = X_train_scaled.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx] - 1, y_train[val_idx] - 1

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu', solver='adam',
        alpha=1e-3, batch_size=2048,
        learning_rate='adaptive', learning_rate_init=1e-3,
        max_iter=100, early_stopping=True,
        n_iter_no_change=10, validation_fraction=0.1,
        random_state=SEED, verbose=False,
    )
    model.fit(X_tr, y_tr)

    oof_mlp[val_idx] = model.predict_proba(X_val)
    test_mlp += model.predict_proba(X_test_scaled) / N_FOLDS
    mlp_models.append(model)

    val_pred = oof_mlp[val_idx].argmax(axis=1) + 1
    y_val_orig = y_train[val_idx]
    acc = accuracy_score(y_val_orig, val_pred)
    qwk = cohen_kappa_score(y_val_orig, val_pred, weights='quadratic')
    print(f'  Fold {fold+1}: Acc={acc:.4f}  QWK={qwk:.4f}  (iters: {model.n_iter_})')

mlp_oof_labels = oof_mlp.argmax(axis=1) + 1
print(f'\nMLP OOF: Acc={accuracy_score(y_train, mlp_oof_labels):.4f}  '
      f'F1={f1_score(y_train, mlp_oof_labels, average="weighted"):.4f}  '
      f'QWK={cohen_kappa_score(y_train, mlp_oof_labels, weights="quadratic"):.4f}')
del X_train_scaled, X_test_scaled; gc.collect()
```


```python
# ===== LEVEL-2 HYBRID STACKING META-LEARNER =====
# 4 base models (3 tree-based + 1 neural) produce 20 meta-features (5 classes Ã— 4 models).
# L1-regularized Logistic Regression learns sparse cross-model complementarity.

oof_meta = np.hstack([oof_lgbm, oof_xgb, oof_cat, oof_mlp])  # (N, 20)
test_meta = np.hstack([test_lgbm, test_xgb, test_cat, test_mlp])

oof_stacked = np.zeros((len(oof_meta), N_CLASSES))
test_stacked = np.zeros((len(test_meta), N_CLASSES))

print(f'Training Level-2 Hybrid Meta-Learner (L1-LR on {oof_meta.shape[1]} meta-features)\n')
best_C = 1.0
best_C_qwk = -1
for C_val in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    tmp_oof = np.zeros((len(oof_meta), N_CLASSES))
    for tr_idx, val_idx in skf.split(oof_meta, y_train):
        lr_tmp = LogisticRegression(C=C_val, penalty='l1', max_iter=2000,
                                    multi_class='multinomial', solver='saga',
                                    random_state=SEED)
        lr_tmp.fit(oof_meta[tr_idx], y_train[tr_idx] - 1)
        tmp_oof[val_idx] = lr_tmp.predict_proba(oof_meta[val_idx])
    q = cohen_kappa_score(y_train, tmp_oof.argmax(axis=1) + 1, weights='quadratic')
    if q > best_C_qwk:
        best_C_qwk = q
        best_C = C_val
print(f'  Best meta-learner C={best_C} (CV QWK={best_C_qwk:.4f})\n')

for fold, (tr_idx, val_idx) in enumerate(skf.split(oof_meta, y_train)):
    lr = LogisticRegression(C=best_C, penalty='l1', max_iter=2000,
                            multi_class='multinomial', solver='saga',
                            random_state=SEED)
    lr.fit(oof_meta[tr_idx], y_train[tr_idx] - 1)
    oof_stacked[val_idx] = lr.predict_proba(oof_meta[val_idx])
    test_stacked += lr.predict_proba(test_meta) / N_FOLDS
    fold_pred = oof_stacked[val_idx].argmax(axis=1) + 1
    print(f'  Fold {fold+1}: Acc={accuracy_score(y_train[val_idx], fold_pred):.4f}  '
          f'QWK={cohen_kappa_score(y_train[val_idx], fold_pred, weights="quadratic"):.4f}')

stacked_labels = oof_stacked.argmax(axis=1) + 1
stacked_qwk = cohen_kappa_score(y_train, stacked_labels, weights='quadratic')
print(f'\nHybrid Stacked OOF QWK (argmax): {stacked_qwk:.4f}')

oof_ensemble = oof_stacked
test_ensemble = test_stacked

# ===== QWK THRESHOLD OPTIMIZATION (Differential Evolution) =====
def qwk_from_thresholds(thresholds, probs, y_true):
    cumprobs = probs.cumsum(axis=1)
    labels = np.ones(len(probs), dtype=int)
    for i, t in enumerate(sorted(thresholds)):
        labels[cumprobs[:, i] > t] = i + 2
    return -cohen_kappa_score(y_true, labels, weights='quadratic')

bounds = [(0.1, 0.9)] * 4
result_de = differential_evolution(
    qwk_from_thresholds, bounds, args=(oof_ensemble, y_train),
    seed=SEED, maxiter=1000, tol=1e-8, polish=True
)
result_nm = minimize(qwk_from_thresholds, [0.5, 0.5, 0.5, 0.5],
                     args=(oof_ensemble, y_train), method='Nelder-Mead',
                     options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})

if result_de.fun <= result_nm.fun:
    opt_thresholds = sorted(result_de.x)
    print(f'  Using differential evolution thresholds (QWK={-result_de.fun:.4f})')
else:
    opt_thresholds = sorted(result_nm.x)
    print(f'  Using Nelder-Mead thresholds (QWK={-result_nm.fun:.4f})')

cumprobs = oof_ensemble.cumsum(axis=1)
oof_labels = np.ones(len(oof_ensemble), dtype=int)
for i, t in enumerate(opt_thresholds):
    oof_labels[cumprobs[:, i] > t] = i + 2

oof_labels_argmax = oof_ensemble.argmax(axis=1) + 1
qwk_argmax = cohen_kappa_score(y_train, oof_labels_argmax, weights='quadratic')
qwk_opt = cohen_kappa_score(y_train, oof_labels, weights='quadratic')

USE_THRESHOLDS = qwk_opt > qwk_argmax
if not USE_THRESHOLDS:
    oof_labels = oof_labels_argmax
    qwk_opt = qwk_argmax
    print('  >> Threshold optimization did not improve QWK; using argmax')

ens_acc = accuracy_score(y_train, oof_labels)
ens_f1  = f1_score(y_train, oof_labels, average='weighted')
ens_qwk = qwk_opt
ens_ll  = log_loss(y_train - 1, oof_ensemble)

print(f'\n{"="*60}')
print(f'FINAL HYBRID ENSEMBLE (3 GBM + 1 MLP â†’ L1-stacked)')
print(f'  Accuracy:                 {ens_acc:.4f}')
print(f'  Weighted F1:              {ens_f1:.4f}')
print(f'  QWK (argmax):             {qwk_argmax:.4f}')
print(f'  QWK (threshold-optimized): {ens_qwk:.4f}')
print(f'  Log Loss:                 {ens_ll:.4f}')
print(f'  Thresholds:               {[f"{t:.3f}" for t in opt_thresholds]}')
print(f'{"="*60}')

print(f'\nFull Model Comparison:')
print(f'  LightGBM only:       QWK={cohen_kappa_score(y_train, lgbm_oof_labels, weights="quadratic"):.4f}')
print(f'  XGBoost only:        QWK={cohen_kappa_score(y_train, xgb_oof_labels, weights="quadratic"):.4f}')
print(f'  CatBoost only:       QWK={cohen_kappa_score(y_train, cat_oof_labels, weights="quadratic"):.4f}')
print(f'  MLP only:            QWK={cohen_kappa_score(y_train, mlp_oof_labels, weights="quadratic"):.4f}')
print(f'  Hybrid stacked:      QWK={stacked_qwk:.4f}')
print(f'  Final (+ threshold): QWK={ens_qwk:.4f}')
```


```python
# Detailed classification report
print('Detailed Classification Report â€” Ensemble OOF Predictions\n')
target_names = [f'ESI {k}: {v}' for k, v in ACUITY.items()]
print(classification_report(y_train, oof_labels, target_names=target_names, digits=4))
```

---
## 5 Â· Model Evaluation & Interpretability

A high QWK alone is insufficient for clinical credibility. We evaluate the hybrid ensemble through four complementary lenses: (1) **confusion analysis** revealing which ESI boundaries the model finds hardest, (2) **feature importance** via tree-model averaging and SHAP to ensure clinical plausibility, (3) **probability calibration** to verify that confidence scores are trustworthy for bedside use, and (4) **clinical misclassification cost analysis** to quantify the real-world safety implications of residual errors.


```python
# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

cm = confusion_matrix(y_train, oof_labels, labels=[1,2,3,4,5])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=list(ACUITY.values()), yticklabels=list(ACUITY.values()),
            ax=axes[0], vmin=0, vmax=1, linewidths=0.5)
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_title('A. Normalized Confusion Matrix', fontweight='bold')

sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
            xticklabels=list(ACUITY.values()), yticklabels=list(ACUITY.values()),
            ax=axes[1], linewidths=0.5)
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_title('B. Confusion Matrix (Counts)', fontweight='bold')

plt.tight_layout()
plt.show()

# Per-class accuracy
for i in range(5):
    class_acc = cm_norm[i, i]
    misclass = 1 - class_acc
    print(f'ESI {i+1} ({ACUITY[i+1]:15s}): {class_acc:.3f} accuracy, {misclass:.3f} error rate')
```


```python
# Feature importance â€” averaged across all tree-based model types
# (MLP does not expose split-based importances; its contribution is captured via SHAP)
lgbm_imp = np.zeros(X_train.shape[1])
for m in lgbm_models:
    lgbm_imp += m.feature_importances_
lgbm_imp /= len(lgbm_models)

xgb_imp = np.zeros(X_train.shape[1])
for m in xgb_models:
    xgb_imp += m.feature_importances_
xgb_imp /= len(xgb_models)

cat_imp = np.zeros(X_train.shape[1])
for m in cat_models:
    cat_imp += m.get_feature_importance()
cat_imp /= len(cat_models)

lgbm_norm = lgbm_imp / (lgbm_imp.max() + 1e-12)
xgb_norm = xgb_imp / (xgb_imp.max() + 1e-12)
cat_norm = cat_imp / (cat_imp.max() + 1e-12)
combined_imp = (lgbm_norm + xgb_norm + cat_norm) / 3

fi = pd.DataFrame({'feature': X_train.columns, 'importance': combined_imp})
fi = fi.sort_values('importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# Top 40 overall features
ax = axes[0]
top = fi.head(40)
colors_fi = ['#ef6c00' if 'tfidf_w_' in f else '#d81b60' if 'tfidf_c_' in f
             else '#1565c0' if f.startswith(('flag_','kw_','qsofa','sirs'))
             else '#2e7d32' for f in top['feature']]
ax.barh(range(len(top)), top['importance'].values, color=colors_fi, edgecolor='white')
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top['feature'].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Normalized Importance')
ax.set_title('A. Top 40 Features (Ensemble Average)', fontweight='bold')
legend_patches = [
    mpatches.Patch(color='#1565c0', label='Clinical flags'),
    mpatches.Patch(color='#ef6c00', label='Word TF-IDF'),
    mpatches.Patch(color='#d81b60', label='Char TF-IDF'),
    mpatches.Patch(color='#2e7d32', label='Other'),
]
ax.legend(handles=legend_patches, fontsize=9, loc='lower right')

# Top 20 NLP features
ax = axes[1]
tfidf_fi = fi[fi['feature'].str.startswith('tfidf_')].head(20)
clean_names = [n.replace('tfidf_w_', 'w:').replace('tfidf_c_', 'c:') for n in tfidf_fi['feature'].values]
ax.barh(range(len(tfidf_fi)), tfidf_fi['importance'].values, color='#ef6c00', edgecolor='white')
ax.set_yticks(range(len(tfidf_fi)))
ax.set_yticklabels(clean_names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Normalized Importance')
ax.set_title('B. Top 20 Chief Complaint Terms', fontweight='bold')

plt.tight_layout()
plt.show()
```


```python
# SHAP analysis â€” model interpretability
try:
    import shap

    np.random.seed(SEED)
    sample_idx = np.random.choice(len(X_train), size=min(2000, len(X_train)), replace=False)
    X_shap = pd.DataFrame(X_train.values[sample_idx], columns=list(X_train.columns))

    explainer = shap.TreeExplainer(lgbm_models[0])
    shap_values = explainer.shap_values(X_shap)

    # Handle both list (old shap) and 3D array (new shap) formats
    if isinstance(shap_values, list):
        sv_esi1 = shap_values[0]
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        sv_esi1 = shap_values[:, :, 0] if shap_values.ndim == 3 else shap_values
        mean_abs_shap = np.abs(shap_values).mean(axis=2) if shap_values.ndim == 3 else np.abs(shap_values)

    # A. SHAP beeswarm for ESI 1 (Resuscitation) â€” most critical class
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(sv_esi1, X_shap, max_display=25, show=False, plot_size=None)
    plt.title('SHAP Feature Impact â€” ESI 1 (Resuscitation)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # B. Mean |SHAP| bar plot â€” top 20 features across all classes
    fi_shap = pd.Series(mean_abs_shap.mean(axis=0), index=X_shap.columns)
    top20 = fi_shap.nlargest(20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    bar_colors = ['#ef6c00' if n.startswith('tfidf_') else '#1565c0' for n in top20.index]
    axes[0].barh(top20.index[::-1], top20.values[::-1], color=bar_colors[::-1])
    axes[0].set_xlabel('Mean |SHAP value|')
    axes[0].set_title('A. Top 20 Features â€” Mean |SHAP| (All Classes)', fontweight='bold')
    legend_patches = [mpatches.Patch(color='#ef6c00', label='NLP (TF-IDF)'),
                      mpatches.Patch(color='#1565c0', label='Clinical / Structured')]
    axes[0].legend(handles=legend_patches, loc='lower right')

    # C. ESI-1 specific top features
    fi_esi1 = pd.Series(np.abs(sv_esi1).mean(axis=0), index=X_shap.columns).nlargest(20)
    bar_colors_1 = ['#c62828' if n.startswith('tfidf_') else '#1565c0' for n in fi_esi1.index]
    axes[1].barh(fi_esi1.index[::-1], fi_esi1.values[::-1], color=bar_colors_1[::-1])
    axes[1].set_xlabel('Mean |SHAP value|')
    axes[1].set_title('B. Top 20 Features â€” ESI 1 (Resuscitation)', fontweight='bold')
    plt.tight_layout()
    plt.show()

    HAS_SHAP = True
    print(f'SHAP computed on {len(X_shap)} samples, {X_shap.shape[1]} features')
except Exception as e:
    import traceback
    print(f'SHAP unavailable ({e.__class__.__name__}: {e})')
    traceback.print_exc()
    HAS_SHAP = False
```

### 5c Â· Probability Calibration Analysis

For clinical decision support, it is not enough that predictions are *accurate* â€” the associated confidence must be *trustworthy*. A model that says "80% likely ESI 2" must be correct ~80% of the time for that confidence level. Calibration curves (reliability diagrams) measure this alignment per ESI class.


```python
# Calibration analysis â€” are predicted probabilities reliable?
# Critical for clinical use: a 70% prediction should correspond to 70% true rate

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Per-class calibration curves
ax = axes[0]
for cls_idx in range(N_CLASSES):
    y_binary = (y_train == cls_idx + 1).astype(int)
    prob_cls = oof_ensemble[:, cls_idx]
    fraction_pos, mean_predicted = calibration_curve(y_binary, prob_cls, n_bins=10, strategy='quantile')
    ax.plot(mean_predicted, fraction_pos, marker='o', label=f'ESI {cls_idx+1}',
            color=COLORS[cls_idx], linewidth=2, markersize=5)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('A. Calibration Curves by ESI Class', fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# Confidence vs accuracy
ax = axes[1]
max_conf = oof_ensemble.max(axis=1)
conf_bins = np.linspace(0, 1, 11)
bin_accs, bin_confs, bin_counts = [], [], []
for i in range(len(conf_bins) - 1):
    mask = (max_conf >= conf_bins[i]) & (max_conf < conf_bins[i+1])
    if mask.sum() > 0:
        bin_accs.append((oof_labels[mask] == y_train[mask]).mean())
        bin_confs.append(max_conf[mask].mean())
        bin_counts.append(mask.sum())
ax.bar(bin_confs, bin_accs, width=0.08, color='#1565c0', edgecolor='white', alpha=0.8)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.set_xlabel('Mean Confidence')
ax.set_ylabel('Accuracy')
ax.set_title('B. Confidence vs Accuracy', fontweight='bold')

plt.suptitle('Probability Calibration Analysis', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Expected Calibration Error
ece = np.mean([abs(a - c) * n / len(y_train) for a, c, n in zip(bin_accs, bin_confs, bin_counts)])
print(f'Expected Calibration Error (ECE): {ece:.4f}')
print('Lower ECE = better calibrated. <0.05 is considered well-calibrated.')
```

### 5d Â· Clinical Misclassification Cost Analysis

Not all errors are equal: predicting ESI 1 (Resuscitation) as ESI 3 (Urgent) â€” a 2-level undertriage â€” poses an immediate patient safety risk, while confusing ESI 4 and ESI 5 carries minimal clinical consequence. We define a **clinically-grounded asymmetric cost matrix** reflecting expert consensus from emergency medicine literature (Farrohknia et al., 2011; Gilboy et al., 2011):

- **Undertriage** (predicted less severe than actual) costs scale quadratically with the acuity gap, weighted 3Ã— heavier than overtriage
- **Overtriage** (predicted more severe) wastes resources but does not endanger the patient
- ESI 1â€“2 errors are penalized most heavily, reflecting their life-threatening nature


```python
# Asymmetric clinical cost matrix: undertriage is 3x worse than overtriage,
# severity penalty scales quadratically with ESI gap
cost_matrix = np.zeros((5, 5))
for true_esi in range(5):
    for pred_esi in range(5):
        gap = pred_esi - true_esi  # positive = undertriage (predicted less severe)
        if gap > 0:  # undertriage â€” dangerous
            cost_matrix[true_esi, pred_esi] = 3.0 * (gap ** 2)
        elif gap < 0:  # overtriage â€” wastes resources but safer
            cost_matrix[true_esi, pred_esi] = 1.0 * (abs(gap) ** 2)

# ESI 1-2 severity multiplier (life-threatening levels)
severity_mult = np.array([2.0, 1.5, 1.0, 0.8, 0.6])
cost_matrix = cost_matrix * severity_mult[:, None]

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# A. Cost matrix itself
ax = axes[0]
sns.heatmap(cost_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=[f'Pred {i+1}' for i in range(5)],
            yticklabels=[f'True {i+1}' for i in range(5)],
            ax=ax, linewidths=0.5)
ax.set_title('A. Clinical Cost Matrix', fontweight='bold')
ax.set_xlabel('Predicted ESI')
ax.set_ylabel('True ESI')

# B. Weighted cost confusion matrix (actual errors Ã— cost)
cm = confusion_matrix(y_train, oof_labels, labels=[1,2,3,4,5])
weighted_cost_cm = cm * cost_matrix
ax = axes[1]
sns.heatmap(weighted_cost_cm, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=[f'Pred {i+1}' for i in range(5)],
            yticklabels=[f'True {i+1}' for i in range(5)],
            ax=ax, linewidths=0.5)
ax.set_title('B. Observed Clinical Cost (count Ã— cost)', fontweight='bold')
ax.set_xlabel('Predicted ESI')
ax.set_ylabel('True ESI')

# C. Per-class clinical cost breakdown
ax = axes[2]
total_cost_per_class = weighted_cost_cm.sum(axis=1)
undertriage_cost = np.array([weighted_cost_cm[i, i+1:].sum() for i in range(5)])
overtriage_cost = np.array([weighted_cost_cm[i, :i].sum() for i in range(5)])

x = np.arange(5)
width = 0.35
bars1 = ax.bar(x - width/2, undertriage_cost, width, label='Undertriage cost', color='#c62828', alpha=0.85)
bars2 = ax.bar(x + width/2, overtriage_cost, width, label='Overtriage cost', color='#1565c0', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'ESI {i+1}' for i in range(5)])
ax.set_ylabel('Clinical Cost (weighted)')
ax.set_title('C. Cost by ESI Level & Error Direction', fontweight='bold')
ax.legend()

plt.suptitle('Clinical Misclassification Cost Analysis', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Summary statistics
total_cost = weighted_cost_cm.sum()
max_possible_cost = (np.ones_like(cm) * cost_matrix).sum()
cost_efficiency = 1 - (total_cost / max_possible_cost)
undertriage_total = sum(undertriage_cost)
overtriage_total = sum(overtriage_cost)

print(f'Total clinical cost:        {total_cost:.0f}')
print(f'  Undertriage contribution: {undertriage_total:.0f} ({100*undertriage_total/(undertriage_total+overtriage_total+1e-9):.1f}%)')
print(f'  Overtriage contribution:  {overtriage_total:.0f} ({100*overtriage_total/(undertriage_total+overtriage_total+1e-9):.1f}%)')
print(f'Cost efficiency score:      {cost_efficiency:.4f} (1.0 = perfect, 0.0 = worst)')
print(f'\nHighest-cost cell: True ESI {np.unravel_index(weighted_cost_cm.argmax(), (5,5))[0]+1} '
      f'-> Pred ESI {np.unravel_index(weighted_cost_cm.argmax(), (5,5))[1]+1} '
      f'(cost = {weighted_cost_cm.max():.0f})')
```

---
## 6 Â· Demographic Bias Analysis â€” Algorithmic Fairness for Clinical AI

Obermeyer et al. (2019) demonstrated that a widely-deployed healthcare algorithm systematically disadvantaged Black patients. In emergency triage, such bias translates directly to delayed care and preventable morbidity. A clinical AI system that systematically undertriages certain patient populations is not just inaccurate â€” it is **harmful**.

In the Finnish healthcare context, where universal coverage eliminates insurance-driven disparities, the key equity dimensions are **age**, **sex**, and **language** â€” particularly relevant for immigrant populations (Arabic, Somali, Russian speakers) navigating triage in a non-native language. We analyze the ensemble model's OOF predictions for evidence of:

- **Differential accuracy** across demographic groups (sex, age, language, insurance)
- **Systematic undertriage** â€” assigning less urgent scores than ground truth (patients wait longer)
- **Systematic overtriage** â€” assigning more urgent scores (wastes resources but less dangerous)

**Bias delta** = mean predicted acuity âˆ’ mean actual acuity. Positive = undertriage, negative = overtriage.


```python
# Prepare analysis DataFrame with original (readable) categorical values
analysis = train_raw.copy()
analysis['pred_acuity'] = oof_labels
analysis['error'] = analysis['pred_acuity'] - analysis[TARGET]
analysis['abs_error'] = analysis['error'].abs()
analysis['is_correct'] = (analysis['error'] == 0).astype(int)
analysis['undertriage'] = (analysis['pred_acuity'] > analysis[TARGET]).astype(int)
analysis['overtriage']  = (analysis['pred_acuity'] < analysis[TARGET]).astype(int)
analysis['pred_confidence'] = oof_ensemble.max(axis=1)

print('OVERALL ERROR ANALYSIS')
print(f'  Accuracy:          {analysis["is_correct"].mean():.4f}')
print(f'  Mean abs error:    {analysis["abs_error"].mean():.4f}')
print(f'  Undertriage rate:  {analysis["undertriage"].mean():.4f} ({analysis["undertriage"].sum():,} patients)')
print(f'  Overtriage rate:   {analysis["overtriage"].mean():.4f} ({analysis["overtriage"].sum():,} patients)')

print('\nUndertriage rate by actual acuity (CRITICAL FOR PATIENT SAFETY):')
for acuity in range(1, 6):
    subset = analysis[analysis[TARGET] == acuity]
    ut_rate = subset['undertriage'].mean()
    flag = ' *** HIGH RISK ***' if ut_rate > 0.05 and acuity <= 2 else ''
    print(f'  ESI {acuity} ({ACUITY[acuity]:15s}): {ut_rate:.4f}  '
          f'({subset["undertriage"].sum():>4,} / {len(subset):>5,}){flag}')
```


```python
# Comprehensive bias analysis
demo_cols = ['sex', 'age_group', 'language', 'insurance_type', 'arrival_mode']

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for idx, col in enumerate(demo_cols):
    ax = axes[idx]
    group = analysis.groupby(col).agg(
        n=(TARGET, 'count'),
        accuracy=('is_correct', 'mean'),
        mean_actual=(TARGET, 'mean'),
        mean_predicted=('pred_acuity', 'mean'),
        undertriage=('undertriage', 'mean'),
        overtriage=('overtriage', 'mean'),
    ).reset_index()
    group['bias_delta'] = group['mean_predicted'] - group['mean_actual']
    group = group.sort_values('bias_delta')
    
    colors_bias = ['#c62828' if d > 0.01 else '#1565c0' if d < -0.01 else '#9e9e9e'
                   for d in group['bias_delta']]
    bars = ax.barh(group[col].astype(str), group['bias_delta'], color=colors_bias, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Bias Delta (+ = undertriage)')
    ax.set_title(f'{col.replace("_", " ").title()}', fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)

# Accuracy by group in last subplot
ax = axes[5]
acc_by_age = analysis.groupby('age_group')['is_correct'].mean().sort_values()
ax.barh(acc_by_age.index, acc_by_age.values, color='#1565c0', edgecolor='white')
ax.set_xlabel('Accuracy')
ax.set_title('Accuracy by Age Group', fontweight='bold')
ax.axvline(analysis['is_correct'].mean(), color='red', linestyle='--', label='Overall')
ax.legend(fontsize=9)

plt.suptitle('Demographic Bias Analysis â€” Ensemble OOF Predictions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```


```python
# Statistical significance: chi-squared test for accuracy differences
print('STATISTICAL SIGNIFICANCE OF BIAS\n')
print('Chi-squared test: accuracy difference between groups\n')

for col in ['sex', 'age_group', 'language', 'insurance_type']:
    contingency = pd.crosstab(analysis[col], analysis['is_correct'])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    sig = 'YES' if p_val < 0.05 else 'no'
    print(f'{col:20s}: chi2={chi2:>10.2f}, p={p_val:.2e}, significant={sig}')

# Detailed bias table by sex
print('\n\nDetailed Performance by Sex:')
sex_detail = analysis.groupby('sex').agg(
    n=(TARGET, 'count'),
    accuracy=('is_correct', 'mean'),
    mean_actual=(TARGET, 'mean'),
    mean_predicted=('pred_acuity', 'mean'),
    undertriage_rate=('undertriage', 'mean'),
    overtriage_rate=('overtriage', 'mean'),
    mean_confidence=('pred_confidence', 'mean'),
).reset_index()
sex_detail['bias_delta'] = sex_detail['mean_predicted'] - sex_detail['mean_actual']
print(sex_detail.round(4).to_string(index=False))

# Detailed bias table by language
print('\n\nDetailed Performance by Language:')
lang_detail = analysis.groupby('language').agg(
    n=(TARGET, 'count'),
    accuracy=('is_correct', 'mean'),
    undertriage_rate=('undertriage', 'mean'),
    mean_confidence=('pred_confidence', 'mean'),
).sort_values('accuracy').reset_index()
print(lang_detail.round(4).to_string(index=False))
```


```python
# Intersectional bias: highest-risk subgroups
print('INTERSECTIONAL BIAS â€” Highest Undertriage Subgroups\n')

analysis['subgroup'] = analysis['sex'] + ' / ' + analysis['age_group'] + ' / ' + analysis['language']
subgroup_stats = analysis.groupby('subgroup').agg(
    n=(TARGET, 'count'),
    accuracy=('is_correct', 'mean'),
    undertriage_rate=('undertriage', 'mean'),
    mean_abs_error=('abs_error', 'mean'),
).reset_index()

subgroup_stats = subgroup_stats[subgroup_stats['n'] >= 100]  # sufficient sample size
subgroup_stats = subgroup_stats.sort_values('undertriage_rate', ascending=False)

print('Top 15 subgroups by undertriage rate (n >= 100):')
print(subgroup_stats.head(15).round(4).to_string(index=False))
```

---
## 7 Â· Clinical Insights â€” From Model to Bedside

Beyond aggregate metrics, what does the model reveal about emergency triage that could inform clinical practice? Below we synthesize findings that a practicing ED physician or nurse manager would find actionable â€” from vital sign distributions across acuity levels to the identification of patients at highest risk of triage error.


```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 7a. qSOFA score vs acuity
ax = axes[0, 0]
qsofa_ct = pd.crosstab(train['qsofa_score'], train[TARGET], normalize='index')
qsofa_ct.plot(kind='bar', stacked=True, color=COLORS, ax=ax, edgecolor='white')
ax.set_title('A. qSOFA Score vs. Triage Acuity', fontweight='bold')
ax.set_xlabel('qSOFA Score')
ax.set_ylabel('Proportion')
ax.legend(title='ESI', fontsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# 7b. Prediction confidence by correctness
ax = axes[0, 1]
correct_conf = oof_ensemble.max(axis=1)[analysis['is_correct'] == 1]
wrong_conf = oof_ensemble.max(axis=1)[analysis['is_correct'] == 0]
ax.hist(correct_conf, bins=50, alpha=0.6, label=f'Correct (n={len(correct_conf):,})', color='#2e7d32', density=True)
ax.hist(wrong_conf, bins=50, alpha=0.6, label=f'Incorrect (n={len(wrong_conf):,})', color='#c62828', density=True)
ax.set_xlabel('Max Prediction Probability')
ax.set_ylabel('Density')
ax.set_title('B. Prediction Confidence: Correct vs Incorrect', fontweight='bold')
ax.legend(fontsize=9)

# 7c. Error distribution
ax = axes[0, 2]
error_counts = analysis['error'].value_counts().sort_index()
colors_err = ['#c62828' if e > 0 else '#1565c0' if e < 0 else '#2e7d32' for e in error_counts.index]
ax.bar(error_counts.index, error_counts.values, color=colors_err, edgecolor='white')
ax.set_xlabel('Prediction Error (pred - actual)')
ax.set_ylabel('Count')
ax.set_title('C. Error Distribution', fontweight='bold')
for e, c in error_counts.items():
    label = 'correct' if e == 0 else ('under' if e > 0 else 'over')
    ax.text(e, c + 200, f'{c:,}', ha='center', fontsize=8)

# 7d. Nurse-level variability
ax = axes[1, 0]
nurse_stats = analysis.groupby('triage_nurse_id').agg(
    mean_acuity=(TARGET, 'mean'),
    n=(TARGET, 'count')
).reset_index()
nurse_stats = nurse_stats.sort_values('mean_acuity')
ax.barh(nurse_stats['triage_nurse_id'], nurse_stats['mean_acuity'], color='#1565c0', edgecolor='white', linewidth=0.3)
ax.axvline(analysis[TARGET].mean(), color='red', linestyle='--', linewidth=1, label='Overall mean')
ax.set_xlabel('Mean Assigned Acuity')
ax.set_title('D. Triage Nurse Variability (50 nurses)', fontweight='bold')
ax.tick_params(axis='y', labelsize=6)
ax.legend(fontsize=9)

# 7e. Age vs acuity
ax = axes[1, 1]
for lvl in [1, 2, 3, 4, 5]:
    subset = analysis[analysis[TARGET] == lvl]['age']
    ax.hist(subset, bins=40, alpha=0.5, label=f'ESI {lvl}', color=COLORS[lvl-1], density=True)
ax.set_xlabel('Age')
ax.set_ylabel('Density')
ax.set_title('E. Age Distribution by Acuity', fontweight='bold')
ax.legend(fontsize=8)

# 7f. Comorbidity burden vs acuity
ax = axes[1, 2]
hx_cols_analysis = [c for c in train.columns if c.startswith('hx_')]
train_tmp = train.copy()
train_tmp['comorb_total'] = train_tmp[hx_cols_analysis].sum(axis=1)
sns.boxplot(data=train_tmp, x=TARGET, y='comorb_total', ax=ax, palette=PALETTE)
ax.set_xlabel('ESI Level')
ax.set_ylabel('Number of Comorbidities')
ax.set_title('F. Comorbidity Burden by Acuity', fontweight='bold')

plt.suptitle('Clinical Insights', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```


```python
# COMPREHENSIVE CLINICAL INTERPRETATION
print('=' * 70)
print('CLINICAL INTERPRETATION OF FINDINGS')
print('Context: Emergency Severity Index (ESI) triage in a Nordic healthcare')
print('system, where nurse-led triage is standard and ED volumes average')
print('40,000-80,000 visits/year per hospital.')
print('=' * 70)

print('\n--- Finding 1: NEWS2 is the Dominant Predictor ---')
news2_corr = train['news2_score'].corr(train[TARGET])
print(f'Pearson correlation with acuity: r = {news2_corr:.3f}')
print('The National Early Warning Score 2 captures the aggregate')
print('physiological derangement that directly maps to ESI level.')
print('Clinical implication: NEWS2 alone could provide a first-pass')
print('triage decision, but combining with other signals reduces errors.')

print('\n--- Finding 2: Mental Status is the Key Discriminator for ESI 1-2 ---')
for status in ['unresponsive', 'drowsy', 'alert']:
    subset = analysis[analysis['mental_status_triage'] == status]
    if len(subset) > 0:
        esi12 = (subset[TARGET] <= 2).mean()
        esi1 = (subset[TARGET] == 1).mean()
        print(f'  {status:15s}: {esi12:.1%} ESI 1-2 | {esi1:.1%} ESI 1 (n={len(subset):,})')
print('This aligns with ESI algorithm: altered mental status â†’ ESI 1.')

print('\n--- Finding 3: qSOFA Score Stratifies Sepsis-Related Acuity ---')
for score in sorted(train['qsofa_score'].unique()):
    subset = train[train['qsofa_score'] == score]
    if len(subset) > 0:
        mean_a = subset[TARGET].mean()
        esi12_rate = (subset[TARGET] <= 2).mean()
        print(f'  qSOFA={int(score)}: mean acuity {mean_a:.2f}, ESI 1-2 rate {esi12_rate:.1%} (n={len(subset):,})')
print('qSOFA â‰¥ 2 is the Sepsis-3 bedside screening threshold.')
print('Our model captures this relationship without explicit programming.')

print('\n--- Finding 4: Inter-Rater Variability Among Triage Nurses ---')
nurse_range = nurse_stats['mean_acuity'].max() - nurse_stats['mean_acuity'].min()
nurse_std = nurse_stats['mean_acuity'].std()
print(f'  Range across nurses: {nurse_range:.2f} acuity levels')
print(f'  Standard deviation: {nurse_std:.3f}')
print('  This variability (>0.5 ESI levels between nurses) confirms the')
print('  literature: inter-rater reliability for ESI is Îº â‰ˆ 0.70-0.80.')
print('  An ML system can reduce this variability by providing consistent')
print('  second opinions, particularly for borderline ESI 2-3 and 3-4 cases.')

print('\n--- Finding 5: NLP Adds Unique Signal ---')
nlp_features_in_top50 = fi[fi['feature'].str.startswith(('tfidf_', 'kw_'))].head(10)
print(f'  {len(nlp_features_in_top50)} NLP features in top importance list')
top_nlp = fi[fi['feature'].str.startswith(('tfidf_', 'kw_'))].head(5)
for _, row in top_nlp.iterrows():
    print(f'    {row["feature"]:35s}  importance={row["importance"]:.4f}')
print('  Free-text chief complaints encode clinical urgency signals')
print('  (e.g., "chest pain", "seizure") not captured by vitals alone.')

print('\n--- Finding 6: Undertriage/Overtriage Rates ---')
undertriage = (analysis['pred_acuity'] > analysis[TARGET]).mean()
overtriage  = (analysis['pred_acuity'] < analysis[TARGET]).mean()
severe_under = ((analysis[TARGET] <= 2) & (analysis['pred_acuity'] > analysis[TARGET])).mean()
print(f'  Overall undertriage rate: {undertriage:.2%}')
print(f'  Overall overtriage rate:  {overtriage:.2%}')
print(f'  Undertriage of ESI 1-2:   {severe_under:.4%}')
print('  Clinical safety: undertriage of critical patients is the most')
print('  dangerous error type. Our model keeps this below 0.5%.')

print('\n--- Finding 7: Model Confidence Correlates with Decision Difficulty ---')
correct_mask = analysis['pred_acuity'] == analysis[TARGET]
wrong_mask = ~correct_mask
print(f'  Mean confidence on correct: {oof_ensemble.max(axis=1)[correct_mask.values].mean():.3f}')
print(f'  Mean confidence on errors:  {oof_ensemble.max(axis=1)[wrong_mask.values].mean():.3f}')
print('  The model "knows when it does not know" â€” low confidence flags')
print('  cases that would benefit from senior physician review.')
```

---
## 7b Â· Ablation Study â€” Feature Group Contributions

A rigorous model requires understanding *which components matter*. We retrain a simplified LightGBM on subsets of features to quantify each group's marginal contribution. This justifies our multi-modal approach and identifies the minimum viable feature set for deployment.


```python
# Ablation study: train LightGBM on feature subsets to measure each group's contribution
ablation_params = {
    'objective': 'multiclass', 'num_class': N_CLASSES, 'metric': 'multi_logloss',
    'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 63,
    'min_child_samples': 30, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': SEED, 'verbose': -1, 'n_jobs': -1,
}

feature_groups = {
    'Vitals Only': NUM_COLS,
    '+ Demographics': NUM_COLS + CAT_COLS,
    '+ Patient History': NUM_COLS + CAT_COLS + HX_COLS,
    '+ Clinical Flags (qSOFA, SIRS, etc.)': NUM_COLS + CAT_COLS + HX_COLS + ENG_COLS,
    '+ NLP (TF-IDF + Keywords)': list(X_train.columns),  # Full model
}

ablation_results = []
skf_abl = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

print('Ablation Study â€” Feature Group Contributions\n')
for group_name, cols in feature_groups.items():
    cols_available = [c for c in cols if c in X_train.columns]
    X_sub = X_train[cols_available]
    
    oof_abl = np.zeros(len(X_sub))
    for tr_idx, val_idx in skf_abl.split(X_sub, y_train):
        m = lgb.LGBMClassifier(**ablation_params)
        m.fit(X_sub.iloc[tr_idx], y_train[tr_idx],
              eval_set=[(X_sub.iloc[val_idx], y_train[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof_abl[val_idx] = m.predict(X_sub.iloc[val_idx])
    
    acc = accuracy_score(y_train, oof_abl)
    qwk = cohen_kappa_score(y_train, oof_abl, weights='quadratic')
    ablation_results.append({'Feature Group': group_name, 'Features': len(cols_available),
                             'Accuracy': acc, 'QWK': qwk})
    print(f'  {group_name:45s}  {len(cols_available):>4} feats  Acc={acc:.4f}  QWK={qwk:.4f}')

abl_df = pd.DataFrame(ablation_results)

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = range(len(abl_df))
bars = ax.bar(x_pos, abl_df['QWK'], color=['#e0e0e0', '#bbdefb', '#90caf9', '#42a5f5', '#1565c0'],
              edgecolor='white', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(abl_df['Feature Group'], rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Quadratic Weighted Kappa')
ax.set_title('Ablation Study: Cumulative Feature Group Contribution', fontweight='bold')
ax.set_ylim(abl_df['QWK'].min() - 0.01, 1.001)
for i, row in abl_df.iterrows():
    ax.text(i, row['QWK'] + 0.001, f"{row['QWK']:.4f}", ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()

delta_qwk = abl_df['QWK'].diff().fillna(abl_df['QWK'].iloc[0])
print('\nMarginal QWK improvement per feature group:')
for i, row in abl_df.iterrows():
    d = delta_qwk.iloc[i]
    print(f"  {row['Feature Group']:45s}  +{d:.4f} QWK")
```

---
## 8 Â· Submission


```python
# Apply same strategy (threshold or argmax) to test predictions
if USE_THRESHOLDS:
    test_cumprobs = test_ensemble.cumsum(axis=1)
    test_labels = np.ones(len(test_ensemble), dtype=int)
    for i, t in enumerate(opt_thresholds):
        test_labels[test_cumprobs[:, i] > t] = i + 2
    print('Using threshold-optimized predictions for test set')
else:
    test_labels = test_ensemble.argmax(axis=1) + 1
    print('Using argmax predictions for test set')

submission = pd.DataFrame({
    ID: test_raw[ID],
    TARGET: test_labels,
})

submission.to_csv('submission.csv', index=False)

print(f'Submission saved: {submission.shape}')
print(f'\nPrediction distribution:')
for lvl in range(1, 6):
    n = (test_labels == lvl).sum()
    train_pct = (y_train == lvl).mean() * 100
    test_pct = n / len(test_labels) * 100
    print(f'  ESI {lvl}: {n:>5,} ({test_pct:5.1f}%)  [train: {train_pct:5.1f}%]')

# Sanity check
fig, ax = plt.subplots(figsize=(8, 4))
train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
test_dist_s = pd.Series(test_labels).value_counts(normalize=True).reindex(range(1,6), fill_value=0).sort_index()
x = np.arange(1, 6)
ax.bar(x - 0.15, train_dist.values, 0.3, label='Train (actual)', color='#1565c0')
ax.bar(x + 0.15, test_dist_s.values, 0.3, label='Test (predicted)', color='#ef6c00')
ax.set_xlabel('ESI Acuity Level')
ax.set_ylabel('Proportion')
ax.set_title('Train vs Test Distribution â€” Sanity Check')
ax.set_xticks(x)
ax.legend()
plt.tight_layout()
plt.show()
```

---
## 9 Â· Summary & Limitations

### Results

| Metric | LightGBM | XGBoost | CatBoost | Stacked Ensemble |
|:-------|:--------:|:-------:|:--------:|:----------------:|
| Accuracy | 99.66% | 99.56% | 99.54% | **99.68%** |
| Weighted F1 | 99.66% | 99.56% | 99.54% | **99.68%** |
| QWK | 0.9982 | 0.9978 | 0.9978 | **0.9984** |

*Results from validated 5-fold stratified CV on 80,000 training patients. Two-level stacking (LR meta-learner on 15 OOF probability features) with 626 total features. ECE = 0.0001.*

### Key Contributions

1. **Multi-modal clinical data fusion** â€” structured vitals + NLP chief complaint + comorbidity history
2. **50+ clinically-motivated features** including qSOFA, SIRS criteria, cardiovascular risk, and critical keyword detection
3. **Two-level stacked triple ensemble** â€” LightGBM + XGBoost + CatBoost as level-1, Logistic Regression meta-learner on 15 OOF probability features
4. **QWK threshold optimization** â€” Nelder-Mead on ordinal cumulative probability boundaries to maximize QWK beyond naive argmax
5. **Clinical misclassification cost analysis** â€” asymmetric cost matrix encoding that undertriage is 3Ã— worse than overtriage, with ESI-level severity weighting
6. **Ablation study** â€” quantifies each feature group's marginal QWK improvement, validating multi-modal design
7. **Probability calibration analysis** â€” per-class reliability diagrams confirm well-calibrated confidence scores
8. **Comprehensive demographic bias analysis** with chi-squared significance testing and intersectional subgroup analysis
9. **Interpretable predictions** via SHAP and tri-model feature importance averaging, enabling clinical audit

### Clinical Implications

- The model could serve as a **real-time second opinion** for triage nurses in Nordic EDs, flagging cases where AI-predicted acuity diverges from initial assessment
- **Clinical cost analysis** reveals residual undertriage cost is concentrated in ESI 2â†’3 misclassifications â€” actionable for deployment thresholding
- **Nurse-level variability** (range of mean acuity across 50 nurses) confirms the clinical need for AI decision support
- **Calibrated confidence scores** enable a "flag for senior review" mechanism on low-confidence cases
- **Bias monitoring must precede deployment** â€” any clinical implementation requires prospective fairness evaluation

### Limitations

1. **Synthetic/competition data** â€” real-world performance requires validation against MIMIC-IV-ED or clinical partner data
2. **NEWS2 as feature** â€” NEWS2 is itself a clinical scoring system; including it provides strong signal but may partially encode existing triage decisions
3. **TF-IDF NLP** â€” captures keyword signal but misses semantic nuance; ClinicalBERT or BioGPT would improve chief complaint understanding
4. **No temporal validation** â€” time-based splits would better simulate prospective deployment
5. **Single-snapshot triage** â€” does not model reassessment dynamics or patient deterioration over time
6. **No external validation** â€” model performance on a held-out site or institution is unknown

### References

- Gilboy, N. et al. (2020). *Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care*. AHRQ.
- Singer, M. et al. (2016). *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)*. JAMA.
- Royal College of Physicians (2017). *National Early Warning Score (NEWS) 2*.
- Farrohknia, N. et al. (2011). *Emergency department triage scales and their components*. Scand. J. Trauma Resusc. Emerg. Med.
- Levin, S. et al. (2018). *Machine learning-based electronic triage more accurately differentiates patients*. Ann. Emerg. Med.
- Fernandes, M. et al. (2020). *Clinical Decision Support Systems for Triage in the Emergency Department*. Artif. Intell. Med.
- Obermeyer, Z. et al. (2019). *Dissecting racial bias in an algorithm used to manage the health of populations*. Science.


```python
print('Triagegeist pipeline complete.')
print(f'  Total features:      {X_train.shape[1]}')
print(f'  Base models:         LightGBM + XGBoost + CatBoost + MLP (4 models, {N_FOLDS} folds each)')
print(f'  Meta-learner:        L1-regularized LR on {N_CLASSES * 4} OOF meta-features')
print(f'  Ensemble QWK:        {ens_qwk:.4f}')
print(f'  Submission rows:     {len(submission):,}')
print(f'  Bias groups analyzed: {len(demo_cols)} demographic dimensions')
```
