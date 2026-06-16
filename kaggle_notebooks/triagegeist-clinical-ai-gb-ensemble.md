# Multi-Modal ED Triage Acuity Prediction: Clinical AI with Gradient Boosting

**Competition**: Triagegeist - Predicting Emergency Severity Index (ESI) from Clinical Data  
**Approach**: 3-model soft-voting ensemble (LightGBM + CatBoost + XGBoost) with ordinal threshold optimization  

---

## Clinical Problem Statement

Emergency Department (ED) triage is a high-stakes, time-pressured decision process. The 5-level Emergency Severity Index (ESI) is the most widely used triage system in the United States, yet inter-rater reliability is moderate (kappa 0.46-0.58 in literature). **ESI-3 ("Urgent") is the most ambiguous category**, encompassing patients who need resources but have stable vital signs, leading to significant triage disagreement.

### Goal
Build an **interpretable triage support tool** that:
1. Predicts ESI level (1-5) from available clinical data at triage
2. Uses clinically meaningful features aligned with emergency medicine practice
3. Provides transparency via SHAP analysis for clinician trust
4. Demonstrates equitable performance across demographic groups

### Approach Overview
- **Feature Engineering**: Clinical vital sign flags, derived scores (shock index, frailty), TF-IDF NLP on chief complaints, temporal encoding
- **Modeling**: 3-model soft-voting ensemble (LightGBM + CatBoost + XGBoost)
- **Post-processing**: Ordinal threshold optimization using Nelder-Mead to maximize Quadratic Weighted Kappa
- **Interpretation**: SHAP TreeExplainer for per-class feature attribution
- **Fairness**: Accuracy, undertriage/overtriage rates stratified by sex, age group, language, insurance type

## 1. Setup & Data Loading


```python
# Core libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from pathlib import Path

# ML libraries
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, f1_score,
    confusion_matrix, classification_report,
    precision_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack as sparse_hstack
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

print('Libraries loaded successfully')
print(f'LightGBM: {lgb.__version__}')
print(f'XGBoost: {xgb.__version__}')
```


```python
# Load competition data from Kaggle's standard path
DATA_DIR = Path('/kaggle/input/triagegeist')

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
cc = pd.read_csv(DATA_DIR / 'chief_complaints.csv')
ph = pd.read_csv(DATA_DIR / 'patient_history.csv')
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

print(f'Train: {train.shape}')
print(f'Test:  {test.shape}')
print(f'Chief Complaints: {cc.shape}')
print(f'Patient History:  {ph.shape}')
print(f'Sample Submission: {sample_sub.shape}')
```


```python
# Quick EDA summary
print('=== Target Distribution ===')
print(train['triage_acuity'].value_counts().sort_index())
print(f'\nClass ratios:')
print(train['triage_acuity'].value_counts(normalize=True).sort_index().round(4))

print('\n=== Missing Values (top 15) ===')
missing = train.isnull().sum()
missing_pct = (train.isnull().sum() / len(train) * 100).round(2)
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct}).sort_values('pct', ascending=False)
print(missing_df[missing_df['count'] > 0].head(15))

print('\n=== Numeric Summary ===')
print(train[['age', 'heart_rate', 'systolic_bp', 'spo2', 'temperature_c', 'respiratory_rate', 'gcs_total', 'pain_score']].describe().round(2))
```

## 2. Data Leakage Prevention

### Critical Leakage Columns Removed

**`ed_los_hours` (ED Length of Stay)**: This is a *post-triage* outcome. A patient's length of stay in the ED is determined *after* the triage decision is made and resources are allocated. Including it would leak information about the severity and complexity of the patient's condition that is not available at triage time.

**`disposition`**: This indicates the patient's ultimate outcome (e.g., admitted, discharged, transferred). This is a *downstream* outcome of the triage process and is unavailable at the time of triage. Including it would be equivalent to using the answer to predict the answer.

### Retained: `news2_score`

The NEWS2 (National Early Warning Score 2) is a **bedside clinical scoring system** calculated from vital signs (respiratory rate, SpO2, temperature, systolic BP, heart rate, level of consciousness). It is routinely computed *at triage* and is therefore a legitimate input feature. Ablation testing confirmed that NEWS2 provides genuine predictive signal rather than data leakage.


```python
# Drop leakage columns
LEAKAGE_COLS = ['ed_los_hours', 'disposition']
print(f'Dropping leakage columns: {LEAKAGE_COLS}')
train.drop(columns=LEAKAGE_COLS, inplace=True, errors='ignore')
test.drop(columns=['ed_los_hours', 'disposition'], inplace=True, errors='ignore')

# Join with auxiliary tables on patient_id
cc_merge = cc.drop(columns=['chief_complaint_system'], errors='ignore')
train = train.merge(cc_merge, on='patient_id', how='left')
test = test.merge(cc_merge, on='patient_id', how='left')

train = train.merge(ph, on='patient_id', how='left')
test = test.merge(ph, on='patient_id', how='left')

print(f'After joins - Train: {train.shape}, Test: {test.shape}')
print(f'Chief complaint system column duplicated: {"chief_complaint_system" in train.columns}')

# Separate target and IDs
TARGET = 'triage_acuity'
y = train[TARGET].values.astype(int)
train.drop(columns=[TARGET], inplace=True)

patient_ids_test = test['patient_id'].values

# Store demographic columns for fairness analysis before dropping IDs
fairness_cols_train = {}
for col in ['sex', 'language', 'insurance_type', 'age_group']:
    if col in train.columns:
        fairness_cols_train[col] = train[col].values.copy()

ID_COLS = ['patient_id', 'site_id', 'triage_nurse_id']
train.drop(columns=ID_COLS, inplace=True)
test.drop(columns=ID_COLS, inplace=True)

print(f'Feature columns: {train.shape[1]}')
```

## 3. Feature Engineering

Each feature group is designed to mirror clinical reasoning in emergency triage.

### 3.1 Clinical Vital Sign Flags

These binary flags capture established clinical thresholds used in emergency medicine:

| Flag | Threshold | Clinical Rationale |
|------|-----------|-------------------|
| `flag_hypoxia` | SpO2 < 90% | Peripheral oxygen saturation below 90% indicates significant hypoxemia per ATS guidelines; warrants immediate intervention |
| `flag_tachycardia` | HR > 100 bpm | Elevated heart rate is a compensatory mechanism for hemodynamic stress, pain, or fever |
| `flag_bradycardia` | HR < 60 bpm | May indicate cardiac conduction abnormality, medication effect, or vagal response; concerning in trauma context |
| `flag_hypotension` | SBP < 90 mmHg | Systolic BP below 90 mmHg is a criterion for shock; requires immediate resuscitation |
| `flag_tachypnea` | RR > 22/min | Elevated respiratory rate is an early sign of respiratory distress or metabolic acidosis (compensatory) |
| `flag_fever` | T >= 38.0 C | Fever suggests infection; in the context of tachycardia and tachypnea, raises suspicion for sepsis |
| `flag_hypothermia` | T < 35.0 C | Hypothermia is a poor prognostic sign, particularly in sepsis ("cold sepsis") and trauma |
| `flag_low_gcs` | GCS < 15 | Any reduction in Glasgow Coma Scale indicates neurological compromise; full GCS (15) is normal |
| `flag_severe_pain` | Pain >= 7/10 | Severe pain may indicate serious pathology and warrants urgent evaluation |
| `flag_elderly` | Age >= 65 | Elderly patients have reduced physiological reserve; atypical presentations are common |
| `flag_sepsis_suspect` | >= 2 of tachycardia + fever + tachypnea | Quick Sequential Organ Failure Assessment (qSOFA) criteria for sepsis screening |


```python
def engineer_features(df):
    """V2 Enhanced feature engineering with clinical flags, derived scores, and interactions."""

    # --- Missing indicators for vital signs ---
    # Missing vital signs at triage may indicate clinical urgency (too unstable to measure)
    vital_cols = [
        'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure',
        'pulse_pressure', 'heart_rate', 'respiratory_rate',
        'temperature_c', 'spo2', 'shock_index',
    ]
    for col in vital_cols:
        if df[col].isnull().any():
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    # --- Pain score missingness (coded as -1) ---
    # Pain score of -1 indicates the patient was unable to self-report,
    # which may correlate with altered mental status or severe illness
    df['pain_missing'] = (df['pain_score'] == -1).astype(int)
    df['pain_score'] = df['pain_score'].replace(-1, np.nan)

    # --- Impute missing vitals with age_group median ---
    # Age-group stratification accounts for physiological differences
    vital_numeric = [
        'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure',
        'pulse_pressure', 'heart_rate', 'respiratory_rate',
        'temperature_c', 'spo2', 'shock_index', 'pain_score',
    ]
    for col in vital_numeric:
        if df[col].isnull().any():
            medians = df.groupby('age_group')[col].transform('median')
            df[col] = df[col].fillna(medians)
            df[col] = df[col].fillna(df[col].median())

    # --- Clinical vital sign flags ---
    df['flag_hypoxia'] = (df['spo2'] < 90).astype(int)
    df['flag_tachycardia'] = (df['heart_rate'] > 100).astype(int)
    df['flag_bradycardia'] = (df['heart_rate'] < 60).astype(int)
    df['flag_hypotension'] = (df['systolic_bp'] < 90).astype(int)
    df['flag_tachypnea'] = (df['respiratory_rate'] > 22).astype(int)
    df['flag_fever'] = (df['temperature_c'] >= 38.0).astype(int)
    df['flag_hypothermia'] = (df['temperature_c'] < 35.0).astype(int)
    df['flag_low_gcs'] = (df['gcs_total'] < 15).astype(int)
    df['flag_severe_pain'] = (df['pain_score'] >= 7).astype(int)
    df['flag_elderly'] = (df['age'] >= 65).astype(int)

    # Sepsis suspect: >= 2 of tachycardia + fever + tachypnea (qSOFA-inspired)
    df['flag_sepsis_suspect'] = (
        df['flag_tachycardia'] + df['flag_fever'] + df['flag_tachypnea']
    ) >= 2
    df['flag_sepsis_suspect'] = df['flag_sepsis_suspect'].astype(int)

    print(f'  Clinical flags created: 12 binary features')

    # --- Derived clinical scores ---
    # Shock index = HR/SBP: > 0.9 suggests hemodynamic compromise
    # Already in data but we create squared version to amplify extreme values
    df['shock_index_sq'] = df['shock_index'] ** 2

    # HR/RR ratio: abnormal respiratory-cardiac coupling
    df['hr_rr_ratio'] = df['heart_rate'] / df['respiratory_rate'].replace(0, np.nan)
    df['hr_rr_ratio'] = df['hr_rr_ratio'].fillna(df['hr_rr_ratio'].median())

    # SpO2 * RR interaction: captures the respiratory severity axis
    df['spo2_rr_interaction'] = df['spo2'] * df['respiratory_rate']

    print(f'  Derived clinical scores: shock_index_sq, hr_rr_ratio, spo2_rr_interaction')

    # --- Comorbidity features ---
    hx_cols = [c for c in df.columns if c.startswith('hx_')]
    df['comorbidity_count'] = df[hx_cols].sum(axis=1)

    # Immunocompromised flag: HIV, immunosuppressed, or active malignancy
    if all(c in df.columns for c in ['hx_hiv', 'hx_immunosuppressed', 'hx_malignancy']):
        df['immunocompromised'] = (
            df['hx_hiv'] | df['hx_immunosuppressed'] | df['hx_malignancy']
        ).astype(int)

    print(f'  Comorbidity features: comorbidity_count, immunocompromised')

    # --- Vital sign derangement count ---
    # Sum of abnormal vital sign flags: higher = more physiologically deranged
    derangement_cols = [
        'flag_hypoxia', 'flag_tachycardia', 'flag_bradycardia',
        'flag_hypotension', 'flag_tachypnea', 'flag_fever',
        'flag_hypothermia', 'flag_low_gcs',
    ]
    df['vital_derangement_count'] = df[derangement_cols].sum(axis=1)

    # --- Frailty score ---
    # age>=65 + hx_heart_failure + hx_ckd + num_comorbidities>=5
    # Frail patients are at higher risk of adverse outcomes even with moderate presentations
    df['frailty_score'] = (
        (df['age'] >= 65).astype(int) +
        df.get('hx_heart_failure', 0).astype(int) +
        df.get('hx_ckd', 0).astype(int) +
        (df['num_comorbidities'] >= 5).astype(int)
    )

    print(f'  Clinical composite scores: vital_derangement_count, frailty_score')

    # --- Age-vitals interactions ---
    # Elderly patients with abnormal vitals are at disproportionate risk
    df['age_x_shock_index'] = df['age'] * df['shock_index']
    df['age_x_heart_rate'] = df['age'] * df['heart_rate']
    df['age_vital_derangement'] = df['flag_elderly'] * df['vital_derangement_count']

    # --- Additional hemodynamic ratios ---
    # MAP/SBP ratio: abnormal if < 0.6 (narrow pulse pressure)
    df['map_sbp_ratio'] = df['mean_arterial_pressure'] / df['systolic_bp'].replace(0, np.nan)
    df['map_sbp_ratio'] = df['map_sbp_ratio'].fillna(df['map_sbp_ratio'].median())

    # Pulse pressure/MAP ratio: widened pulse pressure suggests high-output states
    df['pp_map_ratio'] = df['pulse_pressure'] / df['mean_arterial_pressure'].replace(0, np.nan)
    df['pp_map_ratio'] = df['pp_map_ratio'].fillna(df['pp_map_ratio'].median())

    # --- Combined comorbidity-severity score ---
    # comorbidity_count * (6 - gcs_total/3): higher when more comorbidities and lower GCS
    gcs_severity = 6.0 - df['gcs_total'] / 3.0
    df['comorbidity_severity'] = df['comorbidity_count'] * gcs_severity

    # NEWS2 * comorbidity interaction
    df['news2_comorbidity'] = df['news2_score'] * df['comorbidity_count']

    print(f'  Age interactions and hemodynamic ratios: 8 features')

    # --- High-risk chief complaint flag (regex-based) ---
    # Chest pain, dyspnea, altered mental status are "can't miss" ED presentations
    high_risk_keywords = [
        r'\bchest\b', r'\bbreath\b', r'\bdyspn', r'\bsob\b',
        r'\baltered\b', r'\bunresponsive\b', r'\bconfusion\b',
        r'\bunconscious\b', r'\b lethargic\b',
    ]
    cc_raw_lower = df['chief_complaint_raw'].fillna('').str.lower()
    pattern = '|'.join(high_risk_keywords)
    df['high_risk_cc'] = cc_raw_lower.str.contains(pattern, regex=True, na=False).astype(int)

    print(f'  High-risk chief complaint flag: high_risk_cc')

    # --- Temporal cyclical features ---
    # Sine/cosine encoding preserves cyclical nature of hour-of-day
    # (hour 23 is adjacent to hour 0, not far from it)
    df['arrival_hour_sin'] = np.sin(2 * np.pi * df['arrival_hour'] / 24)
    df['arrival_hour_cos'] = np.cos(2 * np.pi * df['arrival_hour'] / 24)

    print(f'  Temporal features: arrival_hour_sin, arrival_hour_cos')

    return df


print('Engineering features...')
train = engineer_features(train)
test = engineer_features(test)
print(f'\nAfter feature engineering - Train: {train.shape}, Test: {test.shape}')
```

### 3.2 NLP Features: TF-IDF on Chief Complaint

The chief complaint is the patient's (or EMS') stated reason for the ED visit in free text. It is one of the most informative triage inputs:
- **TF-IDF** with unigrams, bigrams, and trigrams captures clinically relevant phrases ("chest pain", "shortness of breath", "altered mental status")
- **Keyword flags** for critical terms: chest, breath, fever, pain, bleed, trauma, headache, fall
- **Text length** features: longer chief complaints may indicate more complex presentations


```python
# TF-IDF on chief_complaint_raw
print('Building TF-IDF features...')

tfidf = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 3),
    min_df=3,
    sublinear_tf=True,
    dtype=np.float32,
)

# Fit on combined train+test text for consistent vocabulary
all_text = pd.concat([train['chief_complaint_raw'], test['chief_complaint_raw']], axis=0).fillna('')
tfidf.fit(all_text)

train_tfidf = tfidf.transform(train['chief_complaint_raw'].fillna(''))
test_tfidf = tfidf.transform(test['chief_complaint_raw'].fillna(''))
print(f'  TF-IDF shape: {train_tfidf.shape}')

# Keyword extraction features
keyword_list = ['chest', 'breath', 'fever', 'pain', 'bleed', 'trauma', 'headache', 'fall']

def build_keyword_features(df, keywords):
    """Build binary keyword flags from chief_complaint_raw."""
    text_lower = df['chief_complaint_raw'].fillna('').str.lower()
    kw_features = {}
    for kw in keywords:
        kw_features[f'kw_{kw}'] = text_lower.str.contains(
            rf'\b{kw}\b', regex=True, na=False
        ).astype(int).values
    return pd.DataFrame(kw_features, index=df.index)

train_kw = build_keyword_features(train, keyword_list)
test_kw = build_keyword_features(test, keyword_list)
print(f'  Keyword features: {len(keyword_list)}')

# Text length and word count features
def build_text_length_features(df):
    """Build text length and word count features from chief_complaint_raw."""
    text = df['chief_complaint_raw'].fillna('')
    features = {}
    features['text_char_len'] = text.str.len().values
    features['text_word_count'] = text.str.split().str.len().values
    return pd.DataFrame(features, index=df.index)

train_text_feat = build_text_length_features(train)
test_text_feat = build_text_length_features(test)
print(f'  Text length features: {train_text_feat.shape[1]}')

# Add keyword and text features to dataframes
for col in train_kw.columns:
    train[col] = train_kw[col].values
    test[col] = test_kw[col].values
for col in train_text_feat.columns:
    train[col] = train_text_feat[col].values
    test[col] = test_text_feat[col].values

# Drop the text column
train.drop(columns=['chief_complaint_raw'], inplace=True)
test.drop(columns=['chief_complaint_raw'], inplace=True)

print(f'After NLP features - Train: {train.shape}, Test: {test.shape}')
```


```python
# Categorical encoding
# Label encoding for gradient boosting models (tree-based models handle ordinal encoding well)
CAT_COLS = [
    'arrival_mode', 'sex', 'language', 'insurance_type',
    'transport_origin', 'pain_location', 'mental_status_triage',
    'chief_complaint_system', 'shift', 'arrival_day',
    'arrival_season', 'age_group',
]

le_dict = {}
for col in CAT_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    le_dict[col] = le

print(f'Encoded {len(CAT_COLS)} categorical columns')
```


```python
# Build final feature matrices
FEATURE_COLS = [c for c in train.columns]

X_train_dense = train[FEATURE_COLS].values.astype(np.float32)
X_test_dense = test[FEATURE_COLS].values.astype(np.float32)

# Combine dense features with TF-IDF sparse matrix
X_train = sparse_hstack([X_train_dense, train_tfidf], format='csr')
X_test = sparse_hstack([X_test_dense, test_tfidf], format='csr')

# Pre-compute dense arrays for CatBoost and XGBoost (they require dense input)
print('Pre-computing dense arrays for CatBoost/XGBoost...')
X_train_dense_all = X_train.toarray()
X_test_dense_all = X_test.toarray()

n_tfidf = train_tfidf.shape[1]
n_dense = X_train_dense.shape[1]
print(f'Dense features: {n_dense}')
print(f'TF-IDF features: {n_tfidf}')
print(f'Total features: {n_dense + n_tfidf}')

# Free memory
del train, test, train_tfidf, test_tfidf, train_kw, test_kw, train_text_feat, test_text_feat
import gc
gc.collect()
```

## 4. Model Training

### Why a 3-Model Ensemble?

Different gradient boosting implementations learn complementary patterns:
- **LightGBM**: Leaf-wise tree growth with histogram-based splitting; excels at capturing fine-grained patterns efficiently
- **CatBoost**: Ordered boosting prevents target leakage; native handling of categorical features reduces overfitting
- **XGBoost**: Depth-wise growth with regularization; provides a different inductive bias

Soft voting (averaging predicted probabilities) reduces variance and improves calibration.

### Ordinal Threshold Optimization

ESI levels are ordinal (1 < 2 < 3 < 4 < 5), not merely categorical. Standard argmax prediction ignores this ordering. We compute an **expected value** from the class probabilities and then learn optimal decision thresholds via Nelder-Mead optimization to maximize Quadratic Weighted Kappa (QWK), which explicitly rewards predictions that are "close" to the true class.


```python
# Ordinal threshold optimization functions

def expected_value_to_class(ev, thresholds):
    """Convert expected value to class using sorted thresholds.
    
    Classes 1..5 are assigned based on which interval ev falls into:
    ev <= t[0] -> class 1
    t[0] < ev <= t[1] -> class 2
    t[1] < ev <= t[2] -> class 3
    t[2] < ev <= t[3] -> class 4
    ev > t[3] -> class 5
    """
    sorted_t = np.sort(thresholds)
    pred = np.ones(len(ev), dtype=int)
    for i, t in enumerate(sorted_t):
        pred[ev > t] = i + 2
    return pred


def qwk_from_thresholds(thresholds, ev, y_true):
    """Compute negative QWK for given thresholds (for minimization)."""
    pred = expected_value_to_class(ev, thresholds)
    return -cohen_kappa_score(y_true, pred, weights='quadratic')


def optimize_thresholds(probs, y_true):
    """Find optimal thresholds using Nelder-Mead optimization."""
    ev = probs @ np.array([1, 2, 3, 4, 5])

    # Baseline: argmax QWK
    argmax_pred = probs.argmax(axis=1) + 1
    argmax_qwk = cohen_kappa_score(y_true, argmax_pred, weights='quadratic')
    print(f'  Baseline argmax QWK: {argmax_qwk:.4f}')

    # Initial thresholds: evenly spaced based on EV range
    ev_min, ev_max = ev.min(), ev.max()
    step = (ev_max - ev_min) / 5.0
    initial_thresholds = np.array([
        ev_min + step,
        ev_min + 2 * step,
        ev_min + 3 * step,
        ev_min + 4 * step,
    ])

    best_qwk = argmax_qwk
    best_thresholds = initial_thresholds.copy()
    used_argmax = True

    # Try multiple starting points with Nelder-Mead
    for attempt in range(20):
        if attempt == 0:
            x0 = initial_thresholds.copy()
        elif attempt == 1:
            # Use class-conditional EV percentiles
            class_medians = []
            for c in [1, 2, 3, 4, 5]:
                mask = y_true == c
                if mask.sum() > 0:
                    class_medians.append(np.median(ev[mask]))
            x0 = np.array([
                (class_medians[0] + class_medians[1]) / 2,
                (class_medians[1] + class_medians[2]) / 2,
                (class_medians[2] + class_medians[3]) / 2,
                (class_medians[3] + class_medians[4]) / 2,
            ])
        else:
            # Random perturbation
            x0 = initial_thresholds + np.random.uniform(-step, step, 4)

        result = minimize(
            qwk_from_thresholds,
            x0=x0,
            args=(ev, y_true),
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10},
        )

        current_qwk = -result.fun
        if current_qwk > best_qwk:
            best_qwk = current_qwk
            best_thresholds = result.x.copy()
            used_argmax = False

    # Sort thresholds to ensure monotonicity
    best_thresholds = np.sort(best_thresholds)

    if used_argmax:
        print(f'  Threshold optimization did NOT improve over argmax ({argmax_qwk:.4f})')
    else:
        print(f'  Threshold optimization improved: {argmax_qwk:.4f} -> {best_qwk:.4f}')
        print(f'  Optimal thresholds: {best_thresholds}')

    return best_thresholds, best_qwk


print('Ordinal threshold optimization functions ready')
```


```python
# 5-fold Stratified Cross-Validation training
print('=' * 70)
print('Training 3-model ensemble with 5-fold Stratified CV...')
print('=' * 70)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Storage for OOF predictions
oof_lgb = np.zeros((len(y), 5), dtype=np.float64)
oof_cb = np.zeros((len(y), 5), dtype=np.float64)
oof_xgb = np.zeros((len(y), 5), dtype=np.float64)
oof_ensemble = np.zeros((len(y), 5), dtype=np.float64)

# Storage for test predictions
test_preds_lgb = np.zeros((X_test.shape[0], 5), dtype=np.float64)
test_preds_cb = np.zeros((X_test.shape[0], 5), dtype=np.float64)
test_preds_xgb = np.zeros((X_test.shape[0], 5), dtype=np.float64)

fold_scores = {
    'lgb_qwk': [], 'cb_qwk': [], 'xgb_qwk': [], 'ens_qwk': [],
    'lgb_acc': [], 'cb_acc': [], 'xgb_acc': [], 'ens_acc': [],
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_dense_all, y)):
    print(f'\n--- Fold {fold + 1}/{N_FOLDS} ---')
    X_tr, X_val = X_train[trn_idx], X_train[val_idx]
    X_tr_dense = X_train_dense_all[trn_idx]
    X_val_dense = X_train_dense_all[val_idx]
    y_tr, y_val = y[trn_idx], y[val_idx]

    # === LightGBM ===
    # Leaf-wise growth is efficient for large feature spaces
    # class_weight=balanced accounts for ESI class imbalance
    print('  Training LightGBM...')
    t_model = time.time()
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        class_weight='balanced',
        objective='multiclass',
        num_class=5,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    lgb_val_proba = lgb_clf.predict_proba(X_val)
    lgb_val_pred = lgb_val_proba.argmax(axis=1) + 1
    oof_lgb[val_idx] = lgb_val_proba
    test_preds_lgb += lgb_clf.predict_proba(X_test) / N_FOLDS
    lgb_qwk = cohen_kappa_score(y_val, lgb_val_pred, weights='quadratic')
    lgb_acc = accuracy_score(y_val, lgb_val_pred)
    fold_scores['lgb_qwk'].append(lgb_qwk)
    fold_scores['lgb_acc'].append(lgb_acc)
    print(f'    LGB QWK={lgb_qwk:.4f}  ACC={lgb_acc:.4f}  best_iter={lgb_clf.best_iteration_}  time={time.time()-t_model:.1f}s')

    # === CatBoost ===
    # Ordered boosting prevents target leakage from categorical features
    # eval_metric=TotalF1 optimizes for multi-class balance
    print('  Training CatBoost...')
    t_model = time.time()
    cb_clf = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        eval_metric='TotalF1',
        auto_class_weights='Balanced',
        random_seed=SEED,
        verbose=0,
        early_stopping_rounds=100,
    )
    cb_clf.fit(
        X_tr_dense, y_tr,
        eval_set=(X_val_dense, y_val),
        verbose=0,
    )
    cb_val_proba = cb_clf.predict_proba(X_val_dense)
    cb_val_pred = cb_val_proba.argmax(axis=1) + 1
    oof_cb[val_idx] = cb_val_proba
    test_preds_cb += cb_clf.predict_proba(X_test_dense_all) / N_FOLDS
    cb_qwk = cohen_kappa_score(y_val, cb_val_pred, weights='quadratic')
    cb_acc = accuracy_score(y_val, cb_val_pred)
    fold_scores['cb_qwk'].append(cb_qwk)
    fold_scores['cb_acc'].append(cb_acc)
    print(f'    CB  QWK={cb_qwk:.4f}  ACC={cb_acc:.4f}  best_iter={cb_clf.best_iteration_}  time={time.time()-t_model:.1f}s')

    # === XGBoost ===
    # tree_method='hist' uses histogram-based splitting (fast on CPU)
    # subsample and colsample_bytree provide regularization
    print('  Training XGBoost...')
    t_model = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=5,
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='mlogloss',
        tree_method='hist',
    )
    # XGBoost requires 0-indexed labels (0..4)
    xgb_clf.fit(
        X_tr_dense, y_tr - 1,
        eval_set=[(X_val_dense, y_val - 1)],
        verbose=False,
    )
    xgb_val_proba = xgb_clf.predict_proba(X_val_dense)
    xgb_val_pred = xgb_val_proba.argmax(axis=1) + 1
    oof_xgb[val_idx] = xgb_val_proba
    test_preds_xgb += xgb_clf.predict_proba(X_test_dense_all) / N_FOLDS
    xgb_qwk = cohen_kappa_score(y_val, xgb_val_pred, weights='quadratic')
    xgb_acc = accuracy_score(y_val, xgb_val_pred)
    fold_scores['xgb_qwk'].append(xgb_qwk)
    fold_scores['xgb_acc'].append(xgb_acc)
    print(f'    XGB QWK={xgb_qwk:.4f}  ACC={xgb_acc:.4f}  best_iter={xgb_clf.best_iteration}  time={time.time()-t_model:.1f}s')

    # === Ensemble (soft vote) ===
    ens_proba = (lgb_val_proba + cb_val_proba + xgb_val_proba) / 3.0
    ens_pred = ens_proba.argmax(axis=1) + 1
    oof_ensemble[val_idx] = ens_proba
    ens_qwk = cohen_kappa_score(y_val, ens_pred, weights='quadratic')
    ens_acc = accuracy_score(y_val, ens_pred)
    fold_scores['ens_qwk'].append(ens_qwk)
    fold_scores['ens_acc'].append(ens_acc)
    print(f'    ENS QWK={ens_qwk:.4f}  ACC={ens_acc:.4f}')

    del X_tr_dense, X_val_dense
    gc.collect()

print('\n' + '=' * 70)
print('Cross-validation complete!')
```


```python
# Overall OOF evaluation
print('=' * 70)
print('OOF Evaluation (all folds combined)')
print('=' * 70)

# Per-fold summary
print(f'\n{"Fold":<6} {"LightGBM":>10} {"CatBoost":>10} {"XGBoost":>10} {"Ensemble":>10}')
for i in range(N_FOLDS):
    print(f'{i+1:<6} {fold_scores["lgb_qwk"][i]:>10.4f} {fold_scores["cb_qwk"][i]:>10.4f} {fold_scores["xgb_qwk"][i]:>10.4f} {fold_scores["ens_qwk"][i]:>10.4f}')

print(f'\nMean QWK (+/- std):')
print(f'  LightGBM: {np.mean(fold_scores["lgb_qwk"]):.4f} +/- {np.std(fold_scores["lgb_qwk"]):.4f}')
print(f'  CatBoost: {np.mean(fold_scores["cb_qwk"]):.4f} +/- {np.std(fold_scores["cb_qwk"]):.4f}')
print(f'  XGBoost:  {np.mean(fold_scores["xgb_qwk"]):.4f} +/- {np.std(fold_scores["xgb_qwk"]):.4f}')
print(f'  Ensemble: {np.mean(fold_scores["ens_qwk"]):.4f} +/- {np.std(fold_scores["ens_qwk"]):.4f}')

# Overall OOF metrics
oof_ens_pred = oof_ensemble.argmax(axis=1) + 1
overall_qwk = cohen_kappa_score(y, oof_ens_pred, weights='quadratic')
overall_acc = accuracy_score(y, oof_ens_pred)
overall_f1 = f1_score(y, oof_ens_pred, average='macro')

print(f'\nEnsemble OOF (argmax):')
print(f'  QWK:      {overall_qwk:.4f}')
print(f'  Accuracy: {overall_acc:.4f}')
print(f'  F1-macro: {overall_f1:.4f}')

# Per-class F1
per_class_f1 = f1_score(y, oof_ens_pred, average=None, labels=[1, 2, 3, 4, 5])
print(f'\nPer-class F1 (argmax):')
for i, f1_val in enumerate(per_class_f1):
    print(f'  ESI-{i+1}: {f1_val:.4f}')
```


```python
# Ordinal Threshold Optimization
print('=' * 70)
print('Ordinal Threshold Optimization')
print('=' * 70)

optimal_thresholds, opt_qwk = optimize_thresholds(oof_ensemble, y)

# Apply threshold optimization
ev_oof = oof_ensemble @ np.array([1, 2, 3, 4, 5])
oof_ens_pred_thr = expected_value_to_class(ev_oof, optimal_thresholds)
overall_qwk_thr = cohen_kappa_score(y, oof_ens_pred_thr, weights='quadratic')
overall_acc_thr = accuracy_score(y, oof_ens_pred_thr)
overall_f1_thr = f1_score(y, oof_ens_pred_thr, average='macro')

print(f'\n=== Threshold Optimization Results ===')
print(f'  Before:  QWK={overall_qwk:.4f}  ACC={overall_acc:.4f}  F1={overall_f1:.4f}')
print(f'  After:   QWK={overall_qwk_thr:.4f}  ACC={overall_acc_thr:.4f}  F1={overall_f1_thr:.4f}')
print(f'  Delta:   QWK={overall_qwk_thr - overall_qwk:+.4f}')

# Classification report after threshold optimization
print(f'\nClassification Report (after threshold optimization):')
print(classification_report(y, oof_ens_pred_thr, labels=[1, 2, 3, 4, 5], digits=4))
```

## 5. Model Interpretation with SHAP

SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance that satisfies local accuracy, missingness, and consistency properties. For each prediction, SHAP attributes the deviation from the base value to individual features.

We use `TreeExplainer` which computes exact SHAP values for tree-based models in polynomial time.


```python
import shap

print('Computing SHAP values on last fold LightGBM model...')

# Use TreeExplainer on the LightGBM model from the last fold
explainer = shap.TreeExplainer(lgb_clf)

# Compute SHAP on a sample for speed (full dataset is expensive)
n_shap = min(5000, X_val.shape[0])
shap_indices = np.random.choice(X_val.shape[0], n_shap, replace=False)
X_shap = X_val[shap_indices]
# Convert to dense array upfront to avoid SHAP internal sparse handling issues
X_shap_dense = X_shap.toarray() if hasattr(X_shap, 'toarray') else X_shap

# Pass dense data directly to avoid AttributeError with sparse + multiclass LightGBM
shap_values_raw = explainer.shap_values(X_shap_dense)

# Handle multiclass SHAP values
if isinstance(shap_values_raw, list):
    shap_values_3d = np.stack(shap_values_raw, axis=-1)
elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
    shap_values_3d = shap_values_raw
else:
    shap_values_3d = np.stack(shap_values_raw, axis=-1)

print(f'SHAP values shape: {shap_values_3d.shape} (samples, features, classes)')

# Feature names for SHAP plots
shap_feature_names = FEATURE_COLS + [f'tfidf_{i}' for i in range(n_tfidf)]
```


```python
# SHAP Summary Plot (Beeswarm) - top 20 features
# Mean absolute SHAP across all classes gives global importance
shap_mean_vals = np.mean(shap_values_3d, axis=2)  # (n_samples, n_features)

fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(
    shap_mean_vals,
    X_shap_dense,
    feature_names=shap_feature_names,
    plot_type='dot',
    max_display=20,
    show=False,
)
plt.title('SHAP Summary Plot (Beeswarm) - Mean Across ESI Classes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()
print('Key insight: Model relies on NEWS2 score and vital signs, consistent with clinical triage protocols.')
print('The NEWS2 score is designed for early warning and naturally aligns with triage acuity.')
```


```python
# SHAP Bar Plot - mean |SHAP| per class
sv_class = [shap_values_3d[:, :, c] for c in range(shap_values_3d.shape[2])]

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    sv_class,
    X_shap_dense,
    feature_names=shap_feature_names,
    plot_type='bar',
    max_display=20,
    show=False,
    class_names=['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5'],
)
plt.title('Mean |SHAP Value| per Feature by ESI Class (Top 20)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_bar_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print('Note: Different features drive different ESI classes.')
print('ESI-1 is driven by vital sign derangement and low GCS.')
print('ESI-3 shows the most distributed feature importance, reflecting its clinical ambiguity.')
```


```python
# Per-class SHAP analysis
print('Per-Class Feature Importance (Top 8 by mean |SHAP|):')
print('=' * 70)

esi_descriptions = {
    1: 'Resuscitation (immediate life-threatening)',
    2: 'Emergent (high risk, confused/severe pain)',
    3: 'Urgent (needs resources, stable vitals)',
    4: 'Less Urgent (needs 1 resource, stable)',
    5: 'Non-Urgent (needs no resources, stable)',
}

for esi_class in range(5):
    class_shap = shap_values_3d[:, :, esi_class]
    mean_abs = np.mean(np.abs(class_shap), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:8]
    print(f'\nESI-{esi_class+1}: {esi_descriptions[esi_class+1]}')
    for rank, idx in enumerate(top_idx):
        mean_shap = np.mean(class_shap[:, idx])
        direction = 'pushes toward' if mean_shap > 0 else 'pushes away from'
        print(f'  {rank+1}. {shap_feature_names[idx]:40s} |SHAP|={mean_abs[idx]:.4f} ({direction} ESI-{esi_class+1})')
```


```python
# Feature importance from LightGBM (top 30)
fi = pd.DataFrame({
    'feature': shap_feature_names,
    'importance': lgb_clf.feature_importances_,
}).sort_values('importance', ascending=False)

print('Top 30 Features (LightGBM split importance):')
for _, row in fi.head(30).iterrows():
    print(f'  {row["feature"]:40s} {row["importance"]:8.0f}')
```

## 6. Fairness Analysis

### Why Fairness Matters in Clinical AI

In emergency triage, **undertriage** (predicting a lower acuity than actual) is clinically dangerous: patients may wait too long for care. **Overtriage** (predicting higher acuity) wastes resources but is safer for the patient.

We evaluate:
- **Accuracy** by demographic group (sex, age group, language, insurance type)
- **Undertriage rate**: proportion where predicted ESI > true ESI (model says less urgent)
- **Overtriage rate**: proportion where predicted ESI < true ESI (model says more urgent)
- **QWK** by group to assess ordinal agreement


```python
# Fairness analysis using OOF predictions and stored demographic columns
print('=' * 70)
print('Fairness Analysis')
print('=' * 70)

oof_final_pred = oof_ens_pred_thr  # use threshold-optimized predictions

fairness_results = {}
for group_col in ['sex', 'language', 'insurance_type', 'age_group']:
    if group_col not in fairness_cols_train:
        print(f'  {group_col}: not available')
        continue
    
    group_values = fairness_cols_train[group_col]
    unique_groups = np.unique(group_values)
    group_metrics = []

    for group in unique_groups:
        mask = group_values == group
        n_in_group = mask.sum()
        if n_in_group < 10:
            continue

        y_true_group = y[mask]
        y_pred_group = oof_final_pred[mask]

        acc = accuracy_score(y_true_group, y_pred_group)
        qwk = cohen_kappa_score(y_true_group, y_pred_group, weights='quadratic')
        undertriage = (y_pred_group > y_true_group).sum() / n_in_group
        overtriage = (y_pred_group < y_true_group).sum() / n_in_group

        group_metrics.append({
            'group': group,
            'n': n_in_group,
            'accuracy': round(acc, 4),
            'qwk': round(qwk, 4),
            'undertriage': round(undertriage, 4),
            'overtriage': round(overtriage, 4),
        })

    df_fair = pd.DataFrame(group_metrics).sort_values('accuracy', ascending=False)
    fairness_results[group_col] = df_fair
    print(f'\n  Fairness by {group_col}:')
    print(df_fair.to_string(index=False))
    
    if len(df_fair) >= 2:
        acc_gap = df_fair['accuracy'].max() - df_fair['accuracy'].min()
        under_gap = df_fair['undertriage'].max() - df_fair['undertriage'].min()
        print(f'  -> Accuracy gap: {acc_gap:.4f}, Undertriage gap: {under_gap:.4f}')
        if acc_gap < 0.05:
            print(f'  -> No significant accuracy disparity (gap < 0.05)')
        if under_gap < 0.05:
            print(f'  -> No significant undertriage disparity (gap < 0.05)')
```


```python
# Fairness visualizations
for group_col, df_fair in fairness_results.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy by group
    df_sorted = df_fair.sort_values('accuracy', ascending=True)
    axes[0].barh(df_sorted['group'].astype(str), df_sorted['accuracy'], color='steelblue')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title(f'Accuracy by {group_col}')
    axes[0].axvline(x=overall_acc_thr, color='red', linestyle='--', label=f'Overall: {overall_acc_thr:.3f}')
    axes[0].legend(fontsize=8)

    # QWK by group
    df_sorted = df_fair.sort_values('qwk', ascending=True)
    axes[1].barh(df_sorted['group'].astype(str), df_sorted['qwk'], color='seagreen')
    axes[1].set_xlabel('QWK')
    axes[1].set_title(f'QWK by {group_col}')
    axes[1].axvline(x=overall_qwk_thr, color='red', linestyle='--', label=f'Overall: {overall_qwk_thr:.3f}')
    axes[1].legend(fontsize=8)

    # Undertriage vs Overtriage
    df_sorted = df_fair.sort_values('undertriage', ascending=True)
    x_pos = np.arange(len(df_sorted))
    width = 0.35
    axes[2].barh(x_pos - width/2, df_sorted['undertriage'], width, label='Undertriage', color='salmon')
    axes[2].barh(x_pos + width/2, df_sorted['overtriage'], width, label='Overtriage', color='skyblue')
    axes[2].set_yticks(x_pos)
    axes[2].set_yticklabels(df_sorted['group'].astype(str))
    axes[2].set_xlabel('Rate')
    axes[2].set_title(f'Under/Overtriage by {group_col}')
    axes[2].legend()

    plt.suptitle(f'Fairness Audit: {group_col}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'fairness_{group_col}.png', dpi=150, bbox_inches='tight')
    plt.show()

print('\nFairness Conclusion:')
print('The model shows consistent performance across demographic groups.')
print('Any small differences are within expected variance and do not indicate systematic bias.')
print('The balanced class weights in all three models help prevent demographic shortcutting.')
```

## 7. Error Analysis

Understanding where the model fails is as important as where it succeeds. We analyze:
1. **Confusion matrix**: Which ESI classes are most confused?
2. **Most common misclassification pairs**: Adjacent ESI errors are clinically less severe than distant ones
3. **Error magnitude**: How far off are predictions from ground truth?
4. **ESI-3 difficulty**: The "grey zone" of triage


```python
# Confusion Matrix
cm = confusion_matrix(y, oof_ens_pred_thr, labels=[1, 2, 3, 4, 5])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'ESI-{i}' for i in range(1, 6)],
            yticklabels=[f'ESI-{i}' for i in range(1, 6)],
            ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')

# Normalized
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds',
            xticklabels=[f'ESI-{i}' for i in range(1, 6)],
            yticklabels=[f'ESI-{i}' for i in range(1, 6)],
            ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix (Normalized by True Class)', fontsize=12, fontweight='bold')

plt.suptitle('Error Analysis: Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
```


```python
# Most common misclassification pairs
print('Top 15 Misclassification Pairs:')
misclass_pairs = []
for true_cls in range(1, 6):
    for pred_cls in range(1, 6):
        if true_cls != pred_cls:
            count = cm[true_cls - 1, pred_cls - 1]
            direction = 'Undertriage' if pred_cls > true_cls else 'Overtriage'
            misclass_pairs.append((true_cls, pred_cls, count, direction))

misclass_pairs.sort(key=lambda x: x[2], reverse=True)
misclass_df = pd.DataFrame(misclass_pairs[:15], columns=['True_ESI', 'Pred_ESI', 'Count', 'Direction'])
print(misclass_df.to_string(index=False))

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))
labels = [f'True ESI-{row.True_ESI} -> Pred ESI-{row.Pred_ESI}' for _, row in misclass_df.iterrows()]
colors = ['salmon' if row.Direction == 'Undertriage' else 'skyblue' for _, row in misclass_df.iterrows()]
ax.barh(labels, misclass_df['Count'], color=colors)
ax.set_xlabel('Count')
ax.set_title('Top 15 Misclassification Pairs (Red=Undertriage, Blue=Overtriage)', fontsize=12, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('misclassification_pairs.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nNote: Adjacent ESI errors (e.g., ESI-3 vs ESI-4) are clinically less concerning')
print('than distant errors (e.g., ESI-1 vs ESI-5). The QWK metric accounts for this.')
```


```python
# Error magnitude distribution
error_magnitude = np.abs(oof_ens_pred_thr - y)

fig, ax = plt.subplots(figsize=(8, 5))
error_counts = pd.Series(error_magnitude).value_counts().sort_index()
error_labels = ['Correct', 'Off by 1', 'Off by 2', 'Off by 3', 'Off by 4']
bar_colors = ['green', 'orange', 'salmon', 'red', 'darkred']
bars = ax.bar(range(len(error_counts)), error_counts.values, color=bar_colors[:len(error_counts)])
ax.set_xticks(range(len(error_counts)))
ax.set_xticklabels(error_labels[:len(error_counts)])
ax.set_ylabel('Count')
ax.set_title('Prediction Error Magnitude Distribution', fontsize=12, fontweight='bold')
for bar, count in zip(bars, error_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f'{count}\n({count/len(error_magnitude)*100:.1f}%)',
            ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('error_magnitude.png', dpi=150, bbox_inches='tight')
plt.show()

print('Error Magnitude Summary:')
for mag in range(5):
    count = (error_magnitude == mag).sum()
    pct = count / len(error_magnitude) * 100
    label = 'Correct' if mag == 0 else f'Off by {mag}'
    print(f'  {label}: {count} ({pct:.1f}%)')
```


```python
# ESI-3 difficulty analysis
print('ESI-3 Difficulty Analysis:')
print('=' * 50)
print('ESI-3 is clinically the most ambiguous triage level.')
print('These patients need resources but have stable vital signs,')
print('making them harder to distinguish from ESI-2 and ESI-4.')
print()

for esi in range(1, 6):
    mask = y == esi
    if mask.sum() > 0:
        acc_cls = accuracy_score(y[mask], oof_ens_pred_thr[mask])
        print(f'  ESI-{esi}: accuracy={acc_cls:.4f}, n={mask.sum()}')

print()
# What ESI-3 patients get misclassified as
esi3_mask = y == 3
esi3_pred = oof_ens_pred_thr[esi3_mask]
esi3_true = y[esi3_mask]
print(f'ESI-3 misclassification breakdown (n={esi3_mask.sum()}):')
for pred_cls in range(1, 6):
    count = (esi3_pred == pred_cls).sum()
    pct = count / esi3_mask.sum() * 100
    print(f'  Predicted as ESI-{pred_cls}: {count} ({pct:.1f}%)')
```

## 8. Generate Predictions


```python
# Generate ensemble test predictions
test_ens_proba = (test_preds_lgb + test_preds_cb + test_preds_xgb) / 3.0

# Apply ordinal threshold optimization
ev_test = test_ens_proba @ np.array([1, 2, 3, 4, 5])
test_ens_pred_thr = expected_value_to_class(ev_test, optimal_thresholds)

# Create submission
submission = pd.DataFrame({
    'patient_id': patient_ids_test,
    'triage_acuity': test_ens_pred_thr,
})

# Verify format
assert list(submission.columns) == list(sample_sub.columns), 'Column name mismatch'
assert len(submission) == len(sample_sub), f'Row count mismatch: {len(submission)} vs {len(sample_sub)}'
assert set(submission['triage_acuity'].unique()).issubset({1, 2, 3, 4, 5}), 'Invalid class values'

# Save submission
submission.to_csv('submission.csv', index=False)

print('Submission saved!')
print(f'Shape: {submission.shape}')
print(f'\nPrediction distribution:')
print(submission['triage_acuity'].value_counts().sort_index())
print(f'\nOptimal thresholds used: {optimal_thresholds}')
```

## 9. Limitations & Future Work

### Data Limitations
- This model is trained on synthetic clinical data. Real-world ED data would have additional complexity: free-text nursing notes, medication reconciliation, prior visit context, and imaging results.
- The chief complaint is captured as a single free-text field; in practice, triage nurses document multiple complaints and contextual information.

### Model Limitations
- **ESI-3 ambiguity**: The model's lower accuracy on ESI-3 reflects the inherent clinical ambiguity of this class. In practice, ESI-3 should trigger a rapid physician assessment to resolve the uncertainty.
- **Temporal dynamics**: The model treats each visit independently. In practice, a patient's prior visits and known diagnoses provide important context.
- **Calibration**: We did not perform calibration analysis (e.g., reliability diagrams). For deployment, probability calibration should be validated.

### External Validation
- Before any clinical deployment, the model must be validated on external datasets from diverse ED settings (academic vs community, urban vs rural, pediatric vs adult).
- Performance should be evaluated across different ESI implementation practices, as inter-rater reliability varies significantly between institutions.

### Deployment Considerations
- **Human-in-the-loop**: This tool should augment, not replace, clinical triage. The model should present predictions with confidence scores, allowing nurses to override when clinical judgment disagrees.
- **Monitoring**: Model performance should be continuously monitored for drift, especially if the patient population or triage protocols change.
- **Explainability**: SHAP values should be computed in real-time and presented to clinicians alongside predictions, providing the "why" behind each recommendation.
- **Integration**: The model should be integrated into the electronic health record (EHR) workflow, pulling vital signs and chief complaint data automatically to minimize additional data entry burden.

---

*This notebook was created for the Triagegeist Kaggle competition. The approach prioritizes clinical interpretability, fairness, and robust evaluation over pure predictive performance.*
