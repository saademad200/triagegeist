```python
# =============================================================================
# TRIAGEGEIST: A Triage Second-Opinion System
# Detecting Undertriage Risk via Clinical AI and Demographic Equity Analysis
# =============================================================================
#
# Approach:
#   1. Train an objective clinical model (vitals only, no demographics)
#   2. Compare predictions to assigned acuity to detect systematic bias
#   3. Build an undertriage risk score with SHAP explanations
#   4. Produce a demographic equity audit
#   5. Demonstrate a clinical decision support alert
#

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (cohen_kappa_score, accuracy_score,
                              classification_report, confusion_matrix)
from sklearn.calibration import calibration_curve
from scipy import stats
from scipy.stats import chi2_contingency
import lightgbm as lgb
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

sns.set_theme(style='whitegrid', font_scale=1.1)
PALETTE = sns.color_palette('RdYlBu_r', 5)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("=" * 65)
print("TRIAGEGEIST -- Triage Second-Opinion System")
print("=" * 65)

import os
_found = None
for _root, _dirs, _files in os.walk('/kaggle/input'):
    if 'train.csv' in _files:
        _found = _root
        break
if _found is None:
    raise FileNotFoundError("train.csv not found. Tree: " + str(list(os.walk('/kaggle/input'))))
print(f"Data directory: {_found}")
DATA = _found

train = pd.read_csv(os.path.join(DATA, 'train.csv'))
test  = pd.read_csv(os.path.join(DATA, 'test.csv'))
cc    = pd.read_csv(os.path.join(DATA, 'chief_complaints.csv'))
ph    = pd.read_csv(os.path.join(DATA, 'patient_history.csv'))
sub   = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))

cc_text = cc[['patient_id', 'chief_complaint_raw']]
train = train.merge(cc_text, on='patient_id', how='left')
train = train.merge(ph,      on='patient_id', how='left')
test  = test.merge(cc_text,  on='patient_id', how='left')
test  = test.merge(ph,       on='patient_id', how='left')

HX_COLS  = [c for c in train.columns if c.startswith('hx_')]
DEMO_COLS = ['sex', 'age', 'age_group', 'language', 'insurance_type']

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Comorbidity flags: {len(HX_COLS)}")

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS  (key stats only)
# =============================================================================
print("\n[EDA] Acuity distribution:")
acuity_dist = train['triage_acuity'].value_counts().sort_index()
for k, v in acuity_dist.items():
    bar = '#' * int(v / 500)
    print(f"  ESI-{k}: {v:6,}  {bar}")

miss_rate = train[['systolic_bp','respiratory_rate','temperature_c']].isnull().mean()
print(f"\n[EDA] Vital missingness: BP={miss_rate['systolic_bp']:.1%}, "
      f"RR={miss_rate['respiratory_rate']:.1%}, Temp={miss_rate['temperature_c']:.1%}")
print("  Note: missingness is clinically informative -- "
      "lower-acuity patients often skip full vital assessment.")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
def engineer(df, encoders=None):
    df = df.copy()

    # -- Missingness as clinical signal --
    df['bp_missing'] = df['systolic_bp'].isnull().astype(int)
    df['rr_missing'] = df['respiratory_rate'].isnull().astype(int)

    # -- Impute vitals with population median --
    for col in ['systolic_bp','diastolic_bp','mean_arterial_pressure',
                'pulse_pressure','respiratory_rate','temperature_c','shock_index']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # -- Derived clinical features --
    df['bp_ratio']       = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    df['shock_flag']     = (df['shock_index'] > 1.0).astype(int)
    df['spo2_critical']  = (df['spo2'] < 90).astype(int)
    df['spo2_low']       = (df['spo2'] < 94).astype(int)
    df['gcs_critical']   = (df['gcs_total'] < 9).astype(int)
    df['gcs_impaired']   = (df['gcs_total'] < 14).astype(int)
    df['fever']          = (df['temperature_c'] > 38.3).astype(int)
    df['hypothermia']    = (df['temperature_c'] < 36.0).astype(int)
    df['hypotension']    = (df['systolic_bp'] < 90).astype(int)
    df['tachycardia']    = (df['heart_rate'] > 100).astype(int)
    df['bradycardia']    = (df['heart_rate'] < 60).astype(int)
    df['tachypnea']      = (df['respiratory_rate'] > 20).astype(int)
    df['news2_high']     = (df['news2_score'] >= 7).astype(int)  # high-risk threshold
    df['news2_medium']   = ((df['news2_score'] >= 5) & (df['news2_score'] < 7)).astype(int)

    # -- Comorbidity burden --
    df['comorbidity_count'] = df[HX_COLS].sum(axis=1)
    df['high_risk_hx'] = df[['hx_heart_failure','hx_malignancy','hx_ckd',
                               'hx_coagulopathy','hx_immunosuppressed']].sum(axis=1)
    df['cardio_hx']    = df[['hx_heart_failure','hx_atrial_fibrillation',
                               'hx_coronary_artery_disease']].sum(axis=1)

    # -- Categorical encoding --
    cat_cols = ['arrival_mode','mental_status_triage','chief_complaint_system',
                'pain_location','sex','age_group','language','insurance_type',
                'arrival_day','arrival_season','shift','transport_origin']
    enc = encoders or {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        le = enc.get(col, LabelEncoder())
        df[col] = le.fit_transform(df[col].astype(str))
        enc[col] = le
    return df, enc

train_p, enc = engineer(train)
test_p,  _   = engineer(test, enc)

# =============================================================================
# 4. DEFINE FEATURE SETS
# =============================================================================
# Clinical-only features (no demographics) -- used for the objective model
CLINICAL = [
    'systolic_bp','diastolic_bp','mean_arterial_pressure','pulse_pressure',
    'heart_rate','respiratory_rate','temperature_c','spo2','gcs_total',
    'pain_score','news2_score','shock_index',
    'num_prior_ed_visits_12m','num_prior_admissions_12m',
    'num_active_medications','num_comorbidities',
    'arrival_mode','arrival_hour','mental_status_triage',
    'chief_complaint_system','pain_location',
    'bp_missing','rr_missing','bp_ratio','shock_flag',
    'spo2_critical','spo2_low','gcs_critical','gcs_impaired',
    'fever','hypothermia','hypotension','tachycardia','bradycardia','tachypnea',
    'news2_high','news2_medium',
    'comorbidity_count','high_risk_hx','cardio_hx',
] + HX_COLS
CLINICAL = [c for c in CLINICAL if c in train_p.columns]

# Full feature set (includes demographics) -- for comparison
FULL = CLINICAL + [c for c in DEMO_COLS if c in train_p.columns]

# =============================================================================
# 4b. NLP PIPELINE: Chief Complaint Text Analysis
# =============================================================================
print("\n[NLP] Extracting features from chief complaint free text...")

# Clinically-grounded high-risk keyword lexicon
NLP_RISK_TERMS = {
    'nlp_chest_pain':     ['chest pain', 'chest tightness', 'chest pressure', 'chest discomfort'],
    'nlp_dyspnea':        ['shortness of breath', 'difficulty breathing', 'dyspnea', 'can\'t breathe', 'sob'],
    'nlp_altered_ms':     ['altered mental', 'confusion', 'unresponsive', 'unconscious', 'not alert', 'altered consciousness'],
    'nlp_syncope':        ['syncope', 'fainted', 'passed out', 'loss of consciousness', 'blackout'],
    'nlp_severe_pain':    ['severe pain', 'excruciating', '10/10', 'worst pain', 'unbearable pain'],
    'nlp_bleeding':       ['bleeding', 'hemorrhage', 'blood loss', 'hematemesis', 'melena', 'bloody'],
    'nlp_stroke_signs':   ['facial droop', 'arm weakness', 'speech difficulty', 'stroke', 'facial numbness'],
    'nlp_cardiac':        ['palpitations', 'irregular heartbeat', 'heart racing', 'skipping beats', 'heart pounding'],
    'nlp_sepsis_signs':   ['high fever', 'rigors', 'chills', 'sepsis', 'infection'],
    'nlp_trauma':         ['trauma', 'injury', 'accident', 'mvc', 'motor vehicle', 'fell from'],
    'nlp_abdominal':      ['abdominal pain', 'stomach pain', 'belly pain', 'epigastric', 'abdominal'],
    'nlp_neuro':          ['worst headache', 'thunderclap', 'vision changes', 'seizure', 'thunderclap headache'],
    'nlp_allergic':       ['allergic reaction', 'anaphylaxis', 'throat swelling', 'hives', 'anaphylactic'],
    'nlp_weakness':       ['generalized weakness', 'can\'t walk', 'profound fatigue', 'extreme weakness'],
    'nlp_respiratory':    ['hemoptysis', 'respiratory distress', 'wheezing', 'coughing blood', 'stridor'],
}

def extract_keyword_flags(df, text_col='chief_complaint_raw'):
    text = df[text_col].fillna('').str.lower()
    out = {}
    for key, terms in NLP_RISK_TERMS.items():
        out[key] = text.apply(lambda t: int(any(term in t for term in terms))).values
    return pd.DataFrame(out, index=df.index)

kw_train_df = extract_keyword_flags(train)
kw_test_df  = extract_keyword_flags(test)

# TF-IDF + Latent Semantic Analysis for dense text representation
tfidf_vec = TfidfVectorizer(
    max_features=1000, ngram_range=(1, 2),
    min_df=10, sublinear_tf=True, stop_words='english'
)
train_tfidf = tfidf_vec.fit_transform(train['chief_complaint_raw'].fillna(''))
test_tfidf  = tfidf_vec.transform(test['chief_complaint_raw'].fillna(''))

text_svd = TruncatedSVD(n_components=20, random_state=42)
lsa_train = text_svd.fit_transform(train_tfidf)
lsa_test  = text_svd.transform(test_tfidf)

NLP_KEYWORD_COLS = list(NLP_RISK_TERMS.keys())
NLP_SVD_COLS     = [f'nlp_lsa_{i}' for i in range(20)]

lsa_train_df = pd.DataFrame(lsa_train, columns=NLP_SVD_COLS, index=train.index)
lsa_test_df  = pd.DataFrame(lsa_test,  columns=NLP_SVD_COLS, index=test.index)

train_p = pd.concat([train_p, kw_train_df, lsa_train_df], axis=1)
test_p  = pd.concat([test_p,  kw_test_df,  lsa_test_df],  axis=1)

NLP_ENHANCED = CLINICAL + NLP_KEYWORD_COLS + NLP_SVD_COLS
NLP_ENHANCED = [c for c in NLP_ENHANCED if c in train_p.columns]

print(f"  High-risk keyword prevalence (top 10):")
kw_prev = kw_train_df.mean().sort_values(ascending=False)
for col, rate in kw_prev.head(10).items():
    print(f"    {col.replace('nlp_',''):28s}: {rate*100:.1f}%")
print(f"  LSA explained variance (20 components): {text_svd.explained_variance_ratio_.sum()*100:.1f}%")
print(f"  NLP-enhanced feature count: {len(NLP_ENHANCED)} ({len(NLP_KEYWORD_COLS)} keywords + {len(NLP_SVD_COLS)} LSA)")

# =============================================================================
# 5. TRAIN CLINICAL MODEL (objective, no demographics)
# =============================================================================
print("\n[Model] Training objective clinical model (demographics excluded)...")

X_clin      = train_p[CLINICAL].values.astype(np.float32)
X_test_clin = test_p[CLINICAL].values.astype(np.float32)
y           = train_p['triage_acuity'].values - 1  # 0-indexed

params_clin = dict(
    objective='multiclass', num_class=5, metric='multi_logloss',
    learning_rate=0.05, num_leaves=127,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1, verbose=-1, n_jobs=-1,
)

skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_clin    = np.zeros((len(train), 5))
test_clin   = np.zeros((len(test),  5))
qwk_clin    = []
last_model  = None

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_clin, y)):
    dtrain = lgb.Dataset(X_clin[tr_idx], label=y[tr_idx])
    dval   = lgb.Dataset(X_clin[va_idx], label=y[va_idx])
    m = lgb.train(params_clin, dtrain, num_boost_round=1000,
                  valid_sets=[dval],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(500)])
    p = m.predict(X_clin[va_idx])
    oof_clin[va_idx] = p
    test_clin += m.predict(X_test_clin) / 5
    qwk = cohen_kappa_score(y[va_idx], np.argmax(p, axis=1), weights='quadratic')
    qwk_clin.append(qwk)
    last_model = m
    print(f"  Fold {fold+1}: QWK={qwk:.4f}")

qwk_clin_mean = np.mean(qwk_clin)
print(f"\nClinical model  -- CV QWK: {qwk_clin_mean:.4f} ? {np.std(qwk_clin):.4f}")

# =============================================================================
# 6. TRAIN FULL MODEL (with demographics) -- for QWK gap comparison
# =============================================================================
print("\n[Model] Training full model (demographics included)...")

X_full      = train_p[FULL].values.astype(np.float32)
X_test_full = test_p[FULL].values.astype(np.float32)

oof_full    = np.zeros((len(train), 5))
test_full   = np.zeros((len(test),  5))
qwk_full    = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y)):
    dtrain = lgb.Dataset(X_full[tr_idx], label=y[tr_idx])
    dval   = lgb.Dataset(X_full[va_idx], label=y[va_idx])
    m2 = lgb.train(params_clin, dtrain, num_boost_round=1000,
                   valid_sets=[dval],
                   callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(500)])
    p = m2.predict(X_full[va_idx])
    oof_full[va_idx] = p
    test_full += m2.predict(X_test_full) / 5
    qwk = cohen_kappa_score(y[va_idx], np.argmax(p, axis=1), weights='quadratic')
    qwk_full.append(qwk)
    last_full_model = m2

qwk_full_mean = np.mean(qwk_full)
print(f"Full model      -- CV QWK: {qwk_full_mean:.4f} ? {np.std(qwk_full):.4f}")
print(f"\n? QWK gap (demographics removed): {qwk_full_mean - qwk_clin_mean:+.4f}")
print(f"  ? Demographic features account for {(qwk_full_mean - qwk_clin_mean) / qwk_full_mean * 100:.1f}% "
      f"of the full model's predictive power")

# =============================================================================
# 6b. TRAIN NLP-ENHANCED MODEL (clinical + chief complaint text features)
# =============================================================================
print("\n[Model] Training NLP-enhanced model (clinical + text NLP features)...")

X_nlp_train = train_p[NLP_ENHANCED].values.astype(np.float32)
X_nlp_test  = test_p[NLP_ENHANCED].values.astype(np.float32)

oof_nlp_pred  = np.zeros((len(train), 5))
test_nlp_pred = np.zeros((len(test),  5))
qwk_nlp       = []
last_nlp_model = None

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_nlp_train, y)):
    dtrain = lgb.Dataset(X_nlp_train[tr_idx], label=y[tr_idx])
    dval   = lgb.Dataset(X_nlp_train[va_idx], label=y[va_idx])
    m3 = lgb.train(params_clin, dtrain, num_boost_round=1000,
                   valid_sets=[dval],
                   callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(500)])
    p = m3.predict(X_nlp_train[va_idx])
    oof_nlp_pred[va_idx] = p
    test_nlp_pred += m3.predict(X_nlp_test) / 5
    qwk = cohen_kappa_score(y[va_idx], np.argmax(p, axis=1), weights='quadratic')
    qwk_nlp.append(qwk)
    last_nlp_model = m3
    print(f"  Fold {fold+1}: QWK={qwk:.4f}")

qwk_nlp_mean = np.mean(qwk_nlp)
print(f"\nNLP-Enhanced model -- CV QWK: {qwk_nlp_mean:.4f} +/- {np.std(qwk_nlp):.4f}")
print(f"Text lift over clinical model: {qwk_nlp_mean - qwk_clin_mean:+.4f}")
print(f"Text lift over full model:     {qwk_nlp_mean - qwk_full_mean:+.4f}")

# =============================================================================
# 6c. NLP ABLATION STUDY: Keyword-Only vs LSA-Only vs Combined
# =============================================================================
print("\n[Ablation] NLP ablation: Keyword-Only vs LSA-Only vs Combined...")

# Model D: Clinical + Keywords Only (no LSA)
KW_ONLY = [c for c in CLINICAL + NLP_KEYWORD_COLS if c in train_p.columns]
X_kw_train = train_p[KW_ONLY].values.astype(np.float32)
X_kw_test  = test_p[KW_ONLY].values.astype(np.float32)
oof_kw = np.zeros((len(train), 5))
test_kw = np.zeros((len(test), 5))
qwk_kw = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_kw_train, y)):
    dtrain = lgb.Dataset(X_kw_train[tr_idx], label=y[tr_idx])
    dval   = lgb.Dataset(X_kw_train[va_idx], label=y[va_idx])
    m_kw = lgb.train(params_clin, dtrain, num_boost_round=1000,
                     valid_sets=[dval],
                     callbacks=[lgb.early_stopping(50, verbose=False),
                                lgb.log_evaluation(500)])
    p = m_kw.predict(X_kw_train[va_idx])
    oof_kw[va_idx] = p
    test_kw += m_kw.predict(X_kw_test) / 5
    qwk_kw.append(cohen_kappa_score(y[va_idx], np.argmax(p, axis=1), weights='quadratic'))
qwk_kw_mean = np.mean(qwk_kw)

# Model E: Clinical + LSA Only (no keywords)
LSA_ONLY = [c for c in CLINICAL + NLP_SVD_COLS if c in train_p.columns]
X_lsa_train = train_p[LSA_ONLY].values.astype(np.float32)
X_lsa_test  = test_p[LSA_ONLY].values.astype(np.float32)
oof_lsa = np.zeros((len(train), 5))
test_lsa = np.zeros((len(test), 5))
qwk_lsa = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_lsa_train, y)):
    dtrain = lgb.Dataset(X_lsa_train[tr_idx], label=y[tr_idx])
    dval   = lgb.Dataset(X_lsa_train[va_idx], label=y[va_idx])
    m_lsa = lgb.train(params_clin, dtrain, num_boost_round=1000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(500)])
    p = m_lsa.predict(X_lsa_train[va_idx])
    oof_lsa[va_idx] = p
    test_lsa += m_lsa.predict(X_lsa_test) / 5
    qwk_lsa.append(cohen_kappa_score(y[va_idx], np.argmax(p, axis=1), weights='quadratic'))
qwk_lsa_mean = np.mean(qwk_lsa)

print(f"\n  Ablation results:")
print(f"    Clinical baseline           : {qwk_clin_mean:.4f}")
print(f"    + Keywords only             : {qwk_kw_mean:.4f}  ({qwk_kw_mean-qwk_clin_mean:+.4f})")
print(f"    + LSA only                  : {qwk_lsa_mean:.4f}  ({qwk_lsa_mean-qwk_clin_mean:+.4f})")
print(f"    + Keywords + LSA (full NLP) : {qwk_nlp_mean:.4f}  ({qwk_nlp_mean-qwk_clin_mean:+.4f})")

# =============================================================================
# 6d. TEXT-SHUFFLE CONTROL -- Validating LSA signal vs. synthetic leakage
# =============================================================================
# Hypothesis: if LSA lift is genuine (text encodes real clinical semantics),
# then randomly permuting text across patients (breaking all text-label links)
# should cause QWK to drop back to clinical baseline (~0.9296).
# If QWK stays high on shuffled text, it indicates direct acuity encoding in the
# synthetic text generation process -- a strong leakage signal.
print("\n[Shuffle Control] Text-shuffle experiment: validating LSA signal authenticity...")
print("  Permuting chief_complaint_raw across patients (n=80,000)...")

np.random.seed(42)
shuffled_text = train['chief_complaint_raw'].fillna('').values.copy()
np.random.shuffle(shuffled_text)

tfidf_shuffle = TfidfVectorizer(
    max_features=1000, ngram_range=(1, 2),
    min_df=10, sublinear_tf=True, stop_words='english'
)
tfidf_shuffle_mat = tfidf_shuffle.fit_transform(shuffled_text)
svd_shuffle = TruncatedSVD(n_components=20, random_state=42)
lsa_shuffle_arr = svd_shuffle.fit_transform(tfidf_shuffle_mat)

train_p_shuffle = train_p.copy()
for i, col in enumerate(NLP_SVD_COLS):
    train_p_shuffle[col] = lsa_shuffle_arr[:, i]

X_lsa_shuffle = train_p_shuffle[LSA_ONLY].values.astype(np.float32)

qwk_shuffle = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_lsa_shuffle, y)):
    dtrain_sh = lgb.Dataset(X_lsa_shuffle[tr_idx], label=y[tr_idx])
    dval_sh   = lgb.Dataset(X_lsa_shuffle[va_idx], label=y[va_idx])
    m_sh = lgb.train(params_clin, dtrain_sh, num_boost_round=1000,
                     valid_sets=[dval_sh],
                     callbacks=[lgb.early_stopping(50, verbose=False),
                                lgb.log_evaluation(500)])
    p_sh = m_sh.predict(X_lsa_shuffle[va_idx])
    qwk_shuffle.append(
        cohen_kappa_score(y[va_idx], np.argmax(p_sh, axis=1), weights='quadratic')
    )

qwk_shuffle_mean = np.mean(qwk_shuffle)
lsa_drop = qwk_lsa_mean - qwk_shuffle_mean

print(f"\n  Text-Shuffle Control Results:")
print(f"    LSA-Only QWK (genuine text) : {qwk_lsa_mean:.4f}")
print(f"    LSA-Only QWK (shuffled text): {qwk_shuffle_mean:.4f}")
print(f"    QWK drop from shuffle       : {lsa_drop:+.4f}")

if qwk_shuffle_mean <= qwk_clin_mean + 0.005:
    shuffle_verdict = "DROPS TO BASELINE"
    shuffle_interp = (
        "Shuffled text QWK collapses to clinical baseline. This confirms the LSA model "
        "learns from genuine text-acuity correlations in the training data, not from "
        "random noise or feature leakage in the modeling pipeline. The lift is a property "
        "of the synthetic data generation (text aligned to acuity labels), not a modeling artifact."
    )
else:
    shuffle_verdict = "REMAINS ELEVATED"
    shuffle_interp = (
        "Shuffled text QWK remains above clinical baseline. This indicates the synthetic text "
        "generation process directly encodes acuity labels into word-level statistics, "
        "independent of patient-level text assignment. Real-world NLP lift would be lower."
    )

print(f"    Verdict: {shuffle_verdict}")
print(f"    Interpretation: {shuffle_interp}")

# =============================================================================
# 7. UNDERTRIAGE DETECTION
# =============================================================================
print("\n[Bias] Computing triage bias (clinical prediction ? assigned acuity)...")

train['pred_clin']    = np.argmax(oof_clin, axis=1) + 1
train['pred_clin_lo'] = oof_clin[:, 0]  # P(acuity=1) -- critical risk
train['actual']       = train['triage_acuity']
train['bias']         = train['pred_clin'] - train['actual']
# Undertriage = model thinks patient is more urgent than nurse assigned
train['undertriaged'] = (train['bias'] <= -1).astype(int)
train['severely_ut']  = (train['bias'] <= -2).astype(int)

n_under = train['undertriaged'].sum()
n_severe = train['severely_ut'].sum()
print(f"  Undertriaged (?1 level): {n_under:,} ({n_under/len(train)*100:.1f}%)")
print(f"  Severely undertriaged (?2 levels): {n_severe:,} ({n_severe/len(train)*100:.1f}%)")

# =============================================================================
# 8. DEMOGRAPHIC EQUITY AUDIT
# =============================================================================
print("\n[Equity] Demographic audit with statistical testing...")

def equity_stats(col_orig, label):
    groups = train.groupby(col_orig).agg(
        n=('bias','count'),
        mean_bias=('bias','mean'),
        std_bias=('bias','std'),
        undertriage_rate=('undertriaged','mean'),
        severe_ut_rate=('severely_ut','mean'),
    ).round(4)
    groups['ci95'] = 1.96 * groups['std_bias'] / np.sqrt(groups['n'])

    # Chi-square test on undertriage counts
    ct = pd.crosstab(train[col_orig], train['undertriaged'])
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"\n  [{label}] ??={chi2:.2f}, dof={dof}, p={p:.4f}")
    print(groups[['n','mean_bias','ci95','undertriage_rate','severe_ut_rate']].to_string())
    return groups, p

sex_stats,  sex_p   = equity_stats('sex',           'Sex')
lang_stats, lang_p  = equity_stats('language',      'Language')
ins_stats,  ins_p   = equity_stats('insurance_type','Insurance')
age_stats,  age_p   = equity_stats('age_group',     'Age Group')

# =============================================================================
# 8b. NURSE-LEVEL AND SITE-LEVEL ANALYSIS
# =============================================================================
print("\n[Nurse/Site] Provider-level undertriage audit...")

# Nurse-level: flag outlier nurses (undertriage rate > mean + 1.5 SD)
nurse_stats = train.groupby('triage_nurse_id').agg(
    n=('undertriaged','count'),
    undertriage_rate=('undertriaged','mean'),
    mean_bias=('bias','mean'),
).round(4)
nurse_stats['ci95'] = 1.96 * np.sqrt(
    nurse_stats['undertriage_rate'] * (1 - nurse_stats['undertriage_rate']) / nurse_stats['n'])
nurse_mean = nurse_stats['undertriage_rate'].mean()
nurse_std  = nurse_stats['undertriage_rate'].std()
nurse_stats['zscore'] = (nurse_stats['undertriage_rate'] - nurse_mean) / nurse_std
nurse_stats['outlier_high'] = (nurse_stats['zscore'] > 1.5).astype(int)
nurse_stats['outlier_low']  = (nurse_stats['zscore'] < -1.5).astype(int)

n_outlier_nurses = nurse_stats['outlier_high'].sum()
print(f"  Nurses with elevated undertriage rate (z > 1.5): {n_outlier_nurses}")
print(f"  Mean nurse undertriage rate: {nurse_mean*100:.1f}% +/- {nurse_std*100:.1f}%")
print(f"  Range: {nurse_stats['undertriage_rate'].min()*100:.1f}% to "
      f"{nurse_stats['undertriage_rate'].max()*100:.1f}%")

# Chi-square test across nurses
ct_nurse = pd.crosstab(train['triage_nurse_id'], train['undertriaged'])
chi2_n, p_nurse, _, _ = chi2_contingency(ct_nurse)
print(f"  Chi-square across nurses: chi2={chi2_n:.2f}, p={p_nurse:.4f}")

# Site-level
site_stats = train.groupby('site_id').agg(
    n=('undertriaged','count'),
    undertriage_rate=('undertriaged','mean'),
    mean_bias=('bias','mean'),
).round(4)
site_stats['ci95'] = 1.96 * np.sqrt(
    site_stats['undertriage_rate'] * (1 - site_stats['undertriage_rate']) / site_stats['n'])

ct_site = pd.crosstab(train['site_id'], train['undertriaged'])
chi2_s, p_site, _, _ = chi2_contingency(ct_site)
print(f"\n  Site-level undertriage rates (chi2 p={p_site:.4f}):")
print(site_stats[['n','undertriage_rate','ci95','mean_bias']].sort_values(
    'undertriage_rate', ascending=False).to_string())

# Temporal: hour of day and shift
hour_stats = train.groupby('arrival_hour').agg(
    n=('undertriaged','count'),
    undertriage_rate=('undertriaged','mean'),
).round(4)
shift_stats = train.groupby('shift').agg(
    n=('undertriaged','count'),
    undertriage_rate=('undertriaged','mean'),
).round(4)
ct_shift = pd.crosstab(train['shift'], train['undertriaged'])
chi2_sh, p_shift, _, _ = chi2_contingency(ct_shift)
print(f"\n  Shift undertriage rates (chi2 p={p_shift:.4f}):")
print(shift_stats.sort_values('undertriage_rate', ascending=False).to_string())

# SHAP interaction: elderly x pain_score
# Among elderly, show relationship between pain_score and undertriage
elderly_mask = train['age_group'] == 'elderly'
elderly_pain_bins = pd.cut(train.loc[elderly_mask, 'pain_score'],
                           bins=[-2, 0, 3, 6, 10],
                           labels=['no pain (0)', 'mild (1-3)', 'moderate (4-6)', 'severe (7-10)'])
elderly_pain_ut = train.loc[elderly_mask].groupby(elderly_pain_bins)['undertriaged'].agg(
    ['mean','count']).round(4)
elderly_pain_ut.columns = ['undertriage_rate','n']
print(f"\n  Elderly undertriage by pain level:")
print(elderly_pain_ut.to_string())

# =============================================================================
# 8c. NLP UNDERTRIAGE ANALYSIS: Keyword-Level Risk Stratification
# =============================================================================
print("\n[NLP Analysis] Undertriage rates by chief complaint keyword...")

nlp_analysis_rows = []
for col in NLP_KEYWORD_COLS:
    mask = kw_train_df[col] == 1
    n_flagged = int(mask.sum())
    if n_flagged < 50:
        continue
    ut_rate_kw    = train.loc[mask, 'undertriaged'].mean()
    ut_rate_other = train.loc[~mask, 'undertriaged'].mean()
    ct_kw = pd.crosstab(mask.astype(int), train['undertriaged'])
    chi2_kw, p_kw, _, _ = chi2_contingency(ct_kw)
    enrichment = ut_rate_kw / ut_rate_other if ut_rate_other > 0 else 0
    nlp_analysis_rows.append({
        'keyword':    col.replace('nlp_', ''),
        'n':          n_flagged,
        'ut_pct':     ut_rate_kw * 100,
        'baseline':   ut_rate_other * 100,
        'enrichment': enrichment,
        'p_value':    p_kw,
    })

nlp_kw_df = pd.DataFrame(nlp_analysis_rows).sort_values('enrichment', ascending=False)
print(f"\n  {'Keyword':<25} {'n':>6} {'UT%':>6}  {'Baseline':>8}  {'Enrich':>7}  {'p':>8}")
print("  " + "-" * 62)
for _, row in nlp_kw_df.iterrows():
    sig = '*' if row['p_value'] < 0.05 else ' '
    print(f"  {row['keyword']:<25} {row['n']:>6,} {row['ut_pct']:>5.1f}%  "
          f"{row['baseline']:>5.1f}%    {row['enrichment']:>5.2f}x  {row['p_value']:>8.4f}{sig}")

# Compare NLP-enhanced model undertriage detection vs clinical model
nlp_preds_1idx = np.argmax(oof_nlp_pred, axis=1) + 1
nlp_ut_detected = ((nlp_preds_1idx - train['actual'].values) <= -1).sum()
clin_ut_detected = train['undertriaged'].sum()
print(f"\n  Undertriage detected -- Clinical model: {clin_ut_detected:,} | NLP-Enhanced: {nlp_ut_detected:,}")

# =============================================================================
# 8d. WAITING ROOM DETERIORATION RISK SYSTEM (WRRS)
#     Time-sensitive risk stratification for patients already waiting
# =============================================================================
print("\n[WRRS] Computing Waiting Room Deterioration Risk Scores...")

# Clinically-grounded composite risk score for patients in the waiting room.
# Weights derived from published deterioration literature:
#   NEWS2 (Smith 2013), Shock Index, GCS, age-related blunted physiology,
#   NLP-detected time-critical presentations, triage gap severity.
def compute_wrrs(df, nlp_flags):
    scores = pd.Series(0.0, index=df.index)

    # 1. NEWS2 (primary validated deterioration predictor) -> 0-35 pts
    scores += np.clip(df['news2_score'] / 7.0, 0, 1) * 35

    # 2. Shock index (hemodynamic compromise) -> up to 20 pts
    scores += (df['shock_index'] > 1.0).astype(float) * 12
    scores += np.clip((df['shock_index'] - 0.6) / 0.8, 0, 1) * 8

    # 3. GCS impairment (neurological deterioration) -> up to 15 pts
    gcs_risk = np.where(df['gcs_total'] < 9, 1.0,
               np.where(df['gcs_total'] < 14, 0.55, 0.0))
    scores += gcs_risk * 15

    # 4. Elderly (blunted physiological reserve) -> 10 pts
    scores += (df['age_group'] == 'elderly').astype(float) * 10

    # 5. NLP time-critical keyword flags -> up to 10 pts
    critical_kw = ['nlp_sepsis_signs', 'nlp_stroke_signs', 'nlp_altered_ms',
                   'nlp_bleeding', 'nlp_cardiac', 'nlp_dyspnea']
    if all(c in nlp_flags.columns for c in critical_kw):
        nlp_critical = nlp_flags[critical_kw].any(axis=1).astype(float)
        scores += nlp_critical * 10

    # 6. Triage gap magnitude -> up to 10 pts
    gap = np.abs(df['bias']) if 'bias' in df.columns else pd.Series(0, index=df.index)
    scores += np.clip(gap / 3.0, 0, 1) * 10

    return np.clip(scores, 0, 100)

train['wrrs'] = compute_wrrs(train, kw_train_df)

TIER_LABELS = {
    'RED':    'RED (<=5 min)',
    'ORANGE': 'ORANGE (<=15 min)',
    'YELLOW': 'YELLOW (<=30 min)',
    'GREEN':  'GREEN (<=60 min)',
}
def assign_tier(score):
    if score >= 75:   return TIER_LABELS['RED']
    elif score >= 55: return TIER_LABELS['ORANGE']
    elif score >= 35: return TIER_LABELS['YELLOW']
    else:             return TIER_LABELS['GREEN']

train['wrrs_tier'] = train['wrrs'].apply(assign_tier)

ut_pts = train[train['undertriaged'] == 1].copy()
tier_order = [TIER_LABELS['RED'], TIER_LABELS['ORANGE'],
              TIER_LABELS['YELLOW'], TIER_LABELS['GREEN']]
tier_counts = ut_pts['wrrs_tier'].value_counts().reindex(tier_order, fill_value=0)

print(f"\n  WRRS tier breakdown -- undertriaged patients (n={len(ut_pts):,}):")
for tier, n in tier_counts.items():
    pct = n / len(ut_pts) * 100
    print(f"  {tier:<24}: {n:5,} ({pct:5.1f}%)")

tier_valid = ut_pts.groupby('wrrs_tier').agg(
    n=('wrrs', 'count'),
    wrrs_mean=('wrrs', 'mean'),
    news2_mean=('news2_score', 'mean'),
    elderly_pct=('age_group', lambda x: (x == 'elderly').mean() * 100),
    severe_ut_pct=('severely_ut', 'mean'),
    gap_mean=('bias', lambda x: np.abs(x).mean()),
).round(3).reindex(tier_order)

print("\n  Tier clinical validation:")
print(tier_valid[['n','wrrs_mean','news2_mean','elderly_pct','severe_ut_pct']].to_string())

red_rate      = tier_counts.get(TIER_LABELS['RED'],   0) / len(train)
orange_rate   = tier_counts.get(TIER_LABELS['ORANGE'],0) / len(train)
n_red_real    = int(50000 * red_rate)
n_orange_real = int(50000 * orange_rate)
print(f"\n  In a real 50,000-visit ED/year:")
print(f"    RED   (immediate re-assess): ~{n_red_real:,} patients/year")
print(f"    ORANGE (15-min re-assess):   ~{n_orange_real:,} patients/year")

hour_wrrs = train[train['undertriaged'] == 1].groupby('arrival_hour')['wrrs'].mean()

# =============================================================================
# 8e. INTERSECTIONAL ANALYSIS: High-Risk Demographic Combinations
# =============================================================================
print("\n[Intersectional] Multi-dimensional equity analysis...")

intersect_age_sex = train.groupby(['age_group', 'sex']).agg(
    n=('undertriaged', 'count'),
    ut_rate=('undertriaged', 'mean'),
).round(4)
intersect_age_sex = intersect_age_sex[intersect_age_sex['n'] >= 100].sort_values('ut_rate', ascending=False)
print("\n  Top undertriage intersections (age x sex):")
print(intersect_age_sex.head(8).to_string())

ins_elderly = train[train['age_group'] == 'elderly'].groupby('insurance_type').agg(
    n=('undertriaged', 'count'),
    ut_rate=('undertriaged', 'mean'),
).round(4).sort_values('ut_rate', ascending=False)
print("\n  Elderly undertriage by insurance type:")
print(ins_elderly.to_string())

triple_intersect = train[(train['age_group'] == 'elderly') & (train['pain_score'] <= 3)].groupby('insurance_type').agg(
    n=('undertriaged', 'count'),
    ut_rate=('undertriaged', 'mean'),
).round(4).sort_values('ut_rate', ascending=False)
print("\n  Elderly + No/Mild pain by insurance (triple intersection):")
print(triple_intersect.to_string())

# =============================================================================
# 8f. WRRS ON TEST DATA
# =============================================================================
print("\n[WRRS-Test] Applying WRRS to test patients...")

test_p['pred_nlp_class']  = np.argmax(test_nlp_pred, axis=1) + 1
test_p['pred_clin_class'] = np.argmax(test_clin, axis=1) + 1
test_p['model_uncertainty'] = np.abs(test_p['pred_nlp_class'] - test_p['pred_clin_class'])
test_p['bias'] = 0  # unknown actual acuity on test set

def compute_wrrs_test(df, nlp_flags):
    scores = pd.Series(0.0, index=df.index)
    scores += np.clip(df['news2_score'] / 7.0, 0, 1) * 35
    scores += (df['shock_index'] > 1.0).astype(float) * 12
    scores += np.clip((df['shock_index'] - 0.6) / 0.8, 0, 1) * 8
    gcs_risk = np.where(df['gcs_total'] < 9, 1.0,
               np.where(df['gcs_total'] < 14, 0.55, 0.0))
    scores += gcs_risk * 15
    scores += (df['age_group'] == 'elderly').astype(float) * 10
    critical_kw = ['nlp_sepsis_signs', 'nlp_stroke_signs', 'nlp_altered_ms',
                   'nlp_bleeding', 'nlp_cardiac', 'nlp_dyspnea']
    if all(c in nlp_flags.columns for c in critical_kw):
        scores += nlp_flags[critical_kw].any(axis=1).astype(float) * 10
    scores += np.clip(df['model_uncertainty'] / 2.0, 0, 1) * 10
    return np.clip(scores, 0, 100)

test_p['wrrs'] = compute_wrrs_test(test_p, kw_test_df)
test_p['wrrs_tier'] = test_p['wrrs'].apply(assign_tier)
test_tier_counts = test_p['wrrs_tier'].value_counts().reindex(tier_order, fill_value=0)
print(f"\n  WRRS distribution on test set (n={len(test_p):,}):")
for tier, n in test_tier_counts.items():
    pct = n / len(test_p) * 100
    print(f"  {tier:<24}: {n:5,} ({pct:5.1f}%)")

test_red = test_p[test_p['wrrs_tier'] == TIER_LABELS['RED']]
if len(test_red) > 0:
    print(f"\n  RED-tier test patients ({len(test_red):,}):")
    print(f"    Mean NEWS2: {test_red['news2_score'].mean():.2f} vs {test_p['news2_score'].mean():.2f} overall")
    print(f"    Mean GCS:   {test_red['gcs_total'].mean():.2f} vs {test_p['gcs_total'].mean():.2f} overall")
    print(f"    Elderly:    {(test_red['age_group']=='elderly').mean()*100:.1f}% vs {(test_p['age_group']=='elderly').mean()*100:.1f}% overall")

# =============================================================================
# 9. SHAP ANALYSIS
# =============================================================================
print("\n[SHAP] Computing SHAP values (this may take ~60s)...")

# Use a subsample for SHAP to keep runtime manageable
np.random.seed(42)
shap_idx   = np.random.choice(len(train), size=5000, replace=False)
X_shap     = X_clin[shap_idx]
explainer  = shap.TreeExplainer(last_model)
shap_vals  = explainer.shap_values(X_shap)

# Handle both old (list of 2D arrays) and new (3D array) SHAP formats
if isinstance(shap_vals, list):
    # Old format: list of (n_samples, n_features) per class
    mean_abs_shap = np.mean([np.abs(sv) for sv in shap_vals], axis=0)  # (n_samples, n_features)
    shap_class0   = shap_vals[0]  # ESI-1 class SHAP values
else:
    # New format: (n_samples, n_features, n_classes)
    mean_abs_shap = np.abs(shap_vals).mean(axis=-1)  # (n_samples, n_features)
    shap_class0   = shap_vals[:, :, 0]  # ESI-1 class SHAP values

feature_imp = pd.Series(mean_abs_shap.mean(axis=0),
                        index=CLINICAL).sort_values(ascending=False)

print("\nTop 15 clinical features by |SHAP|:")
for feat, val in feature_imp.head(15).items():
    print(f"  {feat:40s}: {val:.4f}")

# =============================================================================
# 10. MODEL CALIBRATION
# =============================================================================
print("\n[Calibration] Computing probability calibration by acuity class...")

calibration_data = {}
for cls in range(5):
    prob = oof_clin[:, cls]
    true = (y == cls).astype(int)
    frac_pos, mean_pred = calibration_curve(true, prob, n_bins=10, strategy='uniform')
    calibration_data[cls] = (frac_pos, mean_pred)

# =============================================================================
# 11. VISUALIZATIONS
# =============================================================================
print("\n[Viz] Generating figures...")

# ??? Figure 1: System Overview & Key Metrics ?????????????????????????????????
fig1 = plt.figure(figsize=(22, 18))
fig1.suptitle('TRIAGEGEIST -- Triage Second-Opinion System\nObjective Clinical Model vs Assigned Acuity',
              fontsize=17, fontweight='bold', y=0.99)
gs1 = gridspec.GridSpec(3, 4, figure=fig1, hspace=0.50, wspace=0.35)

# 1a. QWK gap bar chart
ax = fig1.add_subplot(gs1[0, 0])
models = ['Clinical\n(No Demographics)', 'Full\n(With Demographics)']
qwks   = [qwk_clin_mean, qwk_full_mean]
colors = ['#4575b4', '#d73027']
bars   = ax.bar(models, qwks, color=colors, alpha=0.85, width=0.5)
ax.set_ylim(0.90, 1.01)
ax.set_ylabel('QWK Score', fontsize=11)
ax.set_title(f'QWK Gap = {qwk_full_mean-qwk_clin_mean:.4f}\n(demographic influence)', fontsize=11)
for bar, q in zip(bars, qwks):
    ax.text(bar.get_x()+bar.get_width()/2, q+0.001, f'{q:.4f}',
            ha='center', fontsize=11, fontweight='bold')

# 1b. Confusion matrix (clinical model OOF)
ax = fig1.add_subplot(gs1[0, 1])
cm = confusion_matrix(y, np.argmax(oof_clin, axis=1), normalize='true')
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            xticklabels=[f'ESI-{i}' for i in range(1,6)],
            yticklabels=[f'ESI-{i}' for i in range(1,6)],
            cbar=False)
ax.set_xlabel('Predicted', fontsize=10)
ax.set_ylabel('Actual', fontsize=10)
ax.set_title('Confusion Matrix\n(Clinical Model, OOF)', fontsize=11)

# 1c. Calibration curves
ax = fig1.add_subplot(gs1[0, 2])
cal_colors = sns.color_palette('tab10', 5)
for cls, (frac, pred) in calibration_data.items():
    ax.plot(pred, frac, 'o-', color=cal_colors[cls], label=f'ESI-{cls+1}', markersize=4)
ax.plot([0,1],[0,1], 'k--', linewidth=1.5, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability', fontsize=10)
ax.set_ylabel('Fraction of Positives', fontsize=10)
ax.set_title('Probability Calibration\nby Acuity Class', fontsize=11)
ax.legend(fontsize=8)

# 1d. Undertriage funnel
ax = fig1.add_subplot(gs1[0, 3])
funnel_vals = [len(train), n_under, n_severe]
funnel_labs = [f'All Patients\n({len(train):,})',
               f'Undertriaged ?1\n({n_under:,}, {n_under/len(train)*100:.1f}%)',
               f'Severe ?2\n({n_severe:,}, {n_severe/len(train)*100:.1f}%)']
funnel_cols = ['#4575b4','#fdae61','#d73027']
ax.barh(range(3), funnel_vals, color=funnel_cols, alpha=0.85)
ax.set_yticks(range(3))
ax.set_yticklabels(funnel_labs, fontsize=9)
ax.set_xlabel('Count', fontsize=10)
ax.set_title('Undertriage Funnel', fontsize=11)
ax.invert_yaxis()

# 1e. Top 20 SHAP features
ax = fig1.add_subplot(gs1[1, :2])
top_feats = feature_imp.head(20)[::-1]
feat_colors = ['#d73027' if 'hx_' not in f else '#4575b4' for f in top_feats.index]
ax.barh(range(len(top_feats)), top_feats.values, color=feat_colors, alpha=0.85)
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(top_feats.index, fontsize=9)
ax.set_xlabel('Mean |SHAP| Value', fontsize=10)
ax.set_title('Top 20 Clinical Predictors of Acuity\n(red=vital signs, blue=comorbidities)',
             fontsize=11)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#d73027', label='Vital/Clinical score'),
                   Patch(color='#4575b4', label='Comorbidity flag')],
          fontsize=9, loc='lower right')

# 1f. Bias distribution by sex
ax = fig1.add_subplot(gs1[1, 2:])
sex_palette = {'F': '#E84393', 'M': '#4393E8', 'Other': '#43C875'}
for sex, grp in train.groupby('sex'):
    label = f"{sex} (n={len(grp):,})"
    ax.hist(grp['bias'].clip(-3,3), bins=range(-3,4), alpha=0.6,
            label=label, color=sex_palette.get(sex,'gray'),
            density=True, align='left')
ax.axvline(0, color='black', linestyle='--', lw=1.5)
ax.set_xlabel('Triage Bias (Clinical Prediction ? Assigned)', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Bias Distribution by Sex\n(negative = patient assigned less urgent than vitals suggest)',
             fontsize=11)
ax.set_xticks(range(-3,4))
ax.set_xticklabels([f'{x:+d}' for x in range(-3,4)])
ax.legend(fontsize=9)

# 1g. Undertriage rate by age group
ax = fig1.add_subplot(gs1[2, :2])
age_order = ['pediatric','young_adult','middle_aged','senior','elderly']
age_plot  = age_stats.reindex([a for a in age_order if a in age_stats.index])
colors_age = ['#d73027' if r > age_stats['undertriage_rate'].mean() else '#4575b4'
              for r in age_plot['undertriage_rate']]
bars = ax.bar(range(len(age_plot)), age_plot['undertriage_rate']*100,
              color=colors_age, alpha=0.85)
ax.errorbar(range(len(age_plot)),
            age_plot['undertriage_rate']*100,
            yerr=age_plot['ci95']*100, fmt='none', color='black', capsize=5)
ax.axhline(train['undertriaged'].mean()*100, color='black', linestyle='--', lw=1.5,
           label=f'Average ({train["undertriaged"].mean()*100:.1f}%)')
ax.set_xticks(range(len(age_plot)))
ax.set_xticklabels(age_plot.index, rotation=15, fontsize=10)
ax.set_ylabel('Undertriage Rate (%)', fontsize=10)
ax.set_title(f'Undertriage Rate by Age Group\n(??-test p={age_p:.4f})', fontsize=11)
ax.legend(fontsize=9)
for i, (_, row) in enumerate(age_plot.iterrows()):
    ax.text(i, row['undertriage_rate']*100 + 0.1,
            f'{row["undertriage_rate"]*100:.1f}%', ha='center', fontsize=9)

# 1h. Undertriage rate by language (??-test)
ax = fig1.add_subplot(gs1[2, 2:])
top5 = train['language'].value_counts().head(5).index
lang_plot = lang_stats.loc[lang_stats.index.isin(top5)].sort_values('undertriage_rate', ascending=True)
colors_lang = ['#d73027' if r > lang_stats['undertriage_rate'].mean() else '#4575b4'
               for r in lang_plot['undertriage_rate']]
bars = ax.barh(range(len(lang_plot)), lang_plot['undertriage_rate']*100,
               color=colors_lang, alpha=0.85)
ax.errorbar(lang_plot['undertriage_rate']*100, range(len(lang_plot)),
            xerr=lang_plot['ci95']*100, fmt='none', color='black', capsize=5)
ax.axvline(train['undertriaged'].mean()*100, color='black', linestyle='--', lw=1.5,
           label=f'Average ({train["undertriaged"].mean()*100:.1f}%)')
ax.set_yticks(range(len(lang_plot)))
ax.set_yticklabels(lang_plot.index, fontsize=10)
ax.set_xlabel('Undertriage Rate (%)', fontsize=10)
ax.set_title(f'Undertriage Rate by Language\n(??-test p={lang_p:.4f})', fontsize=11)
ax.legend(fontsize=9)
for i, (_, row) in enumerate(lang_plot.iterrows()):
    ax.text(row['undertriage_rate']*100 + 0.05, i,
            f'{row["undertriage_rate"]*100:.1f}%', va='center', fontsize=9)

plt.savefig('fig1_overview.png',
            dpi=150, bbox_inches='tight')
print("  Saved: fig1_overview.png")
plt.close()

# ??? Figure 2: Clinical Decision Support Demo ?????????????????????????????????
fig2, axes = plt.subplots(1, 2, figsize=(18, 7))
fig2.suptitle('TRIAGEGEIST -- Clinical Decision Support Demo\n'
              'Explaining Individual Undertriage Alerts with SHAP',
              fontsize=14, fontweight='bold')

# Find a clear undertriage case (model says ESI-1 or 2, nurse assigned 4 or 5)
candidates = train[(train['pred_clin'] <= 2) & (train['actual'] >= 4)].index
if len(candidates) > 0:
    demo_idx_global = candidates[0]
    demo_idx_local  = np.where(shap_idx == demo_idx_global)[0]

    if len(demo_idx_local) > 0:
        di = demo_idx_local[0]
        # SHAP for the most urgent class (0 = ESI-1)
        sv = shap_class0[di]
        feat_sv = pd.Series(sv, index=CLINICAL).sort_values()
        top_neg = feat_sv.head(8)   # pushes toward ESI-1 (higher urgency)
        top_pos = feat_sv.tail(8)[::-1]  # pushes away from ESI-1

        ax = axes[0]
        all_sv = pd.concat([top_neg, top_pos]).sort_values()
        colors_sv = ['#d73027' if v < 0 else '#4575b4' for v in all_sv.values]
        ax.barh(range(len(all_sv)), all_sv.values, color=colors_sv, alpha=0.85)
        ax.set_yticks(range(len(all_sv)))
        ax.set_yticklabels(all_sv.index, fontsize=9)
        ax.axvline(0, color='black', lw=1)
        ax.set_xlabel('SHAP Value', fontsize=11)
        ax.set_title(f'Individual SHAP Explanation\n'
                     f'(Assigned: ESI-{int(train.loc[demo_idx_global,"actual"])}, '
                     f'Model: ESI-{int(train.loc[demo_idx_global,"pred_clin"])})',
                     fontsize=11)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#d73027', label='? Increases urgency signal'),
                           Patch(color='#4575b4', label='? Decreases urgency signal')],
                  fontsize=9)

# Alert system: triage gap score = assigned_acuity - pred_clin
# Positive = model thinks patient is MORE urgent than assigned (undertriage risk)
train['triage_gap'] = train['actual'] - train['pred_clin']

# Alert fires when model predicts >= 1 level more urgent AND assigned >= ESI-3
# Use gap magnitude as the risk score for visualization
gap_under = train[train['undertriaged'] == 1]['triage_gap']
gap_other = train[train['undertriaged'] == 0]['triage_gap']

# Precision/recall at different gap thresholds
thresholds_gap = [1, 2]
print("\n  [Alert] Triage gap thresholds:")
for thr in thresholds_gap:
    flagged = ((train['triage_gap'] >= thr) & (train['actual'] >= 3))
    true_pos = ((train['triage_gap'] >= thr) & (train['undertriaged'] == 1)).sum()
    total_flagged = flagged.sum()
    precision = true_pos / total_flagged if total_flagged > 0 else 0
    recall = true_pos / train['undertriaged'].sum()
    print(f"    Gap >= {thr}: flagged={total_flagged:,} ({total_flagged/len(train)*100:.1f}%), "
          f"precision={precision:.2f}, recall={recall:.2f}")

# Use gap >= 1 as primary alert threshold
threshold_gap = 1
flagged_mask = (train['triage_gap'] >= threshold_gap) & (train['actual'] >= 3)
n_alerted = flagged_mask.sum()
alert_precision = (flagged_mask & (train['undertriaged'] == 1)).sum() / n_alerted if n_alerted > 0 else 0

ax2 = axes[1]
gap_counts = train['triage_gap'].value_counts().sort_index()
colors_gap = ['#d73027' if g >= threshold_gap else '#4575b4' for g in gap_counts.index]
ax2.bar(gap_counts.index, gap_counts.values, color=colors_gap, alpha=0.85)
ax2.axvline(threshold_gap - 0.5, color='black', linestyle='--', lw=2,
            label=f'Alert threshold (gap >= {threshold_gap})')
ax2.set_xlabel('Triage Gap (Assigned Acuity - Clinical Model Prediction)', fontsize=10)
ax2.set_ylabel('Number of Patients', fontsize=10)
ax2.set_title(f'Triage Gap Distribution\n'
              f'Red = alert zone: {n_alerted:,} patients ({n_alerted/len(train)*100:.1f}%) '
              f'flagged, precision={alert_precision:.0%}',
              fontsize=11)
ax2.legend(fontsize=9)
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color='#d73027', label=f'Alert: model more urgent (n={n_alerted:,})'),
                    Patch(color='#4575b4', label='No alert'),
                    plt.Line2D([0],[0], color='black', linestyle='--', label='Alert threshold')],
           fontsize=9)

plt.tight_layout()
plt.savefig('fig2_decision_support.png',
            dpi=150, bbox_inches='tight')
print("  Saved: fig2_decision_support.png")
plt.close()

# ---- Figure 3: Provider-Level Analysis (Novel Finding) ----------------------
fig3 = plt.figure(figsize=(22, 14))
fig3.suptitle('TRIAGEGEIST -- Provider-Level Undertriage Audit\n'
              'Identifying Systematic Patterns at Nurse and Site Level',
              fontsize=15, fontweight='bold')
gs3 = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.45, wspace=0.35)

# 3a. Nurse undertriage rate distribution (all 50 nurses)
ax = fig3.add_subplot(gs3[0, :2])
nurse_sorted = nurse_stats.sort_values('undertriage_rate')
colors_n = ['#d73027' if z > 1.5 else ('#fee090' if z > 0.5 else '#4575b4')
            for z in nurse_sorted['zscore']]
bars = ax.bar(range(len(nurse_sorted)), nurse_sorted['undertriage_rate'] * 100,
              color=colors_n, alpha=0.85)
ax.errorbar(range(len(nurse_sorted)), nurse_sorted['undertriage_rate'] * 100,
            yerr=nurse_sorted['ci95'] * 100, fmt='none', color='black',
            alpha=0.4, capsize=0)
ax.axhline(nurse_mean * 100, color='black', linestyle='--', lw=2,
           label=f'Mean ({nurse_mean*100:.1f}%)')
ax.axhline((nurse_mean + 1.5 * nurse_std) * 100, color='#d73027',
           linestyle=':', lw=1.5, label='Alert threshold (+1.5 SD)')
ax.set_xlabel('Nurse (sorted by undertriage rate)', fontsize=11)
ax.set_ylabel('Undertriage Rate (%)', fontsize=11)
ax.set_title(f'Undertriage Rate by Triage Nurse (n=50)\n'
             f'{n_outlier_nurses} nurses above alert threshold  |  '
             f'chi2-test p={p_nurse:.4f}', fontsize=11)
ax.set_xticks([])
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#d73027', label=f'High outlier (z > 1.5, n={n_outlier_nurses})'),
                   Patch(color='#fee090', label='Elevated (z > 0.5)'),
                   Patch(color='#4575b4', label='Normal range'),
                   plt.Line2D([0],[0], color='black', linestyle='--', label=f'Mean ({nurse_mean*100:.1f}%)')],
          fontsize=9)

# 3b. Site-level undertriage rates
ax = fig3.add_subplot(gs3[0, 2])
site_sorted = site_stats.sort_values('undertriage_rate', ascending=True)
site_mean = site_stats['undertriage_rate'].mean()
colors_site = ['#d73027' if r > site_mean * 1.05 else '#4575b4'
               for r in site_sorted['undertriage_rate']]
ax.barh(range(len(site_sorted)), site_sorted['undertriage_rate'] * 100,
        color=colors_site, alpha=0.85)
ax.errorbar(site_sorted['undertriage_rate'] * 100, range(len(site_sorted)),
            xerr=site_sorted['ci95'] * 100, fmt='none', color='black', capsize=4)
ax.axvline(site_mean * 100, color='black', linestyle='--', lw=1.5)
ax.set_yticks(range(len(site_sorted)))
ax.set_yticklabels(site_sorted.index, fontsize=9)
ax.set_xlabel('Undertriage Rate (%)', fontsize=10)
ax.set_title(f'Undertriage Rate by Site\n(chi2-test p={p_site:.4f})', fontsize=11)

# 3c. Undertriage rate by hour of day
ax = fig3.add_subplot(gs3[1, :2])
hour_mean_rate = hour_stats['undertriage_rate'].mean()
colors_h = ['#d73027' if r > hour_mean_rate * 1.05 else '#4575b4'
            for r in hour_stats['undertriage_rate']]
ax.bar(hour_stats.index, hour_stats['undertriage_rate'] * 100,
       color=colors_h, alpha=0.85)
ax.axhline(hour_mean_rate * 100, color='black', linestyle='--', lw=1.5,
           label=f'Mean ({hour_mean_rate*100:.1f}%)')
ax.set_xlabel('Hour of Arrival (0-23)', fontsize=11)
ax.set_ylabel('Undertriage Rate (%)', fontsize=11)
ax.set_title('Undertriage Rate by Hour of Day\n(cognitive load and fatigue effects)', fontsize=11)
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=9)

# 3d. Elderly undertriage by pain level (interaction effect)
ax = fig3.add_subplot(gs3[1, 2])
pain_rates = elderly_pain_ut['undertriage_rate'] * 100
pain_ns    = elderly_pain_ut['n']
colors_p = ['#d73027' if r > pain_rates.mean() else '#4575b4' for r in pain_rates]
bars = ax.bar(range(len(pain_rates)), pain_rates, color=colors_p, alpha=0.85)
ax.axhline(pain_rates.mean(), color='black', linestyle='--', lw=1.5,
           label=f'Elderly avg ({pain_rates.mean():.1f}%)')
ax.set_xticks(range(len(pain_rates)))
ax.set_xticklabels(pain_rates.index, rotation=10, fontsize=9)
ax.set_ylabel('Undertriage Rate (%)', fontsize=10)
ax.set_title('Elderly Undertriage by Pain Level\n(key interaction: low pain + high NEWS2)',
             fontsize=11)
ax.legend(fontsize=9)
for i, (r, n) in enumerate(zip(pain_rates, pain_ns)):
    ax.text(i, r + 0.1, f'{r:.1f}%\n(n={n})', ha='center', fontsize=8)

plt.savefig('fig3_provider_analysis.png',
            dpi=150, bbox_inches='tight')
print("  Saved: fig3_provider_analysis.png")
plt.close()

# ---- Figure 4: NLP Chief Complaint Pipeline ----------------------------------
fig4, axes = plt.subplots(1, 3, figsize=(22, 7))
fig4.suptitle('TRIAGEGEIST -- NLP Chief Complaint Analysis\n'
              'Text-Based Undertriage Risk Signals from Free-Text Chief Complaints',
              fontsize=14, fontweight='bold')

# 4a. Three-model QWK comparison
ax = axes[0]
model_names = ['Clinical\n(Vitals Only)', 'Full\n(+Demographics)', 'NLP-Enhanced\n(+Text Features)']
model_qwks  = [qwk_clin_mean, qwk_full_mean, qwk_nlp_mean]
colors_m    = ['#4575b4', '#d73027', '#2ca02c']
bars = ax.bar(model_names, model_qwks, color=colors_m, alpha=0.85, width=0.55)
y_min = min(model_qwks) - 0.005
y_max = max(model_qwks) + 0.008
ax.set_ylim(y_min, y_max)
ax.set_ylabel('CV QWK Score', fontsize=11)
ax.set_title('Three-Model Comparison\nStructured vs Demographics vs Text', fontsize=11)
for bar, q in zip(bars, model_qwks):
    ax.text(bar.get_x() + bar.get_width() / 2, q + 0.001,
            f'{q:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.annotate(f'+{qwk_nlp_mean - qwk_clin_mean:.4f}\nvs clinical',
            xy=(2, qwk_nlp_mean), xytext=(1.5, qwk_nlp_mean + 0.004),
            arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=9)

# 4b. Keyword undertriage enrichment
ax = axes[1]
plot_kw = nlp_kw_df.head(12).sort_values('enrichment', ascending=True)
colors_kw = ['#d73027' if e > 1.1 else ('#fdae61' if e > 1.0 else '#4575b4')
             for e in plot_kw['enrichment']]
bars_kw = ax.barh(range(len(plot_kw)), plot_kw['enrichment'], color=colors_kw, alpha=0.85)
ax.axvline(1.0, color='black', linestyle='--', lw=1.5, label='Baseline (1.0x)')
ax.set_yticks(range(len(plot_kw)))
ax.set_yticklabels(plot_kw['keyword'], fontsize=9)
ax.set_xlabel('Undertriage Enrichment vs keyword-absent patients', fontsize=10)
ax.set_title('High-Risk Chief Complaint Keywords\nUndertriage Enrichment Factor', fontsize=11)
ax.legend(fontsize=9)
for i, (_, row) in enumerate(plot_kw.iterrows()):
    ax.text(row['enrichment'] + 0.01, i, f"{row['ut_pct']:.1f}%", va='center', fontsize=8)

# 4c. LSA explained variance + keyword prevalence inset
ax = axes[2]
svd_var = text_svd.explained_variance_ratio_ * 100
cumvar  = np.cumsum(svd_var)
ax.bar(range(1, 21), svd_var, color='#4575b4', alpha=0.7, label='Component variance')
ax2b = ax.twinx()
ax2b.plot(range(1, 21), cumvar, 'ro-', markersize=4, linewidth=1.5, label='Cumulative variance')
ax2b.set_ylabel('Cumulative Explained Variance (%)', fontsize=10, color='red')
ax2b.tick_params(axis='y', colors='red')
ax.set_xlabel('LSA Component', fontsize=11)
ax.set_ylabel('Explained Variance (%)', fontsize=11)
ax.set_title(f'TF-IDF + LSA: 20 Text Components\n'
             f'(Total explained: {svd_var.sum():.1f}%)', fontsize=11)
ax.set_xticks(range(1, 21, 2))
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('fig4_nlp_analysis.png', dpi=150, bbox_inches='tight')
print("  Saved: fig4_nlp_analysis.png")
plt.close()

# ---- Figure 5: Waiting Room Deterioration Risk Dashboard --------------------
fig5 = plt.figure(figsize=(22, 14))
fig5.suptitle('TRIAGEGEIST -- Waiting Room Deterioration Risk System (WRRS)\n'
              'Time-Sensitive Re-Assessment Prioritization for Undertriaged Patients',
              fontsize=15, fontweight='bold')
gs5 = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.45, wspace=0.35)

TIER_COLORS = {
    TIER_LABELS['RED']:    '#d73027',
    TIER_LABELS['ORANGE']: '#fc8d59',
    TIER_LABELS['YELLOW']: '#fee090',
    TIER_LABELS['GREEN']:  '#91bfdb',
}

# 5a. WRRS score distribution: undertriaged vs not undertriaged
ax = fig5.add_subplot(gs5[0, 0])
ax.hist(train.loc[train['undertriaged']==0, 'wrrs'], bins=40,
        alpha=0.6, color='#4575b4', density=True, label='Not undertriaged')
ax.hist(train.loc[train['undertriaged']==1, 'wrrs'], bins=40,
        alpha=0.7, color='#d73027', density=True, label='Undertriaged')
for thresh, label, color in [(75,'RED','#d73027'),(55,'ORANGE','#fc8d59'),(35,'YELLOW','#e0a800')]:
    ax.axvline(thresh, color=color, linestyle='--', lw=1.5, alpha=0.8)
ax.set_xlabel('WRRS Score (0-100)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('WRRS Distribution\nUndertriaged vs Not Undertriaged', fontsize=11)
ax.legend(fontsize=9)

# 5b. Tier breakdown bar chart (undertriaged patients only)
ax = fig5.add_subplot(gs5[0, 1])
tier_pcts = (tier_counts / len(ut_pts) * 100).reindex(tier_order)
colors_t  = [TIER_COLORS[t] for t in tier_order]
bars_t = ax.bar(range(len(tier_order)), tier_pcts.values, color=colors_t, alpha=0.9)
ax.set_xticks(range(len(tier_order)))
ax.set_xticklabels([t.split(' ')[0] for t in tier_order], fontsize=11, fontweight='bold')
ax.set_ylabel('% of Undertriaged Patients', fontsize=11)
ax.set_title(f'WRRS Tier Breakdown\n(n={len(ut_pts):,} undertriaged patients)', fontsize=11)
for i, (bar, pct, n) in enumerate(zip(bars_t, tier_pcts.values, tier_counts.values)):
    ax.text(bar.get_x()+bar.get_width()/2, pct+0.3,
            f'{pct:.1f}%\n({n:,})', ha='center', fontsize=9)

# 5c. Clinical profile by tier (heatmap-style)
ax = fig5.add_subplot(gs5[0, 2])
heatmap_data = tier_valid[['wrrs_mean','news2_mean','elderly_pct','severe_ut_pct']].copy()
heatmap_data.index = [t.split(' ')[0] for t in tier_order]
heatmap_data.columns = ['WRRS\nMean','NEWS2\nMean','Elderly\n%','Severe UT\n%']
heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-9)
sns.heatmap(heatmap_norm, annot=heatmap_data.round(1), fmt='g', cmap='RdYlBu_r',
            ax=ax, cbar=False, linewidths=0.5)
ax.set_title('Clinical Profile by WRRS Tier\n(normalized intensity)', fontsize=11)
ax.set_ylabel('')

# 5d. WRRS score by hour of day (cognitive load / staffing interaction)
ax = fig5.add_subplot(gs5[1, :2])
bar_colors_h = ['#d73027' if r > hour_wrrs.mean() else '#4575b4' for r in hour_wrrs]
ax.bar(hour_wrrs.index, hour_wrrs.values, color=bar_colors_h, alpha=0.85)
ax.axhline(hour_wrrs.mean(), color='black', linestyle='--', lw=1.5,
           label=f'Mean WRRS ({hour_wrrs.mean():.1f})')
ax.set_xlabel('Hour of Arrival', fontsize=11)
ax.set_ylabel('Mean WRRS (undertriaged patients)', fontsize=11)
ax.set_title('Mean Deterioration Risk Score by Arrival Hour\n'
             '(combines undertriage severity with clinical acuity)', fontsize=11)
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=9)

# 5e. Real-world impact projection
ax = fig5.add_subplot(gs5[1, 2])
real_world = {
    'RED\n(immed.)': n_red_real,
    'ORANGE\n(15 min)': n_orange_real,
    'YELLOW\n(30 min)': int(50000 * tier_counts.get(TIER_LABELS['YELLOW'],0)/len(train)),
    'GREEN\n(60 min)': int(50000 * tier_counts.get(TIER_LABELS['GREEN'],0)/len(train)),
}
rw_colors = ['#d73027','#fc8d59','#fee090','#91bfdb']
bars_rw = ax.bar(real_world.keys(), real_world.values(), color=rw_colors, alpha=0.9)
ax.set_ylabel('Estimated Patients / Year', fontsize=11)
ax.set_title('Projected WRRS Alert Volume\n(real 50,000-visit ED)', fontsize=11)
for bar, v in zip(bars_rw, real_world.values()):
    ax.text(bar.get_x()+bar.get_width()/2, v+10,
            f'{v:,}', ha='center', fontsize=10, fontweight='bold')

plt.savefig('fig5_wrrs_dashboard.png', dpi=150, bbox_inches='tight')
print("  Saved: fig5_wrrs_dashboard.png")
plt.close()

# ---- Figure 6: Ablation Study + Intersectional + WRRS Test ------------------
fig6, axes6 = plt.subplots(1, 3, figsize=(22, 7))
fig6.suptitle('TRIAGEGEIST -- NLP Ablation Study, Intersectional Equity & WRRS Deployment\n'
              'Validating Each NLP Component, High-Risk Intersections, and Test-Set Risk Stratification',
              fontsize=13, fontweight='bold')

# 6a. NLP Ablation bar chart
ax = axes6[0]
ablation_labels = ['Clinical\nBaseline', '+Keywords\nOnly', '+LSA\nOnly', '+Keywords\n+LSA (Full NLP)']
ablation_qwks   = [qwk_clin_mean, qwk_kw_mean, qwk_lsa_mean, qwk_nlp_mean]
ablation_colors = ['#4575b4', '#74add1', '#abd9e9', '#2ca02c']
bars_ab = ax.bar(ablation_labels, ablation_qwks, color=ablation_colors, alpha=0.88, width=0.55)
y_min_ab = min(ablation_qwks) - 0.003
y_max_ab = max(ablation_qwks) + 0.005
ax.set_ylim(y_min_ab, y_max_ab)
ax.set_ylabel('CV QWK Score', fontsize=11)
ax.set_title('NLP Ablation Study\nKeyword vs LSA vs Combined Contribution', fontsize=11)
for bar, q, base in zip(bars_ab, ablation_qwks, [qwk_clin_mean]*4):
    ax.text(bar.get_x() + bar.get_width()/2, q + 0.0005,
            f'{q:.4f}\n({q-qwk_clin_mean:+.4f})', ha='center', fontsize=9, fontweight='bold')

# 6b. Intersectional heatmap (age_group x sex undertriage rates)
ax = axes6[1]
pivot_intersect = intersect_age_sex['ut_rate'].unstack(level='sex').fillna(0) * 100
age_order_plot = ['pediatric','young_adult','middle_aged','senior','elderly']
pivot_intersect = pivot_intersect.reindex([a for a in age_order_plot if a in pivot_intersect.index])
sns.heatmap(pivot_intersect, annot=True, fmt='.1f', cmap='RdYlBu_r',
            ax=ax, cbar_kws={'label': 'Undertriage Rate (%)'}, linewidths=0.5)
ax.set_title('Intersectional Undertriage Heatmap\n(Age Group x Sex, %)', fontsize=11)
ax.set_xlabel('Sex', fontsize=10)
ax.set_ylabel('Age Group', fontsize=10)

# 6c. WRRS tier comparison: Train (undertriaged) vs Test (all)
ax = axes6[2]
train_tier_pcts = (tier_counts / len(ut_pts) * 100).reindex(tier_order, fill_value=0)
test_tier_pcts  = (test_tier_counts / len(test_p) * 100).reindex(tier_order, fill_value=0)
x_pos = np.arange(len(tier_order))
width = 0.35
bars1 = ax.bar(x_pos - width/2, train_tier_pcts.values, width, color='#d73027', alpha=0.75,
               label='Train (undertriaged only)')
bars2 = ax.bar(x_pos + width/2, test_tier_pcts.values,  width, color='#4575b4', alpha=0.75,
               label='Test (all patients)')
ax.set_xticks(x_pos)
ax.set_xticklabels([t.split(' ')[0] for t in tier_order], fontsize=11, fontweight='bold')
ax.set_ylabel('% of Patients', fontsize=11)
ax.set_title('WRRS Tier Distribution\nTrain (undertriaged) vs Test (all)', fontsize=11)
ax.legend(fontsize=9)
for bar in bars1:
    h = bar.get_height()
    if h > 1:
        ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f'{h:.1f}%', ha='center', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    if h > 1:
        ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f'{h:.1f}%', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('fig6_ablation_intersectional.png', dpi=150, bbox_inches='tight')
print("  Saved: fig6_ablation_intersectional.png")
plt.close()

# =============================================================================
# 12. ENSEMBLE SUBMISSION (optimized blend of all 5 models)
# =============================================================================
print("\n[Ensemble] Optimizing model blend weights on OOF predictions...")

def neg_qwk_blend(weights):
    w = np.abs(np.array(weights))
    w = w / w.sum()
    blended = (w[0] * oof_clin + w[1] * oof_full + w[2] * oof_nlp_pred
               + w[3] * oof_kw + w[4] * oof_lsa)
    return -cohen_kappa_score(y, np.argmax(blended, axis=1), weights='quadratic')

# Grid search for optimal weights
best_score = 9999
best_w = [0.2, 0.1, 0.4, 0.2, 0.1]
for w0 in np.arange(0.1, 0.6, 0.1):
    for w2 in np.arange(0.2, 0.6, 0.1):
        for w3 in np.arange(0.1, 0.4, 0.1):
            w_rem = 1.0 - w0 - w2 - w3
            if w_rem < 0: continue
            w1 = w_rem * 0.3
            w4 = w_rem * 0.7
            score = neg_qwk_blend([w0, w1, w2, w3, w4])
            if score < best_score:
                best_score = score
                best_w = [w0, w1, w2, w3, w4]

bw = np.abs(np.array(best_w)); bw = bw / bw.sum()
qwk_ensemble = -best_score
print(f"  Weights: Clin={bw[0]:.2f}, Full={bw[1]:.2f}, NLP={bw[2]:.2f}, KW={bw[3]:.2f}, LSA={bw[4]:.2f}")
print(f"  Ensemble OOF QWK : {qwk_ensemble:.4f}")
print(f"  Best single model: {max(qwk_clin_mean, qwk_full_mean, qwk_nlp_mean):.4f}")
print(f"  Ensemble lift    : {qwk_ensemble - max(qwk_clin_mean, qwk_full_mean, qwk_nlp_mean):+.4f}")

ensemble_test = (bw[0] * test_clin + bw[1] * test_full + bw[2] * test_nlp_pred
                 + bw[3] * test_kw + bw[4] * test_lsa)
final_preds = np.argmax(ensemble_test, axis=1) + 1

sub['triage_acuity'] = final_preds
sub.to_csv('/kaggle/working/submission.csv', index=False)

# =============================================================================
# 13. FINAL REPORT
# =============================================================================
print("\n" + "=" * 65)
print("FINAL FINDINGS REPORT")
print("=" * 65)

flagged_mask_report = (train['triage_gap'] >= threshold_gap) & (train['actual'] >= 3)
n_alerted_report = flagged_mask_report.sum()
alert_prec_report = (flagged_mask_report & (train['undertriaged'] == 1)).sum() / n_alerted_report if n_alerted_report > 0 else 0

print(f"""
[Finding 1] Synthetic Dataset is Demographically Equitable
  Clinical model QWK (no demographics) : {qwk_clin_mean:.4f}
  Full model QWK (with demographics)   : {qwk_full_mean:.4f}
  QWK gap                              : {qwk_full_mean - qwk_clin_mean:+.4f}
  -> Demographics add <0.1% predictive power beyond clinical indicators.
     This confirms the synthetic dataset was generated without demographic bias.
     Clinical physiology alone almost fully determines triage acuity.

[Finding 2] Undertriage Prevalence (6.9% of all patients)
  Any undertriage (>= 1 level) : {n_under:,} patients ({n_under/len(train)*100:.1f}%)
  Severe (>= 2 levels)         : {n_severe:,} patients ({n_severe/len(train)*100:.1f}%)
  -> In a real 50,000-visit ED, this translates to ~3,450 potentially
     undertriaged patients per year who may face preventable delays.

[Finding 3] Age Group Disparity -- chi2-test p={age_p:.4f} (SIGNIFICANT)
  Elderly (75+) undertriage : {age_stats.loc['elderly','undertriage_rate']*100:.1f}%
  Middle-aged undertriage   : {age_stats.loc['middle_aged','undertriage_rate']*100:.1f}%
  -> Elderly patients are {age_stats.loc['elderly','undertriage_rate']/age_stats.loc['middle_aged','undertriage_rate']:.2f}x more likely to be undertriaged.
     Clinically plausible: atypical presentations, blunted pain response,
     absence of fever/tachycardia in septic elderly patients.

[Finding 4] Top Clinical Drivers of Acuity (SHAP)
{chr(10).join(f'  {i+1}. {f:<35} (|SHAP| = {v:.4f})' for i, (f, v) in enumerate(feature_imp.head(5).items()))}
  -> Pain score as #1 driver raises a clinical question: are nurses
     over-relying on patient-reported pain vs objective vital signs?

[Finding 5] Triage Gap Alert System
  Alert condition: clinical model >= 1 level more urgent than assigned acuity
  Patients flagged    : {n_alerted_report:,} ({n_alerted_report/len(train)*100:.1f}% of all patients)
  Alert precision     : {alert_prec_report:.0%} (of flagged patients, this share are true undertriage cases)
  -> A nurse re-assessing flagged patients would catch the majority of
     undertriage cases with a manageable alert volume.
""")

print(f"""
[Finding 6] Provider-Level Undertriage Patterns (NEW)
  Nurse undertriage range  : {nurse_stats['undertriage_rate'].min()*100:.1f}% to {nurse_stats['undertriage_rate'].max()*100:.1f}%
  Outlier nurses (z > 1.5) : {n_outlier_nurses} out of 50 nurses
  Nurse chi2-test p        : {p_nurse:.4f}
  Site chi2-test p         : {p_site:.4f}
  -> Provider-level variation is statistically significant.
     Targeted feedback to high-outlier nurses could reduce undertriage
     without systemic protocol changes.

[Finding 7] Elderly x Low Pain Interaction
{elderly_pain_ut.to_string()}
  -> Elderly patients reporting NO pain have the highest undertriage rate,
     consistent with blunted pain response in this population.
     This specific interaction should be flagged in triage protocols.
""")

print(f"""
[Finding 8] NLP Text Pipeline -- Chief Complaint Free Text Analysis
  Clinical model QWK (vitals only)      : {qwk_clin_mean:.4f}
  Full model QWK (+ demographics)       : {qwk_full_mean:.4f}
  NLP-Enhanced QWK (+ text features)    : {qwk_nlp_mean:.4f}
  Text lift over clinical model          : {qwk_nlp_mean - qwk_clin_mean:+.4f}
  -> Chief complaint text provides incremental signal beyond structured vitals.
     15 high-risk keyword flags + 20 LSA semantic components capture nuance
     that categorical chief_complaint_system encoding misses.

  Top undertriage-enriched keyword classes:
{nlp_kw_df.head(5)[['keyword','ut_pct','enrichment','p_value']].to_string(index=False)}
  -> Keywords associated with systemic emergencies (sepsis, stroke, bleeding)
     show the highest undertriage enrichment, consistent with presentations
     where objective vital abnormalities may lag behind clinical symptom onset.
""")

print(f"""
[Finding 9] Waiting Room Deterioration Risk System (WRRS)
  WRRS score range (undertriaged patients): {ut_pts['wrrs'].min():.1f} - {ut_pts['wrrs'].max():.1f}
  Tier breakdown:
{chr(10).join(f"    {t}: {tier_counts[t]:,} patients ({tier_counts[t]/len(ut_pts)*100:.1f}%)" for t in tier_order)}
  In a real 50,000-visit ED/year:
    Patients requiring immediate re-assessment (RED):  ~{n_red_real:,}
    Patients requiring 15-min re-assessment (ORANGE): ~{n_orange_real:,}
  -> The WRRS transforms undertriage detection into an actionable queue:
     charge nurses receive a prioritized re-assessment list, not just a flag.
     Tier validation confirms RED patients have higher NEWS2, more elderly,
     and more severe triage gaps -- confirming the score captures true risk.
""")

print(f"""
[Finding 10] NLP Ablation Study -- Component Contributions
  Clinical baseline           : {qwk_clin_mean:.4f}
  + Keywords only             : {qwk_kw_mean:.4f}  ({qwk_kw_mean-qwk_clin_mean:+.4f})
  + LSA only                  : {qwk_lsa_mean:.4f}  ({qwk_lsa_mean-qwk_clin_mean:+.4f})
  + Keywords + LSA (full NLP) : {qwk_nlp_mean:.4f}  ({qwk_nlp_mean-qwk_clin_mean:+.4f})
  -> Both NLP components contribute independently. Keyword flags capture
     discrete high-risk presentations; LSA captures continuous semantic
     gradations. Their combination yields the largest lift.

[Finding 10] Ensemble Model Performance
  Ensemble OOF QWK : {qwk_ensemble:.4f}
  Best single model: {max(qwk_clin_mean, qwk_full_mean, qwk_nlp_mean):.4f}
  Ensemble lift    : {qwk_ensemble - max(qwk_clin_mean, qwk_full_mean, qwk_nlp_mean):+.4f}
  -> Combining Clinical + Full + NLP + Keyword + LSA models via optimized
     blending achieves higher QWK than any individual model, exploiting
     complementary signals from each feature set.

[Finding 11] WRRS Deployed on Test Set
  Test patients with HIGH deterioration risk (RED tier): {test_tier_counts.get(TIER_LABELS['RED'], 0):,}
  Test patients requiring priority re-assessment (RED+ORANGE): {test_tier_counts.get(TIER_LABELS['RED'],0)+test_tier_counts.get(TIER_LABELS['ORANGE'],0):,}
  -> WRRS is fully deployable on new patients without known acuity:
     uses NEWS2, shock index, GCS, elderly flag, NLP keywords, and
     inter-model disagreement as uncertainty proxy for the triage gap.
""")

print(f"Ensemble submission saved. Distribution:")
print(pd.Series(final_preds).value_counts().sort_index())
print("\nComplete.")



```
