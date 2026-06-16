#!/usr/bin/env python3
# HGB + text ensemble for Triagegeist (fixed for Kaggle environment)
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import ComplementNB
import warnings, os, re
warnings.filterwarnings('ignore')

SEED = 42
ID_COL = 'patient_id'
TARGET_COL = 'triage_acuity'
TEXT_COL = 'chief_complaint_raw'
INPUT_DIR = '/kaggle/input/competitions/triagegeist'
HIGH_RISK_THRESHOLD = 2
STRUCTURED_WEIGHT = 0.80

print("Loading data...")
train = pd.read_csv(f'{INPUT_DIR}/train.csv', low_memory=False)
test = pd.read_csv(f'{INPUT_DIR}/test.csv', low_memory=False)
cc = pd.read_csv(f'{INPUT_DIR}/chief_complaints.csv')
ph = pd.read_csv(f'{INPUT_DIR}/patient_history.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

train = train.merge(cc, on=ID_COL, how='left').merge(ph, on=ID_COL, how='left')
test = test.merge(cc, on=ID_COL, how='left').merge(ph, on=ID_COL, how='left')
train[TEXT_COL] = train[TEXT_COL].fillna('')
test[TEXT_COL] = test[TEXT_COL].fillna('')

def engineer_features(df):
    df = df.copy()
    df['age_group'] = pd.cut(df['age'], bins=[0,18,35,50,65,80,200],
                             labels=['0-18','19-35','36-50','51-65','66-80','80+'])
    df['bp_map'] = df['diastolic_bp'] + (df['systolic_bp']-df['diastolic_bp'])/3
    df['shock_index'] = df['heart_rate'] / df['systolic_bp'].replace(0,1)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['hr_resp_ratio'] = df['heart_rate'] / df['respiratory_rate'].replace(0,1)
    df['pain_unrecorded'] = (df['pain_score'] == -1).astype(int)
    df['flag_low_oxygen'] = (df['spo2'] < 92).astype(int)
    df['flag_fever'] = (df['temperature_c'] >= 38.0).astype(int)
    df['flag_tachycardia'] = (df['heart_rate'] >= 100).astype(int)
    df['flag_tachypnea'] = (df['respiratory_rate'] >= 22).astype(int)
    df['flag_hypotension'] = (df['systolic_bp'] < 90).astype(int)
    df['flag_gcs_abnormal'] = (df['gcs_total'] < 15).astype(int)
    df['flag_high_news2'] = (df['news2_score'] >= 5).astype(int)
    df['hypotensive'] = (df['systolic_bp'] < 90).astype(int)
    df['tachycardic'] = (df['heart_rate'] > 100).astype(int)
    df['tachypneic'] = (df['respiratory_rate'] > 20).astype(int)
    df['abnormal_count'] = df[['hypotensive','tachycardic','tachypneic']].sum(axis=1)
    df['arrival_hour_sin'] = np.sin(2*np.pi*df['arrival_hour']/24)
    df['arrival_hour_cos'] = np.cos(2*np.pi*df['arrival_hour']/24)
    df['night_arrival'] = ((df['arrival_hour']>=22)|(df['arrival_hour']<6)).astype(int)
    complaint = df[TEXT_COL].fillna("").astype(str).str.lower()
    kw = {'kw_chest_pain':r'chest pain|thoracic pain|crushing chest',
          'kw_stroke_neuro':r'stroke|seizure|thunderclap|loss of vision|weakness|aphasia',
          'kw_respiratory':r'shortness of breath|asthma|hypoxia|wheeze|near-drowning',
          'kw_trauma':r'trauma|fracture|haemothorax|stab|wound|fall|injury',
          'kw_overdose':r'overdose|poison|toxic|substance',
          'kw_bleeding':r'bleed|haemorrhage|melena|hematemesis',
          'kw_pregnancy':r'pregnan|ectopic|postpartum|miscarriage',
          'kw_infection':r'sepsis|fever|necrotising|infection|cellulitis'}
    for name, pat in kw.items():
        df[name] = complaint.str.contains(pat, flags=re.IGNORECASE).astype(int)
    cardio = ['hx_hypertension','hx_heart_failure','hx_atrial_fibrillation',
              'hx_coronary_artery_disease','hx_peripheral_vascular_disease','hx_stroke_prior']
    resp = ['hx_asthma','hx_copd']
    neuro = ['hx_dementia','hx_epilepsy','hx_stroke_prior']
    frailty = ['hx_dementia','hx_ckd','hx_malignancy','hx_immunosuppressed']
    df['cardio_burden'] = df[[c for c in cardio if c in df.columns]].sum(axis=1)
    df['respiratory_burden'] = df[[c for c in resp if c in df.columns]].sum(axis=1)
    df['neuro_burden'] = df[[c for c in neuro if c in df.columns]].sum(axis=1)
    df['frailty_burden'] = df[[c for c in frailty if c in df.columns]].sum(axis=1)
    return df

print("Engineering features...")
train_fe = engineer_features(train)
test_fe = engineer_features(test)
for col in ['disposition','ed_los_hours']:
    if col in train_fe.columns: train_fe.drop(columns=[col], inplace=True)
    if col in test_fe.columns: test_fe.drop(columns=[col], inplace=True)

excluded = {ID_COL, TARGET_COL, TEXT_COL}
numeric_cols = [c for c in train_fe.columns if c not in excluded and pd.api.types.is_numeric_dtype(train_fe[c])]
cat_cols = [c for c in train_fe.columns if c not in excluded and not pd.api.types.is_numeric_dtype(train_fe[c])]

for c in cat_cols:
    le = LabelEncoder()
    train_fe[c] = train_fe[c].astype(str)
    test_fe[c] = test_fe[c].astype(str)
    le.fit(pd.concat([train_fe[c], test_fe[c]]).unique())
    train_fe[c] = le.transform(train_fe[c])
    test_fe[c] = le.transform(test_fe[c])

print("Processing text...")
vec = TfidfVectorizer(max_features=30000, ngram_range=(1,2), sublinear_tf=True, min_df=5)
Xt_tr = vec.fit_transform(train_fe[TEXT_COL])
Xt_te = vec.transform(test_fe[TEXT_COL])
svd = TruncatedSVD(n_components=150, random_state=SEED)
Xtd_tr = svd.fit_transform(Xt_tr)
Xtd_te = svd.transform(Xt_te)
print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

# Shift SVD to non-negative for ComplementNB
Xtd_tr_nn = Xtd_tr - Xtd_tr.min(axis=0)
Xtd_te_nn = Xtd_te - Xtd_te.min(axis=0)

X_struct_tr = train_fe[numeric_cols + cat_cols].values
X_struct_te = test_fe[numeric_cols + cat_cols].values
y_tr = train[TARGET_COL].values
test_ids = test[ID_COL].values

print("3-Fold CV Ensemble...")
skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
classes = np.sort(np.unique(y_tr))
oof_struct, oof_text = np.zeros((len(X_struct_tr), len(classes))), np.zeros((len(X_struct_tr), len(classes)))
test_struct, test_text = np.zeros((len(X_struct_te), len(classes))), np.zeros((len(X_struct_te), len(classes)))

for f, (ti, vi) in enumerate(skf.split(X_struct_tr, y_tr), 1):
    sm = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=300,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
    sm.fit(X_struct_tr[ti], y_tr[ti])
    oof_struct[vi] = sm.predict_proba(X_struct_tr[vi])
    test_struct += sm.predict_proba(X_struct_te) / 3

    tm = ComplementNB(alpha=0.3)
    tm.fit(Xtd_tr_nn[ti], y_tr[ti])
    oof_text[vi] = tm.predict_proba(Xtd_tr_nn[vi])
    test_text += tm.predict_proba(Xtd_te_nn) / 3

    fp = STRUCTURED_WEIGHT * oof_struct[vi] + (1-STRUCTURED_WEIGHT) * oof_text[vi]
    mf1 = f1_score(y_tr[vi], classes[np.argmax(fp, axis=1)], average='macro')
    print(f"  Fold {f}: MF1={mf1:.4f}")

oof_probs = STRUCTURED_WEIGHT * oof_struct + (1-STRUCTURED_WEIGHT) * oof_text
train_preds = classes[np.argmax(oof_probs, axis=1)]
mf1 = f1_score(y_tr, train_preds, average='macro')
hr_recall = np.mean(train_preds[y_tr <= HIGH_RISK_THRESHOLD] <= HIGH_RISK_THRESHOLD)
ut_rate = np.mean((train_preds - y_tr) >= 2)
print(f"\nCV: MF1={mf1:.4f} HR-Recall={hr_recall:.4f} UT={ut_rate:.4f}")

print("\nTraining final model...")
sm_full = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=300,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
sm_full.fit(X_struct_tr, y_tr)
tm_full = ComplementNB(alpha=0.3).fit(Xtd_tr_nn, y_tr)

test_probs = STRUCTURED_WEIGHT * sm_full.predict_proba(X_struct_te) + (1-STRUCTURED_WEIGHT) * tm_full.predict_proba(Xtd_te_nn)
test_preds = classes[np.argmax(test_probs, axis=1)]

sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_preds})
sub.to_csv('/kaggle/working/submission.csv', index=False)
print(f"Submission: {sub.shape}")
d = sub[TARGET_COL].value_counts().sort_index().to_dict()
print(f"Distribution: {d}")
print("DONE")
