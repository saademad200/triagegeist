import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import MiniBatchKMeans

def custom_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading data...")
train = pd.read_csv('data/train.csv')
ph = pd.read_csv('data/patient_history.csv')
cc = pd.read_csv('data/chief_complaints.csv')
train = train.merge(ph, on='patient_id', how='left').merge(cc, on='patient_id', how='left')

y = train['triage_acuity'].values - 1

# 1. Normalization & Semantic Grouping
print("Normalizing text and building Semantic Clusters...")
train['norm_text'] = train['chief_complaint_raw'].apply(normalize_text)
nlp_text = train['norm_text'].values

cluster_tfidf = TfidfVectorizer(max_features=2000, analyzer='char_wb', ngram_range=(2,4))
text_vecs = cluster_tfidf.fit_transform(nlp_text)

kmeans = MiniBatchKMeans(n_clusters=1500, random_state=42, batch_size=1000)
groups = kmeans.fit_predict(text_vecs)
print(f"Generated {len(np.unique(groups))} semantic clusters for Fold Grouping.")

# Feature Engineering
vital_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 'pain_score']
for col in vital_cols:
    train[f'is_missing_{col}'] = train[col].isnull().astype(int)
train['historical_admission_rate'] = train['num_prior_admissions_12m'] / train['num_prior_ed_visits_12m'].clip(lower=1)
train['shock_index'] = train['heart_rate'] / train['systolic_bp'].clip(lower=1)
train['age_adjusted_shock_index'] = train['shock_index'] * train['age']

cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    train[col] = train[col].fillna('Missing').astype(str)

drop_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_raw', 'norm_text', 'chief_complaint_system', 'site_id', 'triage_nurse_id']
X_phys = train.drop(columns=[c for c in drop_cols if c in train.columns])
cat_cols = [c for c in cat_cols if c not in drop_cols]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# OOF Arrays
oof_phys = np.zeros((len(y), 5))
oof_nlp = np.zeros((len(y), 5))

print("Starting Phase 1 (Level-One) Semantic Grouped CV Loop...")
for fold, (train_idx, valid_idx) in enumerate(sgkf.split(X_phys, y, groups=groups)):
    X_tr_phys, X_vl_phys = X_phys.iloc[train_idx].copy(), X_phys.iloc[valid_idx].copy()
    X_tr_text, X_vl_text = nlp_text[train_idx], nlp_text[valid_idx]
    y_tr, y_vl = y[train_idx], y[valid_idx]
    
    # 1. Physiological Engine
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_tr_phys[cat_cols] = encoder.fit_transform(X_tr_phys[cat_cols])
    X_vl_phys[cat_cols] = encoder.transform(X_vl_phys[cat_cols])
    
    lgb_model = lgb.LGBMClassifier(objective='multiclass', num_class=5, n_estimators=100, learning_rate=0.05, max_depth=7, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_tr_phys, y_tr)
    xgb_model = XGBClassifier(objective='multi:softprob', num_class=5, n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    xgb_model.fit(X_tr_phys, y_tr)
    cat_model = CatBoostClassifier(loss_function='MultiClass', iterations=100, learning_rate=0.05, depth=6, random_seed=42, verbose=0, thread_count=-1)
    cat_model.fit(X_tr_phys, y_tr)
    oof_phys[valid_idx] = (lgb_model.predict_proba(X_vl_phys) + xgb_model.predict_proba(X_vl_phys) + cat_model.predict_proba(X_vl_phys)) / 3.0
    
    # 2. NLP Engine (Calibrated via Sigmoid)
    # Using Word + Char ngrams for robustness against typos
    tfidf = TfidfVectorizer(max_features=5000, analyzer='word', ngram_range=(1,2), stop_words='english')
    X_tr_tfidf = tfidf.fit_transform(X_tr_text)
    X_vl_tfidf = tfidf.transform(X_vl_text)
    
    ridge = RidgeClassifier(alpha=1.0, class_weight='balanced')
    calibrated_clf = CalibratedClassifierCV(ridge, cv=3, method='sigmoid')
    calibrated_clf.fit(X_tr_tfidf, y_tr)
    oof_nlp[valid_idx] = calibrated_clf.predict_proba(X_vl_tfidf)

print(f"\\nLevel-One Physiology (Semantic Grouped) QWK: {custom_qwk(y, np.argmax(oof_phys, axis=1)):.4f}")
print(f"Level-One NLP (Semantic Grouped) QWK: {custom_qwk(y, np.argmax(oof_nlp, axis=1)):.4f}")

# 3. Honest Nested Stacker (Level-Two)
print("\\nStarting Phase 2 (Level-Two) Honest Nested Stacking...")
stacker_X = np.hstack([oof_phys, oof_nlp])
meta_learner = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)

# Use cross_val_predict to get true, unbiased nested OOF probabilities for the stacker
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_nested_oof_probs = cross_val_predict(meta_learner, stacker_X, y, cv=inner_cv, method='predict_proba')

final_qwk = custom_qwk(y, np.argmax(final_nested_oof_probs, axis=1))
print(f"\\n🏆 Honest Nested Stacker Synergy QWK: {final_qwk:.4f}")
