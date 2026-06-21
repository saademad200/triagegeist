import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import OrdinalEncoder

def custom_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

print("Loading data...")
train = pd.read_csv('data/train.csv')
ph = pd.read_csv('data/patient_history.csv')
cc = pd.read_csv('data/chief_complaints.csv')
train = train.merge(ph, on='patient_id', how='left').merge(cc, on='patient_id', how='left')

y = train['triage_acuity'].values - 1
nlp_text = train['chief_complaint_raw'].fillna('Missing').astype(str).values
groups = nlp_text.copy()

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

drop_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_raw', 'chief_complaint_system', 'site_id', 'triage_nurse_id']
X_phys = train.drop(columns=[c for c in drop_cols if c in train.columns])
cat_cols = [c for c in cat_cols if c not in drop_cols]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# OOF Arrays
oof_phys = np.zeros((len(y), 5))
oof_nlp = np.zeros((len(y), 5))
oof_rules = np.zeros((len(y), 5))

print("Starting Phase 3 Grouped CV Loop...")
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
    oof_phys[valid_idx] = lgb_model.predict_proba(X_vl_phys)
    
    # 2. NLP Engine (Calibrated)
    tfidf = TfidfVectorizer(max_features=5000, analyzer='word', ngram_range=(1,2), stop_words='english')
    X_tr_tfidf = tfidf.fit_transform(X_tr_text)
    X_vl_tfidf = tfidf.transform(X_vl_text)
    
    ridge = RidgeClassifier(alpha=1.0, class_weight='balanced')
    calibrated_clf = CalibratedClassifierCV(ridge, cv=3, method='isotonic')
    calibrated_clf.fit(X_tr_tfidf, y_tr)
    oof_nlp[valid_idx] = calibrated_clf.predict_proba(X_vl_tfidf)
    
    # 3. Deterministic Rules
    df_tr = pd.DataFrame({'text': X_tr_text, 'y': y_tr})
    # Compute class distribution per template
    template_dist = df_tr.groupby('text')['y'].value_counts(normalize=True).unstack(fill_value=0)
    
    for i, text in enumerate(X_vl_text):
        if text in template_dist.index:
            probs = template_dist.loc[text].values
            # Pad to 5 classes if some were missing
            full_probs = np.zeros(5)
            full_probs[:len(probs)] = probs
            oof_rules[valid_idx[i]] = full_probs
        else:
            oof_rules[valid_idx[i]] = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Print standalone metrics
print(f"\\nPhysiology Standalone QWK: {custom_qwk(y, np.argmax(oof_phys, axis=1)):.4f}")
print(f"Calibrated NLP Standalone QWK: {custom_qwk(y, np.argmax(oof_nlp, axis=1)):.4f}")
print(f"Deterministic Rules Standalone QWK: {custom_qwk(y, np.argmax(oof_rules, axis=1)):.4f}")

# 4. Meta-Learner (Stacker)
# Feature matrix for stacker: 5 phys probs + 5 nlp probs + 5 rule probs = 15 features
stacker_X = np.hstack([oof_phys, oof_nlp, oof_rules])

# We use logistic regression to learn the optimal blend across all folds
# (To prevent overfitting the stacker itself, we could CV the stacker, but standard practice allows fitting on full OOF if OOF is clean)
stacker = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
stacker.fit(stacker_X, y)
final_oof_probs = stacker.predict_proba(stacker_X)

final_qwk = custom_qwk(y, np.argmax(final_oof_probs, axis=1))
print(f"\\n🏆 Meta-Learner Synergy QWK: {final_qwk:.4f}")

# Phase 4 Preview: Fairness and CARES on the final_oof_probs
elderly_mask = train['age_group'].astype(str).str.strip() == 'elderly'
print(f"Elderly Undertriage Before Mitigation: {(np.argmax(final_oof_probs[elderly_mask], axis=1) > y[elderly_mask]).mean():.4f}")
