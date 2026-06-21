import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import OrdinalEncoder

def custom_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

print("Loading data...")
train = pd.read_csv('data/train.csv')
ph = pd.read_csv('data/patient_history.csv')
cc = pd.read_csv('data/chief_complaints.csv')

train = train.merge(ph, on='patient_id', how='left').merge(cc, on='patient_id', how='left')

# Feature Engineering
y = train['triage_acuity'].values - 1
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

# To group by chief_complaint_raw, we save it before dropping
groups = train['chief_complaint_raw'].values

X = train.drop(columns=[c for c in drop_cols if c in train.columns])
cat_cols = [c for c in cat_cols if c not in drop_cols]

# StratifiedGroupKFold on chief_complaint_raw
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), 5))

print("Starting Grouped CV Loop...")
for fold, (train_idx, valid_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    X_tr, X_vl = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    y_tr, y_vl = y[train_idx], y[valid_idx]
    
    # Strict Fold-Scoped Preprocessing
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_tr[cat_cols] = encoder.fit_transform(X_tr[cat_cols])
    X_vl[cat_cols] = encoder.transform(X_vl[cat_cols])
    
    model = lgb.LGBMClassifier(objective='multiclass', num_class=5, n_estimators=100, learning_rate=0.05, max_depth=7, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_tr, y_tr)
    
    oof_preds[valid_idx] = model.predict_proba(X_vl)
    
    fold_qwk = custom_qwk(y_vl, np.argmax(oof_preds[valid_idx], axis=1))
    print(f"Fold {fold} QWK: {fold_qwk:.4f}")

total_qwk = custom_qwk(y, np.argmax(oof_preds, axis=1))
print(f"\\nGrouped CV Leak-Free Baseline QWK: {total_qwk:.4f}")
