import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from src.metrics import compute_metric

SEED = 42

def prepare_data():
    print("Loading data for XGBoost (Leak-Free)...")
    train = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    df = train.merge(ph, on='patient_id', how='left')
    
    y = df['triage_acuity'].values - 1  
    
    # Advanced FE
    df['historical_admission_rate'] = df['num_prior_admissions_12m'] / df['num_prior_ed_visits_12m'].clip(lower=1)
    df['sirs_tachycardia'] = (df['heart_rate'] > 90).astype(int)
    df['sirs_tachypnea'] = (df['respiratory_rate'] > 20).astype(int)
    df['sirs_temp'] = ((df['temperature_c'] > 38) | (df['temperature_c'] < 36)).astype(int)
    df['sirs_score'] = df['sirs_tachycardia'] + df['sirs_tachypnea'] + df['sirs_temp']
    
    if 'shock_index' not in df.columns:
        df['shock_index'] = df['heart_rate'] / df['systolic_bp'].clip(lower=1)
        
    df['age_adjusted_shock_index'] = df['shock_index'] * df['age']
    df['comorbidity_to_age_ratio'] = df['num_comorbidities'] / df['age'].clip(lower=1)
    df['is_hypoxic'] = (df['spo2'] < 92).astype(int)
    df['is_hypotensive'] = (df['systolic_bp'] < 90).astype(int)
    
    drop_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_system', 'site_id', 'triage_nurse_id']
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
            
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in cat_cols:
        df[col] = df[col].fillna('Missing').astype(str)
        
    return df, y, cat_cols

def main():
    X, y, cat_cols = prepare_data()
    print(f"Features: {X.columns.tolist()}")
    print(f"X shape: {X.shape}")
    
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 5,
        'eval_metric': 'mlogloss',
        'n_estimators': 250,
        'learning_rate': 0.05,
        'max_depth': 6,
        'tree_method': 'hist',
        'random_state': SEED,
        'n_jobs': 4,
    }
    
    # Compute class weights manually for XGBoost
    classes, counts = np.unique(y, return_counts=True)
    total = sum(counts)
    class_weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
    sample_weights = np.array([class_weights[cls] for cls in y])
    
    print("Running CV...")
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    oof_preds = np.zeros((len(X), 5))
    qwk_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        y_tr, y_vl = y[train_idx], y[valid_idx]
        w_tr = sample_weights[train_idx]
        
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr[cat_cols] = encoder.fit_transform(X_tr[cat_cols])
        X_vl[cat_cols] = encoder.transform(X_vl[cat_cols])
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr, y_tr, 
            sample_weight=w_tr,
            eval_set=[(X_vl, y_vl)], 
            verbose=20
        )
        
        preds_proba = model.predict_proba(X_vl)
        oof_preds[valid_idx] = preds_proba
        
        preds_class = np.argmax(preds_proba, axis=1)
        fold_qwk = compute_metric(y_vl, preds_class)
        qwk_scores.append(fold_qwk)
        print(f'Fold {fold+1} QWK: {fold_qwk:.4f}')

    overall_qwk = compute_metric(y, np.argmax(oof_preds, axis=1))
    print(f"Overall QWK (Leak-Free XGBoost): {overall_qwk:.4f}")
    
    os.makedirs('results/exp0032_track1_xgboost_fe', exist_ok=True)
    np.save('results/exp0032_track1_xgboost_fe/oof_preds.npy', oof_preds)
    
    with open('results/exp0032_track1_xgboost_fe/metrics.json', 'w') as f:
        json.dump({'overall_qwk': overall_qwk, 'fold_qwk': qwk_scores}, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
