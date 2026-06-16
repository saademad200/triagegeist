import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from src.metrics import compute_metric

SEED = 42

def prepare_data():
    print("Loading data for LightGBM (Leak-Free)...")
    train = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    df = train.merge(ph, on='patient_id', how='left')
    
    y = df['triage_acuity'].values - 1  
    
    # Advanced FE
    # Missingness Indicators (Critical for MNAR variables)
    vital_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 'pain_score']
    for col in vital_cols:
        if col in df.columns:
            df[f'is_missing_{col}'] = df[col].isnull().astype(int)
            
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
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'n_estimators': 250,
        'learning_rate': 0.05,
        'num_leaves': 41,
        'max_depth': 7,
        'class_weight': 'balanced',
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': 4,
    }
    
    print("Running CV...")
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    oof_preds = np.zeros((len(X), 5))
    qwk_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        y_tr, y_vl = y[train_idx], y[valid_idx]
        
        # STRICT LEAK-FREE ENCODING INSIDE FOLD
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_tr[cat_cols] = encoder.fit_transform(X_tr[cat_cols])
        X_vl[cat_cols] = encoder.transform(X_vl[cat_cols])
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dvalid = lgb.Dataset(X_vl, label=y_vl, categorical_feature=cat_cols)
        
        callbacks = [lgb.log_evaluation(period=20), lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model = lgb.train(
            lgb_params, 
            dtrain, 
            valid_sets=[dtrain, dvalid],
            callbacks=callbacks
        )
        
        preds_proba = model.predict(X_vl)
        oof_preds[valid_idx] = preds_proba
        
        preds_class = np.argmax(preds_proba, axis=1)
        fold_qwk = compute_metric(y_vl, preds_class)
        qwk_scores.append(fold_qwk)
        print(f'Fold {fold+1} QWK: {fold_qwk:.4f}')

    overall_qwk = compute_metric(y, np.argmax(oof_preds, axis=1))
    print(f"Overall QWK (Leak-Free LGBM): {overall_qwk:.4f}")
    
    os.makedirs('results/exp0031_track1_leak_free_lgb', exist_ok=True)
    np.save('results/exp0031_track1_leak_free_lgb/oof_preds.npy', oof_preds)
    
    with open('results/exp0031_track1_leak_free_lgb/metrics.json', 'w') as f:
        json.dump({'overall_qwk': overall_qwk, 'fold_qwk': qwk_scores}, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
