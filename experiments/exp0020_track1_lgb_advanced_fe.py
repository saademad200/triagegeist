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
    print("Loading data and engineering advanced clinical features...")
    train = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    df = train.merge(ph, on='patient_id', how='left')
    
    y = df['triage_acuity'].values - 1  
    
    # --- ADVANCED FEATURE ENGINEERING ---
    
    # 1. Historical Admission Rate (Chronicity proxy)
    df['historical_admission_rate'] = df['num_prior_admissions_12m'] / df['num_prior_ed_visits_12m'].clip(lower=1)
    
    # 2. SIRS (Systemic Inflammatory Response Syndrome) proxy score
    # Tachycardia (HR > 90), Tachypnea (RR > 20), Fever/Hypothermia (Temp > 38 or < 36)
    df['sirs_tachycardia'] = (df['heart_rate'] > 90).astype(int)
    df['sirs_tachypnea'] = (df['respiratory_rate'] > 20).astype(int)
    df['sirs_temp'] = ((df['temperature_c'] > 38) | (df['temperature_c'] < 36)).astype(int)
    df['sirs_score'] = df['sirs_tachycardia'] + df['sirs_tachypnea'] + df['sirs_temp']
    
    # 3. Pulse Pressure and MAP are already there, let's add Shock Index
    if 'shock_index' not in df.columns:
        df['shock_index'] = df['heart_rate'] / df['systolic_bp'].clip(lower=1)
        
    # 4. Age-adjusted Shock Index
    df['age_adjusted_shock_index'] = df['shock_index'] * df['age']
    
    # 5. Comorbidity Burden
    df['comorbidity_to_age_ratio'] = df['num_comorbidities'] / df['age'].clip(lower=1)
    
    # 6. Vitals Risk Flags (Hypoxia, Hypotension)
    df['is_hypoxic'] = (df['spo2'] < 92).astype(int)
    df['is_hypotensive'] = (df['systolic_bp'] < 90).astype(int)
    
    # ------------------------------------
    
    drop_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_system', 'site_id', 'triage_nurse_id']
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
            
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))
        
    return df, y, cat_cols

def main():
    X, y, cat_cols = prepare_data()
    print(f"Features: {X.columns.tolist()}")
    print(f"X shape: {X.shape}")
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'n_estimators': 250,  # Increased due to more features
        'learning_rate': 0.05,
        'num_leaves': 41, # Slightly more capacity
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
        X_tr, X_vl = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_vl = y[train_idx], y[valid_idx]
        
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
    print(f"Overall QWK (Advanced FE Baseline): {overall_qwk:.4f}")
    
    os.makedirs('results/exp0020_track1_advanced_fe', exist_ok=True)
    np.save('results/exp0020_track1_advanced_fe/oof_preds.npy', oof_preds)
    
    with open('results/exp0020_track1_advanced_fe/metrics.json', 'w') as f:
        json.dump({'overall_qwk': overall_qwk, 'fold_qwk': qwk_scores}, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
