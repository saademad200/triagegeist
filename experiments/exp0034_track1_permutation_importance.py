import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance
from src.metrics import compute_metric

SEED = 42

def prepare_data():
    train = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    df = train.merge(ph, on='patient_id', how='left')
    
    y = df['triage_acuity'].values - 1  
    
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

def custom_qwk_scorer(estimator, X, y):
    preds = estimator.predict(X)
    return compute_metric(y, preds)

def main():
    print("Running Automated Feature Selection (Permutation Importance)...")
    X, y, cat_cols = prepare_data()
    
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    train_idx, valid_idx = next(skf.split(X, y))
    
    X_tr, X_vl = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    y_tr, y_vl = y[train_idx], y[valid_idx]
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_tr[cat_cols] = encoder.fit_transform(X_tr[cat_cols])
    X_vl[cat_cols] = encoder.transform(X_vl[cat_cols])
    
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=5,
        n_estimators=100, # reduced for speed during permutation
        learning_rate=0.05,
        max_depth=7,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=4
    )
    
    model.fit(X_tr, y_tr)
    
    print("Calculating Permutation Importance on Validation Set...")
    importances = permutation_importance(
        model, X_vl, y_vl, scoring=custom_qwk_scorer,
        n_repeats=3, random_state=SEED, n_jobs=4
    )
    
    feat_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': importances.importances_mean,
        'std': importances.importances_std
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Clinical Features:")
    print(feat_imp.head(10).to_string(index=False))
    
    print("\nBottom 10 (Useless) Features:")
    print(feat_imp.tail(10).to_string(index=False))
    
    os.makedirs('results/exp0034_track1_permutation_importance', exist_ok=True)
    feat_imp.to_csv('results/exp0034_track1_permutation_importance/feature_importances.csv', index=False)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
