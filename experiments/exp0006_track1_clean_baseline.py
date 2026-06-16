import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from src.metrics import compute_metric

SEED = 42

def prepare_data(train_path, ph_path):
    print("Loading data for Track 1: Clean Baseline...")
    train = pd.read_csv(train_path)
    ph = pd.read_csv(ph_path)
    df = train.merge(ph, on='patient_id', how='left')
    
    y = df['triage_acuity'].values - 1  # 0-indexed
    
    # Drop leakage columns and high-cardinality IDs
    drop_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_system', 'site_id', 'triage_nurse_id']
    
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
            
    # Ordinal encode categorical columns to preserve information without LightGBM deadlock
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))
    
    X = df
    
    return X, y, cat_cols

def main():
    X, y, cat_cols = prepare_data("data/train.csv", "data/patient_history.csv")
    print(f"Features: {X.columns.tolist()}")
    print(f"X shape: {X.shape}")
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'n_estimators': 150,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
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
        
        callbacks = [lgb.log_evaluation(period=10), lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model = lgb.train(
            lgb_params, 
            dtrain, 
            valid_sets=[dtrain, dvalid],
            callbacks=callbacks
        )
        
        preds = model.predict(X_vl)
        oof_preds[valid_idx] = preds
        
        fold_qwk = compute_metric(y_vl, np.argmax(preds, axis=1))
        qwk_scores.append(fold_qwk)
        print(f'Fold {fold+1} QWK: {fold_qwk:.4f}')

    overall_qwk = compute_metric(y, np.argmax(oof_preds, axis=1))
    print(f"Overall QWK (Clean Baseline): {overall_qwk:.4f}")
    
    os.makedirs('results/exp0006_track1_clean_baseline', exist_ok=True)
    
    # Save predictions for bias audit
    np.save('results/exp0006_track1_clean_baseline/oof_preds.npy', oof_preds)
    np.save('results/exp0006_track1_clean_baseline/y_true.npy', y)
    
    with open('results/exp0006_track1_clean_baseline/metrics.json', 'w') as f:
        json.dump({
            "cv_qwk": overall_qwk,
            "fold_qwk": qwk_scores
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
