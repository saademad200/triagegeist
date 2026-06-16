import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from src.metrics import compute_metric

SEED = 42

def prepare_data():
    train = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    df = train.merge(ph, on='patient_id', how='left')
    
    y = df['triage_acuity'].values - 1  
    drop_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_system']
    
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
            
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Fill NA for categoricals to prevent CatBoost errors
    for col in cat_cols:
        df[col] = df[col].fillna('Missing').astype(str)
        
    return df, y, cat_cols

def main():
    print("Loading data for Track 1: CatBoost Clean Baseline...")
    X, y, cat_cols = prepare_data()
    print(f"Features: {X.columns.tolist()}")
    print(f"X shape: {X.shape}")
    
    print("Running CV...")
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    oof_preds = np.zeros((len(X), 5))
    qwk_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_vl = y[train_idx], y[valid_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        valid_pool = Pool(X_vl, y_vl, cat_features=cat_cols)
        
        model = CatBoostClassifier(
            iterations=150,
            learning_rate=0.08,
            depth=6,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            auto_class_weights='Balanced',
            random_seed=SEED,
            verbose=20,
            early_stopping_rounds=30,
            thread_count=4
        )
        
        model.fit(train_pool, eval_set=valid_pool)
        
        preds_proba = model.predict_proba(valid_pool)
        oof_preds[valid_idx] = preds_proba
        
        preds_class = np.argmax(preds_proba, axis=1)
        fold_qwk = compute_metric(y_vl, preds_class)
        qwk_scores.append(fold_qwk)
        print(f'Fold {fold+1} QWK: {fold_qwk:.4f}')

    overall_qwk = compute_metric(y, np.argmax(oof_preds, axis=1))
    print(f"Overall QWK (CatBoost Baseline): {overall_qwk:.4f}")
    
    os.makedirs('results/exp0013_track1_catboost', exist_ok=True)
    np.save('results/exp0013_track1_catboost/oof_preds.npy', oof_preds)
    np.save('results/exp0013_track1_catboost/y_true.npy', y)
    
    with open('results/exp0013_track1_catboost/metrics.json', 'w') as f:
        json.dump({'overall_qwk': overall_qwk, 'fold_qwk': qwk_scores}, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
