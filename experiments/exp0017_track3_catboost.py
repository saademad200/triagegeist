import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

SEED = 42

def prepare_data():
    train = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    df = train.merge(ph, on='patient_id', how='left')
    
    # Binary Target for Track 3
    # 1 if admitted/deceased, 0 otherwise
    y = df['disposition'].isin(['admitted', 'deceased']).astype(int).values
    
    # Drop leakages
    drop_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_raw', 'chief_complaint_system']
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
            
    # Engineering
    if 'shock_index' in df.columns and 'age' in df.columns:
        df['shock_age_interaction'] = df['shock_index'] * df['age']
    if 'systolic_bp' in df.columns and 'heart_rate' in df.columns:
        df['bp_hr_product'] = df['systolic_bp'] * df['heart_rate']
            
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna('Missing').astype(str)
        
    return df, y, cat_cols

def main():
    print("Loading data for Track 3: CatBoost Deterioration...")
    X, y, cat_cols = prepare_data()
    
    print("Running CV...")
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    auc_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_vl = y[train_idx], y[valid_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        valid_pool = Pool(X_vl, y_vl, cat_features=cat_cols)
        
        model = CatBoostClassifier(
            iterations=150,
            learning_rate=0.08,
            depth=6,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=SEED,
            verbose=20,
            early_stopping_rounds=30,
            thread_count=4
        )
        
        model.fit(train_pool, eval_set=valid_pool)
        
        preds = model.predict_proba(valid_pool)[:, 1]
        oof_preds[valid_idx] = preds
        
        fold_auc = roc_auc_score(y_vl, preds)
        auc_scores.append(fold_auc)
        print(f'Fold {fold+1} AUC: {fold_auc:.4f}')

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"Overall AUC (CatBoost Deterioration): {overall_auc:.4f}")
    
    os.makedirs('results/exp0017_track3_catboost', exist_ok=True)
    np.save('results/exp0017_track3_catboost/oof_preds.npy', oof_preds)
    np.save('results/exp0017_track3_catboost/y_true.npy', y)
    
    with open('results/exp0017_track3_catboost/metrics.json', 'w') as f:
        json.dump({'overall_auc': overall_auc, 'fold_auc': auc_scores}, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
