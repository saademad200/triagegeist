import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, log_loss

SEED = 42

def prepare_data(train_path, ph_path):
    print("Loading data for Track 3: Deterioration Risk...")
    train = pd.read_csv(train_path)
    ph = pd.read_csv(ph_path)
    df = train.merge(ph, on='patient_id', how='left')
    
    # Target definition: Deterioration/High Risk means admitted or deceased
    # disposition values: 'discharged', 'admitted', 'deceased', 'ama' (against medical advice)
    if 'disposition' not in df.columns:
        raise ValueError("disposition column not found in training data.")
        
    df['deterioration_risk'] = df['disposition'].isin(['admitted', 'deceased']).astype(int)
    
    y = df['deterioration_risk'].values
    
    # Drop leakage and target columns
    drop_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_system', 'site_id', 'triage_nurse_id']
    
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
            
    # Also ignore raw text (no text leakage allowed for clean model)
    # The text isn't in train.csv anyway (it's in chief_complaints.csv) so we just don't merge it.
    
    # Feature Engineering: Clinical Interactions
    if 'shock_index' in df.columns and 'age' in df.columns:
        df['shock_age_interaction'] = df['shock_index'] * df['age']
    if 'systolic_bp' in df.columns and 'heart_rate' in df.columns:
        df['bp_hr_product'] = df['systolic_bp'] * df['heart_rate']
        
    # Ordinal encode categorical columns to preserve information
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))
        
    X = df.drop(['deterioration_risk'], axis=1)
    
    return X, y, cat_cols

def main():
    X, y, cat_cols = prepare_data("data/train.csv", "data/patient_history.csv")
    print(f"Features: {X.columns.tolist()}")
    print(f"X shape: {X.shape}, target mean: {y.mean():.4f}")
    
    lgb_params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'n_estimators': 150,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': 4,
    }
    
    print("Running CV...")
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    auc_scores = []
    
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
        
        fold_auc = roc_auc_score(y_vl, preds)
        auc_scores.append(fold_auc)
        print(f'Fold {fold+1} AUC: {fold_auc:.4f}')

    overall_auc = roc_auc_score(y, oof_preds)
    overall_logloss = log_loss(y, oof_preds)
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Overall Logloss: {overall_logloss:.4f}")
    
    os.makedirs('results/exp0008_track3_deterioration', exist_ok=True)
    np.save('results/exp0008_track3_deterioration/oof_preds.npy', oof_preds)
    np.save('results/exp0008_track3_deterioration/y_true.npy', y)
    with open('results/exp0008_track3_deterioration/metrics.json', 'w') as f:
        json.dump({
            "cv_auc": overall_auc,
            "cv_logloss": overall_logloss,
            "fold_auc": auc_scores
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
