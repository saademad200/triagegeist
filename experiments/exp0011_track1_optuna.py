import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
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
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))
        
    return df, y

def objective(trial, X, y):
    params = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'n_estimators': 150,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'class_weight': 'balanced',
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': 4,
    }
    
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    oof_preds = np.zeros((len(X), 5))
    
    for train_idx, valid_idx in skf.split(X, y):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_vl = y[train_idx], y[valid_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_vl, label=y_vl)
        
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
        model = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid], callbacks=callbacks)
        
        oof_preds[valid_idx] = model.predict(X_vl)
        
    qwk = compute_metric(y, np.argmax(oof_preds, axis=1))
    return qwk

def main():
    print("Running Optuna Hyperparameter Tuning for Track 1...")
    X, y = prepare_data()
    
    study = optuna.create_study(direction='maximize', study_name='track1_lgb_tuning')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=10)
    
    print("Best Trial:")
    print(study.best_trial.value)
    print("Best Params:")
    print(study.best_trial.params)
    
    os.makedirs('results/exp0011_track1_optuna', exist_ok=True)
    with open('results/exp0011_track1_optuna/best_params.json', 'w') as f:
        json.dump(study.best_trial.params, f, indent=4)

if __name__ == "__main__":
    main()
