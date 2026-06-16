import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.data import load_data, build_features
from src.cv import run_cv

SEED = 42

def main():
    print("Loading data...")
    train, test, history, complaints = load_data()
    
    print("Building features with Target Encoding and Clinical Interactions...")
    # fit_tfidf=True fits the TFIDF on training set
    # is_train=True fits the K-Fold Target Encoder on training set
    X_train, tfidf_fitted, target_encoder = build_features(
        train, history, complaints, 
        fit_tfidf=True, 
        tfidf_max_features=2000,
        is_train=True
    )
    
    print("Building features for Test Set...")
    X_test, _ = build_features(
        test, history, complaints, 
        tfidf=tfidf_fitted, 
        fit_tfidf=False, 
        tfidf_max_features=2000,
        target_encoder=target_encoder,
        is_train=False
    )
    
    y_train = train['triage_acuity'].values - 1  # 0-indexed for LightGBM
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # We relax the regularization slightly compared to exp0003, 
    # as we now have powerful, dense target-encoded features (like nurse bias).
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'n_estimators': 1500,
        'learning_rate': 0.01,
        'num_leaves': 63,            # Reduced from 127 to prevent overfitting on the strong TE features
        'max_depth': -1,
        'min_child_samples': 30,     # Increased to ensure robust splits
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.2,            # Relaxed L1
        'reg_lambda': 0.2,           # Relaxed L2
        'class_weight': 'balanced',
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    print("Running CV...")
    result = run_cv(X_train, y_train, lgb_params, n_splits=5, seed=SEED, model_type='lgb')
    
    print(f"Overall QWK CV Score (exp0004 Nurse Bias): {result['cv_score']:.4f}")
    
    # Generate predictions
    test_preds = np.zeros((len(X_test), 5))
    for model in result['models']:
        test_preds += model.predict(X_test) / len(result['models'])
        
    test_preds_class = np.argmax(test_preds, axis=1) + 1
    
    # Save submission
    sub = pd.DataFrame({'patient_id': test['patient_id'], 'triage_acuity': test_preds_class})
    sub.to_csv('submissions/exp0004_nurse_bias.csv', index=False)
    
    # Save results
    os.makedirs('results/exp0004_nurse_bias', exist_ok=True)
    with open('results/exp0004_nurse_bias/metrics.json', 'w') as f:
        json.dump({
            "cv_score": result['cv_score'],
            "fold_scores": result['fold_scores'],
            "runtime": result['runtime']
        }, f, indent=4)
        
    pd.DataFrame(result['oof']).to_parquet('results/exp0004_nurse_bias/oof.parquet')
    
    # Save Feature Importances
    importances = np.zeros(X_train.shape[1])
    for model in result['models']:
        importances += model.feature_importance(importance_type='gain') / len(result['models'])
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv('results/exp0004_nurse_bias/feature_importance.csv', index=False)
    print("Top 10 features by gain:")
    print(importance_df.head(10))
    
    print("Done!")

if __name__ == "__main__":
    main()
