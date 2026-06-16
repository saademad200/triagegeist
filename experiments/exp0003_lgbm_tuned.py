import os
import json
import numpy as np
import pandas as pd
from src.data import load_data, build_features
from src.cv import run_cv

SEED = 42

def main():
    print("Loading data...")
    train, test, history, complaints = load_data()
    
    print("Building features with TFIDF max_features=2000...")
    X_train, tfidf_fitted = build_features(train, history, complaints, fit_tfidf=True, tfidf_max_features=2000)
    X_test, _ = build_features(test, history, complaints, tfidf=tfidf_fitted, fit_tfidf=False, tfidf_max_features=2000)
    
    y_train = train['triage_acuity'].values - 1  # 0-indexed for LightGBM
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'n_estimators': 1500,        # Increased from 800
        'learning_rate': 0.01,       # Decreased from 0.05
        'num_leaves': 127,           # Increased complexity
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,            # Increased L1 regularization to handle 2k features
        'reg_lambda': 0.5,           # Increased L2
        'class_weight': 'balanced',
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    print("Running CV...")
    result = run_cv(X_train, y_train, lgb_params, n_splits=5, seed=SEED, model_type='lgb')
    
    print(f"Overall QWK CV Score (Tuned LGBM): {result['cv_score']:.4f}")
    
    # Generate predictions
    test_preds = np.zeros((len(X_test), 5))
    for model in result['models']:
        test_preds += model.predict(X_test) / len(result['models'])
        
    test_preds_class = np.argmax(test_preds, axis=1) + 1
    
    # Save submission
    sub = pd.DataFrame({'patient_id': test['patient_id'], 'triage_acuity': test_preds_class})
    sub.to_csv('submissions/exp0003_lgbm_tuned.csv', index=False)
    
    # Save results
    os.makedirs('results/exp0003_lgbm_tuned', exist_ok=True)
    with open('results/exp0003_lgbm_tuned/metrics.json', 'w') as f:
        json.dump({
            "cv_score": result['cv_score'],
            "fold_scores": result['fold_scores'],
            "runtime": result['runtime']
        }, f, indent=4)
        
    pd.DataFrame(result['oof']).to_parquet('results/exp0003_lgbm_tuned/oof.parquet')
    print("Done!")

if __name__ == "__main__":
    main()
