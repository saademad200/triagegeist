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
    
    print("Building features...")
    X_train, tfidf_fitted = build_features(train, history, complaints, fit_tfidf=True)
    X_test, _ = build_features(test, history, complaints, tfidf=tfidf_fitted, fit_tfidf=False)
    
    y_train = train['triage_acuity'].values - 1  # 0-indexed for LightGBM
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'n_estimators': 800,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_child_samples': 30,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'class_weight': 'balanced',
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    print("Running CV...")
    result = run_cv(X_train, y_train, lgb_params, n_splits=5, seed=SEED)
    
    print(f"Overall QWK CV Score: {result['cv_score']:.4f}")
    print(f"Runtime: {result['runtime']:.2f} seconds")
    
    # Generate predictions on test set
    print("Generating predictions...")
    test_preds = np.zeros((len(X_test), 5))
    for model in result['models']:
        test_preds += model.predict(X_test) / len(result['models'])
        
    test_preds_class = np.argmax(test_preds, axis=1) + 1  # Convert back to 1-5 scale
    
    # Save submission
    sub = pd.DataFrame({'patient_id': test['patient_id'], 'triage_acuity': test_preds_class})
    sub.to_csv('submissions/exp0000_baseline.csv', index=False)
    
    # Save results
    os.makedirs('results/exp0000_baseline', exist_ok=True)
    with open('results/exp0000_baseline/metrics.json', 'w') as f:
        json.dump({
            "cv_score": result['cv_score'],
            "fold_scores": result['fold_scores'],
            "runtime": result['runtime']
        }, f, indent=4)
    
    # Save OOF
    pd.DataFrame(result['oof']).to_parquet('results/exp0000_baseline/oof.parquet')
    print("Done!")

if __name__ == "__main__":
    main()
