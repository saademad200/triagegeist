import os
import json
import numpy as np
import pandas as pd
from src.data import load_data, build_features
from src.cv import run_cv
from src.metrics import compute_metric

SEED = 42

def main():
    print("Loading data...")
    train, test, history, complaints = load_data()
    
    print("Building features...")
    X_train, tfidf_fitted = build_features(train, history, complaints, fit_tfidf=True)
    X_test, _ = build_features(test, history, complaints, tfidf=tfidf_fitted, fit_tfidf=False)
    
    y_train = train['triage_acuity'].values - 1  # 0-indexed
    
    # We load OOF from the LightGBM baseline
    try:
        lgb_oof = pd.read_parquet('results/exp0000_baseline/oof.parquet').values
        lgb_test_preds = pd.read_csv('submissions/exp0000_baseline.csv')
    except Exception as e:
        print("Could not load LightGBM baseline. Please run exp0000_baseline.py first.")
        return

    cat_params = {
        'loss_function': 'MultiClass',
        'iterations': 800,
        'learning_rate': 0.05,
        'depth': 6,
        'random_seed': SEED,
        'thread_count': -1
    }
    
    print("Running CatBoost CV...")
    result = run_cv(X_train, y_train, cat_params, n_splits=5, seed=SEED, model_type='catboost')
    
    print(f"CatBoost QWK CV Score: {result['cv_score']:.4f}")
    
    print("Generating CatBoost test predictions...")
    test_preds = np.zeros((len(X_test), 5))
    for model in result['models']:
        test_preds += model.predict_proba(X_test) / len(result['models'])
        
    cat_test_class = np.argmax(test_preds, axis=1) + 1
    
    # Blend OOF
    blended_oof = (result['oof'] * 0.5) + (lgb_oof * 0.5)
    blended_qwk = compute_metric(y_train, np.argmax(blended_oof, axis=1))
    print(f"BLENDED QWK CV Score (0.5 LGB + 0.5 CatBoost): {blended_qwk:.4f}")
    
    # Save standalone CatBoost submission
    sub_cat = pd.DataFrame({'patient_id': test['patient_id'], 'triage_acuity': cat_test_class})
    sub_cat.to_csv('submissions/exp0001_catboost_only.csv', index=False)
    
    # Generate blended test predictions
    # Wait, lgb_test_preds only has the final class, not the raw probabilities.
    # We would need the raw test probabilities from LGBM to blend. 
    # Since we didn't save them, we will just save the CatBoost standalone for now.
    
    os.makedirs('results/exp0001_catboost', exist_ok=True)
    with open('results/exp0001_catboost/metrics.json', 'w') as f:
        json.dump({
            "cv_score": result['cv_score'],
            "blended_cv_score": blended_qwk,
            "fold_scores": result['fold_scores'],
            "runtime": result['runtime']
        }, f, indent=4)
        
    pd.DataFrame(result['oof']).to_parquet('results/exp0001_catboost/oof.parquet')
    print("Done!")

if __name__ == "__main__":
    main()
