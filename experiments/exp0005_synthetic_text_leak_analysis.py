import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from src.metrics import compute_metric

SEED = 42

def main():
    print("Loading data for Text-Only Leakage Analysis...")
    train = pd.read_csv("data/train.csv")
    complaints = pd.read_csv("data/chief_complaints.csv")
    
    # Merge and prepare data
    df = train.merge(complaints[['patient_id','chief_complaint_raw']], on='patient_id', how='left')
    df['chief_complaint_raw'] = df['chief_complaint_raw'].fillna('unknown')
    
    y = df['triage_acuity'].values - 1  # 0-indexed
    
    print("Extracting TF-IDF features (Max Features: 2000)...")
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), min_df=5, sublinear_tf=True)
    X = tfidf.fit_transform(df['chief_complaint_raw'])
    X = pd.DataFrame(X.toarray(), columns=[f'cc_{c}' for c in tfidf.get_feature_names_out()])
    
    print(f"X shape: {X.shape}")
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'n_estimators': 300,        # Faster training for just proving a point
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'class_weight': 'balanced',
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    print("Running CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros((len(X), 5))
    qwk_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_vl = y[train_idx], y[valid_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_vl, label=y_vl)
        
        callbacks = [lgb.log_evaluation(period=50), lgb.early_stopping(stopping_rounds=50, verbose=False)]
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
        
        # Save top 10 feature importances for this fold to see what words leak the target
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False).head(10)
        print(f"Top 5 predictive words in Fold {fold+1}:")
        print(importances.head(5)['feature'].tolist())

    overall_qwk = compute_metric(y, np.argmax(oof_preds, axis=1))
    print(f"Overall QWK CV Score (Text ONLY): {overall_qwk:.4f}")
    
    os.makedirs('results/exp0005_text_leak_analysis', exist_ok=True)
    with open('results/exp0005_text_leak_analysis/metrics.json', 'w') as f:
        json.dump({
            "cv_score": overall_qwk,
            "fold_scores": qwk_scores
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
