import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.metrics import compute_metric

SEED = 42

def main():
    print("Loading data for Track 2: NLP Reverse Engineering...")
    train = pd.read_csv("data/train.csv")
    cc = pd.read_csv("data/chief_complaints.csv")
    df = train.merge(cc, on='patient_id', how='left')
    
    df['chief_complaint_raw'] = df['chief_complaint_raw'].fillna('unknown').astype(str)
    y = df['triage_acuity'].values - 1  # 0-indexed
    texts = df['chief_complaint_raw'].values
    
    print("Running 5-Fold CV using TF-IDF + Ridge Classifier...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(df), dtype=int)
    qwk_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(texts, y)):
        text_tr, text_vl = texts[train_idx], texts[valid_idx]
        y_tr, y_vl = y[train_idx], y[valid_idx]
        
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)
        X_tr = vectorizer.fit_transform(text_tr)
        X_vl = vectorizer.transform(text_vl)
        
        clf = RidgeClassifier(alpha=1.0, random_state=SEED)
        clf.fit(X_tr, y_tr)
        
        preds = clf.predict(X_vl)
        oof_preds[valid_idx] = preds
        
        fold_qwk = compute_metric(y_vl, preds)
        qwk_scores.append(fold_qwk)
        print(f'Fold {fold+1} QWK: {fold_qwk:.4f}')

    overall_qwk = compute_metric(y, oof_preds)
    accuracy = np.mean(oof_preds == y)
    print(f"Overall QWK (TF-IDF + Ridge): {overall_qwk:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    os.makedirs('results/exp0007_track2_nlp_lookup', exist_ok=True)
    np.save('results/exp0007_track2_nlp_lookup/oof_preds.npy', oof_preds)
    with open('results/exp0007_track2_nlp_lookup/metrics.json', 'w') as f:
        json.dump({
            "cv_qwk": overall_qwk,
            "accuracy": accuracy,
            "fold_qwk": qwk_scores
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
