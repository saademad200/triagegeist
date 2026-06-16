import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def main():
    print("Loading data for Track 4: Bias Audit...")
    train = pd.read_csv("data/train.csv")
    
    preds_path = 'results/exp0006_track1_clean_baseline/oof_preds.npy'
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy'
    
    if not os.path.exists(preds_path):
        print("Predictions from Track 1 (exp0006) not found. Run exp0006 first.")
        return
        
    oof_preds = np.load(preds_path)
    pred_classes = np.argmax(oof_preds, axis=1)
    y_true = np.load(y_path)
    
    # Undertriage: predicting a HIGHER number (lower acuity) than true
    # E.g. True is 1 (Critical), Pred is 3 (Urgent). Pred > True is undertriage.
    undertriage_mask = pred_classes > y_true
    overtriage_mask = pred_classes < y_true
    
    train['undertriage'] = undertriage_mask.astype(int)
    train['overtriage'] = overtriage_mask.astype(int)
    train['correct'] = (pred_classes == y_true).astype(int)
    
    print(f"Overall Undertriage Rate: {train['undertriage'].mean():.4f}")
    
    audit_results = {}
    audit_cols = ['sex', 'age_group', 'language', 'insurance_type']
    
    for col in audit_cols:
        print(f"\n--- Audit for {col} ---")
        group_stats = train.groupby(col).agg(
            n_patients=('patient_id', 'count'),
            undertriage_rate=('undertriage', 'mean'),
            overtriage_rate=('overtriage', 'mean'),
            accuracy=('correct', 'mean')
        ).reset_index()
        
        # Only consider groups with at least 100 patients
        group_stats = group_stats[group_stats['n_patients'] >= 100]
        print(group_stats.to_string(index=False))
        
        # Save to dict
        audit_results[col] = group_stats.to_dict(orient='records')
        
    os.makedirs('results/exp0009_track4_bias_audit', exist_ok=True)
    with open('results/exp0009_track4_bias_audit/audit_report.json', 'w') as f:
        json.dump(audit_results, f, indent=4)
        
    print("\nDone!")

if __name__ == "__main__":
    main()
