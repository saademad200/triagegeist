import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score

def main():
    print("Running Ensemble (LightGBM + CatBoost) for Track 3...")
    
    lgb_preds_path = 'results/exp0008_track3_deterioration/oof_preds.npy'
    cb_preds_path = 'results/exp0017_track3_catboost/oof_preds.npy'
    y_path = 'results/exp0008_track3_deterioration/y_true.npy'
    
    if not os.path.exists(lgb_preds_path) or not os.path.exists(cb_preds_path):
        print("Predictions not found. Ensure both exp0008 and exp0017 have completed.")
        return
        
    lgb_preds = np.load(lgb_preds_path)
    cb_preds = np.load(cb_preds_path)
    y_true = np.load(y_path)
    
    # Simple Soft-Voting (Averaging Probabilities)
    ensemble_preds = (lgb_preds + cb_preds) / 2.0
    
    # Calculate Metrics
    lgb_auc = roc_auc_score(y_true, lgb_preds)
    cb_auc = roc_auc_score(y_true, cb_preds)
    ens_auc = roc_auc_score(y_true, ensemble_preds)
    
    print(f"LightGBM AUC: {lgb_auc:.4f}")
    print(f"CatBoost AUC: {cb_auc:.4f}")
    print(f"Ensemble AUC: {ens_auc:.4f}")
    
    os.makedirs('results/exp0018_track3_ensemble', exist_ok=True)
    np.save('results/exp0018_track3_ensemble/oof_preds.npy', ensemble_preds)
    
    with open('results/exp0018_track3_ensemble/metrics.json', 'w') as f:
        json.dump({
            "lgb_auc": lgb_auc,
            "cb_auc": cb_auc,
            "ens_auc": ens_auc
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
