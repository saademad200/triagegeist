import os
import json
import numpy as np
from src.metrics import compute_metric

def main():
    print("Running Ensemble (LightGBM + CatBoost) for Track 1...")
    
    lgb_preds_path = 'results/exp0006_track1_clean_baseline/oof_preds.npy'
    cb_preds_path = 'results/exp0013_track1_catboost/oof_preds.npy'
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy'
    
    if not os.path.exists(lgb_preds_path) or not os.path.exists(cb_preds_path):
        print("Predictions not found. Ensure both exp0006 and exp0013 have completed.")
        return
        
    lgb_preds = np.load(lgb_preds_path)
    cb_preds = np.load(cb_preds_path)
    y_true = np.load(y_path)
    
    # Simple Soft-Voting (Averaging Probabilities)
    ensemble_preds = (lgb_preds + cb_preds) / 2.0
    
    # Calculate Metrics
    lgb_class = np.argmax(lgb_preds, axis=1)
    cb_class = np.argmax(cb_preds, axis=1)
    ens_class = np.argmax(ensemble_preds, axis=1)
    
    lgb_qwk = compute_metric(y_true, lgb_class)
    cb_qwk = compute_metric(y_true, cb_class)
    ens_qwk = compute_metric(y_true, ens_class)
    
    print(f"LightGBM QWK: {lgb_qwk:.4f}")
    print(f"CatBoost QWK: {cb_qwk:.4f}")
    print(f"Ensemble QWK: {ens_qwk:.4f}")
    
    os.makedirs('results/exp0016_ensemble', exist_ok=True)
    np.save('results/exp0016_ensemble/oof_preds.npy', ensemble_preds)
    
    with open('results/exp0016_ensemble/metrics.json', 'w') as f:
        json.dump({
            "lgb_qwk": lgb_qwk,
            "cb_qwk": cb_qwk,
            "ens_qwk": ens_qwk
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
