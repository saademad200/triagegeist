import os
import json
import numpy as np
from src.metrics import compute_metric

def main():
    print("Running Advanced Ensemble (LightGBM + CatBoost) for Track 1...")
    
    lgb_preds_path = 'results/exp0020_track1_advanced_fe/oof_preds.npy'
    cb_preds_path = 'results/exp0021_track1_catboost_advanced_fe/oof_preds.npy'
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy' # Target remains the same
    
    if not os.path.exists(lgb_preds_path) or not os.path.exists(cb_preds_path):
        print("Predictions not found. Ensure exp0020 and exp0021 have completed.")
        return
        
    lgb_preds = np.load(lgb_preds_path)
    cb_preds = np.load(cb_preds_path)
    y_true = np.load(y_path)
    
    # Simple Soft-Voting
    ensemble_preds = (lgb_preds + cb_preds) / 2.0
    
    lgb_qwk = compute_metric(y_true, np.argmax(lgb_preds, axis=1))
    cb_qwk = compute_metric(y_true, np.argmax(cb_preds, axis=1))
    ens_qwk = compute_metric(y_true, np.argmax(ensemble_preds, axis=1))
    
    print(f"LightGBM (Advanced FE) QWK: {lgb_qwk:.4f}")
    print(f"CatBoost (Advanced FE) QWK: {cb_qwk:.4f}")
    print(f"Ensemble (Advanced FE) QWK: {ens_qwk:.4f}")
    
    os.makedirs('results/exp0022_track1_ensemble_advanced', exist_ok=True)
    np.save('results/exp0022_track1_ensemble_advanced/oof_preds.npy', ensemble_preds)
    
    with open('results/exp0022_track1_ensemble_advanced/metrics.json', 'w') as f:
        json.dump({
            "lgb_qwk": lgb_qwk,
            "cb_qwk": cb_qwk,
            "ens_qwk": ens_qwk
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
