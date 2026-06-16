import os
import json
import numpy as np
from src.metrics import compute_metric

def main():
    print("Running Mega Ensemble (LightGBM + CatBoost + PyTorch) for Track 1...")
    
    lgb_preds_path = 'results/exp0020_track1_advanced_fe/oof_preds.npy'
    cb_preds_path = 'results/exp0021_track1_catboost_advanced_fe/oof_preds.npy'
    mlp_preds_path = 'results/exp0024_track1_pytorch_mlp/oof_preds.npy'
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy'
    
    if not os.path.exists(lgb_preds_path) or not os.path.exists(cb_preds_path) or not os.path.exists(mlp_preds_path):
        print("Predictions not found. Ensure exp0020, exp0021, and exp0024 have completed.")
        return
        
    lgb_preds = np.load(lgb_preds_path)
    cb_preds = np.load(cb_preds_path)
    mlp_preds = np.load(mlp_preds_path)
    y_true = np.load(y_path)
    
    # Weighted Soft-Voting
    # We give PyTorch a smaller weight because its standalone QWK was 0.913 vs 0.930 for the trees
    ensemble_preds = (0.45 * lgb_preds) + (0.45 * cb_preds) + (0.10 * mlp_preds)
    
    lgb_qwk = compute_metric(y_true, np.argmax(lgb_preds, axis=1))
    cb_qwk = compute_metric(y_true, np.argmax(cb_preds, axis=1))
    mlp_qwk = compute_metric(y_true, np.argmax(mlp_preds, axis=1))
    ens_qwk = compute_metric(y_true, np.argmax(ensemble_preds, axis=1))
    
    print(f"LightGBM QWK: {lgb_qwk:.4f}")
    print(f"CatBoost QWK: {cb_qwk:.4f}")
    print(f"PyTorch MLP QWK: {mlp_qwk:.4f}")
    print(f"Mega Ensemble QWK: {ens_qwk:.4f}")
    
    os.makedirs('results/exp0025_track1_mega_ensemble', exist_ok=True)
    np.save('results/exp0025_track1_mega_ensemble/oof_preds.npy', ensemble_preds)
    
    with open('results/exp0025_track1_mega_ensemble/metrics.json', 'w') as f:
        json.dump({
            "lgb_qwk": lgb_qwk,
            "cb_qwk": cb_qwk,
            "mlp_qwk": mlp_qwk,
            "ens_qwk": ens_qwk
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
