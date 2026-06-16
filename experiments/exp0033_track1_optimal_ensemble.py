import os
import json
import numpy as np
from scipy.optimize import minimize
from src.metrics import compute_metric

def main():
    print("Running Optimal Mega Ensemble Weight Calculation...")
    
    paths = {
        "lgb": 'results/exp0031_track1_leak_free_lgb/oof_preds.npy',
        "cb": 'results/exp0021_track1_catboost_advanced_fe/oof_preds.npy',
        "xgb": 'results/exp0032_track1_xgboost_fe/oof_preds.npy',
        "mlp": 'results/exp0030_track1_leak_free_pytorch/oof_preds.npy',
    }
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy'
    
    preds = []
    names = []
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"Predictions for {name} not found.")
            return
        preds.append(np.load(path))
        names.append(name)
        
    y_true = np.load(y_path)
    
    # preds shape: (4, 80000, 5)
    preds = np.array(preds)
    
    # Objective function to MINIMIZE: -1 * QWK
    def objective(weights):
        # Softmax normalization of weights
        weights = np.array(weights)
        weights = np.exp(weights) / np.sum(np.exp(weights))
        
        # Weighted sum of probabilities
        weighted_preds = np.zeros_like(preds[0])
        for i in range(len(weights)):
            weighted_preds += weights[i] * preds[i]
            
        final_class = np.argmax(weighted_preds, axis=1)
        qwk = compute_metric(y_true, final_class)
        return -qwk
    
    # Initial weights: equal
    initial_weights = [1.0 / len(preds)] * len(preds)
    
    print("Optimizing...")
    # Use Nelder-Mead since QWK is non-differentiable step function
    result = minimize(objective, initial_weights, method='Nelder-Mead', options={'maxiter': 500, 'disp': True})
    
    best_weights = np.exp(result.x) / np.sum(np.exp(result.x))
    print("\nOptimal Weights:")
    for name, weight in zip(names, best_weights):
        print(f"  {name}: {weight:.4f}")
        
    final_preds = np.zeros_like(preds[0])
    for i in range(len(best_weights)):
        final_preds += best_weights[i] * preds[i]
        
    ens_qwk = compute_metric(y_true, np.argmax(final_preds, axis=1))
    print(f"\nFinal Optimized Mega Ensemble QWK: {ens_qwk:.4f}")
    
    os.makedirs('results/exp0033_track1_optimal_ensemble', exist_ok=True)
    np.save('results/exp0033_track1_optimal_ensemble/oof_preds.npy', final_preds)
    
    with open('results/exp0033_track1_optimal_ensemble/metrics.json', 'w') as f:
        json.dump({
            "optimal_qwk": ens_qwk,
            "weights": {name: float(weight) for name, weight in zip(names, best_weights)}
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
