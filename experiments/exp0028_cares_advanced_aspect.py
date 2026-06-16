import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0., 1., n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if bin_mask.sum() > 0:
            bin_acc = y_true[bin_mask].mean()
            bin_conf = y_prob[bin_mask].mean()
            ece += np.abs(bin_acc - bin_conf) * bin_mask.sum() / len(y_true)
            
    return ece

def main():
    print("Running Advanced CARES Selective Risk Control with Compounded Aspect...")
    train_df = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    train = train_df.merge(ph, on='patient_id', how='left')
    
    preds_path = 'results/exp0025_track1_mega_ensemble/oof_preds.npy'
    if not os.path.exists(preds_path):
        preds_path = 'results/exp0006_track1_clean_baseline/oof_preds.npy'
        
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy'
    
    oof_preds = np.load(preds_path)
    pred_class = np.argmax(oof_preds, axis=1)
    y_true = np.load(y_path)
    
    is_correct = (pred_class == y_true).astype(int)
    
    # Compounded Aspect: Arrival Mode + Age Group
    train['compounded_aspect'] = train['arrival_mode'].astype(str) + "_" + train['age_group'].astype(str)
    train['pred_class'] = pred_class
    train['is_correct'] = is_correct
    
    aspect_stats = train.groupby(['compounded_aspect', 'pred_class']).agg(
        n=('is_correct', 'count'),
        n_correct=('is_correct', 'sum')
    ).reset_index()
    
    global_mean = is_correct.mean()
    alpha = 5.0
    
    aspect_stats['calibrated_reliability'] = (aspect_stats['n_correct'] + alpha * global_mean) / (aspect_stats['n'] + alpha)
    
    train = train.merge(aspect_stats[['compounded_aspect', 'pred_class', 'calibrated_reliability']], 
                        on=['compounded_aspect', 'pred_class'], how='left')
    train['calibrated_reliability'] = train['calibrated_reliability'].fillna(global_mean)
    
    # Evaluate Correctness Prediction
    auc_calibrated = roc_auc_score(train['is_correct'], train['calibrated_reliability'])
    print(f"AUROC - CARES Calibrated Reliability: {auc_calibrated:.4f}")
    
    sorted_idx = np.argsort(-train['calibrated_reliability'].values)
    is_correct_sorted = train['is_correct'].values[sorted_idx]
    
    target_error = 0.05
    cum_errors = np.cumsum(1 - is_correct_sorted)
    cum_total = np.arange(1, len(is_correct_sorted) + 1)
    empirical_error_rate = cum_errors / cum_total
    
    from statsmodels.stats.proportion import proportion_confint
    cp_lower, cp_upper = proportion_confint(cum_errors, cum_total, alpha=0.1, method='beta')
    
    valid_idx = np.where(cp_upper <= target_error)[0]
    
    best_coverage = 0
    verified_error = 0
    
    if len(valid_idx) > 0:
        max_valid_idx = valid_idx[-1]
        best_coverage = (max_valid_idx + 1) / len(is_correct_sorted)
        verified_error = empirical_error_rate[max_valid_idx]
        print(f"Selective Risk Control (target error <= {target_error}):")
        print(f"  Coverage: {best_coverage*100:.2f}%")
        print(f"  Verified Error: {verified_error*100:.2f}%")
        print(f"  CP Upper Bound: {cp_upper[max_valid_idx]*100:.2f}%")
    else:
        print(f"Cannot guarantee target error {target_error} with delta=0.1.")
        
    os.makedirs('results/exp0028_cares_advanced_aspect', exist_ok=True)
    with open('results/exp0028_cares_advanced_aspect/metrics.json', 'w') as f:
        json.dump({
            "coverage": best_coverage,
            "error_rate": verified_error
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
