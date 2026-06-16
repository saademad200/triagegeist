import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint

def main():
    print("Loading data for Track 1 CARES Selective Risk Control...")
    train = pd.read_csv("data/train.csv")
    
    preds_path = 'results/exp0006_track1_clean_baseline/oof_preds.npy'
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy'
    
    if not os.path.exists(preds_path):
        print("Predictions from exp0006 not found. Wait for it to finish.")
        return
        
    oof_preds = np.load(preds_path)
    y_true = np.load(y_path)
    
    # Intrinsic confidence
    intrinsic_conf = np.max(oof_preds, axis=1)
    pred_class = np.argmax(oof_preds, axis=1)
    
    is_correct = (pred_class == y_true).astype(int)
    
    # To mimic CARES, we build aspect-answer reliability tables.
    # Aspect = arrival_mode, Answer = pred_class
    train['pred_class'] = pred_class
    train['is_correct'] = is_correct
    
    # Aspect-Answer Reliability
    # Calculate empirical mean correctness per (aspect, predicted_class)
    global_mean = is_correct.mean()
    alpha_shrinkage = 5.0
    
    aspect_answer_stats = train.groupby(['arrival_mode', 'pred_class']).agg(
        n=('is_correct', 'count'),
        n_correct=('is_correct', 'sum')
    ).reset_index()
    
    # Empirical Bayes Shrinkage
    aspect_answer_stats['calibrated_reliability'] = (aspect_answer_stats['n_correct'] + alpha_shrinkage * global_mean) / (aspect_answer_stats['n'] + alpha_shrinkage)
    
    # Map back to train df
    train = train.merge(aspect_answer_stats[['arrival_mode', 'pred_class', 'calibrated_reliability']], 
                        on=['arrival_mode', 'pred_class'], how='left')
    
    # If missing (due to some reason), fill with global mean
    train['calibrated_reliability'] = train['calibrated_reliability'].fillna(global_mean)
    
    # Evaluate Correctness Prediction
    auc_intrinsic = roc_auc_score(train['is_correct'], intrinsic_conf)
    auc_calibrated = roc_auc_score(train['is_correct'], train['calibrated_reliability'])
    
    print(f"AUROC - Intrinsic Confidence: {auc_intrinsic:.4f}")
    print(f"AUROC - CARES Calibrated Reliability: {auc_calibrated:.4f}")
    
    # Conformal Selective Risk Control
    # Sort by calibrated reliability descending
    sorted_idx = np.argsort(-train['calibrated_reliability'].values)
    is_correct_sorted = train['is_correct'].values[sorted_idx]
    
    # We want to find a threshold that guarantees error rate <= 0.05
    target_error = 0.05
    
    # Cumulative errors
    cum_errors = np.cumsum(1 - is_correct_sorted)
    cum_total = np.arange(1, len(is_correct_sorted) + 1)
    empirical_error_rate = cum_errors / cum_total
    
    # Clopper-Pearson Upper Bound (delta=0.1)
    # Using statsmodels proportion_confint
    cp_lower, cp_upper = proportion_confint(cum_errors, cum_total, alpha=0.1, method='beta')
    
    # Find maximum coverage where CP Upper Bound <= target_error
    valid_idx = np.where(cp_upper <= target_error)[0]
    
    if len(valid_idx) > 0:
        max_valid_idx = valid_idx[-1]
        coverage = (max_valid_idx + 1) / len(is_correct_sorted)
        verified_error = empirical_error_rate[max_valid_idx]
        print(f"Selective Risk Control (target error <= {target_error}):")
        print(f"  Coverage: {coverage*100:.2f}%")
        print(f"  Verified Error: {verified_error*100:.2f}%")
        print(f"  CP Upper Bound: {cp_upper[max_valid_idx]*100:.2f}%")
    else:
        print(f"Cannot guarantee target error {target_error} with delta=0.1.")
        
    os.makedirs('results/exp0010_cares_selective_risk', exist_ok=True)
    with open('results/exp0010_cares_selective_risk/metrics.json', 'w') as f:
        json.dump({
            "auc_intrinsic": auc_intrinsic,
            "auc_calibrated": auc_calibrated,
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
