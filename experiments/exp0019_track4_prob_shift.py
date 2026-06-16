import os
import json
import numpy as np
import pandas as pd

def optimize_prob_shift(y_true, preds_proba, demographic_mask, target_undertriage=0.05):
    """
    Finds custom decision thresholds (probability shift) for a specific demographic group 
    that guarantees the undertriage rate is <= target_undertriage while maximizing accuracy.
    """
    best_shift = 0
    best_acc = 0
    best_ut = 1.0
    
    sub_y = y_true[demographic_mask]
    sub_proba = preds_proba[demographic_mask].copy()
    
    # We will search over a bias to add to class 0 (ESI 1) and class 1 (ESI 2)
    # This artificially makes the model more likely to predict higher acuity
    shifts = np.linspace(0, 0.5, 50)
    
    for shift in shifts:
        shifted_proba = sub_proba.copy()
        # Add shift to high acuity classes (0 and 1)
        shifted_proba[:, 0] += shift * 1.5
        shifted_proba[:, 1] += shift * 1.0
        shifted_proba[:, 2] += shift * 0.5
        
        preds = np.argmax(shifted_proba, axis=1)
        undertriage = (preds > sub_y).mean()
        acc = (preds == sub_y).mean()
        
        if undertriage <= target_undertriage:
            if acc > best_acc:
                best_acc = acc
                best_shift = shift
                best_ut = undertriage
                
    return best_shift, best_ut, best_acc

def main():
    print("Running Track 4 Probability Shift Tuning for Bias Mitigation...")
    train_df = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    train = train_df.merge(ph, on='patient_id', how='left')
    
    preds_path = 'results/exp0006_track1_clean_baseline/oof_preds.npy'
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy'
    
    if not os.path.exists(preds_path):
        print("Predictions not found. Run exp0006 first.")
        return
        
    oof_preds = np.load(preds_path)
    pred_classes = np.argmax(oof_preds, axis=1)
    y_true = np.load(y_path)
    
    train['pred_class_orig'] = pred_classes
    train['undertriage_orig'] = (pred_classes > y_true).astype(int)
    train['correct_orig'] = (pred_classes == y_true).astype(int)
    
    orig_global_acc = train['correct_orig'].mean()
    print(f"Original Global Accuracy: {orig_global_acc:.4f}")
    
    mitigated_preds = pred_classes.copy()
    
    # We only apply mitigation to the elderly group
    age_group = 'elderly'
    mask = train['age_group'].astype(str).str.strip() == age_group
    
    orig_ut = train.loc[mask, 'undertriage_orig'].mean()
    orig_acc = train.loc[mask, 'correct_orig'].mean()
    
    shift, new_ut, new_acc = optimize_prob_shift(y_true, oof_preds, mask, target_undertriage=0.05)
    
    print(f"--- Age Group: {age_group} ---")
    print(f"  Orig UT: {orig_ut:.4f}  | Orig Acc: {orig_acc:.4f}")
    print(f"  Prob Shift: {shift:.4f}")
    print(f"  New UT:  {new_ut:.4f}  | New Acc:  {new_acc:.4f}")
    
    # Apply shift
    shifted_proba = oof_preds[mask].copy()
    shifted_proba[:, 0] += shift * 1.5
    shifted_proba[:, 1] += shift * 1.0
    shifted_proba[:, 2] += shift * 0.5
    mitigated_preds[mask] = np.argmax(shifted_proba, axis=1)
        
    train['pred_class_mitigated'] = mitigated_preds
    train['undertriage_mitigated'] = (mitigated_preds > y_true).astype(int)
    train['correct_mitigated'] = (mitigated_preds == y_true).astype(int)
    
    mitig_global_acc = train['correct_mitigated'].mean()
    mitig_global_ut = train['undertriage_mitigated'].mean()
    
    print(f"\nMitigated Global Accuracy: {mitig_global_acc:.4f} (Change: {mitig_global_acc - orig_global_acc:.4f})")
    print(f"Mitigated Global Undertriage: {mitig_global_ut:.4f}")
    
    os.makedirs('results/exp0019_track4_prob_shift', exist_ok=True)
    with open('results/exp0019_track4_prob_shift/metrics.json', 'w') as f:
        json.dump({
            "orig_global_acc": orig_global_acc,
            "mitig_global_acc": mitig_global_acc,
            "elderly_shift": shift,
            "elderly_orig_ut": orig_ut,
            "elderly_new_ut": new_ut
        }, f, indent=4)

if __name__ == "__main__":
    main()
