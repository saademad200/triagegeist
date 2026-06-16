import os
import json
import numpy as np
import pandas as pd

def optimize_prob_shift(y_true, preds_proba, demographic_mask, target_undertriage=0.05):
    best_shift = 0
    best_acc = 0
    best_ut = 1.0
    
    sub_y = y_true[demographic_mask]
    sub_proba = preds_proba[demographic_mask].copy()
    
    shifts = np.linspace(0, 0.5, 50)
    
    for shift in shifts:
        shifted_proba = sub_proba.copy()
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
                
    # If no shift achieves the target, return max shift
    if best_acc == 0:
        return 0.5, best_ut, best_acc
        
    return best_shift, best_ut, best_acc

def main():
    print("Running Comprehensive Algorithmic Fairness Mitigation (Track 4)...")
    train_df = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    train = train_df.merge(ph, on='patient_id', how='left')
    
    preds_path = 'results/exp0006_track1_clean_baseline/oof_preds.npy'
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy'
    
    oof_preds = np.load(preds_path)
    y_true = np.load(y_path)
    pred_classes = np.argmax(oof_preds, axis=1)
    
    train['pred_class_orig'] = pred_classes
    train['undertriage_orig'] = (pred_classes > y_true).astype(int)
    train['correct_orig'] = (pred_classes == y_true).astype(int)
    
    orig_global_acc = train['correct_orig'].mean()
    mitigated_preds = pred_classes.copy()
    shifted_proba = oof_preds.copy()
    
    protected_attributes = ['age_group', 'sex', 'language']
    mitigation_log = {}
    
    for attr in protected_attributes:
        print(f"\n--- Optimizing thresholds for: {attr} ---")
        groups = train[attr].astype(str).str.strip().unique()
        
        attr_log = {}
        for g in groups:
            mask = train[attr].astype(str).str.strip() == g
            
            # Skip very small groups to avoid noise
            if mask.sum() < 100:
                continue
                
            orig_ut = train.loc[mask, 'undertriage_orig'].mean()
            orig_acc = train.loc[mask, 'correct_orig'].mean()
            
            # Only apply mitigation if the group is being undertriaged > 5%
            if orig_ut > 0.05:
                shift, new_ut, new_acc = optimize_prob_shift(y_true, shifted_proba, mask, target_undertriage=0.05)
                
                # Apply shift to global probabilities
                shifted_proba[mask, 0] += shift * 1.5
                shifted_proba[mask, 1] += shift * 1.0
                shifted_proba[mask, 2] += shift * 0.5
                
                attr_log[g] = {
                    "orig_ut": orig_ut, "orig_acc": orig_acc,
                    "shift": shift, "new_ut": new_ut, "new_acc": new_acc
                }
                print(f"Group: {g} | Orig UT: {orig_ut:.4f} -> Mitigated UT: {new_ut:.4f} (Shift: {shift:.4f})")
            else:
                attr_log[g] = {"orig_ut": orig_ut, "orig_acc": orig_acc, "shift": 0, "new_ut": orig_ut, "new_acc": orig_acc}
                
        mitigation_log[attr] = attr_log
        
    mitigated_preds = np.argmax(shifted_proba, axis=1)
    train['pred_class_mitigated'] = mitigated_preds
    train['undertriage_mitigated'] = (mitigated_preds > y_true).astype(int)
    train['correct_mitigated'] = (mitigated_preds == y_true).astype(int)
    
    mitig_global_acc = train['correct_mitigated'].mean()
    mitig_global_ut = train['undertriage_mitigated'].mean()
    
    print(f"\nOriginal Global Accuracy: {orig_global_acc:.4f}")
    print(f"Mitigated Global Accuracy: {mitig_global_acc:.4f} (Change: {mitig_global_acc - orig_global_acc:.4f})")
    print(f"Mitigated Global Undertriage: {mitig_global_ut:.4f}")
    
    os.makedirs('results/exp0023_track4_comprehensive_fairness', exist_ok=True)
    with open('results/exp0023_track4_comprehensive_fairness/metrics.json', 'w') as f:
        json.dump({
            "orig_global_acc": orig_global_acc,
            "mitig_global_acc": mitig_global_acc,
            "mitig_global_ut": mitig_global_ut,
            "group_log": mitigation_log
        }, f, indent=4)

if __name__ == "__main__":
    main()
