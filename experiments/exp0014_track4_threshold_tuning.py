import os
import json
import numpy as np
import pandas as pd

def optimize_thresholds(y_true, preds_proba, demographic_mask, target_undertriage=0.05):
    """
    Finds custom decision thresholds for a specific demographic group 
    that guarantees the undertriage rate is <= target_undertriage.
    For simplicity, we find a global shift parameter `shift` to apply 
    to the predicted classes. But better yet, we shift probabilities.
    Actually, let's find an integer shift for predicted class.
    """
    pred_classes = np.argmax(preds_proba, axis=1)
    
    # We will try shifts from 0 to -4 (shifting to higher acuity)
    best_shift = 0
    best_acc = 0
    
    mask = demographic_mask
    sub_y = y_true[mask]
    sub_preds = pred_classes[mask]
    
    for shift in range(0, -5, -1):
        shifted_preds = np.clip(sub_preds + shift, 0, 4)
        undertriage = (shifted_preds > sub_y).mean()
        acc = (shifted_preds == sub_y).mean()
        
        if undertriage <= target_undertriage:
            return shift, undertriage, acc
            
    return -4, (np.clip(sub_preds - 4, 0, 4) > sub_y).mean(), (np.clip(sub_preds - 4, 0, 4) == sub_y).mean()

def main():
    print("Running Track 4 Threshold Tuning for Bias Mitigation...")
    train_df = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    train = train_df.merge(ph, on='patient_id', how='left')
    
    preds_path = 'results/exp0006_track1_clean_baseline/oof_preds.npy'
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy'
    
    if not os.path.exists(preds_path):
        print("Predictions not found.")
        return
        
    oof_preds = np.load(preds_path)
    pred_classes = np.argmax(oof_preds, axis=1)
    y_true = np.load(y_path)
    
    train['pred_class_orig'] = pred_classes
    train['undertriage_orig'] = (pred_classes > y_true).astype(int)
    train['correct_orig'] = (pred_classes == y_true).astype(int)
    
    orig_global_acc = train['correct_orig'].mean()
    print(f"Original Global Accuracy: {orig_global_acc:.4f}")
    
    age_groups = train['age_group'].unique()
    mitigated_preds = pred_classes.copy()
    
    for age_group in age_groups:
        mask = train['age_group'].astype(str).str.strip() == str(age_group).strip()
        
        orig_ut = train.loc[mask, 'undertriage_orig'].mean()
        orig_acc = train.loc[mask, 'correct_orig'].mean()
        
        shift, new_ut, new_acc = optimize_thresholds(y_true, oof_preds, mask, target_undertriage=0.05)
        
        print(f"--- Age Group: {age_group} ---")
        print(f"  Orig UT: {orig_ut:.4f}  | Orig Acc: {orig_acc:.4f}")
        print(f"  Shift Applied: {shift}")
        print(f"  New UT:  {new_ut:.4f}  | New Acc:  {new_acc:.4f}")
        
        # Apply shift
        mitigated_preds[mask] = np.clip(mitigated_preds[mask] + shift, 0, 4)
        
    train['pred_class_mitigated'] = mitigated_preds
    train['undertriage_mitigated'] = (mitigated_preds > y_true).astype(int)
    train['correct_mitigated'] = (mitigated_preds == y_true).astype(int)
    
    mitig_global_acc = train['correct_mitigated'].mean()
    mitig_global_ut = train['undertriage_mitigated'].mean()
    
    print(f"\nMitigated Global Accuracy: {mitig_global_acc:.4f}")
    print(f"Mitigated Global Undertriage: {mitig_global_ut:.4f}")

if __name__ == "__main__":
    main()
