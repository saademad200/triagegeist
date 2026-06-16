import os
import json
import numpy as np
import pandas as pd

def main():
    print("Running Track 4 Bias Mitigation (Post-Processing)...")
    train_df = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    train = train_df.merge(ph, on='patient_id', how='left')
    
    preds_path = 'results/exp0006_track1_clean_baseline/oof_preds.npy'
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy'
    
    if not os.path.exists(preds_path):
        print("Predictions from Track 1 not found. Run exp0006 first.")
        return
        
    oof_preds = np.load(preds_path)
    pred_classes = np.argmax(oof_preds, axis=1)
    y_true = np.load(y_path)
    
    # 1. Calculate Original Undertriage for Elderly
    # y_true is 0-indexed (0=ESI1, 4=ESI5). Higher class = lower acuity.
    # Undertriage = pred_classes > y_true
    train['pred_class_orig'] = pred_classes
    train['undertriage_orig'] = (pred_classes > y_true).astype(int)
    
    elderly_mask = train['age_group'].astype(str).str.strip() == 'elderly'
    orig_elderly_ut = train.loc[elderly_mask, 'undertriage_orig'].mean()
    print(f"Original Elderly Undertriage Rate: {orig_elderly_ut:.4f}")
    
    # 2. Heuristic Post-Processing Mitigation
    # If age >= 80 and the model predicts Urgent (2), Semi-Urgent (3), or Non-Urgent (4)
    # We automatically upgrade their acuity by 1 level (subtract 1 from class) 
    # to counteract the blunted physiological response bias.
    
    mitigated_preds = pred_classes.copy()
    
    # Identify cases to override
    override_mask = elderly_mask & (pred_classes >= 2)
    mitigated_preds[override_mask] = mitigated_preds[override_mask] - 1
    
    train['pred_class_mitigated'] = mitigated_preds
    train['undertriage_mitigated'] = (mitigated_preds > y_true).astype(int)
    train['correct_orig'] = (pred_classes == y_true).astype(int)
    train['correct_mitigated'] = (mitigated_preds == y_true).astype(int)
    
    mitigated_elderly_ut = train.loc[elderly_mask, 'undertriage_mitigated'].mean()
    print(f"Mitigated Elderly Undertriage Rate: {mitigated_elderly_ut:.4f}")
    
    orig_acc = train['correct_orig'].mean()
    mitig_acc = train['correct_mitigated'].mean()
    print(f"Original Global Accuracy: {orig_acc:.4f}")
    print(f"Mitigated Global Accuracy: {mitig_acc:.4f}")
    
    os.makedirs('results/exp0012_track4_bias_mitigation', exist_ok=True)
    with open('results/exp0012_track4_bias_mitigation/metrics.json', 'w') as f:
        json.dump({
            "orig_elderly_ut": orig_elderly_ut,
            "mitigated_elderly_ut": mitigated_elderly_ut,
            "orig_global_acc": orig_acc,
            "mitig_global_acc": mitig_acc
        }, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
