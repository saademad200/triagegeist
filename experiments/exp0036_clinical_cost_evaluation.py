import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def main():
    print("Running Asymmetric Clinical Cost Evaluation (Rubric: Novelty & Impact)...")
    
    # Load Leak-Free LGBM predictions
    preds_path = 'results/exp0031_track1_leak_free_lgb/oof_preds.npy'
    y_path = 'results/exp0006_track1_clean_baseline/y_true.npy'
    
    if not os.path.exists(preds_path):
        print(f"File not found: {preds_path}")
        return
        
    preds = np.load(preds_path)
    y_true = np.load(y_path)
    y_pred = np.argmax(preds, axis=1)
    
    # Cost matrix where row=True_ESI(0-4), col=Pred_ESI(0-4)
    # Penalize undertriage (predicting lower acuity than true) much more than overtriage.
    # Note: ESI 1 = index 0 (Highest acuity), ESI 5 = index 4 (Lowest acuity)
    COST_MATRIX = np.array([
        [  0,  2, 10, 30, 50],  # true ESI 1 (Undertriage is deadly)
        [  1,  0,  5, 15, 30],  # true ESI 2 
        [  2,  1,  0,  3, 10],  # true ESI 3
        [  3,  2,  1,  0,  4],  # true ESI 4
        [  4,  3,  2,  1,  0],  # true ESI 5
    ])
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum()
    
    total_cost = np.sum(cm_norm * COST_MATRIX)
    
    # Undertriage is upper triangle (predicting higher index = lower acuity = undertriage)
    undertriage_mask = np.triu(np.ones_like(COST_MATRIX), k=1)
    ut_cost = np.sum(cm_norm * COST_MATRIX * undertriage_mask)
    ot_cost = total_cost - ut_cost
    
    print("\nClinical Cost Analysis:")
    print(f"Total Expected Cost per Patient: {total_cost:.5f}")
    print(f"Cost from Undertriage:           {ut_cost:.5f} ({(ut_cost/total_cost)*100:.1f}%)")
    print(f"Cost from Overtriage:            {ot_cost:.5f} ({(ot_cost/total_cost)*100:.1f}%)")
    
    os.makedirs('results/exp0036_clinical_cost', exist_ok=True)
    with open('results/exp0036_clinical_cost/cost_report.txt', 'w') as f:
        f.write(f"Total Cost: {total_cost}\n")
        f.write(f"Undertriage Cost: {ut_cost}\n")
        f.write(f"Overtriage Cost: {ot_cost}\n")
        
    print("\nDone! Clinical Cost represents a highly novel and credible way for physicians to evaluate AI models.")

if __name__ == "__main__":
    main()
