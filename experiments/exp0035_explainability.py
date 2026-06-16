import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

def prepare_data():
    train = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    df = train.merge(ph, on='patient_id', how='left')
    
    y = df['triage_acuity'].values - 1  
    
    # Missingness Indicators
    vital_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 'pain_score']
    for col in vital_cols:
        if col in df.columns:
            df[f'is_missing_{col}'] = df[col].isnull().astype(int)
            
    df['historical_admission_rate'] = df['num_prior_admissions_12m'] / df['num_prior_ed_visits_12m'].clip(lower=1)
    df['sirs_tachycardia'] = (df['heart_rate'] > 90).astype(int)
    df['sirs_tachypnea'] = (df['respiratory_rate'] > 20).astype(int)
    df['sirs_temp'] = ((df['temperature_c'] > 38) | (df['temperature_c'] < 36)).astype(int)
    df['sirs_score'] = df['sirs_tachycardia'] + df['sirs_tachypnea'] + df['sirs_temp']
    
    if 'shock_index' not in df.columns:
        df['shock_index'] = df['heart_rate'] / df['systolic_bp'].clip(lower=1)
        
    df['age_adjusted_shock_index'] = df['shock_index'] * df['age']
    df['comorbidity_to_age_ratio'] = df['num_comorbidities'] / df['age'].clip(lower=1)
    df['is_hypoxic'] = (df['spo2'] < 92).astype(int)
    df['is_hypotensive'] = (df['systolic_bp'] < 90).astype(int)
    
    drop_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_system', 'site_id', 'triage_nurse_id']
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
            
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna('Missing').astype(str)
        
    return df, y, cat_cols

def main():
    print("Running SHAP Explainability Analysis (Rubric: Insight & Findings)...")
    X, y, cat_cols = prepare_data()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, valid_idx = next(skf.split(X, y))
    
    X_tr, X_vl = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
    y_tr, y_vl = y[train_idx], y[valid_idx]
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_tr[cat_cols] = encoder.fit_transform(X_tr[cat_cols])
    X_vl[cat_cols] = encoder.transform(X_vl[cat_cols])
    
    model = lgb.LGBMClassifier(
        objective='multiclass', num_class=5, n_estimators=200, learning_rate=0.05,
        max_depth=7, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    # Take a subsample to speed up
    sample_idx = np.random.RandomState(42).choice(len(X_vl), 2000, replace=False)
    X_shap = X_vl.iloc[sample_idx]
    shap_values = explainer.shap_values(X_shap)
    
    os.makedirs('results/exp0035_explainability', exist_ok=True)
    
    if isinstance(shap_values, list):
        # LightGBM usually returns a list of arrays for multiclass
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=(0, 2))
        
    top25 = pd.Series(mean_shap, index=X_shap.columns).sort_values(ascending=False).head(25)
    
    print("\nTop 10 Global SHAP Drivers (Why the model makes its decisions):")
    print(top25.head(10).to_string())
    
    plt.figure(figsize=(10, 8))
    top25.sort_values(ascending=True).plot(kind='barh')
    plt.title('Top 25 SHAP Feature Importances')
    plt.xlabel('Mean |SHAP Value| (Impact on Model Output)')
    plt.tight_layout()
    plt.savefig('results/exp0035_explainability/shap_summary.png', dpi=150)
    
    top25.to_csv('results/exp0035_explainability/top25_shap.csv')
    print("Done! SHAP plot saved to results/exp0035_explainability/shap_summary.png")

if __name__ == "__main__":
    main()
