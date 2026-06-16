```python
# =========================================================
# TRIAGEGEIST — DUAL MODEL CLINICAL AUDIT SYSTEM
# Full Self-Healing Pipeline (Mobile-Friendly Unified Block)
# Author: Sanco Isaacs
# =========================================================

# ======================
# 0. SETUP & PATH AUTODETECT
# ======================
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

RANDOM_STATE = 42

print("Initializing System & Scanning for Data Directories...")

DATA_DIR = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train.csv' in files:
        DATA_DIR = root + '/'
        break

if DATA_DIR is None:
    raise FileNotFoundError("CRITICAL ERROR: train.csv not found.")

print(f"Data Directory Locked: {DATA_DIR}")

# ======================
# 1. LOAD DATA
# ======================
print("Loading CSVs...")
train = pd.read_csv(f"{DATA_DIR}train.csv")
test = pd.read_csv(f"{DATA_DIR}test.csv")
history = pd.read_csv(f"{DATA_DIR}patient_history.csv")
complaints = pd.read_csv(f"{DATA_DIR}chief_complaints.csv")

# ======================
# 2. MERGE DATASETS
# ======================
print("Executing Left Joins...")
train = train.merge(history, on="patient_id", how="left")
test = test.merge(history, on="patient_id", how="left")
train = train.merge(complaints, on="patient_id", how="left")
test = test.merge(complaints, on="patient_id", how="left")

# ======================
# 3. FEATURE ENGINEERING
# ======================
print("Building Clinical & Auditor Features...")
def create_features(df):
    vital_cols = ["systolic_bp","diastolic_bp","heart_rate", "respiratory_rate","temperature_c","spo2"]
    df["missing_vitals_count"] = df[vital_cols].isna().sum(axis=1)
    
    for col in vital_cols:
        df[col+"_missing"] = df[col].isna().astype(int)
        
    df["pain_score_missing"] = (df["pain_score"] == -1).astype(int)

    df["hypotension_flag"] = (df["systolic_bp"] < 90).astype(int)
    df["tachycardia_flag"] = (df["heart_rate"] > 100).astype(int)
    df["hypoxia_flag"] = (df["spo2"] < 94).astype(int)
    df["fever_flag"] = (df["temperature_c"] > 38).astype(int)
    df["hypothermia_flag"] = (df["temperature_c"] < 35).astype(int)
    df["resp_distress_flag"] = (df["respiratory_rate"] > 22).astype(int)
    
    df["ams_flag"] = df["mental_status_triage"].isin(
        ["confused","drowsy","unresponsive","agitated"]
    ).astype(int)

    df["complaint_len"] = df["chief_complaint_raw"].astype(str).apply(len)
    df["exclaim_count"] = df["chief_complaint_raw"].astype(str).str.count("!")

    hx_cols = [c for c in df.columns if c.startswith("hx_")]
    df["hx_total"] = df[hx_cols].sum(axis=1)
    
    return df

train = create_features(train)
test = create_features(test)

# ======================
# 4. ENCODE CATEGORICALS (PATCHED FOR LEAKAGE)
# ======================
print("Encoding categorical variables...")
cat_cols = train.select_dtypes(include="object").columns.tolist()

columns_to_ignore = ["patient_id", "chief_complaint_raw", "disposition"]
for col in columns_to_ignore:
    if col in cat_cols:
        cat_cols.remove(col)

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]]).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ======================
# 5. DEFINE TARGET
# ======================
target = "triage_acuity"
y_train_mapped = train[target] - 1 

# ======================
# 6. FEATURE SPLIT
# ======================
clinical_exclude = ["triage_acuity", "patient_id", "disposition", "ed_los_hours", "insurance_type", "arrival_mode", "language", "site_id", "triage_nurse_id", "chief_complaint_raw"]
clinical_features = [c for c in train.columns if c not in clinical_exclude]

full_exclude = ["triage_acuity", "patient_id", "disposition", "ed_los_hours", "chief_complaint_raw"]
full_features = [c for c in train.columns if c not in full_exclude]

# ======================
# 7. CROSS VALIDATION 
# ======================
def train_model(features, model_name):
    print(f"\n--- Booting {model_name} ---")
    X = train[features]
    y = y_train_mapped 
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(train))
    test_preds = np.zeros((len(test), 5))

    for fold, (tr, va) in enumerate(folds.split(X, y)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        model = LGBMClassifier(
            n_estimators=1000, learning_rate=0.03, num_leaves=64,
            colsample_bytree=0.8, subsample=0.8, random_state=RANDOM_STATE,
            verbose=-1
        )
        
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[])
        
        oof[va] = model.predict(Xva)
        test_preds += model.predict_proba(test[features]) / folds.n_splits
        print(f"Fold {fold+1} Complete")
        
    print(f"[{model_name}] Accuracy: {accuracy_score(y, oof):.4f}")
    print(f"[{model_name}] F1 (Weighted): {f1_score(y, oof, average='weighted'):.4f}")
    
    final_test_preds = np.argmax(test_preds, axis=1)
    return oof, final_test_preds

# ======================
# 8. MODEL EXECUTION
# ======================
clinical_oof, clinical_test = train_model(clinical_features, "Model A: Clinical Baseline")
full_oof, full_test = train_model(full_features, "Model B: Observed Reality")

# ======================
# 9. BIAS GAP ANALYSIS
# ======================
print("\n--- Auditing Systemic Bias ---")
train["bias_gap"] = full_oof - clinical_oof
print("Bias Gap Distribution (Model B - Model A):")
print(train["bias_gap"].value_counts().sort_index())

# ======================
# 10. SUBMISSION FILE
# ======================
submission = pd.DataFrame({
    "patient_id": test["patient_id"],
    "triage_acuity": (full_test + 1).astype(int) 
})
submission.to_csv("submission.csv", index=False)
print("\n[SUCCESS] Pipeline Complete. 'submission.csv' generated and ready for scoring.")

```


```python
# =========================================================
# 11. THE VIZ PACK: SHAP & BIAS DISTRIBUTION (PATCHED)
# =========================================================
print("Initializing Visual Diagnostics...")
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Plot the Bias Gap Distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=train[train['bias_gap'] != 0], x='bias_gap', palette='coolwarm')
plt.title("Triage Discrepancy: Human Bias vs. Clinical Reality", fontsize=14, fontweight='bold')
plt.xlabel("Bias Gap (Negative = Undertriaged, Positive = Overtriaged)", fontsize=12)
plt.ylabel("Number of Patients", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. Subgroup Analysis: Who is getting undertriaged?
undertriaged = train[train['bias_gap'] < 0]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.countplot(data=undertriaged, y='arrival_mode', order=undertriaged['arrival_mode'].value_counts().index, ax=axes[0], palette='Reds_r')
axes[0].set_title("Undertriage by Arrival Mode")
axes[0].set_xlabel("Patient Count")

sns.countplot(data=undertriaged, y='insurance_type', order=undertriaged['insurance_type'].value_counts().index, ax=axes[1], palette='Reds_r')
axes[1].set_title("Undertriage by Insurance Type")
axes[1].set_xlabel("Patient Count")

plt.tight_layout()
plt.show()

# 3. SHAP Explainability (Model B - The Human Model)
print("Calculating SHAP values for Model B (Using a 10% sample to save compute)...")
final_model_b = LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
final_model_b.fit(train[full_features], y_train_mapped)

shap_sample = train[full_features].sample(n=5000, random_state=RANDOM_STATE)

# FIX: Add perturbation parameter for Kaggle background stability
explainer = shap.TreeExplainer(final_model_b, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(shap_sample)

# FIX: Handle SHAP version format changes (List vs 3D Tensor)
if isinstance(shap_values, list):
    shap_class_0 = shap_values[0] # Older SHAP
else:
    shap_class_0 = shap_values[:, :, 0] # Newer SHAP

# Technical Visualization: Summary Plot
plt.figure(figsize=(10, 8))
plt.title("SHAP Feature Importance: High Acuity Triage Drivers (Class 0)", fontsize=14)
shap.summary_plot(shap_class_0, shap_sample, show=False)
plt.tight_layout()
plt.show()

# Executive Visualization: Bar Plot
plt.figure(figsize=(10, 8))
plt.title("Executive Summary: Mean Absolute SHAP Values", fontsize=14)
shap_explanation = shap.Explanation(values=shap_class_0, data=shap_sample, feature_names=shap_sample.columns)
shap.plots.bar(shap_explanation, show=False)
plt.tight_layout()
plt.show()

print("[SUCCESS] Visual Diagnostics Generated.")

```
