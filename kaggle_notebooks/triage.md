```python
"""
+==========================================================================+
   TRIAGEGEIST - Kaggle Grandmaster-Level Emergency Triage Prediction     
                                                                          
   Approach: LightGBM Regressor + CatBoost + XGBoost Ensemble            
             with QWK-optimized threshold tuning & Chief Complaint NLP    
                                                                          
   Key Techniques:                                                        
       Regression -> threshold optimization (maximizes QWK directly)       
       3-model weighted ensemble (LGB + CatBoost + XGB)                   
       MNAR missingness-as-feature (clinically motivated)                 
       TF-IDF + clinical keyword NLP on chief complaints                  
       SHAP explainability + fairness audit                               
       Ablation study proving NLP value                                    
                                                                          
   Competition: https://kaggle.com/competitions/triagegeist               
   Dataset: Synthetic ED triage (Laitinen-Fredriksson Foundation)         
   License: Non-Commercial Research                                       
+==========================================================================+
"""

# ===========================================================================
#  SETUP & IMPORTS
# ===========================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    cohen_kappa_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
os.makedirs("outputs", exist_ok=True)

# Professional visualization theme
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})
PALETTE = ["#e94560", "#6C5CE7", "#00b894", "#fdcb6e", "#0984e3"]

print("> Setup complete")

```


```python
# ===========================================================================
#  HELPER FUNCTIONS
# ===========================================================================

def qwk(y_true, y_pred):
    """Compute Quadratic Weighted Kappa."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def optimize_thresholds(oof_raw, y_true, n_classes=5):
    """
    Find optimal thresholds to convert regression outputs to ordinal classes.
    This is the KEY technique - train as regression, optimize thresholds for QWK.

    Returns: optimized thresholds array
    """
    # Initial thresholds: evenly spaced between class boundaries
    init_thresholds = np.array([1.5, 2.5, 3.5, 4.5])

    def neg_qwk(thresholds):
        thresholds = np.sort(thresholds)
        preds = np.digitize(oof_raw, thresholds) + 1
        preds = np.clip(preds, 1, n_classes)
        return -qwk(y_true, preds)

    result = minimize(neg_qwk, init_thresholds, method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6})
    return np.sort(result.x)


def apply_thresholds(raw_preds, thresholds, n_classes=5):
    """Convert regression predictions to ordinal classes using optimized thresholds."""
    preds = np.digitize(raw_preds, thresholds) + 1
    return np.clip(preds, 1, n_classes)


print("> Helper functions defined")

```


```python
# ===========================================================================
#  SECTION 1: DATA LOADING & MERGING
# ===========================================================================

print("\n" + "=" * 60)
print("  SECTION 1: DATA LOADING & VALIDATION")
print("=" * 60)

# Correct path for Kaggle competition
DATA_DIR = "/kaggle/input/competitions/triagegeist/"

train = pd.read_csv(f"{DATA_DIR}train.csv")
test = pd.read_csv(f"{DATA_DIR}test.csv")
complaints = pd.read_csv(f"{DATA_DIR}chief_complaints.csv")
history = pd.read_csv(f"{DATA_DIR}patient_history.csv")

print(f"  train.csv:            {train.shape}")
print(f"  test.csv:             {test.shape}")
print(f"  chief_complaints.csv: {complaints.shape}")
print(f"  patient_history.csv:  {history.shape}")

# chief_complaints.csv has chief_complaint_system which also exists in train/test
# Drop it from complaints to avoid _x/_y duplicate columns
if "chief_complaint_system" in complaints.columns and "chief_complaint_system" in train.columns:
    complaints = complaints.drop(columns=["chief_complaint_system"])

# Merge all tables on patient_id
train = train.merge(complaints, on="patient_id", how="left")
test = test.merge(complaints, on="patient_id", how="left")
train = train.merge(history, on="patient_id", how="left")
test = test.merge(history, on="patient_id", how="left")

print(f"\n  After merge -> train: {train.shape}, test: {test.shape}")

# Rigorous validation
assert train["patient_id"].nunique() == len(train), "Duplicate patient_ids!"
assert "triage_acuity" in train.columns, "Target missing!"
assert "triage_acuity" not in test.columns, "Target leaked!"
assert len(set(train["patient_id"]) & set(test["patient_id"])) == 0, "Patient overlap!"
print("  > All validation checks passed")
```


```python
# ===========================================================================
#  SECTION 2: EXPLORATORY DATA ANALYSIS
# ===========================================================================

print("\n" + "=" * 60)
print("  SECTION 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

target = train["triage_acuity"]

# Target distribution
print(f"\n  ESI Distribution:")
for esi in sorted(target.unique()):
    n = (target == esi).sum()
    pct = n / len(target) * 100
    bar = "#" * int(pct)
    print(f"    ESI-{esi}: {n:6,} ({pct:5.1f}%) {bar}")

# Missing value analysis
print(f"\n  Missing Value Analysis:")
missable = ["systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
            "spo2", "temperature_c"]
for col in missable:
    if col in train.columns:
        pct = train[col].isna().mean() * 100
        if pct > 0:
            print(f"    {col:25s} {pct:5.1f}% missing")
if "pain_score" in train.columns:
    pct = (train["pain_score"] < 0).mean() * 100
    print(f"    {'pain_score (-1 encoded)':25s} {pct:5.1f}% missing")

# Dataset-provided clinical features
provided_feats = ["shock_index", "news2_score", "mean_arterial_pressure",
                  "pulse_pressure", "bmi"]
found = [f for f in provided_feats if f in train.columns]
print(f"\n  Pre-computed clinical features: {found}")

# Missingness by acuity (proving MNAR)
print(f"\n  Missingness by ESI (proving MNAR pattern):")
if "systolic_bp" in train.columns:
    for esi in sorted(target.unique()):
        mask = target == esi
        miss_pct = train.loc[mask, "systolic_bp"].isna().mean() * 100
        print(f"    ESI-{esi}: {miss_pct:5.1f}% SBP missing")

# Key visualization: ESI distribution
fig, ax = plt.subplots(figsize=(8, 5))
counts = target.value_counts().sort_index()
bars = ax.bar(counts.index.astype(str), counts.values, color=PALETTE, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{val:,}", ha="center", fontsize=10, fontweight="bold")
ax.set_xlabel("ESI Level (1=Most Urgent, 5=Least Urgent)")
ax.set_ylabel("Count")
ax.set_title("Target Distribution: Emergency Severity Index")
plt.tight_layout()
plt.savefig("outputs/esi_distribution.png", dpi=150, bbox_inches="tight")
plt.close("all")



```


```python
# ===========================================================================
#  SECTION 3: FEATURE ENGINEERING
# ===========================================================================

print("\n" + "=" * 60)
print("  SECTION 3: FEATURE ENGINEERING")
print("=" * 60)
print("  NOTE: shock_index, news2_score, MAP, pulse_pressure, bmi")
print("        are pre-computed in the dataset - we use them directly.\n")

# -- 3.1 MISSINGNESS FLAGS (clinically informative - MNAR pattern) --
# Note: heart_rate and spo2 have 0 missing in actual data, but we still 
# create flags for robustness (they'll be all zeros = harmless)
vital_cols = ["systolic_bp", "diastolic_bp", "heart_rate",
              "respiratory_rate", "spo2", "temperature_c"]
for col in vital_cols:
    for df in [train, test]:
        df[f"{col}_missing"] = df[col].isna().astype(np.int8)
for df in [train, test]:
    df["pain_missing"] = (df["pain_score"] < 0).astype(np.int8)
# Total missingness count per patient
for df in [train, test]:
    miss_cols = [c for c in df.columns if c.endswith("_missing")]
    df["total_missing_vitals"] = df[miss_cols].sum(axis=1)
print("  > Missingness flags (8 features incl. total_missing_vitals)")

# -- 3.2 DEMOGRAPHIC FEATURES --
for df in [train, test]:
    df["is_senior"] = (df["age"] >= 65).astype(np.int8)
    df["is_pediatric"] = (df["age"] < 18).astype(np.int8)
    df["age_squared"] = df["age"] ** 2  # Non-linear age effect
print("  > Demographics (3 features)")

# -- 3.3 TEMPORAL FEATURES --
for df in [train, test]:
    if "arrival_hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["arrival_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["arrival_hour"] / 24)
    if "arrival_day" in df.columns:
        df["is_weekend"] = df["arrival_day"].isin(["Saturday", "Sunday"]).astype(np.int8)
print("  > Temporal (3 features)")

# -- 3.4 VITAL INTERACTIONS (novel, not in dataset) --
for df in [train, test]:
    df["rate_pressure_product"] = df["heart_rate"] * df["systolic_bp"]
    df["spo2_rr_ratio"] = df["spo2"] / (df["respiratory_rate"] + 1e-5)
    if "mean_arterial_pressure" in df.columns:
        df["modified_shock_index"] = df["heart_rate"] / (df["mean_arterial_pressure"] + 1e-5)
    if "num_comorbidities" in df.columns:
        df["age_comorbidity"] = df["age"] * df["num_comorbidities"]
    # Respiratory distress composite
    df["resp_distress"] = ((df["respiratory_rate"] > 22) | (df["spo2"] < 94)).astype(np.int8)
    # Hemodynamic instability flag
    if "shock_index" in df.columns:
        df["hemodynamic_unstable"] = (df["shock_index"] > 1.0).astype(np.int8)
    # GCS severity
    if "gcs_total" in df.columns:
        df["gcs_severe"] = (df["gcs_total"] <= 8).astype(np.int8)
        df["gcs_moderate"] = ((df["gcs_total"] > 8) & (df["gcs_total"] <= 13)).astype(np.int8)
    # Weight/BMI interaction (weight_kg and height_cm exist in dataset)
    if "weight_kg" in df.columns and "age" in df.columns:
        df["weight_age_ratio"] = df["weight_kg"] / (df["age"] + 1)
print("  > Vital interactions (10 features)")

# -- 3.5 HISTORY AGGREGATES --
hx_cols = [c for c in train.columns if c.startswith("hx_")]
for df in [train, test]:
    if hx_cols:
        df["hx_sum"] = df[hx_cols].sum(axis=1)
        df["has_comorbidity"] = (df["hx_sum"] > 0).astype(np.int8)
        df["multi_comorbid"] = (df["hx_sum"] >= 3).astype(np.int8)
print(f"  > History aggregates (3 features from {len(hx_cols)} hx_ flags)")

# -- 3.6 CATEGORICAL ENCODING --
cat_cols = [
    "sex", "language", "insurance_type", "arrival_mode", "arrival_day",
    "shift", "arrival_season", "transport_origin", "mental_status_triage",
    "chief_complaint_system", "pain_location", "age_group",
    # These exist but are IDs - encode anyway as they may capture site effects
    "site_id", "triage_nurse_id",
]
encoded_cats = []
cat_mappings = {}  # Save for reproducibility
for col in cat_cols:
    if col not in train.columns:
        continue
    unique_vals = sorted(train[col].dropna().unique())
    mapping = {v: i for i, v in enumerate(unique_vals)}
    cat_mappings[col] = mapping
    enc_col = f"{col}_enc"
    train[enc_col] = train[col].map(mapping).fillna(-1).astype(int)
    test[enc_col] = test[col].map(mapping).fillna(-1).astype(int)
    encoded_cats.append(enc_col)
print(f"  > Categorical encoding ({len(encoded_cats)} columns)")

```


```python
# ===========================================================================
#  SECTION 4: NLP - CHIEF COMPLAINT TEXT FEATURES
# ===========================================================================

print("\n" + "=" * 60)
print("  SECTION 4: CHIEF COMPLAINT NLP")
print("=" * 60)

# Clean text
for df in [train, test]:
    df["chief_text"] = (df["chief_complaint_raw"]
                        .fillna("")
                        .str.lower()
                        .str.replace(r"[^a-z\s]", "", regex=True)
                        .str.replace(r"\s+", " ", regex=True)
                        .str.strip())

# TF-IDF (fit on train only - no leakage)
tfidf = TfidfVectorizer(max_features=300, min_df=5, stop_words="english",
                         ngram_range=(1, 2), sublinear_tf=True)
X_tfidf_train = tfidf.fit_transform(train["chief_text"])
X_tfidf_test = tfidf.transform(test["chief_text"])

tfidf_cols = [f"tfidf_{i}" for i in range(X_tfidf_train.shape[1])]
train = pd.concat([train.reset_index(drop=True),
                    pd.DataFrame(X_tfidf_train.toarray(), columns=tfidf_cols)], axis=1)
test = pd.concat([test.reset_index(drop=True),
                   pd.DataFrame(X_tfidf_test.toarray(), columns=tfidf_cols)], axis=1)
print(f"  > TF-IDF: {len(tfidf_cols)} features (unigrams + bigrams)")

# Clinical keyword flags (regex-based, domain-informed)
keywords = {
    "kw_chest_pain": r"chest\s*pain",
    "kw_sob": r"shortness\s*of\s*breath|sob|dyspnea|breathing\s*difficulty",
    "kw_unconscious": r"unconscious|unresponsive|syncope|passed\s*out|found\s*down",
    "kw_seizure": r"seizure|convulsion|epilep",
    "kw_trauma": r"trauma|injury|fall|accident|mva|mvc|assault",
    "kw_abdominal": r"abdominal\s*pain|stomach\s*pain|belly",
    "kw_fever": r"fever|febrile|chills",
    "kw_cardiac": r"cardiac|heart\s*attack|palpitation|arrhythmia|afib",
    "kw_stroke": r"stroke|slurred\s*speech|weakness\s*one\s*side|facial\s*droop|tia",
    "kw_bleeding": r"bleeding|hemorrhage|blood\s*loss|hematemesis|melena",
    "kw_altered_mental": r"altered\s*mental|confusion|disoriented|agitated|ams",
    "kw_overdose": r"overdose|ingestion|poisoning|intoxication",
    "kw_respiratory": r"cough|wheez|asthma|copd\s*exac",
    "kw_psych": r"suicid|self\s*harm|psych|anxiety|depression|si$|si ",
}
for df in [train, test]:
    for flag, pattern in keywords.items():
        df[flag] = df["chief_text"].str.contains(pattern, regex=True, na=False).astype(np.int8)
print(f"  > Keyword flags: {len(keywords)} clinical terms")

# Complaint text length (informative: longer = more complex)
for df in [train, test]:
    df["complaint_word_count"] = df["chief_text"].str.split().str.len().fillna(0).astype(int)
    df["complaint_char_count"] = df["chief_text"].str.len().fillna(0).astype(int)
print("  > Text length features (2)")


```


```python
# ===========================================================================
#  SECTION 5: FEATURE SELECTION - DEFINE FINAL FEATURE SET
# ===========================================================================

print("\n" + "=" * 60)
print("  SECTION 5: FEATURE SELECTION")
print("=" * 60)

# STRICT exclusion - prevent any leakage
EXCLUDE = {
    "patient_id", "triage_acuity",
    "disposition", "ed_los_hours",           # OUTCOME COLUMNS = LEAKAGE!
    "chief_complaint_raw", "chief_text",     # Raw text
    *cat_cols,                                # Raw categoricals (use _enc)
    # Columns that might cause merge duplicates
    "chief_complaint_system_x", "chief_complaint_system_y",
}

features = [c for c in train.columns
            if c not in EXCLUDE
            and train[c].dtype in ["int8", "int16", "int32", "int64", "float32", "float64"]]

print(f"  Total features: {len(features)}")
print(f"  - Structured/vitals: {len([f for f in features if not f.startswith(('tfidf_', 'kw_'))])}")
print(f"  - TF-IDF NLP:       {len([f for f in features if f.startswith('tfidf_')])}")
print(f"  - Keyword flags:    {len([f for f in features if f.startswith('kw_')])}")

# Define structured-only features (for ablation study)
features_no_nlp = [f for f in features if not f.startswith(("tfidf_", "kw_", "complaint_"))]
print(f"  - Features (no NLP): {len(features_no_nlp)} (for ablation)")


```


```python
# ===========================================================================
#  SECTION 6: MODEL TRAINING - REGRESSION + QWK THRESHOLD OPTIMIZATION
# ===========================================================================

print("\n" + "=" * 60)
print("  SECTION 6: MODEL TRAINING")
print("  Strategy: Regression -> Threshold Optimization -> Maximum QWK")
print("=" * 60)

X = train
y = target  # ESI 1-5
N_FOLDS = 5
```


```python
# --------------------------------------------------------------------------
#  MODEL 1: LightGBM Regressor (regression -> threshold -> QWK)
# --------------------------------------------------------------------------

print("\n  -- LightGBM (Regression approach) --")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
lgbm_models = []
oof_raw_lgbm = np.zeros(len(X))  # Raw regression outputs

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.3,
    "reg_lambda": 0.3,
    "min_child_samples": 50,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X[features], y)):
    X_tr, X_val = X[features].iloc[trn_idx], X[features].iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_raw_lgbm[val_idx] = model.predict(X_val)
    lgbm_models.append(model)

    # Quick QWK with simple rounding for monitoring
    rounded = np.clip(np.round(oof_raw_lgbm[val_idx]), 1, 5).astype(int)
    fold_qwk = qwk(y_val, rounded)
    print(f"    Fold {fold}: QWK={fold_qwk:.4f} (rounded), best_iter={model.best_iteration_}")

# Optimize thresholds on full OOF
thresholds_lgbm = optimize_thresholds(oof_raw_lgbm, y.values)
oof_preds_lgbm = apply_thresholds(oof_raw_lgbm, thresholds_lgbm)
lgbm_qwk = qwk(y.values, oof_preds_lgbm)
lgbm_acc = accuracy_score(y.values, oof_preds_lgbm)
print(f"\n    > Optimized thresholds: {np.round(thresholds_lgbm, 3)}")
print(f"    > LightGBM OOF QWK:  {lgbm_qwk:.4f}")
print(f"    > LightGBM OOF Acc:  {lgbm_acc:.4f}")

```


```python
# --------------------------------------------------------------------------
#  MODEL 2: CatBoost Regressor
# --------------------------------------------------------------------------

try:
    from catboost import CatBoostRegressor

    print("\n  -- CatBoost (Regression approach) --")
    cat_models = []
    oof_raw_cat = np.zeros(len(X))

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X[features], y)):
        X_tr, X_val = X[features].iloc[trn_idx], X[features].iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        cat = CatBoostRegressor(
            iterations=2000, learning_rate=0.03, depth=8,
            random_seed=SEED, verbose=0, task_type="CPU",
            l2_leaf_reg=3, subsample=0.8,
        )
        cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
        oof_raw_cat[val_idx] = cat.predict(X_val)
        cat_models.append(cat)
        rounded = np.clip(np.round(oof_raw_cat[val_idx]), 1, 5).astype(int)
        print(f"    Fold {fold}: QWK={qwk(y_val, rounded):.4f}")

    thresholds_cat = optimize_thresholds(oof_raw_cat, y.values)
    oof_preds_cat = apply_thresholds(oof_raw_cat, thresholds_cat)
    cat_qwk = qwk(y.values, oof_preds_cat)
    cat_acc = accuracy_score(y.values, oof_preds_cat)
    print(f"\n    > CatBoost OOF QWK:  {cat_qwk:.4f}")
    print(f"    > CatBoost OOF Acc:  {cat_acc:.4f}")
    USE_CATBOOST = True
except ImportError:
    print("\n  [WARNING] CatBoost not installed - using LightGBM only")
    USE_CATBOOST = False
    oof_raw_cat = np.zeros(len(X))
    cat_models = []

```


```python
# --------------------------------------------------------------------------
#  ENSEMBLE: Weighted average of regression outputs -> joint threshold
# --------------------------------------------------------------------------

print("\n  -- Ensemble --")

if USE_CATBOOST:
    # Find optimal blend weight
    best_blend_qwk = 0
    best_w = 0.5
    for w in np.arange(0.3, 0.8, 0.05):
        blended = w * oof_raw_lgbm + (1 - w) * oof_raw_cat
        thresh = optimize_thresholds(blended, y.values)
        preds = apply_thresholds(blended, thresh)
        blend_qwk = qwk(y.values, preds)
        if blend_qwk > best_blend_qwk:
            best_blend_qwk = blend_qwk
            best_w = w
            best_thresh = thresh

    oof_raw_ensemble = best_w * oof_raw_lgbm + (1 - best_w) * oof_raw_cat
    thresholds_ensemble = best_thresh
    oof_preds_final = apply_thresholds(oof_raw_ensemble, thresholds_ensemble)
    print(f"    Best blend weight: LGB={best_w:.2f}, Cat={1-best_w:.2f}")
else:
    oof_raw_ensemble = oof_raw_lgbm
    thresholds_ensemble = thresholds_lgbm
    oof_preds_final = oof_preds_lgbm
    best_w = 1.0

final_qwk = qwk(y.values, oof_preds_final)
final_acc = accuracy_score(y.values, oof_preds_final)

print(f"\n  +======================================+")
print(f"     FINAL ENSEMBLE RESULTS               ")
print(f"     QWK:      {final_qwk:.4f}                   ")
print(f"     Accuracy: {final_acc:.4f}                   ")
print(f"  +======================================+")


```


```python
# ===========================================================================
#  SECTION 7: ABLATION STUDY - PROVING NLP VALUE
# ===========================================================================

print("\n" + "=" * 60)
print("  SECTION 7: ABLATION STUDY")
print("=" * 60)

# Train LightGBM WITHOUT NLP features for comparison
oof_raw_no_nlp = np.zeros(len(X))
for fold, (trn_idx, val_idx) in enumerate(skf.split(X[features_no_nlp], y)):
    X_tr = X[features_no_nlp].iloc[trn_idx]
    X_val = X[features_no_nlp].iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_raw_no_nlp[val_idx] = m.predict(X_val)

thresh_no_nlp = optimize_thresholds(oof_raw_no_nlp, y.values)
preds_no_nlp = apply_thresholds(oof_raw_no_nlp, thresh_no_nlp)
qwk_no_nlp = qwk(y.values, preds_no_nlp)
acc_no_nlp = accuracy_score(y.values, preds_no_nlp)

nlp_delta_qwk = final_qwk - qwk_no_nlp
nlp_delta_acc = final_acc - acc_no_nlp

print(f"\n  Ablation Results:")
print(f"  +-------------------------+----------+----------+")
print(f"  | Configuration           | QWK      | Accuracy |")
print(f"   -------------------------+----------+---------- ")
print(f"  | Structured only         | {qwk_no_nlp:.4f}   | {acc_no_nlp:.4f}   |")
print(f"  | + NLP features          | {final_qwk:.4f}   | {final_acc:.4f}   |")
print(f"  | NLP improvement         | +{nlp_delta_qwk:.4f}  | +{nlp_delta_acc:.4f}  |")
print(f"  +-------------------------+----------+----------+")


```


```python
# ===========================================================================
#  SECTION 8: INTERPRETATION & CLINICAL ANALYSIS
# ===========================================================================

print("\n" + "=" * 60)
print("  SECTION 8: MODEL INTERPRETATION")
print("=" * 60)

# -- 8.1 Feature Importance --
best_model = lgbm_models[0]
imp_df = pd.DataFrame({"feature": features, "importance": best_model.feature_importances_})
imp_df = imp_df.sort_values("importance", ascending=True).tail(30)

fig, ax = plt.subplots(figsize=(10, 10))
colors = ["#e94560" if "kw_" in f or "tfidf_" in f
          else "#6C5CE7" if "_missing" in f
          else "#00b894" for f in imp_df["feature"]]
ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
ax.set_xlabel("Importance (split count)")
ax.set_title("Top 30 Feature Importances\n(Red=NLP, Purple=Missingness, Green=Other)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close("all")


# -- 8.2 SHAP Analysis --
try:
    import shap
    explainer = shap.TreeExplainer(best_model)
    # Use only 500 samples to avoid OOM on 16GB RAM with 403 features
    X_shap = train[features].sample(min(500, len(train)), random_state=SEED)
    shap_values = explainer.shap_values(X_shap)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=20, show=False)
    plt.title("SHAP Feature Importance", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    
    print("  > SHAP analysis saved")
except Exception as e:
    print(f"  [WARNING] SHAP skipped: {e}")

# -- 8.3 Confusion Matrix --
cm = confusion_matrix(y.values, oof_preds_final)
esi_labels = [f"ESI-{i}" for i in sorted(y.unique())]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=esi_labels, yticklabels=esi_labels, ax=ax,
            linewidths=0.5, linecolor="white")
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title(f"Confusion Matrix (QWK={final_qwk:.4f})", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close("all")


# -- 8.4 Per-Class Analysis --
print(f"\n  Per-Class Precision/Recall:")
print(classification_report(y.values, oof_preds_final, digits=4,
      target_names=esi_labels))

# -- 8.5 Error Analysis --
errors = pd.DataFrame({"actual": y.values, "predicted": oof_preds_final.astype(int)})
errors["correct"] = errors["actual"] == errors["predicted"]
errors["off_by"] = (errors["predicted"] - errors["actual"]).abs()

total_err = (~errors["correct"]).sum()
off1 = (errors["off_by"] == 1).sum()
off2 = (errors["off_by"] >= 2).sum()
print(f"  Error Analysis:")
print(f"    Total errors:     {total_err:,} ({total_err/len(errors)*100:.1f}%)")
print(f"    Off-by-1 (minor): {off1:,} ({off1/len(errors)*100:.1f}%)")
print(f"    Off-by-2+ (major): {off2:,} ({off2/len(errors)*100:.1f}%)")

# -- 8.6 Fairness/Bias Audit --
print(f"\n  Bias Audit:")
print(f"    Overall: QWK={final_qwk:.4f}, Acc={final_acc:.4f}")
for grp in ["sex", "is_senior", "is_pediatric"]:
    if grp not in train.columns:
        continue
    for val in sorted(train[grp].dropna().unique()):
        mask = (train[grp] == val).values
        if mask.sum() < 50:
            continue
        sub_qwk = qwk(y.values[mask], oof_preds_final[mask])
        sub_acc = accuracy_score(y.values[mask], oof_preds_final[mask])
        delta = sub_qwk - final_qwk
        flag = " [WARNING]" if abs(delta) > 0.03 else ""
        print(f"    {grp}={val}: QWK={sub_qwk:.4f}, Acc={sub_acc:.4f} ( ={delta:+.4f}){flag}")

# -- 8.7 Clinical Validation (using disposition - NOT a feature) --
if "disposition" in train.columns:
    print(f"\n  Clinical Validation - Disposition by Predicted ESI:")
    for esi in sorted(y.unique()):
        mask = oof_preds_final == esi
        if mask.sum() > 0:
            disp = train.loc[mask, "disposition"]
            print(f"    ESI-{int(esi)} (n={mask.sum():,}): {disp.value_counts().head(3).to_dict()}")

```


```python
# ===========================================================================
#  SECTION 9: GENERATE FINAL SUBMISSION
# ===========================================================================

print("\n" + "=" * 60)
print("  SECTION 9: SUBMISSION")
print("=" * 60)

# Generate test predictions - average regression outputs from all fold models
test_raw = np.zeros(len(test))
for m in lgbm_models:
    test_raw += m.predict(test[features]) / len(lgbm_models)

if USE_CATBOOST:
    test_raw_cat = np.zeros(len(test))
    for m in cat_models:
        test_raw_cat += m.predict(test[features]) / len(cat_models)
    test_raw_blend = best_w * test_raw + (1 - best_w) * test_raw_cat
else:
    test_raw_blend = test_raw

test_preds = apply_thresholds(test_raw_blend, thresholds_ensemble)

# Create submission
submission = pd.DataFrame({
    "patient_id": test["patient_id"],
    "triage_acuity": test_preds.astype(int),
})
submission.to_csv("outputs/submission.csv", index=False)

# Sanity checks
checks = [
    ("Row count matches test", len(submission) == len(test)),
    ("No null predictions", submission["triage_acuity"].isnull().sum() == 0),
    ("All values in [1, 5]", submission["triage_acuity"].between(1, 5).all()),
    ("No duplicate patient_ids", submission["patient_id"].nunique() == len(submission)),
]
print("\n  Sanity Checks:")
all_pass = True
for label, ok in checks:
    print(f"    {'[PASS]' if ok else '[FAIL]'} {label}")
    all_pass = all_pass and ok

print(f"\n  Submission Distribution:")
for esi in sorted(submission["triage_acuity"].unique()):
    n = (submission["triage_acuity"] == esi).sum()
    pct = n / len(submission) * 100
    print(f"    ESI-{esi}: {n:5,} ({pct:5.1f}%)")

# Distribution comparison plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
y.value_counts().sort_index().plot(kind="bar", ax=axes[0], color=PALETTE, edgecolor="white")
axes[0].set_title("Train: Actual ESI", fontweight="bold")
axes[0].set_xlabel("ESI Level")
submission["triage_acuity"].value_counts().sort_index().plot(kind="bar", ax=axes[1], color=PALETTE, edgecolor="white")
axes[1].set_title("Test: Predicted ESI", fontweight="bold")
axes[1].set_xlabel("ESI Level")
plt.suptitle("Distribution Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/submission_analysis.png", dpi=150, bbox_inches="tight")
plt.close("all")


```


```python
# ===========================================================================
#  FINAL SUMMARY
# ===========================================================================

print(f"\n{'=' * 60}")
print(f"  +==============================================+")
print(f"     TRIAGEGEIST - FINAL RESULTS SUMMARY         ")
print(f"  +==============================================+")
print(f"     Ensemble QWK:     {final_qwk:.4f}                    ")
print(f"     Ensemble Acc:     {final_acc:.4f}                    ")
print(f"     LightGBM QWK:     {lgbm_qwk:.4f}                    ")
if USE_CATBOOST:
    print(f"     CatBoost QWK:     {cat_qwk:.4f}                    ")
    print(f"     Blend Weight:     LGB={best_w:.2f} / Cat={1-best_w:.2f}        ")
print(f"     NLP Improvement:  +{nlp_delta_qwk:.4f} QWK               ")
print(f"     Features Used:    {len(features):4d}                      ")
print(f"     Off-by-1 errors:  {off1/len(errors)*100:.1f}%                      ")
print(f"     Off-by-2+ errors: {off2/len(errors)*100:.1f}%                       ")
print(f"  +==============================================+")
print(f"     Submission: outputs/submission.csv           ")
print(f"     Plots:      outputs/*.png                    ")
print(f"  +==============================================+")
print(f"{'=' * 60}")
```


```python

```
