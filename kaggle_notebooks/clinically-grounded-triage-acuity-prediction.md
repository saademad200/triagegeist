# TriageGeist: Clinically Grounded Acuity Prediction from Structured Triage and Free Text

## Clinical Problem Statement
This notebook predicts `triage_acuity` (1 = most urgent, 5 = least urgent) using information available at or near emergency department intake. The goal is not to replace clinician judgment, but to study whether structured triage measurements plus free-text complaints can support early acuity assessment.

## Data Disclosure
This notebook uses the following competition files:
- `train.csv` and `test.csv` for structured triage features and the target label
- `chief_complaints.csv` for free-text chief complaint narratives
- `patient_history.csv` for binary comorbidity indicators

Two post-triage fields, `disposition` and `ed_los_hours`, are removed explicitly because they would leak downstream information unavailable at triage time.

## Modeling Summary
The pipeline combines:
- clinically motivated feature engineering from vital signs
- explicit missingness handling for `pain_score`
- TF-IDF plus lightweight rule-based symptom tokens from chief complaint text
- 5-fold stratified LightGBM with out-of-fold evaluation

## Why This Version Is Submission-Ready
This notebook is designed to run end-to-end, report cross-validated metrics, show clinically interpretable diagnostics, and save `submission.csv` for Kaggle upload.
        



```python
from pathlib import Path
import re
import warnings
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

RANDOM_STATE = 42
N_SPLITS = 5
TEXT_MAX_FEATURES = 50
TOKEN_MIN_FREQ = 10
DROP_PROXY_COLS = ["triage_nurse_id", "site_id", "language", "insurance_type"]
POST_TRIAGE_LEAKAGE_COLS = ["disposition", "ed_los_hours"]

sns.set_theme(style="whitegrid", context="talk")


def resolve_data_dir() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/triagegeist"),
        Path("/kaggle/input/triagegeist"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the Triagegeist data directory. "
        "Update `resolve_data_dir()` with the correct path for your environment."
    )


DATA_DIR = resolve_data_dir()
print(f"Using data directory: {DATA_DIR}")
        

```


```python
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
history = pd.read_csv(DATA_DIR / "patient_history.csv")
complaints = pd.read_csv(DATA_DIR / "chief_complaints.csv")

dataset_summary = pd.DataFrame(
    [
        {"table": "train", "rows": len(train), "cols": train.shape[1]},
        {"table": "test", "rows": len(test), "cols": test.shape[1]},
        {"table": "patient_history", "rows": len(history), "cols": history.shape[1]},
        {"table": "chief_complaints", "rows": len(complaints), "cols": complaints.shape[1]},
    ]
)
print(dataset_summary.to_string(index=False))

class_balance = (
    train["triage_acuity"]
    .value_counts(normalize=True)
    .sort_index()
    .rename("share")
    .reset_index()
    .rename(columns={"index": "triage_acuity"})
)
class_balance["share"] = class_balance["share"].round(4)
print()
print("Triage acuity distribution:")
print(class_balance.to_string(index=False))
        

```

## Exploratory Analysis
The dataset is strongly structured around triage physiology. Before training, it is useful to confirm three patterns that are clinically plausible and important for model design:

1. High-acuity patients cluster in worse hemodynamic and respiratory states.
2. Missingness is informative, especially for lower-acuity workflows.
3. The label distribution is imbalanced, so evaluation should go beyond raw accuracy.
        



```python
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

sns.countplot(data=train, x="triage_acuity", ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title("Class Distribution")
axes[0, 0].set_xlabel("Triage acuity")
axes[0, 0].set_ylabel("Count")

plot_df = train.sample(n=min(6000, len(train)), random_state=RANDOM_STATE).copy()
plot_df["acuity_group"] = np.where(plot_df["triage_acuity"] <= 2, "High acuity (1-2)", "Lower acuity (3-5)")
sns.scatterplot(
    data=plot_df,
    x="systolic_bp",
    y="heart_rate",
    hue="acuity_group",
    alpha=0.45,
    s=30,
    ax=axes[0, 1],
)
axes[0, 1].axhline(120, color="black", linestyle="--", linewidth=1)
axes[0, 1].axvline(90, color="black", linestyle="--", linewidth=1)
axes[0, 1].set_title("Hemodynamics at Triage")
axes[0, 1].set_xlabel("Systolic blood pressure")
axes[0, 1].set_ylabel("Heart rate")

sns.boxenplot(data=train, x="triage_acuity", y="spo2", ax=axes[1, 0], palette="mako")
axes[1, 0].set_title("SpO2 by Triage Acuity")
axes[1, 0].set_xlabel("Triage acuity")
axes[1, 0].set_ylabel("SpO2")

missing_summary = pd.DataFrame(
    {
        "feature": ["systolic_bp", "heart_rate", "respiratory_rate", "temperature_c", "spo2", "pain_score_encoded_missing"],
        "missing_rate": [
            train["systolic_bp"].isna().mean(),
            train["heart_rate"].isna().mean(),
            train["respiratory_rate"].isna().mean(),
            train["temperature_c"].isna().mean(),
            train["spo2"].isna().mean(),
            (train["pain_score"] == -1).mean(),
        ],
    }
)
sns.barplot(data=missing_summary, x="missing_rate", y="feature", ax=axes[1, 1], palette="crest")
axes[1, 1].set_title("Observed Missingness at Intake")
axes[1, 1].set_xlabel("Missing rate")
axes[1, 1].set_ylabel("")

plt.tight_layout()
plt.show()
        

```

## Preprocessing and Feature Engineering
The preprocessing pipeline reflects the triage setting:

- `disposition` and `ed_los_hours` are dropped as post-triage leakage.
- `pain_score = -1` is treated as missing and a separate missingness flag is added.
- Administrative proxy columns (`triage_nurse_id`, `site_id`, `language`, `insurance_type`) are excluded to keep the model focused on clinically meaningful inputs and reduce reliance on site- or workflow-specific artifacts.
- Free-text complaints are represented with both TF-IDF and a lightweight rule-based tokenizer.
- Clinically motivated interaction terms are added for hemodynamics, oxygenation, and age-adjusted severity.
        



```python
class MedicalTokenizer:
    def __init__(self, min_freq=TOKEN_MIN_FREQ):
        self.min_freq = min_freq
        self.tokens = []
        self.connectors = [r"\bwith\b", r"\bworsening\b"]
        self.pattern = "|".join(self.connectors)

    def _clean_and_split(self, text):
        if pd.isna(text) or text == "":
            return []
        text = str(text).lower().replace("，", ",")
        text = re.sub(r"\bin known patient\b", "", text)
        parts = [part.strip() for part in text.split(",")]

        final_parts = []
        for part in parts:
            sub_parts = re.split(self.pattern, part)
            final_parts.extend([sub.strip() for sub in sub_parts if sub.strip()])
        return final_parts

    def fit(self, series):
        counts = series.apply(self._clean_and_split).explode().value_counts()
        self.tokens = sorted(counts[counts >= self.min_freq].index.tolist())
        print(f"Extracted {len(self.tokens)} clinical tokens")
        return self

    def transform(self, series):
        multi_hot = np.zeros((len(series), len(self.tokens)), dtype=np.int8)
        token_to_idx = {token: i for i, token in enumerate(self.tokens)}

        for row_idx, text in enumerate(series):
            for token in self._clean_and_split(text):
                col_idx = token_to_idx.get(token)
                if col_idx is not None:
                    multi_hot[row_idx, col_idx] = 1

        return pd.DataFrame(
            multi_hot,
            columns=[f"token_{token.replace(' ', '_')}" for token in self.tokens],
        )


def apply_clinical_fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "pain_score" in df.columns:
        df["pain_score_missing"] = (df["pain_score"] == -1).astype(np.int8)
        df["pain_score"] = df["pain_score"].replace(-1, np.nan)

    df["fe_msi"] = df["heart_rate"] / (df["mean_arterial_pressure"] + 1e-5)
    df["fe_rpp"] = df["heart_rate"] * df["systolic_bp"]
    df["fe_spo2_rr_ratio"] = df["spo2"] / (df["respiratory_rate"] + 1e-5)
    df["fe_asi"] = df["age"] * (df["heart_rate"] / (df["systolic_bp"] + 1e-5))
    df["fe_is_geriatric"] = (df["age"] >= 65).astype(np.int8)
    df["fe_is_pediatric"] = (df["age"] < 12).astype(np.int8)
    df["fe_hr_class"] = pd.cut(df["heart_rate"], bins=[0, 60, 100, 300], labels=[0, 1, 2]).astype(float)
    df["fe_temp_class"] = pd.cut(
        df["temperature_c"],
        bins=[0, 36, 37.5, 38.5, 50],
        labels=[0, 1, 2, 3],
    ).astype(float)

    return df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame | None = None):
    train_df = train_df.copy()
    test_df = None if test_df is None else test_df.copy()
    encoder_maps = {}
    skip_cols = {"patient_id", "chief_complaint_raw"}

    categorical_cols = []
    for column in train_df.columns:
        dtype_name = str(train_df[column].dtype)
        if column in skip_cols:
            continue
        if (
            dtype_name == "object"
            or dtype_name == "str"
            or dtype_name.startswith("string")
            or dtype_name == "category"
        ):
            categorical_cols.append(column)

    for column in categorical_cols:
        train_values = train_df[column].fillna("__missing__").astype(str)
        if test_df is not None:
            test_values = test_df[column].fillna("__missing__").astype(str)
            categories = pd.Index(sorted(set(train_values).union(set(test_values))))
        else:
            categories = pd.Index(sorted(set(train_values)))

        mapping = {value: idx for idx, value in enumerate(categories)}
        train_df[column] = train_values.map(mapping).astype("int32")
        if test_df is not None:
            test_df[column] = test_values.map(mapping).fillna(-1).astype("int32")
        encoder_maps[column] = mapping

    return train_df, test_df, encoder_maps


def preprocess_pipeline(
    train_df: pd.DataFrame,
    history_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
):
    complaint_text = complaints_df[["patient_id", "chief_complaint_raw"]].copy()

    full_train = (
        train_df
        .merge(history_df, on="patient_id", how="left")
        .merge(complaint_text, on="patient_id", how="left")
    )
    full_test = None
    if test_df is not None:
        full_test = (
            test_df
            .merge(history_df, on="patient_id", how="left")
            .merge(complaint_text, on="patient_id", how="left")
        )

    full_train = full_train.drop(columns=[col for col in POST_TRIAGE_LEAKAGE_COLS if col in full_train.columns])
    full_train = full_train.drop(columns=[col for col in DROP_PROXY_COLS if col in full_train.columns])
    if full_test is not None:
        full_test = full_test.drop(columns=[col for col in DROP_PROXY_COLS if col in full_test.columns])

    full_train["chief_complaint_raw"] = full_train["chief_complaint_raw"].fillna("")
    if full_test is not None:
        full_test["chief_complaint_raw"] = full_test["chief_complaint_raw"].fillna("")

    tfidf = TfidfVectorizer(max_features=TEXT_MAX_FEATURES, stop_words="english")
    tfidf_train = tfidf.fit_transform(full_train["chief_complaint_raw"])
    tfidf_train_df = pd.DataFrame(
        tfidf_train.toarray(),
        columns=[f"tfidf_{term}" for term in tfidf.get_feature_names_out()],
    )

    tokenizer = MedicalTokenizer(min_freq=TOKEN_MIN_FREQ).fit(full_train["chief_complaint_raw"])
    token_train_df = tokenizer.transform(full_train["chief_complaint_raw"])

    tfidf_test_df = None
    token_test_df = None
    if full_test is not None:
        tfidf_test = tfidf.transform(full_test["chief_complaint_raw"])
        tfidf_test_df = pd.DataFrame(
            tfidf_test.toarray(),
            columns=[f"tfidf_{term}" for term in tfidf.get_feature_names_out()],
        )
        token_test_df = tokenizer.transform(full_test["chief_complaint_raw"])

    full_train = apply_clinical_fe(full_train)
    if full_test is not None:
        full_test = apply_clinical_fe(full_test)

    full_train, full_test, encoder_maps = encode_categoricals(full_train, full_test)

    X = pd.concat(
        [
            full_train.drop(columns=["patient_id", "triage_acuity", "chief_complaint_raw"]),
            tfidf_train_df,
            token_train_df,
        ],
        axis=1,
    )
    y = full_train["triage_acuity"].astype(int) - 1

    X_test = None
    if full_test is not None:
        X_test = pd.concat(
            [
                full_test.drop(columns=["patient_id", "chief_complaint_raw"]),
                tfidf_test_df,
                token_test_df,
            ],
            axis=1,
        )
        X_test = X_test.reindex(columns=X.columns, fill_value=0)

    artifacts = {
        "tfidf": tfidf,
        "tokenizer": tokenizer,
        "encoder_maps": encoder_maps,
        "feature_names": X.columns.tolist(),
    }
    return X, y, X_test, artifacts
        

```

## Cross-Validated Modeling
The main submission model is a 5-fold stratified LightGBM ensemble. We evaluate the notebook using out-of-fold predictions, which gives a better estimate of internal performance than a single train/validation split.

In addition to `multi_logloss`, this notebook reports:
- quadratic weighted kappa (QWK), which is appropriate for ordered acuity labels
- overall accuracy
- acute recall for classes 1-2
- acute undertriage rate, where truly acute cases are predicted as lower acuity
        



```python
X, y, X_test, artifacts = preprocess_pipeline(train, history, complaints, test)
print(f"Training matrix shape: {X.shape}")
print(f"Test matrix shape: {X_test.shape}")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

lgb_params = {
    "objective": "multiclass",
    "num_class": 5,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "n_estimators": 500,
    "verbosity": -1,
    "random_state": RANDOM_STATE,
}

oof_proba = np.zeros((len(X), 5))
oof_pred = np.zeros(len(X), dtype=int)
test_proba = np.zeros((len(X_test), 5))
fold_metrics = []
feature_importance_frames = []
models = []
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    valid_proba = model.predict_proba(X_valid)
    valid_pred = np.argmax(valid_proba, axis=1)

    oof_proba[val_idx] = valid_proba
    oof_pred[val_idx] = valid_pred
    test_proba += model.predict_proba(X_test) / N_SPLITS
    models.append(model)
    best_iterations.append(model.best_iteration_)

    fold_metrics.append(
        {
            "fold": fold,
            "best_iteration": model.best_iteration_,
            "logloss": log_loss(y_valid, valid_proba, labels=[0, 1, 2, 3, 4]),
            "qwk": cohen_kappa_score(y_valid, valid_pred, weights="quadratic"),
            "accuracy": accuracy_score(y_valid, valid_pred),
        }
    )

    feature_importance_frames.append(
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": model.feature_importances_,
                "fold": fold,
            }
        )
    )

fold_metrics_df = pd.DataFrame(fold_metrics)
print(fold_metrics_df.round(6).to_string(index=False))

acute_true = (y <= 1).astype(int)
acute_pred = (oof_pred <= 1).astype(int)
summary_metrics = pd.Series(
    {
        "oof_logloss": log_loss(y, oof_proba, labels=[0, 1, 2, 3, 4]),
        "oof_qwk": cohen_kappa_score(y, oof_pred, weights="quadratic"),
        "oof_accuracy": accuracy_score(y, oof_pred),
        "acute_recall_classes_1_2": recall_score(acute_true, acute_pred),
        "acute_undertriage_rate": ((acute_true == 1) & (acute_pred == 0)).sum() / max(acute_true.sum(), 1),
    }
).round(6)

print()
print("Overall out-of-fold metrics:")
print(summary_metrics.to_string())

conf_mat = confusion_matrix(y + 1, oof_pred + 1, labels=[1, 2, 3, 4, 5])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Out-of-Fold Confusion Matrix")
plt.xlabel("Predicted acuity")
plt.ylabel("True acuity")
plt.show()

feature_importance_df = pd.concat(feature_importance_frames, ignore_index=True)
mean_feature_importance = (
    feature_importance_df.groupby("feature", as_index=False)["importance"]
    .mean()
    .sort_values("importance", ascending=False)
)

plt.figure(figsize=(10, 8))
sns.barplot(
    data=mean_feature_importance.head(20),
    x="importance",
    y="feature",
    palette="viridis",
)
plt.title("Mean LightGBM Feature Importance Across Folds")
plt.xlabel("Mean importance")
plt.ylabel("")
plt.tight_layout()
plt.show()
        

```

## Explainability
To keep explanation aligned with the submission model family, we refit a reference LightGBM model on the full training set using the average best iteration from cross-validation. SHAP is used only if the package is available in the current runtime; otherwise the notebook still completes successfully.
        



```python
reference_iterations = int(np.mean(best_iterations))
reference_model = lgb.LGBMClassifier(
    **{
        **lgb_params,
        "n_estimators": max(reference_iterations, 50),
    }
)
reference_model.fit(X, y)

if SHAP_AVAILABLE:
    shap_sample = X.sample(n=min(1000, len(X)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(reference_model)
    shap_values = explainer.shap_values(shap_sample)

    if isinstance(shap_values, list):
        acute_shap_values = shap_values[0]
    elif getattr(shap_values, "ndim", 0) == 3:
        acute_shap_values = shap_values[:, :, 0]
    else:
        acute_shap_values = shap_values

    shap.summary_plot(acute_shap_values, shap_sample, show=False)
    plt.title("SHAP Summary for Class 1 (Most Urgent)")
    plt.tight_layout()
    plt.show()
else:
    print("SHAP is not installed in this environment. Feature importance above can still be used for writeup discussion.")
        

```

## Limitations and Reproducibility Notes
- This is internal validation on a synthetic competition dataset, not external validation on real emergency department data.
- Very high internal performance may indicate that the synthetic label-generation process is highly learnable; results should not be interpreted as ready-for-clinic performance.
- The text pipeline is deliberately lightweight and interpretable. It does not capture full negation handling, chronology, or more complex symptom composition.
- Administrative proxies were dropped on purpose to improve generalizability and reduce the risk of learning site- or workflow-specific shortcuts.
- Reproducibility: the notebook uses `RANDOM_STATE = 42`, fixed fold splits, explicit leakage removal, and writes a deterministic `submission.csv` from averaged fold predictions.
        



```python
final_predictions = np.argmax(test_proba, axis=1) + 1
submission = pd.DataFrame(
    {
        "patient_id": test["patient_id"],
        "triage_acuity": final_predictions,
    }
)

submission.to_csv("submission.csv", index=False)
fold_metrics_df.to_csv("cv_metrics.csv", index=False)
mean_feature_importance.to_csv("feature_importance.csv", index=False)

print("Saved submission.csv, cv_metrics.csv, and feature_importance.csv")
print()
print(submission.head().to_string(index=False))
        

```
