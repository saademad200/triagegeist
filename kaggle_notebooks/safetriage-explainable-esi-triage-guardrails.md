# SafeTriage: Explainable ESI Prediction with NEWS2, Shortcut Audit, and Undertriage Guardrails

This notebook builds a proof-of-concept clinical decision-support system for emergency triage.

The system does four things:

1. predicts ESI acuity level from intake data;
2. explains the decision with clinical features and model importance;
3. flags dangerous undertriage risk;
4. audits subgroup performance for safety monitoring.

The goal is not to replace a triage nurse. The goal is to build a second-read system that can reduce silent undertriage and make model behavior inspectable.

# 1. Imports and configuration

All libraries used in the notebook are imported in the first code cell. Paths are written directly for Kaggle, without environment switching or hidden directory logic.


```python
# If your environment is missing packages, uncomment this line.
# !pip install -q catboost ipywidgets

import re
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import sparse

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    cohen_kappa_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import shap

import ipywidgets as widgets
from IPython.display import display, Markdown, HTML, clear_output

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)
sns.set_theme(style="whitegrid")
```


```python
SEED = 42
N_FOLDS = 5
TARGET = "triage_acuity"
ID_COL = "patient_id"

random.seed(SEED)
np.random.seed(SEED)
```


```python
TRAIN_PATH = "/kaggle/input/competitions/triagegeist/train.csv"
TEST_PATH = "/kaggle/input/competitions/triagegeist/test.csv"
HISTORY_PATH = "/kaggle/input/competitions/triagegeist/patient_history.csv"
COMPLAINTS_PATH = "/kaggle/input/competitions/triagegeist/chief_complaints.csv"
SUBMISSION_PATH = "/kaggle/input/competitions/triagegeist/sample_submission.csv"
NOTE_PATH = "/kaggle/input/competitions/triagegeist/NOTE.md"
```

# 2. Clinical problem statement

Emergency triage is a high-pressure decision process. The most dangerous model failure is not just a wrong class. It is undertriage: a patient who needs urgent care is assigned a less urgent category and waits longer than they should.

This notebook frames the task as ordinal ESI prediction plus safety monitoring.

ESI is a five-level emergency department triage algorithm:

- ESI-1 means most urgent;
- ESI-5 means least urgent;
- the scale is ordered, so errors have different severity depending on distance.

NEWS2 is retained because it is available at triage time. It is computed from current physiological measurements, not from post-triage outcome information.

# 3. Data disclosure

The uploaded `NOTE.md` says this hackathon has no official provided dataset and asks participants to refer to the Kaggle competition data page for inspiration. Therefore this notebook treats the CSV files as a proof-of-concept triage dataset.

The notebook focuses on clinical framing, leakage control, reproducible modeling, safety metrics, explainability, subgroup audit, and a working simulator. It does not claim deployment readiness.

Files used:

| File | Role | Used for |
|---|---|---|
| `train.csv` | labeled patient intake data | training and validation |
| `test.csv` | unlabeled patient intake data | final prediction |
| `patient_history.csv` | comorbidity flags | history features |
| `chief_complaints.csv` | complaint text and system | NLP features |
| `sample_submission.csv` | required format | submission validation |
| `NOTE.md` | competition data note | disclosure |

# 4. Data loading


```python
train_raw = pd.read_csv("/kaggle/input/competitions/triagegeist/train.csv")
```


```python
train_raw.shape
```


```python
train_raw.head()
```


```python
test_raw = pd.read_csv("/kaggle/input/competitions/triagegeist/test.csv")
```


```python
test_raw.shape
```


```python
test_raw.head()
```


```python
history = pd.read_csv("/kaggle/input/competitions/triagegeist/patient_history.csv")
```


```python
history.shape
```


```python
history.head()
```


```python
complaints = pd.read_csv("/kaggle/input/competitions/triagegeist/chief_complaints.csv")
```


```python
complaints.shape
```


```python
complaints.head()
```


```python
sample_submission = pd.read_csv("/kaggle/input/competitions/triagegeist/sample_submission.csv")
```


```python
sample_submission.head()
```

# 5. Basic integrity checks


```python
print("train rows:", len(train_raw))
print("test rows:", len(test_raw))
print("history rows:", len(history))
print("complaints rows:", len(complaints))
```


```python
train_raw[ID_COL].is_unique, test_raw[ID_COL].is_unique, history[ID_COL].is_unique, complaints[ID_COL].is_unique
```


```python
train_raw[TARGET].value_counts().sort_index()
```


```python
train_raw[TARGET].between(1, 5).all()
```


```python
train_only_cols = sorted(list(set(train_raw.columns) - set(test_raw.columns)))
train_only_cols
```

# 6. Leakage audit

The columns `disposition` and `ed_los_hours` exist in train but not in test. More importantly, they are post-triage information.

- `disposition` is the final ED outcome or route.
- `ed_los_hours` is the length of stay after the triage process.

Both are dropped before modeling.

`news2_score` is different. It is derived from physiological measurements available during acute assessment, so we keep it as a valid triage-time clinical score.


```python
leakage_cols = ["disposition", "ed_los_hours"]
leakage_cols
```


```python
train_raw[leakage_cols + [TARGET]].head()
```


```python
train_raw[["ed_los_hours", TARGET]].corr()
```


```python
plt.figure(figsize=(8, 4))
sns.boxplot(data=train_raw, x=TARGET, y="ed_los_hours")
plt.title("Post-triage leakage check: ED length of stay by ESI")
plt.xlabel("ESI acuity")
plt.ylabel("ED length of stay, hours")
plt.show()
```


```python
availability_table = pd.DataFrame({
    "feature_group": [
        "demographics", "arrival", "vitals", "derived vitals", "history", 
        "complaint text", "post-triage outcome", "post-triage duration"
    ],
    "examples": [
        "age, sex", "arrival_mode, shift", "BP, HR, RR, SpO2", "shock_index, MAP, NEWS2", 
        "hx_*", "chief_complaint_raw", "disposition", "ed_los_hours"
    ],
    "available_at_triage": ["yes", "yes", "yes", "yes", "usually yes", "yes", "no", "no"],
    "decision": ["use", "use", "use", "use", "use", "use", "drop", "drop"]
})

availability_table
```

# 7. Merge data


```python
train = train_raw.drop(columns=leakage_cols).copy()
```


```python
test = test_raw.copy()
```


```python
train.shape, test.shape
```


```python
train = train.merge(history, on=ID_COL, how="left")
```


```python
test = test.merge(history, on=ID_COL, how="left")
```


```python
train.shape, test.shape
```


```python
complaints_small = complaints[[ID_COL, "chief_complaint_raw"]].copy()
```


```python
train = train.merge(complaints_small, on=ID_COL, how="left")
```


```python
test = test.merge(complaints_small, on=ID_COL, how="left")
```


```python
train.shape, test.shape
```


```python
train.head()
```


```python
test.head()
```


```python
train.isna().mean().sort_values(ascending=False).head(20)
```

# 8. EDA

## 8.1 Target distribution


```python
target_counts = train[TARGET].value_counts().sort_index()
target_counts
```


```python
target_share = (target_counts / target_counts.sum()).rename("share")
target_share
```


```python
plt.figure(figsize=(7, 4))
sns.barplot(x=target_counts.index, y=target_counts.values)
plt.title("ESI acuity distribution")
plt.xlabel("ESI acuity: 1 most urgent, 5 least urgent")
plt.ylabel("patient count")
plt.show()
```

## 8.2 Vital signs by ESI


```python
vital_cols = [
    "news2_score", "gcs_total", "spo2", "heart_rate", "systolic_bp", 
    "respiratory_rate", "temperature_c", "pain_score", "shock_index"
]

vital_cols
```


```python
train[vital_cols].describe().T
```


```python
for col in vital_cols:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=train, x=TARGET, y=col)
    plt.title(f"{col} by ESI acuity")
    plt.xlabel("ESI acuity")
    plt.ylabel(col)
    plt.show()
```

## 8.3 Missingness as signal


```python
missable_cols = ["systolic_bp", "diastolic_bp", "respiratory_rate", "temperature_c", "spo2"]
missable_cols
```


```python
missing_by_esi = train.groupby(TARGET)[missable_cols].apply(lambda x: x.isna().mean()).T * 100
missing_by_esi
```


```python
plt.figure(figsize=(10, 4))
sns.heatmap(missing_by_esi, annot=True, fmt=".1f", cmap="Oranges")
plt.title("Missing vital signs by ESI acuity, percent")
plt.xlabel("ESI acuity")
plt.ylabel("feature")
plt.show()
```


```python
pain_not_recorded_by_esi = train.assign(pain_not_recorded=(train["pain_score"] == -1)).groupby(TARGET)["pain_not_recorded"].mean() * 100
pain_not_recorded_by_esi
```


```python
plt.figure(figsize=(7, 4))
sns.barplot(x=pain_not_recorded_by_esi.index, y=pain_not_recorded_by_esi.values)
plt.title("Pain score not recorded by ESI")
plt.xlabel("ESI acuity")
plt.ylabel("not recorded, percent")
plt.show()
```

## 8.4 Chief complaint system


```python
complaint_system_table = pd.crosstab(train["chief_complaint_system"], train[TARGET], normalize="index") * 100
complaint_system_table.head()
```


```python
complaint_system_counts = train["chief_complaint_system"].value_counts()
complaint_system_counts.head(20)
```


```python
top_systems = complaint_system_counts.head(12).index
complaint_system_plot = train[train["chief_complaint_system"].isin(top_systems)].copy()
```


```python
plt.figure(figsize=(12, 5))
sns.countplot(data=complaint_system_plot, x="chief_complaint_system", hue=TARGET)
plt.title("Top chief complaint systems by ESI")
plt.xlabel("chief complaint system")
plt.ylabel("count")
plt.xticks(rotation=45, ha="right")
plt.legend(title="ESI")
plt.show()
```

## 8.5 Raw chief complaint text


```python
train["chief_complaint_raw"].fillna("").head(10)
```


```python
train["chief_complaint_len"] = train["chief_complaint_raw"].fillna("").str.len()
test["chief_complaint_len"] = test["chief_complaint_raw"].fillna("").str.len()
```


```python
train.groupby(TARGET)["chief_complaint_len"].describe()
```


```python
plt.figure(figsize=(7, 4))
sns.boxplot(data=train, x=TARGET, y="chief_complaint_len")
plt.title("Chief complaint text length by ESI")
plt.xlabel("ESI acuity")
plt.ylabel("text length")
plt.show()
```

## 8.6 Clinical keyword prevalence


```python
keyword_patterns = {
    "kw_life_threat": r"arrest|shock|unconscious|unresponsive|cpr|resuscitat|apnea",
    "kw_cardio": r"chest pain|chest tightness|stemi|myocardial|cardiac|palpitation",
    "kw_resp": r"dyspnoea|dyspnea|shortness of breath|sob|wheeze|respiratory|breath",
    "kw_neuro": r"stroke|seizure|syncope|thunderclap|weakness|confusion|altered",
    "kw_trauma": r"trauma|fracture|mva|accident|blunt|laceration|bleeding|haemorrhage|hemorrhage",
    "kw_infection": r"sepsis|fever|rigors|infection|cellulitis|meningitis",
    "kw_low_acuity": r"mild|routine|refill|follow.?up|chronic|advice|review",
    "kw_time_sensitive": r"anaphylaxis|aortic|torsion|eclampsia|pulmonary embolism|pe\b",
}

keyword_patterns
```


```python
text_lower = train["chief_complaint_raw"].fillna("").str.lower()

keyword_eda = pd.DataFrame(index=train.index)
for name, pattern in keyword_patterns.items():
    keyword_eda[name] = text_lower.str.contains(pattern, regex=True).astype(int)

keyword_eda[TARGET] = train[TARGET].values
keyword_eda.head()
```


```python
keyword_by_esi = keyword_eda.groupby(TARGET).mean().T * 100
keyword_by_esi
```


```python
plt.figure(figsize=(9, 5))
sns.heatmap(keyword_by_esi, annot=True, fmt=".1f", cmap="Blues")
plt.title("Chief complaint keyword prevalence by ESI, percent")
plt.xlabel("ESI acuity")
plt.ylabel("keyword group")
plt.show()
```

## 8.7 Comorbidity burden


```python
hx_cols = [col for col in train.columns if col.startswith("hx_")]
len(hx_cols), hx_cols[:10]
```


```python
train["comorbidity_count_eda"] = train[hx_cols].fillna(0).sum(axis=1)
test["comorbidity_count_eda"] = test[hx_cols].fillna(0).sum(axis=1)
```


```python
train.groupby(TARGET)["comorbidity_count_eda"].describe()
```


```python
plt.figure(figsize=(7, 4))
sns.boxplot(data=train, x=TARGET, y="comorbidity_count_eda")
plt.title("Comorbidity burden by ESI")
plt.xlabel("ESI acuity")
plt.ylabel("number of comorbidities")
plt.show()
```


```python
burden_curve = train.groupby("comorbidity_count_eda")[TARGET].mean()
burden_curve.head()
```


```python
plt.figure(figsize=(8, 4))
sns.lineplot(x=burden_curve.index, y=burden_curve.values, marker="o")
plt.gca().invert_yaxis()
plt.title("Mean ESI by comorbidity count")
plt.xlabel("comorbidity count")
plt.ylabel("mean ESI, lower means more urgent")
plt.show()
```

# 9. Feature engineering

The feature engineering is intentionally clinical and inspectable. Most features are either direct intake fields, derived vitals, clinical threshold flags, missingness indicators, history aggregates, or complaint text signals.


```python
def add_clinical_features(df):
    df = df.copy()

    hx_cols_local = [col for col in df.columns if col.startswith("hx_")]

    df["pain_not_recorded"] = (df["pain_score"] == -1).astype(int)
    df["pain_score_clean"] = df["pain_score"].replace(-1, np.nan)

    df["bp_missing"] = df["systolic_bp"].isna().astype(int)
    df["rr_missing"] = df["respiratory_rate"].isna().astype(int)
    df["temp_missing"] = df["temperature_c"].isna().astype(int)
    df["spo2_missing"] = df["spo2"].isna().astype(int)
    df["vitals_missing_count"] = df[["bp_missing", "rr_missing", "temp_missing", "spo2_missing"]].sum(axis=1)

    df["shock_index_calc"] = df["heart_rate"] / df["systolic_bp"].replace(0, np.nan)
    df["pulse_pressure_calc"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map_calc"] = (2 * df["diastolic_bp"] + df["systolic_bp"]) / 3

    df["news2_high"] = (df["news2_score"] >= 7).astype(int)
    df["news2_medium"] = ((df["news2_score"] >= 5) & (df["news2_score"] < 7)).astype(int)
    df["gcs_severe"] = (df["gcs_total"] < 9).astype(int)
    df["gcs_moderate"] = ((df["gcs_total"] >= 9) & (df["gcs_total"] < 13)).astype(int)
    df["spo2_critical"] = (df["spo2"] < 90).astype(int)
    df["spo2_concerning"] = ((df["spo2"] >= 90) & (df["spo2"] < 94)).astype(int)
    df["sbp_hypotensive"] = (df["systolic_bp"] < 90).astype(int)
    df["sbp_hypertensive"] = (df["systolic_bp"] > 180).astype(int)
    df["rr_high"] = (df["respiratory_rate"] > 25).astype(int)
    df["rr_low"] = (df["respiratory_rate"] < 8).astype(int)
    df["hr_tachy"] = (df["heart_rate"] > 100).astype(int)
    df["hr_brady"] = (df["heart_rate"] < 50).astype(int)
    df["temp_fever"] = (df["temperature_c"] > 38.3).astype(int)
    df["temp_hypothermia"] = (df["temperature_c"] < 36.0).astype(int)
    df["pain_severe"] = (df["pain_score_clean"] >= 8).astype(int)
    df["shock_index_high"] = (df["shock_index_calc"] >= 1.0).astype(int)
    df["map_critical"] = (df["map_calc"] < 65).astype(int)

    mental_map = {"unresponsive": 0, "drowsy": 1, "agitated": 2, "confused": 3, "alert": 4}
    df["mental_status_encoded"] = df["mental_status_triage"].astype(str).str.lower().map(mental_map).fillna(4).astype(int)
    df["mental_status_unresponsive"] = (df["mental_status_encoded"] == 0).astype(int)
    df["mental_status_not_alert"] = (df["mental_status_encoded"] < 4).astype(int)
    df["mental_status_alert"] = (df["mental_status_encoded"] == 4).astype(int)

    df["comorbidity_count"] = df[hx_cols_local].fillna(0).sum(axis=1)
    df["high_comorbidity"] = (df["comorbidity_count"] >= 5).astype(int)

    cardio_cols = ["hx_heart_failure", "hx_atrial_fibrillation", "hx_coronary_artery_disease", "hx_hypertension", "hx_stroke_prior", "hx_peripheral_vascular_disease"]
    resp_cols = ["hx_asthma", "hx_copd"]
    metabolic_cols = ["hx_diabetes_type1", "hx_diabetes_type2", "hx_obesity", "hx_hypothyroidism", "hx_hyperthyroidism"]
    immuno_cols = ["hx_hiv", "hx_malignancy", "hx_immunosuppressed"]
    neuro_cols = ["hx_dementia", "hx_epilepsy", "hx_stroke_prior"]

    df["cardio_history_count"] = df[[c for c in cardio_cols if c in df.columns]].fillna(0).sum(axis=1)
    df["resp_history_count"] = df[[c for c in resp_cols if c in df.columns]].fillna(0).sum(axis=1)
    df["metabolic_history_count"] = df[[c for c in metabolic_cols if c in df.columns]].fillna(0).sum(axis=1)
    df["immuno_risk"] = (df[[c for c in immuno_cols if c in df.columns]].fillna(0).sum(axis=1) > 0).astype(int)
    df["neuro_history"] = (df[[c for c in neuro_cols if c in df.columns]].fillna(0).sum(axis=1) > 0).astype(int)

    text = df["chief_complaint_raw"].fillna("").str.lower()
    df["chief_complaint_len"] = text.str.len()

    for name, pattern in keyword_patterns.items():
        df[name] = text.str.contains(pattern, regex=True).astype(int)

    df["kw_severity_score"] = (
        3 * df["kw_life_threat"] +
        3 * df["kw_time_sensitive"] +
        2 * df["kw_cardio"] +
        2 * df["kw_resp"] +
        2 * df["kw_neuro"] +
        1 * df["kw_trauma"] +
        1 * df["kw_infection"] -
        2 * df["kw_low_acuity"]
    )

    return df
```


```python
train_fe = add_clinical_features(train)
```


```python
test_fe = add_clinical_features(test)
```


```python
train_fe.shape, test_fe.shape
```


```python
train_fe.head()
```

# 10. Feature groups


```python
hx_cols = [col for col in train_fe.columns if col.startswith("hx_")]
len(hx_cols)
```


```python
cat_features = [
    "site_id", "triage_nurse_id", "arrival_mode", "arrival_day", "arrival_season", "shift",
    "age_group", "sex", "language", "insurance_type", "transport_origin",
    "pain_location", "mental_status_triage", "chief_complaint_system",
]

cat_features = [col for col in cat_features if col in train_fe.columns]
cat_features
```


```python
base_numeric_features = [
    "arrival_hour", "arrival_month", "age", "num_prior_ed_visits_12m", "num_prior_admissions_12m",
    "num_active_medications", "num_comorbidities", "systolic_bp", "diastolic_bp",
    "mean_arterial_pressure", "pulse_pressure", "heart_rate", "respiratory_rate",
    "temperature_c", "spo2", "gcs_total", "pain_score_clean", "weight_kg",
    "height_cm", "bmi", "shock_index", "news2_score", "shock_index_calc",
    "pulse_pressure_calc", "map_calc", "chief_complaint_len",
]

base_numeric_features = [col for col in base_numeric_features if col in train_fe.columns]
base_numeric_features
```


```python
clinical_flag_features = [
    "pain_not_recorded", "bp_missing", "rr_missing", "temp_missing", "spo2_missing",
    "vitals_missing_count", "news2_high", "news2_medium", "gcs_severe", "gcs_moderate",
    "spo2_critical", "spo2_concerning", "sbp_hypotensive", "sbp_hypertensive",
    "rr_high", "rr_low", "hr_tachy", "hr_brady", "temp_fever", "temp_hypothermia",
    "pain_severe", "shock_index_high", "map_critical", "mental_status_encoded",
    "mental_status_unresponsive", "mental_status_not_alert", "mental_status_alert",
]

clinical_flag_features = [col for col in clinical_flag_features if col in train_fe.columns]
clinical_flag_features
```


```python
history_aggregate_features = [
    "comorbidity_count", "high_comorbidity", "cardio_history_count", "resp_history_count",
    "metabolic_history_count", "immuno_risk", "neuro_history",
]

history_aggregate_features = [col for col in history_aggregate_features if col in train_fe.columns]
history_aggregate_features
```


```python
keyword_features = list(keyword_patterns.keys()) + ["kw_severity_score"]
keyword_features
```


```python
all_base_features = (
    cat_features +
    base_numeric_features +
    clinical_flag_features +
    history_aggregate_features +
    keyword_features +
    hx_cols
)

all_base_features = list(dict.fromkeys(all_base_features))
len(all_base_features)
```


```python
all_base_features[:30]
```

# 11. Metrics

Linear weighted kappa is the primary technical metric because ESI is ordered. A one-level error and a four-level error should not be treated the same.

Safety metrics are reported separately because undertriage is the most clinically dangerous failure mode.


```python
def metric_row(name, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    undertriage = y_pred > y_true
    overtriage = y_pred < y_true
    dangerous_undertriage = ((y_true <= 2) & (y_pred >= 3))
    critical_miss = ((y_true == 1) & (y_pred >= 3))

    row = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "linear_kappa": cohen_kappa_score(y_true, y_pred, weights="linear"),
        "quadratic_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae_esi": mean_absolute_error(y_true, y_pred),
        "esi1_recall": recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0),
        "esi2_recall": recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0),
        "undertriage_rate": undertriage.mean(),
        "overtriage_rate": overtriage.mean(),
        "dangerous_undertriage_rate": dangerous_undertriage.mean(),
        "critical_miss_rate": critical_miss.mean(),
    }
    return row
```


```python
def prediction_from_probs(probs):
    return probs.argmax(axis=1) + 1
```


```python
metrics = []
```

# 12. Baseline models

## 12.1 Majority baseline


```python
y = train_fe[TARGET].copy()
```


```python
majority_class = y.value_counts().idxmax()
majority_class
```


```python
majority_pred = np.full(len(y), majority_class)
metrics.append(metric_row("majority_baseline", y, majority_pred))
pd.DataFrame(metrics)
```

## 12.2 Simple clinical rule baseline

This rule is intentionally simple. It is not the final model. It is a transparent clinical reference point.


```python
rule_pred = np.full(len(train_fe), 4)

rule_pred[(train_fe["news2_score"] >= 12) | (train_fe["gcs_total"] < 9)] = 1
rule_pred[((train_fe["news2_score"] >= 7) & (rule_pred != 1)) | (train_fe["sbp_hypotensive"] == 1) | (train_fe["spo2_critical"] == 1)] = 2
rule_pred[((train_fe["news2_score"] >= 3) & (rule_pred > 2)) | (train_fe["pain_severe"] == 1) | (train_fe["kw_cardio"] == 1) | (train_fe["kw_resp"] == 1)] = 3
rule_pred[(train_fe["news2_score"] <= 1) & (train_fe["pain_score_clean"].fillna(0) <= 1) & (train_fe["kw_low_acuity"] == 1)] = 5
```


```python
pd.Series(rule_pred).value_counts().sort_index()
```


```python
metrics.append(metric_row("simple_news2_gcs_rule", y, rule_pred))
pd.DataFrame(metrics)
```

## 12.3 Shallow decision tree bedside card


```python
tree_features = [
    "news2_score", "gcs_total", "spo2", "respiratory_rate", "systolic_bp",
    "heart_rate", "temperature_c", "pain_score_clean", "shock_index_calc",
    "mental_status_encoded", "kw_severity_score", "pain_not_recorded",
]

tree_features = [col for col in tree_features if col in train_fe.columns]
tree_features
```


```python
X_tree = train_fe[tree_features].copy()
```


```python
X_tree = X_tree.fillna(X_tree.median(numeric_only=True))
```


```python
bedside_tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=300,
    class_weight="balanced",
    random_state=SEED,
)
```


```python
bedside_tree.fit(X_tree, y)
```


```python
tree_pred = bedside_tree.predict(X_tree)
metrics.append(metric_row("bedside_tree_depth4_train", y, tree_pred))
pd.DataFrame(metrics)
```


```python
plt.figure(figsize=(24, 10))
plot_tree(
    bedside_tree,
    feature_names=tree_features,
    class_names=[f"ESI-{i}" for i in range(1, 6)],
    filled=True,
    rounded=True,
    impurity=False,
    proportion=True,
    fontsize=9,
)
plt.title("Bedside Card: shallow interpretable decision tree")
plt.show()
```

# 13. Fold-safe feature preparation

This helper is used only where it is genuinely needed: inside cross-validation. It keeps imputation, categorical encoding, and text SVD fit only on the training fold.


```python
def prepare_fold_data(train_part, val_part, test_part, feature_cols, cat_cols, include_text=True, n_text_components=40):
    X_tr = train_part[feature_cols].copy()
    X_val = val_part[feature_cols].copy()
    X_te = test_part[feature_cols].copy()

    cat_cols_used = [col for col in cat_cols if col in feature_cols]
    num_cols_used = [col for col in feature_cols if col not in cat_cols_used]

    medians = X_tr[num_cols_used].median(numeric_only=True)
    X_tr[num_cols_used] = X_tr[num_cols_used].fillna(medians)
    X_val[num_cols_used] = X_val[num_cols_used].fillna(medians)
    X_te[num_cols_used] = X_te[num_cols_used].fillna(medians)

    for col in cat_cols_used:
        X_tr[col] = X_tr[col].fillna("Unknown").astype(str)
        X_val[col] = X_val[col].fillna("Unknown").astype(str)
        X_te[col] = X_te[col].fillna("Unknown").astype(str)

    text_cols = []

    if include_text:
        tr_text = train_part["chief_complaint_raw"].fillna("").astype(str)
        val_text = val_part["chief_complaint_raw"].fillna("").astype(str)
        te_text = test_part["chief_complaint_raw"].fillna("").astype(str)

        word_tfidf = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_features=15000,
        )
        char_tfidf = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=15000,
        )

        tr_word = word_tfidf.fit_transform(tr_text)
        val_word = word_tfidf.transform(val_text)
        te_word = word_tfidf.transform(te_text)

        tr_char = char_tfidf.fit_transform(tr_text)
        val_char = char_tfidf.transform(val_text)
        te_char = char_tfidf.transform(te_text)

        tr_sparse = sparse.hstack([tr_word, tr_char]).tocsr()
        val_sparse = sparse.hstack([val_word, val_char]).tocsr()
        te_sparse = sparse.hstack([te_word, te_char]).tocsr()

        svd = TruncatedSVD(n_components=n_text_components, random_state=SEED)
        tr_svd = svd.fit_transform(tr_sparse)
        val_svd = svd.transform(val_sparse)
        te_svd = svd.transform(te_sparse)

        text_cols = [f"text_svd_{i}" for i in range(n_text_components)]

        X_tr[text_cols] = tr_svd
        X_val[text_cols] = val_svd
        X_te[text_cols] = te_svd

    final_cols = list(X_tr.columns)
    return X_tr, X_val, X_te, final_cols, cat_cols_used, text_cols
```

# 14. CatBoost cross-validation

CatBoost is the main production-style tabular model.


```python
cat_oof_probs = np.zeros((len(train_fe), 5))
cat_test_probs = np.zeros((len(test_fe), 5))
cat_fold_metrics = []
cat_models = []
```


```python
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
```


```python
cat_params = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 2500,
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 5,
    "random_seed": SEED,
    "task_type": "GPU",
    "devices": "0",
    "verbose": False,
    "allow_writing_files": False,
    "early_stopping_rounds": 100,
}

cat_params
```


```python
for fold, (tr_idx, val_idx) in enumerate(skf.split(train_fe, y), start=1):
    print("=" * 80)
    print(f"CatBoost fold {fold}")

    tr_part = train_fe.iloc[tr_idx].reset_index(drop=True)
    val_part = train_fe.iloc[val_idx].reset_index(drop=True)
    te_part = test_fe.copy().reset_index(drop=True)

    X_tr, X_val, X_te, final_cols, cat_cols_used, text_cols = prepare_fold_data(
        tr_part,
        val_part,
        te_part,
        all_base_features,
        cat_features,
        include_text=True,
        n_text_components=40,
    )

    y_tr = tr_part[TARGET]
    y_val = val_part[TARGET]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols_used)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols_used)

    model = CatBoostClassifier(**cat_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)

    val_probs = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_te)

    cat_oof_probs[val_idx] = val_probs
    cat_test_probs += test_probs / N_FOLDS
    cat_models.append(model)

    val_pred = prediction_from_probs(val_probs)
    row = metric_row(f"catboost_fold_{fold}", y_val, val_pred)
    cat_fold_metrics.append(row)
    print(row)
```


```python
cat_oof_pred = prediction_from_probs(cat_oof_probs)
```


```python
metrics.append(metric_row("catboost_oof", y, cat_oof_pred))
pd.DataFrame(metrics)
```


```python
pd.DataFrame(cat_fold_metrics)
```

# 15. LightGBM cross-validation

LightGBM is used as an independent benchmark and ensemble component. 


```python
lgb_oof_probs = np.zeros((len(train_fe), 5))
lgb_test_probs = np.zeros((len(test_fe), 5))
lgb_fold_metrics = []
lgb_models = []
```


```python
lgb_params = {
    "objective": "multiclass",
    "num_class": 5,
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_samples": 30,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
    "device_type": "gpu",
    "max_bin": 63,
}

lgb_params
```


```python
classes = np.array([1, 2, 3, 4, 5])
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))
class_weight_dict
```


```python
for fold, (tr_idx, val_idx) in enumerate(skf.split(train_fe, y), start=1):
    print("=" * 80)
    print(f"LightGBM fold {fold}")

    tr_part = train_fe.iloc[tr_idx].reset_index(drop=True)
    val_part = train_fe.iloc[val_idx].reset_index(drop=True)
    te_part = test_fe.copy().reset_index(drop=True)

    X_tr, X_val, X_te, final_cols, cat_cols_used, text_cols = prepare_fold_data(
        tr_part,
        val_part,
        te_part,
        all_base_features,
        cat_features,
        include_text=True,
        n_text_components=40,
    )

    for col in cat_cols_used:
        freq = X_tr[col].value_counts(normalize=True)
        X_tr[col] = X_tr[col].map(freq).fillna(0)
        X_val[col] = X_val[col].map(freq).fillna(0)
        X_te[col] = X_te[col].map(freq).fillna(0)

    y_tr = tr_part[TARGET] - 1
    y_val = val_part[TARGET] - 1
    sw_tr = tr_part[TARGET].map(class_weight_dict).values

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    val_probs = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_te)

    lgb_oof_probs[val_idx] = val_probs
    lgb_test_probs += test_probs / N_FOLDS
    lgb_models.append(model)

    val_pred = prediction_from_probs(val_probs)
    row = metric_row(f"lightgbm_fold_{fold}", val_part[TARGET], val_pred)
    lgb_fold_metrics.append(row)
    print(row)
```


```python
lgb_oof_pred = prediction_from_probs(lgb_oof_probs)
```


```python
metrics.append(metric_row("lightgbm_oof", y, lgb_oof_pred))
pd.DataFrame(metrics)
```


```python
pd.DataFrame(lgb_fold_metrics)
```

# 16. Text-only model

This model isolates the chief complaint signal. It helps answer whether raw complaint text itself carries triage information.


```python
text_oof_probs = np.zeros((len(train_fe), 5))
text_test_probs = np.zeros((len(test_fe), 5))
text_fold_metrics = []
```


```python
for fold, (tr_idx, val_idx) in enumerate(skf.split(train_fe, y), start=1):
    print("=" * 80)
    print(f"Text-only fold {fold}")

    tr_text = train_fe.iloc[tr_idx]["chief_complaint_raw"].fillna("").astype(str)
    val_text = train_fe.iloc[val_idx]["chief_complaint_raw"].fillna("").astype(str)
    te_text = test_fe["chief_complaint_raw"].fillna("").astype(str)

    word_tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=20000)
    char_tfidf = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=20000)

    tr_word = word_tfidf.fit_transform(tr_text)
    val_word = word_tfidf.transform(val_text)
    te_word = word_tfidf.transform(te_text)

    tr_char = char_tfidf.fit_transform(tr_text)
    val_char = char_tfidf.transform(val_text)
    te_char = char_tfidf.transform(te_text)

    X_tr_text = sparse.hstack([tr_word, tr_char]).tocsr()
    X_val_text = sparse.hstack([val_word, val_char]).tocsr()
    X_te_text = sparse.hstack([te_word, te_char]).tocsr()

    model = LogisticRegression(
        C=3.0,
        max_iter=1000,
        class_weight="balanced",
        multi_class="auto",
        n_jobs=-1,
        random_state=SEED,
    )

    model.fit(X_tr_text, train_fe.iloc[tr_idx][TARGET])

    val_probs = model.predict_proba(X_val_text)
    te_probs = model.predict_proba(X_te_text)

    text_oof_probs[val_idx] = val_probs
    text_test_probs += te_probs / N_FOLDS

    val_pred = prediction_from_probs(val_probs)
    row = metric_row(f"text_only_fold_{fold}", train_fe.iloc[val_idx][TARGET], val_pred)
    text_fold_metrics.append(row)
    print(row)
```


```python
text_oof_pred = prediction_from_probs(text_oof_probs)
metrics.append(metric_row("text_only_oof", y, text_oof_pred))
pd.DataFrame(metrics)
```


```python
pd.DataFrame(text_fold_metrics)
```

# 16.5 Synthetic shortcut audit

The text-only model is extremely strong, so we explicitly check whether the dataset contains repeated complaint templates. This is not treated as a reason to drop chief complaint text. In triage, the complaint is clinically essential. But it must be disclosed because repeated synthetic templates can inflate ordinary StratifiedKFold validation.


```python
complaint_train = train_fe["chief_complaint_raw"].fillna("missing_complaint").astype(str)
complaint_test = test_fe["chief_complaint_raw"].fillna("missing_complaint").astype(str)

complaint_audit = pd.DataFrame({
    "metric": [
        "train_rows",
        "unique_train_complaints",
        "train_duplicate_template_share",
        "test_rows",
        "unique_test_complaints",
        "test_complaints_seen_in_train_share",
    ],
    "value": [
        len(complaint_train),
        complaint_train.nunique(),
        1 - complaint_train.nunique() / len(complaint_train),
        len(complaint_test),
        complaint_test.nunique(),
        complaint_test.isin(set(complaint_train)).mean(),
    ]
})

complaint_audit
```


```python
complaint_counts = complaint_train.value_counts().reset_index()
complaint_counts.columns = ["chief_complaint_raw", "count"]
complaint_counts.head(15)
```


```python
complaint_target_table = pd.crosstab(complaint_train, train_fe[TARGET])
complaint_target_table["n"] = complaint_target_table.sum(axis=1)
complaint_target_table["n_esi_classes"] = (complaint_target_table[[1, 2, 3, 4, 5]] > 0).sum(axis=1)

ambiguous_templates = complaint_target_table[complaint_target_table["n_esi_classes"] > 1].copy()
ambiguous_templates.sort_values("n", ascending=False).head(15)
```


```python
template_oof_pred = np.zeros(len(train_fe), dtype=int)
template_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for tr_idx, val_idx in template_skf.split(train_fe, y):
    tr_part = train_fe.iloc[tr_idx]
    val_part = train_fe.iloc[val_idx]

    template_map = (
        tr_part
        .groupby("chief_complaint_raw")[TARGET]
        .agg(lambda s: s.value_counts().index[0])
    )

    fallback_class = int(tr_part[TARGET].mode()[0])
    val_pred = val_part["chief_complaint_raw"].map(template_map).fillna(fallback_class).astype(int)
    template_oof_pred[val_idx] = val_pred.values

metrics.append(metric_row("complaint_template_mode_oof", y, template_oof_pred))
metric_row("complaint_template_mode_oof", y, template_oof_pred)
```

# 17. Ensemble

The final prediction uses probability averaging. The weights can be tuned on OOF, but we keep the search small and transparent.


```python
weight_grid = []

for w_cat in np.arange(0.3, 0.8, 0.1):
    for w_lgb in np.arange(0.1, 0.7, 0.1):
        w_text = 1 - w_cat - w_lgb
        if w_text < 0:
            continue
        if w_text > 0.4:
            continue
        weight_grid.append((round(w_cat, 2), round(w_lgb, 2), round(w_text, 2)))

weight_grid[:10], len(weight_grid)
```


```python
ensemble_rows = []

for w_cat, w_lgb, w_text in weight_grid:
    probs = w_cat * cat_oof_probs + w_lgb * lgb_oof_probs + w_text * text_oof_probs
    pred = prediction_from_probs(probs)
    row = metric_row(f"ens_cat{w_cat}_lgb{w_lgb}_txt{w_text}", y, pred)
    row["w_cat"] = w_cat
    row["w_lgb"] = w_lgb
    row["w_text"] = w_text
    ensemble_rows.append(row)

ensemble_results = pd.DataFrame(ensemble_rows)
ensemble_results.sort_values(["linear_kappa", "dangerous_undertriage_rate"], ascending=[False, True]).head(10)
```


```python
best_ensemble = ensemble_results.sort_values(["linear_kappa", "dangerous_undertriage_rate"], ascending=[False, True]).iloc[0]
best_ensemble
```


```python
w_cat = best_ensemble["w_cat"]
w_lgb = best_ensemble["w_lgb"]
w_text = best_ensemble["w_text"]

ensemble_oof_probs = w_cat * cat_oof_probs + w_lgb * lgb_oof_probs + w_text * text_oof_probs
ensemble_test_probs = w_cat * cat_test_probs + w_lgb * lgb_test_probs + w_text * text_test_probs

ensemble_oof_pred = prediction_from_probs(ensemble_oof_probs)
ensemble_test_pred = prediction_from_probs(ensemble_test_probs)
```


```python
metrics.append(metric_row("weighted_ensemble_oof", y, ensemble_oof_pred))
pd.DataFrame(metrics).sort_values("linear_kappa", ascending=False)
```

# 18. Safety policy layer

The base model outputs an ESI class by argmax. For clinical workflow, that is not enough. The safety layer is now defined as a **low-acuity override**:

- predicted ESI-1/2 means high-acuity routing, not review workload;
- review is counted only when the model predicts ESI-3/4/5 but probability still contains high-acuity risk or uncertainty;
- this directly targets the dangerous case: a critical patient being placed into a lower-acuity queue.


```python
safety_df = pd.DataFrame({
    "true_esi": y.values,
    "pred_esi": ensemble_oof_pred,
    "p_esi1": ensemble_oof_probs[:, 0],
    "p_esi2": ensemble_oof_probs[:, 1],
    "p_esi3": ensemble_oof_probs[:, 2],
    "p_esi4": ensemble_oof_probs[:, 3],
    "p_esi5": ensemble_oof_probs[:, 4],
})

safety_df["p_high_acuity"] = safety_df["p_esi1"] + safety_df["p_esi2"]
safety_df["max_proba"] = ensemble_oof_probs.max(axis=1)
safety_df["predicted_low_acuity"] = safety_df["pred_esi"] >= 3
safety_df["dangerous_undertriage"] = (safety_df["true_esi"] <= 2) & (safety_df["pred_esi"] >= 3)
safety_df.head()
```


```python
safety_df[["predicted_low_acuity", "dangerous_undertriage"]].mean()
```


```python
policy_rows = []

for high_thr in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    for uncertain_thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        low_pred = safety_df["predicted_low_acuity"]
        review = low_pred & ((safety_df["p_high_acuity"] >= high_thr) | (safety_df["max_proba"] < uncertain_thr))
        dangerous = safety_df["dangerous_undertriage"]
        dangerous_not_reviewed = dangerous & (~review)

        policy_rows.append({
            "p_high_acuity_threshold": high_thr,
            "uncertainty_threshold": uncertain_thr,
            "overall_review_rate": review.mean(),
            "low_acuity_review_rate": review[low_pred].mean(),
            "dangerous_undertriage_rate": dangerous.mean(),
            "dangerous_undertriage_not_reviewed_rate": dangerous_not_reviewed.mean(),
            "dangerous_undertriage_caught_share": 1 - dangerous_not_reviewed.sum() / max(dangerous.sum(), 1),
        })

policy_results = pd.DataFrame(policy_rows)
policy_results.sort_values([
    "dangerous_undertriage_not_reviewed_rate", 
    "low_acuity_review_rate", 
    "overall_review_rate"
]).head(12)
```


```python
selected_policy = policy_results.sort_values([
    "dangerous_undertriage_not_reviewed_rate", 
    "low_acuity_review_rate", 
    "overall_review_rate"
]).iloc[0]

selected_policy
```


```python
HIGH_ACUITY_THRESHOLD = float(selected_policy["p_high_acuity_threshold"])
UNCERTAINTY_THRESHOLD = float(selected_policy["uncertainty_threshold"])

HIGH_ACUITY_THRESHOLD, UNCERTAINTY_THRESHOLD
```


```python
low_pred = safety_df["predicted_low_acuity"]

safety_df["review_flag"] = low_pred & (
    (safety_df["p_high_acuity"] >= HIGH_ACUITY_THRESHOLD) |
    (safety_df["max_proba"] < UNCERTAINTY_THRESHOLD)
)

safety_df["high_acuity_routing"] = safety_df["pred_esi"] <= 2
safety_df["dangerous_undertriage_not_reviewed"] = safety_df["dangerous_undertriage"] & (~safety_df["review_flag"])

safety_summary = pd.Series({
    "high_acuity_routing_rate": safety_df["high_acuity_routing"].mean(),
    "overall_low_acuity_review_rate": safety_df["review_flag"].mean(),
    "review_rate_among_predicted_low_acuity": safety_df.loc[low_pred, "review_flag"].mean(),
    "dangerous_undertriage_rate": safety_df["dangerous_undertriage"].mean(),
    "dangerous_undertriage_not_reviewed_rate": safety_df["dangerous_undertriage_not_reviewed"].mean(),
})

safety_summary
```

This is the metric that matters operationally: how much extra review burden is added among patients who would otherwise be treated as ESI-3/4/5, and whether that catches dangerous undertriage.

# 19. Full evaluation


```python
metrics_df = pd.DataFrame(metrics)
metrics_df.sort_values("linear_kappa", ascending=False)
```


```python
print(classification_report(y, ensemble_oof_pred, digits=4))
```


```python
cm = confusion_matrix(y, ensemble_oof_pred, labels=[1, 2, 3, 4, 5])
cm
```


```python
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
plt.title("Weighted ensemble confusion matrix, raw counts")
plt.xlabel("predicted ESI")
plt.ylabel("true ESI")
plt.show()
```


```python
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(7, 5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
plt.title("Weighted ensemble confusion matrix, row-normalized")
plt.xlabel("predicted ESI")
plt.ylabel("true ESI")
plt.show()
```


```python
error_distance = np.abs(y.values - ensemble_oof_pred)
pd.Series(error_distance).value_counts().sort_index()
```


```python
plt.figure(figsize=(7, 4))
sns.countplot(x=error_distance)
plt.title("Absolute ESI error distance")
plt.xlabel("absolute error in ESI levels")
plt.ylabel("count")
plt.show()
```


```python
error_df = pd.DataFrame({
    "true_esi": y.values,
    "pred_esi": ensemble_oof_pred,
})

error_df["undertriage"] = error_df["pred_esi"] > error_df["true_esi"]
error_df["overtriage"] = error_df["pred_esi"] < error_df["true_esi"]
error_df.head()
```


```python
error_by_esi = error_df.groupby("true_esi")[["undertriage", "overtriage"]].mean() * 100
error_by_esi
```


```python
error_by_esi.plot(kind="bar", figsize=(8, 4))
plt.title("Undertriage and overtriage by true ESI")
plt.xlabel("true ESI")
plt.ylabel("percent")
plt.xticks(rotation=0)
plt.show()
```

# 19.1 Error case analysis

The ensemble makes very few OOF errors, so we inspect every miss instead of hiding behind aggregate metrics. This is especially important for triage because an ESI-1/2 undertriage error matters much more than a harmless one-level low-acuity disagreement.


```python
error_case_df = train_fe[[
    ID_COL, TARGET, "age", "sex", "arrival_mode", "chief_complaint_raw",
    "news2_score", "gcs_total", "spo2", "systolic_bp", "heart_rate", 
    "respiratory_rate", "temperature_c", "pain_score"
]].copy()

error_case_df["pred_esi"] = ensemble_oof_pred
error_case_df["error_distance"] = (error_case_df[TARGET] - error_case_df["pred_esi"]).abs()
error_case_df["undertriage"] = error_case_df["pred_esi"] > error_case_df[TARGET]
error_case_df["overtriage"] = error_case_df["pred_esi"] < error_case_df[TARGET]
error_case_df["dangerous_undertriage"] = (error_case_df[TARGET] <= 2) & (error_case_df["pred_esi"] >= 3)

for i in range(5):
    error_case_df[f"p_esi{i+1}"] = ensemble_oof_probs[:, i]

error_case_df["max_proba"] = ensemble_oof_probs.max(axis=1)
error_case_df["p_high_acuity"] = ensemble_oof_probs[:, 0] + ensemble_oof_probs[:, 1]
error_case_df["review_flag"] = safety_df["review_flag"].values

error_case_df.head()
```


```python
error_case_df["error_distance"].value_counts().sort_index()
```


```python
error_case_df[error_case_df["error_distance"] > 0][
    [ID_COL, TARGET, "pred_esi", "error_distance", "undertriage", "dangerous_undertriage", 
     "review_flag", "p_high_acuity", "max_proba", "news2_score", "gcs_total", 
     "spo2", "pain_score", "chief_complaint_raw"]
].sort_values(["dangerous_undertriage", "error_distance", TARGET], ascending=[False, False, True])
```


```python
dangerous_error_cases = error_case_df[error_case_df["dangerous_undertriage"]].copy()
dangerous_error_cases[
    [ID_COL, TARGET, "pred_esi", "review_flag", "p_esi1", "p_esi2", "p_esi3", 
     "p_high_acuity", "max_proba", "chief_complaint_raw"]
]
```

# 20. Ablation study

This section checks where the model signal comes from. It is intentionally based on a lighter LightGBM setup to keep runtime sane.


```python
ablation_specs = []

vital_core = [
    "age", "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate", "temperature_c",
    "spo2", "gcs_total", "pain_score_clean", "shock_index_calc", "map_calc", "pulse_pressure_calc",
]
vital_core = [col for col in vital_core if col in train_fe.columns]

news2_features = ["news2_score", "news2_high", "news2_medium"]
news2_features = [col for col in news2_features if col in train_fe.columns]

site_nurse_features = ["site_id", "triage_nurse_id"]
site_nurse_features = [col for col in site_nurse_features if col in train_fe.columns]

features_without_site_nurse = [col for col in all_base_features if col not in site_nurse_features]

ablation_specs.append(("vitals_only_no_news2", vital_core, False))
ablation_specs.append(("vitals_plus_news2", vital_core + news2_features, False))
ablation_specs.append(("vitals_news2_history", vital_core + news2_features + history_aggregate_features + hx_cols, False))
ablation_specs.append(("clinical_core_no_raw_text", all_base_features, False))
ablation_specs.append(("full_without_site_nurse", features_without_site_nurse, True))
ablation_specs.append(("full_model", all_base_features, True))

[(name, len(cols), include_text) for name, cols, include_text in ablation_specs]
```


```python
ablation_params = {
    "objective": "multiclass",
    "num_class": 5,
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_samples": 30,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}
```


```python
ablation_rows = []
ablation_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

for exp_name, exp_features, include_text in ablation_specs:
    print("=" * 80)
    print(exp_name)

    exp_oof = np.zeros((len(train_fe), 5))
    exp_cat_cols = [col for col in cat_features if col in exp_features]

    for fold, (tr_idx, val_idx) in enumerate(ablation_skf.split(train_fe, y), start=1):
        tr_part = train_fe.iloc[tr_idx].reset_index(drop=True)
        val_part = train_fe.iloc[val_idx].reset_index(drop=True)
        te_part = test_fe.copy().reset_index(drop=True)

        X_tr, X_val, X_te, final_cols, cat_cols_used, text_cols = prepare_fold_data(
            tr_part,
            val_part,
            te_part,
            exp_features,
            exp_cat_cols,
            include_text=include_text,
            n_text_components=25,
        )

        for col in cat_cols_used:
            freq = X_tr[col].value_counts(normalize=True)
            X_tr[col] = X_tr[col].map(freq).fillna(0)
            X_val[col] = X_val[col].map(freq).fillna(0)

        model = lgb.LGBMClassifier(**ablation_params)
        model.fit(
            X_tr,
            tr_part[TARGET] - 1,
            eval_set=[(X_val, val_part[TARGET] - 1)],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )

        exp_oof[val_idx] = model.predict_proba(X_val)

    exp_pred = prediction_from_probs(exp_oof)
    row = metric_row(exp_name, y, exp_pred)
    row["n_features_base"] = len(exp_features)
    row["text_svd"] = include_text
    ablation_rows.append(row)

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.sort_values("linear_kappa", ascending=False)
```

# 20.1 Reality-check validation: GroupKFold by chief complaint



```python
group_specs = [
    ("group_full_with_text", all_base_features, True),
    ("group_clinical_core_no_raw_text", all_base_features, False),
]

groups = train_fe["chief_complaint_raw"].fillna("missing_complaint").astype(str)
group_specs
```


```python
group_params = ablation_params.copy()
group_params.update({
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "device_type": "gpu",
    "max_bin": 63,
})

group_params
```


```python
group_rows = []
gkf = GroupKFold(n_splits=3)

for exp_name, exp_features, include_text in group_specs:
    print("=" * 80)
    print(exp_name)

    exp_oof = np.zeros((len(train_fe), 5))
    exp_cat_cols = [col for col in cat_features if col in exp_features]

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_fe, y, groups), start=1):
        print(f"fold {fold}")

        tr_part = train_fe.iloc[tr_idx].reset_index(drop=True)
        val_part = train_fe.iloc[val_idx].reset_index(drop=True)
        te_part = val_part.copy().reset_index(drop=True)

        X_tr, X_val, X_te, final_cols, cat_cols_used, text_cols = prepare_fold_data(
            tr_part,
            val_part,
            te_part,
            exp_features,
            exp_cat_cols,
            include_text=include_text,
            n_text_components=20,
        )

        for col in cat_cols_used:
            freq = X_tr[col].value_counts(normalize=True)
            X_tr[col] = X_tr[col].map(freq).fillna(0)
            X_val[col] = X_val[col].map(freq).fillna(0)

        model = lgb.LGBMClassifier(**group_params)
        model.fit(
            X_tr,
            tr_part[TARGET] - 1,
            eval_set=[(X_val, val_part[TARGET] - 1)],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )

        exp_oof[val_idx] = model.predict_proba(X_val)

    exp_pred = prediction_from_probs(exp_oof)
    row = metric_row(exp_name, y, exp_pred)
    row["validation"] = "GroupKFold_by_chief_complaint_raw"
    row["raw_text_svd"] = include_text
    group_rows.append(row)

group_stress_df = pd.DataFrame(group_rows)
group_stress_df
```


```python
two_track_summary = pd.concat([
    pd.DataFrame(metrics).query("model in ['weighted_ensemble_oof', 'text_only_oof', 'complaint_template_mode_oof']"),
    ablation_df.query("model in ['clinical_core_no_raw_text', 'full_model']"),
    group_stress_df,
], ignore_index=True)

two_track_summary[[
    "model", "accuracy", "linear_kappa", "macro_f1", "mae_esi", 
    "esi1_recall", "esi2_recall", "dangerous_undertriage_rate"
]].sort_values("linear_kappa", ascending=False)
```

# 21. Explainability

## 21.1 CatBoost global feature importance


```python
cat_feature_importances = []

for i, model in enumerate(cat_models):
    importance = model.get_feature_importance()
    names = model.feature_names_
    fold_imp = pd.DataFrame({"feature": names, "importance": importance, "fold": i + 1})
    cat_feature_importances.append(fold_imp)

cat_feature_importances = pd.concat(cat_feature_importances, ignore_index=True)
cat_feature_importances.head()
```


```python
cat_importance_mean = cat_feature_importances.groupby("feature")["importance"].mean().sort_values(ascending=False)
cat_importance_mean.head(30)
```


```python
plt.figure(figsize=(9, 8))
cat_importance_mean.head(25).sort_values().plot(kind="barh")
plt.title("CatBoost mean feature importance across folds")
plt.xlabel("importance")
plt.ylabel("feature")
plt.show()
```

## 21.2 LightGBM SHAP on one fold

SHAP is computed on a sample to keep runtime manageable.


```python
shap_model = lgb_models[0]
```


```python
tr_idx, val_idx = list(skf.split(train_fe, y))[0]
tr_part = train_fe.iloc[tr_idx].reset_index(drop=True)
val_part = train_fe.iloc[val_idx].reset_index(drop=True)
te_part = test_fe.copy().reset_index(drop=True)
```


```python
X_tr_shap, X_val_shap, X_te_shap, shap_cols, shap_cat_cols, shap_text_cols = prepare_fold_data(
    tr_part,
    val_part,
    te_part,
    all_base_features,
    cat_features,
    include_text=True,
    n_text_components=40,
)
```


```python
for col in shap_cat_cols:
    freq = X_tr_shap[col].value_counts(normalize=True)
    X_val_shap[col] = X_val_shap[col].map(freq).fillna(0)
```


```python
shap_sample = X_val_shap.sample(min(1500, len(X_val_shap)), random_state=SEED)
shap_sample.shape
```


```python
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(shap_sample)
```


```python
if isinstance(shap_values, list):
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
else:
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))

shap_importance = pd.Series(mean_abs, index=shap_sample.columns).sort_values(ascending=False)
shap_importance.head(30)
```


```python
plt.figure(figsize=(9, 8))
shap_importance.head(25).sort_values().plot(kind="barh")
plt.title("LightGBM SHAP global importance, average over classes")
plt.xlabel("mean absolute SHAP")
plt.ylabel("feature")
plt.show()
```

## 21.3 Local patient explanations


```python
explain_df = train_fe[[ID_COL, TARGET, "age", "sex", "arrival_mode", "chief_complaint_raw", "news2_score", "gcs_total", "spo2", "systolic_bp", "heart_rate", "respiratory_rate", "temperature_c", "pain_score"]].copy()
explain_df["pred_esi"] = ensemble_oof_pred
explain_df["p_esi1"] = ensemble_oof_probs[:, 0]
explain_df["p_esi2"] = ensemble_oof_probs[:, 1]
explain_df["p_esi3"] = ensemble_oof_probs[:, 2]
explain_df["p_esi4"] = ensemble_oof_probs[:, 3]
explain_df["p_esi5"] = ensemble_oof_probs[:, 4]
explain_df["dangerous_undertriage"] = ((explain_df[TARGET] <= 2) & (explain_df["pred_esi"] >= 3))
explain_df.head()
```


```python
examples = []

for esi in [1, 2, 3, 4, 5]:
    correct = explain_df[(explain_df[TARGET] == esi) & (explain_df["pred_esi"] == esi)]
    if len(correct) > 0:
        examples.append(correct.index[0])

examples
```


```python
explain_df.loc[examples]
```

# 22. Fairness and subgroup audit



```python
fairness_df = train_fe[[ID_COL, TARGET, "sex", "age_group", "language", "insurance_type", "arrival_mode", "site_id"]].copy()
fairness_df["pred_esi"] = ensemble_oof_pred
fairness_df["undertriage"] = fairness_df["pred_esi"] > fairness_df[TARGET]
fairness_df["overtriage"] = fairness_df["pred_esi"] < fairness_df[TARGET]
fairness_df["dangerous_undertriage"] = ((fairness_df[TARGET] <= 2) & (fairness_df["pred_esi"] >= 3))
fairness_df["error"] = fairness_df["pred_esi"] - fairness_df[TARGET]
fairness_df.head()
```


```python
def subgroup_metrics(df, group_col):
    rows = []
    for value, part in df.groupby(group_col):
        rows.append({
            "group": value,
            "n": len(part),
            "linear_kappa": cohen_kappa_score(part[TARGET], part["pred_esi"], weights="linear"),
            "macro_f1": f1_score(part[TARGET], part["pred_esi"], average="macro"),
            "undertriage_rate": part["undertriage"].mean(),
            "dangerous_undertriage_rate": part["dangerous_undertriage"].mean(),
            "mean_error": part["error"].mean(),
        })
    return pd.DataFrame(rows).sort_values("linear_kappa")
```


```python
sex_audit = subgroup_metrics(fairness_df, "sex")
sex_audit
```


```python
age_audit = subgroup_metrics(fairness_df, "age_group")
age_audit
```


```python
language_audit = subgroup_metrics(fairness_df, "language")
language_audit
```


```python
insurance_audit = subgroup_metrics(fairness_df, "insurance_type")
insurance_audit
```


```python
arrival_audit = subgroup_metrics(fairness_df, "arrival_mode")
arrival_audit
```


```python
plt.figure(figsize=(8, 4))
sns.barplot(data=age_audit, x="group", y="dangerous_undertriage_rate")
plt.title("Dangerous undertriage rate by age group")
plt.xlabel("age group")
plt.ylabel("dangerous undertriage rate")
plt.xticks(rotation=30, ha="right")
plt.show()
```


```python
pivot_undertriage = fairness_df.pivot_table(
    values="undertriage",
    index="sex",
    columns="insurance_type",
    aggfunc="mean",
) * 100

pivot_undertriage
```


```python
plt.figure(figsize=(10, 4))
sns.heatmap(pivot_undertriage, annot=True, fmt=".1f", cmap="Oranges")
plt.title("Undertriage rate by sex and insurance type, percent")
plt.xlabel("insurance type")
plt.ylabel("sex")
plt.show()
```

# 23. Train final models on full data

The final models are trained on the full training set using the selected feature strategy. These models are used for final test prediction and the interactive simulator.


```python
full_train = train_fe.copy().reset_index(drop=True)
full_test = test_fe.copy().reset_index(drop=True)
```


```python
X_full = full_train[all_base_features].copy()
X_final_test = full_test[all_base_features].copy()
```


```python
cat_cols_final = [col for col in cat_features if col in all_base_features]
num_cols_final = [col for col in all_base_features if col not in cat_cols_final]
```


```python
final_medians = X_full[num_cols_final].median(numeric_only=True)
X_full[num_cols_final] = X_full[num_cols_final].fillna(final_medians)
X_final_test[num_cols_final] = X_final_test[num_cols_final].fillna(final_medians)
```


```python
for col in cat_cols_final:
    X_full[col] = X_full[col].fillna("Unknown").astype(str)
    X_final_test[col] = X_final_test[col].fillna("Unknown").astype(str)
```


```python
full_text = full_train["chief_complaint_raw"].fillna("").astype(str)
test_text = full_test["chief_complaint_raw"].fillna("").astype(str)
```


```python
final_word_tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=15000)
final_char_tfidf = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=15000)
```


```python
full_word = final_word_tfidf.fit_transform(full_text)
test_word = final_word_tfidf.transform(test_text)
```


```python
full_char = final_char_tfidf.fit_transform(full_text)
test_char = final_char_tfidf.transform(test_text)
```


```python
full_sparse = sparse.hstack([full_word, full_char]).tocsr()
test_sparse = sparse.hstack([test_word, test_char]).tocsr()
```


```python
final_svd = TruncatedSVD(n_components=40, random_state=SEED)
full_svd = final_svd.fit_transform(full_sparse)
test_svd = final_svd.transform(test_sparse)
```


```python
final_text_cols = [f"text_svd_{i}" for i in range(40)]
```


```python
X_full[final_text_cols] = full_svd
X_final_test[final_text_cols] = test_svd
```


```python
X_full.shape, X_final_test.shape
```


```python
final_cat_model = CatBoostClassifier(**cat_params)
```


```python
final_cat_model.fit(Pool(X_full, full_train[TARGET], cat_features=cat_cols_final), verbose=False)
```


```python
final_cat_test_probs = final_cat_model.predict_proba(X_final_test)
```


```python
X_full_lgb = X_full.copy()
X_final_test_lgb = X_final_test.copy()
```


```python
final_freq_maps = {}

for col in cat_cols_final:
    freq = X_full_lgb[col].value_counts(normalize=True)
    final_freq_maps[col] = freq
    X_full_lgb[col] = X_full_lgb[col].map(freq).fillna(0)
    X_final_test_lgb[col] = X_final_test_lgb[col].map(freq).fillna(0)
```


```python
final_lgb_model = lgb.LGBMClassifier(**lgb_params)
```


```python
final_lgb_model.fit(X_full_lgb, full_train[TARGET] - 1, sample_weight=full_train[TARGET].map(class_weight_dict).values)
```


```python
final_lgb_test_probs = final_lgb_model.predict_proba(X_final_test_lgb)
```


```python
final_text_model = LogisticRegression(
    C=3.0,
    max_iter=1000,
    class_weight="balanced",
    multi_class="auto",
    n_jobs=-1,
    random_state=SEED,
)
```


```python
final_text_model.fit(full_sparse, full_train[TARGET])
```


```python
final_text_test_probs = final_text_model.predict_proba(test_sparse)
```


```python
final_test_probs = w_cat * final_cat_test_probs + w_lgb * final_lgb_test_probs + w_text * final_text_test_probs
final_test_pred = prediction_from_probs(final_test_probs)
```


```python
pd.Series(final_test_pred).value_counts().sort_index()
```

# 24. Submission


```python
submission = sample_submission.copy()
submission[TARGET] = final_test_pred.astype(int)
submission.head()
```


```python
assert list(submission.columns) == list(sample_submission.columns)
assert len(submission) == len(sample_submission)
assert submission[TARGET].between(1, 5).all()
assert submission[TARGET].isna().sum() == 0
```


```python
submission[TARGET].value_counts().sort_index()
```


```python
submission_path = "submission_safetriage.csv"
submission.to_csv(submission_path, index=False)
submission_path
```

# 25. Clinical simulator

This is a proof-of-concept interface. It does not provide medical advice and does not replace clinical staff.


```python
ESI_LABELS = {
    1: "IMMEDIATE",
    2: "EMERGENT",
    3: "URGENT",
    4: "LESS URGENT",
    5: "NON-URGENT",
}
```


```python
def prepare_single_patient(row_dict):
    row = full_train.drop(columns=[TARGET]).iloc[[0]].copy()

    for col, value in row_dict.items():
        row[col] = value

    row = add_clinical_features(row)

    X_one = row[all_base_features].copy()
    X_one[num_cols_final] = X_one[num_cols_final].fillna(final_medians)

    for col in cat_cols_final:
        X_one[col] = X_one[col].fillna("Unknown").astype(str)

    one_text = row["chief_complaint_raw"].fillna("").astype(str)
    one_word = final_word_tfidf.transform(one_text)
    one_char = final_char_tfidf.transform(one_text)
    one_sparse = sparse.hstack([one_word, one_char]).tocsr()
    one_svd = final_svd.transform(one_sparse)
    X_one[final_text_cols] = one_svd

    return X_one, row
```


```python
def assess_single_patient(row_dict):
    X_one, row = prepare_single_patient(row_dict)
    probs = final_cat_model.predict_proba(X_one)[0]
    pred = int(np.argmax(probs) + 1)

    p_high = probs[0] + probs[1]
    max_p = probs.max()

    if pred <= 2:
        flag = "High-acuity routing"
    elif p_high >= HIGH_ACUITY_THRESHOLD:
        flag = "Low-acuity safety override: high-acuity probability is non-trivial"
    elif max_p < UNCERTAINTY_THRESHOLD:
        flag = "Low-acuity safety override: model uncertainty"
    else:
        flag = "Standard model suggestion"

    return pred, probs, flag, row
```


```python
w_age = widgets.IntText(value=55, description="Age")
w_sex = widgets.Dropdown(options=sorted(full_train["sex"].dropna().astype(str).unique()), value=str(full_train["sex"].mode()[0]), description="Sex")
w_arrival = widgets.Dropdown(options=sorted(full_train["arrival_mode"].dropna().astype(str).unique()), value=str(full_train["arrival_mode"].mode()[0]), description="Arrival")
w_mental = widgets.Dropdown(options=sorted(full_train["mental_status_triage"].dropna().astype(str).unique()), value="alert", description="Mental")
w_gcs = widgets.IntText(value=15, description="GCS")
w_news2 = widgets.IntText(value=1, description="NEWS2")
w_sbp = widgets.IntText(value=120, description="SBP")
w_dbp = widgets.IntText(value=75, description="DBP")
w_hr = widgets.IntText(value=80, description="HR")
w_rr = widgets.FloatText(value=16.0, description="RR")
w_temp = widgets.FloatText(value=37.0, description="Temp C")
w_spo2 = widgets.FloatText(value=98.0, description="SpO2")
w_pain = widgets.IntText(value=0, description="Pain")
w_cc = widgets.Textarea(value="mild headache follow-up", description="Complaint", layout=widgets.Layout(width="700px", height="80px"))

btn = widgets.Button(description="Assess patient", button_style="primary")
out = widgets.Output()
```


```python
def on_click_assess(button):
    with out:
        clear_output()

        row_dict = {
            "age": w_age.value,
            "sex": w_sex.value,
            "arrival_mode": w_arrival.value,
            "mental_status_triage": w_mental.value,
            "gcs_total": w_gcs.value,
            "news2_score": w_news2.value,
            "systolic_bp": w_sbp.value,
            "diastolic_bp": w_dbp.value,
            "heart_rate": w_hr.value,
            "respiratory_rate": w_rr.value,
            "temperature_c": w_temp.value,
            "spo2": w_spo2.value,
            "pain_score": w_pain.value,
            "chief_complaint_raw": w_cc.value,
        }

        pred, probs, flag, row = assess_single_patient(row_dict)

        display(Markdown(f"## Predicted ESI-{pred}: {ESI_LABELS[pred]}"))
        display(Markdown(f"**Safety flag:** {flag}"))

        prob_df = pd.DataFrame({
            "ESI": [1, 2, 3, 4, 5],
            "probability": probs,
        })
        display(prob_df)

        threshold_cols = [
            "news2_high", "gcs_severe", "spo2_critical", "sbp_hypotensive", 
            "rr_high", "shock_index_high", "pain_severe", "kw_life_threat", 
            "kw_cardio", "kw_resp", "kw_neuro", "kw_trauma"
        ]
        active = [col for col in threshold_cols if col in row.columns and int(row[col].iloc[0]) == 1]

        if active:
            display(Markdown("**Active clinical flags:** " + ", ".join(active)))
        else:
            display(Markdown("**Active clinical flags:** none"))
```


```python
btn.on_click(on_click_assess)
```


```python
display(widgets.VBox([
    widgets.HTML("<h3>SafeTriage simulator</h3>"),
    widgets.HBox([w_age, w_sex, w_arrival]),
    widgets.HBox([w_mental, w_gcs, w_news2]),
    widgets.HBox([w_sbp, w_dbp, w_hr]),
    widgets.HBox([w_rr, w_temp, w_spo2]),
    widgets.HBox([w_pain]),
    w_cc,
    btn,
    out,
]))
```

# 26. Limitations

1. The dataset is treated as proof-of-concept or competition-inspired, so real-world generalization is not guaranteed.
2. ESI labels include clinical judgment and may contain inter-rater variability.
3. NEWS2 is valid at triage time. In this dataset it is useful, but the ablation shows it is not the main shortcut.
4. Chief complaint text contains highly repeated templates. The notebook therefore reports a synthetic shortcut audit and a GroupKFold stress test instead of overclaiming the near-perfect score.
5. The final ensemble is optimized for the competition-style in-distribution split, not for prospective hospital deployment.
6. The simulator is not a medical device and does not provide medical advice.
7. Fairness analysis is descriptive and requires real deployment data.
8. External validation on MIMIC-IV-ED, NHAMCS, or hospital-specific data would be required before real use.

# 27. Clinical recommendations

1. Use the model as a second-read triage support layer, not as an autonomous triage decision-maker.
2. Prioritize dangerous undertriage monitoring over raw accuracy.
3. Display probabilities and uncertainty, not only one ESI label.
4. Treat predicted ESI-1/2 as high-acuity routing and use review overrides only for predicted ESI-3/4/5 cases with residual high-acuity risk.
5. Use SHAP, clinical thresholds, and complaint keywords for nurse-facing explanation.
6. Monitor ESI-1 and ESI-2 recall continuously.
7. Audit subgroup performance by age, language, insurance, sex, arrival mode, and site.
8. Recalibrate before use in a new hospital or site.
9. Validate on out-of-template complaint text before making any real-world claims.

# 28. Reproducibility checklist

- Random seeds fixed.
- Kaggle paths are direct and explicit.
- GPU training is used for CatBoost and LightGBM where available.
- Leakage columns `disposition` and `ed_los_hours` removed.
- NEWS2 retained as valid triage-time clinical score.
- Train, history, and complaint data merged by patient ID.
- Cross-validation is stratified for the main competition-style score.
- Text features are fit inside folds during validation.
- OOF predictions are used for reported validation metrics.
- Synthetic shortcut audit reports repeated complaint templates.
- GroupKFold stress test checks unseen complaint-template generalization.
- Safety policy thresholds are selected from OOF predictions.
- Submission file is validated before saving.
- Notebook runs end-to-end.
