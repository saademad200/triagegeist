# Overview

Description:

Every minute counts in the emergency department. Across the world, emergency physicians and nurses make rapid, high-stakes triage decisions under extreme cognitive load, with incomplete information, and in environments that are chronically understaffed. Errors in triage severity scoring are not abstractions: they lead to delayed care, adverse outcomes, and preventable deaths.

The Laitinen-Fredriksson Foundation is a Finnish medical research foundation dedicated to advancing clinical decision support in acute and emergency medicine. Through Triagegeist, we invite data scientists, clinicians, and AI practitioners to tackle one of the most consequential problems in modern healthcare: can AI meaningfully support triage decisions in the emergency department?

Problem: A model that predicts triage acuity level (e.g. ESI, MTS, or equivalent) from structured patient intake data

Objective

Develop a predictive model to estimate Emergency Department (ED) triage acuity level using structured patient intake data available at presentation.

The target variable is:

triage_acuity = {1,2,3,4,5}

where:

1 = most urgent
5 = least urgent

This is an ordinal multiclass classification problem because acuity levels are naturally ordered.

In a professional healthcare analytics setting, the goal is not simply maximizing accuracy, but:

minimizing under-triage (high-risk patients predicted as low acuity),
maintaining acceptable over-triage,
ensuring clinical interpretability,
and building a model robust to missing triage measurements.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
```


```python
train = pd.read_csv("/kaggle/input/competitions/triagegeist/train.csv")
test = pd.read_csv("/kaggle/input/competitions/triagegeist/test.csv")
complaints = pd.read_csv("/kaggle/input/competitions/triagegeist/chief_complaints.csv")
sample_submission = pd.read_csv("/kaggle/input/competitions/triagegeist/sample_submission.csv")
```


```python
print(train.shape)
print(test.shape)
train.head()
```


```python
 #Merge chief complaints
train_df = train.merge(complaints, on="patient_id", how="left")
test_df = test.merge(complaints, on="patient_id", how="left")
train_df.head()

```


```python
print(train.info())
print(train.describe())
```

# Missing values analysis


```python
missing_summary = train.isnull().sum().sort_values(ascending=False)
missing_summary
```


```python
missing_percent = train.isnull().mean().sort_values(ascending=False) * 100
missing_percent
```


```python
missing_percent[missing_percent > 0].plot(kind="bar")
plt.title("Percentage of Missing Values")
plt.ylabel("Missing Percentage")
plt.show()
```

Missingness is clinically informative. Some vitals may be missing because the patient appeared low-risk, so we should not simply drop missing records.


```python
vital_cols = [
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "temperature_c",
    "spo2"
]

for col in vital_cols:
    train[col + "_missing"] = train[col].isna().astype(int)
    test[col + "_missing"] = test[col].isna().astype(int)
```


```python
median_cols = [
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "temperature_c",
    "spo2"
]

for col in median_cols:
    median_value = train[col].median()

    train[col] = train[col].fillna(median_value)
    test[col] = test[col].fillna(median_value)
```


```python
# Pulse pressure
train["pulse_pressure"] = (
    train["systolic_bp"] -
    train["diastolic_bp"]
)

test["pulse_pressure"] = (
    test["systolic_bp"] -
    test["diastolic_bp"]
)

# Mean arterial pressure (MAP)
train["mean_arterial_pressure"] = (
    train["diastolic_bp"] +
    (train["pulse_pressure"] / 3)
)

test["mean_arterial_pressure"] = (
    test["diastolic_bp"] +
    (test["pulse_pressure"] / 3)
)

# Shock index
train["shock_index"] = (
    train["heart_rate"] /
    train["systolic_bp"]
)

test["shock_index"] = (
    test["heart_rate"] /
    test["systolic_bp"]
)
```


```python
train.isnull().sum()
```

# EDA


```python
sns.countplot(data=train, x="triage_acuity")
plt.title("Distribution of Triage Acuity Levels")
plt.show()
```


```python
numeric_cols_for_eda = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "temperature_c",
    "spo2",
    "gcs_total",
    "pain_score",
    "num_prior_ed_visits_12m"
]

# Create subplot grid
fig, ax = plt.subplots(
    4, 3,
    figsize=(18, 16)
)

# Flatten axes for easy looping
ax = ax.flatten()

# Loop through variables
for i, col in enumerate(numeric_cols_for_eda):

    sns.histplot(
        data=train,
        x=col,
        kde=True,
        bins=30,
        ax=ax[i]
    )

    ax[i].set_title(
        f"{col.replace('_', ' ').title()} Distribution"
    )

# Remove extra empty plots
for j in range(len(numeric_cols_for_eda), len(ax)):
    fig.delaxes(ax[j])

plt.tight_layout()
plt.show()
```


```python
# Create subplot layout
fig, ax = plt.subplots(
    4, 3,
    figsize=(20, 16)
)

# Flatten axes for looping
ax = ax.flatten()

# Create boxplots
for i, col in enumerate(numeric_cols_for_eda):

    sns.boxplot(
        data=train,
        x="triage_acuity",
        y=col,
        ax=ax[i]
    )

    ax[i].set_title(
        f"{col.replace('_', ' ').title()} by Triage Acuity"
    )

    ax[i].set_xlabel("Triage Acuity")

# Remove unused subplot spaces
for j in range(len(numeric_cols_for_eda), len(ax)):
    fig.delaxes(ax[j])

plt.tight_layout()
plt.show()
```

Patients classified into higher acuity 


```python
numeric_features = train.select_dtypes(
    include=np.number
)

corr = numeric_features.corr()

plt.figure(figsize=(14,10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix")
plt.show()
```

Strong correlations were observed between systolic and diastolic blood pressure.


```python
#This checks whether arrival mode, sex, or day of arrival is related to triage acuity.
categorical_cols_for_eda = [
    "arrival_mode",
    "arrival_day",
    "sex",
    "arrival_season",
    "shift",
    "age_group",
    "language",
    "insurance_type",
    "transport_origin",
    "pain_location",
    "mental_status_triage",
    "chief_complaint_system"
]

# Create subplot grid
fig, ax = plt.subplots(
    4, 3,
    figsize=(22, 18)
)

# Flatten axes for looping
ax = ax.flatten()

# Create countplots
for i, col in enumerate(categorical_cols_for_eda):

    sns.countplot(
        data=train,
        x=col,
        hue="triage_acuity",
        ax=ax[i]
    )

    ax[i].set_title(
        f"{col.replace('_',' ').title()} by Triage Acuity",
        fontsize=12
    )

    ax[i].tick_params(
        axis="x",
        rotation=45
    )

    ax[i].set_xlabel("")

# Remove extra empty subplots
for j in range(len(categorical_cols_for_eda), len(ax)):
    fig.delaxes(ax[j])

plt.tight_layout()
plt.show()
```

# Data Preprocessing


```python
train["pain_score_missing"] = (
    train["pain_score"] == -1
).astype(int)

test["pain_score_missing"] = (
    test["pain_score"] == -1
).astype(int)

train["pain_score"] = train["pain_score"].replace(-1, np.nan)
test["pain_score"] = test["pain_score"].replace(-1, np.nan)

pain_median = train["pain_score"].median()

train["pain_score"] = train["pain_score"].fillna(pain_median)
test["pain_score"] = test["pain_score"].fillna(pain_median)
```


```python
drop_cols = [
    "patient_id",
    "pulse_pressure",
    "mean_arterial_pressure","triage_nurse_id", "site_id", "disposition", "ed_los_hours", "pain_score_missing"
]

train = train.drop(columns=drop_cols)
drop_col = [
    "patient_id",
    "pulse_pressure",
    "mean_arterial_pressure","triage_nurse_id", "site_id", "pain_score_missing"
]
test = test.drop(columns=drop_col)
```


```python
categorical_cols = [
    "arrival_mode",
    "arrival_day",
    "sex", "arrival_season","shift", "age_group", "language", "insurance_type", "transport_origin", "pain_location",
    "mental_status_triage", "chief_complaint_system"
]
```


```python
from sklearn.preprocessing import LabelEncoder
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()

    train[col] = le.fit_transform(train[col])

    test[col] = le.transform(test[col])

    encoders[col] = le
```


```python
X = train.drop(columns=["triage_acuity"])
y = train["triage_acuity"]

X_test = test.copy()
```


```python
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

# Logistic Regression Model


```python
log_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    multi_class="multinomial",
    solver="lbfgs"
)

log_model.fit(X_train, y_train)
```


```python
val_preds = log_model.predict(X_val)
print("Accuracy:",
      accuracy_score(y_val, val_preds))

print("Macro F1:",
      f1_score(y_val, val_preds,
               average="macro"))

print("Weighted F1:",
      f1_score(y_val, val_preds,
               average="weighted"))

print(
    "Quadratic Weighted Kappa:",
    cohen_kappa_score(
        y_val,
        val_preds,
        weights="quadratic"
    )
)

print(classification_report(
    y_val,
    val_preds
))

cm = confusion_matrix(
    y_val,
    val_preds
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

# ESI Triage Acuity prediction using the testing dataset


```python
test_preds = log_model.predict(X_test)
```


```python
submission = sample_submission.copy()

submission["triage_acuity"] = test_preds

submission.to_csv(
    "submission.csv",
    index=False
)

print("Submission saved.")
```


```python
submission.head()
```

# CatBoost Classifier


```python
from catboost import CatBoostClassifier
```


```python
cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    random_seed=42,
    auto_class_weights="Balanced",
    verbose=100
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=categorical_cols,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100
)
```


```python
val_preds = cat_model.predict(X_val)
val_preds = val_preds.flatten()
print("Accuracy:", accuracy_score(y_val, val_preds))
print("Macro F1:", f1_score(y_val, val_preds, average="macro"))
print("Weighted F1:", f1_score(y_val, val_preds, average="weighted"))
print("Quadratic Weighted Kappa:", cohen_kappa_score(y_val, val_preds, weights="quadratic"))

print(classification_report(y_val, val_preds))
```


```python
cm = confusion_matrix(y_val, val_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - CatBoost")
plt.show()
```


```python
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": cat_model.get_feature_importance()
}).sort_values(by="importance", ascending=False)

feature_importance.head(15)
```


```python
plt.figure(figsize=(10, 7))
sns.barplot(
    data=feature_importance.head(10),
    x="importance",
    y="feature"
)

plt.title("Top 10 Important Features - CatBoost")
plt.show()
```

Pain score, news2 score, Oxygen saturation, Glasgow Coma Scale, temperature, respiratory rate, systolic blood pressure, mental status, number of prior ED visits in the past 12 months, Diastolic blood pressure, and  emerged among the strongest predictors of triage acuity, aligning with established emergency medicine triage principles.


```python
test_preds_cat = cat_model.predict(X_test)
submission_cat = sample_submission.copy()

submission_cat["triage_acuity"] = test_preds

submission_cat.to_csv(
    "submission_cat.csv",
    index=False
)
submission_cat.head()
```
