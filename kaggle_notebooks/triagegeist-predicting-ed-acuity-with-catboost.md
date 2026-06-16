# Overview

**Problem:** A model that predicts triage acuity level (e.g. ESI, MTS, or equivalent) from structured patient intake data


**Objective:** Develop a predictive model to estimate Emergency Department (ED) triage acuity level using structured patient intake data available at presentation.

The target variable is: triage_acuity = {1,2,3,4,5}

**Where:**
1 = most urgent
5 = least urgent

This is an ordinal multiclass classification problem because acuity levels are naturally ordered.
In a professional healthcare analytics setting, the goal is not simply maximizing accuracy, but:

* minimizing under-triage (high-risk patients predicted as low acuity)
* maintaining acceptable over-triage
* ensuring clinical interpretability
* and building a model robust to missing triage measurements

**Model of choice:** Catboost (classification)



# Libraries


```python
# ====================================
# Import Libraries
# ====================================

# standard
import pandas as pd
import numpy as np
import os, glob

# sklearn
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.impute import *
from sklearn.feature_extraction.text import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

# catboost
from catboost import CatBoostClassifier

# random_state
SEED = 412 
```

# Data


```python
# ====================================
# Load Data
# ====================================

# base folder path
folder_path = "/kaggle/input/competitions/triagegeist"

# get all CSV file paths
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# load into dataframes
train = pd.read_csv(csv_files[3])
test = pd.read_csv(csv_files[4])
complaints = pd.read_csv(csv_files[1])
sample_submission = pd.read_csv(csv_files[0])

# explore
print(f"Training data shape: {train.shape} | Test data shape: {test.shape}\n")
train.head()
```


```python
# merge chief complaints
 
train_df = train.merge(complaints, on="patient_id", how="left")
test_df = test.merge(complaints, on="patient_id", how="left")
train_df.head()
```

# EDA


```python
# data analysis
train.describe().T
```


```python
# check missing
print(train.isnull().sum().sort_values(ascending=False))
```

# Imputation


```python
# columns to impute with median values 
median_cols = [
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "temperature_c"
]

# impute
for c in median_cols:
    med_val_tr = train[c].median()
    med_val_te = test[c].median()

    # train
    train[c] = train[c].fillna(med_val_tr)
    test[c] = test[c].fillna(med_val_te)


# impute - pulse_pressure
train["pulse_pressure"] = (train["systolic_bp"] - train["diastolic_bp"])
test["pulse_pressure"] = (test["systolic_bp"] - test["diastolic_bp"])

# impute - mean_arterial_pressure
train["mean_arterial_pressure"] = (train["diastolic_bp"] + (train["pulse_pressure"] / 3))
test["mean_arterial_pressure"] = (test["diastolic_bp"] + (test["pulse_pressure"] / 3))

# impute - shock_index
train["shock_index"] = (train["heart_rate"] / train["systolic_bp"])
test["shock_index"] = (test["heart_rate"] / test["systolic_bp"])

# check missing - again
print(train.isnull().sum().sort_values(ascending=False))
```

# Data Preprocessing


```python
# drop unwanted columns - train & test
drop_cols = [
    "patient_id",
    "triage_nurse_id",
    "site_id",
    "disposition",
    "ed_los_hours"]

train = train.drop(columns=drop_cols)
test  = test.drop(columns=[c for c in drop_cols if c != "disposition" and c != "ed_los_hours"])
```


```python
# categorical columns
categorical_cols = [
    "arrival_mode", "arrival_day", "sex", "arrival_season",
    "shift", "age_group", "language", "insurance_type",
    "transport_origin", "pain_location", "mental_status_triage",
    "chief_complaint_system"]

# ensure object dtype (don't encode)
for col in categorical_cols:
    train[col] = train[col].astype(str)
    test[col]  = test[col].astype(str)

```

# Feature Engineering


```python
# feature & target
X = train.drop(columns=["triage_acuity"])
y = train["triage_acuity"]
X_test = test.copy()
```


```python
# extract complaints data
train_df["chief_complaint_raw"] = train_df["chief_complaint_raw"].fillna("")
test_df["chief_complaint_raw"]  = test_df["chief_complaint_raw"].fillna("")

# TF-IDF on chief complaint text
tfidf = TfidfVectorizer(max_features=100, ngram_range=(1,2))
train_tfidf = tfidf.fit_transform(train_df["chief_complaint_raw"].fillna(""))
test_tfidf  = tfidf.transform(test_df["chief_complaint_raw"].fillna(""))

# Append as new columns
tfidf_cols = [f"cc_{i}" for i in range(100)]
X = pd.concat([X.reset_index(drop=True), pd.DataFrame(train_tfidf.toarray(), columns=tfidf_cols)], axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(test_tfidf.toarray(), columns=tfidf_cols)], axis=1)
```


```python
# data split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"Training: {X_train.shape} | Validation: {X_val.shape}")
```

# Train


```python
# catBoost model

cat_model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.03,        # lower LR → better generalization
    depth=8,                   # deeper trees capture more interactions
    loss_function="MultiClass",
    eval_metric="TotalF1",
    l2_leaf_reg=5,             # regularization to prevent overfitting
    random_strength=1.0,       # adds noise during splits → better generalization
    bagging_temperature=0.8,   # Bayesian bootstrap
    border_count=254,          # more candidate splits for numerical features
    auto_class_weights="Balanced",
    random_seed=SEED,
    verbose=0,
    task_type="GPU",         # uncomment if you have a GPU — massive speedup
)

# model params
print(cat_model.get_params())
```


```python
# train
cat_model.fit(
    X_train, y_train,
    cat_features=categorical_cols,
    eval_set=(X_val, y_val),
    early_stopping_rounds=150,  # stops if no improvement for 150 rounds
)
```


```python
# prediction

val_preds = cat_model.predict(X_val).flatten()
print("Accuracy:               ", accuracy_score(y_val, val_preds))
print("Macro F1:               ", f1_score(y_val, val_preds, average="macro"))
print("Weighted F1:            ", f1_score(y_val, val_preds, average="weighted"))
print("Quadratic Weighted Kappa:", cohen_kappa_score(y_val, val_preds, weights="quadratic"))
print(classification_report(y_val, val_preds))
```


```python
# training vs validation accuracy
train_preds = cat_model.predict(X_train).flatten()
print("Train Accuracy:", accuracy_score(y_train, train_preds))
print("Val Accuracy:  ", accuracy_score(y_val, val_preds))
```


```python
# feature importance

print(cat_model.get_feature_importance(prettified=True).head(20))
```


```python
# submission

test_preds = cat_model.predict(X_test).flatten()
submission = sample_submission.copy()
submission["triage_acuity"] = test_preds
submission.to_csv("submission.csv", index=False)
```
