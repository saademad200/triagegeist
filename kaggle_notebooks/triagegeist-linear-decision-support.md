# Triagegeist: Linear Triage Decision Support

This notebook builds a transparent emergency-triage acuity model from the provided synthetic intake table, chief-complaint text, and patient-history indicators. The goal is not to replace clinicians; it is to show a reproducible decision-support baseline that can flag high-acuity cases consistently and expose where a model would need clinical review before deployment.

## Approach

- Merge structured intake data with chief complaint text and binary patient-history features.
- Exclude post-triage leakage columns (`disposition`, `ed_los_hours`).
- Use a sparse, interpretable pipeline: median-imputed numeric vitals, one-hot categorical fields, TF-IDF chief-complaint text, and a balanced linear SVM.
- Add one conservative safety post-rule discovered in validation error analysis: acute angle-closure glaucoma with unresponsiveness and low GCS is forced to the highest acuity class.


```python
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

SEED = 20260510
random.seed(SEED)
np.random.seed(SEED)

# Kaggle stores competition files under /kaggle/input/<competition-slug>/.
KAGGLE_CANDIDATES = [
    Path('/kaggle/input/triagegeist-local-competition-data'),
    Path('/kaggle/input/triagegeist'),
]
LOCAL_CANDIDATES = [Path('data/raw'), Path('../data/raw'), Path('../../data/raw')]
DATA_DIR = next(
    (p for p in [*KAGGLE_CANDIDATES, *LOCAL_CANDIDATES] if (p / 'train.csv').exists()),
    LOCAL_CANDIDATES[0],
)

OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('../submissions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('DATA_DIR =', DATA_DIR.resolve())
print('OUTPUT_DIR =', OUTPUT_DIR.resolve())
```


```python
def load_frame(data_dir: Path):
    train = pd.read_csv(data_dir / 'train.csv', engine='python')
    test = pd.read_csv(data_dir / 'test.csv', engine='python')
    complaints = pd.read_csv(data_dir / 'chief_complaints.csv', engine='python')
    history = pd.read_csv(data_dir / 'patient_history.csv', engine='python')
    sample = pd.read_csv(data_dir / 'sample_submission.csv', engine='python')

    train = train.merge(complaints, on='patient_id', how='left')
    test = test.merge(complaints, on='patient_id', how='left')
    train = train.merge(history, on='patient_id', how='left')
    test = test.merge(history, on='patient_id', how='left')

    y = train['triage_acuity'].astype(int)
    leakage = {'triage_acuity', 'disposition', 'ed_los_hours'}
    train = train.drop(columns=[c for c in leakage if c in train.columns])
    return train, test, y, sample


def fill_text(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ['chief_complaint_raw', 'chief_complaint_system']:
        if col in out.columns:
            out[col] = out[col].fillna('').astype(str)
    return out

train, test, y, sample = load_frame(DATA_DIR)
train = fill_text(train)
test = fill_text(test)

print('train:', train.shape, 'test:', test.shape)
print('target distribution:')
print(y.value_counts(normalize=True).sort_index().rename('share').to_frame())
```


```python
def build_model(train_frame: pd.DataFrame) -> Pipeline:
    feature_cols = [col for col in train_frame.columns if col != 'patient_id']
    text_cols = [col for col in ['chief_complaint_raw', 'chief_complaint_system'] if col in feature_cols]
    numeric_cols = [
        col for col in feature_cols
        if col not in text_cols and pd.api.types.is_numeric_dtype(train_frame[col])
    ]
    categorical_cols = [col for col in feature_cols if col not in text_cols and col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append((
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scale', StandardScaler(with_mean=False)),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=20, dtype=np.float32)),
            ]),
            categorical_cols,
        ))
    if 'chief_complaint_raw' in text_cols:
        transformers.append((
            'chief_raw_tfidf',
            TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=80_000, sublinear_tf=True),
            'chief_complaint_raw',
        ))
    if 'chief_complaint_system' in text_cols:
        transformers.append((
            'chief_system_tfidf',
            TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True),
            'chief_complaint_system',
        ))

    return Pipeline([
        ('features', ColumnTransformer(transformers=transformers, remainder='drop')),
        ('classifier', LinearSVC(C=0.8, class_weight='balanced', random_state=SEED)),
    ])


def apply_high_acuity_rules(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    out = pred.copy().astype(int)
    required = {'chief_complaint_raw', 'mental_status_triage', 'gcs_total'}
    if not required.issubset(df.columns):
        return out

    complaint = df['chief_complaint_raw'].fillna('').astype(str).str.lower()
    glaucoma_emergency = (
        complaint.str.contains('acute angle closure glaucoma', regex=False)
        & df['mental_status_triage'].eq('unresponsive')
        & (df['gcs_total'] <= 10)
    )
    out[glaucoma_emergency.to_numpy()] = 1
    return out

model = build_model(train)
model
```


```python
x_train, x_valid, y_train, y_valid = train_test_split(
    train,
    y,
    test_size=0.15,
    random_state=SEED,
    stratify=y,
)

model.fit(x_train, y_train)
valid_pred = model.predict(x_valid).astype(int)
valid_pred_post = apply_high_acuity_rules(x_valid, valid_pred)

metrics = {
    'accuracy': accuracy_score(y_valid, valid_pred),
    'macro_f1': f1_score(y_valid, valid_pred, average='macro'),
    'weighted_f1': f1_score(y_valid, valid_pred, average='weighted'),
    'post_accuracy': accuracy_score(y_valid, valid_pred_post),
    'post_macro_f1': f1_score(y_valid, valid_pred_post, average='macro'),
    'post_weighted_f1': f1_score(y_valid, valid_pred_post, average='weighted'),
    'post_changed_valid': int(np.sum(valid_pred != valid_pred_post)),
}
metrics_df = pd.DataFrame([metrics]).T.rename(columns={0: 'value'})
metrics_df
```


```python
report = classification_report(y_valid, valid_pred_post, output_dict=True)
report_df = pd.DataFrame(report).T
report_df[['precision', 'recall', 'f1-score', 'support']]
```


```python
cm = pd.DataFrame(
    confusion_matrix(y_valid, valid_pred_post, labels=[1, 2, 3, 4, 5]),
    index=[f'true_{i}' for i in [1, 2, 3, 4, 5]],
    columns=[f'pred_{i}' for i in [1, 2, 3, 4, 5]],
)
cm
```


```python
error_rows = x_valid.loc[y_valid.to_numpy() != valid_pred_post].copy()
error_rows['true_acuity'] = y_valid.loc[error_rows.index].to_numpy()
error_rows['pred_acuity'] = valid_pred_post[y_valid.to_numpy() != valid_pred_post]
cols = ['patient_id', 'chief_complaint_raw', 'mental_status_triage', 'gcs_total', 'news2_score', 'true_acuity', 'pred_acuity']
error_rows[[c for c in cols if c in error_rows.columns]].head(10)
```

## Train Full Model and Create Submission

The public artifact is a Kaggle notebook, but the code also writes a standard `submission.csv` to `/kaggle/working` for reproducibility and for any host-side checks.


```python
final_model = build_model(train)
final_model.fit(train, y)

test_pred = final_model.predict(test).astype(int)
test_pred = apply_high_acuity_rules(test, test_pred)

submission = sample.copy()
submission['triage_acuity'] = test_pred
submission_path = OUTPUT_DIR / 'submission.csv'
submission.to_csv(submission_path, index=False)

print(submission_path)
print(submission.shape)
print(submission['triage_acuity'].value_counts().sort_index())
submission.head()
```

## Clinical and Deployment Notes

This proof of concept is intentionally simple and auditable. Before any clinical use, it would need external validation on a real hospital dataset, prospective monitoring for subgroup bias and under-triage risk, calibration review, and a workflow where the model supports rather than overrides a licensed clinician. The strongest practical use case is a second-look safety layer that highlights cases whose structured vitals, mental status, and chief complaint suggest higher urgency than the initial queue position.
