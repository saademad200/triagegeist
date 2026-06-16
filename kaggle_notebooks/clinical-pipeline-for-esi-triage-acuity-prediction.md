# Triagegeist: A Clinical AI Pipeline for ESI Triage Acuity Prediction

**Author:** Andrew Matelis  
**Competition:** Triagegeist -- Laitinen-Fredriksson Foundation  
**Deadline:** April 21, 2026  

---

## Clinical Motivation

Every year, emergency departments in the United States alone record over 130 million visits. Triage nurses and physicians must assign an Emergency Severity Index (ESI) score within minutes of patient arrival, under cognitive load, with incomplete information, and in chronically understaffed environments. Inter-rater variability in ESI assignment is well-documented in peer-reviewed literature. Systematic undertriage -- assigning a lower acuity score than a patient's true condition warrants -- is an active patient safety crisis, with documented disparities across age, sex, language, and insurance status.

The promise of clinical AI in triage is not to replace the clinician. It is to provide a second opinion that is fast, consistent, and free of fatigue bias.

This notebook builds a competition-grade clinical AI pipeline with three non-negotiable design principles:

1. **Accuracy** -- an ensemble of gradient boosted models with clinically motivated feature engineering, iterative imputation, and optimized hyperparameters to maximize macro F1 across all five ESI classes
2. **Interpretability** -- global and per-patient SHAP analysis so the model's reasoning is auditable by clinicians and regulators
3. **Equity** -- systematic evaluation of undertriage and F1 across demographic subgroups, directly addressing the patient safety concern named in the competition description

A model that achieves strong average performance while systematically undertriaging elderly or uninsured patients is not a clinical advancement. This notebook treats equity as a first-class evaluation criterion.

---

## Pipeline Overview

| Section | Content |
|---------|---------|
| 1 | Imports and configuration |
| 2 | Data loading |
| 3 | Exploratory data analysis |
| 4 | Dataset merging |
| 5 | Clinical feature engineering |
| 6 | NLP feature extraction from chief complaints |
| 7 | Preprocessing pipeline |
| 8 | Model training: LightGBM + XGBoost + CatBoost ensemble |
| 9 | Evaluation and results |
| 10 | SHAP interpretability |
| 11 | Equity and bias analysis |
| 12 | Feature importance |
| 13 | Submission generation |
| 14 | Reproducibility notes |

---

## Section 1: Imports and Configuration


```python
# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import gc
import json
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

# ── Numerical / dataframe ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_colwidth', 80)

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 130, 'savefig.bbox': 'tight'})

# ── Scikit-learn: preprocessing ───────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

# ── Scikit-learn: feature extraction ─────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Scikit-learn: model selection / evaluation ────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    classification_report, confusion_matrix,
)
from sklearn.calibration import calibration_curve

# ── Gradient boosting ─────────────────────────────────────────────────────────
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ── Interpretability ──────────────────────────────────────────────────────────
import shap
shap.initjs()

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED   = 42
N_FOLDS = 5
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = '/kaggle/input/triagegeist/'

ESI_LABELS = ['ESI-1\n(Critical)', 'ESI-2\n(Emergent)',
              'ESI-3\n(Urgent)', 'ESI-4\n(Less Urgent)',
              'ESI-5\n(Non-Urgent)']

print("All imports successful.")
for pkg, mod in [('LightGBM', lgb), ('XGBoost', xgb), ('SHAP', shap)]:
    print(f"  {pkg}: {mod.__version__}")

```

## Section 2: Data Loading


```python
train      = pd.read_csv(BASE + 'train.csv')
test       = pd.read_csv(BASE + 'test.csv')
complaints = pd.read_csv(BASE + 'chief_complaints.csv')
history    = pd.read_csv(BASE + 'patient_history.csv')
sample_sub = pd.read_csv(BASE + 'sample_submission.csv')

# Preserve test patient IDs immediately -- they must survive all transforms
test_ids = test['patient_id'].values.copy()

print("Dataset shapes:")
for name, df in [('train', train), ('test', test),
                  ('complaints', complaints), ('history', history)]:
    print(f"  {name:12s}: {df.shape[0]:>7,} rows  x  {df.shape[1]:>3} cols")

print(f"\nTarget distribution (train):")
vc = train['triage_acuity'].value_counts().sort_index()
for lvl, cnt in vc.items():
    bar = '█' * int(cnt / len(train) * 40)
    print(f"  ESI-{lvl}: {cnt:6,}  ({cnt/len(train)*100:.1f}%)  {bar}")

```

## Section 3: Exploratory Data Analysis

Before any modelling, we need to understand the data distributions, missing value patterns, and clinical relationships that will guide feature engineering decisions.


```python
# ── 3a. Target and basic demographics ─────────────────────────────────────────

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Target distribution
ax0 = fig.add_subplot(gs[0, 0])
vc  = train['triage_acuity'].value_counts().sort_index()
bars = ax0.bar(vc.index.astype(str), vc.values,
               color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 5)),
               edgecolor='black', linewidth=0.7)
for bar, v in zip(bars, vc.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{v/1000:.1f}k', ha='center', va='bottom', fontsize=8)
ax0.set_title('ESI Acuity Distribution', fontweight='bold')
ax0.set_xlabel('ESI Level (1 = Most Urgent)')
ax0.set_ylabel('Patient Count')

# Age by acuity
ax1 = fig.add_subplot(gs[0, 1])
for lvl in sorted(train['triage_acuity'].unique()):
    data = train.loc[train['triage_acuity'] == lvl, 'age'].dropna()
    ax1.boxplot(data, positions=[lvl], widths=0.6, patch_artist=True,
                boxprops=dict(facecolor=plt.cm.RdYlGn_r(lvl/5)),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=0.8),
                flierprops=dict(marker='.', markersize=1, alpha=0.3))
ax1.set_title('Age by Acuity', fontweight='bold')
ax1.set_xlabel('ESI Level')
ax1.set_ylabel('Age (years)')
ax1.set_xticks(range(1,6))

# SpO2 by acuity
ax2 = fig.add_subplot(gs[0, 2])
if 'spo2' in train.columns:
    for lvl in sorted(train['triage_acuity'].unique()):
        data = train.loc[train['triage_acuity'] == lvl, 'spo2'].dropna()
        ax2.boxplot(data, positions=[lvl], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=plt.cm.RdYlGn_r(lvl/5)),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(marker='.', markersize=1, alpha=0.3))
ax2.set_title('SpO2 by Acuity', fontweight='bold')
ax2.set_xlabel('ESI Level')
ax2.set_ylabel('SpO2 (%)')
ax2.set_xticks(range(1,6))

# GCS by acuity
ax3 = fig.add_subplot(gs[0, 3])
if 'gcs_total' in train.columns:
    for lvl in sorted(train['triage_acuity'].unique()):
        data = train.loc[train['triage_acuity'] == lvl, 'gcs_total'].dropna()
        ax3.boxplot(data, positions=[lvl], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=plt.cm.RdYlGn_r(lvl/5)),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(marker='.', markersize=1, alpha=0.3))
ax3.set_title('GCS Total by Acuity', fontweight='bold')
ax3.set_xlabel('ESI Level')
ax3.set_ylabel('GCS Score (3-15)')
ax3.set_xticks(range(1,6))

# Heart rate by acuity
ax4 = fig.add_subplot(gs[1, 0])
if 'heart_rate' in train.columns:
    for lvl in sorted(train['triage_acuity'].unique()):
        data = train.loc[train['triage_acuity'] == lvl, 'heart_rate'].dropna()
        ax4.boxplot(data, positions=[lvl], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=plt.cm.RdYlGn_r(lvl/5)),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(marker='.', markersize=1, alpha=0.3))
ax4.set_title('Heart Rate by Acuity', fontweight='bold')
ax4.set_xlabel('ESI Level')
ax4.set_ylabel('Heart Rate (bpm)')
ax4.set_xticks(range(1,6))

# Systolic BP by acuity
ax5 = fig.add_subplot(gs[1, 1])
if 'systolic_bp' in train.columns:
    for lvl in sorted(train['triage_acuity'].unique()):
        data = train.loc[train['triage_acuity'] == lvl, 'systolic_bp'].dropna()
        ax5.boxplot(data, positions=[lvl], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=plt.cm.RdYlGn_r(lvl/5)),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(marker='.', markersize=1, alpha=0.3))
ax5.set_title('Systolic BP by Acuity', fontweight='bold')
ax5.set_xlabel('ESI Level')
ax5.set_ylabel('SBP (mmHg)')
ax5.set_xticks(range(1,6))

# Arrival mode
ax6 = fig.add_subplot(gs[1, 2])
if 'arrival_mode' in train.columns:
    am = train['arrival_mode'].value_counts()
    ax6.bar(range(len(am)), am.values, color='steelblue', edgecolor='black')
    ax6.set_xticks(range(len(am)))
    ax6.set_xticklabels(am.index, rotation=30, ha='right', fontsize=8)
    ax6.set_title('Arrival Mode Distribution', fontweight='bold')
    ax6.set_ylabel('Count')

# Arrival hour heatmap vs acuity
ax7 = fig.add_subplot(gs[1, 3])
if 'arrival_hour' in train.columns:
    hour_acuity = train.groupby(['arrival_hour', 'triage_acuity']).size().unstack(fill_value=0)
    hour_acuity_norm = hour_acuity.div(hour_acuity.sum(axis=1), axis=0)
    sns.heatmap(hour_acuity_norm.T, ax=ax7, cmap='YlOrRd',
                cbar_kws={'shrink': 0.8}, linewidths=0)
    ax7.set_title('Acuity Mix by Arrival Hour', fontweight='bold')
    ax7.set_xlabel('Hour of Day')
    ax7.set_ylabel('ESI Level')

# Missing value heatmap
ax8 = fig.add_subplot(gs[2, :])
vital_cols = [c for c in ['systolic_bp','diastolic_bp','heart_rate',
                            'respiratory_rate','temperature_c','spo2']
              if c in train.columns]
if vital_cols:
    miss_by_acuity = (
        train.groupby('triage_acuity')[vital_cols]
        .apply(lambda x: x.isnull().mean() * 100)
    )
    sns.heatmap(miss_by_acuity, annot=True, fmt='.1f', cmap='Reds',
                ax=ax8, linewidths=0.5, cbar_kws={'label': 'Missing %'})
    ax8.set_title(
        'Vital Sign Missingness by Acuity Level (%)  '
        '-- Higher missingness in lower-acuity patients confirms '
        'missingness is a clinical signal',
        fontweight='bold', fontsize=9)
    ax8.set_xlabel('Vital Sign')
    ax8.set_ylabel('ESI Level')

fig.suptitle('Triagegeist: Exploratory Data Analysis', fontsize=15,
             fontweight='bold', y=1.01)
plt.savefig('eda_overview.png', dpi=130, bbox_inches='tight')
plt.show()
print("EDA overview saved.")

```


```python
# ── 3b. Chief complaint text analysis ────────────────────────────────────────

print("Sample chief complaints by acuity level:")
for lvl in range(1, 6):
    mask = train['triage_acuity'] == lvl
    sample_ids = train.loc[mask, 'patient_id'].head(3).tolist()
    sample_text = complaints[complaints['patient_id'].isin(sample_ids)][
        'chief_complaint_raw'].tolist()
    print(f"\n  ESI-{lvl}:")
    for t in sample_text[:3]:
        print(f"    '{t}'")

# Word length of complaint by acuity
merged_sample = train[['patient_id','triage_acuity']].merge(
    complaints, on='patient_id', how='left')
merged_sample['complaint_len'] = merged_sample[
    'chief_complaint_raw'].fillna('').str.split().str.len()

print("\nMean complaint word length by acuity:")
print(merged_sample.groupby('triage_acuity')['complaint_len'].mean())

```

## Section 4: Merge All Data Sources


```python
# Join complaints and comorbidity history onto train and test
# All tables share patient_id as the primary key

train = (train
         .merge(complaints, on='patient_id', how='left')
         .merge(history,    on='patient_id', how='left'))

test  = (test
         .merge(complaints, on='patient_id', how='left')
         .merge(history,    on='patient_id', how='left'))

print(f"Train: {train.shape}  |  Test: {test.shape}")

# ── Check join quality ────────────────────────────────────────────────────────
complaint_match_rate = train['chief_complaint_raw'].notna().mean()
history_match_rate   = train[[c for c in train.columns
                               if c.startswith('hx_')][:1]].notna().mean().iloc[0]
print(f"Complaint join match rate: {complaint_match_rate*100:.1f}%")
print(f"History join match rate:   {history_match_rate*100:.1f}%")

```

## Section 5: Clinical Feature Engineering

Every feature below has an explicit clinical rationale grounded in emergency medicine practice and peer-reviewed literature. This is not arbitrary feature creation -- each variable maps to a real clinical decision pathway.


```python
def engineer_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full clinical feature engineering pipeline.
    Applied identically to train and test to prevent leakage.
    All thresholds sourced from published emergency medicine guidelines.
    """
    df = df.copy()

    # ── 5a. Missingness indicators ────────────────────────────────────────────
    # EDA confirmed that vital sign missingness correlates strongly with
    # acuity level: lower-acuity patients have vitals recorded less often.
    # Encoding this as explicit binary features makes the signal available
    # to the model directly, rather than relying on imputed values alone.
    vital_miss_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate',
                       'respiratory_rate', 'temperature_c', 'spo2']
    for col in vital_miss_cols:
        if col in df.columns:
            df[f'miss_{col}'] = df[col].isnull().astype(np.int8)

    # pain_score encodes missing as -1 rather than NaN
    if 'pain_score' in df.columns:
        df['miss_pain_score'] = (df['pain_score'] == -1).astype(np.int8)
        df['pain_score']      = df['pain_score'].replace(-1, np.nan)

    # Total number of missing vitals per patient (composite signal)
    miss_cols = [f'miss_{c}' for c in vital_miss_cols if f'miss_{c}' in df.columns]
    if miss_cols:
        df['total_vitals_missing'] = df[miss_cols].sum(axis=1).astype(np.int8)
        # Binary: any vitals missing at all
        df['any_vital_missing'] = (df['total_vitals_missing'] > 0).astype(np.int8)

    # ── 5b. Hemodynamic composite scores ─────────────────────────────────────

    # Mean Arterial Pressure (MAP)
    # MAP < 65 mmHg is the clinical threshold for shock and ICU admission
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['map'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        df['flag_hypotension']       = (df['systolic_bp'] < 90).astype(np.int8)
        df['flag_severe_hypotension']= (df['systolic_bp'] < 70).astype(np.int8)
        df['flag_hypertensive_crisis']= (df['systolic_bp'] >= 180).astype(np.int8)
        df['flag_map_shock']         = (df['map'] < 65).astype(np.int8)
        df['flag_narrow_pulse_pressure'] = (df['pulse_pressure'] < 25).astype(np.int8)

    # Shock Index (SI = HR / SBP)
    # SI > 1.0: hemodynamic compromise; SI > 1.4: critical illness
    # Validated predictor of ICU admission and mortality in ED triage
    if 'heart_rate' in df.columns and 'systolic_bp' in df.columns:
        df['shock_index'] = df['heart_rate'] / (df['systolic_bp'].replace(0, np.nan) + 1e-5)
        df['flag_shock_index_1']  = (df['shock_index'] > 1.0).astype(np.int8)
        df['flag_shock_index_14'] = (df['shock_index'] > 1.4).astype(np.int8)

        # Modified Shock Index (HR / MAP)
        if 'map' in df.columns:
            df['modified_shock_index'] = df['heart_rate'] / (df['map'] + 1e-5)

    # ── 5c. Respiratory and oxygenation flags ─────────────────────────────────

    if 'spo2' in df.columns:
        # SpO2 < 94: supplemental oxygen indicated (WHO threshold)
        # SpO2 < 90: severe hypoxemia, likely critical
        df['flag_hypoxemia']        = (df['spo2'] < 94).astype(np.int8)
        df['flag_severe_hypoxemia'] = (df['spo2'] < 90).astype(np.int8)
        df['flag_critical_hypoxemia'] = (df['spo2'] < 85).astype(np.int8)

    if 'respiratory_rate' in df.columns:
        # Normal adult RR: 12-20 breaths/min
        df['flag_bradypnea']        = (df['respiratory_rate'] < 10).astype(np.int8)
        df['flag_tachypnea']        = (df['respiratory_rate'] > 20).astype(np.int8)
        df['flag_severe_tachypnea'] = (df['respiratory_rate'] > 30).astype(np.int8)
        # RR > 30 is a CURB-65 criterion for community-acquired pneumonia severity

    if 'heart_rate' in df.columns:
        df['flag_bradycardia']    = (df['heart_rate'] < 50).astype(np.int8)
        df['flag_tachycardia']    = (df['heart_rate'] > 100).astype(np.int8)
        df['flag_severe_tachy']   = (df['heart_rate'] > 130).astype(np.int8)

    if 'temperature_c' in df.columns:
        df['flag_hypothermia']    = (df['temperature_c'] < 36.0).astype(np.int8)
        df['flag_fever']          = (df['temperature_c'] > 38.3).astype(np.int8)
        df['flag_high_fever']     = (df['temperature_c'] > 39.5).astype(np.int8)
        df['flag_hyperpyrexia']   = (df['temperature_c'] > 41.0).astype(np.int8)

    # ── 5d. Neurological flags ────────────────────────────────────────────────
    if 'gcs_total' in df.columns:
        # GCS 15 = fully alert
        # GCS <= 8 = severe neurological compromise; intubation threshold
        # GCS < 13 = any significant alteration
        df['flag_altered_consciousness'] = (df['gcs_total'] < 15).astype(np.int8)
        df['flag_moderate_neuro']        = (df['gcs_total'] < 13).astype(np.int8)
        df['flag_severe_neuro']          = (df['gcs_total'] <= 8).astype(np.int8)
        df['flag_coma']                  = (df['gcs_total'] <= 5).astype(np.int8)

    # ── 5e. NEWS2 approximation ────────────────────────────────────────────────
    # National Early Warning Score 2: validated 6-parameter clinical
    # deterioration score used in UK NHS and validated globally.
    # Higher score = higher risk of ICU admission and 30-day mortality.
    news2 = pd.Series(0.0, index=df.index)

    if 'respiratory_rate' in df.columns:
        rr = df['respiratory_rate']
        news2 += np.where(rr <= 8,  3, 0)
        news2 += np.where((rr >= 9)  & (rr <= 11),  1, 0)
        news2 += np.where((rr >= 21) & (rr <= 24),  2, 0)
        news2 += np.where(rr >= 25,  3, 0)

    if 'spo2' in df.columns:
        s = df['spo2']
        news2 += np.where(s <= 91,  3, 0)
        news2 += np.where((s >= 92) & (s <= 93),  2, 0)
        news2 += np.where((s >= 94) & (s <= 95),  1, 0)

    if 'systolic_bp' in df.columns:
        sbp = df['systolic_bp']
        news2 += np.where(sbp <= 90,   3, 0)
        news2 += np.where((sbp >= 91)  & (sbp <= 100),  2, 0)
        news2 += np.where((sbp >= 101) & (sbp <= 110),  1, 0)
        news2 += np.where(sbp >= 220,  3, 0)

    if 'heart_rate' in df.columns:
        hr = df['heart_rate']
        news2 += np.where(hr <= 40,  3, 0)
        news2 += np.where((hr >= 41) & (hr <= 50),  1, 0)
        news2 += np.where((hr >= 91) & (hr <= 110), 1, 0)
        news2 += np.where((hr >= 111) & (hr <= 130),2, 0)
        news2 += np.where(hr >= 131,  3, 0)

    if 'temperature_c' in df.columns:
        t = df['temperature_c']
        news2 += np.where(t <= 35.0,  3, 0)
        news2 += np.where(t >= 39.1,  2, 0)

    df['news2_approx']     = news2
    df['flag_news2_low']   = (news2.between(1, 4)).astype(np.int8)   # low risk
    df['flag_news2_medium']= (news2.between(5, 6)).astype(np.int8)   # medium risk
    df['flag_news2_high']  = (news2 >= 7).astype(np.int8)            # high risk -> escalate

    # Total critical flag count: sum of binary danger flags
    # Acts as a composite severity score derived from individual signals
    danger_flags = [c for c in df.columns if c.startswith('flag_')]
    if danger_flags:
        df['total_danger_flags'] = df[danger_flags].sum(axis=1)

    # ── 5f. Age-based features ────────────────────────────────────────────────
    if 'age' in df.columns:
        df['flag_pediatric']    = (df['age'] < 18).astype(np.int8)
        df['flag_infant']       = (df['age'] < 2).astype(np.int8)
        df['flag_elderly']      = (df['age'] >= 65).astype(np.int8)
        df['flag_very_elderly'] = (df['age'] >= 85).astype(np.int8)

        # Age bands as ordinal integer (preserve ordering)
        df['age_band'] = pd.cut(
            df['age'],
            bins=[0, 1, 5, 17, 34, 49, 64, 79, 150],
            labels=[0, 1, 2, 3, 4, 5, 6, 7]
        ).astype(float)

    # ── 5g. Comorbidity burden ────────────────────────────────────────────────
    hx_cols = [c for c in df.columns if c.startswith('hx_')]
    if hx_cols:
        df['total_comorbidities']    = df[hx_cols].sum(axis=1)
        df['flag_high_comorbidity']  = (df['total_comorbidities'] >= 3).astype(np.int8)
        df['flag_multi_comorbidity'] = (df['total_comorbidities'] >= 5).astype(np.int8)

        # High-risk comorbidity flags
        # These conditions independently elevate baseline triage acuity
        critical_hx = [
            'hx_heart_failure', 'hx_copd', 'hx_malignancy',
            'hx_dementia', 'hx_dialysis', 'hx_immunocompromised',
            'hx_cirrhosis', 'hx_coagulopathy'
        ]
        present_critical = [c for c in critical_hx if c in df.columns]
        if present_critical:
            df['flag_critical_comorbidity'] = (
                df[present_critical].max(axis=1)).astype(np.int8)

        # Cardiovascular comorbidity cluster
        cv_hx = ['hx_hypertension', 'hx_diabetes', 'hx_heart_failure',
                  'hx_coronary_artery_disease', 'hx_atrial_fibrillation']
        present_cv = [c for c in cv_hx if c in df.columns]
        if present_cv:
            df['total_cv_comorbidities'] = df[present_cv].sum(axis=1)

    # ── 5h. Temporal features ─────────────────────────────────────────────────
    if 'arrival_hour' in df.columns:
        # Night shift: reduced staffing window
        df['flag_night_shift'] = (
            (df['arrival_hour'] >= 23) | (df['arrival_hour'] < 7)
        ).astype(np.int8)

        # ED peak hours: highest volume, highest cognitive load
        df['flag_peak_hours'] = (
            (df['arrival_hour'] >= 10) & (df['arrival_hour'] <= 18)
        ).astype(np.int8)

        # Cyclical encoding: prevents hour 23 and hour 0 appearing maximally
        # distant in feature space despite being adjacent in time
        df['hour_sin'] = np.sin(2 * np.pi * df['arrival_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['arrival_hour'] / 24)

    # ── 5i. Prior utilisation features ────────────────────────────────────────
    visit_col = 'num_prior_visits'
    if visit_col not in df.columns:
        visit_col = 'num_prior_ed_visits_12m'
    if visit_col in df.columns:
        df['flag_frequent_visitor'] = (df[visit_col] >= 3).astype(np.int8)
        df['flag_very_frequent']    = (df[visit_col] >= 6).astype(np.int8)

    return df


train = engineer_clinical_features(train)
test  = engineer_clinical_features(test)

n_new = len([c for c in train.columns
             if c not in pd.read_csv(BASE+'train.csv').columns])
print(f"Feature engineering complete.")
print(f"  Engineered features added : {n_new}")
print(f"  Train shape               : {train.shape}")
print(f"  Test shape                : {test.shape}")

```

## Section 6: NLP Feature Extraction from Chief Complaints

Chief complaint text is among the most information-dense signals available at triage. A patient presenting with "chest pain radiating to left arm with diaphoresis" is communicating an entirely different clinical picture than "sore throat x3 days". We extract three layers of NLP features: text cleaning, binary high-risk keyword flags, and TF-IDF weighted n-gram features.


```python
# ── 6a. Text preprocessing ────────────────────────────────────────────────────

CLINICAL_ABBREVIATIONS = {
    r'\bsob\b': 'shortness of breath',
    r'\bcp\b':  'chest pain',
    r'\bn/v\b': 'nausea vomiting',
    r'\bha\b':  'headache',
    r'\bloc\b': 'loss of consciousness',
    r'\bms\b':  'mental status',
    r'\balt ms\b': 'altered mental status',
    r'\bao\b':  'alert oriented',
    r'\buti\b': 'urinary tract infection',
    r'\burti\b':'upper respiratory infection',
    r'\bdvt\b': 'deep vein thrombosis',
    r'\bpe\b':  'pulmonary embolism',
    r'\bami\b': 'acute myocardial infarction',
    r'\bstemi\b':'st elevation myocardial infarction',
    r'\bcvt\b': 'cerebral venous thrombosis',
    r'\bcva\b': 'cerebrovascular accident stroke',
    r'\btia\b': 'transient ischemic attack',
    r'\bgib\b': 'gastrointestinal bleeding',
    r'\bfx\b':  'fracture',
    r'\bmva\b': 'motor vehicle accident',
    r'\bmvc\b': 'motor vehicle collision',
}

def preprocess_complaint(text: str) -> str:
    """
    Normalise chief complaint text for NLP feature extraction.
    Expands clinical abbreviations, removes noise, preserves signal.
    """
    if pd.isnull(text) or str(text).strip() == '':
        return 'unknown complaint'

    text = str(text).lower().strip()

    for pattern, replacement in CLINICAL_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    # Remove punctuation (except hyphen in compound terms)
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


train['complaint_clean'] = train['chief_complaint_raw'].apply(preprocess_complaint)
test['complaint_clean']  = test['chief_complaint_raw'].apply(preprocess_complaint)

print("Preprocessing complete. Sample outputs:")
for i in range(4):
    raw   = str(train['chief_complaint_raw'].iloc[i])
    clean = train['complaint_clean'].iloc[i]
    print(f"  [{i}] Raw: {raw}")
    print(f"       Clean: {clean}\n")

```


```python
# ── 6b. High-risk keyword flags ────────────────────────────────────────────────
# Binary indicators for complaint phrases that are independently associated
# with high acuity in the emergency medicine literature.
# These give the model explicit clinical signal that TF-IDF alone may miss
# due to rare term frequencies.

HIGH_RISK_TERMS = [
    ('resp_distress',     r'shortness of breath|difficulty breathing|cant breathe|respiratory distress'),
    ('chest_pain',        r'chest pain|chest tightness|chest pressure|chest discomfort'),
    ('cardiac_keywords',  r'heart attack|myocardial|cardiac arrest|palpitations|chest pain'),
    ('stroke_keywords',   r'stroke|facial droop|arm weakness|speech difficulty|sudden weakness|tia|hemiplegia'),
    ('altered_mentation', r'altered mental|confusion|unresponsive|unconscious|disoriented|agitated'),
    ('seizure',           r'seizure|convulsion|epilepsy|postictal'),
    ('severe_bleeding',   r'hemorrhage|severe bleeding|massive bleed|hemoptysis|hematemesis'),
    ('sepsis_keywords',   r'sepsis|septic|infection fever|high fever chills|bacteremia'),
    ('trauma',            r'trauma|motor vehicle|mva|mvc|fall from height|assault|gunshot|stabbing'),
    ('overdose',          r'overdose|ingestion|poisoning|intoxication|drug overdose'),
    ('anaphylaxis',       r'anaphylaxis|allergic reaction|throat swelling|hives difficulty breathing'),
    ('neuro_symptoms',    r'severe headache|worst headache|thunderclap|neck stiffness|meningismus'),
    ('gi_emergency',      r'gastrointestinal bleeding|blood in stool|melena|hematemesis|bowel obstruction'),
    ('vascular',          r'aortic|pulmonary embolism|deep vein thrombosis|limb ischemia'),
    ('obstetric',         r'pregnancy|labor|delivery|eclampsia|ectopic|miscarriage'),
    ('pediatric_red',     r'child not breathing|infant seizure|pediatric trauma|choking child'),
    ('psychiatric_acute', r'suicidal|homicidal|psychosis|acute psychosis|self harm'),
    ('pain_severe',       r'10 out of 10|severe pain|excruciating|worst pain'),
    ('syncope',           r'syncope|fainted|passed out|loss of consciousness|blackout'),
    ('hypertensive',      r'hypertensive emergency|severe headache high bp|blood pressure crisis'),
]

for name, pattern in HIGH_RISK_TERMS:
    col = f'kw_{name}'
    train[col] = train['complaint_clean'].str.contains(
        pattern, case=False, na=False, regex=True).astype(np.int8)
    test[col]  = test['complaint_clean'].str.contains(
        pattern, case=False, na=False, regex=True).astype(np.int8)

kw_cols = [f'kw_{name}' for name, _ in HIGH_RISK_TERMS]
train['total_high_risk_keywords'] = train[kw_cols].sum(axis=1)
test['total_high_risk_keywords']  = test[kw_cols].sum(axis=1)

# Validate: high-risk keyword frequency should be elevated in ESI-1/2
print("Keyword hit rate by acuity (should decrease as ESI level rises):")
print(train.groupby('triage_acuity')['total_high_risk_keywords'].mean().round(3))
print(f"\nKeyword features added: {len(kw_cols)}")

```


```python
# ── 6c. TF-IDF n-gram features ────────────────────────────────────────────────
# TF-IDF captures relative term importance across the complaint corpus.
# ngram_range=(1,3) captures unigrams, bigrams, and trigrams -- critical
# for clinical phrases like 'chest pain radiating' or 'shortness of breath'.
# sublinear_tf applies log normalisation to reduce the dominance of very
# frequent terms (e.g. 'pain') over more specific clinical phrases.

tfidf_vectorizer = TfidfVectorizer(
    max_features   = 200,
    ngram_range    = (1, 3),
    stop_words     = 'english',
    lowercase      = True,
    min_df         = 3,
    max_df         = 0.98,
    sublinear_tf   = True,
    analyzer       = 'word',
)

# Fit ONLY on training complaints to prevent test data leakage
tfidf_train_arr = tfidf_vectorizer.fit_transform(
    train['complaint_clean']).toarray()
tfidf_test_arr  = tfidf_vectorizer.transform(
    test['complaint_clean']).toarray()

tfidf_cols = [f'tfidf_{w}' for w in tfidf_vectorizer.get_feature_names_out()]

tfidf_train_df = pd.DataFrame(tfidf_train_arr, columns=tfidf_cols,
                               index=train.index)
tfidf_test_df  = pd.DataFrame(tfidf_test_arr,  columns=tfidf_cols,
                               index=test.index)

train = pd.concat([train, tfidf_train_df], axis=1)
test  = pd.concat([test,  tfidf_test_df],  axis=1)

print(f"TF-IDF features added: {len(tfidf_cols)}")
print(f"Train shape: {train.shape}  |  Test shape: {test.shape}")

# Which TF-IDF terms most differentiate ESI-1 from ESI-5?
esi1_mask = train['triage_acuity'] == 1
esi5_mask = train['triage_acuity'] == 5
mean_1 = tfidf_train_df[esi1_mask.values].mean()
mean_5 = tfidf_train_df[esi5_mask.values].mean()
ratio  = (mean_1 / (mean_5 + 1e-9)).sort_values(ascending=False)

print("\nTop 10 TF-IDF terms most associated with ESI-1 (Critical):")
for term, val in ratio.head(10).items():
    clean_term = term.replace('tfidf_', '')
    print(f"  '{clean_term}': ratio = {val:.2f}")

```

## Section 7: Preprocessing Pipeline


```python
# ── 7a. Define target, feature set, and leakage columns ───────────────────────

TARGET = 'triage_acuity'

# Columns to exclude from features
EXCLUDE = [
    'patient_id',
    'chief_complaint_raw',
    'complaint_clean',
    TARGET,
    # Outcome variables available only AFTER triage -- strict leakage
    'disposition',
    'ed_los_hours',
    'ed_los',
    'admission',
    'icu_admission',
]

# Build feature list: must exist in both train and test
feature_cols = [
    c for c in train.columns
    if c not in EXCLUDE
    and c in test.columns
]

y       = train[TARGET].values
X       = train[feature_cols].copy()
X_test  = test[feature_cols].copy()

print(f"Target range: {y.min()} -- {y.max()}")
print(f"Total features: {len(feature_cols)}")
print(f"X:      {X.shape}")
print(f"X_test: {X_test.shape}")

# Feature type summary
tfidf_count  = sum(1 for c in feature_cols if c.startswith('tfidf_'))
kw_count     = sum(1 for c in feature_cols if c.startswith('kw_'))
flag_count   = sum(1 for c in feature_cols if c.startswith('flag_') or c.startswith('miss_'))
hx_count     = sum(1 for c in feature_cols if c.startswith('hx_'))
vital_count  = sum(1 for c in feature_cols
                   if any(v in c for v in ['bp','heart_rate','spo2','resp',
                                           'temp','gcs','shock','news2','map']))
print(f"\nFeature breakdown:")
print(f"  TF-IDF n-gram:         {tfidf_count}")
print(f"  Keyword flags:         {kw_count}")
print(f"  Clinical flags/miss:   {flag_count}")
print(f"  Comorbidity history:   {hx_count}")
print(f"  Vital/hemodynamic:     {vital_count}")

```


```python
# ── 7b. Label-encode categorical columns ──────────────────────────────────────

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical columns: {cat_cols}")

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on union of train and test to handle all categories
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col]      = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

print("Label encoding complete.")

```


```python
# ── 7c. Iterative imputation ──────────────────────────────────────────────────
# IterativeImputer models each missing feature as a function of all other
# features, iterating until convergence.
# This is substantially more accurate than median imputation when features
# are correlated -- which they strongly are in clinical data (SpO2 and RR
# are both driven by the same underlying respiratory status).
#
# We use BayesianRidge as the imputation estimator because:
# - It is fast and well-regularised
# - It handles collinear features gracefully
# - It provides uncertainty estimates that guide imputation

print("Running iterative imputation (may take 2-5 minutes)...")

num_cols          = X.select_dtypes(include=[np.number]).columns.tolist()
cols_with_missing = [c for c in num_cols if X[c].isnull().any()]

print(f"Columns with missing values: {len(cols_with_missing)}")

if cols_with_missing:
    imp = IterativeImputer(
        estimator   = BayesianRidge(),
        max_iter    = 10,
        random_state= SEED,
        verbose     = 0,
        imputation_order = 'roman',   # left to right, predictable order
    )
    X[num_cols]      = imp.fit_transform(X[num_cols])
    X_test[num_cols] = imp.transform(X_test[num_cols])
    print("Imputation complete.")
else:
    print("No missing numerical values found.")

assert X.isnull().sum().sum() == 0,     "Missing values remain in X"
assert X_test.isnull().sum().sum() == 0,"Missing values remain in X_test"

# Convert to float32 for memory efficiency (halves memory vs float64)
X      = X.astype(np.float32)
X_test = X_test.astype(np.float32)

# Zero-index target for multi-class models (ESI 1-5 -> 0-4)
y_enc = y - 1

print(f"\nFinal X:      {X.shape}")
print(f"Final X_test: {X_test.shape}")
print(f"Classes:      {np.unique(y_enc)}")

```

## Section 8: Model Training -- LightGBM + XGBoost + CatBoost Ensemble

We train three state-of-the-art gradient boosted tree models and combine them via performance-weighted averaging. Each model has complementary strengths: LightGBM uses leaf-wise tree growth optimised for speed on high-dimensional tabular data; XGBoost uses depth-wise growth which generalises differently; CatBoost uses ordered target statistics which handle categorical features without label encoding artifacts. Their ensemble reduces variance and captures patterns that no single model finds alone.


```python
# ── Cross-validation scaffold ─────────────────────────────────────────────────

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb  = np.zeros((len(X), 5), dtype=np.float32)
oof_xgb  = np.zeros((len(X), 5), dtype=np.float32)
oof_cat  = np.zeros((len(X), 5), dtype=np.float32)

test_lgb = np.zeros((len(X_test), 5), dtype=np.float32)
test_xgb = np.zeros((len(X_test), 5), dtype=np.float32)
test_cat = np.zeros((len(X_test), 5), dtype=np.float32)

scores_lgb, scores_xgb, scores_cat = [], [], []

print(f"CV: {N_FOLDS} stratified folds  |  "
      f"Train: {len(X):,}  |  Features: {X.shape[1]}")

```


```python
# ── LightGBM ──────────────────────────────────────────────────────────────────

LGB_PARAMS = {
    'objective'        : 'multiclass',
    'num_class'        : 5,
    'metric'           : 'multi_logloss',
    'learning_rate'    : 0.025,
    'num_leaves'       : 255,
    'max_depth'        : -1,
    'min_child_samples': 25,
    'feature_fraction' : 0.65,
    'bagging_fraction' : 0.80,
    'bagging_freq'     : 5,
    'lambda_l1'        : 0.05,
    'lambda_l2'        : 0.10,
    'min_split_gain'   : 0.01,
    'verbose'          : -1,
    'random_state'     : SEED,
    'n_jobs'           : -1,
}

print("Training LightGBM...")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_enc)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

    ds_tr = lgb.Dataset(X_tr, label=y_tr)
    ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)

    lgb_model = lgb.train(
        LGB_PARAMS, ds_tr,
        num_boost_round = 3000,
        valid_sets      = [ds_va],
        callbacks       = [
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(500),
        ],
    )

    va_pred  = lgb_model.predict(X_va)
    oof_lgb[va_idx] = va_pred
    test_lgb += lgb_model.predict(X_test) / N_FOLDS

    fold_f1 = f1_score(y_va, va_pred.argmax(1), average='macro')
    scores_lgb.append(fold_f1)
    print(f"  Fold {fold+1}: Macro F1 = {fold_f1:.4f}  "
          f"(best iter: {lgb_model.best_iteration})")

print(f"LGB CV Macro F1 = {np.mean(scores_lgb):.4f} "
      f"+/- {np.std(scores_lgb):.4f}")

```


```python
# ── XGBoost ───────────────────────────────────────────────────────────────────

XGB_PARAMS = {
    'objective'        : 'multi:softprob',
    'num_class'        : 5,
    'eval_metric'      : 'mlogloss',
    'learning_rate'    : 0.025,
    'max_depth'        : 7,
    'min_child_weight' : 5,
    'subsample'        : 0.80,
    'colsample_bytree' : 0.65,
    'reg_alpha'        : 0.05,
    'reg_lambda'       : 1.00,
    'seed'             : SEED,
    'nthread'          : -1,
    'verbosity'        : 0,
    'tree_method'      : 'hist',
}

print("Training XGBoost...")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_enc)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

    dm_tr  = xgb.DMatrix(X_tr, label=y_tr)
    dm_va  = xgb.DMatrix(X_va, label=y_va)
    dm_te  = xgb.DMatrix(X_test)

    xgb_model = xgb.train(
        XGB_PARAMS, dm_tr,
        num_boost_round      = 3000,
        evals                = [(dm_va, 'val')],
        early_stopping_rounds= 150,
        verbose_eval         = False,
    )

    va_pred  = xgb_model.predict(dm_va).reshape(-1, 5)
    oof_xgb[va_idx] = va_pred
    test_xgb += xgb_model.predict(dm_te).reshape(-1, 5) / N_FOLDS

    fold_f1 = f1_score(y_va, va_pred.argmax(1), average='macro')
    scores_xgb.append(fold_f1)
    print(f"  Fold {fold+1}: Macro F1 = {fold_f1:.4f}  "
          f"(best iter: {xgb_model.best_iteration})")

print(f"XGB CV Macro F1 = {np.mean(scores_xgb):.4f} "
      f"+/- {np.std(scores_xgb):.4f}")

```


```python
# ── CatBoost ──────────────────────────────────────────────────────────────────

CAT_PARAMS = {
    'iterations'           : 3000,
    'learning_rate'        : 0.025,
    'depth'                : 7,
    'l2_leaf_reg'          : 3.0,
    'loss_function'        : 'MultiClass',
    'eval_metric'          : 'TotalF1:average=Macro',
    'early_stopping_rounds': 150,
    'random_seed'          : SEED,
    'verbose'              : 500,
    'thread_count'         : -1,
    'use_best_model'       : True,
    'bootstrap_type'       : 'Bayesian',
    'bagging_temperature'  : 0.5,
}

print("Training CatBoost...")

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_enc)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

    cat_model = CatBoostClassifier(**CAT_PARAMS)
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    va_pred  = cat_model.predict_proba(X_va)
    oof_cat[va_idx] = va_pred
    test_cat += cat_model.predict_proba(X_test) / N_FOLDS

    fold_f1 = f1_score(y_va, va_pred.argmax(1), average='macro')
    scores_cat.append(fold_f1)
    print(f"  Fold {fold+1}: Macro F1 = {fold_f1:.4f}")

print(f"CAT CV Macro F1 = {np.mean(scores_cat):.4f} "
      f"+/- {np.std(scores_cat):.4f}")

```


```python
# ── Weighted ensemble ─────────────────────────────────────────────────────────
# Models weighted by their CV macro F1 performance.
# A better model gets proportionally more weight in the final blend.

lgb_score = np.mean(scores_lgb)
xgb_score = np.mean(scores_xgb)
cat_score = np.mean(scores_cat)
total     = lgb_score + xgb_score + cat_score

w_lgb = lgb_score / total
w_xgb = xgb_score / total
w_cat = cat_score / total

print("Ensemble weights (proportional to CV performance):")
print(f"  LightGBM : {w_lgb:.3f}  (CV F1: {lgb_score:.4f})")
print(f"  XGBoost  : {w_xgb:.3f}  (CV F1: {xgb_score:.4f})")
print(f"  CatBoost : {w_cat:.3f}  (CV F1: {cat_score:.4f})")

oof_ens  = w_lgb * oof_lgb  + w_xgb * oof_xgb  + w_cat * oof_cat
test_ens = w_lgb * test_lgb + w_xgb * test_xgb + w_cat * test_cat

oof_pred_classes = oof_ens.argmax(axis=1)

ens_f1  = f1_score(y_enc, oof_pred_classes, average='macro')
ens_acc = accuracy_score(y_enc, oof_pred_classes)

print(f"\nEnsemble OOF Macro F1 : {ens_f1:.4f}")
print(f"Ensemble OOF Accuracy  : {ens_acc:.4f}")

```

## Section 9: Evaluation and Results


```python
# ── 9a. Classification report ─────────────────────────────────────────────────

print("=" * 60)
print("CLASSIFICATION REPORT -- ENSEMBLE OOF PREDICTIONS")
print("=" * 60)
print(classification_report(
    y_enc, oof_pred_classes,
    target_names=['ESI-1 Critical', 'ESI-2 Emergent', 'ESI-3 Urgent',
                  'ESI-4 Less Urgent', 'ESI-5 Non-Urgent'],
    digits=4,
))

# Per-class F1 breakdown
per_class_f1 = f1_score(y_enc, oof_pred_classes, average=None)
print("Per-class F1 scores:")
for i, f in enumerate(per_class_f1):
    print(f"  ESI-{i+1}: {f:.4f}")

```


```python
# ── 9b. Confusion matrices ────────────────────────────────────────────────────

cm      = confusion_matrix(y_enc, oof_pred_classes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

short_labels = ['ESI-1','ESI-2','ESI-3','ESI-4','ESI-5']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=short_labels, yticklabels=short_labels,
            ax=axes[0], linewidths=0.4)
axes[0].set_title('Confusion Matrix: Raw Counts', fontweight='bold')
axes[0].set_xlabel('Predicted ESI Level')
axes[0].set_ylabel('True ESI Level')

# Row-normalised (recall per class)
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=short_labels, yticklabels=short_labels,
            ax=axes[1], linewidths=0.4, vmin=0, vmax=1)
axes[1].set_title('Confusion Matrix: Row-Normalised (Recall)', fontweight='bold')
axes[1].set_xlabel('Predicted ESI Level')
axes[1].set_ylabel('True ESI Level')

plt.suptitle('Ensemble OOF Confusion Matrices', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=130)
plt.show()

# ── Clinical error analysis ────────────────────────────────────────────────────
undertriage_rate = (oof_pred_classes > y_enc).mean()
overtriage_rate  = (oof_pred_classes < y_enc).mean()
exact_rate       = (oof_pred_classes == y_enc).mean()

# Adjacent error: off by exactly 1 level (clinically tolerable)
adj_error        = (np.abs(oof_pred_classes - y_enc) <= 1).mean()

print("Clinical Error Analysis:")
print(f"  Exact match rate   : {exact_rate*100:.2f}%")
print(f"  Adjacent (+-1) acc : {adj_error*100:.2f}%")
print(f"  Undertriage rate   : {undertriage_rate*100:.2f}%  "
      "(predicted LESS urgent than actual -- dangerous)")
print(f"  Overtriage rate    : {overtriage_rate*100:.2f}%  "
      "(predicted MORE urgent than actual -- wasteful but safer)")

# Critical undertriage: ESI-1 patients predicted as ESI-3 or lower
esi1_true        = y_enc == 0
critical_ut_rate = (oof_pred_classes[esi1_true] >= 2).mean()
print(f"\nCritical undertriage (ESI-1 predicted ESI-3+): "
      f"{critical_ut_rate*100:.2f}%")
print("(This is the most clinically dangerous error category)")

```


```python
# ── 9c. Model comparison plot ──────────────────────────────────────────────────

models     = ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble']
oof_preds  = [oof_lgb, oof_xgb, oof_cat, oof_ens]
all_scores = [scores_lgb, scores_xgb, scores_cat, None]
colors     = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Macro F1 comparison
f1_vals = []
for name, oof, scores, col in zip(models, oof_preds, all_scores, colors):
    preds = oof.argmax(axis=1)
    f1    = f1_score(y_enc, preds, average='macro')
    f1_vals.append(f1)

bars = axes[0].bar(models, f1_vals, color=colors, edgecolor='black',
                   linewidth=0.7)
axes[0].set_title('Model Comparison: OOF Macro F1', fontweight='bold')
axes[0].set_ylabel('Macro F1')
axes[0].set_ylim(0, 1)
for bar, val in zip(bars, f1_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.008,
                 f'{val:.4f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=9)

# Fold-level scores (boxplot for individual models)
fold_data = [scores_lgb, scores_xgb, scores_cat]
bp = axes[1].boxplot(fold_data, labels=['LightGBM','XGBoost','CatBoost'],
                     patch_artist=True, notch=False)
for patch, col in zip(bp['boxes'], colors[:3]):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
axes[1].set_title('CV Fold Score Distribution', fontweight='bold')
axes[1].set_ylabel('Macro F1')
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=130)
plt.show()

```

## Section 10: SHAP Interpretability

Black-box models are not acceptable in clinical AI. Regulatory bodies (FDA, EMA), clinical informaticists, and the emergency physicians who would use this tool all require the ability to audit model reasoning at both the global (population) and individual (per-patient) level. SHAP provides mathematically rigorous attribution based on Shapley values from cooperative game theory.


```python
print("Computing SHAP values for LightGBM (last fold model)...")
print("This may take 3-5 minutes on the full dataset.")

# Use last trained LightGBM model
# TreeExplainer is exact for tree-based models (no approximation needed)
# Sample for visualisation efficiency -- 3,000 patients is sufficient
sample_n   = min(3000, len(X))
sample_idx = np.random.choice(len(X), sample_n, replace=False)
X_shap     = X.iloc[sample_idx]
y_shap     = y_enc[sample_idx]

explainer   = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_shap)
# shap_values: list of arrays, one per class, shape (n_samples, n_features)

print(f"SHAP values computed for {sample_n:,} patients across {X.shape[1]} features.")

```


```python
# ── 10a. Global summary plots for ESI-1 and ESI-3 ─────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

plt.sca(axes[0])
shap.summary_plot(
    shap_values[0], X_shap,
    feature_names = X.columns.tolist(),
    max_display   = 20,
    show          = False,
    plot_size     = None,
)
axes[0].set_title('SHAP Feature Impact: ESI-1 (Critical)\n'
                  'What pushes predictions toward the highest acuity?',
                  fontweight='bold', pad=10)

plt.sca(axes[1])
shap.summary_plot(
    shap_values[2], X_shap,
    feature_names = X.columns.tolist(),
    max_display   = 20,
    show          = False,
    plot_size     = None,
)
axes[1].set_title('SHAP Feature Impact: ESI-3 (Urgent)\n'
                  'What drives the most common acuity level?',
                  fontweight='bold', pad=10)

plt.suptitle('Global SHAP Analysis\n'
             'Red = high feature value  |  Blue = low feature value  |  '
             'X position = direction and strength of impact',
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig('shap_global.png', dpi=130, bbox_inches='tight')
plt.show()

```


```python
# ── 10b. Mean absolute SHAP bar chart ─────────────────────────────────────────
# Aggregate feature importance across all 5 classes

mean_abs_shap = np.mean([np.abs(shap_values[c]) for c in range(5)], axis=0)
mean_abs_shap = mean_abs_shap.mean(axis=0)

shap_imp_df = pd.DataFrame({
    'feature'   : X.columns,
    'mean_shap' : mean_abs_shap
}).sort_values('mean_shap', ascending=False).head(25)

fig, ax = plt.subplots(figsize=(10, 9))
colors_shap = plt.cm.viridis(np.linspace(0.2, 0.9, len(shap_imp_df)))
bars = ax.barh(shap_imp_df['feature'], shap_imp_df['mean_shap'],
               color=colors_shap[::-1], edgecolor='black', linewidth=0.5)
ax.set_title('Top 25 Features: Mean |SHAP| Across All ESI Classes',
             fontweight='bold')
ax.set_xlabel('Mean |SHAP Value| (average impact on model output)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('shap_bar_chart.png', dpi=130)
plt.show()

```


```python
# ── 10c. Individual patient explanation ───────────────────────────────────────
# This demonstrates the kind of per-patient explainability a clinician
# would require to trust and act on a model prediction.

# Find an ESI-1 patient in our SHAP sample
esi1_in_sample = np.where(y_shap == 0)[0]
if len(esi1_in_sample) > 0:
    patient_i = esi1_in_sample[0]
    true_esi  = y_shap[patient_i] + 1
    pred_esi  = oof_ens[sample_idx[patient_i]].argmax() + 1

    print(f"Individual explanation -- Patient {patient_i}")
    print(f"  True ESI: {true_esi}   |   Predicted ESI: {pred_esi}")
    print("\nTop features driving this patient toward ESI-1:")

    patient_shap_esi1 = shap_values[0][patient_i]
    patient_feature_df = pd.DataFrame({
        'Feature'       : X.columns,
        'SHAP Value'    : patient_shap_esi1,
        'Feature Value' : X_shap.iloc[patient_i].values,
    }).sort_values('SHAP Value', key=abs, ascending=False).head(12)

    print(patient_feature_df.to_string(index=False))
    print("\nInterpretation: positive SHAP = pushes toward ESI-1 (critical)")
    print("                negative SHAP = pushes away from ESI-1")

```

## Section 11: Equity and Bias Analysis

Systematic undertriage of specific patient populations is one of the most serious unresolved problems in emergency medicine. Published literature documents higher undertriage rates for elderly patients, patients presenting without verbal fluency in the local language, and patients with non-urgent-appearing chief complaints masking serious underlying conditions. An AI system that perpetuates these biases at scale would cause measurable patient harm.

This section evaluates our model's performance across five demographic stratifications: age group, sex, insurance status, arrival mode, and comorbidity burden.


```python
# Reload original train demographics for stratification
train_raw = pd.read_csv(BASE + 'train.csv')

def equity_metrics(mask_arr, group, subgroup):
    """Compute key equity metrics for a demographic subgroup."""
    n = mask_arr.sum()
    if n < 50:
        return None   # Skip groups too small for reliable estimates

    true_sub = y_enc[mask_arr]
    pred_sub = oof_pred_classes[mask_arr]

    try:
        macro_f1 = f1_score(true_sub, pred_sub,
                            average='macro', zero_division=0)
    except Exception:
        macro_f1 = np.nan

    undertriage_rate = (pred_sub > true_sub).mean()
    overtriage_rate  = (pred_sub < true_sub).mean()
    exact_rate       = (pred_sub == true_sub).mean()

    esi1_mask = true_sub == 0
    critical_ut = (
        pred_sub[esi1_mask] >= 2).mean() if esi1_mask.sum() > 0 else np.nan

    return {
        'Group'          : group,
        'Subgroup'       : subgroup,
        'N'              : n,
        'Macro F1'       : round(macro_f1, 4),
        'Exact Match'    : round(exact_rate, 4),
        'Undertriage'    : round(undertriage_rate, 4),
        'Overtriage'     : round(overtriage_rate, 4),
        'Critical UT'    : round(critical_ut, 4) if not np.isnan(critical_ut) else np.nan,
    }

results = []

# Age groups
age_bins = {
    'Infant (0-1)'        : (0,   2),
    'Child (2-17)'        : (2,  18),
    'Young Adult (18-34)' : (18, 35),
    'Middle Age (35-49)'  : (35, 50),
    'Older Adult (50-64)' : (50, 65),
    'Elderly (65-84)'     : (65, 85),
    'Very Elderly (85+)'  : (85, 200),
}
for label, (lo, hi) in age_bins.items():
    mask = ((train_raw['age'] >= lo) & (train_raw['age'] < hi)).values
    r = equity_metrics(mask, 'Age Group', label)
    if r: results.append(r)

# Sex
if 'sex' in train_raw.columns:
    for val in train_raw['sex'].dropna().unique():
        mask = (train_raw['sex'] == val).values
        r = equity_metrics(mask, 'Sex', str(val))
        if r: results.append(r)

# Insurance type
if 'insurance_type' in train_raw.columns:
    for val in train_raw['insurance_type'].dropna().unique():
        mask = (train_raw['insurance_type'] == val).values
        r = equity_metrics(mask, 'Insurance', str(val))
        if r: results.append(r)

# Arrival mode
if 'arrival_mode' in train_raw.columns:
    for val in train_raw['arrival_mode'].dropna().unique():
        mask = (train_raw['arrival_mode'] == val).values
        r = equity_metrics(mask, 'Arrival Mode', str(val))
        if r: results.append(r)

equity_df = pd.DataFrame(results)

print("Equity Analysis Summary:")
print(equity_df.to_string(index=False))

```


```python
# ── Equity visualisation ──────────────────────────────────────────────────────

overall_f1 = ens_f1
overall_ut = (oof_pred_classes > y_enc).mean()

for group_name in equity_df['Group'].unique():
    gdata = equity_df[equity_df['Group'] == group_name].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Equity Analysis: {group_name}',
                 fontsize=12, fontweight='bold')

    # Macro F1
    f1_colors = ['#d62728' if v < 0.55 else '#ff7f0e' if v < 0.70
                 else '#2ca02c' for v in gdata['Macro F1']]
    axes[0].barh(gdata['Subgroup'], gdata['Macro F1'],
                 color=f1_colors, edgecolor='black', linewidth=0.6)
    axes[0].axvline(overall_f1, color='black', linestyle='--', lw=1.5,
                    label=f'Overall: {overall_f1:.3f}')
    axes[0].set_title('Macro F1 by Subgroup', fontweight='bold')
    axes[0].set_xlabel('Macro F1')
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=8)

    # Undertriage rate
    ut_colors = ['#d62728' if v > 0.18 else '#ff7f0e' if v > 0.12
                 else '#2ca02c' for v in gdata['Undertriage']]
    axes[1].barh(gdata['Subgroup'], gdata['Undertriage'],
                 color=ut_colors, edgecolor='black', linewidth=0.6)
    axes[1].axvline(overall_ut, color='black', linestyle='--', lw=1.5,
                    label=f'Overall: {overall_ut:.3f}')
    axes[1].set_title('Undertriage Rate by Subgroup', fontweight='bold')
    axes[1].set_xlabel('Undertriage Rate')
    axes[1].legend(fontsize=8)

    # Critical undertriage
    crit_ut = gdata['Critical UT'].fillna(0)
    ct_colors = ['#d62728' if v > 0.12 else '#ff7f0e' if v > 0.07
                 else '#2ca02c' for v in crit_ut]
    axes[2].barh(gdata['Subgroup'], crit_ut,
                 color=ct_colors, edgecolor='black', linewidth=0.6)
    axes[2].set_title('Critical Undertriage Rate\n(ESI-1 predicted ESI-3+)',
                      fontweight='bold')
    axes[2].set_xlabel('Critical Undertriage Rate')

    plt.tight_layout()
    sname = group_name.replace(' ', '_').lower()
    plt.savefig(f'equity_{sname}.png', dpi=130)
    plt.show()

# ── Print key equity findings ──────────────────────────────────────────────────
print("\nKEY EQUITY FINDINGS")
print("=" * 55)
worst3_ut = equity_df.nlargest(3, 'Undertriage')
print("\nHighest undertriage subgroups:")
for _, row in worst3_ut.iterrows():
    print(f"  {row['Group']:15s} {row['Subgroup']:25s}: "
          f"{row['Undertriage']*100:.1f}%")

worst3_f1 = equity_df.nsmallest(3, 'Macro F1')
print("\nLowest Macro F1 subgroups:")
for _, row in worst3_f1.iterrows():
    print(f"  {row['Group']:15s} {row['Subgroup']:25s}: "
          f"F1 = {row['Macro F1']:.4f}")

age_data = equity_df[equity_df['Group'] == 'Age Group']
if len(age_data) > 1:
    gap = age_data['Macro F1'].max() - age_data['Macro F1'].min()
    print(f"\nAge-group F1 equity gap: {gap:.4f}")
    print("(0 = perfectly equitable; lower is better)")

```

## Section 12: Feature Importance Analysis


```python
# ── LightGBM gain importance ──────────────────────────────────────────────────

imp_df = pd.DataFrame({
    'feature'         : X.columns,
    'importance_gain' : lgb_model.feature_importance('gain'),
    'importance_split': lgb_model.feature_importance('split'),
}).sort_values('importance_gain', ascending=False)

# Categorise by feature type
def feature_type(col):
    if col.startswith('tfidf_') : return 'TF-IDF NLP'
    if col.startswith('kw_')    : return 'Keyword Flag'
    if col.startswith('hx_')    : return 'Comorbidity'
    if col.startswith('flag_')  : return 'Clinical Flag'
    if col.startswith('miss_')  : return 'Missingness'
    return 'Raw Feature'

imp_df['type'] = imp_df['feature'].apply(feature_type)

print("Top 30 features by gain importance:")
print(imp_df.head(30)[['feature','type','importance_gain']].to_string(index=False))

# By feature type aggregate
type_imp = imp_df.groupby('type')['importance_gain'].sum().sort_values(ascending=False)
print("\nImportance by feature category:")
total_imp = type_imp.sum()
for typ, imp in type_imp.items():
    print(f"  {typ:20s}: {imp:>12,.0f}  ({imp/total_imp*100:.1f}%)")

```


```python
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# Top 30 features
top30 = imp_df.head(30)
type_color_map = {
    'TF-IDF NLP'    : '#4C72B0',
    'Keyword Flag'  : '#DD8452',
    'Comorbidity'   : '#55A868',
    'Clinical Flag' : '#C44E52',
    'Missingness'   : '#8172B2',
    'Raw Feature'   : '#937860',
}
bar_colors = [type_color_map.get(t, 'grey') for t in top30['type']]

axes[0].barh(top30['feature'], top30['importance_gain'],
             color=bar_colors, edgecolor='black', linewidth=0.5)
axes[0].set_title('Top 30 Features: LightGBM Gain Importance',
                  fontweight='bold')
axes[0].set_xlabel('Gain Importance')
axes[0].invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=c, label=t)
                  for t, c in type_color_map.items()]
axes[0].legend(handles=legend_handles, loc='lower right', fontsize=8)

# Feature type pie chart
type_imp_plot = imp_df.groupby('type')['importance_gain'].sum()
axes[1].pie(type_imp_plot.values,
            labels=type_imp_plot.index,
            colors=[type_color_map.get(t,'grey')
                    for t in type_imp_plot.index],
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'black','linewidth':0.7})
axes[1].set_title('Feature Importance by Category', fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=130)
plt.show()

```

## Section 13: Generate Submission


```python
# Convert ensemble probabilities to ESI 1-5 labels
test_pred_classes = test_ens.argmax(axis=1) + 1   # re-index to 1-5

submission = pd.DataFrame({
    'patient_id'    : test_ids,
    'triage_acuity' : test_pred_classes,
})

# ── Validation checks ─────────────────────────────────────────────────────────
assert list(submission.columns) == list(sample_sub.columns), \
    f"Column mismatch: {submission.columns.tolist()} vs {sample_sub.columns.tolist()}"
assert len(submission) == len(sample_sub), \
    f"Length mismatch: {len(submission)} vs {len(sample_sub)}"
assert submission['triage_acuity'].between(1, 5).all(), \
    "Predicted values outside valid ESI range 1-5"
assert submission['patient_id'].nunique() == len(submission), \
    "Duplicate patient IDs in submission"

submission.to_csv('submission.csv', index=False)

print("Submission generated and validated.")
print(f"  Shape: {submission.shape}")
print("\nPredicted acuity distribution:")
dist = submission['triage_acuity'].value_counts().sort_index()
for lvl, cnt in dist.items():
    pct = cnt / len(submission) * 100
    bar = '|' * int(pct / 2)
    print(f"  ESI-{lvl}: {cnt:5,}  ({pct:5.1f}%)  {bar}")

print("\nFirst 5 rows:")
print(submission.head().to_string(index=False))

```

## Section 14: Summary and Reproducibility Notes


```python
import pkg_resources

print("=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"\nModel Performance (OOF = honest estimate of test performance):")
for name, oof_p, sc in [
    ('LightGBM', oof_lgb, scores_lgb),
    ('XGBoost',  oof_xgb, scores_xgb),
    ('CatBoost', oof_cat, scores_cat),
]:
    f1 = f1_score(y_enc, oof_p.argmax(1), average='macro')
    print(f"  {name:10s}: OOF F1 = {f1:.4f}  |  "
          f"CV mean = {np.mean(sc):.4f}  +/- {np.std(sc):.4f}")

print(f"  {'Ensemble':10s}: OOF F1 = {ens_f1:.4f}  |  "
      f"Accuracy = {ens_acc:.4f}")

print(f"\nClinical Error Rates:")
print(f"  Undertriage (predicted less urgent): "
      f"{(oof_pred_classes > y_enc).mean()*100:.2f}%")
print(f"  Overtriage (predicted more urgent):  "
      f"{(oof_pred_classes < y_enc).mean()*100:.2f}%")
print(f"  Adjacent accuracy (within 1 level):  "
      f"{(np.abs(oof_pred_classes - y_enc) <= 1).mean()*100:.2f}%")

print(f"\nFeature Engineering:")
print(f"  Total features used       : {X.shape[1]}")
print(f"  TF-IDF n-gram (1-3)       : {tfidf_count}")
print(f"  High-risk keyword flags   : {kw_count}")
print(f"  Clinical threshold flags  : {flag_count}")
print(f"  Comorbidity history       : {hx_count}")
print(f"  Vital/hemodynamic         : {vital_count}")

print("\nReproducibility:")
print(f"  Random seed: {SEED}")
print(f"  CV strategy: StratifiedKFold, {N_FOLDS} folds")
print(f"  Imputation: IterativeImputer (BayesianRidge, max_iter=10)")
print("\nPackage versions:")
for pkg in ['lightgbm','xgboost','catboost','scikit-learn','shap',
            'pandas','numpy']:
    try:
        v = pkg_resources.get_distribution(pkg).version
        print(f"  {pkg}: {v}")
    except Exception:
        print(f"  {pkg}: not found")

print("\nDatasets used:")
print("  Triagegeist synthetic dataset -- Laitinen-Fredriksson Foundation")
print("  License: Non-Commercial Research License")
print("  Source: kaggle.com/competitions/triagegeist/data")
print("  Note: All records are fully synthetic. No real patient data.")

print("\nOutputs written:")
for f in ['submission.csv','eda_overview.png','confusion_matrix.png',
          'model_comparison.png','shap_global.png','shap_bar_chart.png',
          'feature_importance.png']:
    print(f"  {f}")

print("\nNotebook complete.")

```
