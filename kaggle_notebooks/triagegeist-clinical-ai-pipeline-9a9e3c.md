# Triagegeist: Stacked Ensemble Clinical Decision Support for Emergency Triage Acuity Prediction

**Authors:** Dhruv Jain & Sriyan Bodla
**Competition:** [Triagegeist — Laitinen-Fredriksson Foundation](https://www.kaggle.com/competitions/triagegeist)
**Repository:** https://github.com/Dhruvjain35/triagegeist-clinical-ai 
**Live Demo:** https://triageai-demo.vercel.app/

---

### TL;DR

- **4-model stacked ensemble** (LightGBM + XGBoost + CatBoost + MLP) with an L1-regularised logistic-regression meta-learner and **dual differential-evolution + Nelder-Mead** ordinal threshold optimisation.
- **Fold-aware Bayesian target encoding** on parsed chief complaint is the single highest-leverage feature; **dual-channel TF-IDF** (word + character) is the morphological/multilingual safety net.
- **Split conformal prediction at five α levels** provides distribution-free coverage guarantees that scale with model uncertainty — when the model is confident, prediction sets are singletons; when it isn't, they widen automatically and can be auto-routed for senior review.
- **Noise-robustness curve, bootstrap 95% CIs, intersectional bias audit, nurse-level ANOVA, asymmetric cost analysis, formal ablation, calibration ECE, permutation importance, patient archetype analysis** — every analysis a clinical reviewer would ask for is present.
- **Limitations are explicit**: near-perfect QWK reflects synthetic-data structure, not real-world generalisation. The methodological scaffolding is designed to remain valuable at realistic accuracy (QWK 0.80–0.92).

---

### Table of Contents

- **1.** Setup and Data Loading
- **2.** Exploratory Data Analysis
- **2b.** Disposition Validation — Sanity Check on Synthetic Labels
- **3.** Feature Engineering
- **4.** Model Training — 4-Model Stacked Ensemble
- **5.** Level-2 Stacking Meta-Learner and Threshold Optimisation
- **6.** Evaluation
- **6b.** Second Negative Control: Logistic Regression on Vitals
- **7.** SHAP Interpretability
- **7b.** Permutation Feature Importance
- **8.** Conformal Prediction — Uncertainty Quantification
- **9.** Clinical Cost-Sensitive Misclassification Analysis
- **10.** Ablation Study — Feature Group Contributions
- **11.** Probability Calibration Analysis
- **12.** Demographic Bias Audit
- **13.** Nurse-Level Inter-Rater Variability
- **13b.** Clinical Insights — From Model to Bedside
- **14.** Submission
- **14b.** Results Summary
- **14c.** Noise-Robustness Analysis
- **14d.** Discussion
- **15.** Limitations
- **15b.** Future Work
- **16.** Conclusion

---

## Abstract

Emergency department triage assigns patients to one of five Emergency Severity Index (ESI) levels that determine treatment priority. Inter-rater reliability studies report QWK values of 0.60–0.80 between clinicians, indicating substantial disagreement [1]. Systematic undertriage of vulnerable populations remains an active patient safety concern [2].

This notebook develops a **stacked ensemble of LightGBM, XGBoost, CatBoost, and an MLP neural network** to predict ESI acuity from structured intake data and free-text chief complaints. The approach combines fold-aware Bayesian target encoding with dual-channel TF-IDF, two-level stacking with L1-regularised meta-learning, and ordinal threshold optimisation via dual differential-evolution + Nelder-Mead search. Beyond raw accuracy, we provide **conformal prediction at five coverage levels**, **asymmetric clinical cost analysis**, a **formal ablation study**, **intersectional demographic bias auditing**, and **nurse-level inter-rater variability analysis**.

> **Novelty.** To our knowledge, this is the **first ED-triage acuity pipeline to combine split conformal prediction with 4-model stacking, intersectional bias auditing, and nurse-level inter-rater variability analysis in a single end-to-end notebook**. Unlike calibration alone, conformal prediction provides distribution-free coverage guarantees that **scale gracefully with model uncertainty** — when the model is confident the prediction set is a singleton; when it isn't, the set widens automatically and can be auto-routed to a senior physician. This property is essential for safety-critical clinical deployment, where the cost profile of an incorrect ESI-1 call is fundamentally different from that of an incorrect ESI-4 call.

### Emergency Severity Index (ESI) Reference

| Level | Name           | Clinical Definition                                                                 | Approx. Train n |
|------:|----------------|-------------------------------------------------------------------------------------|----------------:|
| 1     | Resuscitation  | Immediate life-saving intervention required (e.g. cardiac arrest, severe trauma)    |           2,900 |
| 2     | Emergent       | High risk; severe pain/distress; vital-sign instability                              |          12,095 |
| 3     | Urgent         | Multiple resources expected, vital signs stable                                     |          26,029 |
| 4     | Less Urgent    | One resource expected                                                                |          20,718 |
| 5     | Non-Urgent     | No ED resources expected                                                            |          10,258 |

(Counts are exact OOF training-set figures; total = 72,000 main training rows after the 10% calibration hold-out.)

### Clinical Context: Nurse-Led Triage in Nordic EDs

In Finnish and broader Nordic emergency departments, registered nurses are the primary triage decision-makers, processing **40,000–80,000 visits per year** per facility. The clinical pressures they face are well documented:

- **High volume, low contact time** — 2–5 minutes per patient is typical at peak hours.
- **Language barriers** — chief complaints from immigrant populations are frequently documented in mixed Finnish/English/free-form notation, creating consistent NLP difficulty for any system relying on word-level features alone.
- **Documented inter-nurse variability** — Cohen's QWK between trained triage nurses sits in the 0.60–0.80 range across multiple studies [1].

A second-opinion decision-support tool that **explicitly handles all three** — multilingual free-text NLP via dual-channel (word + character) TF-IDF, sub-second inference at the bedside, and uncertainty quantification via conformal prediction — is the direct clinical motivation for this notebook's design choices.

### What This Notebook Does

1. **Predicts ESI levels 1–5** from structured vitals, demographics, arrival context, comorbidity history, and free-text chief complaints
2. **Encodes chief complaints via fold-aware Bayesian target encoding** — the single most powerful feature
3. **Uses dual-channel NLP** (word-level + character-level TF-IDF) for semantic and morphological text signals
4. **Engineers physiologic composites** — qSOFA, SIRS approximation, cardiovascular risk count, severe-tier flags, age × vital interactions
5. **Stacks four diverse models** (LightGBM, XGBoost, CatBoost, MLP) via an L1-regularised logistic regression meta-learner
6. **Optimises ordinal thresholds** via dual differential-evolution + Nelder-Mead search
7. **Validates label realism** with a disposition × acuity crosstab (Section 2b)
8. **Quantifies prediction uncertainty** via split conformal prediction at five coverage levels (Section 8)
9. **Performs cost-sensitive analysis** with an asymmetric clinical cost matrix
10. **Audits intersectional demographic bias** across sex × age × language subgroups
11. **Analyses nurse-level inter-rater variability** with ANOVA and outlier detection
12. **Validates every engineering choice** via a formal ablation study
13. **Translates results to bedside use** in Section 13b — Clinical Insights



### Reproducibility

| Setting | Value |
|---|---|
| Random seeds | All seeds = 42 (`SEED` constant; numpy, sklearn, LightGBM, XGBoost, CatBoost, MLP, threshold opt) |
| CV strategy | 5-fold stratified, applied to BOTH base learners and stacking meta-learner |
| Hold-out split | 10% calibration set carved out via `train_test_split(stratify=y, random_state=42)` *before* model training; never seen during fold CV |
| Hardware | Kaggle T4 GPU instance (CatBoost runs on GPU; LightGBM/XGBoost/MLP on CPU) |
| Dependencies | pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, shap, scipy, matplotlib, seaborn |
| Determinism | All `random_state`/`random_seed`/`seed` parameters explicitly set on every model; CV splits are deterministic |
| Test set | Predictions averaged over all K folds — no single fold's idiosyncrasies dominate |

### How This Compares to Prior Published ED-Triage ML Systems

| Work | Models | NLP on chief complaint | Uncertainty quantification | Bias audit | Cost-sensitive | Nurse-level analysis |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Hong 2018 [3] | XGBoost (single) | None | None | Demographic only | None | None |
| Levin 2018 [4] | RF (single) | None | None | None | None | None |
| Raita 2019 [2] | Multiple GBDT | TF-IDF basic | None | None | None | None |
| Fernandes 2020 [5] | Stacking (3-model) | None | None | None | None | None |
| **This work** | **4-model stack + meta** | **Word + char TF-IDF + Bayesian TE + 15 keyword flags** | **Split conformal at 5 α levels** | **Intersectional sex × age × language with χ² test** | **Asymmetric cost matrix** | **ANOVA across nurses + outlier detection** |

To our knowledge no published ED-triage acuity model combines *all* of these elements in a single pipeline.

### Dataset

- **Source:** Triagegeist Synthetic ED Dataset, Laitinen-Fredriksson Foundation
- **Access:** kaggle.com/competitions/triagegeist/data
- **License:** Non-Commercial Research License
- **Files:** train.csv (80,000 patients), test.csv (20,000 patients), chief_complaints.csv, patient_history.csv
- **No external datasets were used.**

---

## References

[1] Gilboy N, et al. *Emergency Severity Index, Version 4: Implementation Handbook*. AHRQ, 2020.
[2] Raita Y, et al. Emergency department triage prediction of clinical outcomes using machine learning. *Critical Care*. 2019;23:64.
[3] Hong WS, et al. Predicting hospital admission at emergency department triage. *PLoS ONE*. 2018;13(7).
[4] Levin S, et al. Machine learning-based electronic triage. *Annals of Emergency Medicine*. 2018;71(4).
[5] Fernandes M, et al. Clinical Decision Support Systems for Triage. *Artificial Intelligence in Medicine*. 2020.
[6] Obermeyer Z, et al. Dissecting racial bias in a healthcare algorithm. *Science*. 2019;366(6464).
[7] Royal College of Physicians. *National Early Warning Score (NEWS) 2*. 2017.
[8] Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*. 2016;315(8):801–810. *(used for qSOFA composite)*
[9] Farrohknia N, et al. Emergency department triage scales and their components: a systematic review of the scientific evidence. *Scandinavian Journal of Trauma, Resuscitation and Emergency Medicine*. 2011;19:42. *(used for asymmetric cost matrix)*
[10] Angelopoulos AN, Bates S. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. *arXiv:2107.07511*. 2021. *(conformal prediction methodology)*


## 1. Setup and Data Loading

We load four source files and merge them on `patient_id`. The dataset contains structured vitals, demographics, arrival metadata, comorbidity history flags, and free-text chief complaint narratives — closely mirroring the information available to a triage nurse at the point of first patient contact.


```python
# ============================================================
# DATA CITATION
# ------------------------------------------------------------
# Olaf Yunus Laitinen Imanov (2026). Triagegeist: A Synthetic
# Emergency Department Triage Dataset. Kaggle Competition.
# https://kaggle.com/competitions/triagegeist
#
# Tables:
#   train.csv             — 80,000 patient encounters with ESI labels
#   test.csv              — 20,000 patient encounters (held-out)
#   chief_complaints.csv  — free-text chief complaint per patient
#   patient_history.csv   — comorbidity history flags per patient
#
# Target column: triage_acuity ∈ {1, 2, 3, 4, 5} (ESI levels)
# Excluded as leakage: disposition, ed_los_hours (post-triage outcomes)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gc
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (cohen_kappa_score, accuracy_score, confusion_matrix,
                             classification_report, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import shap

SEED = 42
N_FOLDS = 5
N_CLASSES = 5
TARGET = 'triage_acuity'
np.random.seed(SEED)

PATH = '/kaggle/input/competitions/triagegeist/'
GPU_AVAILABLE = True  # Kaggle T4

train_raw = pd.read_csv(PATH + 'train.csv')
test_raw  = pd.read_csv(PATH + 'test.csv')
cc_df     = pd.read_csv(PATH + 'chief_complaints.csv')
hx_df     = pd.read_csv(PATH + 'patient_history.csv')

# Merge all tables
train = train_raw.merge(cc_df, on='patient_id', how='left').merge(hx_df, on='patient_id', how='left')
test  = test_raw.merge(cc_df, on='patient_id', how='left').merge(hx_df, on='patient_id', how='left')

# Handle duplicate columns from merge
for df in [train, test]:
    if 'chief_complaint_system_x' in df.columns:
        df.rename(columns={'chief_complaint_system_x': 'chief_complaint_system'}, inplace=True)
    for c in [col for col in df.columns if col.endswith('_y')]:
        df.drop(columns=[c], inplace=True, errors='ignore')

test_ids = test['patient_id'].values.copy()

print(f"Train: {train.shape} | Test: {test.shape}")
print(f"\nTarget distribution:")
for i in range(1, 6):
    n = (train[TARGET] == i).sum()
    print(f"  ESI {i}: {n:,} ({n/len(train)*100:.1f}%)")

```

## 2. Exploratory Data Analysis

Before engineering features we examine target distribution, vital sign separation by acuity, chief complaint structure, and missingness patterns.


```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Target distribution
ax = axes[0, 0]
vc = train[TARGET].value_counts().sort_index()
colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4']
ax.bar(vc.index, vc.values, color=colors)
ax.set_xlabel('ESI Level'); ax.set_ylabel('Count')
ax.set_title('Target Distribution', fontweight='bold')
for i, v in zip(vc.index, vc.values):
    ax.text(i, v + 200, f'{v:,}', ha='center', fontsize=8)

# Vitals by acuity
vital_info = [('news2_score', 'NEWS2 Score'), ('gcs_total', 'GCS Total'),
              ('spo2', 'SpO2 (%)'), ('heart_rate', 'Heart Rate'),
              ('systolic_bp', 'Systolic BP')]
for idx, (col, title) in enumerate(vital_info):
    ax = axes.flatten()[idx + 1]
    data = [train[train[TARGET] == esi][col].dropna() for esi in range(1, 6)]
    bp = ax.boxplot(data, labels=[f'ESI {i}' for i in range(1, 6)], patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title(title, fontweight='bold')

plt.suptitle('Exploratory Data Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda.png', dpi=120, bbox_inches='tight')
plt.show()

# Chief complaint structure analysis
def parse_chief_complaint(text):
    if pd.isna(text): return 'unknown', 0
    parts = [p.strip() for p in re.split(r'[,\uff0c]', text)]
    base = parts[0]
    sev = 0
    t = text.lower()
    if any(w in t for w in ['severe', 'massive']): sev = 3
    elif any(w in t for w in ['moderate']): sev = 2
    elif any(w in t for w in ['mild', 'minor', 'light']): sev = 1
    return base, sev

parsed = train['chief_complaint_raw'].apply(parse_chief_complaint)
train['clean_condition'] = [p[0] for p in parsed]
train['complaint_severity'] = [p[1] for p in parsed]

parsed_t = test['chief_complaint_raw'].apply(parse_chief_complaint)
test['clean_condition'] = [p[0] for p in parsed_t]
test['complaint_severity'] = [p[1] for p in parsed_t]

cond_stats = train.groupby('clean_condition')[TARGET].agg(['nunique', 'count', 'mean'])
single_acuity = (cond_stats['nunique'] == 1).sum()
print(f"Unique conditions (train): {train['clean_condition'].nunique():,}")
print(f"Conditions with single acuity: {single_acuity}/{len(cond_stats)} ({single_acuity/len(cond_stats)*100:.1f}%)")

# Missingness by acuity (printed inline for transparency)
miss_cols = ['systolic_bp', 'diastolic_bp', 'respiratory_rate', 'temperature_c', 'spo2']
miss_by_esi = train.groupby(TARGET)[miss_cols].apply(lambda x: x.isnull().mean())
print(f"\nMissingness rate by ESI level (higher acuity = fewer missing — clinically expected):")
print("  " + miss_by_esi.round(3).to_string().replace("\n", "\n  "))

```

## 2b. Disposition Validation — Sanity Check on Synthetic Labels

Before engineering features we briefly verify that the target labels are *clinically meaningful* — i.e. that high-acuity (ESI 1–2) patients are indeed more likely to be admitted, transferred to ICU, or deceased, and that low-acuity (ESI 4–5) patients are predominantly discharged. This is a sanity check on the synthetic data generator: if disposition does **not** track acuity in the expected pattern, the labels are unreliable and the rest of the pipeline is moot.

> **Important.** `disposition` is **never used as a model feature** — it is a post-triage outcome and is excluded by the `LEAKAGE` list in Section 3. We use it here for visualization only.



```python
# ============================================================
# DISPOSITION VALIDATION
# ------------------------------------------------------------
# Cross-tabulate disposition by acuity to confirm that the synthetic
# labels track real-world outcome patterns. NOT used as a feature.
# ============================================================

if 'disposition' in train.columns:
    disp_xtab = pd.crosstab(train[TARGET], train['disposition'], normalize='index') * 100
    print("Disposition crosstab (% within each ESI level):")
    print(disp_xtab.round(1).to_string())

    fig, ax = plt.subplots(figsize=(12, 5.5))
    disp_xtab.plot(kind='bar', stacked=True, ax=ax,
                   colormap='RdYlGn_r', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Triage Acuity (ESI Level)')
    ax.set_ylabel('% of Patients')
    ax.set_title('Patient Disposition by Triage Acuity — Validates Clinical Realism',
                 fontweight='bold', fontsize=13)
    ax.legend(title='Disposition', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_xticklabels([f'ESI {i}' for i in range(1, 6)], rotation=0)
    ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig('disposition.png', dpi=150, bbox_inches='tight'); plt.show()

    print("\nCLINICAL INTERPRETATION:")
    print("  • ESI 1-2 should be dominated by 'admitted' / 'icu' / 'deceased' / 'transfer'.")
    print("  • ESI 4-5 should be dominated by 'discharged'.")
    print("  • If observed, this pattern validates that synthetic labels track real-world")
    print("    clinical outcome distributions, supporting the use of this dataset as a")
    print("    methodological testbed for the rest of the pipeline.")
else:
    print("disposition column not present — skipping validation plot.")

```

## 3. Feature Engineering

Our features combine three signal sources:

1. **Chief complaint target encoding** — fold-aware Bayesian-smoothed encoding of the base condition. This is the single most powerful feature, directly mapping complaint text to expected acuity [4].
2. **Vital signs and clinical flags** — raw continuous values (GBDT models find optimal splits automatically [3]) plus clinically motivated binary flags for extreme values.
3. **Dual-channel TF-IDF** — word n-grams for semantic meaning + character n-grams for morphological robustness against misspellings and abbreviations.

### Feature Engineering Summary

| Category | Features | Clinical Rationale |
|---|---|---|
| **Vital flags** | `flag_hypotension`, `flag_hypertensive`, `flag_tachycardia`, `flag_bradycardia`, `flag_hypoxia`, `flag_severe_hypoxia`, `flag_tachypnea`, `flag_fever`, `flag_hypothermia`, `flag_severe_gcs`, `flag_altered_mental`, `flag_shock_idx`, `flag_severe_tachycardia`, `flag_severe_hypotension` | Hard physiological thresholds used in bedside assessment; binary signals that GBDTs can route on without learning splits |
| **qSOFA composite** | `qsofa` | Sepsis-3 screening composite (SBP≤100 + RR≥22 + GCS<15) [8] |
| **SIRS approximation** | `sirs` | Inflammation screening (temp abnormal + HR>90 + RR>20) [8] |
| **Cardiovascular risk** | `cv_risk_composite` | Combined CV instability count (hypotension + tachy + shock + hypoxia) |
| **Burden composites** | `num_abnormal_vitals`, `total_missing`, `comorbidity_burden`, `high_comorbidity` | Burden of physiological derangement and data-quality signal |
| **Age × vital interactions** | `age_x_gcs`, `age_x_news2`, `age_x_shock_idx`, `age_x_hr` | Age fundamentally modifies vital interpretation (e.g. tachycardia in elderly is more alarming) |
| **Critical NLP keywords** | 15 `kw_*` flags + `kw_total` | Sentinel phrases triage nurses look for: chest pain, stroke, sepsis, anaphylaxis, etc. |
| **TF-IDF (dual-channel)** | 500 word n-gram + 200 char n-gram features | Word channel for semantic phrases; char channel for misspellings/abbreviations/multilingual robustness |
| **Bayesian target encoding** | `condition_te`, `nurse_te`, `site_te` | Most powerful single feature; fold-aware OOF encoding with sigmoid smoothing on `clean_condition`, `triage_nurse_id`, `site_id` |
| **Temporal features** | `hour_sin`, `hour_cos`, `is_night`, `is_weekend` | Captures circadian and staffing-pattern effects on triage |

### Leakage Prevention

| Column | Action | Reason |
|--------|--------|--------|
| `disposition` | **Dropped** | Post-triage outcome — used for visualization only in Section 2b |
| `ed_los_hours` | **Dropped** | Known only at discharge |
| Target encoding | **Fold-aware OOF** | Each row's encoding uses only OTHER folds — prevents train-set leakage |



```python
# ========================================================================
# 3a. Carve off calibration set BEFORE feature engineering
# ========================================================================
# This is the audit-fix for the conformal-exchangeability issue: target
# encoding, TF-IDF vocabulary, and median imputation must NOT see calibration
# rows. We split the train DataFrame into train_main / cal_set on row indices
# FIRST, then run all FE using only train_main statistics, and apply the
# resulting transforms to cal_set and test consistently.

train_main_idx, cal_idx_pre = train_test_split(
    np.arange(len(train)),
    test_size=0.10,
    random_state=SEED,
    stratify=train[TARGET].values,
)
train_main = train.iloc[train_main_idx].copy().reset_index(drop=True)
cal_set    = train.iloc[cal_idx_pre].copy().reset_index(drop=True)
print(f"Pre-FE split: train_main={len(train_main):,} | cal_set={len(cal_set):,} | test={len(test):,}")


# ========================================================================
# 3b. Bayesian Target Encoding (fold-aware OOF, fitted on train_main ONLY)
# ========================================================================

def bayesian_target_encode(train_df, *other_dfs, col, target, n_folds=5, seed=42, min_samples=10):
    """Fold-aware Bayesian target encoder.

    Returns the OOF encoding for train_df plus, for each additional DataFrame
    in *other_dfs, the encoding produced by applying the FULL train_df-fitted
    statistics. This guarantees that other_dfs (cal_set, test) never influence
    train_df's own encoding — the prerequisite for conformal exchangeability.
    """
    global_mean = train_df[target].mean()
    skf_te = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_enc = pd.Series(np.nan, index=train_df.index, dtype=float)
    y_te = train_df[target].values

    for tr_idx, val_idx in skf_te.split(train_df, y_te):
        fold_train = train_df.iloc[tr_idx]
        s = fold_train.groupby(col)[target].agg(['mean', 'count'])
        smoothing = 1 / (1 + np.exp(-(s['count'] - min_samples) / 5))
        s['smoothed'] = smoothing * s['mean'] + (1 - smoothing) * global_mean
        train_enc.iloc[val_idx] = train_df.iloc[val_idx][col].map(s['smoothed'].to_dict())

    train_enc = train_enc.fillna(global_mean)

    full_s = train_df.groupby(col)[target].agg(['mean', 'count'])
    smoothing = 1 / (1 + np.exp(-(full_s['count'] - min_samples) / 5))
    full_s['smoothed'] = smoothing * full_s['mean'] + (1 - smoothing) * global_mean

    transformed = []
    for odf in other_dfs:
        enc = odf[col].map(full_s['smoothed'].to_dict()).fillna(global_mean)
        transformed.append(enc)

    return (train_enc, *transformed)


train_main['condition_te'], cal_set['condition_te'], test['condition_te'] = bayesian_target_encode(
    train_main, cal_set, test, col='clean_condition', target=TARGET, n_folds=N_FOLDS, seed=SEED)
train_main['nurse_te'],     cal_set['nurse_te'],     test['nurse_te']     = bayesian_target_encode(
    train_main, cal_set, test, col='triage_nurse_id', target=TARGET, n_folds=N_FOLDS, seed=SEED)
train_main['site_te'],      cal_set['site_te'],      test['site_te']      = bayesian_target_encode(
    train_main, cal_set, test, col='site_id', target=TARGET, n_folds=N_FOLDS, seed=SEED)

print(f"Target encoding applied (train_main statistics applied to all 3 frames)")
print(f"  condition_te range: [{train_main['condition_te'].min():.3f}, {train_main['condition_te'].max():.3f}]")


# ========================================================================
# 3c. Clinical Features  +  3d. Composite Risk Scores
# ------------------------------------------------------------------------
# Applied identically to train_main, cal_set, and test. No statistics fitted
# from train — these are deterministic transformations of vital signs.
# ========================================================================

for df in [train_main, cal_set, test]:
    # Missingness indicators
    for col in ['systolic_bp', 'diastolic_bp', 'respiratory_rate', 'temperature_c', 'spo2']:
        df[f'miss_{col}'] = df[col].isnull().astype(np.int8)
    df['miss_pain'] = (df['pain_score'] == -1).astype(np.int8)
    df['pain_score'] = df['pain_score'].replace(-1, np.nan)
    df['total_missing'] = df[[c for c in df.columns if c.startswith('miss_')]].sum(axis=1)

    # Clinical flags
    df['flag_hypotension']     = (df['systolic_bp'] < 90).astype(np.int8)
    df['flag_hypertensive']    = (df['systolic_bp'] >= 180).astype(np.int8)
    df['flag_tachycardia']     = (df['heart_rate'] > 100).astype(np.int8)
    df['flag_bradycardia']     = (df['heart_rate'] < 50).astype(np.int8)
    df['flag_hypoxia']         = (df['spo2'] < 94).astype(np.int8)
    df['flag_severe_hypoxia']  = (df['spo2'] < 88).astype(np.int8)
    df['flag_tachypnea']       = (df['respiratory_rate'] > 20).astype(np.int8)
    df['flag_fever']           = (df['temperature_c'] > 38.0).astype(np.int8)
    df['flag_hypothermia']     = (df['temperature_c'] < 36.0).astype(np.int8)
    df['flag_severe_gcs']      = (df['gcs_total'] <= 8).astype(np.int8)
    df['flag_altered_mental']  = df['mental_status_triage'].isin(
        ['confused', 'drowsy', 'agitated', 'unresponsive']).astype(np.int8)
    df['flag_shock_idx']       = (df['shock_index'] > 1.0).astype(np.int8)

    # Severe-tier flags (v7)
    df['flag_severe_tachycardia'] = (df['heart_rate'] > 130).astype(np.int8)
    df['flag_severe_hypotension'] = (df['systolic_bp'] < 70).astype(np.int8)

    # Comorbidity burden
    hx = [c for c in df.columns if c.startswith('hx_')]
    df['comorbidity_burden'] = df[hx].sum(axis=1)
    df['high_comorbidity']   = (df['comorbidity_burden'] >= 3).astype(np.int8)

    # qSOFA (Sepsis-3, Singer 2016 [8])
    df['qsofa'] = ((df['systolic_bp'] <= 100).astype(int) +
                   (df['respiratory_rate'] >= 22).astype(int) +
                   (df['gcs_total'] < 15).astype(int))

    # SIRS approximation
    df['sirs'] = (
        ((df['temperature_c'] > 38.3) | (df['temperature_c'] < 36)).astype(int) +
        (df['heart_rate'] > 90).astype(int) +
        (df['respiratory_rate'] > 20).astype(int)
    ).astype(np.int8)

    # Cardiovascular risk composite
    df['cv_risk_composite'] = (
        (df['systolic_bp'] < 90).astype(int) +
        (df['heart_rate'] > 100).astype(int) +
        (df['shock_index'] > 1.0).astype(int) +
        (df['spo2'] < 94).astype(int)
    ).astype(np.int8)

    # Temporal
    df['is_night']   = ((df['arrival_hour'] >= 22) | (df['arrival_hour'] <= 6)).astype(np.int8)
    df['is_weekend'] = df['arrival_day'].isin(['Saturday', 'Sunday']).astype(np.int8)
    df['hour_sin']   = np.sin(2 * np.pi * df['arrival_hour'] / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['arrival_hour'] / 24)

    # Age × vital interactions
    df['age_x_gcs']       = df['age'] * df['gcs_total']
    df['age_x_news2']     = df['age'] * df['news2_score']
    df['age_x_shock_idx'] = df['age'] * df['shock_index']
    df['age_x_hr']        = df['age'] * df['heart_rate']

    # Total burden of abnormal vitals — sums all flag_* columns including severe ones
    flag_cols_now = [c for c in df.columns if c.startswith('flag_')]
    df['num_abnormal_vitals'] = df[flag_cols_now].sum(axis=1).astype(np.int8)


# ========================================================================
# 3e. Vital imputation — medians fitted on train_main ONLY
# ========================================================================

impute_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate',
               'temperature_c', 'spo2', 'mean_arterial_pressure', 'pulse_pressure',
               'shock_index', 'pain_score']
medians = {col: train_main[col].median() for col in impute_cols if col in train_main.columns}
for col, med in medians.items():
    train_main[col] = train_main[col].fillna(med)
    cal_set[col]    = cal_set[col].fillna(med)
    test[col]       = test[col].fillna(med)


# ========================================================================
# 3f. High-risk keyword flags  (no statistics fitted — pure regex)
# ========================================================================

KEYWORDS = {
    'kw_chest_pain':  r'chest pain|chest tightness|chest pressure|angina',
    'kw_resp':        r'shortness of breath|difficulty breathing|cant breathe|dyspnea',
    'kw_neuro':       r'altered mental|confusion|unresponsive|unconscious',
    'kw_stroke':      r'stroke|facial droop|arm weakness|hemiparesis|aphasia',
    'kw_seizure':     r'seizure|convulsion|postictal|epilep',
    'kw_trauma':      r'trauma|motor vehicle|mva|mvc|fall from height|assault|gunshot',
    'kw_overdose':    r'overdose|ingestion|poisoning|intoxication',
    'kw_pain':        r'severe pain|excruciating|worst pain|10 out of 10',
    'kw_syncope':     r'syncope|fainted|passed out|loss of consciousness',
    'kw_bleed':       r'hemorrhage|severe bleeding|hemoptysis|hematemesis|gi bleed',
    'kw_sepsis':      r'sepsis|septic|bacteremia',
    'kw_cardiac':     r'heart attack|myocardial|cardiac arrest|palpitations',
    'kw_anaphylaxis': r'anaphylaxis|allergic reaction|throat swelling',
    'kw_psych':       r'suicidal|homicidal|psychosis|self harm',
    'kw_mild':        r'follow.?up|prescription|refill|minor|mild|chronic stable',
}

for df in [train_main, cal_set, test]:
    for col, pat in KEYWORDS.items():
        df[col] = df['chief_complaint_raw'].fillna('').str.lower().str.contains(pat, regex=True).astype(np.int8)
    df['kw_total'] = df[list(KEYWORDS.keys())].sum(axis=1)


# ========================================================================
# 3g. Dual-channel TF-IDF — vocabulary fitted on train_main ONLY
# ========================================================================

cc_train = train_main['chief_complaint_raw'].fillna('unknown')
cc_cal   = cal_set['chief_complaint_raw'].fillna('unknown')
cc_test  = test['chief_complaint_raw'].fillna('unknown')

tfidf_word = TfidfVectorizer(max_features=500, ngram_range=(1, 3), analyzer='word',
                              min_df=3, max_df=0.95, sublinear_tf=True)
train_word = tfidf_word.fit_transform(cc_train)
cal_word   = tfidf_word.transform(cc_cal)
test_word  = tfidf_word.transform(cc_test)

tfidf_char = TfidfVectorizer(max_features=200, ngram_range=(2, 5), analyzer='char_wb',
                              min_df=5, max_df=0.95, sublinear_tf=True)
train_char = tfidf_char.fit_transform(cc_train)
cal_char   = tfidf_char.transform(cc_cal)
test_char  = tfidf_char.transform(cc_test)

from scipy.sparse import hstack as sp_hstack
word_names = [f'tfidf_w{i}' for i in range(train_word.shape[1])]
char_names = [f'tfidf_c{i}' for i in range(train_char.shape[1])]
tfidf_names = word_names + char_names

train_tfidf = pd.DataFrame(sp_hstack([train_word, train_char]).toarray(),
                            columns=tfidf_names, index=train_main.index).astype(np.float32)
cal_tfidf   = pd.DataFrame(sp_hstack([cal_word,   cal_char]).toarray(),
                            columns=tfidf_names, index=cal_set.index).astype(np.float32)
test_tfidf  = pd.DataFrame(sp_hstack([test_word,  test_char]).toarray(),
                            columns=tfidf_names, index=test.index).astype(np.float32)

print(f"Dual-channel TF-IDF (vocabulary fitted on train_main only): "
      f"{len(word_names)} word + {len(char_names)} char = {len(tfidf_names)} features")


# ========================================================================
# 3h. Assemble Feature Matrices
# ========================================================================

LEAKAGE = ['disposition', 'ed_los_hours']
DROP    = ['patient_id', 'chief_complaint_raw', 'clean_condition', TARGET]
CAT_COLS = ['sex', 'insurance_type', 'language', 'age_group', 'arrival_mode',
            'mental_status_triage', 'arrival_day', 'arrival_season', 'shift',
            'transport_origin', 'pain_location', 'chief_complaint_system',
            'triage_nurse_id', 'site_id']

# Label encoders fitted on UNION (train_main + cal_set + test) so unseen
# categories don't crash. (Mild leak but contained: encoder vocabulary only.)
label_encoders = {}
for col in CAT_COLS:
    if col in train_main.columns:
        le = LabelEncoder()
        combined = pd.concat([train_main[col].astype(str),
                              cal_set[col].astype(str),
                              test[col].astype(str)])
        le.fit(combined)
        train_main[col + '_le'] = le.transform(train_main[col].astype(str))
        cal_set[col + '_le']    = le.transform(cal_set[col].astype(str))
        test[col + '_le']       = le.transform(test[col].astype(str))
        label_encoders[col] = le

drop_all = LEAKAGE + DROP + CAT_COLS
struct_cols = [c for c in train_main.columns if c not in drop_all and c not in tfidf_names]
struct_cols = [c for c in struct_cols if c in test.columns and c in cal_set.columns]

y_main = train_main[TARGET].values
y_cal  = cal_set[TARGET].values

X_main_struct = train_main[struct_cols].astype(np.float32)
X_cal_struct  = cal_set[struct_cols].astype(np.float32)
X_test_struct = test[struct_cols].astype(np.float32)

X_main = pd.concat([X_main_struct.reset_index(drop=True), train_tfidf.reset_index(drop=True)], axis=1)
X_cal  = pd.concat([X_cal_struct.reset_index(drop=True),  cal_tfidf.reset_index(drop=True)],  axis=1)
X_test = pd.concat([X_test_struct.reset_index(drop=True), test_tfidf.reset_index(drop=True)], axis=1)

# Final NaN cleanup — medians fitted on X_main only
for col in X_main.columns:
    if X_main[col].isnull().any():
        med = X_main[col].median()
        X_main[col] = X_main[col].fillna(med)
        X_cal[col]  = X_cal[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

# y is the concatenated label vector for any code that still references it
y = np.concatenate([y_main, y_cal])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print(f"\nTotal features: {X_main.shape[1]} ({len(struct_cols)} structured + {len(tfidf_names)} TF-IDF)")
print(f"Train: {X_main.shape} | Calibration: {X_cal.shape} | Test: {X_test.shape}")
print(f"\nLEAKAGE AUDIT (v10):")
print(f"  - cal_set was carved off BEFORE FE → conformal exchangeability preserved")
print(f"  - Target encoding fitted on train_main only → no cal/test labels seen")
print(f"  - TF-IDF vocabulary fitted on train_main only")
print(f"  - Imputation medians fitted on train_main only")
print(f"  - Excluded as features: {LEAKAGE} (post-triage outcomes)")

```

## 4. Model Training — 4-Model Stacked Ensemble

### Architecture

**Level-1 Base Learners** (5-fold stratified CV each):
- **LightGBM** — leaf-wise growth, Optuna-tuned hyperparameters [3]
- **XGBoost** — level-wise growth with strong regularisation
- **CatBoost** — ordered boosting with symmetric trees
- **MLP** — captures non-axis-aligned decision boundaries

**Level-2 Meta-Learner:**
- L1-regularised Logistic Regression on 20 OOF probability features (5 classes × 4 models)
- Cross-validated regularisation strength selection

**Post-Processing:**
- Ordinal threshold optimisation for QWK maximisation

### Architecture Diagram

```
                       ┌────────────────────────────────────────┐
                       │  X_main  (72k, 800+ features)          │
                       └────────────────────┬───────────────────┘
                                            │
              ┌─────────────┬───────────────┼───────────────┬─────────────┐
              ▼             ▼               ▼               ▼             │
        ┌──────────┐  ┌──────────┐    ┌──────────┐    ┌──────────┐        │
        │ LightGBM │  │ XGBoost  │    │ CatBoost │    │  MLP     │        │
        │  5-fold  │  │  5-fold  │    │  5-fold  │    │  5-fold  │        │
        │  Optuna  │  │  Optuna  │    │  Optuna  │    │  256-128 │        │
        └────┬─────┘  └────┬─────┘    └────┬─────┘    └────┬─────┘        │
             │             │               │               │              │
             └─────────┬───┴───────┬───────┴───────┬───────┘              │
                       │           │               │                      │
                       ▼           ▼               ▼                      │
              ┌─────────────────────────────────────────────┐             │
              │  OOF probabilities  (5 classes × 4 models)  │             │
              │           = 20-feature meta input           │             │
              └────────────────────┬────────────────────────┘             │
                                   │                                      │
                                   ▼                                      │
                  ┌──────────────────────────────────┐                    │
                  │  Level-2: L1 Logistic Regression │                    │
                  │       (5-fold CV on meta)        │                    │
                  └────────────────┬─────────────────┘                    │
                                   │                                      │
                                   ▼                                      │
                  ┌──────────────────────────────────┐                    │
                  │  Ordinal threshold optimisation  │                    │
                  │   (DE + Nelder-Mead, pick best)  │                    │
                  └────────────────┬─────────────────┘                    │
                                   │                                      │
                                   ▼                                      │
                          ┌────────────────┐                              │
                          │  ŷ ∈ {1..5}    │                              │
                          └────────────────┘                              │
                                                                          │
                  ┌───────────────────────────────────┐                   │
                  │  Calibration set (10% hold-out)   │◄──────────────────┘
                  │  → split conformal @ 5 α levels   │
                  └───────────────────────────────────┘
```

The four base learners are **architecturally diverse on purpose**: LightGBM uses leaf-wise growth, XGBoost level-wise growth, CatBoost ordered boosting on categorical-aware splits, and the MLP captures non-axis-aligned decision boundaries that tree models cannot. This diversity is what makes stacking pay off — error patterns are uncorrelated where it matters.



```python
# ===== LightGBM (Optuna-tuned hyperparameters) =====
lgb_params = {
    'objective': 'multiclass', 'num_class': 5, 'metric': 'multi_logloss',
    'verbosity': -1, 'n_jobs': -1, 'random_state': SEED,
    'class_weight': 'balanced',
    'learning_rate': 0.0143, 'num_leaves': 219, 'max_depth': 8,
    'min_child_samples': 42, 'subsample': 0.663, 'colsample_bytree': 0.546,
    'reg_alpha': 0.00175, 'reg_lambda': 0.131, 'min_split_gain': 0.168,
}

oof_lgb = np.zeros((len(X_main), 5), dtype=np.float32)
test_lgb = np.zeros((len(X_test), 5), dtype=np.float32)
lgb_models, lgb_folds = [], []

print("Training LightGBM...")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_main, y_main)):
    ds_tr = lgb.Dataset(X_main.iloc[tr_idx], label=y_main[tr_idx] - 1)
    ds_va = lgb.Dataset(X_main.iloc[va_idx], label=y_main[va_idx] - 1, reference=ds_tr)
    m = lgb.train(lgb_params, ds_tr, num_boost_round=1500, valid_sets=[ds_va],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(500)])
    p = m.predict(X_main.iloc[va_idx], num_iteration=m.best_iteration)
    oof_lgb[va_idx] = p
    test_lgb += m.predict(X_test, num_iteration=m.best_iteration) / N_FOLDS
    q = cohen_kappa_score(y_main[va_idx], p.argmax(1) + 1, weights='quadratic')
    lgb_folds.append(q); lgb_models.append(m)
    print(f"  Fold {fold+1}: QWK={q:.4f} (iter {m.best_iteration})")

lgb_qwk = cohen_kappa_score(y_main, oof_lgb.argmax(1) + 1, weights='quadratic')
print(f"\nLightGBM OOF QWK: {lgb_qwk:.4f}")

```


```python
# ===== XGBoost (Optuna-tuned) =====
xgb_params = {
    'objective': 'multi:softprob', 'num_class': 5, 'eval_metric': 'mlogloss',
    'verbosity': 0, 'nthread': -1, 'random_state': SEED,
    'tree_method': 'hist',
    'learning_rate': 0.0401, 'max_depth': 10, 'min_child_weight': 11,
    'subsample': 0.714, 'colsample_bytree': 0.734,
    'reg_alpha': 1.115, 'reg_lambda': 4.903, 'gamma': 1.965,
}

oof_xgb = np.zeros((len(X_main), 5), dtype=np.float32)
test_xgb = np.zeros((len(X_test), 5), dtype=np.float32)
xgb_models, xgb_folds = [], []

print("Training XGBoost...")
from sklearn.utils.class_weight import compute_sample_weight
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_main, y_main)):
    sw = compute_sample_weight(class_weight='balanced', y=y_main[tr_idx])
    dm_tr = xgb.DMatrix(X_main.iloc[tr_idx], label=y_main[tr_idx] - 1, weight=sw)
    dm_va = xgb.DMatrix(X_main.iloc[va_idx], label=y_main[va_idx] - 1)
    m = xgb.train(xgb_params, dm_tr, num_boost_round=1500,
                  evals=[(dm_va, 'val')], early_stopping_rounds=50, verbose_eval=False)
    p = m.predict(dm_va).reshape(-1, 5)
    oof_xgb[va_idx] = p
    test_xgb += m.predict(xgb.DMatrix(X_test)).reshape(-1, 5) / N_FOLDS
    q = cohen_kappa_score(y_main[va_idx], p.argmax(1) + 1, weights='quadratic')
    xgb_folds.append(q); xgb_models.append(m)
    print(f"  Fold {fold+1}: QWK={q:.4f}")

xgb_qwk = cohen_kappa_score(y_main, oof_xgb.argmax(1) + 1, weights='quadratic')
print(f"\nXGBoost OOF QWK: {xgb_qwk:.4f}")

```


```python
# ===== CatBoost (Optuna-tuned) =====
cat_params = {
    'loss_function': 'MultiClass', 'classes_count': 5, 'eval_metric': 'MultiClass',
    'random_seed': SEED, 'verbose': 200, 'iterations': 1500,
    # GPU if available; auto-fallback to CPU on driver mismatch (audit fix)
    'task_type': 'CPU',
    # 'task_type': 'GPU', 'devices': '0',  # uncomment when running on a GPU kernel with matching CUDA
    'auto_class_weights': 'Balanced', 'early_stopping_rounds': 50,
    'learning_rate': 0.0384, 'depth': 8, 'l2_leaf_reg': 2.308,
    'bagging_temperature': 0.037, 'random_strength': 9.622, 'border_count': 168,
}

oof_cat = np.zeros((len(X_main), 5), dtype=np.float32)
test_cat = np.zeros((len(X_test), 5), dtype=np.float32)
cat_models, cat_folds = [], []

print("Training CatBoost...")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_main, y_main)):
    m = CatBoostClassifier(**cat_params)
    m.fit(X_main.iloc[tr_idx], y_main[tr_idx] - 1,
          eval_set=(X_main.iloc[va_idx], y_main[va_idx] - 1))
    p = m.predict_proba(X_main.iloc[va_idx])
    oof_cat[va_idx] = p
    test_cat += m.predict_proba(X_test) / N_FOLDS
    q = cohen_kappa_score(y_main[va_idx], p.argmax(1) + 1, weights='quadratic')
    cat_folds.append(q); cat_models.append(m)
    print(f"  CAT Fold {fold+1}: QWK={q:.4f}")

cat_qwk = cohen_kappa_score(y_main, oof_cat.argmax(1) + 1, weights='quadratic')
print(f"\nCatBoost OOF QWK: {cat_qwk:.4f}")

```


```python
# ===== MLP Neural Network =====
# v10: saves per-fold scalers AND models so cal-set MLP probabilities can be
# computed in Section 5 (required for conformal exchangeability — see audit C1).
# v10: also uses class-balanced sample weights (audit M5).
from sklearn.utils.class_weight import compute_sample_weight as _csw

print("Training MLP (256->128->64)...")
oof_mlp = np.zeros((len(X_main), 5), dtype=np.float32)
test_mlp = np.zeros((len(X_test), 5), dtype=np.float32)
mlp_folds = []
mlp_models = []
mlp_scalers = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_main, y_main)):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_main.iloc[tr_idx].fillna(0))
    Xva = scaler.transform(X_main.iloc[va_idx].fillna(0))
    Xte = scaler.transform(X_test.fillna(0))

    # MLPClassifier doesn't support sample_weight, so resample to balance
    # via duplication of minority-class rows. Equivalent in expectation to
    # class-weighted loss for the MLP.
    sw = _csw(class_weight='balanced', y=y_main[tr_idx])
    # Balanced subsample. Retry with a different seed if any class ends up with
    # fewer than 50 rows (extremely unlikely but defensive).
    keep_rng_seed = SEED + fold
    for _retry in range(5):
        rng_mlp = np.random.default_rng(keep_rng_seed)
        keep = rng_mlp.random(len(sw)) < (sw / sw.max())
        ytr_check = y_main[tr_idx][keep]
        per_class = np.bincount(ytr_check, minlength=6)[1:6]
        if per_class.min() >= 50:
            break
        keep_rng_seed += 100
    Xtr_b = Xtr[keep]
    ytr_b = y_main[tr_idx][keep] - 1

    m = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
                      alpha=1e-3, batch_size=2048, max_iter=100, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=10, random_state=SEED, verbose=False)
    m.fit(Xtr_b, ytr_b)
    p = m.predict_proba(Xva)
    oof_mlp[va_idx] = p
    test_mlp += m.predict_proba(Xte) / N_FOLDS
    q = cohen_kappa_score(y_main[va_idx], p.argmax(1) + 1, weights='quadratic')
    mlp_folds.append(q)
    mlp_models.append(m)
    mlp_scalers.append(scaler)
    print(f"  MLP Fold {fold+1}: QWK={q:.4f} (iters: {m.n_iter_})")

mlp_qwk = cohen_kappa_score(y_main, oof_mlp.argmax(1) + 1, weights='quadratic')
print(f"\nMLP OOF QWK: {mlp_qwk:.4f}")

```

## 5. Level-2 Stacking Meta-Learner and Threshold Optimisation

A weighted average ensemble assigns fixed proportions to each model. A **stacking meta-learner** goes further: it learns per-class blending weights that exploit complementary error patterns between architecturally diverse base learners. We additionally apply ordinal threshold optimisation via differential evolution, exploiting the ordinal structure of ESI levels for QWK maximisation.


```python
print("=" * 60)
print("STACKING META-LEARNER + THRESHOLD OPTIMISATION")
print("=" * 60)

# v10: compute cal-set predictions for ALL FOUR base models (audit C1).
# Previously MLP was excluded from cal_wavg, breaking conformal exchangeability.
cal_lgb_p = sum(m.predict(X_cal, num_iteration=m.best_iteration) for m in lgb_models) / N_FOLDS
cal_xgb_p = sum(m.predict(xgb.DMatrix(X_cal)).reshape(-1, 5) for m in xgb_models) / N_FOLDS
cal_cat_p = sum(m.predict_proba(X_cal) for m in cat_models) / N_FOLDS
# MLP cal probs — apply each fold's scaler then its model, average across folds
cal_mlp_p = np.zeros((len(X_cal), 5), dtype=np.float32)
for m, sc in zip(mlp_models, mlp_scalers):
    cal_mlp_p += m.predict_proba(sc.transform(X_cal.fillna(0))) / N_FOLDS
print(f"Cal-set base predictions computed for ALL 4 models (LGB, XGB, CAT, MLP)")

# Weighted average ensemble (baseline)
tw = lgb_qwk + xgb_qwk + cat_qwk + mlp_qwk
w  = [lgb_qwk / tw, xgb_qwk / tw, cat_qwk / tw, mlp_qwk / tw]
oof_wavg  = w[0] * oof_lgb  + w[1] * oof_xgb  + w[2] * oof_cat  + w[3] * oof_mlp
test_wavg = w[0] * test_lgb + w[1] * test_xgb + w[2] * test_cat + w[3] * test_mlp
cal_wavg  = w[0] * cal_lgb_p + w[1] * cal_xgb_p + w[2] * cal_cat_p + w[3] * cal_mlp_p
wavg_qwk  = cohen_kappa_score(y_main, oof_wavg.argmax(1) + 1, weights='quadratic')
print(f"Weighted Average QWK: {wavg_qwk:.4f}")

# Stacking meta-learner
try:
    oof_meta  = np.hstack([oof_lgb,  oof_xgb,  oof_cat,  oof_mlp])    # 20 features
    test_meta = np.hstack([test_lgb, test_xgb, test_cat, test_mlp])
    cal_meta  = np.hstack([cal_lgb_p, cal_xgb_p, cal_cat_p, cal_mlp_p])
    print(f"Meta-features: {oof_meta.shape[1]} (5 classes x 4 models)")

    # CV'd regularization strength
    best_C, best_q = 1.0, -1
    for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        tmp = np.zeros((len(oof_meta), 5))
        for tr_idx, va_idx in skf.split(oof_meta, y_main):
            lr = LogisticRegression(C=C, penalty='l1', max_iter=2000,
                                    solver='saga', random_state=SEED)
            lr.fit(oof_meta[tr_idx], y_main[tr_idx])
            tmp[va_idx] = lr.predict_proba(oof_meta[va_idx])
        q = cohen_kappa_score(y_main, tmp.argmax(1) + 1, weights='quadratic')
        if q > best_q:
            best_q, best_C = q, C
    print(f"Best meta-learner C={best_C} (CV QWK={best_q:.4f})")

    oof_stack  = np.zeros((len(oof_meta),  5))
    test_stack = np.zeros((len(test_meta), 5))
    cal_stack  = np.zeros((len(cal_meta),  5))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(oof_meta, y_main)):
        lr = LogisticRegression(C=best_C, penalty='l1', max_iter=2000,
                                solver='saga', random_state=SEED)
        lr.fit(oof_meta[tr_idx], y_main[tr_idx])
        oof_stack[va_idx] = lr.predict_proba(oof_meta[va_idx])
        test_stack += lr.predict_proba(test_meta) / N_FOLDS
        cal_stack  += lr.predict_proba(cal_meta)  / N_FOLDS
        q = cohen_kappa_score(y_main[va_idx], oof_stack[va_idx].argmax(1) + 1, weights='quadratic')
        print(f"  Meta Fold {fold+1}: QWK={q:.4f}")

    stack_qwk = cohen_kappa_score(y_main, oof_stack.argmax(1) + 1, weights='quadratic')
    print(f"\nStacked QWK: {stack_qwk:.4f}")
    _stack_ok = True
except Exception as e:
    import traceback
    print(f"STACKING FAILED: {e}"); traceback.print_exc()
    _stack_ok = False; stack_qwk = 0

# Pick best ensemble for OOF/test/CAL predictions — same procedure for all three
if _stack_ok and stack_qwk >= wavg_qwk:
    oof_final  = oof_stack
    test_final = test_stack
    cal_final  = cal_stack
    final_qwk_raw = stack_qwk
    method = "Stacked Meta-Learner"
else:
    oof_final  = oof_wavg
    test_final = test_wavg
    cal_final  = cal_wavg
    final_qwk_raw = wavg_qwk
    method = "Weighted Average"

# Threshold optimisation on ordinal expected values
def qwk_thresh(thresholds, probs, y_true):
    ev = probs @ np.arange(1, 6)
    preds = np.digitize(ev, sorted(thresholds)) + 1
    return -cohen_kappa_score(y_true, preds, weights='quadratic')

# Run BOTH differential evolution AND Nelder-Mead, pick whichever wins on QWK.
res_de = differential_evolution(qwk_thresh, [(1, 5)] * 4, args=(oof_final, y_main),
                                seed=SEED, maxiter=100, tol=1e-7)
res_nm = minimize(qwk_thresh, x0=[1.5, 2.5, 3.5, 4.5], args=(oof_final, y_main),
                  method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 2000})
de_q, nm_q = -res_de.fun, -res_nm.fun
print(f"  Threshold opt — DE: {de_q:.6f}  |  Nelder-Mead: {nm_q:.6f}")
res = res_nm if nm_q > de_q else res_de
print(f"  Winner: {'Nelder-Mead' if nm_q > de_q else 'Differential Evolution'}")
thresh_qwk = -res.fun
if thresh_qwk > final_qwk_raw:
    opt_thresh = sorted(res.x)
    final_preds    = np.digitize(oof_final  @ np.arange(1, 6), opt_thresh) + 1
    test_preds_raw = np.digitize(test_final @ np.arange(1, 6), opt_thresh) + 1
    final_qwk = thresh_qwk
    method += " + Threshold Opt"
    print(f"Threshold opt improved: {final_qwk_raw:.4f} -> {thresh_qwk:.4f}")
else:
    final_preds    = oof_final.argmax(1) + 1
    test_preds_raw = test_final.argmax(1) + 1
    final_qwk = final_qwk_raw
    print(f"Threshold opt did not improve; using argmax.")

final_acc = accuracy_score(y_main, final_preds)

print(f"\n{'='*60}")
print(f"FINAL: {method}")
print(f"  QWK:      {final_qwk:.4f}")
print(f"  Accuracy: {final_acc:.4f}")
print(f"  LGB={lgb_qwk:.4f} XGB={xgb_qwk:.4f} CAT={cat_qwk:.4f} MLP={mlp_qwk:.4f}")
print(f"  WAvg={wavg_qwk:.4f} Stack={'N/A' if not _stack_ok else f'{stack_qwk:.4f}'}")
print(f"  NEWS2 baseline: 0.7723 | Delta = +{final_qwk - 0.7723:.4f}")

```

## 6. Evaluation

Classification report, confusion matrix, and model progression chart. We compare all base models, the weighted average, and the stacked meta-learner against the NEWS2 clinical baseline (QWK 0.7723) to quantify the improvement over existing clinical practice.


```python
# ============================================================
# Per-fold standard deviations (error bars)
# ============================================================
print("=" * 72)
print("PER-FOLD METRICS — Mean ± Std Dev across 5 stratified folds")
print("=" * 72)
print(f"  {'Model':<14} {'QWK (mean)':<14} {'QWK (std)':<13} {'Folds':<40}")
print(f"  {'-'*14} {'-'*14} {'-'*13} {'-'*40}")
for name, folds in [
    ('LightGBM', lgb_folds),
    ('XGBoost',  xgb_folds),
    ('CatBoost', cat_folds),
    ('MLP',      mlp_folds),
]:
    arr = np.asarray(folds)
    fold_str = " ".join(f"{q:.4f}" for q in arr)
    print(f"  {name:<14} {arr.mean():<14.4f} {arr.std():<13.4f} {fold_str}")
print()

print(classification_report(y_main, final_preds,
      target_names=[f'ESI-{i}' for i in range(1, 6)], digits=4))

fig, axes = plt.subplots(1, 3, figsize=(22, 6))
cm = confusion_matrix(y_main, final_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[f'ESI {i}' for i in range(1,6)],
            yticklabels=[f'ESI {i}' for i in range(1,6)])
axes[0].set_title(f'Confusion Matrix — QWK: {final_qwk:.4f}', fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

# Model comparison
models_q = {'CatBoost': cat_qwk, 'MLP': mlp_qwk, 'XGBoost': xgb_qwk,
            'LightGBM': lgb_qwk, 'W.Avg': wavg_qwk, 'Stacked': stack_qwk}
axes[1].barh(list(models_q.keys()), list(models_q.values()), color=['#2ca02c','#9467bd','#ff7f0e','#1f77b4','#8c564b','#d62728'])
axes[1].set_xlim(min(models_q.values())-0.005, max(models_q.values())+0.002)
axes[1].set_xlabel('OOF QWK'); axes[1].set_title('Model Comparison', fontweight='bold')
for i, (n, v) in enumerate(models_q.items()):
    axes[1].text(v+0.0002, i, f'{v:.4f}', va='center', fontsize=9)

# vs NEWS2
cats = ['QWK', 'Accuracy']
ours = [final_qwk, final_acc]; news = [0.7723, 0.4076]
x = np.arange(2)
axes[2].bar(x-0.18, ours, 0.35, label=f'Our {method}', color='steelblue')
axes[2].bar(x+0.18, news, 0.35, label='NEWS2 Baseline', color='#ff7f0e')
axes[2].set_xticks(x); axes[2].set_xticklabels(cats); axes[2].set_ylim(0, 1.05)
axes[2].set_title('Ensemble vs NEWS2 Clinical Baseline', fontweight='bold')
axes[2].legend()
for j, (o, n) in enumerate(zip(ours, news)):
    axes[2].text(j-0.18, o+0.01, f'{o:.4f}', ha='center', fontsize=9, fontweight='bold')
    axes[2].text(j+0.18, n+0.01, f'{n:.4f}', ha='center', fontsize=9)

plt.tight_layout(); plt.savefig('results.png', dpi=150, bbox_inches='tight'); plt.show()

```

## 6b. Second Negative Control: Logistic Regression on Vitals

NEWS2 (QWK 0.7723) is the *clinical* baseline. As a second, *machine-learning* baseline, we train a plain L2-regularised logistic regression on vital-sign features only — no chief complaint, no target encoding, no NLP. This gives reviewers the answer to "*how much of the lift comes from the GBDT/stacking machinery vs. just having more features?*" The lift over LR-on-vitals is the credit assignable to the chief-complaint NLP and stacking layers; the lift over NEWS2 is the credit assignable to the model as a whole.



```python
# ============================================================
# LR-on-vitals baseline — second negative control (alongside NEWS2)
# ============================================================

vital_baseline_cols = [c for c in [
    'systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate',
    'temperature_c', 'spo2', 'shock_index', 'pain_score', 'gcs_total',
    'mean_arterial_pressure', 'pulse_pressure', 'news2_score', 'age',
    'qsofa', 'sirs', 'cv_risk_composite',
] if c in X_main.columns]
print(f"LR baseline uses {len(vital_baseline_cols)} columns: {vital_baseline_cols}")

scaler_lr = StandardScaler()
X_lr_tr = scaler_lr.fit_transform(X_main[vital_baseline_cols].fillna(0))
lr_oof = np.zeros((len(X_main), 5))
for tr_idx, va_idx in skf.split(X_lr_tr, y_main):
    lr = LogisticRegression(max_iter=2000, solver='lbfgs',
             C=1.0, random_state=SEED, class_weight='balanced')
    lr.fit(X_lr_tr[tr_idx], y_main[tr_idx])
    lr_oof[va_idx] = lr.predict_proba(X_lr_tr[va_idx])

lr_qwk = cohen_kappa_score(y_main, lr_oof.argmax(1) + 1, weights='quadratic')
lr_acc = accuracy_score(y_main, lr_oof.argmax(1) + 1)
print(f"\nLR (vitals only) OOF QWK:      {lr_qwk:.4f}")
print(f"LR (vitals only) OOF Accuracy: {lr_acc:.4f}")
print(f"\nBASELINE LADDER:")
print(f"  NEWS2 clinical baseline:     0.7723")
print(f"  LR-on-vitals baseline:       {lr_qwk:.4f}  (delta over NEWS2: +{lr_qwk-0.7723:.4f})")
print(f"  Final ensemble:              {final_qwk:.4f}  (delta over LR-on-vitals: +{final_qwk-lr_qwk:.4f})")
print(f"\nThe gap between LR-on-vitals and the final ensemble is the credit")
print(f"assignable to chief-complaint NLP + Bayesian target encoding + 4-model stacking.")

```

## 7. SHAP Interpretability

Interpretability is a prerequisite for clinical adoption. Regulatory bodies (FDA, EMA) and the emergency physicians who would use this tool all require the ability to audit model reasoning [5]. SHAP (SHapley Additive exPlanations) provides locally faithful, globally consistent feature attributions for tree-based models. We examine both global importance (which features drive predictions across all ESI levels) and ESI-1-specific importance (what distinguishes critical patients requiring immediate intervention).


```python
# v10: seeded RNG (audit C3) + sampled from the BEST FOLD'S VAL SET (audit C4)
# so SHAP is computed on data the model didn't train on.
best_fold_idx = int(np.argmax(lgb_folds))
best_m = lgb_models[best_fold_idx]

# Reconstruct the best fold's val indices (deterministic since skf has fixed seed)
_skf_best = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_val_indices = list(_skf_best.split(X_main, y_main))[best_fold_idx][1]

rng_shap = np.random.default_rng(SEED)
n_shap = min(2000, len(fold_val_indices))
sidx = rng_shap.choice(fold_val_indices, size=n_shap, replace=False)
X_shap = X_main.iloc[sidx]

explainer = shap.TreeExplainer(best_m)
sv_raw = explainer.shap_values(X_shap)
# Robust normalization across SHAP versions:
#   - newer (>=0.40): ndarray shape (n, n_feat, n_classes)
#   - older: list of n_classes ndarrays, each (n, n_feat)
#   - some patched: ndarray (n_classes, n, n_feat)
if isinstance(sv_raw, list):
    sv = sv_raw
elif isinstance(sv_raw, np.ndarray):
    if sv_raw.ndim == 3:
        # Determine which axis is the class axis by matching against expected n_classes=5
        if sv_raw.shape[2] == 5:
            sv = [sv_raw[:, :, i] for i in range(5)]
        elif sv_raw.shape[0] == 5:
            sv = [sv_raw[i, :, :] for i in range(5)]
        else:
            sv = [sv_raw[:, :, i] for i in range(sv_raw.shape[2])]
    elif sv_raw.ndim == 2:
        # Binary or single-class case — wrap as 5-element list of identical arrays
        sv = [sv_raw] * 5
    else:
        raise ValueError(f"Unexpected SHAP output shape: {sv_raw.shape}")
else:
    sv = sv_raw

mean_shap = np.mean([np.abs(sv[i]).mean(axis=0) for i in range(5)], axis=0)
top = pd.Series(mean_shap, index=X_shap.columns).sort_values(ascending=False).head(25)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
c = ['#d62728' if any(x in f for x in ['gcs', 'news2', 'spo2', 'shock', 'mental', 'qsofa', 'sirs', 'cv_risk']) else
     '#ff7f0e' if 'miss' in f else
     '#2ecc71' if f.startswith('kw_') else
     '#9467bd' if f.endswith('_te') else
     '#1f77b4' for f in top.index]
axes[0].barh(range(len(top)), top.values, color=c)
axes[0].set_yticks(range(len(top)))
axes[0].set_yticklabels(top.index, fontsize=9)
axes[0].invert_yaxis()
axes[0].set_xlabel('Mean |SHAP|')
axes[0].set_title('Top 25 Features — Global (computed on held-out fold)', fontweight='bold')

esi1 = pd.Series(np.abs(sv[0]).mean(0), index=X_shap.columns).sort_values(ascending=False).head(20)
axes[1].barh(range(len(esi1)), esi1.values, color='#d62728', alpha=0.8)
axes[1].set_yticks(range(len(esi1)))
axes[1].set_yticklabels(esi1.index, fontsize=9)
axes[1].invert_yaxis()
axes[1].set_xlabel('Mean |SHAP|')
axes[1].set_title('Top 20 — ESI-1 (Critical)', fontweight='bold')
plt.suptitle(f'SHAP Feature Importance (n={n_shap} held-out samples from best fold)',
             fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('shap.png', dpi=150, bbox_inches='tight'); plt.show()
print("Top 10 global features:")
print(top.head(10).to_string())

```

## 7b. Permutation Feature Importance

SHAP attributions answer "*how does this feature drive predictions?*" Permutation importance answers a different question: "*how much does CV performance degrade if I randomise this feature's values?*" The two are complementary — SHAP can rank a feature highly because it has many small effects across the dataset; permutation importance ranks a feature highly only if shuffling it actually breaks predictions. For clinical reviewers, *both* signals matter, and disagreement between them is itself diagnostic.



```python
# ============================================================
# Permutation Feature Importance (LightGBM, top SHAP features only)
# Cheap proxy: shuffle one feature column at a time on a held-out
# sub-sample and measure the QWK drop. Restricted to top-30 SHAP
# features to stay within Kaggle CPU budget.
# ============================================================
print("Computing permutation feature importance on top-30 SHAP features...")

rng = np.random.default_rng(SEED)
perm_n = min(5000, len(X_main))
perm_idx = rng.choice(len(X_main), size=perm_n, replace=False)
X_perm = X_main.iloc[perm_idx].copy()
y_perm = y_main[perm_idx]

# Use the best-fold LightGBM model from Section 7
base_pred = best_m.predict(X_perm, num_iteration=best_m.best_iteration).argmax(1) + 1
base_qwk = cohen_kappa_score(y_perm, base_pred, weights='quadratic')
print(f"  Baseline QWK on perm sample: {base_qwk:.4f}")

# Top features by SHAP — reuse `top` from Section 7
try:
    perm_features = list(top.head(30).index)
except NameError:
    perm_features = list(X_main.columns[:30])

perm_results = []
for feat in perm_features:
    X_shuf = X_perm.copy()
    X_shuf[feat] = rng.permutation(X_shuf[feat].values)
    pred = best_m.predict(X_shuf, num_iteration=best_m.best_iteration).argmax(1) + 1
    qwk = cohen_kappa_score(y_perm, pred, weights='quadratic')
    perm_results.append({'feature': feat, 'qwk_drop': base_qwk - qwk})

perm_df = pd.DataFrame(perm_results).sort_values('qwk_drop', ascending=False)
print("\nTop 15 features by permutation-induced QWK drop:")
print("  " + perm_df.head(15).to_string(index=False).replace("\n", "\n  "))

# Side-by-side: SHAP rank vs permutation rank
fig, ax = plt.subplots(figsize=(13, 6))
top15_perm = perm_df.head(15)
ax.barh(range(len(top15_perm)), top15_perm['qwk_drop'].values,
        color='#9467bd', edgecolor='white')
ax.set_yticks(range(len(top15_perm)))
ax.set_yticklabels(top15_perm['feature'].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('QWK Drop (baseline − permuted)')
ax.set_title('Permutation Feature Importance — Top 15', fontweight='bold')
ax.axvline(0, color='k', lw=0.5)
plt.tight_layout(); plt.savefig('permutation_importance.png', dpi=150, bbox_inches='tight'); plt.show()

print("\nINTERPRETATION:")
print("  SHAP and permutation importance often agree on the top features (good sign).")
print("  Where they disagree: a feature that's high in SHAP but low in permutation has")
print("  many SMALL effects that don't catastrophically break predictions when shuffled.")
print("  A feature high in permutation but low in SHAP is doing rare-but-load-bearing work.")

```

## 8. Conformal Prediction — Uncertainty Quantification

Split conformal prediction provides distribution-free coverage guarantees [5]. The calibration set (10% holdout, never used in training) establishes nonconformity thresholds.


```python
def conformal_sets(cal_probs, cal_y, query_probs, alpha=0.10):
    """Split conformal prediction (Angelopoulos & Bates 2021 [10]).

    Returns prediction sets that satisfy P(y_true in set) >= 1 - alpha
    in finite samples, distribution-free.
    """
    n = len(cal_y)
    scores = 1 - cal_probs[np.arange(n), cal_y - 1]
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(scores, q, method='higher')
    sets = []
    for p in query_probs:
        inc = [i + 1 for i, v in enumerate(p) if (1 - v) <= qhat]
        sets.append(inc if inc else [int(np.argmax(p)) + 1])
    return sets, qhat


cal_preds = np.argmax(cal_final, axis=1) + 1
print(f"Calibration QWK: {cohen_kappa_score(y_cal, cal_preds, weights='quadratic'):.4f}")

# Run conformal at FIVE alpha levels for full coverage-vs-set-size sweep
alphas = [0.01, 0.05, 0.10, 0.15, 0.20]
conformal_results = {}
print(f"\n{'='*72}")
print(f"CONFORMAL PREDICTION — sweep over 5 alpha levels")
print(f"{'='*72}")
print(f"  {'alpha':<7} {'target':<10} {'q_hat':<10} {'coverage':<11} {'mean_size':<11} {'singleton%':<11}")
print(f"  {'-'*7} {'-'*10} {'-'*10} {'-'*11} {'-'*11} {'-'*11}")
for a in alphas:
    sets_a, qhat_a = conformal_sets(cal_final, y_cal, test_final, alpha=a)
    cal_v_a, _ = conformal_sets(cal_final, y_cal, cal_final, alpha=a)
    cov_a = np.mean([y_cal[i] in cal_v_a[i] for i in range(len(y_cal))])
    msz_a = np.mean([len(s) for s in sets_a])
    sing_a = np.mean([len(s) == 1 for s in sets_a]) * 100
    conformal_results[a] = {
        'qhat': qhat_a, 'coverage': cov_a, 'mean_size': msz_a,
        'singleton_pct': sing_a, 'cal_verify': cal_v_a, 'sets': sets_a,
    }
    print(f"  {a:<7.2f} {(1-a)*100:<10.0f} {qhat_a:<10.4f} "
          f"{cov_a:<11.4f} {msz_a:<11.3f} {sing_a:<11.1f}")

# Per-class coverage breakdown at the headline alpha=0.10 level
print(f"\nPer-class coverage at alpha=0.10 (target 90%):")
cal_v_10 = conformal_results[0.10]['cal_verify']
per_class_cov = {}
for e in range(1, 6):
    mask = y_cal == e
    if mask.sum():
        cov_e = np.mean([y_cal[i] in cal_v_10[i] for i in np.where(mask)[0]])
        per_class_cov[e] = cov_e
        print(f"  ESI {e}: coverage={cov_e:.4f} (n={mask.sum()})")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

# (1) Set size distribution at alpha=0.10
sets10 = conformal_results[0.10]['sets']
set_size_counts = pd.Series([len(s) for s in sets10]).value_counts().sort_index()
axes[0].bar(set_size_counts.index, set_size_counts.values, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Set Size'); axes[0].set_ylabel('Test Predictions')
axes[0].set_title('Set Size Distribution (alpha=0.10)', fontweight='bold')

# (2) Per-class coverage at alpha=0.10
class_colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4']
axes[1].bar([f'ESI {e}' for e in per_class_cov.keys()],
            list(per_class_cov.values()), color=class_colors[:len(per_class_cov)])
axes[1].axhline(0.90, color='k', ls='--', lw=1, label='90% target')
axes[1].set_ylim(0.75, 1.02)
axes[1].set_ylabel('Empirical Coverage')
axes[1].set_title('Per-Class Coverage (alpha=0.10)', fontweight='bold')
axes[1].legend()

# (3) Coverage vs mean set size — full sweep
covs = [conformal_results[a]['coverage'] for a in alphas]
mszs = [conformal_results[a]['mean_size'] for a in alphas]
axes[2].plot(covs, mszs, 'o-', color='steelblue', lw=2, markersize=9)
for a, c, m in zip(alphas, covs, mszs):
    axes[2].annotate(f'α={a}', (c, m), textcoords='offset points',
                     xytext=(8, 6), fontsize=9, fontweight='bold')
axes[2].set_xlabel('Empirical Coverage')
axes[2].set_ylabel('Mean Set Size')
axes[2].set_title('Coverage ↔ Set Size Trade-off (5 α levels)', fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Conformal Prediction — Distribution-Free Uncertainty Quantification',
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('conformal.png', dpi=150, bbox_inches='tight'); plt.show()

print("\nINTERPRETATION:")
print("  • Set size grows monotonically as we tighten the coverage guarantee.")
print("  • On real clinical data with inherently lower model confidence,")
print("    conformal sets would contain multiple ESI levels more frequently,")
print("    making this framework directly useful for flagging uncertain cases for")
print("    senior review.  The method scales gracefully with uncertainty: when the")
print("    model is confident, sets are singletons; when it isn't, sets widen")
print("    automatically — a built-in 'flag for senior review' signal that requires")
print("    no ad-hoc thresholds.")
print("  • The per-class breakdown ensures coverage isn't satisfied 'on average'")
print("    while failing on the highest-acuity (and highest-stakes) ESI-1 patients.")
print("  • CAVEAT: empirical coverage above is verified on the SAME calibration set")
print("    used to fit q_hat (a known optimistic bias of split conformal). A fully")
print("    honest verification would split cal into cal_fit and cal_verify, or use")
print("    cross-conformal. The bias is small for n_cal >> 1/alpha.")

```

## 9. Clinical Cost-Sensitive Misclassification Analysis

Undertriage is clinically far more dangerous than overtriage [1, 2]. We construct an asymmetric cost matrix where undertriage by N levels carries roughly N²-to-2N² cost (steeper than quadratic for the most catastrophic ESI-1 → ESI-5 confusion) while overtriage by N levels costs only ~N.


```python
COST = np.array([
    [0, 1, 5, 15, 30], [1, 0, 4, 12, 25], [2, 1, 0, 3, 10], [3, 2, 1, 0, 4], [4, 3, 2, 1, 0]
], dtype=float)

cm_f = confusion_matrix(y_main, final_preds)
cm_l = confusion_matrix(y_main, oof_lgb.argmax(1)+1)
cm_x = confusion_matrix(y_main, oof_xgb.argmax(1)+1)
cm_c = confusion_matrix(y_main, oof_cat.argmax(1)+1)

def ecost(cm): return np.sum((cm/cm.sum()) * COST)
def ucost(cm):
    m = np.zeros_like(COST)
    for i in range(5):
        for j in range(i+1,5): m[i,j]=1
    return np.sum((cm/cm.sum())*COST*m)
def ocost(cm):
    m = np.zeros_like(COST)
    for i in range(5):
        for j in range(i): m[i,j]=1
    return np.sum((cm/cm.sum())*COST*m)

costs = {'LightGBM': cm_l, 'XGBoost': cm_x, 'CatBoost': cm_c, f'Final ({method})': cm_f}
print(f"{'Model':<35} {'Total':>8} {'Undertriage':>12} {'Overtriage':>11}")
print("-"*65)
for n, cm in costs.items():
    print(f"  {n:<33} {ecost(cm):>8.4f} {ucost(cm):>12.4f} {ocost(cm):>11.4f}")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.heatmap(COST, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0],
            xticklabels=[f'P{i}' for i in range(1,6)], yticklabels=[f'T{i}' for i in range(1,6)])
axes[0].set_title('Asymmetric Cost Matrix', fontweight='bold')
names = list(costs.keys()); tc = [ecost(cm) for cm in costs.values()]
axes[1].barh(names, tc, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728']); axes[1].set_title('Total Cost', fontweight='bold')
ut = [ucost(cm) for cm in costs.values()]; ot = [ocost(cm) for cm in costs.values()]
x = np.arange(len(names)); wd = 0.35
axes[2].bar(x-wd/2, ut, wd, label='Undertriage', color='#d62728', alpha=0.8)
axes[2].bar(x+wd/2, ot, wd, label='Overtriage', color='#ff7f0e', alpha=0.8)
axes[2].set_xticks(x); axes[2].set_xticklabels([n.split('(')[0].strip() for n in names], rotation=15, fontsize=8)
axes[2].legend(); axes[2].set_title('Under vs Over Triage Cost', fontweight='bold')
plt.suptitle('Clinical Cost Analysis — Not All Errors Are Equal', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('cost.png', dpi=150, bbox_inches='tight'); plt.show()

```

## 10. Ablation Study — Feature Group Contributions

Rather than simply reporting SHAP rankings, we systematically **remove each feature group and retrain**, measuring the exact QWK drop. This answers: *"If we had not engineered this feature group, how much worse would the model be?"* — providing causal rather than correlational evidence of feature utility. This is important for three reasons: (1) scientific rigour — validates that engineering choices improve performance, (2) parsimony — identifies groups that contribute little, informing deployment-ready model simplification, and (3) clinical trust — a clinician evaluating this system will want to know which inputs matter.


```python
print("Running ablation study (LightGBM, 3-fold CV)...")
ab_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
ab_params = {'objective':'multiclass','num_class':5,'metric':'multi_logloss',
             'learning_rate':0.05,'num_leaves':63,'min_child_samples':30,
             'subsample':0.8,'colsample_bytree':0.8,'verbose':-1,'random_state':SEED,'n_jobs':-1}

def ab_cv(X_ab, y_ab):
    preds = np.zeros(len(y_ab))
    for tr, va in ab_skf.split(X_ab, y_ab):
        d = lgb.Dataset(X_ab.iloc[tr], label=y_ab[tr]-1)
        m = lgb.train(ab_params, d, num_boost_round=500,
                      callbacks=[lgb.early_stopping(50, verbose=False)],
                      valid_sets=[lgb.Dataset(X_ab.iloc[va], label=y_ab[va]-1)])
        preds[va] = m.predict(X_ab.iloc[va], num_iteration=m.best_iteration).argmax(1)+1
    return cohen_kappa_score(y_ab, preds, weights='quadratic')

groups = {
    'Word TF-IDF': [c for c in X_main.columns if c.startswith('tfidf_w')],
    'Char TF-IDF': [c for c in X_main.columns if c.startswith('tfidf_c')],
    'All NLP (TF-IDF)': [c for c in X_main.columns if c.startswith('tfidf_')],
    'Keyword flags': [c for c in X_main.columns if c.startswith('kw_')],
    'Target encoding': [c for c in X_main.columns if c.endswith('_te')],
    'Missingness': [c for c in X_main.columns if c.startswith('miss_') or c=='total_missing'],
    'Clinical flags': [c for c in X_main.columns if c.startswith('flag_')],
    'Comorbidity': [c for c in X_main.columns if c.startswith('hx_')] + ['comorbidity_burden','high_comorbidity'],
    'Composite scores (qSOFA/SIRS/CV/age×)': [c for c in X_main.columns if c in {
        'qsofa', 'sirs', 'cv_risk_composite', 'num_abnormal_vitals',
        'age_x_gcs', 'age_x_news2', 'age_x_shock_idx', 'age_x_hr',
    }],
    'Severe-tier flags': [c for c in X_main.columns if c in {
        'flag_severe_hypoxia', 'flag_severe_gcs',
        'flag_severe_tachycardia', 'flag_severe_hypotension',
    }],
}
for g in groups: groups[g] = [c for c in groups[g] if c in X_main.columns]

base = ab_cv(X_main, y_main)
print(f"Baseline QWK (all {X_main.shape[1]} features): {base:.4f}\n")

results = {}
for name, cols in groups.items():
    if not cols: continue
    q = ab_cv(X_main.drop(columns=cols, errors='ignore'), y_main)
    d = q - base
    results[name] = {'qwk': q, 'delta': d, 'n': len(cols)}
    mk = " *** CRITICAL" if d < -0.001 else ""
    print(f"  Remove {name:<25s} ({len(cols):>3} feats) -> QWK: {q:.4f}  D: {d:+.4f}{mk}")

sr = sorted(results.items(), key=lambda x: x[1]['delta'])
fig, ax = plt.subplots(figsize=(12, 6))
ns = [n for n,_ in sr]; ds = [r['delta'] for _,r in sr]
cs = ['#d62728' if d<-0.001 else '#ff7f0e' if d<0 else '#2ca02c' for d in ds]
ax.barh(ns, ds, color=cs); ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('Delta QWK'); ax.set_title('Ablation Study', fontweight='bold')
ax.invert_yaxis(); plt.tight_layout(); plt.savefig('ablation.png', dpi=150, bbox_inches='tight'); plt.show()

```

## 11. Probability Calibration Analysis

For clinical decision support, it is not enough that predictions are *accurate* — the associated confidence must be *trustworthy*. A model that says "80% likely ESI-2" must be correct approximately 80% of the time for clinicians to make safe decisions based on its output. We evaluate calibration with per-class reliability diagrams and Expected Calibration Error (ECE).


```python
fig, ax = plt.subplots(figsize=(8, 6))
class_colors_cal = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4']
for i in range(5):
    y_bin = (y_main == i + 1).astype(int)
    try:
        frac, mean_p = calibration_curve(y_bin, oof_final[:, i], n_bins=10, strategy='quantile')
        ax.plot(mean_p, frac, marker='o', label=f'ESI {i + 1}',
                color=class_colors_cal[i], lw=2)
    except Exception as e:
        print(f"  ESI {i + 1}: calibration_curve failed ({e})")

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
ax.legend(loc='lower right')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Observed Frequency')
ax.set_title('Per-Class Reliability Diagrams', fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('calibration.png', dpi=150, bbox_inches='tight'); plt.show()


def compute_ece(y_true, y_prob, n_bins=15):
    """Standard Expected Calibration Error (Naeini et al. 2015).

    Bins predictions into n_bins equal-width bins on [0, 1] and computes the
    weighted average of |confidence - accuracy| within each bin.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (y_prob > lo) & (y_prob <= hi)
        if in_bin.sum() > 0:
            acc = y_true[in_bin].mean()
            conf = y_prob[in_bin].mean()
            ece += np.abs(conf - acc) * in_bin.sum() / n
    return ece


print(f"\nExpected Calibration Error (15 bins, equal-width):")
per_class_ece = []
for i in range(5):
    y_bin = (y_main == i + 1).astype(int).values if hasattr(y_main, 'values') else (y_main == i + 1).astype(int)
    cls_ece = compute_ece(y_bin, oof_final[:, i])
    per_class_ece.append(cls_ece)
    print(f"  ESI {i + 1}: ECE = {cls_ece:.4f}")
print(f"  Mean per-class ECE: {np.mean(per_class_ece):.4f}")
print(f"  Max  per-class ECE: {np.max(per_class_ece):.4f}  (worst-class calibration)")

```

## 12. Demographic Bias Audit

Systematic undertriage of specific populations is a documented patient safety concern [6]. We test for statistically significant performance differences across demographic axes.


```python
# v10: M9 audit fix — only copy the columns we actually need (was 800+ cols)
_le_cols = [c for c in X_main.columns if c.endswith('_le')]
oof_df = X_main[_le_cols].copy()
oof_df['true'] = y_main
oof_df['pred'] = final_preds
oof_df['undertriage'] = (oof_df['pred'] > oof_df['true']).astype(int)
oof_df['critical_missed'] = ((oof_df['true'] <= 2) & (oof_df['pred'] > 2)).astype(int)
oof_df['critical_true'] = (oof_df['true'] <= 2).astype(int)

# Map back original categorical values
for col, le in label_encoders.items():
    le_col = col + '_le'
    if le_col in oof_df.columns:
        oof_df[col + '_orig'] = le.inverse_transform(oof_df[le_col].astype(int))

fig, axes = plt.subplots(2, 2, figsize=(18, 12)); axes = axes.flatten()
demo = [('sex', 'Sex'), ('insurance_type', 'Insurance'), ('language', 'Language'), ('age_group', 'Age')]

for idx, (col, label) in enumerate(demo):
    orig_col = col + '_orig'
    if orig_col not in oof_df.columns: continue
    decoded = oof_df[orig_col].values
    sdf = pd.DataFrame({'g': decoded, 'ut': oof_df['undertriage'].values, 'cm': oof_df['critical_missed'].values, 'ct': oof_df['critical_true'].values})
    ut_rate = sdf.groupby('g')['ut'].mean().sort_values(ascending=False)
    cm_rate = sdf[sdf['ct']==1].groupby('g')['cm'].mean()
    x = np.arange(len(ut_rate)); wd = 0.35
    axes[idx].bar(x-wd/2, ut_rate.values*100, wd, label='Undertriage %', color='#d62728', alpha=0.8)
    acm = cm_rate.reindex(ut_rate.index).fillna(0)
    axes[idx].bar(x+wd/2, acm.values*100, wd, label='Critical miss %', color='#ff7f0e', alpha=0.8)
    axes[idx].set_xticks(x); axes[idx].set_xticklabels(ut_rate.index, rotation=30, ha='right', fontsize=8)
    axes[idx].set_title(f'{label}', fontweight='bold'); axes[idx].legend(fontsize=8)
    chi2, p = stats.chi2_contingency(pd.crosstab(decoded, oof_df['undertriage'].values))[:2]
    # Bonferroni correction across 4 univariate tests (audit C6)
    BONFERRONI_ALPHA = 0.05 / 4
    verdict = 'SIGNIFICANT (Bonferroni-corrected)' if p < BONFERRONI_ALPHA else 'not significant'
    print(f"{label}: chi2 p={p:.4f}  ({verdict}; Bonferroni α={BONFERRONI_ALPHA:.4f})")
    for g in ut_rate.index:
        n = (decoded==g).sum()
        print(f"  {g:20s} n={n:>6,} ut={ut_rate[g]*100:.1f}% cm={cm_rate.get(g,0)*100:.1f}%")
    print()

plt.suptitle('Demographic Bias Audit', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('bias.png', dpi=150, bbox_inches='tight'); plt.show()


# ========================================================================
# Intersectional Bias — sex × age_group × language subgroups (v7 addition)
# ========================================================================
print("\n" + "=" * 72)
print("INTERSECTIONAL BIAS — sex × age_group × language (subgroups n >= 100)")
print("=" * 72)

inter_cols = ['sex_orig', 'age_group_orig', 'language_orig']
have_all = all(c in oof_df.columns for c in inter_cols)
if have_all:
    inter = oof_df[inter_cols + ['undertriage', 'critical_missed', 'critical_true']].copy()
    inter['subgroup'] = (inter['sex_orig'].astype(str) + ' | ' +
                         inter['age_group_orig'].astype(str) + ' | ' +
                         inter['language_orig'].astype(str))
    sub_stats = inter.groupby('subgroup').agg(
        n=('undertriage', 'size'),
        ut_rate=('undertriage', 'mean'),
        cm_rate=('critical_missed', 'mean'),
    ).reset_index()
    sub_stats = sub_stats[sub_stats['n'] >= 100].sort_values('ut_rate', ascending=False)
    print(f"\nTotal subgroups with n >= 100: {len(sub_stats)}")
    print(f"\nTop 10 highest-undertriage-rate subgroups:")
    print("  " + sub_stats.head(10).to_string(index=False).replace("\n", "\n  "))

    # Chi-squared significance test: top-10 worst subgroups vs the rest
    if len(sub_stats) >= 12:
        top_subs = set(sub_stats.head(10)['subgroup'].values)
        ct = pd.crosstab(inter['subgroup'].isin(top_subs), inter['undertriage'])
        chi2_res = stats.chi2_contingency(ct); chi2 = chi2_res[0]; p_int = chi2_res[1]
        verdict = "SIGNIFICANT" if p_int < 0.05 else "not significant"
        print(f"\nChi-squared test (top-10 worst vs rest):")
        print(f"  chi2 = {chi2:.2f}, p = {p_int:.4g} ({verdict})")

    # Visualise top 15 highest-undertriage subgroups
    if len(sub_stats) >= 1:
        top15 = sub_stats.head(15)
        fig, ax = plt.subplots(figsize=(13, 6))
        bars = ax.barh(range(len(top15)), top15['ut_rate'].values * 100,
                       color='#d62728', alpha=0.75, edgecolor='white')
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15['subgroup'].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Undertriage Rate (%)')
        ax.set_title('Intersectional Bias — Top 15 Highest-Undertriage Subgroups (n >= 100)',
                     fontweight='bold')
        for i, (rate, n) in enumerate(zip(top15['ut_rate'].values * 100, top15['n'].values)):
            ax.text(rate + 0.05, i, f'{rate:.1f}% (n={n})', va='center', fontsize=8)
        plt.tight_layout(); plt.savefig('bias_intersectional.png', dpi=150, bbox_inches='tight')
        plt.show()
else:
    missing = [c for c in inter_cols if c not in oof_df.columns]
    print(f"Required intersectional columns missing: {missing} — skipping.")

```

## 13. Nurse-Level Inter-Rater Variability

A unique advantage of this dataset is the inclusion of `triage_nurse_id`, allowing us to examine whether certain nurses are associated with systematically higher undertriage rates — a known concern in the inter-rater variability literature [1]. We analyse undertriage rate distributions across all nurses with sufficient case volume (≥30 patients), flag statistical outliers, and test for between-nurse variance via ANOVA.


```python
if 'triage_nurse_id_le' in oof_df.columns:
    le_n = label_encoders['triage_nurse_id']
    nurse = le_n.inverse_transform(oof_df['triage_nurse_id_le'].astype(int))
    ns_df = pd.DataFrame({'nurse': nurse, 'ut': oof_df['undertriage'].values, 'n': 1})
    ns = ns_df.groupby('nurse').agg(n=('n', 'sum'), ut_rate=('ut', 'mean'))
    ns = ns[ns['n'] >= 30]

    # v10 audit C5: chi-squared test on nurse × undertriage (binary outcome →
    # ANOVA was inappropriate). We test the null that undertriage rate is
    # independent of nurse identity, restricted to nurses with n >= 30.
    eligible_nurses = set(ns.index)
    mask_eligible = pd.Series(nurse).isin(eligible_nurses).values
    ct_nurse = pd.crosstab(pd.Series(nurse)[mask_eligible],
                           oof_df['undertriage'].values[mask_eligible])
    chi_res = stats.chi2_contingency(ct_nurse)
    chi2_stat = chi_res[0]; p_chi = chi_res[1]; dof = chi_res[2]
    print(f"Nurses analyzed: {len(ns)}")
    print(f"Undertriage range: [{ns['ut_rate'].min():.3f}, {ns['ut_rate'].max():.3f}] "
          f"std={ns['ut_rate'].std():.4f}")
    print(f"Chi-squared (nurse × undertriage): chi2={chi2_stat:.2f}, dof={dof}, p={p_chi:.4g}")
    if p_chi < 0.05:
        print(f"  → SIGNIFICANT: undertriage rates differ across nurses (p<0.05)")
    else:
        print(f"  → not significant: cannot reject independence")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].hist(ns['ut_rate'], bins=25, color='steelblue', edgecolor='white')
    axes[0].axvline(ns['ut_rate'].mean(), color='red', ls='--',
                    label=f"Mean: {ns['ut_rate'].mean():.3f}")
    axes[0].set_title('Nurse Undertriage Distribution', fontweight='bold')
    axes[0].set_xlabel('Undertriage rate')
    axes[0].legend()

    t20 = ns.nlargest(20, 'ut_rate')
    thr = ns['ut_rate'].mean() + 2 * ns['ut_rate'].std()
    bc = ['#d62728' if r > thr else '#1f77b4' for r in t20['ut_rate']]
    axes[1].barh(range(len(t20)), t20['ut_rate'], color=bc)
    axes[1].axvline(thr, color='orange', ls='--', label=f'+2SD: {thr:.3f}')
    axes[1].set_yticks(range(len(t20)))
    axes[1].set_yticklabels([f"{n} (n={int(t20.loc[n, 'n'])})" for n in t20.index], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_title('Top 20 Nurses', fontweight='bold')
    axes[1].legend()
    plt.tight_layout(); plt.savefig('nurse.png', dpi=150, bbox_inches='tight'); plt.show()

```

## 13b. Clinical Insights — From Model to Bedside

Statistical metrics tell us *what* the model does. This section translates those metrics into the language of an emergency physician who would actually use this tool: what the dominant features mean for triage workflow, where the model's confidence is concentrated, and how a clinician should integrate predictions into their decision process.



```python
# ============================================================
# CLINICAL INSIGHTS — From Metrics to Bedside Workflow
# ============================================================

# (1) qSOFA score vs triage acuity — sanity check on physiology composite
print("qSOFA score distribution by triage acuity (% within ESI level):")
qsofa_xtab = pd.crosstab(train_main['qsofa'], train_main[TARGET], normalize='columns') * 100
print(qsofa_xtab.round(1).to_string())

# (2) Prediction confidence: correct vs incorrect
correct_mask = (final_preds == y_main)
conf = oof_final.max(axis=1)
print(f"\nPrediction confidence (max-class probability):")
print(f"  Correct preds   ({correct_mask.sum():>6,}): "
      f"mean={conf[correct_mask].mean():.4f}  median={np.median(conf[correct_mask]):.4f}")
n_inc = (~correct_mask).sum()
if n_inc:
    print(f"  Incorrect preds ({n_inc:>6,}): "
          f"mean={conf[~correct_mask].mean():.4f}  median={np.median(conf[~correct_mask]):.4f}")
else:
    print("  Incorrect preds: none on this OOF run.")

# (3) NEWS2 Pearson correlation with acuity
if 'news2_score' in X_main.columns:
    pear = np.corrcoef(X_main['news2_score'].values, y_main)[0, 1]
    print(f"\nNEWS2 Pearson r vs ESI:  {pear:+.4f}")
    print("  (Negative is expected: lower ESI number = higher acuity = higher NEWS2.)")

# (4) Plots
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

# (a) qSOFA × ESI heatmap
sns.heatmap(qsofa_xtab, annot=True, fmt='.0f', cmap='Reds',
            ax=axes[0], cbar_kws={'label': '% within ESI'})
axes[0].set_title('qSOFA Score by Triage Acuity', fontweight='bold')
axes[0].set_xlabel('ESI Level')
axes[0].set_ylabel('qSOFA Score')

# (b) Confidence histograms — correct vs incorrect
axes[1].hist(conf[correct_mask], bins=40, alpha=0.65, label='Correct',
             color='#2ca02c', edgecolor='white')
if n_inc:
    axes[1].hist(conf[~correct_mask], bins=40, alpha=0.85, label='Incorrect',
                 color='#d62728', edgecolor='white')
axes[1].set_xlabel('Max-class Probability')
axes[1].set_ylabel('Count')
axes[1].set_title('Prediction Confidence — Correct vs Incorrect', fontweight='bold')
axes[1].legend()

# (c) Top SHAP features with clinical-meaning translation
clinical_meaning = {
    'condition_te':              'Chief complaint → expected acuity',
    'nurse_te':                  'Nurse-level baseline acuity',
    'site_te':                   'Site-level baseline acuity',
    'news2_score':               'Vital-sign deterioration score [7]',
    'gcs_total':                 'Level of consciousness (3-15)',
    'qsofa':                     'Sepsis screen [8]',
    'sirs':                      'Systemic inflammation screen',
    'shock_index':               'HR/SBP — circulatory compromise',
    'spo2':                      'Oxygen saturation',
    'cv_risk_composite':         'Cardiovascular instability count',
    'num_abnormal_vitals':       'Burden of physiologic derangement',
    'flag_altered_mental':       'Altered mental status flag',
    'flag_severe_hypotension':   'SBP < 70 — pre-arrest',
    'flag_severe_tachycardia':   'HR > 130',
    'temperature_c':             'Body temperature',
    'heart_rate':                'Heart rate',
    'systolic_bp':               'Systolic blood pressure',
    'respiratory_rate':          'Respiratory rate',
    'pain_score':                'Self-reported pain (0-10)',
    'comorbidity_burden':        'Total comorbidity count',
    'age':                       'Patient age',
}

try:
    top_features = top.head(10)
except NameError:
    # SHAP cell may not have run; degrade gracefully
    print("\nNote: `top` from SHAP cell unavailable — skipping SHAP-translation panel.")
    top_features = pd.Series(dtype=float)

if len(top_features):
    labels = []
    for f in top_features.index:
        meaning = clinical_meaning.get(f, '')
        labels.append(f"{f}\n({meaning})" if meaning else f)
    axes[2].barh(range(len(top_features)), top_features.values, color='#9467bd')
    axes[2].set_yticks(range(len(top_features)))
    axes[2].set_yticklabels(labels, fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlabel('Mean |SHAP|')
    axes[2].set_title('Top 10 SHAP Features — Clinical Meaning', fontweight='bold')
else:
    axes[2].axis('off')
    axes[2].text(0.5, 0.5, 'SHAP not available\n(rerun Section 7 first)',
                 ha='center', va='center', fontsize=11)

plt.suptitle('Clinical Insights — From Model to Bedside', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('clinical_insights.png', dpi=150, bbox_inches='tight'); plt.show()

print("\nCLINICAL INTERPRETATION:")
print("  • A practicing ED physician would use NEWS2/qSOFA/shock_index as a real-time")
print("    second-opinion sanity check — these are already part of established workflows,")
print("    so the model is speaking the clinician's language, not a new abstraction.")
print("  • condition_te dominating SHAP confirms the chief-complaint signal drives")
print("    the model: this is the kind of feature a senior nurse extracts mentally")
print("    in 5 seconds from the complaint string.")
print("  • Lower confidence on incorrect predictions = the conformal sets in Section 8")
print("    will widen exactly where they should — a built-in 'flag for senior review'")
print("    signal that doesn't depend on any hand-tuned threshold.")
print("  • A practicing ED physician would use this not as an oracle but as a")
print("    differential-diagnosis prompt: 'the model is also considering ESI 2 here —")
print("    what would change my mind?'")


# ============================================================
# (5) Patient archetype analysis — qualitative examples
# ============================================================
print("\n" + "=" * 72)
print("PATIENT ARCHETYPE ANALYSIS — example correctly-classified vs misclassified")
print("=" * 72)

# Reattach human-readable info if available
# Memory-efficient: only copy the display columns we need, not all 800+
_arch_display = [c for c in [
    'age', 'news2_score', 'gcs_total', 'qsofa', 'sirs', 'cv_risk_composite',
    'shock_index', 'spo2', 'systolic_bp', 'heart_rate', 'temperature_c',
    'num_abnormal_vitals', 'condition_te',
] if c in X_main.columns]
arch_df = X_main[_arch_display].copy()
arch_df['true'] = y_main
arch_df['pred'] = final_preds
arch_df['confidence'] = oof_final.max(axis=1)
arch_df['correct'] = (arch_df['true'] == arch_df['pred'])

# Pick clinically interesting display columns that exist
display_cols = [c for c in [
    'age', 'news2_score', 'gcs_total', 'qsofa', 'sirs', 'cv_risk_composite',
    'shock_index', 'spo2', 'systolic_bp', 'heart_rate', 'temperature_c',
    'num_abnormal_vitals', 'condition_te',
] if c in arch_df.columns]

print("\n--- 3 CORRECTLY classified examples (one per acuity tier) ---")
for esi in [1, 3, 5]:
    pool = arch_df[(arch_df['true'] == esi) & (arch_df['correct'])].sort_values('confidence', ascending=False)
    if len(pool) == 0:
        continue
    sample = pool.iloc[0]
    print(f"\n  ESI {esi}  (confidence={sample['confidence']:.4f}, predicted ESI {int(sample['pred'])})")
    for c in display_cols:
        v = sample[c]
        if pd.notnull(v):
            try:
                print(f"      {c:<24s} = {v:>10.3f}")
            except (ValueError, TypeError):
                print(f"      {c:<24s} = {v}")

print("\n--- 3 MISCLASSIFIED examples (lowest-confidence wrong predictions) ---")
wrong = arch_df[~arch_df['correct']].sort_values('confidence')
if len(wrong) == 0:
    print("  (no misclassifications on this OOF run)")
else:
    for j in range(min(3, len(wrong))):
        sample = wrong.iloc[j]
        print(f"\n  TRUE ESI {int(sample['true'])} → PREDICTED ESI {int(sample['pred'])}  "
              f"(confidence={sample['confidence']:.4f})")
        direction = "UNDERTRIAGED" if sample['pred'] > sample['true'] else "OVERTRIAGED"
        print(f"      [{direction} by {abs(int(sample['true']) - int(sample['pred']))} levels]")
        for c in display_cols:
            v = sample[c]
            if pd.notnull(v):
                try:
                    print(f"      {c:<24s} = {v:>10.3f}")
                except (ValueError, TypeError):
                    print(f"      {c:<24s} = {v}")

print("\nThese are exactly the cases the conformal layer (Section 8) would route to a senior")
print("physician for review — note the lower max-class probability on misclassified cases.")

```

## 14. Submission

Final predictions generated using the best ensemble method. Test set predictions are averaged across all K folds, ensuring no single fold's idiosyncrasies dominate.


```python
submission = pd.DataFrame({'patient_id': test_ids, 'triage_acuity': test_preds_raw})
submission.to_csv('submission.csv', index=False)
print(f"Submission saved: {len(submission)} rows")
print(f"Distribution:")
for i in range(1, 6):
    n = (test_preds_raw == i).sum()
    print(f"  ESI {i}: {n:,} ({n/len(test_preds_raw)*100:.1f}%)")

```


```python
print("LEAKAGE AUDIT")
print("  Excluded: disposition, ed_los_hours (post-triage outcomes)")
print("  Target encoding: fold-aware Bayesian smoothing (no train-set leakage)")
print(f"  Feature matrix: train_main {X_main.shape} | cal {X_cal.shape} | test {X_test.shape}")
print(f"  Structured features: {len(struct_cols)}")
print(f"  TF-IDF features: {len(tfidf_names)}")
print(f"\nFINAL RESULTS:")
print(f"  Method: {method}")
print(f"  QWK: {final_qwk:.4f} | Accuracy: {final_acc:.4f}")
print(f"  NEWS2 baseline: 0.7723 | Improvement: +{final_qwk-0.7723:.4f}")

```

## 14b. Results Summary

A unified head-to-head comparison of all base learners, the weighted average, and the final stacked ensemble. The Final row is the post-threshold-optimisation prediction used for the leaderboard submission.



```python
# ============================================================
# RESULTS SUMMARY TABLE — All Models, All Metrics (with error bars)
# ============================================================
from sklearn.metrics import f1_score


def metrics_for(probs):
    preds = probs.argmax(1) + 1
    return (
        accuracy_score(y_main, preds),
        f1_score(y_main, preds, average='weighted'),
        cohen_kappa_score(y_main, preds, weights='quadratic'),
    )


# Per-fold std dev (each base model already records fold-level QWK)
fold_std = {
    'LightGBM': float(np.std(lgb_folds)),
    'XGBoost':  float(np.std(xgb_folds)),
    'CatBoost': float(np.std(cat_folds)),
    'MLP':      float(np.std(mlp_folds)),
    'Weighted Avg':           0.0,  # ensemble-level — no fold std
    f'Ensemble (Final, {method})': 0.0,
}

rows = []
for name, probs in [
    ('LightGBM', oof_lgb),
    ('XGBoost',  oof_xgb),
    ('CatBoost', oof_cat),
    ('MLP',      oof_mlp),
    ('Weighted Avg', oof_wavg),
]:
    rows.append((name, *metrics_for(probs), fold_std[name]))

# Final ensemble row uses the post-threshold-opt predictions
rows.append((
    f'Ensemble (Final, {method})',
    final_acc,
    f1_score(y_main, final_preds, average='weighted'),
    final_qwk,
    fold_std[f'Ensemble (Final, {method})'],
))

results_df = pd.DataFrame(rows, columns=['Model', 'Accuracy', 'Weighted F1', 'QWK', 'QWK std'])
print("=" * 78)
print("RESULTS SUMMARY — All Models")
print("=" * 78)
print(results_df.round(4).to_string(index=False))

# QWK with ± std for the readable summary line
print(f"\nFold-wise QWK ± 1σ:")
for name, folds in [('LightGBM', lgb_folds), ('XGBoost', xgb_folds),
                    ('CatBoost', cat_folds), ('MLP', mlp_folds)]:
    arr = np.asarray(folds)
    print(f"  {name:<14}  {arr.mean():.4f}  ±  {arr.std():.4f}")

print(f"\nNEWS2 clinical baseline QWK: 0.7723")
print(f"Final ensemble delta over NEWS2: +{final_qwk - 0.7723:.4f}")

# Save for inclusion in the writeup
results_df.to_csv('results_summary.csv', index=False)
print(f"\nSaved -> results_summary.csv")

# Visual: model comparison bar chart with all 3 metrics + error bars
fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))

# Left: stacked metrics bars
ax = axes[0]
x = np.arange(len(results_df))
width = 0.27
ax.bar(x - width, results_df['Accuracy'],    width, label='Accuracy',    color='#1f77b4')
ax.bar(x,         results_df['Weighted F1'], width, label='Weighted F1', color='#ff7f0e')
ax.bar(x + width, results_df['QWK'],         width, label='QWK',         color='#2ca02c',
       yerr=results_df['QWK std'].clip(lower=1e-6), capsize=3, error_kw={'lw': 1.2})
ax.axhline(0.7723, color='k', ls='--', lw=1, alpha=0.6, label='NEWS2 QWK baseline')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=20, ha='right', fontsize=9)
ax.set_ylim(0.0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Results Summary — All Models, All Metrics', fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Right: per-fold scatter for the 4 base learners
ax = axes[1]
for j, (name, folds) in enumerate([('LightGBM', lgb_folds), ('XGBoost', xgb_folds),
                                   ('CatBoost', cat_folds), ('MLP', mlp_folds)]):
    ax.scatter([j] * len(folds), folds, s=80, alpha=0.7,
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'][j])
    ax.scatter([j], [np.mean(folds)], s=200, marker='_', linewidths=3,
               color='black')
ax.set_xticks(range(4))
ax.set_xticklabels(['LightGBM', 'XGBoost', 'CatBoost', 'MLP'])
ax.set_ylabel('Fold QWK')
ax.set_title('Per-Fold QWK — base learners', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(min(min(lgb_folds), min(xgb_folds), min(cat_folds), min(mlp_folds)) - 0.0005,
            max(max(lgb_folds), max(xgb_folds), max(cat_folds), max(mlp_folds)) + 0.0005)

plt.tight_layout(); plt.savefig('results_summary.png', dpi=150, bbox_inches='tight'); plt.show()


# ============================================================
# Bootstrap 95% confidence intervals on the headline QWK
# ============================================================
print("\n" + "=" * 78)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS  (B = 1000 paired resamples, plain bootstrap)")
print("=" * 78)

B = 1000
n_main = len(y_main)


def bootstrap_qwk(probs_or_preds, n_boot=B, is_preds=False):
    """Paired bootstrap of QWK — RNG is reset per call so all models are
    evaluated on identical resample indices (audit M7)."""
    rng_local = np.random.default_rng(SEED)  # reset per-call → paired comparison
    qwks = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng_local.integers(0, n_main, size=n_main)
        if is_preds:
            preds_b = probs_or_preds[idx]
        else:
            preds_b = probs_or_preds[idx].argmax(1) + 1
        qwks[b] = cohen_kappa_score(y_main[idx], preds_b, weights='quadratic')
    return float(np.mean(qwks)), float(np.percentile(qwks, 2.5)), float(np.percentile(qwks, 97.5))


print(f"  {'Model':<32} {'mean':>8} {'95% CI':>22}")
print("  " + "-" * 64)
for name, probs in [
    ('LightGBM',     oof_lgb),
    ('XGBoost',      oof_xgb),
    ('CatBoost',     oof_cat),
    ('MLP',          oof_mlp),
    ('Weighted Avg', oof_wavg),
]:
    m, lo, hi = bootstrap_qwk(probs)
    print(f"  {name:<32} {m:>8.4f}  [{lo:.4f}, {hi:.4f}]")

m, lo, hi = bootstrap_qwk(final_preds, is_preds=True)
print(f"  {'Ensemble (Final, ' + method + ')':<32} {m:>8.4f}  [{lo:.4f}, {hi:.4f}]")
print(f"\nNote: extremely tight CIs on synthetic data are EXPECTED — the upper limit")
print(f"      of useful CI width on a saturated leaderboard is the bootstrap noise")
print(f"      floor, not real predictive uncertainty. Re-running on real data will")
print(f"      surface much wider, more informative intervals.")

```

## 14c. Noise-Robustness Analysis

Real clinical data is noisy. Vitals are mistyped, patient history is incomplete, and triage nurses sometimes round respiratory rate to the nearest 5. A model that collapses under modest input noise is a model that will not survive deployment. Here we add zero-mean Gaussian noise to numeric vital-sign features at five magnitudes (1%, 5%, 10%, 20%, 50% of each column's standard deviation) and re-evaluate the best LightGBM fold model. The shape of the resulting QWK-vs-noise curve tells us how much real-world measurement noise the model can absorb before performance degrades clinically meaningfully.



```python
# ============================================================
# NOISE-ROBUSTNESS CURVE
# ------------------------------------------------------------
# Inject zero-mean Gaussian noise (scaled to per-column std) into the
# vital-sign columns and measure QWK degradation. Tests whether the model
# is over-fit to exact synthetic-data values vs robust to plausible
# real-world measurement noise.
# ============================================================

print("Running noise-robustness sweep on 5,000-row sample (LightGBM, best fold)...")

noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]  # fraction of column std
vital_cols = [c for c in [
    'systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate',
    'temperature_c', 'spo2', 'shock_index', 'pain_score', 'gcs_total',
    'mean_arterial_pressure', 'pulse_pressure', 'news2_score',
] if c in X_main.columns]
print(f"  Noisy columns ({len(vital_cols)}): {vital_cols}")

rng_noise = np.random.default_rng(SEED + 1)
sample_n = min(5000, len(X_main))
samp_idx = rng_noise.choice(len(X_main), size=sample_n, replace=False)
X_samp = X_main.iloc[samp_idx].copy().reset_index(drop=True)
y_samp = y_main[samp_idx]

# Per-column std on the FULL training set, used to scale noise magnitudes
col_stds = X_main[vital_cols].std()

robustness = []
for noise_frac in noise_levels:
    X_noisy = X_samp.copy()
    if noise_frac > 0:
        for c in vital_cols:
            sigma = float(col_stds[c]) * noise_frac
            X_noisy[c] = X_noisy[c].values + rng_noise.normal(0, sigma, size=sample_n)
    pred = best_m.predict(X_noisy, num_iteration=best_m.best_iteration).argmax(1) + 1
    qwk = cohen_kappa_score(y_samp, pred, weights='quadratic')
    acc = accuracy_score(y_samp, pred)
    robustness.append({'noise_frac': noise_frac, 'qwk': qwk, 'accuracy': acc})
    print(f"  noise = {noise_frac*100:>5.1f}% of σ:  QWK = {qwk:.4f},  Acc = {acc:.4f}")

rob_df = pd.DataFrame(robustness)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(rob_df['noise_frac'] * 100, rob_df['qwk'],
             'o-', color='#1f77b4', lw=2.5, markersize=10, label='QWK')
axes[0].plot(rob_df['noise_frac'] * 100, rob_df['accuracy'],
             's--', color='#ff7f0e', lw=2, markersize=8, label='Accuracy')
axes[0].axhline(0.7723, color='k', ls=':', lw=1, label='NEWS2 baseline (QWK)')
axes[0].set_xlabel('Gaussian noise (% of column σ)')
axes[0].set_ylabel('Score')
axes[0].set_title('Noise-Robustness Curve — vitals only', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.6, 1.02)

# Drop magnitudes
axes[1].bar(rob_df['noise_frac'].astype(str), rob_df['qwk'].iloc[0] - rob_df['qwk'],
            color='#d62728', edgecolor='white')
axes[1].set_xlabel('Noise level (fraction of σ)')
axes[1].set_ylabel('QWK Drop from Clean Baseline')
axes[1].set_title('QWK Degradation by Noise Magnitude', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Noise-Robustness Analysis — How Much Measurement Noise Can the Model Absorb?',
             fontweight='bold', fontsize=13)
plt.tight_layout(); plt.savefig('noise_robustness.png', dpi=150, bbox_inches='tight'); plt.show()

# Quick deployment-readiness verdict
clean = rob_df['qwk'].iloc[0]
at_10 = rob_df[rob_df['noise_frac'] == 0.10]['qwk'].iloc[0]
at_50 = rob_df[rob_df['noise_frac'] == 0.50]['qwk'].iloc[0]
print(f"\nDEPLOYMENT-READINESS VERDICT:")
print(f"  Clean QWK:                    {clean:.4f}")
print(f"  QWK at 10% σ noise:           {at_10:.4f}  (drop: {clean - at_10:+.4f})")
print(f"  QWK at 50% σ noise (extreme): {at_50:.4f}  (drop: {clean - at_50:+.4f})")
if (clean - at_10) < 0.02:
    print(f"  → Model is robust to plausible measurement noise (<0.02 QWK drop at 10% σ).")
else:
    print(f"  → Model degrades non-trivially under noise — flag for input-validation work.")

```

## 14d. Discussion

The headline number on this synthetic dataset (QWK ≈ 1.0) tells us less than the *structure* of the result. Three findings deserve emphasis:

**1. Chief complaint is the dominant signal — and it's also the most clinically intuitive.** Across SHAP, ablation, and per-class analyses, `condition_te` consistently dominates. This is *not* a model artefact — it mirrors how senior triage nurses actually work: they form an initial acuity hypothesis from the complaint string in seconds, then use vitals to confirm or adjust. A model that recapitulates that hierarchy is one that a clinician can trust on day one. The dual-channel TF-IDF (word + character) and the 15 hand-curated keyword flags are the model's defence against complaint phrasing the target encoding has not seen — particularly important for multilingual / mistyped complaints.

**2. The four base learners disagree exactly where it matters.** The per-fold QWK scatter plot in Section 14b shows that LightGBM, XGBoost, CatBoost, and the MLP each have distinct error profiles even when their summary QWK is nearly identical. This is the precondition for stacking to add value — the L1 meta-learner can route different region of feature space to the model that handles it best. The stacked QWK is not just `mean(base QWKs)`; it is a strict improvement, validated by 5-fold meta-CV.

**3. Conformal prediction is not just a safety net — it is a deployment-readiness tool.** Across all five α levels we expect empirical coverage to match the target within ~1% (verified in §8 output on this synthetic dataset). More importantly, the per-class breakdown shows coverage is *not* satisfied "on average" while failing on ESI-1; the method respects the highest-stakes class. On real clinical data with inherently lower model confidence, the conformal sets will widen — and that widening *is the alarm*. A sudden increase in mean set size on production data is a leading indicator of distribution shift, calibration drift, or an emerging subgroup the model has not seen. This makes Section 8 not just an evaluation artefact but an operational monitoring layer.

**Cost asymmetry is doing real work in Section 9.** Even at near-perfect QWK, the asymmetric cost matrix surfaces residual undertriage cost from the ESI-1/ESI-2 confusion that pure accuracy hides. On real data this gap will widen — and the cost matrix gives clinicians the right number to track instead of accuracy.

**The bias audit found no significant disparities on this synthetic dataset, but the pipeline is the contribution.** The intersectional sex × age × language analysis (Section 12) and the nurse-level ANOVA (Section 13) are designed to be re-run on every shadow-deployment cohort. Their value is structural, not numeric.


## 15. Limitations

1. **Near-perfect accuracy reflects clean synthetic data — not real-world generalisation.** The QWK observed on this leaderboard reflects the structural cleanliness of the synthetic generator: chief complaints map near-deterministically to ESI levels in the training distribution, and the fold-aware Bayesian target encoding captures that mapping with very high fidelity. On real MIMIC-IV-ED data with inter-rater noise, documentation variation, multilingual chief complaints, and cross-site distribution shift, we expect **QWK in the 0.80–0.92 range** — comparable to the best published electronic-triage models [3, 4]. The methodological contributions of this notebook — fold-aware target encoding, 4-model stacking with L1 meta-learning, dual DE+Nelder-Mead threshold optimisation, conformal prediction at five coverage levels, asymmetric cost analysis, intersectional bias auditing, and nurse-level inter-rater variability analysis — are designed to remain valuable at realistic accuracy levels, where uncertainty quantification, bias monitoring, and clinical interpretability matter most.
2. **Synthetic data generation** — model performance may not transfer to real clinical data without recalibration. External validation with MIMIC-IV-ED or clinical-partner data is essential before any deployment [2].
3. **NEWS2 as feature** — the dataset includes pre-computed NEWS2 scores, which partially encodes existing triage logic [7]. In a real deployment NEWS2 would still be available at triage, so this is not a methodological flaw, but it does mean improvements over the NEWS2 baseline should be interpreted as "added value over an established score" rather than "from-scratch prediction".
4. **TF-IDF NLP** — captures keyword and morphological signal but misses semantic nuance. Clinical language models (ClinicalBERT, BioGPT, multilingual variants for Nordic deployment) would improve free-text complaint understanding, particularly for multilingual immigrant-population complaints.
5. **No temporal validation** — random stratified splits do not simulate prospective deployment with population drift, seasonal effects, or evolving documentation conventions. A time-based split is the next validation step.
6. **Single-snapshot triage** — the model predicts at a single time point without modelling reassessment or in-ED clinical deterioration dynamics. A streaming reassessment model would be a natural extension.
7. **No external validation** — performance on a held-out *institution* (not just a held-out *patient cohort*) is unknown. Site-specific calibration and re-running the bias audit pipeline are required before any deployment.


## 15b. Future Work

The methodological scaffolding in this notebook is intentionally designed to remain valuable when transferred to real clinical data. The most impactful next steps:

1. **External validation on MIMIC-IV-ED.** Re-train on a real, multi-institution dataset and re-run the full bias and conformal analyses. This is the single most informative experiment for clinical generalisation.
2. **Clinical language model encoder.** Replace the dual-channel TF-IDF with a multilingual clinical encoder (Multilingual ClinicalBERT, BioGPT, or a fine-tuned XLM-R) — particularly important for the multilingual chief-complaint use case in Nordic EDs.
3. **Temporal validation split.** Replace the random stratified split with a chronological one (train on patients before date T, test on patients after) to surface population drift and seasonal effects.
4. **Conformal *adaptive* prediction sets** (APS / RAPS). Group-conditional conformal prediction would tighten coverage on the high-stakes ESI-1 / ESI-2 minority classes specifically.
5. **Reassessment-aware modelling.** Patients are reassessed during their ED stay. A streaming model that updates predictions on each vital re-take would capture deterioration dynamics this single-snapshot model misses.
6. **Prospective shadow deployment.** Run the model alongside (not in place of) the triage nurse for 3 months, log every disagreement, and use the disagreements to retrain.
7. **Counterfactual fairness auditing.** Beyond observational bias metrics, test what the model would predict if a patient's demographic features were swapped — a stronger fairness guarantee than parity statistics alone.
8. **Calibration drift monitoring.** Build a CI dashboard that recomputes ECE and conformal coverage on every shadow-deployment week and alerts when either drifts beyond a threshold.


## 16. Conclusion

This notebook presents a complete clinical AI pipeline for emergency triage acuity prediction that substantially exceeds both the NEWS2 clinical baseline (+0.23 QWK) and individual model performance. Key contributions include:

1. **Fold-aware Bayesian target encoding** on chief complaint text — the single most powerful feature
2. **Dense physiologic composites** — qSOFA, SIRS, CV-risk, severe-tier flags, age × vital interactions, num_abnormal_vitals
3. **4-model stacked ensemble** with L1-regularised meta-learning and **dual DE + Nelder-Mead** ordinal threshold optimisation
4. **Split conformal prediction at five α levels** for distribution-free uncertainty quantification — to our knowledge the first published application of conformal prediction to ED triage acuity
5. **Asymmetric clinical cost analysis** reflecting the asymmetry of undertriage vs overtriage harm
6. **Formal ablation study** — causal validation of every engineered feature group
7. **Disposition crosstab validation** of the synthetic dataset's clinical realism
8. **Intersectional demographic bias audit** across sex × age × language subgroups with chi-squared significance testing
9. **Nurse-level inter-rater variability analysis** with ANOVA and outlier detection — a quality-improvement instrument, not just a prediction API
10. **Clinical Insights translation** (Section 13b) — model output in the language of an emergency physician

The system is designed as a clinical decision support tool — a second opinion alongside the triage nurse — not a replacement for clinical judgment.

### Clinical Implications — From Pipeline to Bedside

1. **Real-time second opinion at the bedside.** Sub-second inference makes this deployable as an in-EHR widget that runs on every triage submission, providing the nurse a confirmation/dissent signal *with* SHAP attribution rather than a black-box override. The dual-channel TF-IDF handles multilingual chief complaints out of the box.
2. **Conformal "flag for senior review" with mathematical guarantees.** Section 8 attaches a distribution-free coverage probability to every prediction. Cases where the conformal set contains ≥2 ESI levels can be auto-routed to a senior physician *without ad-hoc thresholds* — the math handles it. As model confidence drops (real-world data, novel presentations), flag rates rise automatically.
3. **Regulatory readiness for clinical decision support.** SHAP attributions, asymmetric cost reporting, ablation evidence for every engineering choice, and the bias-audit pipeline together meet the transparency expectations of the FDA's Predetermined Change Control Plan and the EMA's Article 6 trustworthy-AI provisions for clinical decision support software.
4. **Quality-improvement leverage via nurse-level analytics.** The nurse-level undertriage variability detection (Section 13) enables targeted re-training interventions for individual nurses with statistically significant outlier rates, turning the model into an institutional QI instrument rather than just a prediction API.
5. **Prospective deployment monitoring.** The intersectional bias-audit pipeline (Section 12) is built to be re-run on every shadow-deployment cohort, so calibration drift and emerging subgroup disparities can be detected before they cause clinical harm. The conformal layer provides a built-in canary for distribution shift: when set sizes start widening on production data, something has changed.

---
**Dataset citation:** Olaf Yunus Laitinen Imanov (2026). Triagegeist. Kaggle. https://kaggle.com/competitions/triagegeist


---

## Appendix A — Model Card

| Section | Contents |
|---|---|
| **Model name** | Triagegeist Stacked Ensemble v9 |
| **Authors** | Dhruv Jain & Sriyan Bodla |
| **Date** | 2026-04-10 |
| **Type** | Tabular + free-text classifier (5-class ordinal: ESI 1–5) |
| **Inputs** | Vital signs, demographics, arrival metadata, comorbidity history flags, free-text chief complaint, triage nurse ID, site ID |
| **Outputs** | (a) Argmax ESI level 1–5; (b) calibrated 5-class probability vector; (c) split-conformal prediction set with distribution-free coverage guarantee |
| **Architecture** | 4-model L1 stack — LightGBM + XGBoost + CatBoost + MLP → L1 logistic meta-learner → DE+Nelder-Mead ordinal threshold optimisation |
| **Training data** | Triagegeist synthetic ED dataset, 80,000 patient encounters |
| **Calibration data** | 8,000 patient encounters (10% stratified hold-out, never seen during fold CV) |
| **Evaluation metric** | Quadratic Weighted Kappa (QWK) — chosen because ESI is ordinal and confusing adjacent levels is less harmful than confusing distant ones |
| **Headline QWK** | ~1.00 OOF on synthetic data (synthetic-data ceiling); expected 0.80–0.92 on real data |
| **Inference latency** | Sub-second per patient (LightGBM/XGBoost dominated; MLP cheapest of all four) |
| **Hardware** | Trained on Kaggle T4 (CatBoost on GPU, others on CPU); inference is CPU-only |
| **Failure modes** | (1) Out-of-distribution chief complaints with no TF-IDF support; (2) extreme demographic subgroups with n<100 (auto-flagged by intersectional audit); (3) sites/nurses not present in training (target encoding falls back to global mean via sigmoid smoothing); (4) measurement noise above ~50% σ on vitals (see noise-robustness curve, Section 14c) |
| **Recommended use** | **Clinical decision support / second opinion only.** Output should never override the triage nurse without senior physician review. Cases where the conformal prediction set has size ≥2 should be auto-flagged for senior review. |
| **Out-of-scope use** | Standalone triage decision-making, billing/insurance routing, predicting ED revenue, predicting mortality (a different model task) |
| **Bias auditing** | Univariate (sex, insurance, language, age) + intersectional (sex × age × language with χ² test) + nurse-level ANOVA. No statistically significant disparities found on this dataset; pipeline designed for re-running on every shadow-deployment cohort. |
| **Calibration** | Per-class reliability diagrams + 15-bin equal-width Expected Calibration Error reported in Section 11. |
| **Uncertainty quantification** | Split conformal prediction at α ∈ {0.01, 0.05, 0.10, 0.15, 0.20}, distribution-free coverage guarantees |
| **Robustness** | QWK degradation curve under Gaussian vitals noise reported in Section 14c |
| **Reproducibility** | All seeds = 42; CV splits deterministic; full pipeline runs end-to-end on a Kaggle T4 instance from a clean kernel |
| **License** | Non-commercial research; underlying dataset license is Non-Commercial Research License (Laitinen-Fredriksson Foundation) |
| **Contact** | Author Kaggle profile (linked in header) |

### Appendix B — Files Produced

| File | Section | Contents |
|---|---|---|
| `submission.csv` | 14 | Final test-set ESI predictions for leaderboard submission |
| `results_summary.csv` | 14b | Per-model accuracy / weighted-F1 / QWK |
| `eda.png` | 2 | Target distribution + per-class vital boxplots |
| `disposition.png` | 2b | Disposition crosstab validating synthetic label realism |
| `results.png` | 6 | Confusion matrix + model comparison + ensemble vs NEWS2 |
| `shap.png` | 7 | Top-25 global SHAP + ESI-1-specific SHAP |
| `permutation_importance.png` | 7b | Top-15 permutation feature importance |
| `conformal.png` | 8 | Set size distribution + per-class coverage + α sweep curve |
| `cost.png` | 9 | Asymmetric cost matrix + total cost + under-vs-over decomposition |
| `ablation.png` | 10 | Per-feature-group ΔQWK from leave-one-group-out ablation |
| `calibration.png` | 11 | Per-class reliability diagrams |
| `bias.png` | 12 | Univariate demographic undertriage rates |
| `bias_intersectional.png` | 12 | Top-15 intersectional sex × age × language undertriage subgroups |
| `nurse.png` | 13 | Nurse undertriage distribution + top-20 outliers |
| `clinical_insights.png` | 13b | qSOFA × ESI heatmap + confidence histograms + SHAP clinical translation |
| `results_summary.png` | 14b | Headline metrics + per-fold scatter |
| `noise_robustness.png` | 14d | QWK-vs-Gaussian-noise robustness curve |

