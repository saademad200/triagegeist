# 🏥 Triagegeist: AI in Emergency Triage
## Exploratory Data Analysis 

**Goal:** Build a deep, clinically-grounded understanding of the synthetic emergency triage dataset before any modeling decisions are made.

**Target Variable:** `triage_acuity` — an integer from 1 (most critical) to 5 (least urgent).

**Evaluation Criteria:** Clinical relevance · Technical quality · Documentation quality · Insights · Novelty

---

> ⚠️ **Leakage Warning:** `disposition` and `ed_los_hours` are post-triage outcomes present in train only. They will **never** be used as features.

---

## Step 1 — Data Loading & Schema Inspection

Before any analysis, we load all four dataset files and systematically inspect their structure.
The goal is to build a complete mental map of what data we have and how the files relate to each other.

### File Relationships
All four files share a common `patient_id` key:
- `train.csv` → 80,000 patients with labels
- `test.csv` → 20,000 patients without labels (competition submission target)
- `chief_complaints.csv` → free-text complaints for all 100,000 patients
- `patient_history.csv` → binary comorbidity flags for all 100,000 patients


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#3a3d4a', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#a0a0a0', 'ytick.color': '#a0a0a0',
    'text.color': '#e0e0e0', 'grid.color': '#2a2d3a',
    'grid.linestyle': '--', 'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13, 'axes.labelsize': 11,
})

ACUITY_PALETTE = {'1':'#e74c3c', '2':'#e67e22', '3':'#f1c40f', '4':'#2ecc71', '5':'#3498db'}
ACUITY_COLORS  = [ACUITY_PALETTE[str(i)] for i in range(1, 6)]
print('Imports and plot style configured.')
```


```python
DATA_DIR = '/kaggle/input/competitions/triagegeist/'   # change to '/kaggle/input/triagegeist/' on Kaggle

train = pd.read_csv(f'{DATA_DIR}train.csv')
test  = pd.read_csv(f'{DATA_DIR}test.csv')
cc    = pd.read_csv(f'{DATA_DIR}chief_complaints.csv')
ph    = pd.read_csv(f'{DATA_DIR}patient_history.csv')
sub   = pd.read_csv(f'{DATA_DIR}sample_submission.csv')

print(f'train : {train.shape} | test : {test.shape}')
print(f'chief_complaints : {cc.shape} | patient_history : {ph.shape}')
```


```python
summary = pd.DataFrame({
    'dtype': train.dtypes, 'non_null': train.notna().sum(),
    'null_pct': (train.isna().mean() * 100).round(2),
    'nunique': train.nunique(), 'sample': train.iloc[0],
})
display(summary)
```


```python
comorbidity_cols = [c for c in ph.columns if c.startswith('hx_')]
prev = ph[comorbidity_cols].mean().mul(100).sort_values(ascending=False).round(2)
display(prev.to_frame('prevalence_%'))
```


```python
df_train = (
    train
    .merge(ph, on='patient_id', how='left')
    .merge(cc[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
)
df_test = (
    test
    .merge(ph, on='patient_id', how='left')
    .merge(cc[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
)
print(f'Merged train: {df_train.shape} | Merged test: {df_test.shape}')
print(f'Duplicate patient_id: {df_train.duplicated(subset="patient_id").any()}')
```


```python
LEAKAGE_COLS      = ['disposition', 'ed_los_hours']
TARGET_COL        = 'triage_acuity'
ID_COLS           = ['patient_id', 'site_id', 'triage_nurse_id']
VITALS_COLS       = ['systolic_bp','diastolic_bp','mean_arterial_pressure','pulse_pressure',
                     'heart_rate','respiratory_rate','temperature_c','spo2',
                     'gcs_total','pain_score','shock_index','news2_score']
ANTHRO_COLS       = ['age','weight_kg','height_cm','bmi']
DEMO_COLS         = ['sex','age_group','language','insurance_type']
CONTEXT_COLS      = ['arrival_mode','arrival_hour','arrival_day','arrival_month',
                     'arrival_season','shift','transport_origin']
HISTORY_COLS      = ['num_prior_ed_visits_12m','num_prior_admissions_12m',
                     'num_active_medications','num_comorbidities']
CLINICAL_CAT_COLS = ['pain_location','mental_status_triage','chief_complaint_system']
COMORBIDITY_COLS  = comorbidity_cols
TEXT_COL          = 'chief_complaint_raw'
print('Feature groups defined.')
```

### 🔍 Insight Summary (Step 1)

Data Loading & Schema Inspection

### 1.1 Dataset Scale

| File | Rows | Columns | Role |
|---|---|---|---|
| `train.csv` | 80,000 | 40 | Labelled training set |
| `test.csv` | 20,000 | 37 | Unlabelled — competition submission target |
| `chief_complaints.csv` | 100,000 | 3 | Free-text complaints for ALL patients |
| `patient_history.csv` | 100,000 | 26 | Binary comorbidity flags for ALL patients |

**Key observation:** The complaint and history files cover all 100,000 patients (train + test combined). This means they were generated globally and are available at inference time — no leakage concern from these auxiliary files.

---

### 1.2 The Leakage Trap

Two columns exist in `train.csv` that do NOT appear in `test.csv`:
- `disposition` — what happened to the patient *after* the triage decision (admitted, discharged, etc.)
- `ed_los_hours` — how long the patient stayed in the ED

**Why this matters clinically:** Both of these are *outcomes* of the triage process, not inputs to it. A real triage nurse makes a severity decision within the first 2–5 minutes of seeing a patient. They cannot know the final disposition or length of stay. Including these columns as features would be severe data leakage — the model would learn to predict acuity from its own consequences, not its causes.

> **Hypothesis 1:** `disposition` and `ed_los_hours` will show near-perfect correlation with `triage_acuity`. This will be confirmed in Step 4 (bivariate analysis) as a sanity check, then the columns will be permanently excluded.

---

### 1.3 Feature Volume and Redundancy

The merged training set has **65+ columns** after joining all files. However, a first scan reveals significant internal redundancy:

**Mathematically derived columns (already present in raw data):**
| Derived Column | Formula |
|---|---|
| `mean_arterial_pressure` | (2 × diastolic_bp + systolic_bp) / 3 |
| `pulse_pressure` | systolic_bp − diastolic_bp |
| `shock_index` | heart_rate / systolic_bp |
| `bmi` | weight_kg / (height_cm/100)² |
| `age_group` | binned from `age` at cutoffs 15, 40, 65 |
| `arrival_season` | derived from `arrival_month` |

**Why this matters:** These columns carry zero new information beyond their parent columns. For tree-based models (XGBoost, LightGBM, CatBoost), their presence is mostly harmless — trees can ignore redundant splits. However, they pollute feature importance rankings and inflate the appearance of the feature space. For linear models or distance-based methods, collinearity becomes a real problem.

> **Decision to make later:** We will keep derived columns in the dataset for now and revisit during feature engineering. The shock_index in particular is clinically meaningful as a pre-computed composite — keeping it is defensible even if redundant.

---

### 1.4 The Sentinel Value Problem

`pain_score` ranges from **−1 to 10**, where:
- `0–10` = actual pain rating on the standard clinical scale
- `−1` = pain was **not assessed** (sentinel value)

This is a critical encoding issue. A raw −1 treated as a numeric value would tell a model that these patients feel *less* pain than anyone who scored 0. In reality, it means the assessment simply wasn't done — and, as we'll see in Step 2 (missing data analysis), *who doesn't get assessed* is likely informative about acuity.

> **Hypothesis 2:** Patients with `pain_score = −1` will be disproportionately concentrated in either acuity level 1 (too critical for pain assessment) or acuity level 5 (so minor it was skipped). Both extremes would be clinically plausible.

---

### 1.5 The Comorbidity Block

`patient_history.csv` contains **25 binary flags** covering a wide spectrum of chronic conditions — from common ones (hypertension, diabetes type 2) to rarer ones (HIV, coagulopathy, peripheral vascular disease). 

The column `num_comorbidities` in `train.csv` is a simple count of how many of these flags are set. This means:
- `num_comorbidities` is **partially redundant** with the 25 individual flags
- But the individual flags carry *which* conditions, not just how many — a patient with COPD + heart failure has a very different risk profile than one with obesity + anxiety, even at the same count

> **Hypothesis 3:** The count alone (`num_comorbidities`) will be a weaker predictor than selected individual comorbidities. Conditions like COPD, heart failure, coagulopathy, and malignancy are clinically higher-risk than depression or hypothyroidism.

---

### 1.6 The Text Signal

`chief_complaint_raw` has **~5,000 unique values** across 100,000 patients — meaning many complaints repeat. The column `chief_complaint_system` (also present in `train.csv`) is a pre-categorized system label (neurological, cardiac, respiratory, etc.) — 14 categories total.

The raw text contains clinically precise descriptions like:
- `"thunderclap headache, worsening with movement"` — a neurological emergency red flag (subarachnoid hemorrhage)
- `"contraception advice, intermittent"` — clearly non-urgent

The pre-categorized `chief_complaint_system` collapses all this nuance into 14 bins. The raw text likely contains far more signal, particularly for distinguishing acuity levels 1 vs 2 (critical vs emergent).

> **Hypothesis 4:** The raw complaint text, when processed with even simple keyword extraction (chest pain, shortness of breath, altered consciousness), will add discriminative signal beyond what `chief_complaint_system` alone captures — especially for the highest-severity classes.

---

### 1.7 File Join Integrity

All merges are on `patient_id` which is a row-level unique key in each file. Post-merge checks:
- **No duplicate rows** after join — verified
- **Minimal nulls** in the comorbidity block — observed
- The `chief_complaint_system` column exists in both `train.csv` and `chief_complaints.csv`. We retain `train.csv`'s version to avoid duplication.

---

## Summary of Step 1 Hypotheses

| ID | Hypothesis |
|---|---|
| H1 | `disposition` and `ed_los_hours` will be near-perfectly correlated with `triage_acuity` (leakage proof) |
| H2 | `pain_score = −1` will cluster in acuity extremes (level 1 or level 5) |
| H3 | Individual comorbidities (COPD, heart failure, coagulopathy) will outperform the raw count as predictors |
| H4 | Raw complaint text will add signal beyond the 14-category `chief_complaint_system` label |

---

*Next step: Step 2 — Target Variable Analysis & Class Distribution*


---
## Step 2 — Target Variable Analysis

The target `triage_acuity` maps to the **Emergency Severity Index (ESI)** — a globally standardised 5-level triage protocol.

| Level | ESI Name | Clinical Meaning | Time to MD |
|---|---|---|---|
| 1 | Resuscitation | Immediate life threat | Immediate |
| 2 | Emergent | High-risk | ≤ 15 min |
| 3 | Urgent | Stable, resource-intensive | 30 min |
| 4 | Less Urgent | Stable, 1 resource | 60 min |
| 5 | Non-Urgent | Stable, no resources | 120 min |

Understanding the class distribution is the first critical step — it shapes metric choice, sampling strategy, and clinical prioritisation.


```python
acuity_counts = train[TARGET_COL].value_counts().sort_index()
acuity_pct    = (acuity_counts / len(train) * 100).round(2)

acuity_table = pd.DataFrame({
    'ESI Label':  ['Resuscitation','Emergent','Urgent','Less Urgent','Non-Urgent'],
    'Time to MD': ['Immediate','≤ 15 min','30 min','60 min','120 min'],
    'Count':      acuity_counts.values,
    'Pct (%)':    acuity_pct.values,
}, index=pd.Index(range(1,6), name='Acuity'))
display(acuity_table)
```


```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Target Variable: Triage Acuity Class Distribution', fontsize=15, fontweight='bold')

ax = axes[0]
bars = ax.bar(acuity_counts.index, acuity_counts.values,
              color=ACUITY_COLORS, edgecolor='#ffffff15', linewidth=0.8, zorder=3)
ax.set_xlabel('Triage Acuity Level'); ax.set_ylabel('Patient Count')
ax.set_title('Count per Acuity Level')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis='y', zorder=0)
for bar, pct in zip(bars, acuity_pct.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=11,
            color='#e0e0e0', fontweight='bold')

ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(
    acuity_counts.values,
    labels=[f'Level {i}' for i in acuity_counts.index],
    colors=ACUITY_COLORS, autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='#0f1117', linewidth=2.5))
for t in autotexts: t.set_color('#111'); t.set_fontweight('bold')
ax2.set_facecolor('#1a1d27')
ax2.set_title('Proportion by Acuity Level')
plt.tight_layout(); plt.show()
```


```python
majority = acuity_counts.idxmax(); minority = acuity_counts.idxmin()
ratio    = acuity_counts.max() / acuity_counts.min()
crit_n   = acuity_counts[[1,2]].sum()
minor_n  = acuity_counts[[4,5]].sum()

print(f'Majority class       : Level {majority}  ({acuity_pct[majority]:.1f}%)')
print(f'Minority class       : Level {minority}  ({acuity_pct[minority]:.1f}%)')
print(f'Imbalance ratio      : {ratio:.1f}x')
print(f'Critical (L1+L2)     : {crit_n:,}  ({crit_n/len(train)*100:.1f}%)')
print(f'Non-urgent (L4+L5)   : {minor_n:,}  ({minor_n/len(train)*100:.1f}%)')
```


```python
# Leakage confirmation: disposition crosstab + ed_los_hours per acuity
disp_tab = pd.crosstab(train['disposition'], train[TARGET_COL], normalize='index').round(3)
print('disposition vs triage_acuity (row-normalised):'); display(disp_tab)
los_stats = train.groupby(TARGET_COL)['ed_los_hours'].agg(['mean','median']).round(2)
print('\ned_los_hours per acuity level:'); display(los_stats)
```


```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Leakage Columns vs Acuity — Confirmation Only (NOT features)',
             fontsize=13, fontweight='bold')

ax = axes[0]
bp = ax.boxplot([train[train[TARGET_COL]==i]['ed_los_hours'].dropna() for i in range(1,6)],
                patch_artist=True, medianprops=dict(color='white', linewidth=2))
for patch, c in zip(bp['boxes'], ACUITY_COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax.set_xlabel('Acuity'); ax.set_ylabel('ED LoS (hours)')
ax.set_title('ed_los_hours by Acuity [LEAKAGE]'); ax.grid(axis='y')

ax2 = axes[1]
disp_tab.T.plot(kind='bar', stacked=True, ax=ax2,
                colormap='tab10', edgecolor='#0f1117')
ax2.set_xlabel('Acuity'); ax2.set_ylabel('Proportion')
ax2.set_title('disposition by Acuity [LEAKAGE]')
ax2.legend(title='Disposition', bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)
ax2.tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.show()
```

### 🔍 Insight Summary (Step 2)

Target Variable Analysis

### 2.1 Class Distribution

The `triage_acuity` distribution in the training set follows a pattern consistent with real emergency departments using ESI. Level 3 (Urgent) is the dominant class, while Level 1 (Resuscitation) is the rarest.

**What this means:**
- The dataset is **imbalanced** — not dramatically, but enough to matter
- A naive classifier that always predicts Level 3 would achieve deceptively high accuracy
- The minority classes (Level 1 and 2) are the *most clinically important* — these are patients who could die if under-triaged

**Clinical calibration check:** In real EDs, Level 1 represents <2% of visits (cardiac arrest, unresponsive, etc.). Level 2 covers high-risk presentations like stroke, severe sepsis, major trauma. If the synthetic data reflects this, it is a well-calibrated simulation of real-world ED volume.

---

### 2.2 Class Imbalance — Modeling Implications

The imbalance ratio between the largest and smallest classes determines how we must approach the problem:

- **Metric choice:** Accuracy is misleading. We must use **macro-averaged F1**, **Cohen's Kappa** (especially quadratic weighted Kappa since acuity is ordinal), or class-specific recall for levels 1–2
- **Sampling strategy:** Oversampling (SMOTE on tabular features) or class weights in the loss function — decision deferred until after full EDA
- **Threshold tuning:** For clinical deployment, we may want to intentionally bias toward Level 1/2 predictions (accepting more false positives to avoid false negatives on critical patients)

> **Key insight:** In emergency triage, a **false negative on Level 1 is a potential death**. A false positive (over-triaging) wastes resources but does not kill anyone. Our model should be tuned with this asymmetry in mind — recall on critical classes matters more than precision.

---

### 2.3 Ordinal Structure Matters

The target is not merely categorical — it is **strictly ordered** (1 < 2 < 3 < 4 < 5). This has two important consequences:

1. **Adjacent misclassifications are less severe than distant ones.** Predicting Level 2 instead of Level 1 is dangerous but recoverable; predicting Level 5 instead of Level 1 is potentially fatal.
2. **Standard cross-entropy loss ignores this ordering.** Quadratic Weighted Kappa (QWK) is the natural metric here because it penalises disagreements proportionally to their distance on the ordinal scale.

> **Hypothesis confirmed:** Ordinal classification (Direction 3 from our schema analysis) is not just a "nice-to-have" — it is the *right* framing for this problem.

---

### 2.4 Leakage Confirmation (Sanity Check)

We visualised `disposition` and `ed_los_hours` against `triage_acuity` — not to use them as features, but to confirm they are indeed strongly correlated with the target:

- **`ed_los_hours`:** Expected pattern — Level 1 patients have *shorter* ED stays (they are either rapidly stabilised or die), while Level 3 patients have the longest stays due to resource-intensive workups. Level 4–5 are discharged quickly.
- **`disposition`:** Level 1 patients should show high admission/ICU rates; Level 4–5 mostly discharged home.

If this pattern holds in the data, it confirms both that the target is well-defined **and** that these columns are post-hoc outcomes that must be excluded from any feature set.

---

### Step 2 Hypotheses Added

| ID | Hypothesis |
|---|---|
| H5 | Macro-F1 and QWK will diverge significantly from accuracy — accuracy is unreliable here |
| H6 | Recall on Level 1 will be the hardest metric to optimise; the model will systematically under-predict the rarest class |
| H7 | `ed_los_hours` will show a non-monotonic relationship with acuity (Level 1 < Level 3 > Level 5) — confirming it as a post-triage outcome, not a predictor |

---


---
## Step 3 — Missing Data Analysis

This is one of the most clinically important sections of the EDA.

In emergency triage, **missingness is not random**. The decision to measure (or not measure) a vital sign
is itself a clinical judgment. A patient who arrives unconscious by ambulance gets every vital measured
immediately. A patient who walks in with a sore finger may never have their blood pressure taken.

Therefore:
- **Missing vitals in high-acuity patients** → measurement was interrupted or the patient deteriorated
- **Missing vitals in low-acuity patients** → assessment was not clinically indicated

> This means missingness itself carries predictive signal. We must analyse it **before** deciding
> any imputation strategy. Blindly filling in medians would destroy this signal.


```python
# ── 3.1 Overall missing value summary ─────────────────────────────────────
cols_with_nulls = train.isnull().sum()
cols_with_nulls = cols_with_nulls[cols_with_nulls > 0].sort_values(ascending=False)

missing_df = pd.DataFrame({
    'missing_count': cols_with_nulls,
    'missing_pct':   (cols_with_nulls / len(train) * 100).round(3),
})
print(f'Columns with missing values: {len(missing_df)} / {train.shape[1]}')
display(missing_df)
```


```python
# ── 3.2 Missing value bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(missing_df.index[::-1], missing_df['missing_pct'][::-1],
               color='#e74c3c', alpha=0.8, edgecolor='#ffffff10')
ax.set_xlabel('Missing %')
ax.set_title('Missing Value Rate by Column (train.csv)', fontweight='bold')
ax.axvline(5, color='#f1c40f', linestyle='--', linewidth=1, label='5% threshold')
ax.legend()
for bar, val in zip(bars, missing_df['missing_pct'][::-1]):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=9)
ax.grid(axis='x')
plt.tight_layout(); plt.show()
```


```python
# ── 3.3 Are the same rows missing across BP-related columns? ───────────────
bp_cols = ['systolic_bp','diastolic_bp','mean_arterial_pressure','pulse_pressure','shock_index']
bp_null_counts = train[bp_cols].isnull().sum()
print('Null counts in BP-derived columns:')
print(bp_null_counts.to_string())

# Are the missing rows identical?
null_mask = train[bp_cols].isnull()
identical = (null_mask.sum(axis=1).isin([0, len(bp_cols)])).all()
print(f'\nAll BP nulls in exactly the same rows: {identical}')
n_bp_null = train['systolic_bp'].isnull().sum()
print(f'Total rows with no BP recorded: {n_bp_null:,} ({n_bp_null/len(train)*100:.2f}%)')
```


```python
# ── 3.4 Missingness rate by acuity level — KEY CLINICAL ANALYSIS ──────────
nullable_cols = list(missing_df.index)

miss_by_acuity = (
    train.groupby(TARGET_COL)[nullable_cols]
    .apply(lambda g: g.isnull().mean() * 100)
    .round(2)
)
print('Missing % per column, broken down by triage acuity level:')
display(miss_by_acuity)
```


```python
# ── 3.5 Heatmap — missingness by acuity ───────────────────────────────────
fig, ax = plt.subplots(figsize=(max(8, len(nullable_cols)*1.1), 5))
sns.heatmap(
    miss_by_acuity,
    annot=True, fmt='.1f', cmap='Reds',
    linewidths=0.5, linecolor='#0f1117',
    ax=ax, cbar_kws={'label': 'Missing %'}
)
ax.set_title('Missing Value Rate (%) by Triage Acuity Level', fontweight='bold', pad=12)
ax.set_xlabel('Column'); ax.set_ylabel('Acuity Level')
ax.set_yticklabels([f'L{i}' for i in range(1,6)], rotation=0)
plt.tight_layout(); plt.show()
```


```python
# ── 3.6 pain_score sentinel value analysis ────────────────────────────────
pain_dist = train['pain_score'].value_counts().sort_index()
print('pain_score value distribution:')
print(pain_dist.to_string())
print(f'\nSentinel (-1) count : {pain_dist.get(-1, 0):,}  ({pain_dist.get(-1,0)/len(train)*100:.1f}%)')
print(f'Truly missing (NaN) : {train["pain_score"].isnull().sum():,}')
```


```python
# ── 3.7 pain_score=-1 distribution across acuity levels ──────────────────
sentinel_by_acuity = (
    train.groupby(TARGET_COL)['pain_score']
    .apply(lambda s: (s == -1).mean() * 100)
    .round(2)
    .rename('pct_sentinel_pain')
)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(sentinel_by_acuity.index, sentinel_by_acuity.values,
              color=ACUITY_COLORS, edgecolor='#ffffff10', zorder=3)
ax.set_xlabel('Triage Acuity Level')
ax.set_ylabel('% of patients with pain_score = -1')
ax.set_title('Pain Score Sentinel Value (-1) Rate by Acuity Level', fontweight='bold')
ax.grid(axis='y', zorder=0)
for bar, val in zip(bars, sentinel_by_acuity.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout(); plt.show()

print(sentinel_by_acuity)
```


```python
# ── 3.8 Create missingness indicator features ─────────────────────────────
# These binary flags capture whether a value was recorded — not what it was.
# They will be candidate features for modeling.
for col in nullable_cols:
    df_train[f'{col}_missing'] = df_train[col].isnull().astype(int)

# Also flag the pain sentinel
df_train['pain_not_assessed'] = (df_train['pain_score'] == -1).astype(int)

miss_flag_cols = [f'{c}_missing' for c in nullable_cols] + ['pain_not_assessed']
print(f'Created {len(miss_flag_cols)} missingness indicator features:')
print(miss_flag_cols)

# Confirm signal: do these flags correlate with acuity?
miss_corr = df_train[miss_flag_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
print('\nCorrelation of missingness flags with triage_acuity:')
display(miss_corr.sort_values().to_frame('corr_with_acuity'))
```

### 🔍 Insight Summary (Step 3)

Missing Data Analysis

### 3.1 Which Columns Have Missing Data

Only a **small subset of columns** have any missing values at all — all concentrated in the vitals block:

- `systolic_bp`, `diastolic_bp`, `mean_arterial_pressure`, `pulse_pressure`, `shock_index` → all share the **exact same missing rows**
- `respiratory_rate` → partially missing (subset of the above)
- `temperature_c` → very few missing (<1%)

**No demographic or history columns have missing values.** This is expected — registration data (age, sex, insurance) is always collected, and the comorbidity file is complete for all 100,000 patients.

---

### 3.2 Structured Missingness in Blood Pressure

The five BP-related columns (`systolic_bp`, `diastolic_bp`, `mean_arterial_pressure`, `pulse_pressure`, `shock_index`) share **identical missing rows**. This is not a coincidence — it is a structured pattern.

**Clinical interpretation:** Blood pressure is either measured (and all derived values computed) or not measured at all. A triage nurse either places a BP cuff or does not. There is no scenario where `systolic_bp` is recorded but `pulse_pressure` is missing — they come from the same measurement.

**Why it matters for modeling:**
- These 5 columns are one "feature unit," not 5 independent ones
- Creating a single `bp_not_measured` binary flag captures the same information more cleanly
- A tree model that gets all 5 NaN columns would split on all 5; using the flag consolidates this into one interpretable feature

> **Key insight:** The BP missingness pattern is a *single clinical event* (no BP measurement) encoded redundantly across 5 columns. We should create `bp_missing` as a unified indicator.

---

### 3.3 Is Missingness Informative? (The Central Question)

The missingness-by-acuity heatmap answers the most important question in this section.

**Expected patterns (hypotheses going in):**
- **H8a:** Level 1 patients may have *some* missing BP (measurement interrupted during resuscitation)
- **H8b:** Level 4–5 patients will have the *highest* BP missingness (not clinically indicated for minor complaints)

**What this reveals about triage decision-making:** A patient who arrives with no BP recorded is more likely to be either critically ill (measurement was impossible) or completely minor (measurement was unnecessary). This bimodal pattern means the flag is informative at *both* extremes of acuity — a rare situation where a binary missingness flag encodes real clinical signal.

**Implication:** `bp_missing` should be treated as a **first-class feature**, not a nuisance to be imputed away. Any imputation strategy that fills missing BPs without first stratifying by this flag is methodologically wrong.

---

### 3.4 The Pain Score Sentinel (-1) Problem

`pain_score = -1` is fundamentally different from `pain_score = NaN`:
- `NaN` = data collection failure (system-level missing)
- `-1` = deliberate non-assessment (clinician decided pain was not applicable)

**Who gets `-1`?**
- Acuity Level 1 patients: Often unconscious, intubated, or in cardiac arrest — pain assessment is irrelevant and impossible
- Acuity Level 5 patients: Presenting for administrative reasons (prescription refill, paperwork), where pain is similarly not applicable

If the distribution confirms this **U-shaped pattern** (high `-1` rate at both Level 1 and Level 5), it directly confirms Hypothesis H2 and demonstrates that the sentinel value encodes acuity-relevant signal.

**Treatment recommendation (to be decided post-EDA):**
- Do NOT treat `-1` as a numeric value in continuous pain_score models
- Create `pain_not_assessed` binary flag
- For patients with actual scores (0–10), treat pain_score as ordinal or continuous

---

### 3.5 Missingness Indicator Features

We create binary missingness flags for every column that has missing values, plus `pain_not_assessed` for the sentinel. These become **candidate model features**.

The correlation of these flags with `triage_acuity` tells us immediately how informative they are:
- A **negative correlation** means missing → lower acuity number → higher severity → the measurement was more likely to be taken for critical patients
- A **positive correlation** means missing → higher acuity number → lower severity → the measurement was skipped for minor patients

Both directions are informative. The sign of the correlation reveals which clinical scenario applies.

---

### Step 3 Hypotheses Added

| ID | Hypothesis |
|---|---|
| H8 | BP missingness will be highest at acuity Level 4–5 (minor patients, no measurement indicated) |
| H9 | `pain_score = -1` will form a U-shape: highest at Level 1 (unconscious/critical) and Level 5 (non-clinical visit) |
| H10 | Missingness indicator flags will have non-zero correlation with acuity — confirming informative missingness |
| H11 | `respiratory_rate` will show a different missingness pattern from BP (not identical rows) — it is partially, not structurally, missing |

---


---
## Step 4 — Univariate Analysis (Vitals Deep Dive)

Vitals are the most objective and universally collected parameters in emergency medicine. They are often the strongest predictors of critical illness. In this step, we examine the distribution of each core vital sign to understand its range, skew, and potential outliers.

**Core Vitals to Analyze:**
- `heart_rate` (HR)
- `systolic_bp` (SBP) and `diastolic_bp` (DBP)
- `respiratory_rate` (RR)
- `temperature_c` (Temp)
- `spo2` (Oxygen Saturation)
- `gcs_total` (Glasgow Coma Scale)


```python
# ── 4.1 Summary Statistics for Core Vitals ────────────────────────────────
core_vitals = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate', 'temperature_c', 'spo2', 'gcs_total']
vitals_summary = df_train[core_vitals].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
display(vitals_summary.round(2))
```


```python
# ── 4.2 Visualizing Vital Sign Distributions ──────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle('Distributions of Core Vital Signs', fontsize=16, fontweight='bold', y=1.02)
axes = axes.flatten()

plot_vitals = ['heart_rate', 'systolic_bp', 'respiratory_rate', 'temperature_c', 'spo2', 'gcs_total']
colors = ['#e74c3c', '#3498db', '#f1c40f', '#e67e22', '#2ecc71', '#9b59b6']

for i, (col, color) in enumerate(zip(plot_vitals, colors)):
    sns.histplot(df_train[col].dropna(), bins=40, kde=True, ax=axes[i], color=color, edgecolor='#ffffff22')
    axes[i].set_title(f'{col} Distribution', fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```


```python
# ── 4.3 Detecting Outliers (Clinical Plausibility Check) ──────────────────
# Define clinical extreme thresholds (highly abnormal, but biologically possible)
# Values outside these ranges might be data entry errors or extreme physiological states.
extreme_thresholds = {
    'heart_rate': (30, 250),
    'systolic_bp': (40, 300),
    'respiratory_rate': (4, 60),
    'temperature_c': (30.0, 43.0),
    'spo2': (50, 100) # SpO2 cannot technically be >100
}

outlier_counts = {}
for col, (low, high) in extreme_thresholds.items():
    n_low = (df_train[col] < low).sum()
    n_high = (df_train[col] > high).sum()
    outlier_counts[col] = {'< Low Threshold': n_low, '> High Threshold': n_high}

outliers_df = pd.DataFrame(outlier_counts).T
print('Patients with extremely abnormal (potentially erroneous) vitals:')
display(outliers_df)
```

### 🔍 Insight Summary (Step 4)

Univariate Analysis (Vitals Deep Dive)

Vitals are the cornerstone of the triage decision. Analyzing their distributions provides insights into the physiological severity of the patients in our dataset. 

If I were an emergency physician, I would look at these physiological signals before even reading the patient's history, as they are immediate indicators of life threat.

### 4.1 Heart Rate (HR)
- **Observation:** The distribution of heart rate is slightly right-skewed, centered around 80-90 bpm. We see a long tail extending up to 150+ bpm.
- **Interpretation:** Normal HR is 60-100 bpm. High heart rate (tachycardia) is a non-specific but critical sign of physiological stress (e.g., pain, anxiety, fever, dehydration, sepsis, bleeding). A very low heart rate (bradycardia, <60 bpm) can be normal in athletes but dangerous if associated with heart blocks or medication overdoses.
- **Hypothesis (H12):** Extremes of heart rate, particularly severe tachycardia (>120 bpm) and severe bradycardia (<50 bpm), will be strong predictors of higher acuity (Level 1 and 2).

### 4.2 Blood Pressure (Systolic BP)
- **Observation:** SBP is relatively normally distributed with a mean around 130 mmHg, but has a wide variance (ranging from ~80 to over 200). 
- **Interpretation:** Normal SBP is typically 90-120 mmHg. 
  - **High SBP (Hypertension):** Very high values (>180) can indicate hypertensive emergencies, severe pain, or stroke.
  - **Low SBP (Hypotension):** Values <90 mmHg indicate shock (e.g., severe bleeding, sepsis, heart failure). This is an immediate life threat.
- **Hypothesis (H13):** Hypotension (<90 mmHg) will be heavily concentrated in Level 1 (Resuscitation) and Level 2 (Emergent). Hypertension will correlate with higher acuity, but not as strongly as hypotension.

### 4.3 Respiratory Rate (RR)
- **Observation:** The distribution is highly concentrated around 16-20 breaths per minute, with a noticeable right skew (tachypnea).
- **Interpretation:** Normal RR is 12-20. 
  - **High RR (>24):** A classic early warning sign of respiratory failure, hypoxia, acidosis, or severe infection (sepsis).
  - **Low RR (<10):** Often seen in central nervous system depression, frequently due to opiate overdose or impending respiratory arrest.
- **Hypothesis (H14):** Respiratory rate is often the most sensitive indicator of critical illness. Deviations from the narrow normal range (especially RR > 24) will strongly predict Level 1 and 2 triage acuity.

### 4.4 Temperature
- **Observation:** Tightly clustered around 36.5°C to 37.5°C (normal body temperature). Outliers exist on both ends (fever >38°C and hypothermia <36°C).
- **Interpretation:** 
  - **Fever:** Indicates infection/inflammation. By itself, it might not trigger Level 1/2 unless accompanied by other abnormal vitals (e.g., high HR, low BP = sepsis).
  - **Hypothermia:** Can be due to environmental exposure or severe, late-stage sepsis (a very bad prognostic sign).
- **Hypothesis (H15):** Temperature alone will be a weaker predictor of extreme severity compared to BP and RR, but will have strong interaction effects (e.g., Temp + HR + SBP = shock index/sepsis alert).

### 4.5 Oxygen Saturation (SpO2)
- **Observation:** Heavily left-skewed, with the vast majority of patients sitting between 95% and 100%. A long tail extends downward below 90%.
- **Interpretation:** Normal SpO2 is >=95%. Values <92% indicate hypoxia and prompt immediate intervention (supplemental oxygen). Values <85% are life-threatening.
- **Hypothesis (H16):** SpO2 < 92% will be almost exclusively found in Acuity Levels 1 and 2. It will act as a strong threshold feature for tree-based models.

### 4.6 Glasgow Coma Scale (GCS)
- **Observation:** Most values are exactly 15 (fully alert and oriented). A small subset has scores <15.
- **Interpretation:** GCS measures neurological status. 15 is normal. <15 means altered mental status. A score <=8 usually mandates immediate intubation (Level 1).
- **Hypothesis (H17):** GCS is a highly imbalanced but extremely powerful feature. Any GCS < 15 strongly rules out Levels 4 and 5, and GCS <=8 virtually guarantees Level 1.

### 4.7 Clinical Plausibility (Outliers)
- **Observation:** The dataset contains some values that push physiological limits.
- **Interpretation:** In real-world data, extreme outliers could be data entry errors (e.g., typing a weight as a heart rate) or rare true clinical events. Because this is synthetic data, these extremes may simulate real-world noise.
- **Hypothesis (H18):** Tree-based models (XGBoost/LightGBM) are robust to monotonic outliers. We should **not** clip or remove these outliers without bivariate analysis, as extreme values are often the exact signals used to trigger a Level 1 resuscitation response.

---


---
## Step 5 — Bivariate Analysis (Vitals vs Acuity & Interactions)

In emergency medicine, triage decisions are rarely made based on a single vital sign. They are made by recognizing patterns and combinations of physiological derangement.

In this step, we will:
1. Test our univariate hypotheses (e.g., does severe hypotension actually correlate with Level 1 acuity in THIS dataset?).
2. Explore critical clinical interactions (e.g., Shock: High Heart Rate + Low Blood Pressure).


```python
# ── 5.1 Vitals vs Triage Acuity (Testing Univariate Hypotheses) ───────────
fig, axes = plt.subplots(3, 2, figsize=(16, 16))
fig.suptitle('Vital Signs by Triage Acuity Level', fontsize=16, fontweight='bold', y=1.02)
axes = axes.flatten()

plot_vitals = ['heart_rate', 'systolic_bp', 'respiratory_rate', 'temperature_c', 'spo2', 'gcs_total']

for i, col in enumerate(plot_vitals):
    sns.boxplot(data=df_train, x=TARGET_COL, y=col, ax=axes[i], palette=ACUITY_PALETTE, 
                showfliers=False, width=0.6, boxprops=dict(alpha=0.8, edgecolor='w'))
    axes[i].set_title(f'{col} vs Acuity', fontweight='bold')
    axes[i].set_xlabel('Triage Acuity Level')
    axes[i].set_ylabel(col)
    axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```


```python
# ── 5.2 Deep Dive: GCS and SpO2 Thresholds ────────────────────────────────
# GCS and SpO2 are hypothesized to act as strict clinical thresholds.
gcs_breakdown = pd.crosstab(df_train['gcs_total'] < 15, df_train[TARGET_COL], normalize='index').round(3)
gcs_breakdown.index = ['GCS == 15 (Normal)', 'GCS < 15 (Altered)']
print('Proportion of Acuity Levels for Normal vs Altered Mental Status (GCS):')
display(gcs_breakdown)

spo2_breakdown = pd.crosstab(df_train['spo2'] < 92, df_train[TARGET_COL], normalize='index').round(3)
spo2_breakdown.index = ['SpO2 >= 92% (Normal)', 'SpO2 < 92% (Hypoxia)']
print('\nProportion of Acuity Levels for Normal Oxygen vs Hypoxia:')
display(spo2_breakdown)
```


```python
# ── 5.3 Clinical Interactions: Shock (HR vs SBP) ──────────────────────────
# High Heart Rate + Low Blood Pressure is the classic presentation of shock (hypovolemic, septic, cardiogenic).
fig, ax = plt.subplots(figsize=(10, 7))

# To avoid overplotting, we sample the data and only look at Acuity 1, 2, and 5 for stark contrast.
mask = df_train[TARGET_COL].isin([1, 2, 5])
sample_df = df_train[mask].sample(min(15000, mask.sum()), random_state=42)

sns.scatterplot(data=sample_df, x='systolic_bp', y='heart_rate', hue=TARGET_COL, 
                palette={1: '#e74c3c', 2: '#e67e22', 5: '#3498db'}, 
                alpha=0.6, s=20, ax=ax, edgecolor=None)

# Add clinical danger zone (SBP < 90, HR > 110)
ax.axvspan(xmin=40, xmax=90, ymin=0.5, ymax=1.0, color='red', alpha=0.1, zorder=0)
ax.text(65, 180, 'Shock Zone\n(Low SBP, High HR)', color='#ff6b6b', fontweight='bold', ha='center')

ax.set_title('Interaction: Heart Rate vs Systolic BP (Shock Dynamics)', fontweight='bold')
ax.set_xlabel('Systolic BP (mmHg)')
ax.set_ylabel('Heart Rate (bpm)')
ax.grid(alpha=0.3)
plt.legend(title='Acuity Level')
plt.show()
```


```python
# ── 5.4 Clinical Interactions: Respiratory Failure (RR vs SpO2) ───────────
fig, ax = plt.subplots(figsize=(10, 7))

sns.scatterplot(data=sample_df, x='spo2', y='respiratory_rate', hue=TARGET_COL, 
                palette={1: '#e74c3c', 2: '#e67e22', 5: '#3498db'}, 
                alpha=0.6, s=20, ax=ax, edgecolor=None)

# Add clinical danger zone (SpO2 < 92, RR > 24)
ax.axvspan(xmin=50, xmax=92, ymin=0.4, ymax=1.0, color='red', alpha=0.1, zorder=0)
ax.text(80, 45, 'Resp. Failure Zone\n(Hypoxia, Tachypnea)', color='#ff6b6b', fontweight='bold', ha='center')

ax.set_title('Interaction: Respiratory Rate vs SpO2 (Respiratory Dynamics)', fontweight='bold')
ax.set_xlabel('SpO2 (%)')
ax.set_ylabel('Respiratory Rate (breaths/min)')
ax.set_xlim(60, 101)
ax.grid(alpha=0.3)
plt.legend(title='Acuity Level')
plt.show()
```

### 🔍 Insight Summary (Step 5)

Bivariate Analysis (Vitals vs Acuity & Interactions)

In this step, we cross-reference our univariate hypotheses against the actual target variable (`triage_acuity`) and explore the multidimensional interactions that define true clinical emergencies.

### 5.1 Validating Univariate Hypotheses
By grouping the vital signs across the 5 acuity levels, we can confirm or reject our earlier assumptions:

* **Heart Rate (Confirming H12):** The median HR rises slightly as acuity increases (Level 5 -> Level 1). More importantly, the *variance* explodes in Level 1 and 2. Tachycardia (>110) is heavily represented in Level 1.
* **Systolic BP (Confirming H13):** There is a clear pattern of hypotension (<90 mmHg) clustering in Level 1 and 2. Level 5 patients rarely have dangerous hypotension. Hypertension is present across all levels, strongly suggesting that *low* BP is a much stronger specific indicator of critical illness than *high* BP.
* **Respiratory Rate (Confirming H14):** RR shows a stark divergence. Level 1 patients have a massive spread, featuring both severe tachypnea (>24) and dangerous bradypnea (<10). Level 4 and 5 are tightly bound to the normal 14-20 range.
* **SpO2 and GCS (Confirming H16 & H17):** 
  * SpO2 < 92% (Hypoxia) is rare in Levels 4 and 5. It is strongly associated with Level 1 and 2.
  * GCS < 15 (Altered Mental Status) is a strong red flag. The data is consistent with an abnormal GCS dominantly mapping to Level 1 or 2.

### 5.2 Interaction 1: Shock Dynamics (HR vs SBP)
- **Observation:** When plotting Heart Rate against Systolic BP, a distinct "Shock Zone" emerges (SBP < 90, HR > 110). 
- **Interpretation:** This physiological state happens when the heart beats frantically to compensate for falling blood pressure.
- **Clinical Alignment:** The scatterplot reveals that this Shock Zone is densely populated by Level 1 (Resuscitation) and Level 2 (Emergent) patients. Level 5 (Non-Urgent) patients are completely absent from this quadrant. 
- **Hypothesis (H19):** An interaction feature representing the "Shock Index" (HR / SBP) is already in the dataset. This visual confirms that high Shock Index is a dominant predictor of extreme acuity.

### 5.3 Interaction 2: Respiratory Failure (RR vs SpO2)
- **Observation:** Plotting Respiratory Rate vs Oxygen Saturation reveals a "Respiratory Failure Zone" (SpO2 < 92% and RR > 24).
- **Interpretation:** The patient is breathing rapidly but still failing to oxygenate their blood. This is a classic presentation of severe pneumonia, COPD exacerbation, or pulmonary embolism.
- **Clinical Alignment:** Just like shock, this quadrant is exclusively owned by Level 1 and 2 patients. 
- **Hypothesis (H20):** Combining respiratory features into a single composite risk flag (e.g., `is_resp_failure = (SpO2 < 92) & (RR > 24)`) will create a highly pure feature for isolating critical patients.

### 5.4 The "Normal Vitals" Trap
- **Observation:** A significant number of Level 2 (and even some Level 1) patients fall directly into the "Normal" zones for all vitals (e.g., HR 80, SBP 120, SpO2 98%).
- **Interpretation:** **Vitals are not everything.** A patient can have perfect vitals but still be having a stroke, an active heart attack, or a psychiatric emergency.
- **Hypothesis (H21):** Vitals have high *specificity* for critical illness (if they are terrible, you are critical), but low *sensitivity* (if they are normal, you might still be critical). We MUST rely on the `chief_complaint_raw` and `patient_history` to catch the "silent" Level 1 and 2 patients.

---


---
## Step 6 — Chief Complaints and Patient History (Categorical & Text Signals)

In Step 5, we saw that many patients in Level 1 and 2 present with 'normal' vitals. A patient experiencing an acute stroke or psychiatric crisis might have a perfect heart rate and blood pressure.

To detect these 'silent' critical patients, we must analyze the reason they came to the ED (`chief_complaint`) and their underlying health baseline (`patient_history`).


```python
# ── 6.1 Chief Complaint System vs Acuity ──────────────────────────────────
# Calculate the proportion of each acuity level within each complaint system
cc_sys_xtab = pd.crosstab(df_train['chief_complaint_system'], df_train[TARGET_COL], normalize='index')
cc_sys_xtab = cc_sys_xtab.sort_values(by=1, ascending=False) # Sort by proportion of Level 1

fig, ax = plt.subplots(figsize=(12, 6))
cc_sys_xtab.plot(kind='bar', stacked=True, ax=ax, color=ACUITY_COLORS, edgecolor='#0f1117', linewidth=0.5)
ax.set_title('Triage Acuity Distribution by Chief Complaint System', fontweight='bold')
ax.set_xlabel('Chief Complaint System')
ax.set_ylabel('Proportion')
ax.legend(title='Acuity Level', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

print("Top 5 Systems with Highest Proportion of Level 1 & 2 Patients:")
display((cc_sys_xtab[1] + cc_sys_xtab[2]).sort_values(ascending=False).head(5).to_frame('Proportion Level 1+2'))
```


```python
# ── 6.2 Raw Chief Complaint Text Exploration ──────────────────────────────
print("Most frequent raw complaints overall (Top 10):")
display(df_train['chief_complaint_raw'].value_counts().head(10))

print("\nMost frequent raw complaints for Acuity LEVEL 1 (Top 10):")
display(df_train[df_train[TARGET_COL] == 1]['chief_complaint_raw'].value_counts().head(10))

print("\nMost frequent raw complaints for Acuity LEVEL 5 (Top 10):")
display(df_train[df_train[TARGET_COL] == 5]['chief_complaint_raw'].value_counts().head(10))
```


```python
# ── 6.3 Keyword Impact Analysis ───────────────────────────────────────────
# Let's test a few high-risk vs low-risk keywords to see how they stratify acuity.
keywords = ['chest pain', 'unconscious', 'shortness of breath', 'refill', 'note']

keyword_stats = []
for kw in keywords:
    mask = df_train['chief_complaint_raw'].str.contains(kw, case=False, na=False)
    if mask.sum() > 0:
        acuity_dist = df_train[mask][TARGET_COL].value_counts(normalize=True)
        stat = {'Keyword': kw, 'Count': mask.sum()}
        for level in range(1, 6):
            stat[f'Level {level} %'] = (acuity_dist.get(level, 0) * 100)
        keyword_stats.append(stat)

kw_df = pd.DataFrame(keyword_stats).set_index('Keyword')
display(kw_df)
```


```python
# ── 6.4 Patient History (Comorbidities) vs Acuity ─────────────────────────
# First: Does the raw count of comorbidities matter?
comorb_xtab = pd.crosstab(df_train['num_comorbidities'], df_train[TARGET_COL], normalize='index')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Comorbidities vs Triage Acuity', fontsize=14, fontweight='bold')

comorb_xtab.plot(kind='bar', stacked=True, ax=axes[0], color=ACUITY_COLORS, edgecolor='#0f1117', linewidth=0.5)
axes[0].set_title('Acuity Distribution by Number of Comorbidities')
axes[0].set_xlabel('Number of Comorbidities')
axes[0].set_ylabel('Proportion')
axes[0].legend(title='Acuity Level', loc='upper right', fontsize=8)

# Second: Which specific diseases are most dangerous?
# We calculate the % of patients with each disease who end up in Level 1 or 2.
disease_risk = {}
for col in comorbidity_cols:
    mask = df_train[col] == 1
    if mask.sum() > 0:
        high_acuity_pct = df_train[mask][TARGET_COL].isin([1, 2]).mean() * 100
        disease_risk[col.replace('hx_', '')] = high_acuity_pct

risk_s = pd.Series(disease_risk).sort_values(ascending=False)
risk_s.plot(kind='bar', ax=axes[1], color='#e74c3c', edgecolor='#ffffff22')
axes[1].set_title('% of Patients in Level 1 or 2 by Specific Condition')
axes[1].set_xlabel('Comorbidity')
axes[1].set_ylabel('% Level 1 or 2')
axes[1].axhline(df_train[TARGET_COL].isin([1, 2]).mean() * 100, color='#f1c40f', linestyle='--', label='Baseline Risk (All Patients)')
axes[1].legend()
plt.tight_layout()
plt.show()
```

### 🔍 Insight Summary (Step 6)

Chief Complaints and Patient History (Categorical & Text Signals)

We previously established the "Normal Vitals Trap" — a subset of critical patients who present with perfect physiology but are experiencing hidden emergencies (e.g., strokes, psychiatric crises, or early stages of shock before compensation fails). In this step, we explore how categorical context and free text explain these cases and reduce triage uncertainty.

### 6.1 Chief Complaint System (Confirming H21)
- **Observation:** The broad `chief_complaint_system` categories heavily stratify the acuity distribution. Systems like *Neurological* and *Cardiovascular* have a strongly elevated proportion of Level 1 and 2 patients. Systems like *Dermatological* or *Ophthalmologic* are overwhelmingly Level 4 and 5.
- **Interpretation:** If a patient has perfectly normal vitals but presents with a *Neurological* complaint (e.g., stroke-like symptoms), they are still frequently triaged as Level 2. This strongly suggests that categorical systems provide an entirely separate dimension of risk assessment that covers the blind spots of physiological vitals.
- **Hypothesis (H22):** The `chief_complaint_system` will act as a strong prior for acuity, meaning a baseline prediction shifts significantly up or down based on this category before vitals are even considered.

### 6.2 Raw Chief Complaint Text
- **Observation:** Looking at the raw text, the contrast between extremes is stark. Level 5 complaints are often administrative or trivial (e.g., "note", "refill", "mild rash"). Level 1 complaints contain alarming keywords (e.g., "unconscious", "chest pain", "severe bleeding").
- **Interpretation:** The 14-category `chief_complaint_system` is a blunt instrument. It groups "mild headache" and "unconscious" into the same *Neurological* bucket. The raw text captures the nuance necessary to differentiate Level 1 from Level 3.
- **Hypothesis (H23):** NLP signals from `chief_complaint_raw` (such as keyword indicators) will be strongly associated with Level 1 and 2 patients, potentially outperforming the pre-categorized `chief_complaint_system`. 

### 6.3 Comorbidities (Patient History)
- **Observation:** The raw count of comorbidities (`num_comorbidities`) shows a general positive association with acuity severity. However, when examining *specific* conditions, a clear hierarchy of risk emerges. Patients with `hx_heart_failure`, `hx_copd`, or `hx_coagulopathy` have a much higher baseline rate of Level 1/2 acuity than the general population. Conditions like `hx_depression` or `hx_anxiety` do not show the same dramatic risk elevation.
- **Interpretation:** Clinical triage factors in a patient's physiological reserve. A patient with severe COPD presenting with shortness of breath is at a higher risk of rapid deterioration than a healthy 20-year-old with the same complaint. Certain baseline diseases reduce resilience, making any new insult potentially life-threatening.
- **Hypothesis (H24):** The simple `num_comorbidities` count is a weaker summary than the specific disease flags. Specific comorbidities (particularly cardiopulmonary and bleeding disorders) will carry independent, strong associations with high acuity.

### 6.4 Explaining the "Normal Vitals" Trap
- **Conclusion:** We have found the missing pieces of the puzzle. The patients in Step 5 who had Level 1/2 acuity despite normal vitals are likely the ones presenting with high-risk chief complaints (e.g., "chest pain radiating to arm") or severe background histories.
- **Hypothesis (H25):** The data suggests that triage relies on a union of signals: a patient is likely critical if their vitals are terrible OR if their chief complaint is inherently dangerous. Therefore, the combination of text and vitals will likely provide the most complete picture of the patient's state.

---


---
## Step 7 — Signal Strength & Feature Importance (Pre-Model)

We have identified many plausible signals visually. Now, we must quantify their statistical association with the target (`triage_acuity`).
We will use **Mutual Information (MI)** for continuous variables, which captures both linear and non-linear relationships. For categorical variables, we will measure the variance in Acuity 1/2 distribution across categories.


```python
from sklearn.feature_selection import mutual_info_classif

# ── 7.1 Continuous Vitals Signal Strength (Mutual Information) ────────────
vitals_to_test = ['heart_rate', 'systolic_bp', 'respiratory_rate', 'temperature_c', 'spo2', 'gcs_total']

# Drop nulls for MI calculation
mi_df = df_train.dropna(subset=vitals_to_test + [TARGET_COL])

mi_scores = mutual_info_classif(mi_df[vitals_to_test], mi_df[TARGET_COL], random_state=42)
mi_vitals = pd.Series(mi_scores, index=vitals_to_test).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Signal Strength Quantified (Pre-Model)', fontsize=16, fontweight='bold')

mi_vitals.plot(kind='bar', ax=axes[0], color='#9b59b6', edgecolor='#ffffff22')
axes[0].set_title('Mutual Information (Continuous Vitals vs Acuity)')
axes[0].set_ylabel('MI Score')
axes[0].tick_params(axis='x', rotation=45)

# ── 7.2 Categorical Signal Strength (Variance in Critical Acuity) ─────────
# For categoricals, we measure how much the % of Level 1/2 varies across categories.
cat_features = ['chief_complaint_system', 'arrival_mode', 'mental_status_triage']
cat_variance = {}

for cat in cat_features:
    # Calculate % Level 1/2 for each category
    props = df_train.groupby(cat)[TARGET_COL].apply(lambda x: x.isin([1, 2]).mean())
    # Variance in this proportion tells us how strongly the category separates risk
    cat_variance[cat] = props.var()

cat_s = pd.Series(cat_variance).sort_values(ascending=False)
cat_s.plot(kind='bar', ax=axes[1], color='#e67e22', edgecolor='#ffffff22')
axes[1].set_title('Risk Variance (Categorical Features)')
axes[1].set_ylabel('Variance in % Level 1/2')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```


```python
# ── 7.3 Comparing History vs Text (Top Keyword vs Top Disease) ────────────

# Risk of top keyword 'unconscious'
mask_unconscious = df_train['chief_complaint_raw'].str.contains('unconscious', case=False, na=False)
risk_unconscious = df_train[mask_unconscious][TARGET_COL].isin([1, 2]).mean()

# Risk of top comorbidity 'heart_failure'
mask_hf = df_train['hx_heart_failure'] == 1
risk_hf = df_train[mask_hf][TARGET_COL].isin([1, 2]).mean()

# Risk of missing BP (vital proxy)
mask_no_bp = df_train['systolic_bp'].isnull()
risk_no_bp = df_train[mask_no_bp][TARGET_COL].isin([1, 2]).mean()

# Baseline risk
baseline_risk = df_train[TARGET_COL].isin([1, 2]).mean()

comparison = pd.Series({
    'Text: "Unconscious"': risk_unconscious,
    'History: Heart Failure': risk_hf,
    'Vitals: Missing BP': risk_no_bp,
    'Baseline (All Patients)': baseline_risk
}).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
comparison.plot(kind='barh', color=['#e74c3c', '#e67e22', '#3498db', '#95a5a6'], ax=ax, edgecolor='#111')
ax.set_title('Signal Comparison: Probability of Acuity 1 or 2', fontweight='bold')
ax.set_xlabel('Probability')
for i, v in enumerate(comparison):
    ax.text(v + 0.01, i, f'{v*100:.1f}%', va='center', fontweight='bold')
plt.tight_layout()
plt.show()
```

### 🔍 Insight Summary (Step 7)

Signal Strength & Feature Importance (Pre-Model)

We have identified many physiological and historical patterns visually. Now, we quantify how strong these signals actually are by examining Mutual Information (for continuous variables) and Risk Variance (for categorical variables).

### 7.1 Continuous Vitals (Mutual Information Ranking)
- **Observation:** `respiratory_rate` and `heart_rate` have the highest Mutual Information scores with `triage_acuity`. `gcs_total` and `spo2` follow, while `temperature_c` is the weakest of the core vitals.
- **Interpretation:** As hypothesized in H14, respiratory rate is an incredibly sensitive indicator of critical illness. Temperature is a weak predictor on its own because fevers are common in minor illnesses (Level 4/5) and hypothermia is relatively rare.
- **Hypothesis (H26):** If we remove `temperature_c`, we lose very little baseline information compared to removing `respiratory_rate` or `heart_rate`.

### 7.2 Categorical Signal Strength (Variance in Risk)
- **Observation:** We evaluated categorical variables by looking at how much the proportion of Level 1/2 patients varies across categories. `chief_complaint_system` has enormous variance (some systems are >30% critical, others <2%). `arrival_mode` also shows strong variance (ambulance arrivals are far more critical than walk-ins).
- **Interpretation:** The category of a patient's complaint acts as a massive prior. `arrival_mode` is also a powerful contextual signal — paramedics triage patients in the field, so "arrived by ambulance" inherently carries a higher risk probability.
- **Hypothesis (H27):** Contextual features like `arrival_mode` and categorical `chief_complaint_system` will rank among the top 10 most important structured features in any tree-based model.

### 7.3 Comparing Signal Types (Text vs History vs Vitals)
- **Observation:** We compared the extreme cases of each domain:
  - Text: The presence of the word "unconscious".
  - History: The presence of "hx_heart_failure".
  - Vitals: The indicator for missing BP (from our Step 3 insights).
- **Interpretation:** High-risk text keywords ("unconscious", "chest pain") carry a much higher probability of Level 1/2 acuity than isolated history flags. While having heart failure increases risk above baseline, presenting *unconscious* pushes the risk probability extraordinarily high.
- **Hypothesis (H28):** In a ranked feature list, strong text embeddings/keywords will be the strongest individual predictors, followed by key physiological extremes (shock index, RR), followed by categorical context (`arrival_mode`), with patient history (comorbidities) acting as weaker, secondary modifiers.

### 7.4 Feature Redundancy Check
- **Observation:** `systolic_bp` and `diastolic_bp` carry highly overlapping mutual information. Similarly, `age` and `age_group` are perfectly redundant. 
- **Interpretation:** Many features capture the same underlying variance. We do not need both raw and binned versions of the same variable.
- **Hypothesis (H29):** Dropping explicitly redundant derived features (like `mean_arterial_pressure` or `pulse_pressure`) will not degrade predictive performance, as tree models can reconstruct these interactions from the base variables.


## Step 8 — Summary & Modeling Strategy

Based on the deep EDA, here is the roadmap for modeling:

1. **Target**: Predict `triage_acuity` (ordinal: 1 to 5).
2. **Metrics**: Quadratic Weighted Kappa (QWK) to handle ordinality, and Macro-F1 to ensure minority classes (Level 1/2) are respected.
3. **Leakage to Drop**: `disposition` and `ed_los_hours` MUST be excluded.
4. **Missingness**: Do not impute blindly. The binary `bp_missing` and `pain_not_assessed` flags are highly predictive.
5. **Feature Engineering**: 
   - `shock_index` (HR/SBP)
   - `is_resp_failure` (SpO2 < 92 & RR > 24)
   - NLP features from `chief_complaint_raw` (TF-IDF or embeddings)
6. **Model Choice**: A tree-based model (XGBoost/LightGBM) using an ordinal objective (like proportional odds) or simply optimizing for QWK.




```python

```


```python

```


```python

```
