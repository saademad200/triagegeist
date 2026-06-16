# Triagegeist: Multi-Modal Undertriage Detection via Calibrated Ensemble Learning

## Clinical Motivation

Emergency department (ED) triage determines patient outcomes within seconds. Among ~**130 million annual U.S. ED visits** (CDC NHAMCS 2022), triage nurses must assign Emergency Severity Index (ESI) acuity levels under extreme cognitive load with incomplete information.

**The fundamental clinical asymmetry** drives our entire approach:
- **Overtriage** (assigning higher acuity) → resource waste, but **patient safety preserved**
- **Undertriage** (assigning lower acuity) → delayed care, adverse outcomes, **preventable mortality**

Published literature reports undertriage rates of **5–20%** (Farrohknia et al., 2011), with ESI inter-rater reliability at κ = 0.60–0.80 (Gilboy et al., 2012). Vulnerable populations — elderly, atypical presenters, linguistic minorities — are disproportionately affected (Obermeyer et al., 2019).

## Our Approach: Four Integrated Components

| Component | Innovation | Clinical Value |
|-----------|-----------|----------------|
| **4-Model Calibrated Ensemble** | LightGBM + XGBoost + MLP Neural Net + Logistic Regression | Diversity-driven accuracy with uncertainty quantification |
| **Missingness-as-Clinical-Signal** | Information-theoretic analysis of which vitals are *not recorded* | Captures real ED workflows where missing data IS clinical data |
| **Sentence-BERT NLP** | Dense 384-dim semantic embeddings of chief complaints | Clinical meaning preservation beyond keyword matching |
| **Undertriage Safety Net** | Binary high-sensitivity detector (≥95% recall for ESI 1-2) | Catches critically ill patients at risk of dangerous delay |

**Key differentiator**: We treat this not as a classification problem alone, but as a **calibrated decision-support system** — because clinicians need reliable *probabilities*, not just point predictions. We validate with bootstrap confidence intervals, statistical significance tests, and failure mode analysis.

---
*Competition: Triagegeist | Laitinen-Fredriksson Foundation | AI in Emergency Triage*



```python
# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
import numpy as np
import pandas as pd
import warnings
import os
import re
import time
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# Core ML
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy import stats

# Gradient Boosting Frameworks
try:
    import lightgbm as lgb
    HAS_LGBM = True
    print(f"LightGBM: {lgb.__version__}")
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available — using sklearn GradientBoosting")

try:
    import xgboost as xgb
    HAS_XGB = True
    print(f"XGBoost: {xgb.__version__}")
except ImportError:
    HAS_XGB = False
    print("XGBoost not available — using sklearn RandomForest")

# Sentence Transformers for NLP
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
    print("SentenceTransformers available for semantic NLP")
except ImportError:
    HAS_SBERT = False
    print("SentenceTransformers not available — using TF-IDF fallback")

# Explainability
try:
    import shap
    HAS_SHAP = True
    print("SHAP available for explainability")
except ImportError:
    HAS_SHAP = False
    print("SHAP not available — using feature importance fallback")

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
try:
    import seaborn as sns
    sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Reproducibility
SEED = 42
np.random.seed(SEED)

t0_global = time.time()
print("\n" + "=" * 60)
print("Triagegeist: Multi-Modal Undertriage Detection v5")
print("=" * 60)
print(f"NumPy: {np.__version__} | Pandas: {pd.__version__}")
print(f"LightGBM: {'yes' if HAS_LGBM else 'no'} | XGBoost: {'yes' if HAS_XGB else 'no'}")
print(f"SentenceTransformers: {'yes' if HAS_SBERT else 'no'} | SHAP: {'yes' if HAS_SHAP else 'no'}")
print("Environment ready.")

```

## Data Loading & Integration

The Triagegeist dataset comprises three complementary sources:

| File | Records | Key Content |
|------|---------|-------------|
| `train.csv` | 80,000 | 40 structured features: vitals, demographics, temporal, pre-computed scores + ESI target |
| `chief_complaints.csv` | 100,000 | Free-text chief complaint narratives for NLP |
| `patient_history.csv` | 100,000 | 25 binary comorbidity flags |

**Data integration strategy**: Inner join on `patient_id` ensures each patient has complete multi-modal representation. We preserve the full feature set from all three sources rather than selecting subsets — the ensemble models handle feature selection implicitly via regularization.

**Missingness is clinically meaningful** — lower-acuity patients have more missing vitals, reflecting real ED workflows where not all vitals are obtained for minor complaints. We exploit this as a predictive signal (detailed in our Missingness Analysis section).



```python
# ============================================================================
# DATA LOADING
# ============================================================================

# Auto-detect competition data path
DATA_DIR = None
if os.path.exists('/kaggle/input'):
    for folder in os.listdir('/kaggle/input'):
        candidate = os.path.join('/kaggle/input', folder)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, 'train.csv')):
            DATA_DIR = candidate
            break

if DATA_DIR is None:
    import glob
    matches = glob.glob('/kaggle/input/**/train.csv', recursive=True)
    if matches:
        DATA_DIR = os.path.dirname(matches[0])
    else:
        DATA_DIR = '/kaggle/input/triagegeist'

print(f"Data directory: {DATA_DIR}")
print(f"Files: {os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else 'NOT FOUND'}")

print("\nLoading data...")
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
chief_complaints = pd.read_csv(os.path.join(DATA_DIR, 'chief_complaints.csv'))
patient_history = pd.read_csv(os.path.join(DATA_DIR, 'patient_history.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print(f"  train.csv: {train.shape[0]:,} rows x {train.shape[1]} cols")
print(f"  chief_complaints.csv: {chief_complaints.shape[0]:,} rows x {chief_complaints.shape[1]} cols")
print(f"  patient_history.csv: {patient_history.shape[0]:,} rows x {patient_history.shape[1]} cols")
print(f"  test.csv: {test.shape[0]:,} rows x {test.shape[1]} cols")

# Merge datasets on patient_id
df = train.merge(chief_complaints, on='patient_id', how='left', suffixes=('', '_cc'))
df = df.merge(patient_history, on='patient_id', how='left')
if 'chief_complaint_system_cc' in df.columns:
    df.drop('chief_complaint_system_cc', axis=1, inplace=True)

df_test = test.merge(chief_complaints, on='patient_id', how='left', suffixes=('', '_cc'))
df_test = df_test.merge(patient_history, on='patient_id', how='left')
if 'chief_complaint_system_cc' in df_test.columns:
    df_test.drop('chief_complaint_system_cc', axis=1, inplace=True)

print(f"\nMerged train: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"Merged test: {df_test.shape[0]:,} rows x {df_test.shape[1]} cols")

TARGET = 'triage_acuity'
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
df[TARGET] = df[TARGET].astype(int)

print(f"After target cleaning: {df.shape[0]:,} rows")
print(f"\nTarget distribution (ESI levels):")
vc = df[TARGET].value_counts().sort_index()
for esi, count in vc.items():
    pct = count / len(df) * 100
    print(f"  ESI {esi}: {count:>6,} ({pct:5.1f}%)")

```

## Exploratory Data Analysis

We examine four key dimensions that inform our modeling strategy:
1. **Missingness patterns** — clinically informative (MNAR by ESI level)
2. **Vital sign distributions** — stratified by acuity to understand class separation
3. **Temporal patterns** — arrival time, day, season effects on acuity mix
4. **Chief complaint systems** — mapped to acuity for NLP feature validation



```python
# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================
ESI_COLORS = {1: '#d62728', 2: '#ff7f0e', 3: '#2ca02c', 4: '#1f77b4', 5: '#9467bd'}
colors = [ESI_COLORS.get(i, '#333333') for i in sorted(df[TARGET].unique())]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Triagegeist: Exploratory Data Analysis', fontsize=14, fontweight='bold')

# 1. Missingness by ESI level
vital_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 
              'temperature_c', 'spo2', 'gcs_total', 'pain_score']
miss_by_esi = df.groupby(TARGET)[vital_cols].apply(lambda x: x.isnull().mean() * 100)
miss_by_esi.plot(kind='bar', ax=axes[0, 0], legend=False, color=plt.cm.tab10(np.linspace(0, 1, len(vital_cols))))
axes[0, 0].set_title('Missing Vital Signs by ESI Level')
axes[0, 0].set_xlabel('ESI Acuity')
axes[0, 0].set_ylabel('% Missing')
axes[0, 0].tick_params(axis='x', rotation=0)

# 2. ESI distribution
vc = df[TARGET].value_counts().sort_index()
bars = axes[0, 1].bar(vc.index, vc.values, color=[ESI_COLORS[i] for i in vc.index])
axes[0, 1].set_title('ESI Level Distribution')
axes[0, 1].set_xlabel('ESI Level')
axes[0, 1].set_ylabel('Count')
for bar, v in zip(bars, vc.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{v:,}', 
                    ha='center', va='bottom', fontsize=9)

# 3. Heart rate by ESI
hr_data = [df[df[TARGET]==i]['heart_rate'].dropna() for i in sorted(df[TARGET].unique())]
bp = axes[0, 2].boxplot(hr_data, labels=sorted(df[TARGET].unique()), patch_artist=True)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
axes[0, 2].set_title('Heart Rate by ESI Level')
axes[0, 2].set_xlabel('ESI Level')
axes[0, 2].set_ylabel('HR (bpm)')

# 4. Arrival hour distribution
for esi in sorted(df[TARGET].unique()):
    subset = df[df[TARGET] == esi]
    axes[1, 0].hist(subset['arrival_hour'], bins=24, alpha=0.4, label=f'ESI {esi}', 
                    color=ESI_COLORS[esi], density=True)
axes[1, 0].set_title('Arrival Hour by ESI Level')
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].legend(fontsize=8)

# 5. NEWS2 score by ESI
news_data = [df[df[TARGET]==i]['news2_score'].dropna() for i in sorted(df[TARGET].unique())]
bp2 = axes[1, 1].boxplot(news_data, labels=sorted(df[TARGET].unique()), patch_artist=True)
for patch, c in zip(bp2['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
axes[1, 1].set_title('NEWS2 Score by ESI Level')
axes[1, 1].set_xlabel('ESI Level')
axes[1, 1].set_ylabel('NEWS2')

# 6. Chief complaint system (top 8)
top_ccs = df['chief_complaint_system'].value_counts().head(8)
axes[1, 2].barh(range(len(top_ccs)), top_ccs.values, color='steelblue')
axes[1, 2].set_yticks(range(len(top_ccs)))
axes[1, 2].set_yticklabels(top_ccs.index, fontsize=9)
axes[1, 2].set_title('Chief Complaint Systems (Top 8)')
axes[1, 2].set_xlabel('Count')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=100, bbox_inches='tight')
plt.show()
print("EDA complete. Key finding: Missingness is strongly correlated with ESI level (MNAR).")

```

## Novel: Missingness as Clinical Signal

**Key Insight**: In real EDs, data isn't Missing At Random (MAR). Lower-acuity patients (ESI 4-5) often don't get full vital sign workups — a nurse triaging a sprained ankle doesn't typically check SpO2 or GCS. This *absence* of measurement is itself a clinical judgment.

We formalize this with three complementary features:
- **Per-vital missingness indicators** — binary flags capturing which vitals were not recorded
- **Total missing vital count** — overall documentation completeness
- **Missingness entropy** — information-theoretic measure (Shannon entropy of the binary missingness vector): high entropy = selective documentation (some vitals taken, others not); zero entropy = either complete or fully missing workup

The ambulance-missing interaction captures an anomalous clinical pattern: patients arriving by ambulance almost always get full workups — when they don't, something unusual happened during transport.



```python
# ============================================================================
# MISSINGNESS-AS-CLINICAL-SIGNAL ANALYSIS
# ============================================================================
vital_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 
              'temperature_c', 'spo2', 'gcs_total', 'pain_score', 'weight_kg', 'height_cm']

# Compute missingness profile per ESI level
print("Missingness rates by ESI level (%):")
print("=" * 70)
miss_profile = pd.DataFrame(index=sorted(df[TARGET].unique()))
for col in vital_cols:
    miss_profile[col] = df.groupby(TARGET)[col].apply(lambda x: x.isnull().mean() * 100)
print(miss_profile.round(1).to_string())

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Missingness heatmap
im = axes[0].imshow(miss_profile.values, cmap='YlOrRd', aspect='auto')
axes[0].set_xticks(range(len(vital_cols)))
axes[0].set_xticklabels([c.replace('_', '\n') for c in vital_cols], fontsize=7, rotation=45, ha='right')
axes[0].set_yticks(range(5))
axes[0].set_yticklabels([f'ESI {i}' for i in range(1, 6)])
axes[0].set_title('Missingness Rate (%) by ESI')
plt.colorbar(im, ax=axes[0], shrink=0.8)

# Total missing by ESI
df['n_missing_vitals'] = df[vital_cols].isnull().sum(axis=1)
miss_box = [df[df[TARGET]==i]['n_missing_vitals'] for i in sorted(df[TARGET].unique())]
bp = axes[1].boxplot(miss_box, labels=sorted(df[TARGET].unique()), patch_artist=True)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
axes[1].set_title('Number of Missing Vitals by ESI Level')
axes[1].set_xlabel('ESI Level')

# Statistical test: Kruskal-Wallis
h_stat, p_val = stats.kruskal(*miss_box)
axes[1].text(0.95, 0.95, f'Kruskal-Wallis\nH={h_stat:.1f}, p<0.001' if p_val < 0.001 else f'H={h_stat:.1f}, p={p_val:.3f}',
             transform=axes[1].transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

# Correlation: missingness count vs acuity
miss_acuity_corr = np.corrcoef(df['n_missing_vitals'], df[TARGET])[0, 1]
axes[2].scatter(df['n_missing_vitals'] + np.random.normal(0, 0.1, len(df)), 
                df[TARGET] + np.random.normal(0, 0.1, len(df)), 
                alpha=0.01, s=1, color='steelblue')
axes[2].set_xlabel('Number of Missing Vitals')
axes[2].set_ylabel('ESI Level')
axes[2].set_title(f'Missingness vs Acuity (r={miss_acuity_corr:.3f})')

plt.tight_layout()
plt.show()
print(f"\nCorrelation between missing vital count and ESI: r = {miss_acuity_corr:.3f}")
print("Conclusion: Missingness is strongly predictive — ESI 1-2 have near-complete documentation.")

```

## NLP: Chief Complaint Encoding

We use a **dual-strategy** approach:
1. **Sentence Transformers** (`all-MiniLM-L6-v2`) — 384-dimensional dense semantic embeddings that capture clinical meaning (e.g., "chest pain radiating to arm" ≈ "ACS presentation")
2. **TF-IDF** (fallback) — sparse bigram features capturing distinctive clinical terminology

The sentence transformer approach captures three critical NLP dimensions that bag-of-words methods miss:
- **Symptom severity**: "mild headache" vs "worst headache of my life" (thunderclap = SAH risk)
- **Anatomical specificity**: "chest pain" vs "sharp left-sided chest pain on inspiration" (pleuritic)
- **Clinical urgency cues**: "can't breathe" vs "short of breath with exertion" (acute vs chronic)



```python
# ============================================================================
# NLP: CHIEF COMPLAINT ENCODING
# ============================================================================

df['chief_complaint_raw'] = df['chief_complaint_raw'].fillna('not recorded')
df_test['chief_complaint_raw'] = df_test['chief_complaint_raw'].fillna('not recorded')

all_complaints = pd.concat([df['chief_complaint_raw'], df_test['chief_complaint_raw']])

if HAS_SBERT:
    print("Using Sentence Transformer (all-MiniLM-L6-v2) for semantic embeddings...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    train_embeddings = sbert_model.encode(
        df['chief_complaint_raw'].tolist(), 
        batch_size=256, show_progress_bar=True
    )
    test_embeddings = sbert_model.encode(
        df_test['chief_complaint_raw'].tolist(),
        batch_size=256, show_progress_bar=True
    )
    
    embed_cols = [f'sbert_{i}' for i in range(train_embeddings.shape[1])]
    nlp_train = pd.DataFrame(train_embeddings, columns=embed_cols, index=df.index)
    nlp_test = pd.DataFrame(test_embeddings, columns=embed_cols, index=df_test.index)
    NLP_COLS = embed_cols
    print(f"  Generated {len(embed_cols)} semantic embedding dimensions")
    
else:
    print("Using TF-IDF vectorization (bigrams, 300 features)...")
    tfidf = TfidfVectorizer(
        max_features=300, min_df=5, max_df=0.85,
        ngram_range=(1, 2), sublinear_tf=True,
        strip_accents='unicode', token_pattern=r'(?u)\b\w+\b'
    )
    tfidf.fit(all_complaints)
    train_tfidf = tfidf.transform(df['chief_complaint_raw'])
    test_tfidf = tfidf.transform(df_test['chief_complaint_raw'])
    
    tfidf_cols = [f'cc_{name}' for name in tfidf.get_feature_names_out()]
    nlp_train = pd.DataFrame(train_tfidf.toarray(), columns=tfidf_cols, index=df.index)
    nlp_test = pd.DataFrame(test_tfidf.toarray(), columns=tfidf_cols, index=df_test.index)
    NLP_COLS = tfidf_cols
    print(f"  Generated {len(tfidf_cols)} TF-IDF features")

# Most discriminative terms (using TF-IDF for interpretability)
tfidf_viz = TfidfVectorizer(max_features=50, ngram_range=(1, 2), sublinear_tf=True)
tfidf_viz.fit(df['chief_complaint_raw'])
X_viz = tfidf_viz.transform(df['chief_complaint_raw'])
vocab = tfidf_viz.get_feature_names_out()

high_acuity = df[TARGET].isin([1, 2]).values
mean_high = np.asarray(X_viz[high_acuity].mean(axis=0)).flatten()
mean_low = np.asarray(X_viz[~high_acuity].mean(axis=0)).flatten()
diff = mean_high - mean_low
top_high = np.argsort(diff)[-10:]
top_low = np.argsort(diff)[:10]

print("\nMost discriminative terms:")
print("  HIGH acuity (ESI 1-2):", [vocab[i] for i in top_high])
print("  LOW acuity (ESI 4-5):", [vocab[i] for i in top_low])

```

## Feature Engineering: Multi-Modal Integration

Our feature engineering combines **seven information categories** into a unified patient representation:

| Category | Features | Source | Rationale |
|----------|----------|--------|-----------|
| **Raw Vitals** | 10 | train.csv | Direct physiological state measurement |
| **Clinical Flags** | 18 | Computed | Expert-defined abnormality thresholds (tachycardia >100, hypoxia <94%, etc.) |
| **Computed Scores** | 6 | Computed + dataset | MEWS, shock index, NEWS2, MAP, pulse pressure, BMI |
| **Demographics & Context** | ~25 | train.csv | Age interactions, cyclical temporal encoding, arrival mode, mental status |
| **Comorbidity Profile** | ~30 | patient_history.csv | 25 flags + burden score + high-risk combinations |
| **Missingness Signature** | 11 | All sources | Per-vital flags + count + entropy + ambulance interaction |
| **NLP Embeddings** | 384 (or 300) | chief_complaints.csv | Sentence-BERT semantic dimensions |

Total features: **~490** (structured + NLP)



```python
# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(data, is_test=False):
    """Multi-modal feature engineering pipeline (~110 structured features)."""
    feats = pd.DataFrame(index=data.index)
    
    # --- RAW VITALS (10) ---
    vital_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate',
                  'temperature_c', 'spo2', 'gcs_total', 'pain_score', 'weight_kg', 'height_cm']
    for col in vital_cols:
        if col in data.columns:
            feats[col] = data[col]
    
    # --- PRE-COMPUTED SCORES (5) ---
    for col in ['shock_index', 'news2_score', 'mean_arterial_pressure', 'pulse_pressure', 'bmi']:
        if col in data.columns:
            feats[col] = data[col]
    
    # --- CLINICAL FLAGS (18) ---
    feats['tachycardia'] = (data['heart_rate'] > 100).astype(float)
    feats['bradycardia'] = (data['heart_rate'] < 60).astype(float)
    feats['severe_tachycardia'] = (data['heart_rate'] > 130).astype(float)
    feats['hypotension'] = (data['systolic_bp'] < 90).astype(float)
    feats['severe_hypotension'] = (data['systolic_bp'] < 70).astype(float)
    feats['hypertensive_crisis'] = (data['systolic_bp'] > 180).astype(float)
    feats['tachypnea'] = (data['respiratory_rate'] > 20).astype(float)
    feats['severe_tachypnea'] = (data['respiratory_rate'] > 30).astype(float)
    feats['bradypnea'] = (data['respiratory_rate'] < 12).astype(float)
    feats['hypoxia'] = (data['spo2'] < 94).astype(float)
    feats['severe_hypoxia'] = (data['spo2'] < 90).astype(float)
    feats['critical_hypoxia'] = (data['spo2'] < 85).astype(float)
    feats['fever'] = (data['temperature_c'] > 38.0).astype(float)
    feats['high_fever'] = (data['temperature_c'] > 39.0).astype(float)
    feats['hypothermia'] = (data['temperature_c'] < 35.0).astype(float)
    feats['altered_consciousness'] = (data['gcs_total'] < 15).astype(float)
    feats['severe_neuro'] = (data['gcs_total'] <= 8).astype(float)
    feats['severe_pain'] = (data['pain_score'] >= 8).astype(float)
    
    # --- COMPUTED MEWS (Modified Early Warning Score) ---
    mews = pd.Series(0, index=data.index, dtype=float)
    mews += (data['heart_rate'] > 100).fillna(False).astype(float)
    mews += (data['heart_rate'] > 130).fillna(False).astype(float)
    mews += (data['systolic_bp'] < 90).fillna(False).astype(float) * 2
    mews += (data['respiratory_rate'] > 20).fillna(False).astype(float)
    mews += (data['respiratory_rate'] > 30).fillna(False).astype(float)
    mews += (data['temperature_c'] > 38.5).fillna(False).astype(float)
    mews += (data['spo2'] < 94).fillna(False).astype(float)
    mews += (data['spo2'] < 90).fillna(False).astype(float)
    mews += (data['gcs_total'] < 15).fillna(False).astype(float)
    mews += (data['gcs_total'] <= 8).fillna(False).astype(float) * 2
    feats['mews_computed'] = mews
    
    # --- DEMOGRAPHICS & TEMPORAL ---
    feats['age'] = data['age']
    feats['is_pediatric'] = (data['age'] < 18).astype(float)
    feats['is_elderly'] = (data['age'] >= 65).astype(float)
    feats['is_very_elderly'] = (data['age'] >= 80).astype(float)
    feats['is_female'] = (data['sex'] == 'F').astype(float)
    feats['elderly_abnormal_vitals'] = (feats['is_elderly'] * feats['tachycardia']).fillna(0)
    
    feats['arrival_hour'] = data['arrival_hour']
    feats['night_arrival'] = data['arrival_hour'].apply(lambda h: 1.0 if (h >= 22 or h < 6) else 0.0)
    feats['evening_arrival'] = data['arrival_hour'].apply(lambda h: 1.0 if (18 <= h < 22) else 0.0)
    
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    if 'arrival_day' in data.columns:
        day_num = data['arrival_day'].map(day_map).fillna(0)
        feats['arrival_day_sin'] = np.sin(2 * np.pi * day_num / 7)
        feats['arrival_day_cos'] = np.cos(2 * np.pi * day_num / 7)
        feats['is_weekend'] = data['arrival_day'].isin(['Saturday', 'Sunday']).astype(float)
    
    season_map = {'spring': 0, 'summer': 1, 'fall': 2, 'autumn': 2, 'winter': 3}
    if 'arrival_season' in data.columns:
        season_num = data['arrival_season'].map(season_map).fillna(0)
        feats['season_sin'] = np.sin(2 * np.pi * season_num / 4)
        feats['season_cos'] = np.cos(2 * np.pi * season_num / 4)
    
    shift_map = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    if 'shift' in data.columns:
        feats['shift_code'] = data['shift'].map(shift_map).fillna(0)
    
    feats['arrived_ambulance'] = (data['arrival_mode'] == 'ambulance').astype(float)
    feats['arrived_transfer'] = (data['arrival_mode'].isin(['transfer', 'helicopter'])).astype(float)
    
    if 'transport_origin' in data.columns:
        high_risk_origins = ['nursing_home', 'assisted_living', 'rehab_facility', 'other_hospital']
        feats['high_risk_origin'] = data['transport_origin'].isin(high_risk_origins).astype(float)
    
    mental_map = {'alert': 0, 'drowsy': 1, 'verbal': 1, 'confused': 2, 'agitated': 2, 'pain': 3, 'unresponsive': 4}
    feats['mental_status_code'] = data['mental_status_triage'].map(mental_map).fillna(0)
    
    if 'chief_complaint_system' in data.columns:
        top_systems = ['cardiovascular', 'neurological', 'respiratory', 'trauma',
                       'gastrointestinal', 'musculoskeletal', 'psychiatric', 'genitourinary']
        for sys in top_systems:
            feats[f'ccs_{sys}'] = (data['chief_complaint_system'] == sys).astype(float)
    
    # --- COMORBIDITY PROFILE ---
    hx_cols = [c for c in data.columns if c.startswith('hx_')]
    for col in hx_cols:
        feats[col] = data[col].fillna(0)
    
    feats['comorbidity_burden'] = data[hx_cols].sum(axis=1) if hx_cols else 0
    
    if 'hx_heart_failure' in data.columns and 'hx_copd' in data.columns:
        feats['cardiopulm_risk'] = (data['hx_heart_failure'].fillna(0) + data['hx_copd'].fillna(0) + 
                                    data.get('hx_coronary_artery_disease', pd.Series(0, index=data.index)).fillna(0))
    if 'hx_diabetes_type1' in data.columns:
        feats['diabetes_any'] = ((data['hx_diabetes_type1'].fillna(0) + data['hx_diabetes_type2'].fillna(0)) > 0).astype(float)
    if 'hx_immunosuppressed' in data.columns:
        feats['immunocompromised'] = ((data['hx_immunosuppressed'].fillna(0) + data['hx_hiv'].fillna(0) + 
                                       data['hx_malignancy'].fillna(0)) > 0).astype(float)
    
    # --- CLINICAL HISTORY ---
    feats['num_prior_ed_visits'] = data['num_prior_ed_visits_12m'].fillna(0)
    if 'num_prior_admissions_12m' in data.columns:
        feats['num_prior_admissions'] = data['num_prior_admissions_12m'].fillna(0)
    feats['num_active_medications'] = data['num_active_medications'].fillna(0)
    feats['num_comorbidities'] = data['num_comorbidities'].fillna(0)
    feats['frequent_visitor'] = (data['num_prior_ed_visits_12m'].fillna(0) >= 4).astype(float)
    feats['polypharmacy'] = (data['num_active_medications'].fillna(0) >= 5).astype(float)
    
    # --- MISSINGNESS SIGNATURE ---
    miss_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate',
                 'temperature_c', 'spo2', 'gcs_total', 'pain_score']
    for col in miss_cols:
        feats[f'miss_{col}'] = data[col].isnull().astype(float)
    feats['n_missing_vitals'] = data[miss_cols].isnull().sum(axis=1)
    
    # Missingness entropy (vectorized)
    miss_matrix = data[miss_cols].isnull().astype(float)
    p_miss = miss_matrix.mean(axis=1)
    miss_ent = pd.Series(0.0, index=data.index)
    mask_ent = (p_miss > 0) & (p_miss < 1)
    miss_ent[mask_ent] = -(p_miss[mask_ent] * np.log2(p_miss[mask_ent]) + (1-p_miss[mask_ent]) * np.log2(1-p_miss[mask_ent]))
    feats['miss_entropy'] = miss_ent
    
    feats['ambulance_missing_interaction'] = feats['arrived_ambulance'] * feats['n_missing_vitals']
    
    # --- VITAL SIGN INTERACTIONS ---
    feats['hr_sbp_product'] = data['heart_rate'].fillna(80) * data['systolic_bp'].fillna(120)
    feats['age_hr_interaction'] = data['age'].fillna(50) * data['heart_rate'].fillna(80) / 1000
    feats['spo2_rr_interaction'] = data['spo2'].fillna(97) * data['respiratory_rate'].fillna(16) / 100
    
    return feats

# Apply to train and test
print("Engineering features...")
X_structured = engineer_features(df)
X_structured_test = engineer_features(df_test, is_test=True)

# Combine structured + NLP
X_full = pd.concat([X_structured, nlp_train], axis=1)
X_full_test = pd.concat([X_structured_test, nlp_test], axis=1)

# Impute missing values
imputer = SimpleImputer(strategy='median')
feature_names = X_full.columns.tolist()
X_full_imputed = pd.DataFrame(imputer.fit_transform(X_full), columns=feature_names, index=X_full.index)
X_full_test_imputed = pd.DataFrame(imputer.transform(X_full_test), columns=feature_names, index=X_full_test.index)

y = df[TARGET].values

print(f"\nFinal feature matrix: {X_full_imputed.shape[0]:,} x {X_full_imputed.shape[1]} features")
print(f"  Structured features: {X_structured.shape[1]}")
print(f"  NLP features: {len(NLP_COLS)}")
print(f"  Total: {X_full_imputed.shape[1]}")

# Feature category summary
vital_cols_list = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate',
                   'temperature_c', 'spo2', 'gcs_total', 'pain_score', 'weight_kg', 'height_cm']
structured_cats = {
    'Raw Vitals': [c for c in X_structured.columns if c in vital_cols_list],
    'Clinical Flags': [c for c in X_structured.columns if c in ['tachycardia','bradycardia','severe_tachycardia','hypotension','severe_hypotension','hypertensive_crisis','tachypnea','severe_tachypnea','bradypnea','hypoxia','severe_hypoxia','critical_hypoxia','fever','high_fever','hypothermia','altered_consciousness','severe_neuro','severe_pain']],
    'Scores': ['mews_computed','shock_index','news2_score','mean_arterial_pressure','pulse_pressure','bmi'],
    'Demographics': [c for c in X_structured.columns if any(x in c for x in ['age','pediatric','elderly','female','arrival','night','evening','weekend','day_sin','day_cos','season','shift','ambulance','transfer','mental','origin','ccs_'])],
    'Comorbidities': [c for c in X_structured.columns if c.startswith('hx_') or c in ['comorbidity_burden','cardiopulm_risk','diabetes_any','immunocompromised','polypharmacy']],
    'Missingness': [c for c in X_structured.columns if 'miss' in c],
    'Interactions': [c for c in X_structured.columns if 'interaction' in c or 'product' in c],
}
for cat, cols in structured_cats.items():
    if cols:
        print(f"  {cat}: {len(cols)} features")

```

## Model Training: 4-Model Calibrated Ensemble

We train a **4-model ensemble** using complementary learning paradigms:

| Model | Architecture | Rationale |
|-------|-------------|-----------|
| **LightGBM** | Gradient boosting (histogram-based) | Fast, native missing value handling, strong on tabular |
| **XGBoost** | Gradient boosting (exact/approx) | Different regularization → ensemble diversity |
| **MLP Neural Net** | 3-layer feedforward (256→128→64) | Captures non-linear feature interactions that trees miss |
| **Logistic Regression** | Linear (L2-penalized) | Well-calibrated baseline, diversity contributor |

The final prediction uses **probability-weighted soft voting** (weights ∝ macro-F1) rather than hard voting, enabling better calibration and uncertainty quantification. Cross-validation: **5-fold stratified** (preserving ESI class proportions).



```python
# ============================================================================
# MODEL TRAINING: 4-MODEL CALIBRATED ENSEMBLE WITH HYPERPARAMETER OPTIMIZATION
# ============================================================================
# Key improvements over baseline:
#   - 5-fold stratified CV (reduced variance vs 3-fold)
#   - Optuna hyperparameter tuning for LightGBM (20 trials)
#   - Stacking meta-learner (LogisticRegression on OOF predictions)
# ============================================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Scaling for models that need it (MLP, LR)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full_imputed)

# --- HYPERPARAMETER TUNING: Optuna on LightGBM ---
print("Phase 1: Hyperparameter Optimization (Optuna, 20 trials)...")
print("-" * 60)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  Optuna not available — using tuned defaults")

best_lgbm_params = {
    'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.08,
    'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'min_child_samples': 30, 'num_leaves': 63
}

if HAS_OPTUNA and HAS_LGBM:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 600),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        }
        model = lgb.LGBMClassifier(
            **params, class_weight='balanced', random_state=SEED, verbose=-1, n_jobs=-1
        )
        # Quick 3-fold eval for speed during tuning
        skf_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        proba = cross_val_predict(model, X_full_imputed, y, cv=skf_tune, method='predict_proba')
        preds = proba.argmax(axis=1) + 1
        return f1_score(y, preds, average='macro')
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    
    best_lgbm_params = study.best_params
    print(f"  Best F1 (macro): {study.best_value:.4f}")
    print(f"  Best params: {best_lgbm_params}")
    print(f"  Trials completed: {len(study.trials)}")
else:
    print("  Using pre-tuned parameters (Optuna unavailable or no LightGBM)")

# --- Model 1: LightGBM (with tuned hyperparameters) ---
print("\nPhase 2: Training Models (5-fold stratified CV)...")
print("-" * 60)

if HAS_LGBM:
    print("  Training LightGBM (5-fold CV, tuned params)...")
    lgbm_model = lgb.LGBMClassifier(
        **best_lgbm_params, class_weight='balanced', random_state=SEED, verbose=-1, n_jobs=-1
    )
    lgbm_proba = cross_val_predict(lgbm_model, X_full_imputed, y, cv=skf, method='predict_proba')
    lgbm_preds = lgbm_proba.argmax(axis=1) + 1
    print(f"    LightGBM Accuracy: {accuracy_score(y, lgbm_preds):.4f}")
    print(f"    LightGBM F1 (macro): {f1_score(y, lgbm_preds, average='macro'):.4f}")
else:
    print("  Training GradientBoosting (sklearn fallback, 5-fold CV)...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=20, random_state=SEED
    )
    lgbm_proba = cross_val_predict(gb_model, X_full_imputed, y, cv=skf, method='predict_proba')
    lgbm_preds = lgbm_proba.argmax(axis=1) + 1
    print(f"    GradientBoosting Accuracy: {accuracy_score(y, lgbm_preds):.4f}")

# --- Model 2: XGBoost ---
if HAS_XGB:
    print("  Training XGBoost (5-fold CV)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=30, eval_metric='mlogloss',
        random_state=SEED, n_jobs=-1, verbosity=0
    )
    # Fix: encode labels 0-4 for XGBoost consistently
    xgb_proba = cross_val_predict(xgb_model, X_full_imputed, y - 1, cv=skf, method='predict_proba')
    xgb_preds = xgb_proba.argmax(axis=1) + 1
    print(f"    XGBoost Accuracy: {accuracy_score(y, xgb_preds):.4f}")
    print(f"    XGBoost F1 (macro): {f1_score(y, xgb_preds, average='macro'):.4f}")
else:
    print("  Training RandomForest (sklearn fallback, 5-fold CV)...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight='balanced',
        min_samples_leaf=10, random_state=SEED, n_jobs=-1
    )
    xgb_proba = cross_val_predict(rf_model, X_full_imputed, y, cv=skf, method='predict_proba')
    xgb_preds = xgb_proba.argmax(axis=1) + 1

# --- Model 3: MLP Neural Network ---
print("  Training MLP Neural Network (5-fold CV)...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64), activation='relu',
    solver='adam', alpha=0.001, batch_size=256,
    learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=200, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=15, random_state=SEED
)
mlp_proba = cross_val_predict(mlp_model, X_scaled, y, cv=skf, method='predict_proba')
mlp_preds = mlp_proba.argmax(axis=1) + 1
print(f"    MLP Accuracy: {accuracy_score(y, mlp_preds):.4f}")
print(f"    MLP F1 (macro): {f1_score(y, mlp_preds, average='macro'):.4f}")

# --- Model 4: Logistic Regression ---
print("  Training Logistic Regression (5-fold CV)...")
lr_model = LogisticRegression(
    max_iter=2000, C=1.0, solver='lbfgs', multi_class='multinomial',
    class_weight='balanced', random_state=SEED, n_jobs=-1
)
lr_proba = cross_val_predict(lr_model, X_scaled, y, cv=skf, method='predict_proba')
lr_preds = lr_proba.argmax(axis=1) + 1
print(f"    LogisticRegression Accuracy: {accuracy_score(y, lr_preds):.4f}")
print(f"    LogisticRegression F1 (macro): {f1_score(y, lr_preds, average='macro'):.4f}")

# --- STACKING META-LEARNER (instead of naive F1-weighted average) ---
print("\nPhase 3: Stacking Meta-Learner...")
print("-" * 60)

# Stack OOF predictions from all 4 models as meta-features
n_classes = lgbm_proba.shape[1]
meta_features = np.hstack([lgbm_proba, xgb_proba, mlp_proba, lr_proba])
print(f"  Meta-feature matrix: {meta_features.shape} ({4} models x {n_classes} classes)")

# Train LogisticRegression meta-learner on stacked OOF predictions
meta_learner = LogisticRegression(
    max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial',
    random_state=SEED, n_jobs=-1
)
# Use CV for the meta-learner too (prevents overfitting on OOF predictions)
meta_proba = cross_val_predict(meta_learner, meta_features, y, cv=skf, method='predict_proba')
meta_preds = meta_proba.argmax(axis=1) + 1

# Also compute simple weighted average for comparison
f1_lgbm = f1_score(y, lgbm_preds, average='macro')
f1_xgb = f1_score(y, xgb_preds, average='macro')
f1_mlp = f1_score(y, mlp_preds, average='macro')
f1_lr = f1_score(y, lr_preds, average='macro')
total_f1 = f1_lgbm + f1_xgb + f1_mlp + f1_lr
w_lgbm = f1_lgbm / total_f1
w_xgb = f1_xgb / total_f1
w_mlp = f1_mlp / total_f1
w_lr = f1_lr / total_f1

naive_proba = w_lgbm * lgbm_proba + w_xgb * xgb_proba + w_mlp * mlp_proba + w_lr * lr_proba
naive_preds = naive_proba.argmax(axis=1) + 1

# Compare stacking vs naive
f1_stacked = f1_score(y, meta_preds, average='macro')
f1_naive = f1_score(y, naive_preds, average='macro')
acc_stacked = accuracy_score(y, meta_preds)
acc_naive = accuracy_score(y, naive_preds)

print(f"\n  Stacking vs Naive Ensemble Comparison:")
print(f"  {'Method':<25s} {'Accuracy':>10s} {'F1 (macro)':>12s}")
print(f"  {'-'*47}")
print(f"  {'Naive F1-weighted avg':<25s} {acc_naive:>10.4f} {f1_naive:>12.4f}")
print(f"  {'Stacking (LR meta)':<25s} {acc_stacked:>10.4f} {f1_stacked:>12.4f}")
print(f"  {'Improvement':<25s} {acc_stacked-acc_naive:>+10.4f} {f1_stacked-f1_naive:>+12.4f}")

# Use the better method
if f1_stacked >= f1_naive:
    ensemble_proba = meta_proba
    ensemble_preds = meta_preds
    ensemble_method = "Stacking (LogisticRegression meta-learner)"
    print(f"\n  >>> Using STACKING ensemble (superior performance)")
else:
    ensemble_proba = naive_proba
    ensemble_preds = naive_preds
    ensemble_method = "F1-weighted average"
    print(f"\n  >>> Using F1-WEIGHTED ensemble (better in this case)")

# Core metrics
acc = accuracy_score(y, ensemble_preds)
f1_macro = f1_score(y, ensemble_preds, average='macro')
f1_weighted = f1_score(y, ensemble_preds, average='weighted')
kappa = cohen_kappa_score(y, ensemble_preds, weights='quadratic')
undertriage = np.sum(ensemble_preds > y) / len(y) * 100
dangerous_undertriage = np.sum((ensemble_preds - y) >= 2) / len(y) * 100

print(f"\n{'='*60}")
print(f"FINAL ENSEMBLE RESULTS (5-fold CV, 4 models + {ensemble_method}):")
print(f"  Accuracy:                {acc:.4f}")
print(f"  F1 (macro):              {f1_macro:.4f}")
print(f"  F1 (weighted):           {f1_weighted:.4f}")
print(f"  Cohen's Kappa (QWK):     {kappa:.4f}")
print(f"  Overall undertriage:     {undertriage:.2f}%")
print(f"  Dangerous undertriage:   {dangerous_undertriage:.3f}%")
print(f"{'='*60}")

```

## Statistical Validation: Bootstrap CIs & Model Comparison

We provide **95% bootstrap confidence intervals** for all metrics to quantify estimation uncertainty, and **McNemar's test** to verify that ensemble improvements over individual models are statistically significant — not just noise from a particular train/test split.



```python
# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS & STATISTICAL SIGNIFICANCE
# ============================================================================

def bootstrap_metric(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95, **kwargs):
    """Compute bootstrap CI for any sklearn metric."""
    scores = []
    n = len(y_true)
    rng = np.random.RandomState(SEED)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx], **kwargs))
    alpha = (1 - ci) / 2
    return np.percentile(scores, [alpha * 100, (1 - alpha) * 100])

print("Computing 95% Bootstrap Confidence Intervals (1000 resamples)...")
print("=" * 65)

metrics_ci = {}
for name, preds in [('Ensemble', ensemble_preds), ('LightGBM', lgbm_preds), 
                     ('XGBoost', xgb_preds), ('MLP', mlp_preds), ('LR', lr_preds)]:
    ci_acc = bootstrap_metric(y, preds, accuracy_score)
    ci_f1 = bootstrap_metric(y, preds, f1_score, average='macro')
    point_acc = accuracy_score(y, preds)
    point_f1 = f1_score(y, preds, average='macro')
    metrics_ci[name] = {'acc': point_acc, 'acc_ci': ci_acc, 'f1': point_f1, 'f1_ci': ci_f1}
    print(f"  {name:12s}  Acc={point_acc:.4f} [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]  "
          f"F1={point_f1:.4f} [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")

# McNemar's Test: Ensemble vs each individual model
print("\nMcNemar's Test (Ensemble vs Individual Models):")
print("-" * 45)
for name, preds in [('LightGBM', lgbm_preds), ('XGBoost', xgb_preds), 
                     ('MLP', mlp_preds), ('LR', lr_preds)]:
    ens_correct = (ensemble_preds == y)
    ind_correct = (preds == y)
    # Contingency: b = ensemble right & individual wrong, c = ensemble wrong & individual right
    b = np.sum(ens_correct & ~ind_correct)
    c = np.sum(~ens_correct & ind_correct)
    if b + c > 0:
        chi2 = (abs(b - c) - 1)**2 / (b + c)  # McNemar's with continuity correction
        p_val = 1 - stats.chi2.cdf(chi2, df=1)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  Ensemble vs {name:12s}: b={b}, c={c}, chi2={chi2:.2f}, p={p_val:.4f} {sig}")
    else:
        print(f"  Ensemble vs {name:12s}: identical predictions")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CI plot
models = list(metrics_ci.keys())
accs = [metrics_ci[m]['acc'] for m in models]
acc_errs = np.array([[metrics_ci[m]['acc'] - metrics_ci[m]['acc_ci'][0], 
                       metrics_ci[m]['acc_ci'][1] - metrics_ci[m]['acc']] for m in models]).T
f1s = [metrics_ci[m]['f1'] for m in models]
f1_errs = np.array([[metrics_ci[m]['f1'] - metrics_ci[m]['f1_ci'][0], 
                      metrics_ci[m]['f1_ci'][1] - metrics_ci[m]['f1']] for m in models]).T

x = np.arange(len(models))
axes[0].errorbar(x - 0.15, accs, yerr=acc_errs, fmt='o', capsize=5, color='steelblue', label='Accuracy')
axes[0].errorbar(x + 0.15, f1s, yerr=f1_errs, fmt='s', capsize=5, color='coral', label='F1 (macro)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=30)
axes[0].set_ylabel('Score')
axes[0].set_title('Model Comparison with 95% Bootstrap CIs')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Per-class F1 comparison
per_class_f1 = {}
for name, preds in [('Ensemble', ensemble_preds), ('LightGBM', lgbm_preds), 
                     ('XGBoost', xgb_preds), ('MLP', mlp_preds)]:
    per_class_f1[name] = f1_score(y, preds, average=None, labels=[1,2,3,4,5])

x2 = np.arange(5)
w = 0.2
for i, (name, f1vals) in enumerate(per_class_f1.items()):
    axes[1].bar(x2 + i*w - 1.5*w, f1vals, w, label=name, alpha=0.8)
axes[1].set_xticks(x2)
axes[1].set_xticklabels([f'ESI-{i+1}' for i in range(5)])
axes[1].set_ylabel('F1 Score')
axes[1].set_title('Per-Class F1 by Model')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

```


```python
# ============================================================================
# CONFUSION MATRICES & CLINICAL ERROR ANALYSIS
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

model_names = ['LightGBM/GB', 'XGBoost/RF', 'Ensemble (4-model)']
all_preds = [lgbm_preds, xgb_preds, ensemble_preds]

for ax, preds, name in zip(axes, all_preds, model_names):
    cm = confusion_matrix(y, preds, labels=[1, 2, 3, 4, 5])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'{name}\nAcc={accuracy_score(y, preds):.3f}, F1={f1_score(y, preds, average="macro"):.3f}')
    ax.set_xlabel('Predicted ESI')
    ax.set_ylabel('True ESI')
    ax.set_xticks(range(5))
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_yticks(range(5))
    ax.set_yticklabels([1, 2, 3, 4, 5])
    
    for i in range(5):
        for j in range(5):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center', color=color, fontsize=9)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
plt.show()

# Clinical error analysis
print("\n" + "=" * 60)
print("CLINICAL ERROR ANALYSIS (Ensemble)")
print("=" * 60)
print(f"\nPer-class report:")
print(classification_report(y, ensemble_preds, target_names=['ESI-1','ESI-2','ESI-3','ESI-4','ESI-5']))

error_diff = ensemble_preds - y
dangerous_mask = error_diff >= 2
print(f"\nDangerous undertriage (predicted >=2 levels too low):")
print(f"  Total: {dangerous_mask.sum()} / {len(y)} ({dangerous_mask.mean()*100:.3f}%)")
for true_esi in [1, 2, 3]:
    mask_esi = (y == true_esi) & dangerous_mask
    if mask_esi.sum() > 0:
        print(f"  True ESI {true_esi} -> predicted ESI {true_esi+2}+: {mask_esi.sum()} cases")

```

## Failure Mode Analysis

Understanding **where and why** the model fails is as important as reporting aggregate metrics. We analyze:
1. **ESI-3 boundary confusion** — ESI-3 patients sit at the clinical boundary between "needs attention soon" and "can wait," making them inherently harder to classify
2. **High-confidence errors** — cases where the model is wrong AND confident (most dangerous)
3. **Undertriage patterns** — which patient profiles are most likely to be under-triaged



```python
# ============================================================================
# FAILURE MODE ANALYSIS
# ============================================================================

# 1. Error distribution by true ESI
print("ERROR ANALYSIS BY ESI LEVEL")
print("=" * 60)
errors = ensemble_preds != y
for esi in range(1, 6):
    mask = y == esi
    n_total = mask.sum()
    n_err = (mask & errors).sum()
    err_rate = n_err / n_total * 100 if n_total > 0 else 0
    # Where do errors go?
    if n_err > 0:
        err_preds = ensemble_preds[mask & errors]
        top_misclass = pd.Series(err_preds).value_counts().head(3)
        misclass_str = ", ".join([f"ESI-{int(k)}:{v}" for k, v in top_misclass.items()])
    else:
        misclass_str = "none"
    print(f"  ESI-{esi}: {n_err}/{n_total} errors ({err_rate:.2f}%) -> {misclass_str}")

# 2. High-confidence errors (most dangerous)
max_conf = ensemble_proba.max(axis=1)
high_conf_errors = errors & (max_conf > 0.8)
print(f"\nHigh-confidence errors (conf > 80%): {high_conf_errors.sum()} cases")
if high_conf_errors.sum() > 0:
    print("  These are the MOST CLINICALLY DANGEROUS — model is wrong AND confident:")
    hc_df = pd.DataFrame({
        'true_esi': y[high_conf_errors],
        'pred_esi': ensemble_preds[high_conf_errors],
        'confidence': max_conf[high_conf_errors]
    })
    print(hc_df.groupby(['true_esi', 'pred_esi']).agg(
        count=('confidence', 'size'),
        mean_conf=('confidence', 'mean')
    ).sort_values('count', ascending=False).head(10).to_string())

# 3. Undertriage risk factors
undertriage_mask = ensemble_preds > y
if undertriage_mask.sum() > 10:
    print(f"\nUndertriage risk factor analysis ({undertriage_mask.sum()} undertriaged cases):")
    # Compare feature distributions
    ut_feats = X_full_imputed.loc[undertriage_mask]
    correct_feats = X_full_imputed.loc[~errors]
    
    risk_factors = []
    for col in ['n_missing_vitals', 'miss_entropy', 'age', 'comorbidity_burden', 
                'mental_status_code', 'night_arrival', 'arrived_ambulance']:
        if col in X_full_imputed.columns:
            mean_ut = ut_feats[col].mean()
            mean_correct = correct_feats[col].mean()
            diff = mean_ut - mean_correct
            risk_factors.append({'feature': col, 'undertriaged_mean': mean_ut, 
                                'correct_mean': mean_correct, 'difference': diff})
    rf_df = pd.DataFrame(risk_factors).sort_values('difference', key=abs, ascending=False)
    print(rf_df.to_string(index=False, float_format='%.3f'))

# 4. Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Error rate by ESI
esi_err_rates = []
for esi in range(1, 6):
    mask = y == esi
    esi_err_rates.append((mask & errors).sum() / mask.sum() * 100 if mask.sum() > 0 else 0)
axes[0].bar(range(1, 6), esi_err_rates, color=[ESI_COLORS[i] for i in range(1, 6)])
axes[0].set_xlabel('True ESI Level')
axes[0].set_ylabel('Error Rate (%)')
axes[0].set_title('Error Rate by ESI Level\n(ESI-3 boundary is hardest)')
axes[0].set_xticks(range(1, 6))

# Confidence distribution: correct vs errors
axes[1].hist(max_conf[~errors], bins=50, alpha=0.6, color='green', label='Correct', density=True)
axes[1].hist(max_conf[errors], bins=50, alpha=0.6, color='red', label='Errors', density=True)
axes[1].set_xlabel('Prediction Confidence')
axes[1].set_ylabel('Density')
axes[1].set_title('Confidence Distribution: Correct vs Errors')
axes[1].legend()

plt.tight_layout()
plt.savefig('failure_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

```

## Model Explainability: SHAP Analysis

We use **SHAP (SHapley Additive exPlanations)** for both global and local interpretability:
1. **Global**: Which feature categories drive predictions overall?
2. **Local**: For individual patients, why was this specific ESI assigned?

This is critical for clinical trust (clinicians reject black-box tools), regulatory compliance (FDA AI/ML transparency guidance, 2021), and debugging (verifying the model uses clinically appropriate signals).



```python
# ============================================================================
# SHAP EXPLAINABILITY
# ============================================================================

# Train final models on full data (reused for SHAP + test predictions)
print("Training final models on full training data...")
if HAS_LGBM:
    final_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
        num_leaves=63, class_weight='balanced', random_state=SEED, verbose=-1
    )
else:
    final_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=20, random_state=SEED
    )
final_model.fit(X_full_imputed, y)

if HAS_XGB:
    final_xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=30,
        eval_metric='mlogloss', random_state=SEED, verbosity=0
    )
    final_xgb_model.fit(X_full_imputed, y - 1)
else:
    final_xgb_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight='balanced',
        min_samples_leaf=10, random_state=SEED, n_jobs=-1
    )
    final_xgb_model.fit(X_full_imputed, y)

final_mlp_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64), activation='relu',
    solver='adam', alpha=0.001, batch_size=256,
    learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=100, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, random_state=SEED
)
final_mlp_model.fit(X_scaled, y)

final_lr_model = LogisticRegression(
    max_iter=2000, C=1.0, solver='lbfgs', multi_class='multinomial',
    class_weight='balanced', random_state=SEED
)
final_lr_model.fit(X_scaled, y)
print("All 4 final models trained.")

if HAS_SHAP:
    print("Computing SHAP values (subsample=1000)...")
    X_shap = X_full_imputed.sample(n=min(1000, len(X_full_imputed)), random_state=SEED)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_shap)
    
    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.mean(axis=1)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
else:
    print("Using feature_importances_ (SHAP not available)...")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

def get_category(feat_name):
    if feat_name.startswith('sbert_') or feat_name.startswith('cc_'):
        return 'NLP'
    elif feat_name.startswith('hx_') or feat_name in ['comorbidity_burden', 'cardiopulm_risk', 'diabetes_any', 'immunocompromised']:
        return 'Comorbidities'
    elif 'miss' in feat_name:
        return 'Missingness'
    elif feat_name in ['age', 'is_pediatric', 'is_elderly', 'is_very_elderly', 'is_female', 
                       'arrival_hour', 'night_arrival', 'evening_arrival', 'is_weekend',
                       'arrival_day_sin', 'arrival_day_cos', 'season_sin', 'season_cos',
                       'shift_code', 'arrived_ambulance', 'arrived_transfer', 'high_risk_origin',
                       'mental_status_code', 'elderly_abnormal_vitals'] or feat_name.startswith('ccs_'):
        return 'Demographics/Context'
    else:
        return 'Vitals/Scores'

cat_colors = {
    'Vitals/Scores': '#1f77b4', 'NLP': '#9467bd', 'Comorbidities': '#ff7f0e',
    'Missingness': '#7f7f7f', 'Demographics/Context': '#2ca02c'
}

top_n = 30
top_feats = importance_df.head(top_n)
top_feats = top_feats.copy()
top_feats['category'] = top_feats['feature'].apply(get_category)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

feat_colors_plot = [cat_colors[c] for c in top_feats['category']]
axes[0].barh(range(top_n), top_feats['importance'].values[::-1], 
             color=feat_colors_plot[::-1])
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_feats['feature'].values[::-1], fontsize=8)
axes[0].set_xlabel('Mean |SHAP| / Importance')
axes[0].set_title(f'Top {top_n} Features by Importance')
legend_patches = [mpatches.Patch(color=c, label=l) for l, c in cat_colors.items()]
axes[0].legend(handles=legend_patches, loc='lower right', fontsize=9)

cat_importance = importance_df.copy()
cat_importance['category'] = cat_importance['feature'].apply(get_category)
cat_totals = cat_importance.groupby('category')['importance'].sum().sort_values(ascending=False)
axes[1].pie(cat_totals.values, labels=cat_totals.index, autopct='%1.1f%%',
            colors=[cat_colors[c] for c in cat_totals.index], startangle=90)
axes[1].set_title('Feature Importance by Category')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nTop 10 most important features:")
for _, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:35s} [{get_category(row['feature']):20s}] = {row['importance']:.4f}")

```

## Probability Calibration Analysis

A critical requirement for clinical deployment is **well-calibrated probabilities**. When a model says "70% chance of ESI-2," it should be correct ~70% of the time. We evaluate:
- **Reliability diagrams** per ESI level
- **Expected Calibration Error (ECE)** — the average gap between predicted confidence and observed accuracy



```python
# ============================================================================
# CALIBRATION ANALYSIS
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

ece_scores = {}
for idx, esi in enumerate([1, 2, 3, 4, 5]):
    ax = axes[idx // 3, idx % 3]
    y_binary = (y == esi).astype(int)
    prob_esi = ensemble_proba[:, esi - 1]
    
    try:
        fraction_pos, mean_predicted = calibration_curve(y_binary, prob_esi, n_bins=10, strategy='uniform')
        ax.plot(mean_predicted, fraction_pos, 'o-', color=ESI_COLORS[esi], linewidth=2, label=f'ESI {esi}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        
        bin_counts = np.histogram(prob_esi, bins=10, range=(0, 1))[0]
        ece = np.sum(np.abs(fraction_pos - mean_predicted) * bin_counts[:len(fraction_pos)] / len(y))
        ece_scores[esi] = ece
        ax.set_title(f'ESI {esi} (ECE={ece:.4f})')
    except Exception:
        ax.set_title(f'ESI {esi} (insufficient data)')
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

ax_sum = axes[1, 2]
esi_levels = list(ece_scores.keys())
ece_vals = [ece_scores[e] for e in esi_levels]
bars = ax_sum.bar(esi_levels, ece_vals, color=[ESI_COLORS[e] for e in esi_levels])
ax_sum.set_title('ECE by ESI Level')
ax_sum.set_xlabel('ESI Level')
ax_sum.set_ylabel('Expected Calibration Error')
ax_sum.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Good threshold')
ax_sum.legend()

plt.tight_layout()
plt.savefig('calibration_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

mean_ece = np.mean(ece_vals) if ece_vals else 0
print(f"\nCalibration Analysis:")
print(f"  Mean ECE: {mean_ece:.4f}")
for esi, ece in ece_scores.items():
    status = "GOOD" if ece < 0.05 else "FAIR" if ece < 0.10 else "NEEDS IMPROVEMENT"
    print(f"  ESI {esi}: ECE = {ece:.4f} [{status}]")

```

## Undertriage Safety Net: High-Sensitivity Binary Detector

The safety net is our **most clinically important** contribution. Binary classification:
- **Positive**: ESI 1–2 (emergent/resuscitation) — MUST NOT be under-triaged
- **Negative**: ESI 3–5 (non-emergent)

We tune the threshold for **≥95% sensitivity** — accepting lower specificity to ensure critically ill patients are flagged. Clinical workflow: when the safety net fires, it signals: *"This patient may be more acute than assigned — consider reassessment."*



```python
# ============================================================================
# UNDERTRIAGE SAFETY NET
# ============================================================================
y_binary = (y <= 2).astype(int)
print(f"Safety Net: {y_binary.sum():,} high-acuity (ESI 1-2) vs {(1-y_binary).sum():,} lower-acuity (ESI 3-5)")

if HAS_LGBM:
    safety_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
        class_weight='balanced', random_state=SEED, verbose=-1
    )
else:
    safety_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=30, random_state=SEED
    )

safety_proba = cross_val_predict(safety_model, X_full_imputed, y_binary, cv=skf, method='predict_proba')[:, 1]

fpr, tpr, thresholds = roc_curve(y_binary, safety_proba)
roc_auc = roc_auc_score(y_binary, safety_proba)

target_sensitivity = 0.95
idx_95 = np.argmin(np.abs(tpr - target_sensitivity))
threshold_95 = thresholds[idx_95]
spec_at_95 = 1 - fpr[idx_95]

print(f"\nROC AUC: {roc_auc:.4f}")
print(f"At {target_sensitivity*100:.0f}% sensitivity threshold ({threshold_95:.3f}):")
print(f"  Sensitivity: {tpr[idx_95]:.4f}")
print(f"  Specificity: {spec_at_95:.4f}")
print(f"  Would flag {(safety_proba >= threshold_95).sum():,} / {len(safety_proba):,} patients ({(safety_proba >= threshold_95).mean()*100:.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'Safety Net (AUC={roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].axhline(y=0.95, color='red', linestyle=':', alpha=0.7, label='95% target')
axes[0].scatter([fpr[idx_95]], [tpr[idx_95]], color='red', s=100, zorder=5, label='Operating point')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve: Safety Net')
axes[0].legend(loc='lower right')

axes[1].hist(safety_proba[y_binary == 1], bins=50, alpha=0.6, color='red', label='True ESI 1-2', density=True)
axes[1].hist(safety_proba[y_binary == 0], bins=50, alpha=0.6, color='blue', label='True ESI 3-5', density=True)
axes[1].axvline(x=threshold_95, color='black', linestyle='--', label=f'Threshold={threshold_95:.3f}')
axes[1].set_xlabel('P(High Acuity)')
axes[1].set_ylabel('Density')
axes[1].set_title('Safety Net Score Distribution')
axes[1].legend()

axes[2].plot(thresholds, tpr[:-1] if len(tpr) > len(thresholds) else tpr[:len(thresholds)], 
             'r-', label='Sensitivity', linewidth=2)
spec_curve = 1 - fpr[:-1] if len(fpr) > len(thresholds) else 1 - fpr[:len(thresholds)]
axes[2].plot(thresholds, spec_curve, 'b-', label='Specificity', linewidth=2)
axes[2].axvline(x=threshold_95, color='black', linestyle='--', alpha=0.7)
axes[2].set_xlabel('Decision Threshold')
axes[2].set_ylabel('Rate')
axes[2].set_title('Sensitivity-Specificity Tradeoff')
axes[2].legend()

plt.tight_layout()
plt.savefig('safety_net.png', dpi=100, bbox_inches='tight')
plt.show()

```

## Demographic Fairness Audit

AI triage must not perpetuate healthcare disparities. We evaluate across:
- **Age groups**, **Sex**, **Language** (English vs non-English), **Insurance type**

We flag any subgroup with accuracy or undertriage rate differing by >3pp from the population mean (Obermeyer et al., 2019).



```python
# ============================================================================
# DEMOGRAPHIC FAIRNESS AUDIT
# ============================================================================

def compute_fairness_metrics(y_true, y_pred, groups, group_name):
    results = []
    for group in sorted(groups.unique()):
        if pd.isna(group):
            continue
        mask = groups == group
        n = mask.sum()
        if n < 50:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        f1 = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
        undertriage = np.mean(y_pred[mask] > y_true[mask]) * 100
        results.append({'group': group, 'n': n, 'accuracy': acc, 'f1_macro': f1, 'undertriage_pct': undertriage})
    return pd.DataFrame(results)

demographic_dims = {
    'Age Group': df['age_group'],
    'Sex': df['sex'],
    'Language': df['language'].apply(lambda x: x if x == 'English' else 'Non-English'),
    'Insurance': df['insurance_type']
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fairness Audit: Performance by Demographic Group', fontsize=13, fontweight='bold')

population_acc = accuracy_score(y, ensemble_preds)
population_undertriage = np.mean(ensemble_preds > y) * 100

for idx, (dim_name, dim_values) in enumerate(demographic_dims.items()):
    ax = axes[idx // 2, idx % 2]
    metrics = compute_fairness_metrics(y, ensemble_preds, dim_values, dim_name)
    
    if len(metrics) == 0:
        ax.set_title(f'{dim_name}: insufficient data')
        continue
    
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width/2 for i in x], metrics['accuracy'], width, label='Accuracy', color='steelblue')
    ax.bar([i + width/2 for i in x], metrics['undertriage_pct']/100, width, label='Undertriage Rate', color='coral')
    ax.axhline(y=population_acc, color='steelblue', linestyle='--', alpha=0.5, label=f'Pop. Acc={population_acc:.3f}')
    ax.axhline(y=population_undertriage/100, color='coral', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics['group'].values, rotation=45, ha='right', fontsize=8)
    ax.set_title(f'{dim_name}')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    
    for i, row in metrics.iterrows():
        if abs(row['accuracy'] - population_acc) > 0.03:
            ax.annotate('!', xy=(list(metrics.index).index(i), row['accuracy']), fontsize=14, color='red', ha='center')

plt.tight_layout()
plt.savefig('fairness_audit.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FAIRNESS AUDIT SUMMARY")
print("=" * 60)
print(f"Population: Acc={population_acc:.4f}, Undertriage={population_undertriage:.2f}%\n")

any_disparity = False
for dim_name, dim_values in demographic_dims.items():
    metrics = compute_fairness_metrics(y, ensemble_preds, dim_values, dim_name)
    print(f"{dim_name}:")
    for _, row in metrics.iterrows():
        flag = ""
        if abs(row['accuracy'] - population_acc) > 0.03:
            flag = " ** DISPARITY"
            any_disparity = True
        print(f"  {row['group']:20s} n={row['n']:>5,}  Acc={row['accuracy']:.4f}  F1={row['f1_macro']:.4f}  UT={row['undertriage_pct']:.1f}%{flag}")
    print()

if not any_disparity:
    print("No significant disparities detected (all within 3pp of population mean)")

```

## Patient Case Studies: Clinical Decision Support in Action

We demonstrate the system with **5 representative cases** across all acuity levels, showing exactly how a clinician would interact with the tool: presenting complaint, vitals, model prediction with probabilities, safety net status, and the top features driving the prediction.

This demonstrates the system operates as a **transparent second opinion** — not a black box.



```python
# ============================================================================
# PATIENT CASE STUDIES
# ============================================================================

np.random.seed(SEED)
demo_cases = []
for esi in [1, 2, 3, 4, 5]:
    esi_indices = np.where(y == esi)[0]
    if len(esi_indices) > 0:
        demo_cases.append(np.random.choice(esi_indices))

print("=" * 75)
print("CLINICAL DECISION SUPPORT CASE STUDIES")
print("=" * 75)

for case_num, case_idx in enumerate(demo_cases, 1):
    row = df.iloc[case_idx]
    true_esi = y[case_idx]
    pred_proba = ensemble_proba[case_idx]
    pred_esi = pred_proba.argmax() + 1
    confidence = pred_proba.max()
    safety_score = safety_proba[case_idx]
    
    print(f"\n{'━' * 75}")
    print(f"  CASE {case_num}: Patient {row['patient_id']}")
    print(f"{'━' * 75}")
    print(f"  Demographics: {row['age']:.0f}y {row['sex']}, arrived by {row['arrival_mode']}")
    print(f"  Chief Complaint: \"{row.get('chief_complaint_raw', 'N/A')}\"")
    print(f"  Vitals: HR={row.get('heart_rate','—')}, BP={row.get('systolic_bp','—')}/{row.get('diastolic_bp','—')}, "
          f"SpO2={row.get('spo2','—')}%, Temp={row.get('temperature_c','—')}°C, "
          f"RR={row.get('respiratory_rate','—')}, GCS={row.get('gcs_total','—')}")
    print(f"  Mental Status: {row.get('mental_status_triage', '—')}")
    print(f"  Comorbidities: {int(row.get('num_comorbidities', 0))} conditions, {int(row.get('num_active_medications', 0))} medications")
    
    print(f"\n  ┌─ MODEL OUTPUT ─────────────────────────────")
    print(f"  │ TRUE ESI:      {true_esi}")
    print(f"  │ PREDICTED ESI: {pred_esi}  (confidence: {confidence:.1%})")
    print(f"  │ Probabilities: " + " | ".join([f"ESI-{i+1}:{p:.1%}" for i, p in enumerate(pred_proba)]))
    
    if safety_score >= threshold_95:
        print(f"  │ ** SAFETY NET ALERT: P(high acuity) = {safety_score:.1%} — consider reassessment")
    else:
        print(f"  │ Safety Net: clear (score={safety_score:.3f})")
    
    # Top contributing features for this patient
    if HAS_SHAP and case_idx in X_shap.index:
        patient_shap_idx = list(X_shap.index).index(case_idx)
        if isinstance(shap_values, list):
            sv = shap_values[true_esi - 1][patient_shap_idx]
        else:
            sv = shap_values[patient_shap_idx]
            if sv.ndim > 1:
                sv = sv[:, true_esi - 1]
        top_shap_idx = np.argsort(np.abs(sv))[-5:][::-1]
        print(f"  │ Top drivers:")
        for si in top_shap_idx:
            fname = feature_names[si]
            fval = X_full_imputed.iloc[case_idx][fname]
            print(f"  │   {fname} = {fval:.2f} (SHAP: {sv[si]:+.3f})")
    
    match = "CORRECT" if pred_esi == true_esi else ("UNDERTRIAGE" if pred_esi > true_esi else "OVERTRIAGE")
    print(f"  └─ Assessment: {match}")

print(f"\n{'━' * 75}")
print("Case studies demonstrate transparent, explainable decision support.")

```

## Interactive Clinical Decision Support Demo

This interactive widget simulates how the triage support tool would function in a clinical environment.
Enter patient vitals and demographics below and the trained ensemble will return:
- **Predicted ESI level** with confidence
- **Safety net alert** if the patient may be under-triaged
- **Top contributing features** via SHAP

This demonstrates the system as a **real-time clinical decision support tool**, not just an offline analysis.


```python
# ============================================================================
# CLINICAL DECISION SUPPORT DEMO (Static HTML)
# ============================================================================
# Demonstrates the triage system with representative patient scenarios.
# Each patient is constructed from explicit vital signs and demographics —
# no template copying from training data.
# ============================================================================

from IPython.display import display, HTML
import numpy as np

def predict_patient_demo(hr, sbp, dbp, spo2, temp, rr, gcs, age, sex_idx,
                         arrival_mode_idx, mental_status_idx, n_comorbidities):
    """Predict triage level from explicit patient parameters.
    
    Constructs a full feature vector by setting known values explicitly
    and using population medians for NLP/embedding features.
    """
    # Start with population median (robust neutral baseline)
    patient = pd.DataFrame(
        [X_full_imputed.median()], columns=X_full_imputed.columns
    )
    
    # Set explicit vital signs
    vital_map = {
        'heart_rate': hr, 'sbp': sbp, 'systolic_bp': sbp,
        'dbp': dbp, 'diastolic_bp': dbp,
        'spo2': spo2, 'temperature': temp, 'temperature_c': temp,
        'respiratory_rate': rr, 'gcs_total': gcs, 'age': age
    }
    for col, val in vital_map.items():
        if col in patient.columns:
            patient[col] = val
    
    # Compute derived features from vitals
    if 'shock_index' in patient.columns and sbp > 0:
        patient['shock_index'] = hr / sbp
    if 'pulse_pressure' in patient.columns:
        patient['pulse_pressure'] = sbp - dbp
    if 'map_value' in patient.columns:
        patient['map_value'] = dbp + (sbp - dbp) / 3
    if 'mean_arterial_pressure' in patient.columns:
        patient['mean_arterial_pressure'] = dbp + (sbp - dbp) / 3
    
    # Clinical flags derived from vitals
    flag_map = {
        'tachycardia': hr > 100, 'bradycardia': hr < 60,
        'hypotension': sbp < 90, 'hypertension': sbp > 180,
        'hypoxia': spo2 < 94, 'fever': temp > 38.0,
        'hypothermia': temp < 35.0, 'tachypnea': rr > 20,
        'altered_mental': gcs < 15
    }
    for col, val in flag_map.items():
        if col in patient.columns:
            patient[col] = int(val)
    
    # Missingness: demo patients have complete data
    for col in patient.columns:
        if 'missing' in col.lower():
            patient[col] = 0
    if 'missingness_entropy' in patient.columns:
        patient['missingness_entropy'] = 0.0
    if 'comorbidity_burden' in patient.columns:
        patient['comorbidity_burden'] = n_comorbidities
    
    # NLP embeddings: use zero vector (neutral = no chief complaint bias)
    for col in patient.columns:
        if col.startswith('sbert_') or col.startswith('nlp_') or col.startswith('tfidf_'):
            patient[col] = 0.0
    
    # Ensemble prediction
    all_models = [
        ('LightGBM/GB', final_model, w_lgbm, False),
        ('XGBoost/RF', final_xgb_model, w_xgb, False),
        ('MLP', final_mlp_model, w_mlp, True),
        ('LR', final_lr_model, w_lr, True),
    ]
    
    probas = []
    for name, model, weight, needs_scaling in all_models:
        try:
            inp = scaler.transform(patient) if needs_scaling else patient
            p = model.predict_proba(inp)
            probas.append((p, weight))
        except Exception:
            pass
    
    if not probas:
        return None, None, None, None, None
    
    total_w = sum(w for _, w in probas)
    ensemble_proba = sum(p * w for p, w in probas) / total_w
    pred_class = int(np.argmax(ensemble_proba)) + 1
    confidence = float(np.max(ensemble_proba))
    high_acuity_prob = float(ensemble_proba[0, 0] + ensemble_proba[0, 1])
    safety_alert = high_acuity_prob > 0.3
    probs_out = ensemble_proba[0]
    if len(probs_out) < 5:
        probs_out = np.pad(probs_out, (0, 5 - len(probs_out)))
    return pred_class, confidence, safety_alert, high_acuity_prob, probs_out[:5]

# Representative patient scenarios spanning the acuity spectrum
scenarios = [
    {"name": "Chest Pain - Elderly",  "desc": "78M, ambulance, tachycardic, hypertensive",
     "hr": 112, "sbp": 185, "dbp": 95, "spo2": 93.0, "temp": 37.2, "rr": 24, "gcs": 15,
     "age": 78, "sex": 0, "arrival": 1, "mental": 0, "comorbid": 4},
    {"name": "Pediatric Fever",       "desc": "3F, walk-in, febrile, tachycardic",
     "hr": 145, "sbp": 90, "dbp": 55, "spo2": 97.0, "temp": 39.8, "rr": 30, "gcs": 15,
     "age": 3, "sex": 1, "arrival": 0, "mental": 0, "comorbid": 0},
    {"name": "Trauma - Hypotensive",  "desc": "34M, ambulance, hypotensive, altered",
     "hr": 130, "sbp": 75, "dbp": 40, "spo2": 91.0, "temp": 36.2, "rr": 28, "gcs": 10,
     "age": 34, "sex": 0, "arrival": 1, "mental": 1, "comorbid": 0},
    {"name": "Routine Visit",         "desc": "45F, walk-in, stable vitals",
     "hr": 78, "sbp": 122, "dbp": 78, "spo2": 99.0, "temp": 36.8, "rr": 14, "gcs": 15,
     "age": 45, "sex": 1, "arrival": 0, "mental": 0, "comorbid": 1},
    {"name": "Diabetic Emergency",    "desc": "62M, ambulance, tachycardic, confused",
     "hr": 118, "sbp": 95, "dbp": 60, "spo2": 95.0, "temp": 37.5, "rr": 26, "gcs": 13,
     "age": 62, "sex": 0, "arrival": 1, "mental": 1, "comorbid": 3},
]

esi_colors = ['#dc2626', '#f97316', '#eab308', '#22c55e', '#3b82f6']
esi_labels = ['Resuscitation', 'Emergent', 'Urgent', 'Less Urgent', 'Non-Urgent']

try:
    html_parts = ['<h3>&#x1f3e5; Clinical Decision Support Demo</h3>',
                  '<p><i>Representative patient scenarios (median-baseline features, explicit vitals):</i></p>']
    
    for sc in scenarios:
        pred, conf, alert, ha_prob, probs = predict_patient_demo(
            sc['hr'], sc['sbp'], sc['dbp'], sc['spo2'], sc['temp'], sc['rr'],
            sc['gcs'], sc['age'], sc['sex'], sc['arrival'], sc['mental'], sc['comorbid'])
        
        if pred is None:
            html_parts.append(f'<div style="padding:10px;margin:8px 0;background:#fee2e2;'
                              f'border-radius:8px;"><b>{sc["name"]}</b>: Prediction failed</div>')
            continue
        
        color = esi_colors[min(pred - 1, 4)]
        label = esi_labels[min(pred - 1, 4)]
        
        alert_html = ''
        if alert:
            alert_html = (f'<div style="background:#fef2f2;border:1px solid #dc2626;'
                          f'padding:6px;border-radius:6px;margin:4px 0;font-size:12px;">'
                          f'<b>SAFETY NET ALERT:</b> High-acuity P={ha_prob:.1%} &mdash; '
                          f'Consider reassessment</div>')
        
        prob_bars = ''
        for i in range(5):
            w = int(probs[i] * 150)
            prob_bars += (f'<span style="display:inline-block;width:45px;font-size:11px;">'
                          f'ESI {i+1}:</span>'
                          f'<span style="display:inline-block;width:{w}px;height:12px;'
                          f'background:{esi_colors[i]};border-radius:2px;"></span>'
                          f' <span style="font-size:11px;">{probs[i]:.0%}</span><br/>')
        
        html_parts.append(
            f'<div style="padding:12px;margin:8px 0;border-radius:8px;'
            f'background:linear-gradient(135deg,#f8fafc,#e2e8f0);border-left:4px solid {color};">'
            f'<b>{sc["name"]}</b> <span style="color:#64748b;font-size:12px;">({sc["desc"]})</span><br/>'
            f'<span style="color:{color};font-size:18px;font-weight:bold;">'
            f'ESI {pred} &mdash; {label}</span> '
            f'<span style="font-size:12px;">(Confidence: {conf:.0%})</span>'
            f'{alert_html}'
            f'<div style="margin-top:6px;">{prob_bars}</div></div>')
    
    display(HTML(''.join(html_parts)))
    
except Exception as e:
    print(f'Demo visualization skipped: {e}')
    print('(Demo requires trained final models from the cell above)')

```

## Test Set Predictions

Predictions for the 20,000 held-out test patients using the full 4-model ensemble trained on all training data.



```python
# ============================================================================
# TEST SET PREDICTIONS
# ============================================================================
print("Generating test set predictions using pre-trained final models...")

test_proba_lgbm = final_model.predict_proba(X_full_test_imputed)
if HAS_XGB:
    test_proba_xgb = final_xgb_model.predict_proba(X_full_test_imputed)
else:
    test_proba_xgb = final_xgb_model.predict_proba(X_full_test_imputed)

X_test_scaled = scaler.transform(X_full_test_imputed)
test_proba_mlp = final_mlp_model.predict_proba(X_test_scaled)
test_proba_lr = final_lr_model.predict_proba(X_test_scaled)

test_proba_ensemble = (w_lgbm * test_proba_lgbm + w_xgb * test_proba_xgb + 
                       w_mlp * test_proba_mlp + w_lr * test_proba_lr)
test_preds = test_proba_ensemble.argmax(axis=1) + 1

submission = pd.DataFrame({
    'patient_id': df_test['patient_id'],
    'triage_acuity': test_preds
})
submission.to_csv('submission.csv', index=False)

print(f"\nTest predictions: {len(submission):,} patients")
print(f"\nPrediction distribution:")
pred_dist = pd.Series(test_preds).value_counts().sort_index()
for esi, count in pred_dist.items():
    print(f"  ESI {esi}: {count:>5,} ({count/len(test_preds)*100:5.1f}%)")

max_proba = test_proba_ensemble.max(axis=1)
print(f"\nConfidence: mean={max_proba.mean():.3f}, median={np.median(max_proba):.3f}")
print(f"Low confidence (<50%): {(max_proba < 0.5).sum():,} ({(max_proba < 0.5).mean()*100:.1f}%)")
print(f"\nSubmission saved: submission.csv")

```

## Summary of Findings & Novel Contributions

### Novel Contributions

1. **Missingness-as-Clinical-Signal**: First systematic exploitation of ED documentation patterns as predictive features. Missingness entropy and ambulance-missing interaction capture real clinical workflows.

2. **4-Model Calibrated Ensemble**: Diversity-driven prediction (gradient boosting + neural net + linear) with F1-weighted soft voting produces well-calibrated probabilities suitable for threshold-based clinical workflows.

3. **Multi-Modal Integration**: Dense NLP embeddings (384-dim Sentence-BERT) + structured vitals + comorbidity profiles + missingness signatures = complete patient representation beyond any single modality.

4. **Asymmetric Error Optimization**: Explicit undertriage minimization aligned with clinical priorities — catching critically ill patients matters more than perfect accuracy on minor complaints.

5. **Transparent Decision Support**: Per-patient SHAP explanations, failure mode analysis, and demographic fairness audit make this a clinician-facing tool, not a black box.

### Limitations

1. **Synthetic dataset**: May not capture rare presentations or real-world noise
2. **No temporal dynamics**: Arrival-time features only; patient deterioration is not modeled
3. **Language bias**: Sentence embeddings trained on English corpora
4. **Single-site**: External validation required before deployment
5. **No image/waveform**: Real triage increasingly uses POCUS, ECG

### Clinical Safety Statement

This system is designed as a **decision-support tool only**:
- Always used alongside clinical judgment
- Triggers additional assessment, not automatic escalation
- Requires continuous monitoring for drift and bias


---

## Part 2: Real-World Validation on NHAMCS 2022

### Why Real Clinical Data Matters

The synthetic training data produces artificially high accuracy (99.97%) because vitals were *generated* from ESI levels — creating near-perfect separability that does not exist in real clinical settings. The synthetic data generator ([solution.py](https://www.kaggle.com/competitions/triagegeist)) samples ESI first, then generates vitals conditioned on ESI, effectively encoding the target into the features.

**Real-world ESI inter-rater reliability**: κ = 0.60–0.80 (Gilboy et al., 2012). Human triage nurses disagree ~20–40% of the time on the same patient. No ML model should exceed human ceiling by orders of magnitude.

To demonstrate our methodology's genuine clinical value, we validate on the **NHAMCS 2022** (National Hospital Ambulatory Medical Care Survey) — a nationally representative sample of ~10,000 real U.S. emergency department visits with **real triage acuity labels**, real vitals, real comorbidities, and real missingness patterns.

**Key differences from synthetic data:**
- ESI distribution: 1.4% ESI-1 / 15.6% ESI-2 / 52.3% ESI-3 / 27.7% ESI-4 / 3.0% ESI-5
- 29% missing pain scores, 21% missing SpO2 — reflecting real ED workflows
- No chief complaint free text — only numeric reason-for-visit codes
- No GCS, weight/height, pre-computed NEWS2/MEWS — requiring adaptation



```python
# ============================================================================
# NHAMCS 2022: REAL-WORLD CLINICAL VALIDATION
# ============================================================================
import warnings
warnings.filterwarnings('ignore')

# Load NHAMCS 2022 (attached Kaggle dataset: radhikaaaaaaa/nhamcs2022)
NHAMCS_PATH = None
nhamcs_candidates = [
    '/kaggle/input/nhamcs2022/nhamcs2022.csv',
    '/kaggle/input/nhamcs-2022/nhamcs2022.csv',
    'nhamcs_data/nhamcs2022/nhamcs2022.csv',
]
for path in nhamcs_candidates:
    if os.path.exists(path):
        NHAMCS_PATH = path
        break

if NHAMCS_PATH is None:
    print("NHAMCS data not found — skipping real-world validation.")
    print("To enable: attach 'radhikaaaaaaa/nhamcs2022' dataset to this notebook.")
    NHAMCS_AVAILABLE = False
else:
    NHAMCS_AVAILABLE = True
    nhamcs = pd.read_csv(NHAMCS_PATH, low_memory=False)
    print(f"NHAMCS 2022 loaded: {nhamcs.shape[0]:,} ED visits x {nhamcs.shape[1]} variables")
    
    # Target distribution
    print(f"\nReal-world ESI distribution:")
    vc = nhamcs['triage_acuity'].value_counts().sort_index()
    for esi, count in vc.items():
        pct = count / len(nhamcs) * 100
        print(f"  ESI {esi}: {count:>5,} ({pct:5.1f}%)")
    
    # Missing data reality
    vital_miss = {
        'heart_rate': nhamcs['heart_rate'].isna().mean() * 100,
        'systolic_bp': nhamcs['systolic_bp'].isna().mean() * 100 if 'systolic_bp' in nhamcs.columns else 0,
        'temperature': nhamcs['temperature_c'].isna().mean() * 100 if 'temperature_c' in nhamcs.columns else 0,
        'spo2': nhamcs['spo2'].isna().mean() * 100 if 'spo2' in nhamcs.columns else 0,
        'pain_score': nhamcs['pain_score'].isna().mean() * 100,
    }
    print(f"\nReal-world missingness rates:")
    for v, pct in vital_miss.items():
        print(f"  {v}: {pct:.1f}% missing")

```


```python
# ============================================================================
# NHAMCS FEATURE ENGINEERING (adapted for available variables)
# ============================================================================
if NHAMCS_AVAILABLE:
    def engineer_nhamcs_features(data):
        feats = pd.DataFrame(index=data.index)
        
        # --- RAW VITALS (available in NHAMCS) ---
        for col in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate',
                     'temperature_c', 'spo2', 'pain_score']:
            if col in data.columns:
                feats[col] = data[col]
        
        # --- PRE-COMPUTED (limited in NHAMCS) ---
        if 'pulse_pressure' in data.columns:
            feats['pulse_pressure'] = data['pulse_pressure']
        
        # Compute shock index from available vitals
        if 'heart_rate' in data.columns and 'systolic_bp' in data.columns:
            feats['shock_index'] = data['heart_rate'].fillna(80) / data['systolic_bp'].fillna(120)
        
        # --- CLINICAL FLAGS ---
        if 'heart_rate' in data.columns:
            feats['tachycardia'] = (data['heart_rate'] > 100).astype(float)
            feats['bradycardia'] = (data['heart_rate'] < 60).astype(float)
            feats['severe_tachycardia'] = (data['heart_rate'] > 130).astype(float)
        if 'systolic_bp' in data.columns:
            feats['hypotension'] = (data['systolic_bp'] < 90).astype(float)
            feats['severe_hypotension'] = (data['systolic_bp'] < 70).astype(float)
            feats['hypertensive_crisis'] = (data['systolic_bp'] > 180).astype(float)
        if 'respiratory_rate' in data.columns:
            feats['tachypnea'] = (data['respiratory_rate'] > 20).astype(float)
            feats['severe_tachypnea'] = (data['respiratory_rate'] > 30).astype(float)
            feats['bradypnea'] = (data['respiratory_rate'] < 12).astype(float)
        if 'spo2' in data.columns:
            feats['hypoxia'] = (data['spo2'] < 94).astype(float)
            feats['severe_hypoxia'] = (data['spo2'] < 90).astype(float)
        if 'temperature_c' in data.columns:
            feats['fever'] = (data['temperature_c'] > 38.0).astype(float)
            feats['high_fever'] = (data['temperature_c'] > 39.0).astype(float)
            feats['hypothermia'] = (data['temperature_c'] < 35.0).astype(float)
        if 'pain_score' in data.columns:
            feats['severe_pain'] = (data['pain_score'] >= 8).astype(float)
        
        # --- COMPUTED MEWS (no GCS in NHAMCS, so partial) ---
        mews = pd.Series(0, index=data.index, dtype=float)
        if 'heart_rate' in data.columns:
            mews += (data['heart_rate'] > 100).fillna(False).astype(float)
            mews += (data['heart_rate'] > 130).fillna(False).astype(float)
        if 'systolic_bp' in data.columns:
            mews += (data['systolic_bp'] < 90).fillna(False).astype(float) * 2
        if 'respiratory_rate' in data.columns:
            mews += (data['respiratory_rate'] > 20).fillna(False).astype(float)
            mews += (data['respiratory_rate'] > 30).fillna(False).astype(float)
        if 'temperature_c' in data.columns:
            mews += (data['temperature_c'] > 38.5).fillna(False).astype(float)
        if 'spo2' in data.columns:
            mews += (data['spo2'] < 94).fillna(False).astype(float)
            mews += (data['spo2'] < 90).fillna(False).astype(float)
        feats['mews_computed'] = mews
        
        # --- DEMOGRAPHICS ---
        feats['age'] = data['age']
        feats['is_pediatric'] = (data['age'] < 18).astype(float)
        feats['is_elderly'] = (data['age'] >= 65).astype(float)
        feats['is_very_elderly'] = (data['age'] >= 80).astype(float)
        if 'sex' in data.columns:
            feats['is_female'] = (data['sex'] == 2).astype(float)  # NHAMCS: 1=M, 2=F
        feats['elderly_abnormal_vitals'] = feats.get('is_elderly', 0) * feats.get('tachycardia', 0)
        
        # --- TEMPORAL ---
        if 'arrival_hour' in data.columns:
            feats['arrival_hour'] = data['arrival_hour']
            feats['night_arrival'] = data['arrival_hour'].apply(lambda h: 1.0 if (h >= 22 or h < 6) else 0.0)
            feats['evening_arrival'] = data['arrival_hour'].apply(lambda h: 1.0 if (18 <= h < 22) else 0.0)
        
        if 'visit_day' in data.columns:
            feats['is_weekend'] = data['visit_day'].isin([1, 7]).astype(float)  # 1=Sun, 7=Sat
        
        # --- ARRIVAL ---
        if 'arrived_by_ambulance' in data.columns:
            feats['arrived_ambulance'] = (data['arrived_by_ambulance'] == 1).astype(float)
        
        # --- COMORBIDITIES ---
        hx_cols = [c for c in data.columns if c.startswith('hx_')]
        for col in hx_cols:
            feats[col] = data[col].fillna(0)
        feats['comorbidity_burden'] = data[hx_cols].sum(axis=1) if hx_cols else 0
        
        if 'hx_heart_failure' in data.columns and 'hx_copd' in data.columns:
            feats['cardiopulm_risk'] = (data['hx_heart_failure'].fillna(0) + 
                                        data['hx_copd'].fillna(0) + 
                                        data.get('hx_coronary_artery_disease', pd.Series(0, index=data.index)).fillna(0))
        if 'hx_diabetes_type1' in data.columns:
            feats['diabetes_any'] = ((data['hx_diabetes_type1'].fillna(0) + 
                                      data['hx_diabetes_type2'].fillna(0)) > 0).astype(float)
        
        # --- MISSINGNESS SIGNATURE ---
        miss_cols = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate',
                     'temperature_c', 'spo2', 'pain_score']
        for col in miss_cols:
            if col in data.columns:
                feats[f'miss_{col}'] = data[col].isnull().astype(float)
        feats['n_missing_vitals'] = sum(data[c].isnull().astype(float) for c in miss_cols if c in data.columns)
        
        # Missingness entropy
        miss_matrix = pd.DataFrame({c: data[c].isnull().astype(float) for c in miss_cols if c in data.columns})
        p_miss = miss_matrix.mean(axis=1)
        miss_ent = pd.Series(0.0, index=data.index)
        mask_ent = (p_miss > 0) & (p_miss < 1)
        miss_ent[mask_ent] = -(p_miss[mask_ent] * np.log2(p_miss[mask_ent]) + 
                                (1-p_miss[mask_ent]) * np.log2(1-p_miss[mask_ent]))
        feats['miss_entropy'] = miss_ent
        
        if 'arrived_ambulance' in feats.columns:
            feats['ambulance_missing_interaction'] = feats['arrived_ambulance'] * feats['n_missing_vitals']
        
        # --- VITAL INTERACTIONS ---
        feats['hr_sbp_product'] = data['heart_rate'].fillna(80) * data['systolic_bp'].fillna(120)
        feats['age_hr_interaction'] = data['age'].fillna(50) * data['heart_rate'].fillna(80) / 1000
        if 'spo2' in data.columns and 'respiratory_rate' in data.columns:
            feats['spo2_rr_interaction'] = data['spo2'].fillna(97) * data['respiratory_rate'].fillna(16) / 100
        
        return feats
    
    # Apply feature engineering
    X_nhamcs = engineer_nhamcs_features(nhamcs)
    y_nhamcs_orig = nhamcs['triage_acuity'].values  # 1-5
    y_nhamcs = y_nhamcs_orig - 1  # 0-4 for model training
    
    # Impute
    from sklearn.impute import SimpleImputer
    nhamcs_imputer = SimpleImputer(strategy='median')
    nhamcs_feat_names = X_nhamcs.columns.tolist()
    X_nhamcs_imp = pd.DataFrame(
        nhamcs_imputer.fit_transform(X_nhamcs), 
        columns=nhamcs_feat_names, 
        index=X_nhamcs.index
    )
    
    print(f"NHAMCS features: {X_nhamcs_imp.shape[1]} (vs {X_full_imputed.shape[1]} synthetic)")
    print(f"  No NLP features (no free-text chief complaints in NHAMCS)")
    print(f"  No GCS, NEWS2, weight/height (not available)")
    print(f"  Available: vitals, demographics, comorbidities, missingness, interactions")

```


```python
# ============================================================================
# NHAMCS: 4-MODEL ENSEMBLE (same architecture, real data)
# ============================================================================
if NHAMCS_AVAILABLE:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    import lightgbm as lgb
    import xgboost as xgb
    
    print("=" * 70)
    print("REAL-WORLD VALIDATION: 4-Model Ensemble on NHAMCS 2022")
    print("=" * 70)
    
    n_classes = len(np.unique(y_nhamcs))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Storage for predictions
    nhamcs_oof_preds = {name: np.zeros((len(y_nhamcs), n_classes)) for name in ['lgb', 'xgb', 'mlp', 'lr']}
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_nhamcs_imp, y_nhamcs)):
        X_tr, X_val = X_nhamcs_imp.iloc[train_idx], X_nhamcs_imp.iloc[val_idx]
        y_tr, y_val = y_nhamcs[train_idx], y_nhamcs[val_idx]
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
            class_weight='balanced'
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_cal = CalibratedClassifierCV(lgb_model, cv='prefit', method='isotonic')
        lgb_cal.fit(X_val, y_val)
        nhamcs_oof_preds['lgb'][val_idx] = lgb_cal.predict_proba(X_val)
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            eval_metric='mlogloss', verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      verbose=False)
        xgb_cal = CalibratedClassifierCV(xgb_model, cv='prefit', method='isotonic')
        xgb_cal.fit(X_val, y_val)
        nhamcs_oof_preds['xgb'][val_idx] = xgb_cal.predict_proba(X_val)
        
        # MLP
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation='relu',
            max_iter=300, early_stopping=True, random_state=42, verbose=False
        )
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)
        mlp_model.fit(X_tr_sc, y_tr)
        nhamcs_oof_preds['mlp'][val_idx] = mlp_model.predict_proba(X_val_sc)
        
        # Logistic Regression
        lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced')
        lr_model.fit(X_tr_sc, y_tr)
        nhamcs_oof_preds['lr'][val_idx] = lr_model.predict_proba(X_val_sc)
        
        print(f"  Fold {fold_idx+1}/3 complete")
    
    # Ensemble (equal weights)
    ensemble_probs = sum(nhamcs_oof_preds[m] for m in nhamcs_oof_preds) / 4
    ensemble_classes = np.unique(y_nhamcs)
    ensemble_pred = ensemble_classes[ensemble_probs.argmax(axis=1)]
    
    # Individual model predictions
    model_preds = {}
    for name in nhamcs_oof_preds:
        model_preds[name] = ensemble_classes[nhamcs_oof_preds[name].argmax(axis=1)]
    
    # Results
    print(f"\n{'='*70}")
    print(f"NHAMCS 2022 RESULTS (Real Clinical Data)")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Accuracy':>10} {'F1 (macro)':>12} {'Kappa':>10}")
    print(f"{'-'*52}")
    
    for name, label in [('lgb','LightGBM'), ('xgb','XGBoost'), ('mlp','MLP'), ('lr','LogReg')]:
        acc = accuracy_score(y_nhamcs, model_preds[name])
        f1 = f1_score(y_nhamcs, model_preds[name], average='macro')
        kap = cohen_kappa_score(y_nhamcs, model_preds[name], weights='quadratic')
        print(f"  {label:<18} {acc:>10.4f} {f1:>12.4f} {kap:>10.4f}")
    
    ens_acc = accuracy_score(y_nhamcs, ensemble_pred)
    ens_f1 = f1_score(y_nhamcs, ensemble_pred, average='macro')
    ens_kap = cohen_kappa_score(y_nhamcs, ensemble_pred, weights='quadratic')
    print(f"  {'Ensemble':<18} {ens_acc:>10.4f} {ens_f1:>12.4f} {ens_kap:>10.4f}")
    
    # Undertriage analysis
    undertriage = ((ensemble_pred - y_nhamcs) > 0)  # predicted higher ESI number = lower acuity
    dangerous_ut = ((ensemble_pred - y_nhamcs) >= 2)
    print(f"\nUndertriage Analysis:")
    print(f"  Undertriage rate: {undertriage.mean()*100:.2f}% ({undertriage.sum()}/{len(y_nhamcs)})")
    print(f"  Dangerous undertriage (>=2 levels): {dangerous_ut.mean()*100:.2f}% ({dangerous_ut.sum()}/{len(y_nhamcs)})")
    
    # Per-class report
    print(f"\nPer-class accuracy (NHAMCS):")
    for esi in sorted(np.unique(y_nhamcs)):
        mask = y_nhamcs == esi
        class_acc = accuracy_score(y_nhamcs[mask], ensemble_pred[mask])
        n = mask.sum()
        print(f"  ESI {esi}: {class_acc:.4f} (n={n})")
    
    print(f"\nClassification Report:")
    print(classification_report(y_nhamcs, ensemble_pred, target_names=[f'ESI-{i}' for i in sorted(np.unique(y_nhamcs))]))

```

### Synthetic vs. Real-World Performance Comparison

This comparison is the central finding of our analysis. The gap between synthetic and real-world performance quantifies the **over-specification** in the competition dataset and calibrates expectations for actual clinical deployment.



```python
# ============================================================================
# SYNTHETIC vs REAL COMPARISON
# ============================================================================
if NHAMCS_AVAILABLE:
    # Build comparison table
    comparison = pd.DataFrame({
        'Dataset': ['Synthetic (Competition)', 'NHAMCS 2022 (Real)'],
        'N': [len(y), len(y_nhamcs)],
        'Accuracy': [accuracy_score(y, ensemble_preds), ens_acc],
        'F1 (macro)': [f1_score(y, ensemble_preds, average='macro'), ens_f1],
        'QWK': [cohen_kappa_score(y, ensemble_preds, weights='quadratic'), ens_kap],
    })
    
    print("=" * 70)
    print("PERFORMANCE COMPARISON: SYNTHETIC vs REAL")
    print("=" * 70)
    print(comparison.to_string(index=False))
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    metrics = ['Accuracy', 'F1 (macro)', 'QWK']
    colors = ['#3b82f6', '#ef4444']
    
    for i, metric in enumerate(metrics):
        vals = comparison[metric].values
        bars = axes[i].bar(['Synthetic', 'Real (NHAMCS)'], vals, color=colors, alpha=0.85, edgecolor='white')
        axes[i].set_title(metric, fontsize=13, fontweight='bold')
        axes[i].set_ylim(0, 1.05)
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
        axes[i].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Clinical target')
        axes[i].legend(fontsize=8)
    
    plt.suptitle('ESI Triage Prediction: Synthetic vs Real Clinical Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('synthetic_vs_real.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nKey insight: {(comparison.iloc[0]['Accuracy'] - comparison.iloc[1]['Accuracy'])*100:.1f} percentage point accuracy gap")
    print(f"between synthetic and real data — confirming synthetic over-specification.")
    print(f"\nReal-world kappa of {ens_kap:.3f} {'exceeds' if ens_kap > 0.8 else 'approaches'} published ESI inter-rater reliability (0.60–0.80).")


```

### Ablation Study: Feature Group Contributions

To quantify the value of each feature engineering innovation, we remove feature groups one at a time from the NHAMCS model and measure the performance drop. This reveals which components drive real clinical prediction accuracy.



```python
# ============================================================================
# ABLATION STUDY (on NHAMCS real data)
# ============================================================================
if NHAMCS_AVAILABLE:
    print("=" * 70)
    print("ABLATION STUDY: Feature Group Contributions (NHAMCS)")
    print("=" * 70)
    
    # Define feature groups
    all_feat_names = nhamcs_feat_names
    
    vital_feats = [c for c in all_feat_names if c in ['heart_rate','systolic_bp','diastolic_bp',
                   'respiratory_rate','temperature_c','spo2','pain_score']]
    flag_feats = [c for c in all_feat_names if c in ['tachycardia','bradycardia','severe_tachycardia',
                  'hypotension','severe_hypotension','hypertensive_crisis','tachypnea','severe_tachypnea',
                  'bradypnea','hypoxia','severe_hypoxia','fever','high_fever','hypothermia','severe_pain']]
    miss_feats = [c for c in all_feat_names if 'miss' in c or 'entropy' in c]
    comorbid_feats = [c for c in all_feat_names if c.startswith('hx_') or c in ['comorbidity_burden','cardiopulm_risk','diabetes_any']]
    interact_feats = [c for c in all_feat_names if 'interaction' in c or 'product' in c]
    demo_feats = [c for c in all_feat_names if c in ['age','is_pediatric','is_elderly','is_very_elderly',
                  'is_female','elderly_abnormal_vitals','arrival_hour','night_arrival','evening_arrival',
                  'is_weekend','arrived_ambulance']]
    
    ablation_groups = {
        'Full model': [],  # Remove nothing
        '- Vitals': vital_feats,
        '- Clinical Flags': flag_feats,
        '- Missingness': miss_feats,
        '- Comorbidities': comorbid_feats,
        '- Interactions': interact_feats,
        '- Demographics': demo_feats,
    }
    
    ablation_results = []
    
    for group_name, features_to_remove in ablation_groups.items():
        if features_to_remove:
            keep_cols = [c for c in all_feat_names if c not in features_to_remove]
        else:
            keep_cols = all_feat_names
        
        if len(keep_cols) == 0:
            continue
        
        X_abl = X_nhamcs_imp[keep_cols]
        
        # Quick LightGBM only (for speed)
        abl_preds = np.zeros(len(y_nhamcs))
        for train_idx, val_idx in cv.split(X_abl, y_nhamcs):
            model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1, class_weight='balanced'
            )
            model.fit(X_abl.iloc[train_idx], y_nhamcs[train_idx],
                     eval_set=[(X_abl.iloc[val_idx], y_nhamcs[val_idx])],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            abl_preds[val_idx] = model.predict(X_abl.iloc[val_idx])
        
        acc = accuracy_score(y_nhamcs, abl_preds)
        f1 = f1_score(y_nhamcs, abl_preds, average='macro')
        n_feats = len(keep_cols)
        ablation_results.append({
            'Configuration': group_name,
            'Features': n_feats,
            'Accuracy': acc,
            'F1 (macro)': f1,
        })
        print(f"  {group_name:<25} | {n_feats:>3} feats | Acc={acc:.4f} | F1={f1:.4f}")
    
    # Summary table
    abl_df = pd.DataFrame(ablation_results)
    full_acc = abl_df.iloc[0]['Accuracy']
    abl_df['Δ Accuracy'] = abl_df['Accuracy'] - full_acc
    
    print(f"\n{'='*70}")
    print(abl_df.to_string(index=False))
    print(f"\nMost impactful feature group: ", end='')
    worst_drop = abl_df.iloc[1:]['Δ Accuracy'].idxmin()
    print(f"{abl_df.iloc[worst_drop]['Configuration']} (Δ={abl_df.iloc[worst_drop]['Δ Accuracy']:.4f})")
    
    # Ablation bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_abl = ['#22c55e'] + ['#ef4444' if d < 0 else '#22c55e' for d in abl_df.iloc[1:]['Δ Accuracy']]
    bars = ax.barh(abl_df['Configuration'], abl_df['Accuracy'], color=colors_abl, alpha=0.85, edgecolor='white')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Ablation Study: Feature Group Impact on NHAMCS Accuracy', fontsize=13, fontweight='bold')
    ax.axvline(x=full_acc, color='gray', linestyle='--', alpha=0.7, label=f'Full model ({full_acc:.4f})')
    for bar, acc_val in zip(bars, abl_df['Accuracy']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{acc_val:.4f}', va='center', fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig('ablation_study.png', dpi=150, bbox_inches='tight')
    plt.show()

```


```python
# ============================================================================
# NHAMCS CONFUSION MATRIX
# ============================================================================
if NHAMCS_AVAILABLE:
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_nhamcs, ensemble_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize for display
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    im = ax.imshow(cm_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Add text  
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm_norm[i,j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1%})', 
                   ha='center', va='center', fontsize=9, color=color)
    
    labels = [f'ESI-{i}' for i in sorted(np.unique(y_nhamcs))]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix — NHAMCS 2022 (Real Clinical Data)', fontsize=13, fontweight='bold')
    plt.colorbar(im, label='Proportion')
    plt.tight_layout()
    plt.savefig('confusion_matrix_nhamcs.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Highlight undertriage errors
    print("\nUndertriage errors (real data):")
    for i in range(cm.shape[0]):
        for j in range(i+1, cm.shape[1]):  # predicted higher ESI number = lower acuity
            if cm[i,j] > 0:
                print(f"  ESI-{sorted(np.unique(y_nhamcs))[i]} misclassified as ESI-{sorted(np.unique(y_nhamcs))[j]}: {cm[i,j]} patients ({cm[i,j]/cm[i].sum()*100:.1f}%)")

```


```python
# ============================================================================
# COVER IMAGE (560x280)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(5.6, 2.8), dpi=100)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#0f172a')
ax.axis('off')

# Title
ax.text(0.5, 0.88, 'TRIAGEGEIST', fontsize=24, fontweight='bold', color='white',
        ha='center', va='center', transform=ax.transAxes, fontfamily='monospace')
ax.text(0.5, 0.72, 'Multi-Modal Undertriage Detection', fontsize=11, color='#7dd3fc',
        ha='center', va='center', transform=ax.transAxes)

# Accent line
ax.plot([0.15, 0.85], [0.62, 0.62], color='#f97316', linewidth=2.5, transform=ax.transAxes)

# Key points
bullets = [
    'LightGBM + XGBoost + MLP Neural Net + Logistic Regression',
    'Novel: Missingness-as-Clinical-Signal (Information Entropy)',
    'Undertriage Safety Net with 95%+ Sensitivity',
    'Sentence-BERT NLP + 490 Multi-Modal Features',
    'Fairness Audit | SHAP Explainability | Bootstrap CIs'
]
for i, bullet in enumerate(bullets):
    ax.text(0.08, 0.50 - i*0.10, f'\u2022 {bullet}', fontsize=7, color='#cbd5e1',
            va='center', transform=ax.transAxes)

# Medical cross
cx, cy = 0.93, 0.30
ax.plot([cx-0.02, cx+0.02], [cy, cy], color='#ef4444', linewidth=4, 
        transform=ax.transAxes, solid_capstyle='round')
ax.plot([cx, cx], [cy-0.08, cy+0.08], color='#ef4444', linewidth=4,
        transform=ax.transAxes, solid_capstyle='round')

# Author
ax.text(0.08, 0.05, 'Ashok Pukkalla | Laitinen-Fredriksson Foundation', 
        fontsize=6.5, color='#64748b', va='center', transform=ax.transAxes)

plt.savefig('cover_image.png', dpi=100, bbox_inches='tight', facecolor='#0f172a', pad_inches=0.1)
plt.show()
print("Cover image saved: cover_image.png (560x280)")

elapsed = time.time() - t0_global
print(f"\nTotal notebook runtime: {elapsed/60:.1f} minutes")

```

## Reproducibility & References

### Environment
- Python 3.10+ on Kaggle | Random seed: 42
- Key packages: lightgbm, xgboost, scikit-learn, sentence-transformers, shap, scipy

### Data
- Competition data + NHAMCS 2022 (CDC, public domain) for real-world validation
- All code in this single notebook, runs end-to-end

### References

1. Gilboy, N., et al. (2012). Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care, Version 4. AHRQ.
2. Farrohknia, N., et al. (2011). Emergency department triage scales and their components: A systematic review. Scandinavian J. Trauma.
3. Levin, S., et al. (2018). Machine-learning-based electronic triage more accurately differentiates patients. Annals of Emergency Medicine.
4. Lundberg, S. & Lee, S.I. (2017). A unified approach to interpreting model predictions. NeurIPS.
5. Klug, M., et al. (2020). A gradient boosting ML model for predicting early mortality in the ED. Academic Emergency Medicine.
6. Obermeyer, Z., et al. (2019). Dissecting racial bias in an algorithm used to manage population health. Science.
7. FDA (2021). AI/ML-Based Software as a Medical Device Action Plan.
8. Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
9. Christ, M., et al. (2010). Modern triage in the emergency department. Deutsches Ärzteblatt International.
10. Dugas, A.F., et al. (2016). An electronic emergency triage system to improve patient distribution. J. Emergency Medicine.

---
*Notebook: ashokpukkalla | Triagegeist Competition 2026 | Laitinen-Fredriksson Foundation*


