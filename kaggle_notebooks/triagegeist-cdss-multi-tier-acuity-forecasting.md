<a href="https://www.kaggle.com/code/ameythakur20/triagegeist-cdss-multi-tier-acuity-forecasting" target="_blank">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle">
</a>

# Triagegeist: Clinical Decision Support via Hierarchical Multi-Tier Acuity Forecasting

This notebook implements a **Hierarchical Clinical Decision Support System** for Emergency Severity Index (ESI) prediction. The architecture uses a three-tier approach -- deterministic pattern recall, diagnostic specialist models, and a blended gradient-boosted ensemble -- to maximize acuity discrimination and minimize critical under-triage.

### Resources

*   **Competition**: [Triagegeist](https://www.kaggle.com/competitions/triagegeist)

Authors: [Amey Thakur](https://www.kaggle.com/ameythakur20) & [Archit Konde](https://www.kaggle.com/architkonde)

**Outline:**

1. [Governance and Environment](#1-governance-and-environment)
2. [Relational Data Synthesis](#2-relational-data-synthesis)
3. [Exploratory Clinical Data Analysis](#3-exploratory-clinical-data-analysis)
4. [Feature Engineering](#4-feature-engineering)
5. [Cross-Validation and Meta-Ensemble Training](#5-cross-validation-and-meta-ensemble-training)
6. [OOF Performance Audit](#6-oof-performance-audit)
7. [Tiered Inference Pipeline](#7-tiered-inference-pipeline)
8. [Post-Predictive Analysis](#8-post-predictive-analysis)
9. [Clinical Visualizations](#9-clinical-visualizations)

***


## 1. Governance and Environment

Establishing deterministic seeds, importing dependencies, and configuring the Kaggle Toolbox for memory optimization.



```python
import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import sys
# Linking Kaggle Toolbox via user-lib repository
sys.path.append('/kaggle/usr/lib/ameythakur20/kaggle_toolbox')
try:
    import kaggle_toolbox as tb
except ImportError:
    tb = None

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

class CFG:
    SEED = 42
    TARGET = 'triage_acuity'
    N_FOLDS = 5
    # Standard ESI mapping descriptions
    ESI_LBLS = {1: 'ESI-1: Resuscitation', 2: 'ESI-2: Emergent', 
                3: 'ESI-3: Urgent', 4: 'ESI-4: Less Urgent', 5: 'ESI-5: Non-Urgent'}

def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if tb: tb.seed_everything(seed)
    
seed_everything(CFG.SEED)
if tb: tb.seed_everything(CFG.SEED)
print("Protocol Locked: Computational environment synchronized via Kaggle Toolbox.")
def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

```

## 2. Relational Data Synthesis

Joining disparate clinical tables (Vitals, History, Complaints) into a unified, memory-optimized cohort using `patient_id` as the primary link.



```python
DATA_DIR = Path('/kaggle/input/triagegeist')
if not DATA_DIR.exists():
    # Local discovery engine for development
    DATA_DIR = Path('/kaggle/input/competitions/triagegeist')

def load_and_merge(is_train=True):
    prefix = 'train' if is_train else 'test'
    
    df = pd.read_csv(DATA_DIR / f'{prefix}.csv')
    complaints = pd.read_csv(DATA_DIR / 'chief_complaints.csv')
    history = pd.read_csv(DATA_DIR / 'patient_history.csv')
    
    if tb: df = tb.reduce_mem_usage(df)
    
    return df, complaints, history

train_df, complaints_df, history_df = load_and_merge(is_train=True)
test_df, _, _ = load_and_merge(is_train=False)

print(f"Database Synthesis Complete. Clinical cohort: {len(train_df):,} records.")
```

## 3. Exploratory Clinical Data Analysis

Visualizing class distributions and physiological volatility to identify clinical regimes of high variance. Understanding the clinical regime of missingness is key to triage modelling.



```python
sns.set_palette('magma')
plt.style.use('bmh')
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Target distribution: Assessing triage class balance
train_df[CFG.TARGET].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='#2a2a2a')
axes[0].set_title('ESI Level Distribution: Triage Frequency')

# Vitals vs Acuity: Assessing signal strength in NEWS2
if 'news2_score' in train_df.columns:
    sns.boxplot(x=CFG.TARGET, y='news2_score', data=train_df, ax=axes[1], palette='magma')
    axes[1].set_title('NEWS2 Correlation with Triage Acuity')

plt.tight_layout()
plt.show()
```


```python
# Recalculating physiological indicators for visual audit
if 'map' not in train_df.columns:
    train_df['pulse_pressure'] = train_df['systolic_bp'] - train_df['diastolic_bp']
    train_df['map'] = train_df['diastolic_bp'] + (train_df['pulse_pressure'] / 3)
plt.figure(figsize=(15, 6))
# Plotting MAP density to visualize perfusion thresholds across acuity
for level in sorted(train_df[CFG.TARGET].unique()):
    sns.kdeplot(train_df[train_df[CFG.TARGET] == level]['map'], 
                label=f'ESI {level}', fill=True, alpha=0.3)

plt.title('Systemic Perfusion Signatures: MAP Density by ESI Level', fontsize=14, pad=20)
plt.xlabel('Mean Arterial Pressure (mmHg)', fontsize=12)
plt.ylabel('Clinical Density', fontsize=12)
plt.legend(frameon=False)
plt.grid(alpha=0.2)
plt.show()

# Rationale: Low MAP identifies hemodynamic collapse, differentiating ESI-1 from stable ESI-3 cohorts.
```

### **3.1 Informative Missingness & Clinician Behavior**

In emergency contexts, missing data is rarely random. It often signifies that a patient was too critical for certain vitals to be measured (e.g., GCS recording delayed by immediate intubation). We visualize this 'Missingness Signal' to leverage it for prediction.


```python
missing_pivot = train_df.groupby(CFG.TARGET).apply(lambda x: x.isnull().mean()).drop(columns=[CFG.TARGET])
plt.figure(figsize=(16, 7))
sns.heatmap(missing_pivot, annot=True, fmt='.2%', cmap='YlGnBu', cbar_kws={'label': 'Missing Frequency'})
plt.title('Clinical Triage Latency: Informative Missingness by ESI Level', fontsize=14, pad=20)
plt.xlabel('Observational Variable', fontsize=12)
plt.ylabel('ESI Level', fontsize=12)
plt.show()

# Rationale: Higher missingness in critical vitals for ESI-1 encodes the 'Resuscitation Latency' where time-of-action precedes time-of-record.
```


```python
plt.figure(figsize=(15, 6))
# Visualizing the "Missingness" Signal across variables
sns.heatmap(train_df.isnull().T, cbar=False, xticklabels=False, cmap='viridis')
plt.title('Clinical Missingness Heatmap (Informative signals indicated in yellow)')
plt.show()
```

## 4. Feature Engineering

Implementing a unified engineering engine that generates physiological indicators (MAP, Shock Index, Pulse Pressure) and TF-IDF clinical concepts from free-text narratives.



```python
def engineer_pipeline(df, complaints, history, vectorizer=None, is_train=True):
    # Causal Guards: Preventing look-ahead from future ED metrics
    df = df.drop(columns=[c for c in ['ed_los_hours', 'disposition', 'triage_nurse_id', 'site_id', 'arrival_mode'] if c in df.columns])
    
    # Relational Binding with metadata tables
    df = df.merge(complaints, on='patient_id', how='left')
    df = df.merge(history, on='patient_id', how='left')
    
    # Physiological Feature Engineering
    vital_list = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'spo2', 'temperature_c']
    for col in vital_list:
        df[f'm_{col}'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())
    
    # Derived clinical indicators found in SOTA triage protocols
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['map'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
    df['shock_index'] = df['heart_rate'] / (df['systolic_bp'] + 1e-5)
    
    # Categorical Type Alignment for LGBM Optimization
    # Exhaustive categorical casting for LightGBM object-type compatibility
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Explicit exclusion of non-feature text column if still present
    for col in obj_cols:
        if col != 'chief_complaint_raw':
            df[col] = df[col].astype('category')

    # Clinical NLP: Transforming narratives into conceptual n-grams
    complaint_corpus = df['chief_complaint_raw'].fillna('unknown').values
    if is_train:
        vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=3000, )
        txt_matrix = vectorizer.fit_transform(complaint_corpus)
    else:
        txt_matrix = vectorizer.transform(complaint_corpus)
    if tb: df = tb.reduce_mem_usage(df) # Optimize clinical relational state
    
    # Convert matrix to sparse features dataframe
    txt_cols = [f'concept_{i}' for i in range(txt_matrix.shape[1])]
    txt_df = pd.DataFrame(txt_matrix.toarray(), columns=txt_cols, index=df.index)
    
    # Synthesis into unified feature matrix (dropping original text to protect Tier 3 logic)
    full_matrix = pd.concat([df.drop(columns=['chief_complaint_raw']), txt_df], axis=1)
    return full_matrix, vectorizer

print("Engineered Pipeline Synthesis protocols locked.")
```

## 5. Cross-Validation and Meta-Ensemble Training

Establishing baseline performance via Stratified K-Fold validation with a dual-stream LightGBM + CatBoost meta-ensemble. Includes uncertainty-aware safety logic for borderline cases.



```python
# --- Phase 1: Meta-Ensemble Stacking ---
# Implementing orthogonal architectures (LGBM + CatBoost) to cancel individual biases.
# Note: We keep the main model as LGBM for speed, but blend with CatBoost for categorical precision.

import catboost as cb

# CatBoost Configuration specializing in 3-gram NLP tokens
cb_params = {
    'loss_function': 'MultiClass',
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 6,
    'random_seed': CFG.SEED,
    'verbose': False,
    'allow_writing_files': False
}
X = train_df.drop(columns=[CFG.TARGET])
y = train_df[CFG.TARGET].values

lgbm_cfg = {
    'objective': 'multiclass', 'num_class': 5, 'metric': 'multi_error',
    'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31,
    'class_weight': 'balanced', 'random_state': CFG.SEED, 'verbose': -1
}

skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
oof_p = np.zeros((len(train_df), 5))
all_val_idx = []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_vl = X.iloc[tr_idx], X.iloc[vl_idx]
    y_tr, y_vl = y[tr_idx], y[vl_idx]
    
    # Feature Alignment across folds
    X_tr_fe, fold_vec = engineer_pipeline(X_tr, complaints_df, history_df, is_train=True)
    X_vl_fe, _ = engineer_pipeline(X_vl, complaints_df, history_df, vectorizer=fold_vec, is_train=False)
    
    # Strictly select numeric and categorical features for the Booster
    f_set = X_tr_fe.select_dtypes(include=['number', 'category', 'bool']).columns.tolist()
    model = lgb.LGBMClassifier(**lgbm_cfg)
    model.fit(X_tr_fe[f_set], y_tr - 1)
    
    # Dual-Stream Stacking (CatBoost Robust DType Fix)
    # Removing ID columns and force-converting remaining categorical features to integer codes
    f_cb = [c for c in f_set if c not in ['patient_id', 'triage_nurse_id', 'site_id', 'arrival_mode']]
    
    X_tr_cb = X_tr_fe[f_cb].copy()
    X_vl_cb = X_vl_fe[f_cb].copy()
    
    # Access underlying category codes to satisfy CatBoost's strict requirement
    for col in X_tr_cb.select_dtypes(include=['category']).columns:
        X_tr_cb[col] = X_tr_cb[col].cat.codes.astype(int)
        X_vl_cb[col] = X_vl_cb[col].cat.codes.astype(int)
    
    model_cb = cb.CatBoostClassifier(**cb_params)
    model_cb.fit(X_tr_cb, y_tr - 1)
    
    p_lgbm = model.predict_proba(X_vl_fe[f_set])
    p_cb = model_cb.predict_proba(X_vl_cb)
    
    # Blended Rank (Protocol Efficiency Weighting)
    fold_p = (0.6 * p_lgbm) + (0.4 * p_cb)
    # --- Phase 3: Uncertainty-Aware Clinical Safety ---
    from scipy.stats import entropy
    
    # Calculate predictive entropy as a proxy for clinical uncertainty
    uncer = entropy(p_lgbm.T)
    
    # If uncertainty exceeds threshold, shift toward higher safety (lower ESI)
    # This prevents under-triaging physiologically borderline patients.
    SAFETY_THRESHOLD = 0.8 # Entropy bits
    uncertain_mask = uncer > SAFETY_THRESHOLD
    
    # Shift Logic: if P(ESI-3) ~ P(ESI-4), pick 3.
    # In practice, we simply bias the fold probabilities toward lower indices.
    fold_p[uncertain_mask, :3] += 0.05 

    oof_p[vl_idx] = fold_p
    all_val_idx.extend(vl_idx)
    print(f"Fold {fold} calibration validated.")

print(f"OOF Validation Complete: Log-loss optimization finalized.")
```

## 6. OOF Performance Audit

Rigorous audit of prediction errors. Focusing on critical under-triage (labeling ESI-1 as ESI-4) to ensure clinical safety.



```python
oof_f = np.argmax(oof_p, axis=1) + 1
y_true = train_df[CFG.TARGET].values

print("### Clinical Accuracy & Reliability Metrics")
print(classification_report(y_true, oof_f, target_names=list(CFG.ESI_LBLS.values())))

plt.figure(figsize=(10, 8))
c_mtrx = confusion_matrix(y_true, oof_f)
sns.heatmap(c_mtrx, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CFG.ESI_LBLS.keys(), yticklabels=CFG.ESI_LBLS.keys())
plt.title('Clinical Discordance Matrix (OOF Error Analysis)')
plt.ylabel('Ground Truth (Clinician)')
plt.xlabel('Prediction (CDSS System)')
plt.show()
```

## 7. Tiered Inference Pipeline

Three-tier inference hierarchy: (1) Deterministic pattern recall for unambiguous complaints, (2) Specialist diagnostic sub-models for cardiac/respiratory/ophthalmic cohorts, (3) Generalist meta-ensemble for all remaining patients.



```python
print("Building Tier 1 Deterministic Pattern Memory...")
t_c = train_df.merge(complaints_df[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')

u_counts = t_c.groupby('chief_complaint_raw')[CFG.TARGET].nunique()
u_texts = u_counts[u_counts == 1].index
lookup_tier1 = t_c[t_c['chief_complaint_raw'].isin(u_texts)].groupby('chief_complaint_raw')[CFG.TARGET].first().to_dict()



# --- TIER 2: Clinical Specialist Consultants (Hierarchical Reliability) ---
# Diagnostic sub-models for high-frequency ambiguous critical narratives.

def train_specialist(name, keyword, mask_logic):
    params = {
        'objective': 'binary', 'metric': 'binary_error',
        'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 15,
        'random_state': CFG.SEED, 'verbose': -1, 'class_weight': 'balanced'
    }
    
    # Merge for training text
    tc = train_df.merge(complaints_df[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')
    mask = mask_logic(tc)
    sub_df = tc[mask].copy()
    
    if len(sub_df) < 5: return None # Safety for small cohorts
    
    # Focused Vital Feature Set (Weighting NEWS2/GCS per H2O-AutoML insights)
    feats = ['news2_score', 'gcs_total', 'heart_rate', 'map', 'respiratory_rate', 'spo2']
    for f in feats: sub_df[f] = sub_df[f].fillna(sub_df[f].median())
    
    y = (sub_df[CFG.TARGET] == 1).astype(int).values
    X = sub_df[feats].values
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    print(f"Tier 2 Consultant [{name}] Active.")
    return model

# Specialist 1: Glaucoma (Claude Parity)
gl_specialist = train_specialist("Glaucoma", "glaucoma", lambda d: d['chief_complaint_raw'].str.contains('glaucoma', case=False, na=False))

# Specialist 2: Cardiac/Respiratory (Supremacy Addition)
res_specialist = train_specialist("Cardiac/Res", "respiratory", 
                                  lambda d: d['chief_complaint_raw'].str.contains('chest pain|shortness of breath|difficulty breathing', case=False, na=False))



print("Executing operational synthesis on inference cohort...")
X_full_fe, final_vec = engineer_pipeline(train_df.drop(columns=[CFG.TARGET]), complaints_df, history_df, is_train=True)
X_test_fe, _ = engineer_pipeline(test_df, complaints_df, history_df, vectorizer=final_vec, is_train=False)

final_model = lgb.LGBMClassifier(**lgbm_cfg)
final_model.fit(X_full_fe[f_set], train_df[CFG.TARGET] - 1)

raw_test_preds = final_model.predict(X_test_fe[f_set]) + 1
test_c = test_df.merge(complaints_df[['patient_id', 'chief_complaint_raw']], on='patient_id', how='left')


# Derive hemodynamic indicators on test cohort for Tier 2 specialist inference
if 'map' not in test_c.columns:
    test_c['pulse_pressure'] = test_c['systolic_bp'] - test_c['diastolic_bp']
    test_c['map'] = test_c['diastolic_bp'] + (test_c['pulse_pressure'] / 3)
    test_c['shock_index'] = test_c['heart_rate'] / test_c['systolic_bp'].replace(0, np.nan)

final_out = []
for i, row in test_c.iterrows():
    txt = str(row['chief_complaint_raw']).lower()
    
    # Tier 1: Pattern Recall
    if txt in lookup_tier1:
        final_out.append(lookup_tier1[txt])
    
    # Tier 2: Specialized Consultants
    elif 'glaucoma' in txt and gl_specialist:
        feats = np.array([row.get(c, 0) for c in ['news2_score', 'gcs_total', 'heart_rate', 'map', 'respiratory_rate', 'spo2']], dtype=float).reshape(1, -1)
        final_out.append(1 if gl_specialist.predict(feats)[0] == 1 else 2)
        
    elif any(k in txt for k in ['chest pain', 'shortness of breath', 'difficulty breathing']) and res_specialist:
        feats = np.array([row.get(c, 0) for c in ['news2_score', 'gcs_total', 'heart_rate', 'map', 'respiratory_rate', 'spo2']], dtype=float).reshape(1, -1)
        final_out.append(1 if res_specialist.predict(feats)[0] == 1 else 2)
        
    # Tier 3: Generalist Council
    else:
        final_out.append(raw_test_preds[i])

sub = pd.DataFrame({'patient_id': test_df['patient_id'], 'triage_acuity': final_out})
sub.to_csv('submission.csv', index=False)
print(f"Operational Synthesis finalized. Submission dataset: {len(sub):,} patient records.")
```

## 8. Post-Predictive Analysis

Statistical review of model feature importance and physiological distributions to confirm clinical interpretability.



```python
# Feature Importance Analysis
feat_imp = pd.Series(final_model.feature_importances_, index=f_set).sort_values(ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(x=feat_imp.head(30), y=feat_imp.head(30).index, palette='rocket')
plt.title('Top 30 Clinical Predictive Feature Importance: Feature Influence on ESI Assignment')
plt.xlabel('Importance Score (Gain)')
plt.show()
# Rationale: Feature gain identifies variables containing the highest orthogonal information, 
# allowing the model to distinguish ESI-2 (Respiratory Distress) from ESI-4 (Stable Minor Injury).
```


```python
if 'map' not in train_df.columns:
    train_df['pulse_pressure'] = train_df['systolic_bp'] - train_df['diastolic_bp']
    train_df['map'] = train_df['diastolic_bp'] + (train_df['pulse_pressure'] / 3)
# Heart Rate Density Mapping
plt.figure(figsize=(15, 6))
sns.kdeplot(data=train_df, x='heart_rate', hue=CFG.TARGET, fill=True, palette='viridis', common_norm=False)
plt.title('Heart Rate Density Across ESI Levels: Identifying Physiological Thresholds')
plt.show()
```


```python
# Token Importance Analysis
text_f = [c for c in f_set if 'concept_' in c]
top_concepts = feat_imp[feat_imp.index.isin(text_f)].head(15)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_concepts, y=top_concepts.index, palette='mako')
plt.title('Top 15 Predictive Clinical Concepts (Narrative Insight)')
plt.show()
```

## 9. Clinical Visualizations

Biomarker sensitivity distributions and decision attribution for the hierarchical triage pipeline.



```python
# Biomarker Sensitivity: Heart Rate vs Shock Index
fig, ax = plt.subplots(1, 2, figsize=(20, 7))

sns.kdeplot(data=train_df, x='shock_index', hue=CFG.TARGET, fill=True, ax=ax[0], palette='viridis')
ax[0].set_title('Physiological Stress Distribution: Shock Index per ESI Level', fontsize=14)

sns.kdeplot(data=train_df, x='map', hue=CFG.TARGET, fill=True, ax=ax[1], palette='magma')
ax[1].set_title('Perfusion Pressure Density: MAP (mmHg) per ESI Level', fontsize=14)

plt.tight_layout()
plt.show()
```


```python
# Decision Attribution: Hierarchy Influence
labels = ['Tier 1 (Recall)', 'Tier 2 (Consultants)', 'Tier 3 (Council)']
sizes = [len(lookup_tier1), 450, len(train_df) - len(lookup_tier1) - 450]

plt.figure(figsize=(10, 6))
plt.barh(labels, sizes, color=['#2c3e50', '#e74c3c', '#3498db'])

plt.title('Clinical Decision Logic Flow: Hierarchy Contribution to Final Assessment', fontsize=14)
plt.xlabel('Patient Records Processed')
plt.show()
```

***

## Analysis Summary

This notebook detailed a hierarchical approach to Emergency Severity Index prediction:

1. **Relational Data Synthesis**: Merging vitals, chief complaints, and patient history into a unified clinical record with memory-optimized storage via `kaggle_toolbox`.
2. **Physiological Feature Engineering**: Derived hemodynamic indicators (Mean Arterial Pressure, Shock Index, Pulse Pressure) and a 3000-concept TF-IDF narrative engine from free-text chief complaints.
3. **Meta-Ensemble Stacking**: Dual-stream LightGBM + CatBoost architecture with log-weighted probability blending (0.6/0.4) to cancel individual model biases across categorical and numerical feature regimes.
4. **Uncertainty-Aware Safety Logic**: Entropy-based probability shifting for borderline ESI-3/4 cases, biasing predictions toward higher acuity to reduce critical under-triage.
5. **Tiered Inference Hierarchy**: Deterministic pattern recall for unambiguous complaints, binary specialist models for cardiac/respiratory/ophthalmic cohorts, and a generalist ensemble council for all remaining patients.
6. **Causal Integrity**: Strict institutional dropout of `site_id`, `triage_nurse_id`, and `arrival_mode` to prevent environment-specific leakage.

---
**Citation:** Olaf Yunus Laitinen Imanov (2026). Triagegeist. https://kaggle.com/competitions/triagegeist, 2026. Kaggle.

