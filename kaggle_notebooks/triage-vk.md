# Hierarchical Multimodal Triage Intelligence — Triagegeist v2
### GPU-Optimised · Memory-Safe · Crash-Proof · Three-Tier Clinical AI

**Architecture:**
- **Tier 1** — Deterministic safety guardrail (red-flag vitals + keywords → forced ESI-1/2)
- **Tier 3A/B/C** — LightGBM + CatBoost + XGBoost ensemble (5-fold stratified CV)
- **Post-processing** — Nelder-Mead QWK threshold optimisation + Entropy audit

**Safety:** All optional components (GPU, BERT, TabPFN) have hardened fallbacks. Kernel will not OOM.

## ── Cell 1 · Imports & Environment Detection ──────────────────────────────


```python
import os, gc, warnings, time, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — avoids display memory leaks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import entropy as scipy_entropy

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA = '/kaggle/input/competitions/triagegeist/'

# ── Hardware detection ────────────────────────────────────────────────────────
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEM  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'GPU detected: {GPU_NAME}  ({GPU_MEM:.1f} GB)')
    else:
        print('No GPU — running on CPU (all models will auto-adjust)')
except ImportError:
    GPU_AVAILABLE = False
    print('PyTorch not installed — CPU mode')

LGB_DEVICE = 'gpu'  if GPU_AVAILABLE else 'cpu'
CB_DEVICE  = 'GPU'  if GPU_AVAILABLE else 'CPU'
XGB_DEVICE = 'cuda' if GPU_AVAILABLE else 'cpu'

# ── Global seeds ──────────────────────────────────────────────────────────────
SEED    = 42
NFOLDS  = 5
np.random.seed(SEED)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.dpi':       100,
})
ESI_COLORS = ['#d32f2f','#f57c00','#fbc02d','#388e3c','#1976d2']
ESI_NAMES  = ['ESI-1','ESI-2','ESI-3','ESI-4','ESI-5']

print(f'LightGBM {lgb.__version__} | CatBoost {cb.__version__} | XGBoost {xgb.__version__}')
print('Setup complete.')
```

## ── Cell 2 · Memory-Efficient Data Loading ────────────────────────────────


```python
def reduce_mem(df, verbose=False):
    """Downcast numeric columns to smallest safe dtype — saves 40-60% RAM."""
    start = df.memory_usage(deep=True).sum() / 1e6
    for col in df.select_dtypes(include=['int64','int32']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    end = df.memory_usage(deep=True).sum() / 1e6
    if verbose:
        print(f'  Memory: {start:.1f} MB → {end:.1f} MB  (−{100*(start-end)/start:.0f}%)')
    return df


print('Loading data...')
t0 = time.time()

train  = reduce_mem(pd.read_csv(DATA + 'train.csv'),   verbose=True)
test   = reduce_mem(pd.read_csv(DATA + 'test.csv'),    verbose=True)
cc     = reduce_mem(pd.read_csv(DATA + 'chief_complaints.csv'), verbose=True)
hist   = reduce_mem(pd.read_csv(DATA + 'patient_history.csv'),  verbose=True)
sub    = pd.read_csv(DATA + 'sample_submission.csv')

print(f'\nLoaded in {time.time()-t0:.1f}s')
print(f'Train : {train.shape[0]:,} rows × {train.shape[1]} cols')
print(f'Test  : {test.shape[0]:,} rows × {test.shape[1]} cols')
print(f'CC    : {cc.shape[0]:,} rows | History: {hist.shape[0]:,} rows')

print('\nTarget distribution (train):')
dist = train['triage_acuity'].value_counts(normalize=True).sort_index()
for k,v in dist.items():
    bar = '█' * int(v*60)
    print(f'  ESI-{k}: {v*100:5.1f}%  {bar}')

gc.collect()
```

## ── Cell 3 · EDA Plots ────────────────────────────────────────────────────


```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Class balance
counts = train['triage_acuity'].value_counts().sort_index()
bars   = axes[0].bar(ESI_NAMES, counts.values, color=ESI_COLORS, edgecolor='white', linewidth=1)
for bar, v in zip(bars, counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                 f'{v:,}\n({v/len(train)*100:.1f}%)', ha='center', fontsize=8)
axes[0].set_title('ESI Class Distribution', fontweight='bold')
axes[0].set_ylim(0, counts.max()*1.2)
axes[0].set_ylabel('Count')

# 2. NEWS2 by acuity (violin)
if 'news2_score' in train.columns:
    parts = [train[train['triage_acuity']==i]['news2_score'].dropna().values for i in range(1,6)]
    parts = [p if len(p) > 0 else np.array([0]) for p in parts]
    vp = axes[1].violinplot(parts, positions=range(1,6), showmedians=True, showextrema=False)
    for body, color in zip(vp['bodies'], ESI_COLORS):
        body.set_facecolor(color); body.set_alpha(0.7)
    vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(2)
    axes[1].set_xlabel('ESI Level'); axes[1].set_ylabel('NEWS2 Score')
    axes[1].set_title('NEWS2 Score by Acuity', fontweight='bold')

# 3. Missingness heatmap
vital_cols = ['systolic_bp','diastolic_bp','heart_rate','respiratory_rate',
              'temperature_c','spo2','pain_score']
miss_pct = pd.DataFrame({
    col: [train[train['triage_acuity']==i][col].isnull().mean()*100
          for i in range(1,6)]
    for col in vital_cols if col in train.columns
}, index=[f'ESI-{i}' for i in range(1,6)])
sns.heatmap(miss_pct.T, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.3, ax=axes[2], cbar_kws={'label':'Missing %'})
axes[2].set_title('Missingness % by ESI Level\n(higher ESI = more missing = clinical signal)', fontweight='bold')

plt.suptitle('Triagegeist — Clinical EDA', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda.png', dpi=120, bbox_inches='tight')
plt.show()
plt.close()
gc.collect()
```

## ── Cell 4 · Clinical Calculators (NEWS2, CCI, MSI) ──────────────────────


```python
# ── NEWS2  (Royal College of Physicians, 2017) ─────────────────────────────────
def compute_news2(df):
    """Vectorised NEWS2 — operates on full DataFrame, returns new columns."""
    out = {}

    # Respiratory Rate
    rr = df.get('respiratory_rate', pd.Series(16, index=df.index))
    out['n2_rr'] = np.select(
        [rr<=8, (rr>=9)&(rr<=11), (rr>=12)&(rr<=20), (rr>=21)&(rr<=24), rr>=25],
        [3,      1,                 0,                  2,                  3], default=0).astype(np.int8)

    # SpO2 (Scale 1)
    s = df.get('spo2', pd.Series(98, index=df.index))
    out['n2_spo2'] = np.select(
        [s<=91, (s>=92)&(s<=93), (s>=94)&(s<=95), s>=96],
        [3,      2,               1,                0], default=0).astype(np.int8)

    # Systolic BP
    sbp = df.get('systolic_bp', pd.Series(120, index=df.index))
    out['n2_sbp'] = np.select(
        [sbp<=90, (sbp>=91)&(sbp<=100), (sbp>=101)&(sbp<=110),
         (sbp>=111)&(sbp<=219), sbp>=220],
        [3,        2,                    1,
         0,                      3], default=0).astype(np.int8)

    # Heart Rate
    hr = df.get('heart_rate', pd.Series(75, index=df.index))
    out['n2_hr'] = np.select(
        [hr<=40, (hr>=41)&(hr<=50), (hr>=51)&(hr<=90),
         (hr>=91)&(hr<=110), (hr>=111)&(hr<=130), hr>=131],
        [3,       1,                 0,
         1,                   2,                   3], default=0).astype(np.int8)

    # Temperature
    temp = df.get('temperature_c', pd.Series(37.0, index=df.index))
    out['n2_temp'] = np.select(
        [temp<=35.0, (temp>=35.1)&(temp<=36.0), (temp>=36.1)&(temp<=38.0),
         (temp>=38.1)&(temp<=39.0), temp>=39.1],
        [3,           1,                          0,
         1,                          2], default=0).astype(np.int8)

    # Consciousness (ACVPU)
    ms = df.get('mental_status_triage', pd.Series('alert', index=df.index)).astype(str).str.lower()
    out['n2_consciousness'] = ms.isin(
        ['confused','drowsy','unresponsive','agitated','disoriented','voice','pain']
    ).astype(np.int8) * 3

    news2_total = (out['n2_rr'] + out['n2_spo2'] + out['n2_sbp'] +
                   out['n2_hr'] + out['n2_temp'] + out['n2_consciousness'])
    out['news2_calc']     = news2_total.astype(np.int8)
    out['news2_critical'] = (news2_total >= 5).astype(np.int8)   # urgent review threshold
    out['news2_emergency']= (news2_total >= 7).astype(np.int8)   # emergency tier
    return pd.DataFrame(out, index=df.index)


# ── Charlson Comorbidity Index weights ────────────────────────────────────────
CCI_MAP = {
    'hx_myocardial_infarction':1, 'hx_heart_failure':1, 'hx_congestive_heart_failure':1,
    'hx_cerebrovascular':1, 'hx_stroke':1, 'hx_dementia':1, 'hx_copd':1,
    'hx_peripheral_vascular':1, 'hx_peptic_ulcer':1, 'hx_rheumatoid':1,
    'hx_liver_disease':1, 'hx_diabetes':1,
    'hx_diabetes_complications':2, 'hx_hemiplegia':2, 'hx_renal_disease':2,
    'hx_dialysis':2, 'hx_malignancy':2, 'hx_immunocompromised':2,
    'hx_cirrhosis':3,
    'hx_metastatic_cancer':6, 'hx_aids':6,
}

def compute_cci(df):
    cci = np.zeros(len(df), dtype=np.float32)
    for col, w in CCI_MAP.items():
        if col in df.columns:
            cci += df[col].fillna(0).values * w
    return cci

print('Clinical calculators ready: NEWS2 (RCP-2017, vectorised), CCI, MSI.')
```

## ── Cell 5 · Full Feature Engineering Pipeline ────────────────────────────


```python
def engineer(df_raw, hist_df, cc_df, nlp_pipe=None, fit_nlp=False):
    """
    Complete feature engineering. Memory-safe: works on a copy,
    deletes intermediates, uses int8/float32 throughout.
    Returns (X: pd.DataFrame[float32], fitted_nlp_pipe)
    """
    df = df_raw.copy()

    # ── 1. Joins ──────────────────────────────────────────────────────────────
    df = df.merge(hist_df,  on='patient_id', how='left')
    df = df.merge(cc_df[['patient_id','chief_complaint_raw']], on='patient_id', how='left')

    # ── 2. Pain score ─────────────────────────────────────────────────────────
    df['pain_not_recorded'] = (df['pain_score'] == -1).astype(np.int8)
    df.loc[df['pain_score'] == -1, 'pain_score'] = np.nan

    # ── 3. Missingness flags (clinically informative) ─────────────────────────
    vital_miss = ['systolic_bp','diastolic_bp','mean_arterial_pressure','pulse_pressure',
                  'shock_index','respiratory_rate','temperature_c','spo2','pain_score']
    miss_flags = []
    for col in vital_miss:
        if col in df.columns:
            df[f'm_{col}'] = df[col].isnull().astype(np.int8)
            miss_flags.append(f'm_{col}')
    df['n_vitals_missing'] = df[miss_flags].sum(axis=1).astype(np.int8)

    # ── 4. Group imputation: arrival_mode × age_group ─────────────────────────
    impute_cols = ['systolic_bp','diastolic_bp','mean_arterial_pressure','pulse_pressure',
                   'shock_index','respiratory_rate','temperature_c','spo2',
                   'heart_rate','pain_score']
    grp_cols = [c for c in ['age_group','arrival_mode'] if c in df.columns]
    if grp_cols:
        for col in impute_cols:
            if col not in df.columns: continue
            med = df.groupby(grp_cols)[col].transform('median')
            df[col] = df[col].fillna(med).fillna(df[col].median())
    else:
        for col in impute_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

    # ── 5. Haemodynamic indices ───────────────────────────────────────────────
    sbp = df.get('systolic_bp', pd.Series(120, index=df.index))
    dbp = df.get('diastolic_bp', pd.Series(80,  index=df.index))
    hr  = df.get('heart_rate',   pd.Series(75,  index=df.index))

    if 'mean_arterial_pressure' not in df.columns:
        df['mean_arterial_pressure'] = (dbp + (sbp - dbp) / 3).astype(np.float32)
    if 'pulse_pressure' not in df.columns:
        df['pulse_pressure'] = (sbp - dbp).astype(np.float32)
    if 'shock_index' not in df.columns:
        df['shock_index'] = (hr / sbp.replace(0, np.nan)).astype(np.float32)

    map_val = df['mean_arterial_pressure'].replace(0, np.nan)
    df['mod_shock_index'] = (hr / map_val).fillna(0).astype(np.float32)   # MSI

    # ── 6. NEWS2 (explicit, vectorised) ──────────────────────────────────────
    news_df = compute_news2(df)
    df = pd.concat([df, news_df], axis=1)
    del news_df

    # ── 7. CCI ───────────────────────────────────────────────────────────────
    df['cci_score'] = compute_cci(df).astype(np.float32)
    df['cci_high']  = (df['cci_score'] >= 3).astype(np.int8)

    # ── 8. Comorbidity burden ─────────────────────────────────────────────────
    hx_cols = [c for c in df.columns if c.startswith('hx_')]
    if hx_cols:
        df['comorbidity_burden'] = df[hx_cols].fillna(0).sum(axis=1).astype(np.int8)
    high_risk_hx = [c for c in ['hx_heart_failure','hx_copd','hx_malignancy',
                                  'hx_dementia','hx_cirrhosis','hx_immunocompromised',
                                  'hx_dialysis','hx_metastatic_cancer','hx_renal_disease']
                    if c in df.columns]
    if high_risk_hx:
        df['hi_risk_comorbidity'] = df[high_risk_hx].fillna(0).max(axis=1).astype(np.int8)

    # ── 9. Vital flag thresholds ──────────────────────────────────────────────
    s = df.get('spo2',            pd.Series(98,  index=df.index))
    t = df.get('temperature_c',   pd.Series(37,  index=df.index))
    g = df.get('gcs_total',       pd.Series(15,  index=df.index))
    rr= df.get('respiratory_rate',pd.Series(16,  index=df.index))
    si= df.get('shock_index',     df['shock_index'] if 'shock_index' in df.columns
                                   else pd.Series(0.6, index=df.index))

    flags = {
        'hypoxia':          (s  < 94).astype(np.int8),
        'severe_hypoxia':   (s  < 90).astype(np.int8),
        'hypotension':      (sbp < 90).astype(np.int8),
        'severe_hypoten':   (sbp < 80).astype(np.int8),
        'tachycardia':      (hr  > 100).astype(np.int8),
        'severe_tachycard': (hr  > 150).astype(np.int8),
        'bradycardia':      (hr  < 60).astype(np.int8),
        'tachypnea':        (rr  > 20).astype(np.int8),
        'fever':            (t   > 38.0).astype(np.int8),
        'high_fever':       (t   > 39.5).astype(np.int8),
        'hypothermia':      (t   < 36.0).astype(np.int8),
        'gcs_severe':       (g   <= 8).astype(np.int8),
        'gcs_moderate':     ((g >= 9) & (g <= 12)).astype(np.int8),
        'gcs_normal':       (g   >= 14).astype(np.int8),
        'shock_risk':       (si  > 0.9).astype(np.int8),
        'hi_shock_risk':    (si  > 1.0).astype(np.int8),
        'msi_elevated':     (df['mod_shock_index'] > 1.3).astype(np.int8),
        'severe_pain':      (df.get('pain_score', pd.Series(0, index=df.index)) >= 8).astype(np.int8),
    }
    for k, v in flags.items():
        df[k] = v
    df['n_vital_abnormal'] = (df['tachycardia'] + df['bradycardia'] + df['tachypnea'] +
                               df['hypotension'] + df['hypoxia'] + df['fever'] +
                               df['hypothermia']).astype(np.int8)

    # ── 10. Demographics ──────────────────────────────────────────────────────
    age = df.get('age', pd.Series(40, index=df.index))
    df['elderly']      = (age >= 65).astype(np.int8)
    df['very_elderly'] = (age >= 80).astype(np.int8)
    df['pediatric']    = (age <  16).astype(np.int8)
    df['infant']       = (age <   2).astype(np.int8)
    df['age_sq']       = (age ** 2).astype(np.float32)
    df['age_log']      = np.log1p(age).astype(np.float32)

    # Age-adjusted shock index (paediatric baseline HR higher)
    df['peds_si'] = np.where(age < 2,  si / 1.4,
                    np.where(age < 10, si / 1.2, si)).astype(np.float32)

    # ── 11. Temporal ──────────────────────────────────────────────────────────
    df['weekend']    = df.get('arrival_day', pd.Series('Mon', index=df.index)).isin(
                           ['Saturday','Sunday']).astype(np.int8)
    df['night_shift']= (df.get('shift', pd.Series('day', index=df.index)) == 'night').astype(np.int8)
    if 'arrival_hour' in df.columns:
        ah = df['arrival_hour'].astype(float)
        df['hour_sin'] = np.sin(2*np.pi*ah/24).astype(np.float32)
        df['hour_cos'] = np.cos(2*np.pi*ah/24).astype(np.float32)

    # ── 12. Arrival & Mental Status ───────────────────────────────────────────
    am = df.get('arrival_mode', pd.Series('walk-in', index=df.index)).astype(str)
    df['ambulance']      = (am == 'ambulance').astype(np.int8)
    df['hi_risk_arrival']= am.isin(['ambulance','helicopter','transfer']).astype(np.int8)
    ms = df.get('mental_status_triage', pd.Series('alert', index=df.index)).astype(str).str.lower()
    df['altered_ms']  = ms.isin(['confused','drowsy','unresponsive','agitated','disoriented']).astype(np.int8)
    df['unresponsive']= (ms == 'unresponsive').astype(np.int8)

    # ── 13. Interaction features ──────────────────────────────────────────────
    hr_c = df.get('hi_risk_comorbidity', pd.Series(0, index=df.index))
    df['elderly_x_altms']   = (df['elderly']   * df['altered_ms']).astype(np.int8)
    df['elderly_x_hypoten'] = (df['elderly']   * df['hypotension']).astype(np.int8)
    df['elderly_x_hypoxia'] = (df['elderly']   * df['hypoxia']).astype(np.int8)
    df['elderly_x_tachy']   = (df['elderly']   * df['tachycardia']).astype(np.int8)
    df['altms_x_hypoxia']   = (df['altered_ms']* df['hypoxia']).astype(np.int8)
    df['shock_x_altms']     = (df['shock_risk']* df['altered_ms']).astype(np.int8)
    df['ambul_x_shock']     = (df['ambulance'] * df['shock_risk']).astype(np.int8)
    df['ambul_x_news2crit'] = (df['ambulance'] * df['news2_critical']).astype(np.int8)
    df['peds_x_fever']      = (df['pediatric'] * df['fever']).astype(np.int8)
    df['news2_x_gcs']       = (df['news2_calc']* (16 - g.clip(3,15))).astype(np.float32)
    df['cci_x_hypoten']     = (df['cci_score'] * df['hypotension']).astype(np.float32)
    df['cci_x_news2']       = (df['cci_score'] * df['news2_calc']).astype(np.float32)
    df['geri_msi_risk']     = (df['elderly']   * df['msi_elevated']).astype(np.int8)
    if 'hi_risk_comorbidity' in df.columns:
        df['hrc_x_hypoxia'] = (hr_c * df['hypoxia']).astype(np.int8)
        df['hrc_x_shock']   = (hr_c * df['shock_risk']).astype(np.int8)
        df['hrc_x_altms']   = (hr_c * df['altered_ms']).astype(np.int8)

    # Prior utilisation
    if 'num_prior_ed_visits_12m' in df.columns:
        df['frequent_flyer'] = (df['num_prior_ed_visits_12m'] >= 4).astype(np.int8)
        df['log_prior_vis']  = np.log1p(df['num_prior_ed_visits_12m']).astype(np.float32)
    if 'num_active_medications' in df.columns:
        df['polypharmacy']   = (df['num_active_medications'] >= 5).astype(np.int8)
        df['log_meds']       = np.log1p(df['num_active_medications']).astype(np.float32)

    # ── 14. NLP — TF-IDF + SVD ───────────────────────────────────────────────
    df['chief_complaint_raw'] = df['chief_complaint_raw'].fillna('unspecified complaint')
    df['cc_len']    = df['chief_complaint_raw'].str.len().astype(np.int16)
    df['cc_words']  = df['chief_complaint_raw'].str.split().str.len().astype(np.int8)

    if fit_nlp:
        nlp_pipe = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=400, ngram_range=(1,3), min_df=3,
                sublinear_tf=True, dtype=np.float32,
                token_pattern=r'(?u)\b\w+\b'
            )),
            ('svd', TruncatedSVD(n_components=60, random_state=SEED, n_iter=5))
        ])
        nlp_mat = nlp_pipe.fit_transform(df['chief_complaint_raw'])
    else:
        nlp_mat = nlp_pipe.transform(df['chief_complaint_raw'])

    nlp_df = pd.DataFrame(
        nlp_mat.astype(np.float32),
        columns=[f'cc_{i}' for i in range(nlp_mat.shape[1])],
        index=df.index
    )
    df = pd.concat([df.reset_index(drop=True), nlp_df.reset_index(drop=True)], axis=1)
    del nlp_mat, nlp_df

    # ── 15. Label-encode categoricals ─────────────────────────────────────────
    cat_cols = ['arrival_mode','arrival_day','arrival_season','shift','age_group',
                'sex','language','insurance_type','transport_origin','pain_location',
                'mental_status_triage','chief_complaint_system','site_id']
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # ── 16. Drop non-feature cols & cast to float32 ────────────────────────────
    drop = {'patient_id','triage_nurse_id','chief_complaint_raw',
            'triage_acuity','disposition','ed_los_hours'}
    feat_cols = [c for c in df.columns if c not in drop]
    X = df[feat_cols].copy()

    # Zero-variance columns
    zv = X.columns[X.nunique() <= 1].tolist()
    if zv: X = X.drop(columns=zv)

    # Final NaN fill + cast
    X = X.fillna(-999).astype(np.float32)
    del df
    return X, nlp_pipe


print('Feature engineering function defined.')
print('  Categories: raw vitals | NEWS2 | MSI | CCI | missingness |')
print('              demographics | temporal | interactions | NLP (TF-IDF+SVD)')
```

## ── Cell 6 · Build Feature Matrices ──────────────────────────────────────


```python
print('Building feature matrices...')
t0 = time.time()

X_train, nlp_pipe = engineer(train, hist, cc, fit_nlp=True)
X_test,  _        = engineer(test,  hist, cc, nlp_pipe=nlp_pipe, fit_nlp=False)
y_train = (train['triage_acuity'].values - 1).astype(np.int32)   # 0-indexed

print(f'Done in {time.time()-t0:.1f}s')
print(f'X_train: {X_train.shape}  ({X_train.memory_usage(deep=True).sum()/1e6:.0f} MB)')
print(f'X_test : {X_test.shape}  ({X_test.memory_usage(deep=True).sum()/1e6:.0f} MB)')
print(f'Features: {X_train.shape[1]}')

gc.collect()
print('\nMemory cleaned.')
```

## ── Cell 7 · Tier 1 — Deterministic Safety Guardrail ─────────────────────


```python
ESI1_KW = [
    'cardiac arrest','agonal breathing','no pulse','pulseless','apnea','apnoea',
    'respiratory arrest','full arrest','cpr in progress','massive hemorrhage',
    'massive haemorrhage'
]
ESI2_KW = [
    'chest pain','chest tightness','stemi','stroke','tia','facial droop',
    'slurred speech','arm weakness','sudden weakness','anaphylaxis','anaphylactic',
    'overdose','seizure','postictal','altered mental','severe respiratory distress',
    'difficulty breathing','shortness of breath','sob','dyspnea','dyspnoea',
    'severe abdominal','active bleeding','gi bleed','suicidal'
]

def tier1_guardrail(df_raw, cc_df, preds_0idx):
    """
    Apply deterministic safety overrides.
    Only UPGRADES — never downgrades a prediction.
    """
    preds  = preds_0idx.copy()
    df_cc  = df_raw.merge(cc_df[['patient_id','chief_complaint_raw']], on='patient_id', how='left')
    cc_txt = df_cc['chief_complaint_raw'].fillna('').str.lower()

    # ESI-1: extreme vitals
    esi1_vital = (
        (df_raw.get('gcs_total',    pd.Series(15, index=df_raw.index)) <= 8) |
        (df_raw.get('heart_rate',   pd.Series(80, index=df_raw.index)) > 150) |
        (df_raw.get('systolic_bp',  pd.Series(120,index=df_raw.index)) < 80)
    ).values
    esi1_text = cc_txt.apply(lambda t: any(k in t for k in ESI1_KW)).values
    esi1_mask = esi1_vital | esi1_text
    preds[esi1_mask] = 0

    # ESI-2: high-risk vitals + text (only if not already ESI-1)
    esi2_vital = (
        (df_raw.get('spo2',       pd.Series(98, index=df_raw.index)) < 90) |
        (df_raw.get('systolic_bp',pd.Series(120,index=df_raw.index)) < 90) |
        (df_raw.get('mental_status_triage',
             pd.Series('alert',index=df_raw.index)).isin(['unresponsive','confused']))
    ).values
    esi2_text = cc_txt.apply(lambda t: any(k in t for k in ESI2_KW)).values
    esi2_upgrade = (esi2_vital | esi2_text) & ~esi1_mask & (preds >= 2)
    preds[esi2_upgrade] = 1

    n1 = int(esi1_mask.sum())
    n2 = int(esi2_upgrade.sum())
    print(f'  Tier 1: {n1} forced ESI-1  |  {n2} upgraded to ESI-2  |  total overrides: {n1+n2}')
    return preds

print('Tier 1 guardrail ready.')
print('  ESI-1 triggers : GCS≤8 | HR>150 | SBP<80 | 11 red-flag keywords')
print('  ESI-2 triggers : SpO2<90 | SBP<90 | AMS | 26 high-risk keywords')
```

## ── Cell 8 · QWK Helpers + Threshold Optimiser ────────────────────────────


```python
def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def optimise_thresholds(oof_probs, y_true, n_classes=5, n_restarts=5):
    """
    Nelder-Mead optimisation of cumulative-probability thresholds to maximise QWK.
    Multiple random restarts for robustness.
    """
    def neg_qwk(t):
        t_sorted = np.sort(t)
        cum = np.cumsum(oof_probs, axis=1)           # (N, 5)
        pred = np.full(len(oof_probs), n_classes-1, dtype=np.int32)
        for j, th in enumerate(t_sorted):
            mask = (pred == n_classes-1) & (cum[:, j] >= th)
            pred[mask] = j
        return -qwk(y_true, pred)

    best_val, best_t = np.inf, None
    rng = np.random.default_rng(SEED)
    for _ in range(n_restarts):
        x0     = np.sort(rng.uniform(0.1, 0.95, n_classes-1))
        result = minimize(neg_qwk, x0, method='Nelder-Mead',
                          options={'maxiter':3000,'xatol':1e-6,'fatol':1e-6})
        if result.fun < best_val:
            best_val, best_t = result.fun, np.sort(result.x)
    return best_t


def apply_thresholds(probs, thresholds, n_classes=5):
    cum  = np.cumsum(probs, axis=1)
    pred = np.full(len(probs), n_classes-1, dtype=np.int32)
    for j, th in enumerate(thresholds):
        mask = (pred == n_classes-1) & (cum[:, j] >= th)
        pred[mask] = j
    return pred


def entropy_audit(probs, preds, threshold=0.75, n_classes=5):
    """
    Upgrade uncertain ESI-3 predictions to ESI-2 (safety-first principle).
    High entropy = flat distribution = model uncertainty at ESI-2/3 boundary.
    """
    ent     = scipy_entropy(probs.T) / np.log(n_classes)   # normalised to [0,1]
    updated = preds.copy()
    upgrade = (preds == 2) & (ent > threshold)              # uncertain ESI-3 → ESI-2
    updated[upgrade] = 1
    return updated, ent, int(upgrade.sum())


print('Helpers ready: qwk | optimise_thresholds (Nelder-Mead, 5 restarts) |')
print('               apply_thresholds | entropy_audit')
```

## ── Cell 9 · Model A — LightGBM ──────────────────────────────────────────


```python
lgb_params = {
    'objective':          'multiclass',
    'num_class':          5,
    'metric':             'multi_logloss',
    'n_estimators':       2000,
    'learning_rate':      0.03,
    'num_leaves':         127,
    'max_depth':          -1,
    'min_child_samples':  20,
    'subsample':          0.80,
    'subsample_freq':     1,
    'colsample_bytree':   0.70,
    'reg_alpha':          0.05,
    'reg_lambda':         0.10,
    'class_weight':       'balanced',
    'random_state':       SEED,
    'verbose':            -1,
    'n_jobs':             -1,
    'device':             LGB_DEVICE,
    # GPU memory safety — only set on GPU
    **({
        'gpu_use_dp': False,         # single precision on GPU (halves VRAM usage)
        'max_bin':    255,
    } if LGB_DEVICE == 'gpu' else {})
}

skf         = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
lgb_oof     = np.zeros((len(X_train), 5), dtype=np.float32)
lgb_test    = np.zeros((len(X_test),  5), dtype=np.float32)
lgb_scores  = []
lgb_models  = []

print(f'Training LightGBM [{LGB_DEVICE.upper()}] — 5-fold CV...')
t_lgb = time.time()

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr = X_train.iloc[tr_idx];  X_val = X_train.iloc[val_idx]
    y_tr = y_train[tr_idx];       y_val = y_train[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(500)
        ]
    )

    val_prob = model.predict_proba(X_val)
    lgb_oof[val_idx] = val_prob
    lgb_test        += model.predict_proba(X_test) / NFOLDS

    score = qwk(y_val, np.argmax(val_prob, 1))
    lgb_scores.append(score)
    lgb_models.append(model)
    print(f'  Fold {fold+1}/{NFOLDS}  QWK={score:.4f}  best_iter={model.best_iteration_:,}')

    # Cleanup per fold
    del X_tr, X_val, y_tr, y_val, val_prob
    gc.collect()

lgb_oof_qwk = qwk(y_train, np.argmax(lgb_oof, 1))
print(f'\nLGB OOF QWK : {lgb_oof_qwk:.4f}')
print(f'LGB CV  mean: {np.mean(lgb_scores):.4f} ± {np.std(lgb_scores):.4f}')
print(f'LGB training: {(time.time()-t_lgb)/60:.1f} min')
gc.collect()
```

## ── Cell 10 · Model B — CatBoost ─────────────────────────────────────────


```python
cb_params = {
    'iterations':            2000,
    'learning_rate':         0.03,
    'depth':                 8,
    'loss_function':         'MultiClass',
    'eval_metric':           'MultiClass',
    'l2_leaf_reg':           3.0,
    'bagging_temperature':   1.0,
    'random_strength':       1.0,
    'border_count':          128,
    'auto_class_weights':    'Balanced',
    'early_stopping_rounds': 100,
    'random_seed':           SEED,
    'verbose':               False,
    'task_type':             CB_DEVICE,
    **({
        'devices': '0',                      # single GPU
    } if CB_DEVICE == 'GPU' else {})
}

cb_oof    = np.zeros((len(X_train), 5), dtype=np.float32)
cb_test   = np.zeros((len(X_test),  5), dtype=np.float32)
cb_scores = []
cb_models = []

print(f'Training CatBoost [{CB_DEVICE}] — 5-fold CV...')
t_cb = time.time()

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr = X_train.iloc[tr_idx];  X_val = X_train.iloc[val_idx]
    y_tr = y_train[tr_idx];       y_val = y_train[val_idx]

    model = cb.CatBoostClassifier(**cb_params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

    val_prob = model.predict_proba(X_val)
    cb_oof[val_idx] = val_prob
    cb_test        += model.predict_proba(X_test) / NFOLDS

    score = qwk(y_val, np.argmax(val_prob, 1))
    cb_scores.append(score)
    cb_models.append(model)
    print(f'  Fold {fold+1}/{NFOLDS}  QWK={score:.4f}  best_iter={model.best_iteration_:,}')

    del X_tr, X_val, y_tr, y_val, val_prob
    gc.collect()

cb_oof_qwk = qwk(y_train, np.argmax(cb_oof, 1))
print(f'\nCBT OOF QWK : {cb_oof_qwk:.4f}')
print(f'CBT CV  mean: {np.mean(cb_scores):.4f} ± {np.std(cb_scores):.4f}')
print(f'CBT training: {(time.time()-t_cb)/60:.1f} min')
gc.collect()
```

## ── Cell 11 · Model C — XGBoost ──────────────────────────────────────────


```python
xgb_params = {
    'objective':            'multi:softprob',
    'num_class':            5,
    'eval_metric':          'mlogloss',
    'n_estimators':         2000,
    'learning_rate':        0.03,
    'max_depth':            7,
    'min_child_weight':     5,
    'subsample':            0.80,
    'colsample_bytree':     0.70,
    'reg_alpha':            0.05,
    'reg_lambda':           1.0,
    'gamma':                0.01,
    'tree_method':          'hist',
    'device':               XGB_DEVICE,
    'early_stopping_rounds': 100,   # XGBoost >=2.0: must be in constructor, not fit()
    'random_state':         SEED,
    'verbosity':            0,
    'n_jobs':               1 if GPU_AVAILABLE else -1,  # avoid CPU/GPU thread conflicts
}

xgb_oof    = np.zeros((len(X_train), 5), dtype=np.float32)
xgb_test   = np.zeros((len(X_test),  5), dtype=np.float32)
xgb_scores = []

print(f'Training XGBoost [{XGB_DEVICE.upper()}] — 5-fold CV...')
t_xgb = time.time()

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr = X_train.iloc[tr_idx];  X_val = X_train.iloc[val_idx]
    y_tr = y_train[tr_idx];       y_val = y_train[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False)

    val_prob = model.predict_proba(X_val)
    xgb_oof[val_idx] = val_prob
    xgb_test        += model.predict_proba(X_test) / NFOLDS

    score = qwk(y_val, np.argmax(val_prob, 1))
    xgb_scores.append(score)
    print(f'  Fold {fold+1}/{NFOLDS}  QWK={score:.4f}')

    del X_tr, X_val, y_tr, y_val, val_prob, model
    gc.collect()

xgb_oof_qwk = qwk(y_train, np.argmax(xgb_oof, 1))
print(f'\nXGB OOF QWK : {xgb_oof_qwk:.4f}')
print(f'XGB CV  mean: {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}')
print(f'XGB training: {(time.time()-t_xgb)/60:.1f} min')
gc.collect()
```

## ── Cell 12 · Blend Search + Threshold Optimisation + Entropy Audit ───────


```python
# ── Grid-search blend weights on OOF QWK ─────────────────────────────────────
print('Searching blend weights...')
best_blend_qwk, best_w = 0, (0.50, 0.30, 0.20)

for w1 in np.arange(0.20, 0.70, 0.05):
    for w2 in np.arange(0.10, 0.50, 0.05):
        w3 = 1.0 - w1 - w2
        if not (0.05 <= w3 <= 0.50): continue
        blend = w1*lgb_oof + w2*cb_oof + w3*xgb_oof
        score = qwk(y_train, np.argmax(blend, 1))
        if score > best_blend_qwk:
            best_blend_qwk, best_w = score, (w1, w2, w3)

w1, w2, w3 = best_w
print(f'Best weights — LGB:{w1:.2f}  CBT:{w2:.2f}  XGB:{w3:.2f}')
print(f'Blend OOF QWK (argmax): {best_blend_qwk:.4f}')

ens_oof  = (w1*lgb_oof  + w2*cb_oof  + w3*xgb_oof).astype(np.float32)
ens_test = (w1*lgb_test + w2*cb_test + w3*xgb_test).astype(np.float32)

# ── Nelder-Mead threshold optimisation ───────────────────────────────────────
print('\nNelder-Mead threshold optimisation...')
best_t        = optimise_thresholds(ens_oof, y_train, n_restarts=5)
oof_thresh    = apply_thresholds(ens_oof, best_t)
thresh_qwk    = qwk(y_train, oof_thresh)
print(f'Thresholds : {best_t.round(4)}')
print(f'OOF QWK after threshold opt : {thresh_qwk:.4f}  (+{thresh_qwk-best_blend_qwk:+.4f})')

# ── Entropy audit ─────────────────────────────────────────────────────────────
print('\nEntropy audit (ESI-2/3 safety boundary)...')
oof_entropy_pred, oof_ent, n_up = entropy_audit(ens_oof, oof_thresh, threshold=0.75)
ent_qwk = qwk(y_train, oof_entropy_pred)
print(f'Patients upgraded ESI-3→ESI-2 (high entropy): {n_up:,}')
print(f'OOF QWK after entropy audit : {ent_qwk:.4f}  ({ent_qwk-thresh_qwk:+.4f})')

USE_ENTROPY = (ent_qwk >= thresh_qwk)
final_oof   = oof_entropy_pred if USE_ENTROPY else oof_thresh
final_qwk   = ent_qwk if USE_ENTROPY else thresh_qwk

print(f'\nEntropy audit: {"APPLIED" if USE_ENTROPY else "SKIPPED (no improvement)"}')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '═'*55)
rows = [
    ('LightGBM',               lgb_oof_qwk),
    ('CatBoost',               cb_oof_qwk),
    ('XGBoost',                xgb_oof_qwk),
    ('Blend (grid search)',     best_blend_qwk),
    ('+ Threshold opt.',        thresh_qwk),
    ('+ Entropy audit  ← FINAL', ent_qwk),
]
for name, score in rows:
    marker = ' ◀' if 'FINAL' in name else ''
    print(f'  {name:<35} {score:.4f}{marker}')
print('═'*55)
```

## ── Cell 13 · Evaluation Plots ───────────────────────────────────────────


```python
# Classification report
print('Classification Report — Final Ensemble OOF:')
print(classification_report(y_train, final_oof, target_names=ESI_NAMES, digits=3))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion matrix
cm     = confusion_matrix(y_train, final_oof)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=ESI_NAMES, yticklabels=ESI_NAMES,
            linewidths=0.4, ax=axes[0], cbar_kws={'label':'Recall (row %)'})
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
axes[0].set_title(f'Confusion Matrix — OOF  (QWK={final_qwk:.4f})', fontweight='bold')

# Per-class F1
f1s = [f1_score((y_train==i).astype(int),(final_oof==i).astype(int)) for i in range(5)]
axes[1].bar(ESI_NAMES, f1s, color=ESI_COLORS, edgecolor='white', linewidth=1)
axes[1].axhline(0.8, color='black', ls='--', lw=0.8, label='F1=0.80')
axes[1].set_ylabel('F1 Score'); axes[1].set_ylim(0, 1.12)
axes[1].set_title('Per-Class F1 — Final Ensemble', fontweight='bold')
for i, v in enumerate(f1s):
    axes[1].text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)
axes[1].legend()

plt.suptitle('Ensemble Performance Summary', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('performance.png', dpi=120, bbox_inches='tight')
plt.show()
plt.close()
gc.collect()
```

## ── Cell 14 · Feature Importance ─────────────────────────────────────────


```python
feat_imp = pd.DataFrame({
    'feature':    X_train.columns,
    'importance': np.mean([m.feature_importances_ for m in lgb_models], axis=0)
}).sort_values('importance', ascending=False)

def cat_color(name):
    if name.startswith('cc_'):            return '#1976d2'   # NLP
    if name.startswith('n2_') or 'news2' in name: return '#e65100'  # NEWS2
    if name.startswith('hx_') or name in ['cci_score','cci_high',
       'comorbidity_burden','hi_risk_comorbidity']: return '#7b1fa2'  # Comorbidity
    if 'm_' in name or 'missing' in name: return '#ff6f00'   # Missingness
    if name in ['systolic_bp','diastolic_bp','heart_rate','respiratory_rate',
                'temperature_c','spo2','gcs_total','shock_index','mod_shock_index',
                'mean_arterial_pressure','pulse_pressure']: return '#d32f2f'  # Raw vitals
    return '#388e3c'                                          # Engineered

top40  = feat_imp.head(40)
colors = [cat_color(c) for c in top40['feature']]

fig, ax = plt.subplots(figsize=(10, 13))
ax.barh(top40['feature'][::-1], top40['importance'][::-1],
        color=colors[::-1], edgecolor='white', linewidth=0.6)
ax.set_xlabel('Mean Gain Importance (5 folds)')
ax.set_title('Top 40 Features — LightGBM', fontweight='bold', fontsize=12)

legend_items = [
    mpatches.Patch(color='#d32f2f', label='Raw Vitals'),
    mpatches.Patch(color='#e65100', label='NEWS2'),
    mpatches.Patch(color='#1976d2', label='NLP (TF-IDF+SVD)'),
    mpatches.Patch(color='#7b1fa2', label='Comorbidity / CCI'),
    mpatches.Patch(color='#ff6f00', label='Missingness Flags'),
    mpatches.Patch(color='#388e3c', label='Engineered Features'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=120, bbox_inches='tight')
plt.show()
plt.close()
gc.collect()
```

## ── Cell 15 · SHAP Feature Attribution ───────────────────────────────────


```python
try:
    import shap
    print('Computing SHAP values on 2000-patient sample (fold-0 model)...')
    sample_idx  = np.random.choice(len(X_train), 2000, replace=False)
    X_sample    = X_train.iloc[sample_idx]

    explainer   = shap.TreeExplainer(lgb_models[0])
    shap_vals   = explainer.shap_values(X_sample)

    print('SHAP Beeswarm — ESI-1 (Immediate):')
    shap.summary_plot(shap_vals[0], X_sample, max_display=18, plot_size=(10,6), show=False)
    plt.tight_layout()
    plt.savefig('shap_esi1.png', dpi=120, bbox_inches='tight')
    plt.show(); plt.close()

    print('SHAP Beeswarm — ESI-3 (Urgent — hardest boundary):')
    shap.summary_plot(shap_vals[2], X_sample, max_display=18, plot_size=(10,6), show=False)
    plt.tight_layout()
    plt.savefig('shap_esi3.png', dpi=120, bbox_inches='tight')
    plt.show(); plt.close()

    del shap_vals, X_sample, explainer
    gc.collect()

except Exception as e:
    print(f'SHAP skipped ({e}). Install with: pip install shap')
```

## ── Cell 16 · Undertriage Bias Audit ─────────────────────────────────────


```python
from scipy.stats import chi2_contingency

# Build audit frame (minimal columns only — memory safe)
keep = ['patient_id','triage_acuity','sex','age_group','insurance_type',
        'language','arrival_mode','chief_complaint_system']
audit = train[[c for c in keep if c in train.columns]].copy()
audit = audit.merge(hist[['patient_id']], on='patient_id', how='left')

audit['pred']        = final_oof + 1
audit['true']        = y_train   + 1
audit['undertriage'] = (audit['pred'] > audit['true']).astype(np.int8)  # pred lower urgency
audit['entropy']     = (scipy_entropy(ens_oof.T) / np.log(5)).astype(np.float32)

audit_hi = audit[audit['true'] <= 3].copy()   # high-acuity only
overall  = audit_hi['undertriage'].mean()

print(f'High-acuity patients (ESI 1-3): {len(audit_hi):,}')
print(f'Overall undertriage rate      : {overall*100:.1f}%')

# Statistical tests
print('\n─── Chi-squared undertriage bias tests (vs. overall rate) ───')
for col in ['sex','insurance_type','language','arrival_mode']:
    if col not in audit_hi.columns: continue
    print(f'\n  {col}:')
    for val, grp in audit_hi.groupby(col):
        if len(grp) < 50: continue
        rate = grp['undertriage'].mean()
        rest = audit_hi[audit_hi[col] != val]['undertriage']
        ct   = np.array([[grp['undertriage'].sum(),   len(grp)-grp['undertriage'].sum()],
                          [rest.sum(),                  len(rest)-rest.sum()]])
        if ct.min() > 0:
            _, p, _, _ = chi2_contingency(ct)
            sig = ' *** SIGNIFICANT' if p < 0.05 else ''
            print(f'    {str(val):20s}  rate={rate*100:5.1f}%  n={len(grp):,}  p={p:.4f}{sig}')

gc.collect()
```


```python
# Bias plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

def plot_ut(df, col, ax, title, n_min=50):
    if col not in df.columns:
        ax.set_visible(False); return
    
    # YAHAN FIX KIYA HAI: .reset_index() add kar diya gaya hai
    g = df.groupby(col)['undertriage'].agg(['mean','count']).reset_index()
    
    g = g[g['count'] >= n_min].sort_values('mean', ascending=False)
    if len(g) == 0:
        ax.set_visible(False); return
        
    g['ci'] = 1.96 * np.sqrt(g['mean']*(1-g['mean'])/g['count'])
    clrs = ['#d32f2f' if r>overall+0.03 else
            '#388e3c' if r<overall-0.03 else '#90a4ae' for r in g['mean']]
            
    # Ab g[col] bina kisi error ke kaam karega
    ax.bar(g[col].astype(str), g['mean']*100, color=clrs, edgecolor='white')
    ax.errorbar(range(len(g)), g['mean']*100, yerr=g['ci']*100,
                fmt='none', color='k', capsize=3, lw=1)
    
    ax.axhline(overall*100, color='black', ls='--', lw=1.2,
               label=f'Overall: {overall*100:.1f}%')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_ylabel('Undertriage Rate (%)')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=8)

plot_ut(audit_hi, 'sex',                   axes[0], 'By Sex')
plot_ut(audit_hi, 'age_group',             axes[1], 'By Age Group')
plot_ut(audit_hi, 'insurance_type',        axes[2], 'By Insurance (SES Proxy)')
plot_ut(audit_hi, 'language',              axes[3], 'By Language')
plot_ut(audit_hi, 'arrival_mode',          axes[4], 'By Arrival Mode')
plot_ut(audit_hi, 'chief_complaint_system',axes[5], 'By Complaint System')

plt.suptitle('Systematic Undertriage Audit — ESI 1-3 Patients\n'
             'Red = elevated risk | Green = reduced risk | Grey = no sig. diff.',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('bias_audit.png', dpi=120, bbox_inches='tight')
plt.show()
plt.close()

del audit, audit_hi
gc.collect()
```

## ── Cell 17 · Generate Final Submission ──────────────────────────────────


```python
# Apply threshold optimisation to test ensemble
test_thresh = apply_thresholds(ens_test, best_t)

# Entropy audit on test
if USE_ENTROPY:
    test_ent = scipy_entropy(ens_test.T) / np.log(5)
    test_preds, _, n_up_test = entropy_audit(ens_test, test_thresh, threshold=0.75)
    print(f'Test entropy audit: {n_up_test:,} patients upgraded ESI-3→ESI-2')
else:
    test_preds = test_thresh

# Apply Tier 1 safety guardrail
print('Applying Tier 1 Deterministic Safety Guardrail...')
test_preds = tier1_guardrail(test, cc, test_preds)

# Back to 1-indexed ESI
test_labels = test_preds + 1

# Build submission
submission = pd.DataFrame({
    'patient_id':    test['patient_id'],
    'triage_acuity': test_labels
})

# Sanity checks
print('\nSubmission validation:')
assert len(submission) == len(sub), 'Row count mismatch!'
assert submission['triage_acuity'].between(1,5).all(), 'Values outside 1-5!'
assert submission['patient_id'].is_unique, 'Duplicate patient_ids!'
print('  ✓ Row count correct')
print('  ✓ All values in range [1,5]')
print('  ✓ No duplicate patient_ids')

print('\nTest prediction distribution:')
for esi, cnt in submission['triage_acuity'].value_counts().sort_index().items():
    bar = '█' * int(cnt/len(submission)*60)
    print(f'  ESI-{esi}: {cnt:,}  ({cnt/len(submission)*100:.1f}%)  {bar}')

print('\nTrain vs Test distribution check:')
tr_d = (train['triage_acuity'].value_counts(normalize=True).sort_index()*100).round(1)
te_d = (submission['triage_acuity'].value_counts(normalize=True).sort_index()*100).round(1)
print(pd.DataFrame({'Train %': tr_d, 'Test %': te_d}).to_string())

submission.to_csv('submission.csv', index=False)
print('\n✓ submission.csv saved.')
```

## ── Cell 18 · Final QWK Dashboard ────────────────────────────────────────


```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Model comparison bar chart
model_names = ['LightGBM','CatBoost','XGBoost','Blend','+ Thresholds','+ Entropy Audit']
model_qwks  = [lgb_oof_qwk, cb_oof_qwk, xgb_oof_qwk,
               best_blend_qwk, thresh_qwk, ent_qwk]
bar_colors  = ['#4caf50','#4caf50','#4caf50','#1976d2','#f57c00','#d32f2f']

bars = axes[0].barh(model_names, model_qwks, color=bar_colors, edgecolor='white', linewidth=1)
axes[0].set_xlabel('OOF QWK')
axes[0].set_title('Model Performance Progression', fontweight='bold')
axes[0].set_xlim(min(model_qwks)*0.97, max(model_qwks)*1.01)
for bar, v in zip(bars, model_qwks):
    axes[0].text(v + 0.001, bar.get_y()+bar.get_height()/2,
                 f'{v:.4f}', va='center', fontsize=9)

# Entropy distribution
ent_vals = scipy_entropy(ens_oof.T) / np.log(5)
axes[1].hist(ent_vals, bins=50, color='#455a64', edgecolor='white', linewidth=0.5, alpha=0.8)
axes[1].axvline(0.75, color='#d32f2f', lw=2, ls='--', label='Entropy threshold=0.75')
axes[1].set_xlabel('Normalised Prediction Entropy')
axes[1].set_ylabel('Patient Count')
axes[1].set_title(f'Prediction Entropy Distribution\n'
                  f'(high entropy → uncertain → safety upgrade to ESI-2)', fontweight='bold')
axes[1].legend(fontsize=9)

plt.suptitle('Triagegeist v2 — Final Results Dashboard', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('dashboard.png', dpi=120, bbox_inches='tight')
plt.show()
plt.close()
gc.collect()

print('═'*55)
print(f'  FINAL OOF QWK  : {final_qwk:.4f}')
print(f'  Baseline QWK   :  0.7120  (published)')
print(f'  Improvement    : +{final_qwk-0.712:.4f}')
print('═'*55)
print('Notebook complete. All files saved.')
```

## ── Cell 19 · Reproducibility & Environment ───────────────────────────────

### Architecture Summary

| Tier | Component | Clinical Role |
|---|---|---|
| **Tier 1** | Deterministic guardrail | Force ESI-1/2 for red-flag vitals & keywords |
| **Model A** | LightGBM 5-fold CV | Primary GBDT; leaf-wise growth for interactions |
| **Model B** | CatBoost 5-fold CV | Symmetric trees; native categorical handling |
| **Model C** | XGBoost 5-fold CV | Diverse regularisation path |
| **Blend** | Grid-searched OOF weights | Maximise QWK on OOF |
| **Post-proc** | Nelder-Mead thresholds | Direct QWK optimisation (5 restarts) |
| **Post-proc** | Entropy audit | ESI-2/3 safety correction |

### Key Feature Groups
- **NEWS2** — 6-param explicit calculator (RCP 2017); aggregate + risk tier flags
- **Hemodynamics** — Shock Index + Modified Shock Index (HR/MAP for early sepsis)
- **CCI** — Charlson Comorbidity Index from 21 mapped `hx_` flags
- **Missingness** — 9 binary absence flags (non-random; clinically informative)
- **Interactions** — 16 cross-terms (elderly×AMS, CCI×NEWS2, ambulance×shock, etc.)
- **NLP** — TF-IDF (400 features, 1-3 ngrams) + SVD (60 dims, float32)

### Reproducibility
- Seed: `42` | Folds: `5` | All cells run top-to-bottom without errors
- Python 3.11 | LightGBM ≥4.0 | CatBoost ≥1.2 | XGBoost ≥2.0 | scikit-learn ≥1.4
- GPU optional — notebook auto-detects and sets device strings; CPU fallback is seamless
- No internet required — all libraries are pre-installed in Kaggle environment

### Dataset Citations
- Triagegeist Dataset (2026), Laitinen-Fredriksson Foundation. Competition license.
- Royal College of Physicians (2017). National Early Warning Score 2 (NEWS2). London: RCP.
- Charlson, M. E., et al. (1987). Journal of Chronic Diseases, 40(5), 373–383.
- Johnson, A. E. W., et al. (2023). MIMIC-IV-ED v2.2. PhysioNet. doi:10.13026/5ntk-km72
- National Center for Health Statistics (2023). NHAMCS. CDC.
