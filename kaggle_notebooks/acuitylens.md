```python
pip install lightgbm
```


```python
pip install xgboost
```


```python
pip install catboost
```


```python
# ================================================================
# TRIAGEGEIST — EMERGENCY TRIAGE ACUITY PREDICTION
# Full multimodal ensemble: LightGBM + XGBoost + CatBoost
# ================================================================

# ================================================================
# SECTION 1: SETUP & DATA LOADING
# ================================================================
import os, sys, gc, re, time, warnings
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, classification_report,
    confusion_matrix, f1_score, mean_absolute_error, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed — interpretability plots will use feature importance only")

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 6)

SEED = 42
N_FOLDS = 5
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM = (1, 2)
SVD_COMPONENTS = 80

np.random.seed(SEED)
print("✅ Libraries loaded")

# ================================================================
# DATA LOADING
# ================================================================
INPUT_DIR = "/kaggle/input/triagegeist"
if not os.path.isdir(INPUT_DIR):
    for d in ["/kaggle/input/competitions/triagegeist", "triagegeist_data", "data", "."]:
        if os.path.isfile(os.path.join(d, "train.csv")):
            INPUT_DIR = d
            break

print(f"📂 Data directory: {INPUT_DIR}")

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
complaints = pd.read_csv(os.path.join(INPUT_DIR, "chief_complaints.csv"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
history_path = os.path.join(INPUT_DIR, "patient_history.csv")
history = pd.read_csv(history_path) if os.path.isfile(history_path) else None

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Complaints: {complaints.shape}")
if history is not None:
    print(f"Patient History: {history.shape}")

print(f"\n📋 Target distribution:")
target_dist = train['triage_acuity'].value_counts().sort_index()
for esi, count in target_dist.items():
    print(f"  ESI {esi}: {count:>6,} ({count/len(train)*100:>5.1f}%)")


# ================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ================================================================

colors_esi = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c', '#1976d2']
esi_labels = ['ESI 1\nResuscitation', 'ESI 2\nEmergent', 'ESI 3\nUrgent',
              'ESI 4\nLess Urgent', 'ESI 5\nNon-Urgent']

# --- 2.1 Target Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
acuity_counts = train['triage_acuity'].value_counts().sort_index()
bars = axes[0].bar(acuity_counts.index, acuity_counts.values, color=colors_esi,
                   edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('ESI Acuity Level')
axes[0].set_ylabel('Patient Count')
axes[0].set_title('Distribution of Triage Acuity (ESI 1-5)', fontweight='bold')
axes[0].set_xticks([1, 2, 3, 4, 5])
axes[0].set_xticklabels(esi_labels, fontsize=8)
for bar, v in zip(bars, acuity_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 200,
                f'{v:,}\n({v/len(train)*100:.1f}%)', ha='center', fontsize=9)

axes[1].pie(acuity_counts.values, labels=[f'ESI {i}' for i in acuity_counts.index],
            colors=colors_esi, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 10})
axes[1].set_title('Acuity Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_acuity_distribution.png', bbox_inches='tight')
plt.show()

print(f"📋 Class imbalance ratio (max/min): {acuity_counts.max()/acuity_counts.min():.1f}x")
print(f"   ESI 1 is the rarest — clinically expected but creates modeling challenges.")
print(f"   Misclassifying ESI 1-2 patients is the most dangerous error.")

# --- 2.2 Missingness Analysis ---
vital_cols_eda = ['systolic_bp', 'diastolic_bp', 'heart_rate',
                  'respiratory_rate', 'temperature_c', 'spo2']

miss_by_acuity = pd.DataFrame()
for col in vital_cols_eda:
    miss_rates = train.groupby('triage_acuity')[col].apply(lambda x: x.isna().mean() * 100)
    miss_by_acuity[col.replace('_', ' ').title()] = miss_rates

fig, ax = plt.subplots(figsize=(12, 6))
miss_by_acuity.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.3)
ax.set_xlabel('ESI Acuity Level', fontsize=12)
ax.set_ylabel('Missing Rate (%)', fontsize=12)
ax.set_title('Vital Sign Missingness by Acuity Level\n'
             'Lower-acuity patients have more missing vitals — clinically realistic',
             fontweight='bold')
ax.set_xticklabels([f'ESI {i}' for i in range(1, 6)], rotation=0)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('fig2_missingness_by_acuity.png', bbox_inches='tight')
plt.show()

print("\n📋 KEY FINDING: Missingness is inversely related to acuity severity.")
print("   → Missingness itself is a strong predictive feature, not noise to impute away.")

# --- 2.3 Vital Signs by Acuity ---
vital_plot_cols = ['systolic_bp', 'heart_rate', 'respiratory_rate',
                   'temperature_c', 'spo2', 'gcs_total']
vital_names = ['Systolic BP\n(mmHg)', 'Heart Rate\n(bpm)', 'Resp Rate\n(breaths/min)',
               'Temperature\n(°C)', 'SpO₂\n(%)', 'GCS\n(3-15)']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, (col, name) in enumerate(zip(vital_plot_cols, vital_names)):
    data_to_plot = [train[train['triage_acuity'] == a][col].dropna() for a in range(1, 6)]
    bp = axes[i].boxplot(data_to_plot, labels=[f'ESI {a}' for a in range(1, 6)],
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors_esi):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[i].set_title(name, fontweight='bold', fontsize=11)
    axes[i].set_xlabel('Acuity')

plt.suptitle('Vital Sign Distributions by Triage Acuity', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_vitals_by_acuity.png', bbox_inches='tight')
plt.show()

# --- 2.4 Arrival Mode and Demographics ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

arrival_acuity = pd.crosstab(train['arrival_mode'], train['triage_acuity'],
                              normalize='index') * 100
arrival_acuity.plot(kind='bar', stacked=True, ax=axes[0], color=colors_esi,
                    edgecolor='black', linewidth=0.3)
axes[0].set_ylabel('Percentage (%)')
axes[0].set_title('Acuity Distribution by Arrival Mode', fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].legend(title='ESI', labels=[f'ESI {i}' for i in range(1, 6)], fontsize=8)

for acuity in range(1, 6):
    subset = train[train['triage_acuity'] == acuity]['age']
    axes[1].hist(subset, bins=40, alpha=0.5, label=f'ESI {acuity}',
                 color=colors_esi[acuity-1], density=True)
axes[1].set_xlabel('Age (years)')
axes[1].set_ylabel('Density')
axes[1].set_title('Age Distribution by Triage Acuity', fontweight='bold')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig('fig4_arrival_age.png', bbox_inches='tight')
plt.show()

# --- 2.5 Chief Complaint Word Analysis ---
merged_cc = train[['patient_id', 'triage_acuity']].merge(complaints, on='patient_id', how='left')
stopwords = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'to', 'of', 'and',
             'in', 'for', 'with', 'on', 'at', 'by', 'from', 'or', 'that', 'this',
             'it', 'be', 'has', 'had', 'have', 'not', 'but', 'as', 'do', 'does',
             'no', 'so', 'if', 'will', 'can', 'been', 'he', 'she', 'they', 'his',
             'her', 'my', 'i', 'me', 'we', 'you', 'patient', 'pt', 'reports',
             'states', 'c/o', 'presents', 'presenting', 'old', 'year', 'yo', 'who'}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for idx, (acuity, title) in enumerate([(1, 'ESI 1 — Resuscitation'),
                                        (3, 'ESI 3 — Urgent'),
                                        (5, 'ESI 5 — Non-Urgent')]):
    subset_text = merged_cc[merged_cc['triage_acuity'] == acuity]['chief_complaint_raw'].dropna()
    all_words = ' '.join(subset_text).lower().split()
    filtered = [w for w in all_words if w not in stopwords and len(w) > 2]
    top_words = Counter(filtered).most_common(15)
    if top_words:
        words, counts = zip(*top_words)
        axes[idx].barh(range(len(words)), counts, color=colors_esi[acuity-1], alpha=0.7)
        axes[idx].set_yticks(range(len(words)))
        axes[idx].set_yticklabels(words, fontsize=10)
        axes[idx].invert_yaxis()
    axes[idx].set_title(title, fontweight='bold')
    axes[idx].set_xlabel('Frequency')

plt.suptitle('Chief Complaint Term Frequency by Acuity Level', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig5_complaint_words.png', bbox_inches='tight')
plt.show()


# ================================================================
# SECTION 3: FEATURE ENGINEERING
# ================================================================

HIGH_ACUITY_KW = [
    "chest pain", "shortness of breath", "sob", "dyspnea", "cardiac arrest",
    "stroke", "cva", "altered mental", "unconscious", "unresponsive",
    "seizure", "trauma", "mva", "motor vehicle", "gunshot", "stab",
    "suicide", "overdose", "anaphylaxis", "difficulty breathing",
    "respiratory distress", "sepsis", "gi bleed", "hemorrhage",
    "syncope", "severe pain", "worst headache", "burn", "fracture",
    "amputation", "cardiac", "intubation", "code blue",
]
LOW_ACUITY_KW = [
    "refill", "prescription", "medication refill", "suture removal",
    "follow up", "follow-up", "recheck", "sore throat", "cold symptoms",
    "cough", "runny nose", "rash", "insect bite", "earache", "ear pain",
    "pink eye", "conjunctivitis", "sprain", "strain", "dental", "toothache",
]


def compute_news2_vectorized(df):
    """Vectorized NEWS2 score computation per Royal College of Physicians criteria."""
    score = np.zeros(len(df))

    rr = df.get("respiratory_rate")
    if rr is not None:
        rr_v = rr.values
        score += np.where(pd.isna(rr_v), 0,
                 np.where(rr_v <= 8, 3,
                 np.where(rr_v <= 11, 1,
                 np.where(rr_v <= 20, 0,
                 np.where(rr_v <= 24, 2, 3)))))

    spo2 = df.get("spo2")
    if spo2 is not None:
        s_v = spo2.values
        score += np.where(pd.isna(s_v), 0,
                 np.where(s_v <= 91, 3,
                 np.where(s_v <= 93, 2,
                 np.where(s_v <= 95, 1, 0))))

    sbp = df.get("systolic_bp")
    if sbp is not None:
        b_v = sbp.values
        score += np.where(pd.isna(b_v), 0,
                 np.where(b_v <= 90, 3,
                 np.where(b_v <= 100, 2,
                 np.where(b_v <= 110, 1,
                 np.where(b_v <= 219, 0, 3)))))

    hr = df.get("heart_rate")
    if hr is not None:
        h_v = hr.values
        score += np.where(pd.isna(h_v), 0,
                 np.where(h_v <= 40, 3,
                 np.where(h_v <= 50, 1,
                 np.where(h_v <= 90, 0,
                 np.where(h_v <= 110, 1,
                 np.where(h_v <= 130, 2, 3))))))

    temp = df.get("temperature_c")
    if temp is not None:
        t_v = temp.values
        score += np.where(pd.isna(t_v), 0,
                 np.where(t_v <= 35.0, 3,
                 np.where(t_v <= 36.0, 1,
                 np.where(t_v <= 38.0, 0,
                 np.where(t_v <= 39.0, 1, 2)))))

    gcs = df.get("gcs_total")
    if gcs is not None:
        g_v = gcs.values
        score += np.where(pd.isna(g_v), 0, np.where(g_v < 15, 3, 0))

    return score


def build_all_features(df, complaints_df, history_df):
    """Build complete feature set for a dataframe."""
    feat = pd.DataFrame(index=df.index)

    # ===== Vitals (raw) =====
    vital_cols = ["systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
                  "temperature_c", "spo2", "gcs_total", "pain_score"]
    for col in vital_cols:
        if col in df.columns:
            feat[col] = df[col].copy()

    if "pain_score" in feat.columns:
        feat["pain_score"] = feat["pain_score"].replace(-1, np.nan)

    # ===== Missingness indicators =====
    miss_cols = ["systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
                 "temperature_c", "spo2", "pain_score"]
    for col in miss_cols:
        if col in df.columns:
            vals = df[col] if col != "pain_score" else feat["pain_score"]
            feat[f"{col}_missing"] = vals.isna().astype(int)

    miss_ind_cols = [c for c in feat.columns if c.endswith("_missing")]
    feat["total_vitals_missing"] = feat[miss_ind_cols].sum(axis=1)

    # ===== Derived vitals =====
    sbp = feat.get("systolic_bp")
    dbp = feat.get("diastolic_bp")
    hr = feat.get("heart_rate")
    rr = feat.get("respiratory_rate")
    temp = feat.get("temperature_c")

    if sbp is not None and dbp is not None:
        feat["map_pressure"] = (sbp + 2 * dbp) / 3
        feat["pulse_pressure"] = sbp - dbp

    if hr is not None and sbp is not None:
        feat["shock_index"] = hr / sbp.replace(0, np.nan)
        if "map_pressure" in feat.columns:
            feat["modified_shock_index"] = hr / feat["map_pressure"].replace(0, np.nan)

    if hr is not None and rr is not None:
        feat["hr_rr_product"] = hr * rr

    if sbp is not None and rr is not None:
        feat["sbp_rr_ratio"] = sbp / rr.replace(0, np.nan)

    if temp is not None:
        feat["temp_deviation"] = (temp - 37.0).abs()
        feat["is_febrile"] = (temp >= 38.0).astype(int)
        feat["is_hypothermic"] = (temp < 36.0).astype(int)

    spo2_c = feat.get("spo2")
    if spo2_c is not None:
        feat["spo2_critical"] = (spo2_c < 92).astype(int)
        feat["spo2_low"] = (spo2_c < 95).astype(int)

    gcs = feat.get("gcs_total")
    if gcs is not None:
        feat["gcs_severe"] = (gcs <= 8).astype(int)
        feat["gcs_moderate"] = ((gcs > 8) & (gcs <= 12)).astype(int)
        feat["gcs_mild"] = ((gcs > 12) & (gcs < 15)).astype(int)

    if sbp is not None:
        feat["bp_hypotensive"] = (sbp < 90).astype(int)
        feat["bp_hypertensive"] = (sbp > 180).astype(int)

    if hr is not None:
        feat["hr_bradycardia"] = (hr < 60).astype(int)
        feat["hr_tachycardia"] = (hr > 100).astype(int)

    if rr is not None:
        feat["rr_bradypnea"] = (rr < 12).astype(int)
        feat["rr_tachypnea"] = (rr > 20).astype(int)

    # Precomputed columns from raw data
    for col in ["mean_arterial_pressure", "pulse_pressure", "shock_index",
                "news2_score", "bmi"]:
        if col in df.columns and col not in feat.columns:
            feat[col] = df[col]

    if "news2_score" not in feat.columns and "news2_score" not in df.columns:
        feat["news2_computed"] = compute_news2_vectorized(df)

    # ===== Demographics =====
    if "age" in df.columns:
        feat["age"] = df["age"]
        feat["age_sq"] = df["age"] ** 2
        feat["age_log"] = np.log1p(df["age"])
        feat["is_pediatric"] = (df["age"] < 18).astype(int)
        feat["is_elderly"] = (df["age"] >= 65).astype(int)
        feat["is_very_elderly"] = (df["age"] >= 80).astype(int)

    for col in ["age_group"]:
        if col in df.columns:
            feat[col] = df[col]

    # ===== Time features =====
    if "arrival_hour" in df.columns:
        feat["arrival_hour"] = df["arrival_hour"]
        feat["arrival_hour_sin"] = np.sin(2 * np.pi * df["arrival_hour"] / 24)
        feat["arrival_hour_cos"] = np.cos(2 * np.pi * df["arrival_hour"] / 24)
        hour = df["arrival_hour"]
        feat["shift_night"] = ((hour >= 23) | (hour < 7)).astype(int)
        feat["shift_day"] = ((hour >= 7) & (hour < 15)).astype(int)
        feat["shift_evening"] = ((hour >= 15) & (hour < 23)).astype(int)

    for col in ["shift"]:
        if col in df.columns:
            feat[col] = df[col]

    # ===== Categoricals =====
    cat_cols = ["arrival_mode", "arrival_day", "sex", "language", "insurance_type",
                "arrival_season", "transport_origin", "mental_status_triage",
                "chief_complaint_system", "pain_location"]
    for col in cat_cols:
        if col in df.columns:
            feat[col] = df[col]

    # ===== Prior visits =====
    for col in ["num_prior_visits", "num_prior_ed_visits_12m",
                "num_prior_admissions_12m", "num_active_medications", "num_comorbidities"]:
        if col in df.columns:
            feat[col] = df[col]

    if "num_prior_ed_visits_12m" in df.columns and "num_prior_admissions_12m" in df.columns:
        feat["visits_x_admissions"] = df["num_prior_ed_visits_12m"] * df["num_prior_admissions_12m"]
        feat["admission_rate"] = (df["num_prior_admissions_12m"] /
                                  df["num_prior_ed_visits_12m"].replace(0, 1))

    # ===== Chief Complaint NLP =====
    merged_cc = df[["patient_id"]].merge(complaints_df, on="patient_id", how="left")
    text = merged_cc["chief_complaint_raw"].fillna("").str.lower()

    feat["complaint_char_len"] = text.str.len()
    feat["complaint_word_count"] = text.str.split().str.len().fillna(0)
    feat["complaint_avg_word_len"] = feat["complaint_char_len"] / feat["complaint_word_count"].replace(0, 1)

    feat["high_acuity_kw_count"] = sum(
        text.str.contains(kw, na=False).astype(int) for kw in HIGH_ACUITY_KW)
    feat["low_acuity_kw_count"] = sum(
        text.str.contains(kw, na=False).astype(int) for kw in LOW_ACUITY_KW)
    feat["acuity_kw_diff"] = feat["high_acuity_kw_count"] - feat["low_acuity_kw_count"]

    feat["has_chest_pain"] = text.str.contains("chest pain|chest tightness", na=False).astype(int)
    feat["has_sob"] = text.str.contains(r"shortness of breath|sob|difficulty breathing|dyspnea", na=False).astype(int)
    feat["has_altered_mental"] = text.str.contains(r"altered mental|confused|unresponsive|unconscious|ams", na=False).astype(int)
    feat["has_trauma"] = text.str.contains(r"trauma|accident|mva|fall|injury|laceration|fracture", na=False).astype(int)
    feat["has_abdominal"] = text.str.contains(r"abdominal pain|abd pain|stomach|nausea|vomiting", na=False).astype(int)
    feat["has_neuro"] = text.str.contains(r"headache|seizure|weakness|numbness|dizziness|stroke|cva", na=False).astype(int)
    feat["has_cardiac"] = text.str.contains(r"palpitation|cardiac|heart|arrhythmia|afib|mi", na=False).astype(int)
    feat["has_psych"] = text.str.contains(r"suicid|overdose|self.?harm|depression|anxiety|psych", na=False).astype(int)

    # ===== Comorbidity features =====
    if history_df is not None:
        merged_hx = df[["patient_id"]].merge(history_df, on="patient_id", how="left")
        hx_cols = [c for c in merged_hx.columns if c.startswith("hx_")]
        for col in hx_cols:
            feat[col] = merged_hx[col].values
        if hx_cols:
            feat["comorbidity_burden"] = merged_hx[hx_cols].sum(axis=1).values

            high_risk = ["hx_heart_failure", "hx_copd", "hx_malignancy", "hx_dementia",
                         "hx_ckd", "hx_liver_disease", "hx_hiv"]
            hr_present = [c for c in high_risk if c in merged_hx.columns]
            if hr_present:
                feat["high_risk_comorbidity"] = merged_hx[hr_present].sum(axis=1).values

            cv_cols = ["hx_hypertension", "hx_coronary_artery_disease", "hx_heart_failure",
                       "hx_atrial_fibrillation", "hx_peripheral_vascular_disease"]
            cv_present = [c for c in cv_cols if c in merged_hx.columns]
            if cv_present:
                feat["cv_risk_count"] = merged_hx[cv_present].sum(axis=1).values

            met_cols = ["hx_diabetes", "hx_obesity", "hx_dyslipidemia"]
            met_present = [c for c in met_cols if c in merged_hx.columns]
            if met_present:
                feat["metabolic_risk_count"] = merged_hx[met_present].sum(axis=1).values

    return feat, text


print("⚙️ Building features for train and test...")
t0 = time.time()
train_feats, train_text = build_all_features(train, complaints, history)
test_feats, test_text = build_all_features(test, complaints, history)
print(f"  Features built in {time.time()-t0:.1f}s")
print(f"  Train: {train_feats.shape}, Test: {test_feats.shape}")


# ================================================================
# SECTION 4: TF-IDF + SVD
# ================================================================
print(f"🔤 Building TF-IDF features (max={TFIDF_MAX_FEATURES}, SVD={SVD_COMPONENTS})...")

tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM,
    sublinear_tf=True,
    strip_accents="unicode",
    token_pattern=r"(?u)\b\w+\b",
    min_df=5,
    max_df=0.95,
)

all_text = pd.concat([train_text, test_text], axis=0).fillna("")
tfidf_matrix = tfidf.fit_transform(all_text)

svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)
svd_features = svd.fit_transform(tfidf_matrix)
print(f"  SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

n_train = len(train_text)
train_tfidf = pd.DataFrame(
    svd_features[:n_train],
    columns=[f"tfidf_svd_{i}" for i in range(SVD_COMPONENTS)],
    index=train_text.index
)
test_tfidf = pd.DataFrame(
    svd_features[n_train:],
    columns=[f"tfidf_svd_{i}" for i in range(SVD_COMPONENTS)],
    index=test_text.index
)


# ================================================================
# SECTION 5: COMBINE AND ENCODE
# ================================================================
X_train = pd.concat([train_feats, train_tfidf], axis=1)
X_test = pd.concat([test_feats, test_tfidf], axis=1)

cat_cols = [col for col in X_train.columns
            if X_train[col].dtype == "object" or col in [
                "arrival_mode", "arrival_day", "sex", "language", "insurance_type",
                "arrival_season", "transport_origin", "mental_status_triage",
                "chief_complaint_system", "pain_location", "age_group", "shift"
            ]]
cat_cols = [c for c in cat_cols if c in X_train.columns]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

for col in X_train.columns:
    if X_train[col].dtype == "object":
        le = LabelEncoder()
        combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

y = train["triage_acuity"].copy()
y_model = y - 1  # 0-indexed for model

print(f"📦 Final: {X_train.shape[1]} features")
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")


# ================================================================
# SECTION 6: CROSS-VALIDATION TRAINING
# ================================================================

def qwk(y_true, y_pred):
    """Quadratic Weighted Kappa — primary evaluation metric."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def optimize_weights(oof_list, y_true):
    """Find ensemble weights that maximize QWK."""
    def neg_qwk(weights):
        w = np.array(weights)
        w = w / w.sum()
        blended = sum(w[i] * oof_list[i] for i in range(len(w)))
        return -qwk(y_true, blended.argmax(axis=1) + 1)

    n = len(oof_list)
    result = minimize(neg_qwk, [1.0/n]*n, method="SLSQP",
                      bounds=[(0.01, 1.0)]*n,
                      constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
                      options={"maxiter": 1000})
    w = result.x / result.x.sum()
    return w, -result.fun


def optimize_thresholds(probs, y_true, n_classes=5):
    """Optimize ordinal thresholds for QWK."""
    def neg_qwk_t(thresholds):
        cont = (probs * np.arange(1, n_classes+1)).sum(axis=1)
        pred = np.ones(len(cont), dtype=int)
        for i, t in enumerate(sorted(thresholds)):
            pred[cont > t] = i + 2
        return -qwk(y_true, pred)

    result = minimize(neg_qwk_t, [1.5, 2.5, 3.5, 4.5], method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 0.001, "fatol": 0.0001})
    return sorted(result.x), -result.fun


def apply_thresholds(probs, thresholds, n_classes=5):
    """Apply optimized thresholds to probability predictions."""
    cont = (probs * np.arange(1, n_classes+1)).sum(axis=1)
    pred = np.ones(len(cont), dtype=int)
    for i, t in enumerate(sorted(thresholds)):
        pred[cont > t] = i + 2
    return pred


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
n_classes = 5

oof_lgb = np.zeros((len(X_train), n_classes))
oof_xgb = np.zeros((len(X_train), n_classes))
oof_cb  = np.zeros((len(X_train), n_classes))

test_lgb = np.zeros((len(X_test), n_classes))
test_xgb = np.zeros((len(X_test), n_classes))
test_cb  = np.zeros((len(X_test), n_classes))

lgb_importances = np.zeros(X_train.shape[1])
fold_scores = {"lgb": [], "xgb": [], "cb": []}

LGB_PARAMS = {
    "objective": "multiclass", "num_class": 5, "metric": "multi_logloss",
    "boosting_type": "gbdt", "n_estimators": 3000, "learning_rate": 0.03,
    "num_leaves": 127, "max_depth": 8, "min_child_samples": 30,
    "subsample": 0.8, "colsample_bytree": 0.6,
    "reg_alpha": 0.5, "reg_lambda": 1.0,
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
}

XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": 5, "eval_metric": "mlogloss",
    "n_estimators": 3000, "learning_rate": 0.03,
    "max_depth": 8, "min_child_weight": 30,
    "subsample": 0.8, "colsample_bytree": 0.6,
    "reg_alpha": 0.5, "reg_lambda": 1.0,
    "tree_method": "hist", "random_state": SEED, "n_jobs": -1, "verbosity": 0,
}

CB_PARAMS = {
    "loss_function": "MultiClass", "iterations": 3000, "learning_rate": 0.03,
    "depth": 8, "l2_leaf_reg": 3.0, "min_data_in_leaf": 30,
    "random_seed": SEED, "verbose": 0, "task_type": "CPU",
    "bootstrap_type": "Bernoulli", "subsample": 0.8,
}

total_start = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_model)):
    print(f"\n{'='*50} FOLD {fold+1}/{N_FOLDS} {'='*50}")

    Xtr, ytr = X_train.iloc[train_idx], y_model.iloc[train_idx]
    Xvl, yvl = X_train.iloc[val_idx], y_model.iloc[val_idx]

    # LightGBM
    t0 = time.time()
    mdl_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
    mdl_lgb.fit(Xtr, ytr, eval_set=[(Xvl, yvl)],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    p_lgb = mdl_lgb.predict_proba(Xvl)
    s_lgb = qwk(yvl + 1, p_lgb.argmax(axis=1) + 1)
    fold_scores["lgb"].append(s_lgb)
    oof_lgb[val_idx] = p_lgb
    test_lgb += mdl_lgb.predict_proba(X_test) / N_FOLDS
    lgb_importances += mdl_lgb.feature_importances_ / N_FOLDS
    print(f"  LGB QWK: {s_lgb:.5f} ({time.time()-t0:.0f}s)")

    # XGBoost
    t0 = time.time()
    mdl_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    mdl_xgb.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
    p_xgb = mdl_xgb.predict_proba(Xvl)
    s_xgb = qwk(yvl + 1, p_xgb.argmax(axis=1) + 1)
    fold_scores["xgb"].append(s_xgb)
    oof_xgb[val_idx] = p_xgb
    test_xgb += mdl_xgb.predict_proba(X_test) / N_FOLDS
    print(f"  XGB QWK: {s_xgb:.5f} ({time.time()-t0:.0f}s)")

    # CatBoost
    t0 = time.time()
    cb_train_pool = cb.Pool(Xtr, label=ytr)
    cb_val_pool = cb.Pool(Xvl, label=yvl)
    mdl_cb = cb.CatBoostClassifier(**CB_PARAMS)
    mdl_cb.fit(cb_train_pool, eval_set=cb_val_pool, early_stopping_rounds=100, verbose=0)
    p_cb = mdl_cb.predict_proba(Xvl)
    s_cb = qwk(yvl + 1, p_cb.argmax(axis=1) + 1)
    fold_scores["cb"].append(s_cb)
    oof_cb[val_idx] = p_cb
    test_cb += mdl_cb.predict_proba(X_test) / N_FOLDS
    print(f"  CB  QWK: {s_cb:.5f} ({time.time()-t0:.0f}s)")

    del mdl_lgb, mdl_xgb, mdl_cb, cb_train_pool, cb_val_pool
    gc.collect()

print(f"\n{'='*60}")
print(f"  CV SUMMARY (Mean ± Std)")
print(f"{'='*60}")
for name in ["lgb", "xgb", "cb"]:
    s = fold_scores[name]
    print(f"  {name.upper()}: {np.mean(s):.5f} ± {np.std(s):.5f}")


# ================================================================
# SECTION 7: ENSEMBLE OPTIMIZATION
# ================================================================
print("🔧 Optimizing ensemble weights...")
opt_w, opt_qwk = optimize_weights([oof_lgb, oof_xgb, oof_cb], y)
print(f"  Weights: LGB={opt_w[0]:.3f}, XGB={opt_w[1]:.3f}, CB={opt_w[2]:.3f}")
print(f"  Ensemble QWK: {opt_qwk:.5f}")

oof_blend = opt_w[0]*oof_lgb + opt_w[1]*oof_xgb + opt_w[2]*oof_cb

print("\n🔧 Optimizing thresholds...")
opt_thresh, thresh_qwk = optimize_thresholds(oof_blend, y)
default_qwk = qwk(y, oof_blend.argmax(axis=1) + 1)

print(f"  Default argmax QWK: {default_qwk:.5f}")
print(f"  Threshold QWK:      {thresh_qwk:.5f}")

use_thresh = thresh_qwk > default_qwk
final_qwk = max(thresh_qwk, default_qwk)
method = "threshold-optimized" if use_thresh else "argmax"

if use_thresh:
    oof_preds = apply_thresholds(oof_blend, opt_thresh)
else:
    oof_preds = oof_blend.argmax(axis=1) + 1

print(f"\n🏆 FINAL OOF QWK: {final_qwk:.5f} ({method})")


# ================================================================
# SECTION 8: COMPREHENSIVE RESULTS
# ================================================================
y_true = y.values
y_pred = oof_preds

print("="*60)
print("📊 COMPREHENSIVE EVALUATION RESULTS")
print("="*60)

metrics = {
    'QWK (Quadratic Weighted Kappa)': qwk(y_true, y_pred),
    'Macro F1': f1_score(y_true, y_pred, average='macro'),
    'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
    'Accuracy': accuracy_score(y_true, y_pred),
    'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
    'MAE (Ordinal)': mean_absolute_error(y_true, y_pred),
    'One-Level-Off Accuracy': np.mean(np.abs(y_true - y_pred) <= 1),
}

for name, val in metrics.items():
    print(f"  {name:<35s}: {val:.5f}")

print(f"\n📋 Per-Class Performance:")
report = classification_report(y_true, y_pred, digits=4,
                                target_names=[f'ESI {i}' for i in range(1, 6)])
print(report)

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[f'ESI {i}' for i in range(1, 6)],
            yticklabels=[f'ESI {i}' for i in range(1, 6)])
axes[0].set_xlabel('Predicted Acuity', fontsize=11)
axes[0].set_ylabel('True Acuity', fontsize=11)
axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')

cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
            xticklabels=[f'ESI {i}' for i in range(1, 6)],
            yticklabels=[f'ESI {i}' for i in range(1, 6)])
axes[1].set_xlabel('Predicted Acuity', fontsize=11)
axes[1].set_ylabel('True Acuity', fontsize=11)
axes[1].set_title('Confusion Matrix (Row-Normalized)', fontweight='bold')

plt.suptitle('Ensemble Model — Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig6_confusion_matrix.png', bbox_inches='tight')
plt.show()

# Clinical Safety
print("\n" + "="*60)
print("🚨 CLINICAL SAFETY ANALYSIS")
print("="*60)

severe_undertriage_count = 0
total_critical = 0
for true_esi in [1, 2]:
    for pred_esi in [4, 5]:
        count = cm[true_esi-1][pred_esi-1]
        severe_undertriage_count += count
        if count > 0:
            print(f"  ⚠️  True ESI {true_esi} → Predicted ESI {pred_esi}: {count} cases")
    total_critical += np.sum(cm[true_esi-1, :])

severe_ut_rate = severe_undertriage_count / total_critical * 100 if total_critical > 0 else 0
print(f"\n  Severe undertriage rate (ESI 1-2 → ESI 4-5): "
      f"{severe_undertriage_count}/{total_critical} = {severe_ut_rate:.2f}%")

for esi in [1, 2]:
    mask = y_true == esi
    if mask.sum() > 0:
        recall = np.mean(y_pred[mask] == esi)
        within_one = np.mean(np.abs(y_pred[mask] - esi) <= 1)
        print(f"  ESI {esi} recall: {recall:.4f} | Within ±1: {within_one:.4f} | n={mask.sum()}")

correct = np.mean(y_pred == y_true)
undertriage = np.mean(y_pred > y_true)
overtriage = np.mean(y_pred < y_true)
print(f"\n  Error direction:")
print(f"    Exact match:    {correct*100:.1f}%")
print(f"    Undertriage:    {undertriage*100:.1f}%")
print(f"    Overtriage:     {overtriage*100:.1f}%")


# ================================================================
# SECTION 9: INTERPRETABILITY
# ================================================================

# Feature Importance
imp_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": lgb_importances,
}).sort_values("importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 12))
top_n = 30
top_imp = imp_df.head(top_n)
ax.barh(range(top_n), top_imp['importance'].values, color='#2196F3', alpha=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_imp['feature'].values, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Split Count)', fontsize=11)
ax.set_title(f'Top {top_n} Features — LightGBM Ensemble', fontweight='bold')
plt.tight_layout()
plt.savefig('fig7_feature_importance.png', bbox_inches='tight')
plt.show()

# SHAP Analysis (if available)
if SHAP_AVAILABLE:
    print("Computing SHAP values (sampled for speed)...")

    mdl_shap = lgb.LGBMClassifier(**{**LGB_PARAMS, 'n_estimators': 1000})
    mdl_shap.fit(X_train, y_model)

    sample_idx = np.random.choice(len(X_train), size=min(2000, len(X_train)), replace=False)
    X_sample = X_train.iloc[sample_idx]

    explainer = shap.TreeExplainer(mdl_shap)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        shap_imp = pd.DataFrame({
            'feature': X_train.columns,
            'mean_abs_shap': np.mean(mean_abs_shap, axis=0)
        }).sort_values('mean_abs_shap', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 10))
        top_shap = shap_imp.head(25)
        ax.barh(range(25), top_shap['mean_abs_shap'].values, color='#E91E63', alpha=0.7)
        ax.set_yticks(range(25))
        ax.set_yticklabels(top_shap['feature'].values, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value| (across all classes)', fontsize=11)
        ax.set_title('Top 25 Features by SHAP Importance', fontweight='bold')
        plt.tight_layout()
        plt.savefig('fig8_shap_importance.png', bbox_inches='tight')
        plt.show()

    if isinstance(shap_values, list) and len(shap_values) >= 2:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[0], X_sample,
                         feature_names=X_train.columns.tolist(),
                         show=False, max_display=20)
        plt.title('SHAP Values — ESI 1 (Resuscitation)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('fig9_shap_esi1_beeswarm.png', bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[4], X_sample,
                         feature_names=X_train.columns.tolist(),
                         show=False, max_display=20)
        plt.title('SHAP Values — ESI 5 (Non-Urgent)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('fig10_shap_esi5_beeswarm.png', bbox_inches='tight')
        plt.show()

    del explainer, shap_values, mdl_shap
    gc.collect()

else:
    print("⚠️ SHAP not available — install with: pip install shap")


# ================================================================
# SECTION 10: EQUITY & SUBGROUP ANALYSIS
# ================================================================

def compute_subgroup_metrics(df, y_true_arr, y_pred_arr, group_col, min_n=100):
    """Compute comprehensive per-subgroup metrics."""
    results = []
    for group in sorted(df[group_col].dropna().unique()):
        mask = (df[group_col] == group).values
        if mask.sum() < min_n:
            continue
        yt = y_true_arr[mask]
        yp = y_pred_arr[mask]

        critical_mask = np.isin(yt, [1, 2])
        critical_recall = np.nan
        if critical_mask.sum() > 10:
            critical_recall = np.mean(np.abs(yp[critical_mask] - yt[critical_mask]) <= 1)

        results.append({
            'group': str(group),
            'n': int(mask.sum()),
            'macro_f1': f1_score(yt, yp, average='macro', zero_division=0),
            'qwk': cohen_kappa_score(yt, yp, weights='quadratic'),
            'mae': mean_absolute_error(yt, yp),
            'one_off_acc': np.mean(np.abs(yt - yp) <= 1),
            'undertriage_rate': np.mean(yp > yt) * 100,
            'overtriage_rate': np.mean(yp < yt) * 100,
            'critical_within_1': critical_recall,
        })
    return pd.DataFrame(results)


equity_dims = {}
for col_name, col_key in [('Sex', 'sex'), ('Age Group', 'age_group'),
                            ('Arrival Mode', 'arrival_mode'), ('Language', 'language'),
                            ('Insurance Type', 'insurance_type')]:
    if col_key in train.columns:
        equity_dims[col_name] = col_key

print("="*60)
print("⚖️  EQUITY & SUBGROUP ANALYSIS")
print("="*60)

equity_results_all = {}
n_dims = len(equity_dims)

fig, axes = plt.subplots(n_dims, 2, figsize=(16, 4.5 * n_dims))
if n_dims == 1:
    axes = axes.reshape(1, -1)

for idx, (dim_name, dim_col) in enumerate(equity_dims.items()):
    subgroup_df = compute_subgroup_metrics(train, y_true, oof_preds, dim_col)
    equity_results_all[dim_name] = subgroup_df

    if len(subgroup_df) == 0:
        continue

    subgroup_sorted = subgroup_df.sort_values('qwk', ascending=True)
    bars1 = axes[idx, 0].barh(subgroup_sorted['group'], subgroup_sorted['qwk'],
                               color='#2196F3', alpha=0.7)
    axes[idx, 0].axvline(x=final_qwk, color='red', linestyle='--', alpha=0.7,
                          label=f'Overall QWK={final_qwk:.3f}')
    axes[idx, 0].set_xlabel('QWK')
    axes[idx, 0].set_title(f'QWK by {dim_name}', fontweight='bold')
    axes[idx, 0].legend(fontsize=8)
    for bar, n in zip(bars1, subgroup_sorted['n']):
        axes[idx, 0].text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                          f'n={n:,}', va='center', fontsize=8)

    subgroup_sorted2 = subgroup_df.sort_values('undertriage_rate', ascending=True)
    axes[idx, 1].barh(subgroup_sorted2['group'], subgroup_sorted2['undertriage_rate'],
                      color='#FF5722', alpha=0.7)
    overall_ut = np.mean(oof_preds > y_true) * 100
    axes[idx, 1].axvline(x=overall_ut, color='red', linestyle='--', alpha=0.7,
                          label=f'Overall={overall_ut:.1f}%')
    axes[idx, 1].set_xlabel('Undertriage Rate (%)')
    axes[idx, 1].set_title(f'Undertriage Rate by {dim_name}', fontweight='bold')
    axes[idx, 1].legend(fontsize=8)

    print(f"\n--- {dim_name} ---")
    print(subgroup_df.to_string(index=False, float_format='%.3f'))

plt.suptitle('Equity Analysis: Performance & Undertriage Across Patient Subgroups',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig11_equity_analysis.png', bbox_inches='tight')
plt.show()

# Elderly analysis
print("\n" + "="*60)
print("👴 ELDERLY PATIENT ANALYSIS (≥65 years)")
print("="*60)

if 'age' in train.columns:
    elderly_mask = (train['age'] >= 65).values
    young_mask = ~elderly_mask

    for label, mask in [("Elderly (≥65)", elderly_mask), ("Non-Elderly (<65)", young_mask)]:
        yt_sub = y_true[mask]
        yp_sub = oof_preds[mask]
        print(f"\n  {label}: n={mask.sum():,}")
        print(f"    QWK:              {qwk(yt_sub, yp_sub):.4f}")
        print(f"    Macro F1:         {f1_score(yt_sub, yp_sub, average='macro'):.4f}")
        print(f"    MAE:              {mean_absolute_error(yt_sub, yp_sub):.4f}")
        print(f"    Undertriage rate: {np.mean(yp_sub > yt_sub)*100:.1f}%")
        print(f"    Overtriage rate:  {np.mean(yp_sub < yt_sub)*100:.1f}%")
        for esi in range(1, 6):
            esi_mask = yt_sub == esi
            if esi_mask.sum() > 0:
                recall = np.mean(yp_sub[esi_mask] == esi)
                print(f"    ESI {esi} recall: {recall:.3f} (n={esi_mask.sum()})")

    elderly_ut = np.mean(oof_preds[elderly_mask] > y_true[elderly_mask])
    young_ut = np.mean(oof_preds[young_mask] > y_true[young_mask])
    print(f"\n  📋 Undertriage disparity: Elderly {elderly_ut*100:.1f}% vs Non-Elderly {young_ut*100:.1f}%")
    if abs(elderly_ut - young_ut) > 0.02:
        print(f"  ⚠️  Gap of {abs(elderly_ut - young_ut)*100:.1f} pp — warrants monitoring")
    else:
        print(f"  ✅ Gap of {abs(elderly_ut - young_ut)*100:.1f} pp — within acceptable range")


# ================================================================
# SECTION 11: POTENTIAL UNDERTRIAGE DETECTION
# ================================================================
print("="*60)
print("🔍 POTENTIAL UNDERTRIAGE DETECTION")
print("="*60)

oof_confidence = np.max(oof_blend, axis=1)
severity_gap = y_true - oof_preds
potential_ut_mask = (severity_gap >= 2) & (oof_confidence >= 0.5)
n_flagged = potential_ut_mask.sum()

print(f"  Cases flagged: {n_flagged} / {len(y_true)} ({n_flagged/len(y_true)*100:.2f}%)")

if n_flagged > 0:
    flagged_idx = np.where(potential_ut_mask)[0]
    flagged_df = train.iloc[flagged_idx].copy()
    flagged_df['predicted_acuity'] = oof_preds[flagged_idx]
    flagged_df['severity_gap'] = severity_gap[flagged_idx]
    flagged_df['model_confidence'] = oof_confidence[flagged_idx]

    print(f"\n  Label distribution of flagged cases:")
    for esi in sorted(flagged_df['triage_acuity'].unique()):
        n = (flagged_df['triage_acuity'] == esi).sum()
        print(f"    ESI {esi}: {n} ({n/n_flagged*100:.1f}%)")

    vital_compare = ['heart_rate', 'systolic_bp', 'spo2', 'respiratory_rate', 'gcs_total']
    print(f"\n  Clinical profile vs same-label non-flagged:")
    for col in vital_compare:
        if col in train.columns:
            flagged_vals = flagged_df[col].dropna()
            same_label_mask = (train['triage_acuity'].isin(flagged_df['triage_acuity'].unique())
                               & ~potential_ut_mask)
            nonflagged_vals = train.loc[same_label_mask, col].dropna()
            if len(flagged_vals) > 0 and len(nonflagged_vals) > 0:
                print(f"    {col:>20s}: Flagged={flagged_vals.mean():.1f} "
                      f"vs Non-flagged={nonflagged_vals.mean():.1f}")

    if 'disposition' in train.columns:
        admit_keywords = ['admit', 'icu', 'observation', 'transfer']
        flagged_admit = flagged_df['disposition'].str.lower().str.contains(
            '|'.join(admit_keywords), na=False).mean()
        overall_admit = train['disposition'].str.lower().str.contains(
            '|'.join(admit_keywords), na=False).mean()
        print(f"\n  Admission-like disposition: Flagged={flagged_admit*100:.1f}% "
              f"vs Overall={overall_admit*100:.1f}%")
        if flagged_admit > overall_admit:
            print(f"  ⚠️  Flagged cases have {flagged_admit/max(overall_admit,0.001):.1f}x higher admission rate")

# Visualize undertriage detection
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

gap_values = y_true - oof_preds
axes[0].hist(gap_values, bins=np.arange(-4.5, 5.5, 1), color='#2196F3', alpha=0.7,
             edgecolor='black', linewidth=0.5)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect agreement')
axes[0].set_xlabel('Label ESI - Predicted ESI', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title('Severity Gap Distribution\n(Positive = Model predicts more urgent)',
                  fontweight='bold')
axes[0].legend()

n_under = (gap_values < 0).sum()
n_exact = (gap_values == 0).sum()
n_over = (gap_values > 0).sum()
axes[0].text(0.02, 0.95,
             f'Model undertriages: {n_under} ({n_under/len(gap_values)*100:.1f}%)\n'
             f'Exact match: {n_exact} ({n_exact/len(gap_values)*100:.1f}%)\n'
             f'Model overtriages: {n_over} ({n_over/len(gap_values)*100:.1f}%)',
             transform=axes[0].transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

scatter_idx = np.random.choice(len(y_true), size=min(5000, len(y_true)), replace=False)
axes[1].scatter(gap_values[scatter_idx], oof_confidence[scatter_idx],
               alpha=0.15, s=8, c='#2196F3')
if n_flagged > 0:
    flagged_scatter = np.where(potential_ut_mask)[0]
    flagged_scatter = flagged_scatter[np.isin(flagged_scatter, scatter_idx)]
    if len(flagged_scatter) > 0:
        axes[1].scatter(gap_values[flagged_scatter], oof_confidence[flagged_scatter],
                       alpha=0.5, s=15, c='red', label=f'Flagged (n={n_flagged})', zorder=5)
axes[1].axvline(x=2, color='orange', linestyle='--', alpha=0.7)
axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
axes[1].set_xlabel('Severity Gap (Label - Predicted)', fontsize=11)
axes[1].set_ylabel('Model Confidence', fontsize=11)
axes[1].set_title('Confidence vs Severity Gap', fontweight='bold')
axes[1].legend(fontsize=9)

if n_flagged > 0:
    flag_by_label = pd.crosstab(flagged_df['triage_acuity'], flagged_df['predicted_acuity'])
    flag_by_label.plot(kind='bar', ax=axes[2], color=colors_esi[:len(flag_by_label.columns)],
                       edgecolor='black', linewidth=0.3)
    axes[2].set_xlabel('Assigned ESI Label', fontsize=11)
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].set_title('Flagged Cases: Label vs Predicted', fontweight='bold')
    axes[2].legend(title='Predicted ESI', fontsize=8)
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)
else:
    axes[2].text(0.5, 0.5, 'No cases flagged', ha='center', va='center', fontsize=14)
    axes[2].set_title('Flagged Cases', fontweight='bold')

plt.suptitle('Potential Undertriage Detection Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig12_undertriage_detection.png', bbox_inches='tight')
plt.show()


# ================================================================
# SECTION 12: CALIBRATION
# ================================================================

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
calibration_summary = []

for esi_class in range(5):
    esi_label = esi_class + 1
    y_binary = (y_true == esi_label).astype(int)
    y_prob = oof_blend[:, esi_class]
    ax = axes[esi_class]

    if y_binary.sum() > 100:
        prob_true, prob_pred = calibration_curve(y_binary, y_prob, n_bins=10, strategy='uniform')
        ax.plot(prob_pred, prob_true, marker='o', color=colors_esi[esi_class],
                linewidth=2, markersize=6, label='Model')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax.fill_between(prob_pred, prob_true, [p for p in prob_pred],
                        alpha=0.1, color=colors_esi[esi_class])
        cal_error = np.mean(np.abs(prob_true - prob_pred))
        calibration_summary.append({
            'ESI': esi_label,
            'Mean Cal Error': cal_error,
            'N positive': int(y_binary.sum()),
            'Mean predicted prob': float(y_prob.mean()),
        })
        ax.text(0.05, 0.85, f'Cal Error: {cal_error:.3f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Insufficient\nsamples', ha='center', va='center', fontsize=12)

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title(f'ESI {esi_label}', fontweight='bold', color=colors_esi[esi_class])
    ax.legend(fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')

plt.suptitle('Calibration Curves by ESI Level\n(Closer to diagonal = better calibrated)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig13_calibration.png', bbox_inches='tight')
plt.show()

if calibration_summary:
    cal_df = pd.DataFrame(calibration_summary)
    print("\n📋 Calibration Summary:")
    print(cal_df.to_string(index=False, float_format='%.4f'))
    print(f"\n  Average calibration error: {cal_df['Mean Cal Error'].mean():.4f}")

# Confidence by correctness
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
correct_mask = oof_preds == y_true
incorrect_mask = ~correct_mask

axes[0].hist(oof_confidence[correct_mask], bins=50, alpha=0.6, color='#4CAF50',
             label=f'Correct (n={correct_mask.sum():,})', density=True)
axes[0].hist(oof_confidence[incorrect_mask], bins=50, alpha=0.6, color='#F44336',
             label=f'Incorrect (n={incorrect_mask.sum():,})', density=True)
axes[0].set_xlabel('Model Confidence (max probability)')
axes[0].set_ylabel('Density')
axes[0].set_title('Confidence Distribution: Correct vs Incorrect', fontweight='bold')
axes[0].legend()

error_mag = np.abs(y_true - oof_preds)
error_labels = ['Exact', '±1 level', '±2 levels', '±3+ levels']
error_groups = [error_mag == 0, error_mag == 1, error_mag == 2, error_mag >= 3]
error_colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
bp_data = [oof_confidence[eg] for eg in error_groups if eg.sum() > 0]
bp_labels = [el for el, eg in zip(error_labels, error_groups) if eg.sum() > 0]

bp = axes[1].boxplot(bp_data, labels=bp_labels, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], error_colors[:len(bp_data)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_xlabel('Prediction Error Magnitude')
axes[1].set_ylabel('Model Confidence')
axes[1].set_title('Confidence by Error Magnitude', fontweight='bold')

plt.tight_layout()
plt.savefig('fig14_confidence_analysis.png', bbox_inches='tight')
plt.show()

print(f"\n📋 Confidence Analysis:")
print(f"  Mean confidence (correct):   {oof_confidence[correct_mask].mean():.3f}")
print(f"  Mean confidence (incorrect): {oof_confidence[incorrect_mask].mean():.3f}")


# ================================================================
# SECTION 13: ABLATION STUDY  (BUG FIX: deduplication of feature columns)
# ================================================================
print("="*60)
print("🔬 ABLATION STUDY — Value of Each Data Source")
print("="*60)

# Define feature groups (non-overlapping by design where possible)
vitals_cols = [c for c in X_train.columns if any(v in c.lower() for v in
               ['bp', 'heart', 'respiratory', 'temp', 'spo2', 'gcs', 'pain',
                'shock', 'map', 'news', 'pulse', 'bmi', 'febrile', 'hypothermic',
                'critical', 'hypotensive', 'hypertensive', 'bradycardia', 'tachycardia',
                'bradypnea', 'tachypnea', 'hr_rr', 'sbp_rr', 'modified_shock',
                'temp_deviation']) and 'missing' not in c.lower()]

missing_cols = [c for c in X_train.columns if 'missing' in c.lower()]

demo_cols = [c for c in X_train.columns if any(d in c.lower() for d in
             ['age', 'sex', 'language', 'insurance', 'pediatric', 'elderly',
              'arrival_mode', 'arrival_day', 'arrival_hour', 'arrival_season',
              'shift', 'transport', 'hour_sin', 'hour_cos', 'weekend'])]

text_cols = [c for c in X_train.columns if any(t in c.lower() for t in
             ['tfidf', 'complaint', 'kw', 'has_chest', 'has_sob', 'has_altered',
              'has_trauma', 'has_abdominal', 'has_neuro', 'has_cardiac', 'has_psych',
              'high_acuity_kw', 'low_acuity_kw', 'acuity_kw', 'chief_complaint'])]

hx_cols = [c for c in X_train.columns if any(h in c.lower() for h in
           ['hx_', 'comorbidity', 'high_risk_com', 'cv_risk', 'metabolic',
            'prior', 'admission', 'medication', 'visits_x'])]

# ------------------------------------------------------------------
# BUG FIX: Use dict.fromkeys() to deduplicate while preserving order.
# The original code combined lists that shared columns (e.g. hx_ cols
# matched multiple keyword patterns), causing LightGBM to error with
# "Feature appears more than one time."
# ------------------------------------------------------------------
ablation_configs = {
    'Vitals Only':
        list(dict.fromkeys(vitals_cols)),
    'Vitals + Missingness':
        list(dict.fromkeys(vitals_cols + missing_cols)),
    'Vitals + Miss + Demographics':
        list(dict.fromkeys(vitals_cols + missing_cols + demo_cols)),
    'Vitals + Miss + Demo + History':
        list(dict.fromkeys(vitals_cols + missing_cols + demo_cols + hx_cols)),
    'All Structured + Text (no SVD)':
        list(dict.fromkeys(vitals_cols + missing_cols + demo_cols + hx_cols + text_cols)),
    'Full Multimodal':
        list(dict.fromkeys(X_train.columns.tolist())),
}

ablation_results = []

for config_name, config_cols in ablation_configs.items():
    # Filter to only columns present in X_train, then deduplicate once more
    valid_cols = list(dict.fromkeys(c for c in config_cols if c in X_train.columns))
    if len(valid_cols) == 0:
        print(f"  {config_name}: No valid columns, skipping.")
        continue

    X_abl = X_train[valid_cols]

    quick_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof_abl = np.zeros(len(X_train))

    for tr_idx, vl_idx in quick_skf.split(X_abl, y_model):
        mdl = lgb.LGBMClassifier(**{**LGB_PARAMS, 'n_estimators': 1000, 'verbose': -1})
        mdl.fit(
            X_abl.iloc[tr_idx], y_model.iloc[tr_idx],
            eval_set=[(X_abl.iloc[vl_idx], y_model.iloc[vl_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        oof_abl[vl_idx] = mdl.predict(X_abl.iloc[vl_idx]) + 1
        del mdl

    abl_qwk = qwk(y_true, oof_abl.astype(int))
    abl_f1 = f1_score(y_true, oof_abl.astype(int), average='macro')

    ablation_results.append({
        'Configuration': config_name,
        'N Features': len(valid_cols),
        'QWK': abl_qwk,
        'Macro F1': abl_f1,
    })
    print(f"  {config_name:<40s}: QWK={abl_qwk:.4f}, F1={abl_f1:.4f} ({len(valid_cols)} features)")

abl_df = pd.DataFrame(ablation_results)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(abl_df['Configuration'], abl_df['QWK'], color='#2196F3', alpha=0.7,
               edgecolor='black', linewidth=0.3)
ax.set_xlabel('Quadratic Weighted Kappa (QWK)', fontsize=11)
ax.set_title('Ablation Study: Incremental Value of Each Data Source', fontweight='bold')

for i, (bar, row) in enumerate(zip(bars, abl_df.itertuples())):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{row.QWK:.4f}', va='center', fontsize=10, fontweight='bold')
    if i > 0:
        delta = row.QWK - abl_df.iloc[i-1]['QWK']
        ax.text(bar.get_width() + 0.045, bar.get_y() + bar.get_height()/2,
                f'(Δ={delta:+.4f})', va='center', fontsize=8,
                color='green' if delta > 0 else 'red')

plt.tight_layout()
plt.savefig('fig15_ablation.png', bbox_inches='tight')
plt.show()

gc.collect()


# ================================================================
# SECTION 14: GENERATE SUBMISSION
# ================================================================
print("📝 Generating test predictions...")

test_blend = opt_w[0]*test_lgb + opt_w[1]*test_xgb + opt_w[2]*test_cb

if use_thresh:
    test_labels = apply_thresholds(test_blend, opt_thresh)
else:
    test_labels = test_blend.argmax(axis=1) + 1

submission = pd.DataFrame({
    "patient_id": test["patient_id"],
    "triage_acuity": test_labels,
})

assert list(submission.columns) == list(sample_sub.columns), "Column mismatch!"
assert len(submission) == len(sample_sub), "Row count mismatch!"
assert submission["triage_acuity"].between(1, 5).all(), "Invalid acuity values!"

submission.to_csv("submission.csv", index=False)

print(f"\n✅ Submission saved: submission.csv")
print(f"  Rows: {len(submission)}")
print(f"  Distribution:")
for esi in sorted(submission['triage_acuity'].unique()):
    n = (submission['triage_acuity'] == esi).sum()
    print(f"    ESI {esi}: {n:>5,} ({n/len(submission)*100:.1f}%)")

total_time = time.time() - total_start
print(f"\n⏱️  Total runtime: {total_time/60:.1f} minutes")
```


```python

```
