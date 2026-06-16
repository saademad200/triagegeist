# Emergency Department Triage Prediction

---

## Quick Navigation

| Section | Description | 
|---------|-------------|
| [Clinical Context](#clinical-context) | Understanding emergency triage |
| [Data Overview](#data-overview) | Dataset structure & distributions |
| [Configuration](#configuration) | Model settings & hyperparameters |
| [Feature Engineering](#feature-engineering) | Clinical features & text processing |
| [Model Training](#model-training) | LightGBM with cross-validation |
| [Results & Analysis](#results) | Performance metrics & insights |
| [Predictions](#predictions) | Generate submission file |



---


<a id='clinical-context'></a>

# Clinical Context: Understanding Emergency Triage
 
## The Emergency Severity Index (ESI)
 
# The ESI is a 5-level triage system widely used in emergency departments:
 
| Level | Category | Expected Resources | Typical Presentation |
|-------|----------|-------------------|---------------------|
| 1 | Resuscitation | Immediate life-saving interventions | Cardiac arrest, severe respiratory distress, unresponsive |
| 2 | Emergent | High-risk situation or multiple resources | Chest pain with ECG changes, severe trauma, stroke symptoms |
| 3 | Urgent | Multiple resources needed | Moderate asthma exacerbation, abdominal pain, complex laceration |
| 4 | Less Urgent | One resource needed | Simple laceration, minor burn, urinary symptoms |
| 5 | Non-Urgent | No resources needed | Medication refill, minor rash, chronic back pain |
 
## Why Triage Accuracy Matters
 
# **Under-triage** (assigning too low acuity) can:
 - Delay time-sensitive interventions for stroke, MI, and sepsis
 - Increase morbidity and mortality
 - Lead to patient deterioration while waiting
 
# **Over-triage** (assigning too high acuity) can:
 - Consume scarce resources unnecessarily
 - Increase wait times for truly sick patients
 - Contribute to ED crowding and staff burnout
 
## Current Challenges
 
# Studies show 20-30% disagreement between nurses on moderate acuity cases due to:
 - Different training and experience levels
 - Implicit biases (age, race, language)
 - Inconsistent vital sign interpretation
 - Cognitive load during busy shifts
 
# This model aims to provide **decision support**, not replacement—offering a second opinion to help standardize triage decisions.

<a id='configuration'></a>
# Configuration & Setup

## Model Hyperparameters

These settings control the model's behavior. Adjust them to experiment:

| Parameter | Value | Purpose |
|-----------|-------|----------|
| `SEED` | 42 | Reproducibility |
| `N_SPLITS` | 5 | Cross-validation folds |
| `TARGET` | triage_acuity | What we're predicting |
| `USE_TEXT` | True | Include chief complaint text features |
| `USE_MISSING_INDICATORS` | True | Create features for missing values |
| `USE_DERIVED_FEATURES` | True | Add clinical calculations (shock index, MAP, etc.) |
| `DEBUG` | False | Use subset of data for testing |

---



```python
SEED = 42
N_SPLITS = 5
TARGET = "triage_acuity"

USE_TEXT = True
USE_MISSING_INDICATORS = True
USE_DERIVED_FEATURES = True

DEBUG = False
```

## Import Libraries



```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
```


```python
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(SEED)
```

<a id='data-overview'></a>
# Data Loading & Overview

## Dataset Structure

We're working with three CSV files:
- **train.csv**: 80,000 patients with triage labels
- **test.csv**: 20,000 patients (need to predict)
- **chief_complaints.csv**: Raw text complaints for all patients

---



```python
train = pd.read_csv("/kaggle/input/competitions/triagegeist/train.csv")
test = pd.read_csv("/kaggle/input/competitions/triagegeist/test.csv")
complaints = pd.read_csv("/kaggle/input/competitions/triagegeist/chief_complaints.csv")

train = train.merge(complaints, on="patient_id", how="left")
test = test.merge(complaints, on="patient_id", how="left")
test_ids = test[["patient_id"]].copy()

if DEBUG:
    train = train.sample(5000, random_state=SEED)
```


```python
print("*"*60)
print("TRAINING DATA OVERVIEW")
print("*"*60)
print(f"Training shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Complaints shape: {complaints.shape}")


print("\nTraining columns:")
print(train.columns.tolist())

print("\nData types:")
print(train.dtypes.value_counts())

print("\nMissing values in training:")
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)


# target distribution
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

target_counts = train[TARGET].value_counts().sort_index()
colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#5f27cd']
bars = plt.bar(target_counts.index, target_counts.values, color=colors)
plt.xlabel('Triage Acuity Level (1=Most Severe, 5=Least Severe)')
plt.ylabel('Count')
plt.title('Distribution of Triage Acuity Levels')
for bar, count in zip(bars, target_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
             f'{count}\n({count/len(train)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=10)

plt.subplot(1, 2, 2)

plt.pie(target_counts.values, labels=[f'Level {i}' for i in target_counts.index], 
        autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Triage Acuity Distribution (Proportion)')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nClass distribution summary:")

print(f"Most common: Level {target_counts.idxmax()} ({target_counts.max()} patients)")
print(f"Least common: Level {target_counts.idxmin()} ({target_counts.min()} patients)")
print(f"Imbalance ratio: {target_counts.max()/target_counts.min():.1f}:1")
```

<a id='feature-engineering'></a>
# Feature Engineering Pipeline

This section creates features from raw data. the key innovations are:

1. **Clinical Features** (Shock Index, MAP, SOFA components)
2. **Missing Indicators** (Flag when vitals are not measured)
3. **Text Features** (TF-IDF from chief complaints)
4. **Derived Flags** (Hypotension, tachycardia, hypoxia, etc.)



---



```python
print("\n" + "*"*60)
print("DEMOGRAPHIC ANALYSIS")
print("*"*60)

# age distribution
plt.figure(figsize=(16, 12))

# age histogram
plt.subplot(3, 3, 1)

train['age'].hist(bins=30, edgecolor='black', color='skyblue', figsize=(16, 12))
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Age Distribution', fontsize=14, fontweight='bold')
plt.axvline(train['age'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {train["age"].mean():.1f}')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# age by triage level boxplot
plt.subplot(3, 3, 2)

# prepare data for boxplot
box_data = [train[train[TARGET] == i]['age'].dropna() for i in range(1, 6)]
bp = plt.boxplot(box_data, patch_artist=True, labels=['1', '2', '3', '4', '5'])
colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#5f27cd']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel('Triage Level', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.title('Age Distribution by Triage Level', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# sex distribution pie chart
plt.subplot(3, 3, 3)

if 'sex' in train.columns:
    sex_counts = train['sex'].value_counts()
    wedges, texts, autotexts = plt.pie(sex_counts.values, labels=sex_counts.index, 
                                        autopct='%1.1f%%', colors=['#3498db', '#e74c3c', '#2ecc71'],
                                        startangle=90, textprops={'fontsize': 12})
    plt.title('Sex Distribution', fontsize=14, fontweight='bold')

# arrival mode horizontal bar chart
plt.subplot(3, 3, 4)

if 'arrival_mode' in train.columns:
    arrival_counts = train['arrival_mode'].value_counts().head(8)
    y_pos = range(len(arrival_counts))
    bars = plt.barh(y_pos, arrival_counts.values, color='#3498db')
    plt.yticks(y_pos, arrival_counts.index, fontsize=11)
    plt.xlabel('Count', fontsize=12)
    plt.title('Top Arrival Modes', fontsize=14, fontweight='bold')
    # adding value labels
    for i, (bar, val) in enumerate(zip(bars, arrival_counts.values)):
        plt.text(val + 100, bar.get_y() + bar.get_height()/2, f'{val}', 
                 va='center', fontsize=10)

# Insurance type vertical bar chart
plt.subplot(3, 3, 5)

if 'insurance_type' in train.columns:
    insurance_counts = train['insurance_type'].value_counts().head(6)
    x_pos = range(len(insurance_counts))
    bars = plt.bar(x_pos, insurance_counts.values, color='#e74c3c')
    plt.xticks(x_pos, insurance_counts.index, rotation=45, ha='right', fontsize=10)
    plt.ylabel('Count', fontsize=12)
    plt.title('Insurance Types', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, insurance_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                 f'{val}', ha='center', fontsize=10)

# language horizontal bar chart
plt.subplot(3, 3, 6)

if 'language' in train.columns:
    lang_counts = train['language'].value_counts().head(6)
    y_pos = range(len(lang_counts))
    bars = plt.barh(y_pos, lang_counts.values, color='#2ecc71')
    plt.yticks(y_pos, lang_counts.index, fontsize=11)
    plt.xlabel('Count', fontsize=12)
    plt.title('Top Languages', fontsize=14, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, lang_counts.values)):
        plt.text(val + 50, bar.get_y() + bar.get_height()/2, f'{val}', 
                 va='center', fontsize=10)

# additional demographic: arrival da
plt.subplot(3, 3, 7)

if 'arrival_day' in train.columns:
    day_counts = train['arrival_day'].value_counts().sort_index()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.plot(days[:len(day_counts)], day_counts.values, marker='o', linewidth=2, 
             markersize=8, color='#9b59b6')
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Arrivals by Day', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

# additional demographic: Arrival hour
plt.subplot(3, 3, 8)

if 'arrival_hour' in train.columns:
    hour_counts = train['arrival_hour'].value_counts().sort_index()
    plt.bar(hour_counts.index, hour_counts.values, color='#f39c12', width=0.8)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Arrivals by Hour', fontsize=14, fontweight='bold')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3, axis='y')

# additional demographic: Shift
plt.subplot(3, 3, 9)

if 'shift' in train.columns:
    shift_counts = train['shift'].value_counts()
    plt.pie(shift_counts.values, labels=shift_counts.index, autopct='%1.1f%%',
            colors=['#1abc9c', '#e67e22', '#9b59b6'], startangle=90)
    plt.title('Arrivals by Shift', fontsize=14, fontweight='bold')

plt.suptitle('Demographic Analysis of Emergency Department Patients', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('demographic_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
```


```python
print("\n" + "*"*60)
print("VITAL SIGNS ANALYSIS")
print("*"*60)

vital_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 
              'temperature_c', 'spo2', 'pain_score', 'gcs_total']

# summary statistics
print("\nVital Signs Summary Statistics:")
print(train[vital_cols].describe().round(1))

# outlier checks
print("\nPotential outliers (values beyond clinical normal ranges):")
clinical_ranges = {
    'systolic_bp': (70, 220),
    'diastolic_bp': (40, 130),
    'heart_rate': (30, 200),
    'respiratory_rate': (6, 40),
    'temperature_c': (32, 42),
    'spo2': (50, 100),
    'pain_score': (0, 10),
    'gcs_total': (3, 15)
}

for col, (low, high) in clinical_ranges.items():
    if col in train.columns:
        outliers = train[(train[col] < low) | (train[col] > high)][col].count()
        if outliers > 0:
            print(f"  {col}: {outliers} outliers ({outliers/len(train)*100:.2f}%)")

print("\nMissing values in vital signs:")
vital_missing = train[vital_cols].isnull().sum()
vital_missing = vital_missing[vital_missing > 0].sort_values(ascending=False)
print(vital_missing)

# visualizing key vitals by triage level
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(vital_cols):
    if col in train.columns:
        ax = axes[i]
        # Boxplot by triage level
        train.boxplot(column=col, by=TARGET, ax=ax)
        ax.set_title(f'{col} by Triage Level')
        ax.set_xlabel('Triage Level')
        ax.set_ylabel(col)
        ax.set_xticklabels([1,2,3,4,5])

# hHide unused subplot
if len(vital_cols) < 8:
    axes[-1].set_visible(False)

plt.suptitle('Vital Signs Distribution by Triage Level', y=1.02)
plt.tight_layout()
plt.savefig('vitals_by_triage.png', dpi=150, bbox_inches='tight')
plt.show()
```


```python
print("\n" + "*"*60)
print("CORRELATION ANALYSIS")
print("*"*60)

# select numerical columns
numerical_cols = train.select_dtypes(include=[np.number]).columns
numerical_cols = [col for col in numerical_cols if col != TARGET]

#calculate correlations with target
target_corr = []

for col in numerical_cols:
    if col in train.columns and train[col].nunique() > 1:
        corr = train[[col, TARGET]].corr().iloc[0,1]
        target_corr.append({'feature': col, 'correlation': abs(corr)})

corr_df = pd.DataFrame(target_corr).sort_values('correlation', ascending=False)

print("\nTop 10 features correlated with triage acuity:")
print(corr_df.head(10).to_string(index=False))

# correlation heatmap (top features)
plt.figure(figsize=(12, 10))
top_features = corr_df.head(15)['feature'].tolist()
top_features.append(TARGET)
corr_matrix = train[top_features].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)

plt.title('Correlation Heatmap - Top 15 Features with Target')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```


```python
print("\n" + "*"*60)
print("TEXT DATA ANALYSIS")
print("*"*60)

# basic text stats
train_text = train['chief_complaint_raw'].fillna("")
test_text = test['chief_complaint_raw'].fillna("")

print(f"Missing chief complaints in train: {train['chief_complaint_raw'].isnull().sum()}")
print(f"Missing chief complaints in test: {test['chief_complaint_raw'].isnull().sum()}")

# text length statistics
train_text_len = train_text.str.len()
test_text_len = test_text.str.len()

print(f"\nTrain text length - Mean: {train_text_len.mean():.1f}, Std: {train_text_len.std():.1f}")
print(f"Test text length - Mean: {test_text_len.mean():.1f}, Std: {test_text_len.std():.1f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)

plt.hist(train_text_len, bins=50, alpha=0.7, label='Train', color='blue')
plt.hist(test_text_len, bins=50, alpha=0.7, label='Test', color='orange')
plt.xlabel('Text Length (characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Chief Complaint Length')
plt.legend()

plt.subplot(1, 2, 2)

train_text_len[train_text_len < 200].hist(bins=50, color='blue', alpha=0.7)
plt.xlabel('Text Length (characters) - Zoomed')
plt.ylabel('Frequency')
plt.title('Text Length Distribution (capped at 200 chars)')

plt.tight_layout()
plt.savefig('text_length_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# most common words
def get_most_common_words(text_series, n=20):
    words = []
    for text in text_series.dropna():
        # Simple tokenization
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        words.extend(tokens)
    return Counter(words).most_common(n)

print("\nMost common words in chief complaints:")
common_words = get_most_common_words(train['chief_complaint_raw'], 30)
for word, count in common_words[:20]:
    print(f"  {word}: {count}")

# Check for severity words
severity_words = ['mild', 'moderate', 'severe', 'acute', 'minor', 'critical']
print("\nChecking for severity words (potential leakage):")
for word in severity_words:
    count = train_text.str.contains(word, case=False, na=False).sum()
    if count > 0:
        print(f"  '{word}' appears in {count} complaints ({count/len(train)*100:.2f}%)")
```


```python
LEAKAGE_COLS = [
    "ed_los_hours",
    "disposition",
    "news2_score"
]

for col in LEAKAGE_COLS:
    if col in train.columns:
        train.drop(columns=col, inplace=True)
    if col in test.columns:
        test.drop(columns=col, inplace=True)

DROP_ID_COLS = [
    "patient_id",
    "triage_nurse_id"
]

for col in DROP_ID_COLS:
    if col in train.columns:
        train.drop(columns=col, inplace=True)
    if col in test.columns:
        test.drop(columns=col, inplace=True)
```


```python
def add_missing_indicators(df):
    
    vital_cols = [
        "systolic_bp",
        "diastolic_bp",
        "mean_arterial_pressure",
        "heart_rate",
        "respiratory_rate",
        "temperature_c",
        "spo2",
        "pain_score",
        "shock_index"
    ]
    
    for col in vital_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    
    df["pain_score"] = df["pain_score"].replace(-1, np.nan)
    
    return df

if USE_MISSING_INDICATORS:
    train = add_missing_indicators(train)
    test = add_missing_indicators(test)
```


```python
def add_derived_features(df):
    
    # Physiologic flags
    df["hypotension_flag"] = (df["systolic_bp"] < 90).astype(int)
    df["tachycardia_flag"] = (df["heart_rate"] > 100).astype(int)
    df["hypoxia_flag"] = (df["spo2"] < 92).astype(int)
    df["fever_flag"] = (df["temperature_c"] > 38).astype(int)
    
    # GCS alertness
    df["gcs_low_flag"] = (df["gcs_total"] < 15).astype(int)
    
    # Age risk buckets
    df["pediatric_flag"] = (df["age"] < 18).astype(int)
    df["elderly_flag"] = (df["age"] >= 75).astype(int)
    
    # Pulse pressure
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]

    # Fill NaN in derived features (flags should be 0 if vital is missing)
    flag_cols = [col for col in df.columns if col.endswith('_flag')]
    for col in flag_cols:
        df[col] = df[col].fillna(0).astype(int)
    
    return df

if USE_DERIVED_FEATURES:
    train = add_derived_features(train)
    test = add_derived_features(test)
```


```python
categorical_cols = [
    "site_id",
    "arrival_mode",
    "arrival_day",
    "arrival_season",
    "shift",
    "sex",
    "chief_complaint_system_y",
    "age_group",
    "language",
    "insurance_type",
    "transport_origin",
    "pain_location",
    "mental_status_triage",
    "chief_complaint_system_x"
]

# Remove duplicates if they exist
categorical_cols = list(dict.fromkeys(categorical_cols))

# Fill NaN in categorical columns with 'MISSING'
for col in categorical_cols:
    if col in train.columns:
        train[col] = train[col].fillna('MISSING').astype('category')
    if col in test.columns:
        test[col] = test[col].fillna('MISSING').astype('category')
```


```python
# Prepare target (FIX: convert from 1-5 to 0-4)
y = train[TARGET] - 1  

# Check class distribution
print("Class distribution (0-4 scale):")
print(y.value_counts().sort_index())
```


```python
features = [
    col for col in train.columns 
    if col != TARGET and col != "chief_complaint_raw"
]

train_text = train["chief_complaint_raw"].fillna("")
test_text = test["chief_complaint_raw"].fillna("")
```

<a id='model-training'></a>
# Model Training

## Algorithm: LightGBM Classifier

I use LightGBM because:
- Handles sparse matrices efficiently (2000 text features)
- Built in support for multiclass classification
- Fast training with good performance
- Robust to imbalanced classes

### Model Parameters
```python
LGBMClassifier(
    n_estimators=100,      # Number of boosting rounds
    max_depth=6,           # Tree depth (prevents overfitting)
    learning_rate=0.05,    # Slow learning = better generalization
    num_leaves=31,         # Leaf nodes per tree
    class_weight='balanced', # Handle class imbalance
    random_state=SEED
)
```

### Training Strategy
- **5-Fold Stratified Cross-Validation** (ensures all classes in each fold)
- **Custom class weights** to handle imbalanced data
- **Out-of-fold predictions** for unbiased performance estimates

---

## Training Process

*The next cell will take ~3-4 minutes to run. It trains 5 models (one per fold) and generates predictions.*



```python
if USE_TEXT:
    # Create safe symptom vocabulary
    safe_vocab = [
    'head', 'skull', 'scalp', 'face', 'forehead', 'temple', 'jaw', 'chin',
    'neck', 'throat', 'voice', 'vocal', 'cords', 'larynx', 'pharynx',
    'eye', 'eyes', 'eyeball', 'eyelid', 'vision', 'blurred vision', 'double vision',
    'ear', 'ears', 'hearing', 'tinnitus', 'nose', 'nasal', 'sinus', 'sinuses',
    'mouth', 'lip', 'lips', 'tongue', 'gum', 'gums', 'tooth', 'teeth', 'dental',
    'chest', 'rib', 'ribs', 'sternum', 'breast', 'back', 'spine', 'spinal',
    'vertebra', 'vertebrae', 'disc', 'shoulder', 'shoulders',
    'abdomen', 'abdominal', 'belly', 'stomach', 'gut', 'intestine', 'bowel',
    'colon', 'rectum', 'anal', 'anus', 'pelvis', 'pelvic', 'groin', 'hip', 'hips',
    'bladder', 'kidney', 'kidneys', 'renal', 'liver', 'gallbladder', 'pancreas',
    'spleen', 'appendix', 'hernia',
    'arm', 'arms', 'forearm', 'elbow', 'wrist', 'hand', 'hands', 'finger', 'fingers',
    'thumb', 'nail', 'nails', 'knuckle', 'palm',
    'leg', 'legs', 'thigh', 'knee', 'knees', 'calf', 'shin', 'ankle', 'foot',
    'feet', 'toe', 'toes', 'heel', 'sole',
    'skin', 'rash', 'lesion', 'wound', 'cut', 'laceration', 'abrasion', 'bruise',
    'contusion', 'burn', 'blister', 'ulcer', 'spot', 'bump', 'lump',
    'mass', 'swelling', 'redness', 'inflammation',
    'pain', 'ache', 'aching', 'sore', 'tenderness', 'discomfort', 'cramp', 'cramping',
    'throbbing', 'pulsating', 'stabbing', 'sharp', 'dull', 'burning', 'gnawing',
    'pressure', 'tightness', 'heaviness',
    'cough', 'coughing', 'wheeze', 'wheezing', 'shortness', 'breath', 'breathing',
    'respiratory', 'congestion', 'phlegm', 'mucus', 'sputum', 'hemoptysis',
    'blood-tinged', 'aspiration', 'choke', 'choking', 'sneeze', 'sneezing',
    'snore', 'snoring', 'apnea',
    'palpitation', 'palpitations', 'racing', 'fluttering', 'skipping', 'irregular',
    'heartbeat', 'pulse', 'bp', 'blood pressure', 'circulation', 'cold extremities',
    'nausea', 'vomit', 'vomiting', 'emesis', 'regurgitation', 'diarrhea', 'loose stools',
    'constipation', 'bloody stools', 'melena', 'hematochezia', 'indigestion',
    'heartburn', 'reflux', 'gas', 'bloating', 'distension', 'swallow', 'swallowing',
    'dysphagia', 'appetite', 'anorexia',
    'dizzy', 'dizziness', 'vertigo', 'faint', 'fainting', 'syncope', 'lightheaded',
    'headache', 'migraine', 'seizure', 'convulsion', 'fit', 'tremor', 'shaking',
    'twitch', 'twitching', 'spasm', 'numb', 'numbness', 'tingle', 'tingling',
    'pins and needles', 'paralysis', 'weakness', 'gait', 'walking', 'balance',
    'coordination', 'speech', 'slurred', 'aphasia', 'confusion', 'disorientation',
    'memory', 'forgetfulness', 'aura',
    'swell', 'stiff', 'stiffness', 'joint pain', 'arthritis', 'gout',
    'muscle pain', 'myalgia', 'strain', 'sprain', 'fracture', 'broken', 'crack',
    'snap', 'pop', 'click', 'locking', 'giving way', 'instability',
    'urinate', 'urination', 'urinary', 'dysuria', 'frequency', 'urgency',
    'hesitancy', 'retention', 'incontinence', 'blood in urine', 'hematuria',
    'discharge', 'penile', 'testicular', 'scrotal', 'vaginal', 'menstrual',
    'period', 'bleeding', 'spotting', 'pregnancy', 'pregnant',
    'fever', 'chill', 'chills', 'rigor', 'sweat', 'sweating', 'night sweats',
    'fatigue', 'tired', 'lethargy', 'malaise', 'weak', 'weakness', 'energy',
    'appetite', 'weight loss', 'weight gain', 'dehydration', 'thirst',
    'vision change', 'blurred', 'double', 'blind spot', 'floater', 'flash',
    'photophobia', 'hearing loss', 'muffled', 'ringing', 'tinnitus', 'roaring',
    'smell', 'taste', 'metallic taste',
    'alert', 'responsive', 'oriented', 'anxious', 'agitated', 'restless',
    'withdrawn', 'depressed', 'sad', 'crying', 'hallucination', 'paranoid',
    'heart attack', 'mi', 'myocardial', 'infarction', 'angina', 'cad',
    'chf', 'heart failure', 'afib', 'arrhythmia', 'svt', 'vtach', 'bradycardia',
    'tachycardia', 'htn', 'hypertension', 'hypotension', 'pvd', 'dvt', 'pe',
    'claudication', 'aneurysm',
    'copd', 'emphysema', 'bronchitis', 'pneumonia', 'asthma', 'bronchospasm',
    'pleurisy', 'pleural', 'effusion', 'pneumothorax', 'hemothorax', 'croup',
    'epiglottitis', 'covid', 'influenza', 'flu', 'rsv',
    'cva', 'stroke', 'tia', 'brain attack', 'hemorrhage', 'ich', 'sah',
    'tbi', 'head injury', 'concussion', 'ms', 'multiple sclerosis', 'parkinson',
    'alzheimer', 'dementia', 'meningitis', 'encephalitis', 'neuropathy',
    'gerd', 'reflux', 'ulcer', 'gastric', 'peptic', 'gastritis', 'gastroenteritis',
    'appendicitis', 'diverticulitis', 'cholecystitis', 'pancreatitis', 'hepatitis',
    'cirrhosis', 'ibs', 'colitis', 'crohn', 'ulcerative',
    'diabetes', 'dm', 'dka', 'hyperglycemia', 'hypoglycemia', 'thyroid',
    'hyperthyroid', 'hypothyroid', 'addison', 'cushing',
    'fracture', 'dislocation', 'subluxation', 'sprain', 'strain', 'tear',
    'rupture', 'arthritis', 'oa', 'ra', 'gout', 'fibromyalgia', 'sciatica',
    'stenosis', 'herniated disc', 'slipped disc',
    'infection', 'sepsis', 'cellulitis', 'abscess', 'uti', 'pyelo', 'pid',
    'std', 'sti', 'hiv', 'aids', 'shingles', 'herpes', 'zoster', 'fungal',
    'candida', 'tinea',
    'fall', 'fell', 'trip', 'slip', 'collapse', 'accident', 'mva', 'motor vehicle',
    'car accident', 'bike accident', 'pedestrian', 'struck', 'hit', 'assault',
    'fight', 'stab', 'gunshot', 'gsw', 'overdose', 'poison', 'ingestion',
    'burn', 'scald', 'electrical', 'lightning', 'drowning', 'near-drowning',
    'suffocation', 'strangulation', 'hanging',
    'surgery', 'post-op', 'postoperative', 'procedure', 'line', 'tube', 'catheter',
    'port', 'picc', 'trach', 'vent', 'ventilator', 'oxygen', 'cpap', 'bipap',
    'dialysis', 'pacemaker', 'icd',
    'new', 'recurrent', 'chronic', 'acute onset', 'gradual', 'sudden',
    'intermittent', 'constant', 'waxing', 'waning', 'worse', 'better',
    'improving', 'worsening', 'progressive', 'stable', 'unchanged',
    'morning', 'night', 'nocturnal', 'positional', 'exertional', 'rest',
    'activity', 'movement', 'touch', 'pressure',
    'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
    'hour', 'hours', 'minute', 'minutes', 'second', 'seconds', 'morning',
    'afternoon', 'evening', 'night', 'today', 'yesterday', 'last night',
]

    safe_vocab = list(dict.fromkeys(safe_vocab))
    
    # Create TF-IDF with vocabulary restriction
    tfidf = TfidfVectorizer(
        vocabulary=safe_vocab,  # Only use these words!
        ngram_range=(1,2),
        min_df=3,
        max_df=0.9,
        stop_words="english",
        sublinear_tf=True
    )
    
    X_text = tfidf.fit_transform(train_text)
    X_test_text = tfidf.transform(test_text)
    print(f"Text features shape: {X_text.shape}")
else:
    # Create empty sparse matrices with no features
    X_text = csr_matrix((train.shape[0], 0))
    X_test_text = csr_matrix((test.shape[0], 0))
    print("Text features disabled")
```


```python
# Encode categorical features properly (avoid one-hot explosion)
def encode_categorical_features(df, categorical_cols, encoder_dict=None):
    """Encode categorical features using label encoding"""
    df_encoded = pd.DataFrame(index=df.index)
    
    if encoder_dict is None:
        encoder_dict = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                encoder_dict[col] = le
    else:
        for col in categorical_cols:
            if col in df.columns and col in encoder_dict:
                # Handle unseen categories
                le = encoder_dict[col]
                df[col] = df[col].astype(str)
                # Map unseen to -1
                df_encoded[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    return df_encoded, encoder_dict

# Fit encoders on train
X_struct_encoded, encoders = encode_categorical_features(train[features], categorical_cols)

# Transform test
X_test_struct_encoded, _ = encode_categorical_features(test[features], categorical_cols, encoders)

# Add numerical features
numerical_cols = [col for col in features if col not in categorical_cols and col in train.columns]
for col in numerical_cols:
    if col in train.columns:
        X_struct_encoded[col] = train[col].fillna(train[col].median())
    if col in test.columns:
        X_test_struct_encoded[col] = test[col].fillna(train[col].median())  # Use train median for test

# Convert to float32
X_struct_encoded = X_struct_encoded.astype('float32')
X_test_struct_encoded = X_test_struct_encoded.astype('float32')

# Convert to sparse
X_struct_sparse = csr_matrix(X_struct_encoded.values)
X_test_struct_sparse = csr_matrix(X_test_struct_encoded.values)

# Combine with text features
X_final = hstack([X_struct_sparse, X_text])
X_test_final = hstack([X_test_struct_sparse, X_test_text])

print(f"Final feature matrix shape: {X_final.shape}")
```


```python
def train_lgb(
    X_final,
    y,
    X_test_final,
    n_splits=5,
    random_state=SEED
):
    
    # Train LightGBM on combined structured + TF-IDF sparse features.
    n_classes = 5
    oof_preds = np.zeros((X_final.shape[0], n_classes))
    test_preds = np.zeros((X_test_final.shape[0], n_classes))
    
    # Store feature importance
    feature_importance = []

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    # FIX: Correct class weights for 0-4 classes
    # Heavier weights for more severe classes (3,4,5 on original scale -> 2,3,4 on 0-4 scale)
    class_weights = {0:1.0, 1:1.0, 2:2.0, 3:3.0, 4:4.0}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")

        X_train, X_val = X_final[train_idx], X_final[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            boosting_type="gbdt",
            learning_rate=0.03,  # Slightly lower
            n_estimators=2000,
            class_weight=class_weights,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,  # Reduce verbosity
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(50),  # Reduced from 100
                lgb.log_evaluation(100)
            ]
        )

        # OOF predictions
        oof_preds[val_idx] = model.predict_proba(X_val)

        # Test predictions (averaged across folds)
        test_preds += model.predict_proba(X_test_final) / n_splits
        
        # Store feature importance
        feature_importance.append(model.feature_importances_)

    # ============================
    # Evaluation (convert back to 1-5 scale)
    # ============================
    y_true = y + 1  # Convert back to 1-5
    oof_labels = np.argmax(oof_preds, axis=1) + 1  # Convert back to 1-5

    macro_f1 = f1_score(y_true, oof_labels, average="macro")
    weighted_f1 = f1_score(y_true, oof_labels, average="weighted")
    accuracy = np.mean(oof_labels == y_true)

    print("\n Overall CV Evaluation ")
    print(f"Macro F1     : {macro_f1:.6f}")
    print(f"Weighted F1  : {weighted_f1:.6f}")
    print(f"Accuracy     : {accuracy:.6f}")

    cm = confusion_matrix(y_true, oof_labels)
    print("\n Confusion Matrix (rows=true, cols=predicted):")
    print(cm)

    # Under/Over triage analysis (ordinal relevance)
    under_triage = np.sum(oof_labels < y_true)
    over_triage = np.sum(oof_labels > y_true)
    total = len(y_true)

    print("\n Ordinal Error Analysis ")
    print(f"Under-triage cases : {under_triage} ({under_triage/total:.3%})")
    print(f"Over-triage cases  : {over_triage} ({over_triage/total:.3%})")
    
    # Per class metrics
    print("\n Per Class Performance ")
    print(classification_report(y_true, oof_labels, digits=4))

    return oof_preds, test_preds, feature_importance
```

<a id='results'></a>
# Results & Analysis

This section contains:
1. **Overall Performance Metrics** - Macro F1, accuracy, weighted F1
2. **Confusion Matrix** - Where does the model make mistakes?
3. **Per-Class Performance** - Precision, recall, F1 for each triage level
4. **Feature Importance** - Which features matter most?
5. **Error Analysis** - Which patient subgroups are hardest to predict?
6. **Confidence Calibration** - How certain is the model?

---

## Performance Metrics & Visualizations



```python
oof_preds, test_preds, feature_importance = train_lgb(
    X_final,
    y,
    X_test_final,
    n_splits=5,
    random_state=SEED
)
```


```python
feature_names = list(X_struct_encoded.columns) + list(tfidf.get_feature_names_out())
avg_importance = np.mean(feature_importance, axis=0)

#create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': avg_importance
}).sort_values('importance', ascending=False)

# separate clinical vs text features
importance_df['feature_type'] = 'text'
importance_df.loc[importance_df['feature'].isin(X_struct_encoded.columns), 'feature_type'] = 'clinical'

# ploting top 20 features
plt.figure(figsize=(12, 8))

top_features = importance_df.head(20)
colors = ['#1f77b4' if t == 'clinical' else '#ff7f0e' for t in top_features['feature_type']]
plt.barh(range(len(top_features)), top_features['importance'].values, color=colors)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance Score')
plt.title('Top 20 Most Important Features')
plt.gca().invert_yaxis()
plt.legend(['Clinical Features', 'Text Features'], loc='lower right')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# Print top features by category
print("\n" + "="*60)
print("TOP 10 CLINICAL FEATURES")
print("="*60)
print(importance_df[importance_df['feature_type'] == 'clinical'].head(10).to_string(index=False))


print("\n" + "="*60)
print("TOP 10 TEXT FEATURES (Symptoms)")
print("="*60)
text_features = importance_df[importance_df['feature_type'] == 'text'].head(10)
for idx, row in text_features.iterrows():
    print(f"  {row['feature']}: {row['importance']:.1f}")
```


```python
y_true = y + 1  # Convert back to 1-5
oof_labels = np.argmax(oof_preds, axis=1) + 1

# plotting calibration curves for each class
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(5):
    # One-vs-rest calibration
    y_true_binary = (y_true == i+1).astype(int)
    y_pred_binary = oof_preds[:, i]
    
    fraction_pos, mean_pred = calibration_curve(y_true_binary, y_pred_binary, n_bins=10)
    
    ax = axes[i]
    ax.plot(mean_pred, fraction_pos, marker='o', linewidth=2, label=f'Class {i+1}')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Curve - Class {i+1} (ESI Level {i+1})')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Hide the 6th subplot
axes[5].set_visible(False)
plt.suptitle('Model Calibration by Triage Level', fontsize=14)
plt.tight_layout()
plt.savefig('calibration_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("Calibration analysis complete - saved to calibration_curves.png")
```


```python
results_df = pd.DataFrame({
    'true': y_true,
    'pred': oof_labels,
    'age': train['age'].values,
    'sex': train['sex'].values,
    'arrival_mode': train['arrival_mode'].values,
    'mental_status': train['mental_status_triage'].values if 'mental_status_triage' in train.columns else 'Unknown'
})

# calculate errors
results_df['correct'] = results_df['true'] == results_df['pred']
results_df['error'] = ~results_df['correct']
results_df['error_magnitude'] = abs(results_df['true'] - results_df['pred'])
results_df['under_triage'] = results_df['pred'] < results_df['true']
results_df['over_triage'] = results_df['pred'] > results_df['true']

print("="*60)
print("ERROR ANALYSIS BY PATIENT SUBGROUP")
print("="*60)

# error by sex
print("\n1. Error Rate by Sex:")
sex_errors = results_df.groupby('sex')['error'].agg(['mean', 'count'])
sex_errors['mean'] = sex_errors['mean'] * 100
print(sex_errors.round(2))

# error by age group
results_df['age_group'] = pd.cut(results_df['age'], 
                                  bins=[0, 18, 40, 65, 100], 
                                  labels=['Pediatric (0-17)', 'Adult (18-39)', 'Middle-aged (40-64)', 'Elderly (65+)'])

print("\n2. Error Rate by Age Group:")
age_errors = results_df.groupby('age_group', observed=True)['error'].agg(['mean', 'count'])
age_errors['mean'] = age_errors['mean'] * 100
print(age_errors.round(2))

# error by arrival mode
print("\n3. Error Rate by Arrival Mode:")
arrival_errors = results_df.groupby('arrival_mode')['error'].agg(['mean', 'count'])
arrival_errors['mean'] = arrival_errors['mean'] * 100
print(arrival_errors.sort_values('mean', ascending=False).head(10).round(2))

# most common error patterns
print("\n4. Most Common Error Patterns:")
error_patterns = results_df[results_df['error']].groupby(['true', 'pred']).size().reset_index(name='count')
error_patterns = error_patterns.sort_values('count', ascending=False)
for _, row in error_patterns.head(5).iterrows():
    print(f"  True={int(row['true'])}, Pred={int(row['pred'])}: {row['count']} cases ({row['count']/len(results_df)*100:.2f}%)")

# confidence analysis
results_df['max_prob'] = np.max(oof_preds, axis=1)
low_conf = results_df[results_df['max_prob'] < 0.7]

print(f"\n5. Low Confidence Predictions (<70% probability):")
print(f"   Count: {len(low_conf)} ({len(low_conf)/len(results_df)*100:.1f}% of all predictions)")
if len(low_conf) > 0:
    low_conf_acc = low_conf['correct'].mean() * 100
    print(f"   Accuracy on low confidence: {low_conf_acc:.1f}%")
```


```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error by age group
ax = axes[0, 0]
age_errors = results_df.groupby('age_group', observed=True)['error'].mean() * 100
colors = ['#2ecc71' if e < 6.2 else '#f39c12' for e in age_errors.values]
bars = ax.bar(age_errors.index, age_errors.values, color=colors)
ax.set_ylabel('Error Rate (%)')
ax.set_title('Error Rate by Age Group')
ax.set_ylim(0, 8)
for bar, val in zip(bars, age_errors.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{val:.2f}%', ha='center', va='bottom')

# Error by arrival mode
ax = axes[0, 1]
arrival_errors = results_df.groupby('arrival_mode')['error'].mean() * 100
arrival_errors = arrival_errors.sort_values()
bars = ax.barh(arrival_errors.index, arrival_errors.values, color='#3498db')
ax.set_xlabel('Error Rate (%)')
ax.set_title('Error Rate by Arrival Mode')
for bar, val in zip(bars, arrival_errors.values):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
            f'{val:.2f}%', va='center')

# Error magnitude heatmap
ax = axes[1, 0]
error_matrix = pd.crosstab(results_df['true'], results_df['pred'], 
                           normalize='index') * 100
sns.heatmap(error_matrix, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
            xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
ax.set_xlabel('Predicted Level')
ax.set_ylabel('True Level')
ax.set_title('Error Patterns (% of each true class)')

# Confidence vs accuracy
ax = axes[1, 1]
# Bin by confidence
results_df['conf_bin'] = pd.cut(results_df['max_prob'], 
                                 bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                 labels=['<50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'])
conf_accuracy = results_df.groupby('conf_bin', observed=True)['correct'].mean() * 100
conf_counts = results_df.groupby('conf_bin', observed=True).size()

bars = ax.bar(range(len(conf_accuracy)), conf_accuracy.values, color='#27ae60')
ax.set_xticks(range(len(conf_accuracy)))
ax.set_xticklabels(conf_accuracy.index, rotation=45)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy by Confidence Level')
ax.set_ylim(0, 100)

# Add count labels
for i, (bar, count) in enumerate(zip(bars, conf_counts.values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'n={count}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('error_analysis_detailed.png', dpi=150, bbox_inches='tight')
plt.show()
```


```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(results_df['max_prob'], bins=50, edgecolor='black', color='#3498db')
plt.axvline(0.7, color='red', linestyle='--', label='Low confidence threshold (70%)')
plt.xlabel('Maximum Prediction Probability')
plt.ylabel('Count')
plt.title('Distribution of Model Confidence')
plt.legend()

plt.subplot(1, 2, 2)
# Confidence by true class
box_data = [results_df[results_df['true'] == i]['max_prob'].values for i in range(1, 6)]
bp = plt.boxplot(box_data, labels=[1,2,3,4,5], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#5f27cd']):
    patch.set_facecolor(color)
plt.xlabel('True Triage Level')
plt.ylabel('Confidence Score')
plt.title('Model Confidence by True Class')
plt.axhline(0.7, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('confidence_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nConfidence Summary:")
print(f"Mean confidence: {results_df['max_prob'].mean():.3f}")
print(f"Median confidence: {results_df['max_prob'].median():.3f}")
print(f"% low confidence (<70%): {(results_df['max_prob'] < 0.7).mean()*100:.1f}%")
print(f"% high confidence (>90%): {(results_df['max_prob'] > 0.9).mean()*100:.1f}%")
```

<a id='predictions'></a>
# Generate Final Predictions

This section:
1. Trains on the **full training set** (all 80,000 patients)
2. Generates predictions for the **test set** (20,000 patients)
3. Creates **submission.csv** ready for upload

---



```python
# Create submission from test_preds generated during cross-validation
print("="*60)
print("CREATING SUBMISSION FILE")
print("="*60)

# test_preds is the average of predictions from all 5 folds
# Shape: (n_test_samples, 5) - probabilities for each class

# Get the predicted class (argmax of probabilities)
final_predictions = np.argmax(test_preds, axis=1) + 1  # +1 to convert from 0-4 to 1-5

# Create submission dataframe
submission = pd.DataFrame({
    'patient_id': test_ids['patient_id'].values,
    'triage_acuity': final_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

print("\nPrediction Distribution:")
pred_dist = submission['triage_acuity'].value_counts().sort_index()
print(pred_dist)
print(f"\nTotal predictions: {len(submission)}")

# Show confidence distribution
print("\nAverage Confidence per Class:")
for i in range(1, 6):
    mask = final_predictions == i
    if mask.sum() > 0:
        # Get the probability for the predicted class
        class_probs = test_preds[mask, i-1]
        avg_conf = class_probs.mean()
        print(f"  Level {i}: {avg_conf:.1%} (n={mask.sum():,})")

print("\n" + "="*60)
print("SUBMISSION READY FOR DOWNLOAD!")
print("="*60)

# Display first few rows
print("\nFirst 10 predictions:")
print(submission.head(10))
```
