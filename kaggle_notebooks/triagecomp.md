```python
SEED = 42
N_SPLITS = 5
TARGET = "triage_acuity"

USE_TEXT = True
USE_MISSING_INDICATORS = True
USE_DERIVED_FEATURES = True

DEBUG = False
```


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
```


```python
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(SEED)
```


```python
train = pd.read_csv("/kaggle/input/competitions/triagegeist/train.csv")
test = pd.read_csv("/kaggle/input/competitions/triagegeist/test.csv")
complaints = pd.read_csv("/kaggle/input/competitions/triagegeist/chief_complaints.csv")

train = train.merge(complaints, on="patient_id", how="left")
test = test.merge(complaints, on="patient_id", how="left")

if DEBUG:
    train = train.sample(5000, random_state=SEED)
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

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': avg_importance
}).sort_values('importance', ascending=False)

print("Top 20 most predictive features:")
print(importance_df.head(20))

# Check top symptom words
symptom_importance = importance_df[importance_df['feature'].str.len() < 20].head(10)
print("\nTop 10 symptom words:")
print(symptom_importance)
```
