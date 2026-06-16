# Triagegeist: ESI Prediction and Risk Analysis

## Clinical Motivation
Emergency department triage requires rapid assessment under high cognitive load. This project provides a decision support framework that predicts the Emergency Severity Index (ESI) while simultaneously identifying patients at risk of clinical deterioration. By measuring prediction uncertainty, the system flags cases where human re-evaluation is most critical.

## 1. Environment Setup


```python
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
```

## 2. Data Integration and Preprocessing
We integrate patient history and chief complaints with the core triage data. To ensure model integrity, we apply strict text cleaning to remove administrative jargon and potential label leakage.


```python
def clean_text_clinical(text):
    if not isinstance(text, str): return "unknown"
    text = text.lower()
    # Remove administrative keywords and digits to prevent leakage
    text = re.sub(r'esi|level|priority|p[1-5]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_data():
    train = pd.read_csv('/kaggle/input/competitions/triagegeist/train.csv')
    test = pd.read_csv('/kaggle/input/competitions/triagegeist/test.csv')
    cc = pd.read_csv('/kaggle/input/competitions/triagegeist/chief_complaints.csv')
    ph = pd.read_csv('/kaggle/input/competitions/triagegeist/patient_history.csv')
    
    y_acuity = train['triage_acuity']
    y_mort = (train['disposition'] == 'deceased').astype(int)
    
    cc['chief_complaint_raw'] = cc['chief_complaint_raw'].apply(clean_text_clinical)
    
    train_full = train.merge(ph, on='patient_id', how='left').merge(cc, on='patient_id', how='left')
    test_full = test.merge(ph, on='patient_id', how='left').merge(cc, on='patient_id', how='left')
    
    return train_full, test_full, y_acuity, y_mort

train_df, test_df, y_acuity, y_mort = load_data()
```

## 3. Feature Extraction (SVD-NLP)
We use dimensionality reduction (Truncated SVD) on TF-IDF vectors to capture both word-level semantics and character-level patterns. This approach maintains high performance while managing feature sparsity and handling typographical variations in clinical notes.


```python
all_text = pd.concat([train_df['chief_complaint_raw'], test_df['chief_complaint_raw']])

# Word-level features
tfidf_w = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
svd_w = TruncatedSVD(n_components=50, random_state=SEED)
svd_w.fit(tfidf_w.fit_transform(all_text))

# Char-level features (for typo robustness)
tfidf_c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000)
svd_c = TruncatedSVD(n_components=50, random_state=SEED)
svd_c.fit(tfidf_c.fit_transform(all_text))

def extract_features(df):
    cc_col = 'chief_complaint_system_x' if 'chief_complaint_system_x' in df.columns else 'chief_complaint_system'
    cat_f = ['arrival_mode', 'pain_location', 'mental_status_triage', 'sex', 'arrival_day', cc_col]
    leakers = ['disposition', 'ed_los_hours', 'discharge_disposition', 'admit_unit', 'news2_score']
    num_f = [c for c in df.columns if df[c].dtype in ['int64', 'float64'] 
             and c not in ['triage_acuity', 'patient_id'] + leakers]
    
    for c in cat_f: df[c] = df[c].fillna('None').astype('category')
    
    w_svd = svd_w.transform(tfidf_w.transform(df['chief_complaint_raw']))
    c_svd = svd_c.transform(tfidf_c.transform(df['chief_complaint_raw']))
    
    return pd.concat([
        df[cat_f + num_f].reset_index(drop=True),
        pd.DataFrame(w_svd, columns=[f'w_svd_{i}' for i in range(50)]),
        pd.DataFrame(c_svd, columns=[f'c_svd_{i}' for i in range(50)])
    ], axis=1)

X_train = extract_features(train_df)
X_test = extract_features(test_df)
```

## 4. Stochastic Cross-Validation
We employ a 5-fold ensemble to estimate prediction uncertainty. The variance in mortality risk across folds serves as a proxy for prediction frailty, identifying patients whose clinical profile is unstable or ambiguous.


```python
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
test_acuity_probs = np.zeros((len(test_df), 5))
test_mort_probs = []
oof_acuity = np.zeros(len(train_df))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_acuity)):
    # Acuity Model
    model_a = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, random_state=SEED+fold, verbose=-1)
    model_a.fit(X_train.iloc[tr_idx], y_acuity.iloc[tr_idx]-1, 
                eval_set=[(X_train.iloc[val_idx], y_acuity.iloc[val_idx]-1)], 
                callbacks=[lgb.early_stopping(50)])
    
    test_acuity_probs += model_a.predict_proba(X_test) / N_FOLDS
    oof_acuity[val_idx] = model_a.predict(X_train.iloc[val_idx]) + 1
    
    # Deterioration Model
    model_m = lgb.LGBMClassifier(n_estimators=300, scale_pos_weight=100, random_state=SEED+fold, verbose=-1)
    model_m.fit(X_train.iloc[tr_idx], y_mort.iloc[tr_idx])
    test_mort_probs.append(model_m.predict_proba(X_test)[:, 1])
    
    print(f"Fold {fold+1} metrics calculated.")
```

## 5. Results and Clinical Dashboard
The final output provides a multi-dimensional view of patient risk. We flag patients with high deterioration risk despite stable ESI scores and patients with high prediction frailty who require immediate human re-triage.


```python
print("--- Classification Report ---")
print(classification_report(y_acuity, oof_acuity, digits=4))

mort_matrix = np.array(test_mort_probs)
test_df['final_esi'] = np.argmax(test_acuity_probs, axis=1) + 1
test_df['deterioration_risk'] = np.mean(mort_matrix, axis=0)
test_df['prediction_frailty'] = np.std(mort_matrix, axis=0)

test_df['Action_Plan'] = 'Routine Care'
test_df.loc[(test_df['final_esi'] >= 3) & (test_df['deterioration_risk'] > 0.15), 'Action_Plan'] = 'High Deterioration Risk'
test_df.loc[(test_df['prediction_frailty'] > 0.08), 'Action_Plan'] = 'High Prediction Frailty'

test_df[['patient_id', 'final_esi']].rename(columns={'final_esi': 'triage_acuity'}).to_csv('submission.csv', index=False)
print("\n--- Clinical Dashboard Preview ---")
print(test_df[['patient_id', 'final_esi', 'deterioration_risk', 'prediction_frailty', 'Action_Plan']].sort_values('deterioration_risk', ascending=False).head(10))
```
