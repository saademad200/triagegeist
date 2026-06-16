```python
import gc
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

SEED = 42
np.random.seed(SEED)
ON_KAGGLE = os.path.exists('/kaggle/input')
DATA_DIR = Path('/kaggle/input/competitions/triagegeist') if ON_KAGGLE else Path('.')

print(f'Running on: {"Kaggle" if ON_KAGGLE else "Local"}')
```


```python
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
chief_complaints = pd.read_csv(DATA_DIR / 'chief_complaints.csv')

if 'chief_complaint_raw' not in train.columns:
    train = train.merge(chief_complaints, on='patient_id', how='left')
    test = test.merge(chief_complaints, on='patient_id', how='left')

for col in list(train.columns):
    if col.endswith('_x') and col.replace('_x', '_y') in train.columns:
        base = col[:-2]
        ycol = col.replace('_x', '_y')
        train[base] = train[col].fillna(train[ycol])
        if col in test.columns and ycol in test.columns:
            test[base] = test[col].fillna(test[ycol])
        train.drop([col, ycol], axis=1, inplace=True, errors='ignore')
        test.drop([col, ycol], axis=1, inplace=True, errors='ignore')

train['chief_complaint_raw'] = train['chief_complaint_raw'].fillna('not recorded')
test['chief_complaint_raw'] = test['chief_complaint_raw'].fillna('not recorded')

print('train:', train.shape)
print('test :', test.shape)
print('acuity distribution:')
print(train['triage_acuity'].value_counts().sort_index())
```


```python
vital_cols = [c for c in ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2'] if c in train.columns]
colors = ['#d32f2f', '#f57c00', '#fbc02d', '#4caf50', '#1976d2']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
target_dist = train['triage_acuity'].value_counts().sort_index()
axes[0].bar(target_dist.index, target_dist.values, color=colors[:len(target_dist)])
axes[0].set_title('Triage Acuity Distribution')
axes[0].set_xlabel('ESI')
axes[0].set_ylabel('Count')

if vital_cols:
    miss_by_acuity = train.groupby('triage_acuity')[vital_cols].apply(lambda x: x.isna().mean() * 100)
    miss_by_acuity.plot(kind='bar', ax=axes[1], colormap='RdYlGn_r')
    axes[1].set_title('Vital Missingness by Acuity')
    axes[1].set_xlabel('ESI')
    axes[1].set_ylabel('Missing %')

plt.tight_layout()
plt.show()
```


```python
def engineer_features(df):
    feat = df.copy()
    available = set(feat.columns)

    vital_check = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2']
    present_vitals = [c for c in vital_check if c in available]

    for col in present_vitals:
        feat[f'{col}_missing'] = feat[col].isna().astype(int)

    if present_vitals:
        feat['n_vitals_missing'] = feat[[f'{c}_missing' for c in present_vitals]].sum(axis=1)
        feat['all_vitals_present'] = (feat['n_vitals_missing'] == 0).astype(int)
        feat['no_vitals_taken'] = (feat['n_vitals_missing'] == len(present_vitals)).astype(int)
        miss_pattern = feat[[f'{c}_missing' for c in present_vitals]].astype(str).agg(''.join, axis=1)
        feat['miss_pattern_freq'] = miss_pattern.map(miss_pattern.value_counts(normalize=True).to_dict())

    if 'pain_score' in available:
        feat['pain_missing'] = (feat['pain_score'] == -1).astype(int)
        feat['pain_score_clean'] = feat['pain_score'].replace(-1, np.nan)
        feat['high_pain'] = (feat['pain_score_clean'] >= 8).astype(float)

    if 'spo2' in available:
        feat['spo2_critical'] = (feat['spo2'] < 92).astype(float)
        feat['spo2_low'] = (feat['spo2'] < 95).astype(float)
    if 'gcs_total' in available:
        feat['gcs_impaired'] = (feat['gcs_total'] < 15).astype(float)
        feat['gcs_severe'] = (feat['gcs_total'] <= 8).astype(float)
    if 'heart_rate' in available:
        feat['tachycardic'] = (feat['heart_rate'] > 100).astype(float)
        feat['bradycardic'] = (feat['heart_rate'] < 50).astype(float)
    if 'systolic_bp' in available:
        feat['hypotensive'] = (feat['systolic_bp'] < 90).astype(float)
        feat['hypertensive_crisis'] = (feat['systolic_bp'] > 180).astype(float)
    if 'respiratory_rate' in available:
        feat['tachypneic'] = (feat['respiratory_rate'] > 20).astype(float)
    if 'temperature_c' in available:
        feat['febrile'] = (feat['temperature_c'] > 38.0).astype(float)
        feat['hypothermic'] = (feat['temperature_c'] < 35.0).astype(float)
        feat['temp_deviation'] = (feat['temperature_c'] - 37.0).abs()

    sirs_cols = [c for c in ['tachycardic', 'tachypneic', 'febrile', 'hypothermic'] if c in feat.columns]
    if sirs_cols:
        feat['sirs_count'] = feat[sirs_cols].fillna(0).sum(axis=1)

    if 'shock_index' in available:
        feat['shock_index_critical'] = (feat['shock_index'] > 1.0).astype(float)
    if 'news2_score' in available:
        feat['news2_high'] = (feat['news2_score'] >= 7).astype(float)

    if 'arrival_hour' in available:
        feat['is_night'] = ((feat['arrival_hour'] >= 22) | (feat['arrival_hour'] <= 6)).astype(int)
        feat['hour_sin'] = np.sin(2 * np.pi * feat['arrival_hour'] / 24)
        feat['hour_cos'] = np.cos(2 * np.pi * feat['arrival_hour'] / 24)

    if 'arrival_day' in available:
        feat['is_weekend'] = feat['arrival_day'].isin(['Saturday', 'Sunday']).astype(int)

    if 'age' in available:
        feat['pediatric'] = (feat['age'] < 18).astype(int)
        feat['elderly'] = (feat['age'] >= 65).astype(int)

    if 'arrival_mode' in available:
        feat['ambulance_arrival'] = (feat['arrival_mode'] == 'ambulance').astype(int)
        feat['walkin'] = (feat['arrival_mode'] == 'walk-in').astype(int)

    if 'num_prior_ed_visits_12m' in available:
        feat['frequent_visitor'] = (feat['num_prior_ed_visits_12m'] >= 4).astype(int)

    return feat

train_feat = engineer_features(train)
test_feat = engineer_features(test)
print('engineered columns:', train_feat.shape[1])
```


```python
keyword_groups = {
    'kw_resuscitation': ['cardiac arrest', 'unresponsive', 'intubat', 'apneic', 'pulseless', 'code blue', 'cpr', 'no pulse'],
    'kw_emergent': ['chest pain', 'stroke', 'altered mental', 'seizure', 'overdose', 'suicidal', 'hemorrhag', 'severe bleed', 'dyspnea', 'syncope', 'anaphylax', 'sepsis', 'shortness of breath', 'sob', 'difficulty breathing', 'crushing', 'worst headache', 'gi bleed'],
    'kw_urgent': ['abdominal pain', 'vomiting', 'dizziness', 'laceration', 'fracture', 'dislocation', 'burn', 'allergic reaction', 'high fever', 'dehydrat', 'asthma', 'cellulitis'],
    'kw_low_acuity': ['prescription refill', 'med refill', 'suture removal', 'follow up', 'clearance', 'sore throat', 'congestion', 'rash', 'earache', 'insect bite', 'cold symptoms', 'medication refill', 'cough', 'sti check']
}

for group_name, keywords in keyword_groups.items():
    pattern = '|'.join(keywords)
    train_feat[group_name] = train_feat['chief_complaint_raw'].str.lower().str.contains(pattern, na=False, regex=True).astype(int)
    test_feat[group_name] = test_feat['chief_complaint_raw'].str.lower().str.contains(pattern, na=False, regex=True).astype(int)

for df in [train_feat, test_feat]:
    df['complaint_length'] = df['chief_complaint_raw'].str.len()
    df['complaint_word_count'] = df['chief_complaint_raw'].str.split().str.len()
    df['complaint_has_numbers'] = df['chief_complaint_raw'].str.contains(r'\d', na=False, regex=True).astype(int)

tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), sublinear_tf=True, min_df=5, strip_accents='unicode', token_pattern=r'(?u)\b\w+\b')
tfidf_train = tfidf.fit_transform(train_feat['chief_complaint_raw']).toarray()
tfidf_test = tfidf.transform(test_feat['chief_complaint_raw']).toarray()
tfidf_cols = [f'tfidf_{i}' for i in range(tfidf_train.shape[1])]

USE_DEBERTA = False
if USE_DEBERTA:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
            model = AutoModel.from_pretrained('microsoft/deberta-v3-small').to(device).eval()

            def extract_embeddings(texts, batch_size=64):
                all_embs = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
                    with torch.no_grad():
                        out = model(**enc)
                        mask = enc['attention_mask'].unsqueeze(-1).float()
                        embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    all_embs.append(embs.cpu().numpy())
                return np.vstack(all_embs)

            train_embs = extract_embeddings(train_feat['chief_complaint_raw'].tolist())
            test_embs = extract_embeddings(test_feat['chief_complaint_raw'].tolist())
            emb_cols = [f'emb_{i}' for i in range(train_embs.shape[1])]
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        else:
            USE_DEBERTA = False
    except Exception:
        USE_DEBERTA = False

print('TF-IDF features:', tfidf_train.shape[1])
print('Use DeBERTa:', USE_DEBERTA)
```


```python
LEAKAGE_COLS = {'disposition', 'ed_los_hours'}
ID_COLS = {'patient_id'}
TARGET_COL = 'triage_acuity'
TEXT_COLS = {'chief_complaint_raw'}

cat_cols = []
for col in train_feat.columns:
    if col in LEAKAGE_COLS | ID_COLS | TEXT_COLS | {TARGET_COL}:
        continue
    if train_feat[col].dtype == 'object':
        cat_cols.append(col)

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_feat[col].astype(str), test_feat[col].astype(str)])
    le.fit(combined)
    train_feat[col + '_enc'] = le.transform(train_feat[col].astype(str))
    test_feat[col + '_enc'] = le.transform(test_feat[col].astype(str))

exclude_cols = LEAKAGE_COLS | ID_COLS | TEXT_COLS | {TARGET_COL} | set(cat_cols)
tab_feature_cols = [c for c in train_feat.columns if c not in exclude_cols and train_feat[c].dtype in ['int64', 'float64', 'int32', 'float32', 'int8', 'uint8', 'float16']]

X_tab_train = train_feat[tab_feature_cols].values.astype(np.float32)
X_tab_test = test_feat[tab_feature_cols].values.astype(np.float32)

if USE_DEBERTA:
    X_train = np.hstack([X_tab_train, tfidf_train, train_embs]).astype(np.float32)
    X_test = np.hstack([X_tab_test, tfidf_test, test_embs]).astype(np.float32)
    all_feature_names = tab_feature_cols + tfidf_cols + emb_cols
else:
    X_train = np.hstack([X_tab_train, tfidf_train]).astype(np.float32)
    X_test = np.hstack([X_tab_test, tfidf_test]).astype(np.float32)
    all_feature_names = tab_feature_cols + tfidf_cols

y_train = train_feat[TARGET_COL].values
X_train_clean = np.nan_to_num(X_train, nan=-999)
X_test_clean = np.nan_to_num(X_test, nan=-999)

def qwk_score(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

print('tabular features:', len(tab_feature_cols))
print('X_train shape:', X_train_clean.shape)
```


```python
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

lgb_params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': SEED
}

oof_lgb = np.zeros((len(X_train_clean), 5))
test_lgb = np.zeros((len(X_test_clean), 5))
lgb_models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_clean, y_train), 1):
    X_tr, X_va = X_train_clean[tr_idx], X_train_clean[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr - 1, feature_name=all_feature_names)
    dvalid = lgb.Dataset(X_va, label=y_va - 1, feature_name=all_feature_names)

    model = lgb.train(
        lgb_params, dtrain, num_boost_round=3000, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
    )

    oof_lgb[va_idx] = model.predict(X_va)
    test_lgb += model.predict(X_test_clean) / N_FOLDS
    lgb_models.append(model)
    print(f'LGB fold {fold} QWK:', qwk_score(y_va, oof_lgb[va_idx].argmax(axis=1) + 1))

lgb_qwk = qwk_score(y_train, oof_lgb.argmax(axis=1) + 1)
print('LGB OOF QWK:', lgb_qwk)

oof_cat = np.zeros((len(X_train_clean), 5))
test_cat = np.zeros((len(X_test_clean), 5))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_clean, y_train), 1):
    X_tr, X_va = X_train_clean[tr_idx], X_train_clean[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    cat_model = CatBoostClassifier(
        iterations=3000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
        loss_function='MultiClass', eval_metric='WKappa', early_stopping_rounds=100,
        verbose=500, random_seed=SEED + fold, task_type='CPU'
    )
    cat_model.fit(X_tr, y_tr - 1, eval_set=(X_va, y_va - 1))

    oof_cat[va_idx] = cat_model.predict_proba(X_va)
    test_cat += cat_model.predict_proba(X_test_clean) / N_FOLDS
    print(f'CAT fold {fold} QWK:', qwk_score(y_va, oof_cat[va_idx].argmax(axis=1) + 1))

cat_qwk = qwk_score(y_train, oof_cat.argmax(axis=1) + 1)
print('CAT OOF QWK:', cat_qwk)
```


```python
def optimize_weights(oof_list, y_true):
    def neg_qwk(w):
        w = np.abs(w) / np.abs(w).sum()
        blended = sum(wi * pi for wi, pi in zip(w, oof_list))
        return -qwk_score(y_true, blended.argmax(axis=1) + 1)
    result = minimize(neg_qwk, np.ones(len(oof_list)) / len(oof_list), method='Nelder-Mead', options={'maxiter': 1000})
    w = np.abs(result.x) / np.abs(result.x).sum()
    return w, -result.fun

def optimize_thresholds(probs, y_true):
    expected = (probs * np.arange(1, 6)).sum(axis=1)
    def neg_qwk(t):
        preds = np.clip(np.digitize(expected, sorted(t)) + 1, 1, 5)
        return -qwk_score(y_true, preds)
    result = minimize(neg_qwk, [1.5, 2.5, 3.5, 4.5], method='Nelder-Mead', options={'maxiter': 5000})
    return sorted(result.x), -result.fun

weights, ens_qwk_raw = optimize_weights([oof_lgb, oof_cat], y_train)
oof_blend = weights[0] * oof_lgb + weights[1] * oof_cat
test_blend = weights[0] * test_lgb + weights[1] * test_cat

thresholds, ens_qwk_opt = optimize_thresholds(oof_blend, y_train)

oof_expected = (oof_blend * np.arange(1, 6)).sum(axis=1)
test_expected = (test_blend * np.arange(1, 6)).sum(axis=1)

oof_predictions = np.clip(np.digitize(oof_expected, thresholds) + 1, 1, 5)
test_predictions = np.clip(np.digitize(test_expected, thresholds) + 1, 1, 5)

print('weights:', weights)
print('QWK argmax:', ens_qwk_raw)
print('QWK optimized:', ens_qwk_opt)
print('accuracy:', accuracy_score(y_train, oof_predictions))
```


```python
cm = confusion_matrix(y_train, oof_predictions, labels=[1, 2, 3, 4, 5])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[f'ESI-{i}' for i in range(1, 6)],
            yticklabels=[f'ESI-{i}' for i in range(1, 6)])
axes[0].set_title('Confusion Matrix (Counts)')

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
            xticklabels=[f'ESI-{i}' for i in range(1, 6)],
            yticklabels=[f'ESI-{i}' for i in range(1, 6)])
axes[1].set_title('Confusion Matrix (Normalized)')

plt.tight_layout()
plt.show()

print(classification_report(y_train, oof_predictions, target_names=[f'ESI-{i}' for i in range(1, 6)], digits=3))
```


```python
import shap

SHAP_N = min(2000, len(X_train_clean))
shap_idx = np.random.choice(len(X_train_clean), SHAP_N, replace=False)
X_shap = X_train_clean[shap_idx]

explainer = shap.TreeExplainer(lgb_models[0])
shap_values = explainer.shap_values(X_shap)

# Handle both possible return shapes:
if isinstance(shap_values, list):
    # list of (N, F) per class
    mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)  # (N, F)
elif shap_values.ndim == 3:
    if shap_values.shape[0] == SHAP_N:
        # Shape (N, F, C) — newer SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=2)  # (N, F)
    else:
        # Shape (C, N, F)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)  # (N, F)
else:
    mean_abs_shap = np.abs(shap_values)

# Now mean_abs_shap is guaranteed (N, F)
importance_per_feature = mean_abs_shap.mean(axis=0)  # (F,) — 1D

feat_imp = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importance_per_feature
}).sort_values('importance', ascending=False)

top = feat_imp.head(min(25, len(feat_imp)))
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(range(len(top)), top['importance'].values[::-1], color='steelblue', alpha=0.8)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top['feature'].values[::-1], fontsize=9)
ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
ax.set_title('Top 25 Features by Clinical Importance (SHAP)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```


```python
def subgroup_metrics(y_true, y_pred, mask, name):
    if mask.sum() < 20:
        return None
    yt, yp = y_true[mask], y_pred[mask]
    return {
        'Group': name,
        'N': int(mask.sum()),
        'QWK': round(qwk_score(yt, yp), 4),
        'Accuracy': round(accuracy_score(yt, yp), 4),
        'Undertriage%': round((yp > yt).mean() * 100, 1),
        'Overtriage%': round((yp < yt).mean() * 100, 1)
    }

results = []

if 'sex' in train_feat.columns:
    for val in train_feat['sex'].unique():
        r = subgroup_metrics(y_train, oof_predictions, (train_feat['sex'] == val).values, f'Sex: {val}')
        if r:
            results.append(r)

if 'age' in train_feat.columns:
    age_groups = {
        'Pediatric (0-17)': train_feat['age'] < 18,
        'Young Adult (18-39)': (train_feat['age'] >= 18) & (train_feat['age'] < 40),
        'Middle-aged (40-64)': (train_feat['age'] >= 40) & (train_feat['age'] < 65),
        'Geriatric (65+)': train_feat['age'] >= 65
    }
    for name, cond in age_groups.items():
        r = subgroup_metrics(y_train, oof_predictions, cond.values, f'Age: {name}')
        if r:
            results.append(r)

equity_df = pd.DataFrame(results)
print(equity_df.to_string(index=False))
```


```python
submission = pd.DataFrame({
    'patient_id': test_feat['patient_id'],
    'triage_acuity': test_predictions.astype(int)
})
submission.to_csv('submission.csv', index=False)

print('submission saved:', submission.shape)
print('Final QWK:', round(ens_qwk_opt, 4))
```
