import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""
# TriageGeist Holistic Suite: Multi-Signal Synergy Stacker

**Clinical Relevance**
Emergency triage relies on rapid human judgment under extreme cognitive load. Undertriage delays care and causes preventable harm. Many AI models achieve perfect Kaggle scores by memorizing dataset artifacts (e.g., synthetic text templates) rather than learning true physiological deterioration. We present a highly interpretable, dual-track decision support system that rejects data leakage, evaluates on an Asymmetric Clinical Cost matrix to penalize undertriage, and estimates selective risk via Conformal Prediction.

### Data Citation & Disclosure
* **Citation**: Olaf Yunus Laitinen Imanov (2026). Triagegeist. https://kaggle.com/competitions/triagegeist, 2026. Kaggle.
* **Compliance**: We confirm that our use of the Triagegeist dataset complies with its terms of access. No external datasets were utilized for this submission.
    """))
    
    # 2. Imports
    cells.append(nbf.v4.new_code_cell("""
import os
import gc
import re
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import hstack
from scipy.optimize import minimize
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings('ignore')
print("Setup Complete")
    """))
    
    # 3. Data Loading
    cells.append(nbf.v4.new_markdown_cell("""
## Data Loading & The "Clean Room" Protocol
To ensure rigorous **technical quality**, we explicitly exclude `disposition`, `ed_los_hours`, and `chief_complaint_raw` from our physiological engine. This prevents outcome leakage and ensures the model learns true medical deterioration, rather than reverse-engineering the synthetic data generation rules.
    """))
    
    cells.append(nbf.v4.new_code_cell("""
KAGGLE_PATH = '/kaggle/input/competitions/triagegeist/'
LOCAL_PATH = '../data/' 
FALLBACK_PATH = 'data/' 

if os.path.exists(KAGGLE_PATH + 'train.csv'):
    PATH = KAGGLE_PATH
elif os.path.exists(LOCAL_PATH + 'train.csv'):
    PATH = LOCAL_PATH
else:
    PATH = FALLBACK_PATH
    
train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')
ph = pd.read_csv(PATH + 'patient_history.csv')
cc = pd.read_csv(PATH + 'chief_complaints.csv')

train = train.merge(ph, on='patient_id', how='left').merge(cc, on='patient_id', how='left')
test = test.merge(ph, on='patient_id', how='left').merge(cc, on='patient_id', how='left')
print(f"Train Shape: {train.shape} | Test Shape: {test.shape}")
    """))
    
    # 4. Feature Engineering
    cells.append(nbf.v4.new_markdown_cell("""
## Semantic Grouping & MNAR Missingness Strategy
To prevent validation leakage from synonymous complaint templates drifting across folds, we normalize the text and use K-Means clustering on TF-IDF character n-grams to generate rigorous semantic clusters. We use these clusters as fold boundaries to simulate a strict hidden test split.

For physiology, Missing vitals in the ED are *Missing Not At Random* (MNAR). We extract explicit missingness indicators before letting the tree algorithms handle NaNs natively.
    """))
    
    cells.append(nbf.v4.new_code_cell("""
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

print("Normalizing text and building Semantic Clusters...")
train['norm_text'] = train['chief_complaint_raw'].apply(normalize_text)
nlp_train = train['norm_text'].values

test['norm_text'] = test['chief_complaint_raw'].apply(normalize_text)
nlp_test = test['norm_text'].values

cluster_tfidf = TfidfVectorizer(max_features=2000, analyzer='char_wb', ngram_range=(2,4))
text_vecs = cluster_tfidf.fit_transform(nlp_train)

kmeans = MiniBatchKMeans(n_clusters=1500, random_state=42, batch_size=1000)
groups = kmeans.fit_predict(text_vecs)

def process_data(df, is_train=True):
    d = df.copy()
    y = d['triage_acuity'].values - 1 if is_train else None
        
    vital_cols = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 'pain_score']
    for col in vital_cols:
        if col in d.columns:
            d[f'is_missing_{col}'] = d[col].isnull().astype(int)
            
    d['historical_admission_rate'] = d['num_prior_admissions_12m'] / d['num_prior_ed_visits_12m'].clip(lower=1)
    if 'shock_index' not in d.columns:
        d['shock_index'] = d['heart_rate'] / d['systolic_bp'].clip(lower=1)
    d['age_adjusted_shock_index'] = d['shock_index'] * d['age']
    
    cat_cols = d.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        d[col] = d[col].fillna('Missing').astype(str)
        
    return d, y, cat_cols

X_train, y_train, cat_cols = process_data(train, is_train=True)
X_test, _, _ = process_data(test, is_train=False)

# Drop Leakage for Physiological Track
drop_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_raw', 'norm_text', 'chief_complaint_system', 'site_id', 'triage_nurse_id']
X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
cat_cols = [c for c in cat_cols if c not in drop_cols]
    """))
    
    # 5. Modeling
    cells.append(nbf.v4.new_markdown_cell("""
## Rigorous Validation & Multi-Signal Synergy Stacker
We employ a rigorous `StratifiedGroupKFold` (grouping by our semantic clusters) to eliminate target leakage across folds. We use **native categorical handling** for our tree algorithms (LightGBM, XGBoost, CatBoost) to maximize physiological predictive power without throwing away information via ordinal encoding. We use deep iterations with early stopping on the validation folds.

For NLP, we use a union of Word and Character TF-IDF matrices to ensure extreme robustness against typos and punctuation drift.

Finally, we use nested cross-validation (`cross_val_predict` grouped by semantic clusters) on a Logistic Regression Meta-Learner to ensure our level-two stacker is evaluated completely free of meta-leakage.
    """))
    
    cells.append(nbf.v4.new_code_cell("""
def custom_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_phys = np.zeros((len(X_train), 5))
test_phys = np.zeros((len(X_test), 5))

oof_nlp = np.zeros((len(X_train), 5))
test_nlp = np.zeros((len(X_test), 5))

lgb_shap_model = None
X_shap_sample = None

print("Training Dual-Track Holistic Suite...")
for fold, (train_idx, valid_idx) in enumerate(sgkf.split(X_train, y_train, groups=groups)):
    X_tr_phys, X_vl_phys = X_train.iloc[train_idx].copy(), X_train.iloc[valid_idx].copy()
    X_tr_text, X_vl_text = nlp_train[train_idx], nlp_train[valid_idx]
    X_ts_phys, X_ts_text = X_test.copy(), nlp_test
    y_tr, y_vl = y_train[train_idx], y_train[valid_idx]
    
    # --- Native Categorical Handling ---
    cat_idx = [X_tr_phys.columns.get_loc(c) for c in cat_cols]
    
    X_tr_cat = X_tr_phys.copy()
    X_vl_cat = X_vl_phys.copy()
    X_ts_cat = X_ts_phys.copy()
    for col in cat_cols:
        X_tr_cat[col] = X_tr_cat[col].astype("category")
        X_vl_cat[col] = pd.Categorical(X_vl_cat[col], categories=X_tr_cat[col].cat.categories)
        X_ts_cat[col] = pd.Categorical(X_ts_cat[col], categories=X_tr_cat[col].cat.categories)
    
    # === TRACK 1: PHYSIOLOGICAL ENSEMBLE ===
    lgb_model = lgb.LGBMClassifier(objective='multiclass', num_class=5, n_estimators=1500, learning_rate=0.03, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_tr_cat, y_tr, categorical_feature=cat_cols, eval_set=[(X_vl_cat, y_vl)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    xgb_model = XGBClassifier(objective='multi:softprob', num_class=5, n_estimators=1500, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss', enable_categorical=True, early_stopping_rounds=50)
    xgb_model.fit(X_tr_cat, y_tr, eval_set=[(X_vl_cat, y_vl)], verbose=False)
    
    cat_model = CatBoostClassifier(loss_function='MultiClass', iterations=1500, learning_rate=0.03, depth=6, random_seed=42, verbose=0, thread_count=-1, early_stopping_rounds=50)
    cat_model.fit(X_tr_phys, y_tr, cat_features=cat_idx, eval_set=(X_vl_phys, y_vl))
    
    oof_phys[valid_idx] = (lgb_model.predict_proba(X_vl_cat) + xgb_model.predict_proba(X_vl_cat) + cat_model.predict_proba(X_vl_phys)) / 3.0
    test_phys += (lgb_model.predict_proba(X_ts_cat) + xgb_model.predict_proba(X_ts_cat) + cat_model.predict_proba(X_ts_phys)) / 15.0
    
    if fold == 0:
        lgb_shap_model = lgb_model
        X_shap_sample = X_vl_cat.head(500)
        
    # === TRACK 2: CALIBRATED NLP MODEL (WORD+CHAR UNION) ===
    word_tfidf = TfidfVectorizer(max_features=20000, analyzer='word', ngram_range=(1,2), stop_words='english', sublinear_tf=True)
    char_tfidf = TfidfVectorizer(max_features=40000, analyzer='char_wb', ngram_range=(3,5), sublinear_tf=True)
    
    X_tr_word = word_tfidf.fit_transform(X_tr_text)
    X_vl_word = word_tfidf.transform(X_vl_text)
    X_ts_word = word_tfidf.transform(X_ts_text)
    
    X_tr_char = char_tfidf.fit_transform(X_tr_text)
    X_vl_char = char_tfidf.transform(X_vl_text)
    X_ts_char = char_tfidf.transform(X_ts_text)
    
    X_tr_tfidf = hstack([X_tr_word, X_tr_char])
    X_vl_tfidf = hstack([X_vl_word, X_vl_char])
    X_ts_tfidf = hstack([X_ts_word, X_ts_char])
    
    ridge = RidgeClassifier(alpha=1.0, class_weight='balanced')
    calibrated_clf = CalibratedClassifierCV(ridge, cv=3, method='sigmoid')
    calibrated_clf.fit(X_tr_tfidf, y_tr)
    
    oof_nlp[valid_idx] = calibrated_clf.predict_proba(X_vl_tfidf)
    test_nlp += calibrated_clf.predict_proba(X_ts_tfidf) / 5.0

print(f"\\nPhysiological Engine (Native Cat) QWK: {custom_qwk(y_train, np.argmax(oof_phys, axis=1)):.4f}")
print(f"Calibrated NLP Engine (Word+Char) QWK: {custom_qwk(y_train, np.argmax(oof_nlp, axis=1)):.4f}")

# === HONEST NESTED SYNERGY META-LEARNER ===
stacker_X_train = np.hstack([oof_phys, oof_nlp])
stacker_X_test = np.hstack([test_phys, test_nlp])

meta_learner = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
meta_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Use nested cross_val_predict to get the true, unbiased OOF QWK
final_oof_probs = cross_val_predict(meta_learner, stacker_X_train, y_train, cv=meta_cv, groups=groups, method='predict_proba')

# Fit on full stacker matrix to predict test set
meta_learner.fit(stacker_X_train, y_train)
final_test_probs = meta_learner.predict_proba(stacker_X_test)

print(f"Honest Grouped Nested Meta-Learner Synergy QWK: {custom_qwk(y_train, np.argmax(final_oof_probs, axis=1)):.4f}")
    """))

    # 6. Bias Tuning
    cells.append(nbf.v4.new_markdown_cell("""
## QWK Class Bias Optimization
Because Quadratic Weighted Kappa is highly sensitive to ordinal boundaries, relying on a naive `argmax()` over probabilities leaves significant performance on the table. We use scipy's Powell optimizer to dynamically learn the mathematically optimal class probability biases to maximize QWK. We do this inside a nested outer loop to ensure the reported tuned score is truly out-of-fold and un-optimistic.
    """))
    
    cells.append(nbf.v4.new_code_cell("""
def qwk_score_bias(y_true, probs, bias):
    logp = np.log(np.clip(probs, 1e-12, 1.0)) + bias[None, :]
    pred = np.argmax(logp, axis=1)
    return cohen_kappa_score(y_true, pred, weights="quadratic")

def objective(bias, y_true, probs):
    return -qwk_score_bias(y_true, probs, bias)

print("Starting Nested Class Bias Tuning for Ordinal QWK...")
nested_tuned_preds = np.zeros_like(y_train)

# Nested bias tuning loop
for train_idx, valid_idx in meta_cv.split(final_oof_probs, y_train, groups=groups):
    x0 = np.zeros(5)
    res = minimize(objective, x0, args=(y_train[train_idx], final_oof_probs[train_idx]), method="Powell")
    fold_bias = res.x
    logp_val = np.log(np.clip(final_oof_probs[valid_idx], 1e-12, 1.0)) + fold_bias[None, :]
    nested_tuned_preds[valid_idx] = np.argmax(logp_val, axis=1)

nested_tuned_qwk = cohen_kappa_score(y_train, nested_tuned_preds, weights="quadratic")
print(f"🚀 Honest Nested Tuned Stacker QWK: {nested_tuned_qwk:.4f}")

# Final bias tuned on the entire train set to apply to the test set
final_res = minimize(objective, np.zeros(5), args=(y_train, final_oof_probs), method="Powell")
best_bias_global = final_res.x
print(f"Final Global Bias Array for Test Set: {best_bias_global}")
    """))

    
    # 7. SHAP
    cells.append(nbf.v4.new_markdown_cell("""
## Interpretability & Fairness Mitigation
We dynamically implement an explicit probability shift during inference on the final, nested probabilities to computationally mitigate elderly undertriage.
    """))
    
    cells.append(nbf.v4.new_code_cell("""
def optimize_prob_shift(y_true, preds_proba, demographic_mask, target_ut=0.05):
    best_shift, best_acc, best_ut = 0, 0, 1.0
    sub_y = y_true[demographic_mask]
    sub_proba = preds_proba[demographic_mask].copy()
    
    for shift in np.linspace(0, 0.5, 50):
        shifted = sub_proba.copy()
        shifted[:, 0] += shift * 1.5
        shifted[:, 1] += shift * 1.0
        shifted[:, 2] += shift * 0.5
        
        # Apply optimal class bias internally as well
        logp = np.log(np.clip(shifted, 1e-12, 1.0)) + best_bias_global[None, :]
        preds = np.argmax(logp, axis=1)
        
        ut = (preds > sub_y).mean()
        acc = (preds == sub_y).mean()
        
        if ut <= target_ut and acc > best_acc:
            best_acc, best_shift, best_ut = acc, shift, ut
            
    return best_shift

elderly_mask = train['age_group'].astype(str).str.strip() == 'elderly'
computed_shift = optimize_prob_shift(y_train, final_oof_probs, elderly_mask, target_ut=0.05)
print(f"Dynamically computed Elderly Shift: +{computed_shift:.4f}")

# Apply strictly after meta-learner
final_oof_probs_fair = final_oof_probs.copy()
final_oof_probs_fair[elderly_mask, 0] += computed_shift * 1.5
final_oof_probs_fair[elderly_mask, 1] += computed_shift * 1.0
final_oof_probs_fair[elderly_mask, 2] += computed_shift * 0.5

# Re-apply bias tuning to get final OOF predictions for Cost Matrix
final_tuned_oof_preds = np.argmax(np.log(np.clip(final_oof_probs_fair, 1e-12, 1.0)) + best_bias_global[None, :], axis=1)

print("Calculating SHAP values for Physiological Engine...")
explainer = shap.TreeExplainer(lgb_shap_model)
shap_values = explainer.shap_values(X_shap_sample)

if isinstance(shap_values, list):
    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
else:
    if len(shap_values.shape) == 3:
        mean_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

top_features = pd.Series(mean_shap, index=X_shap_sample.columns).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
top_features.sort_values(ascending=True).plot(kind='barh')
plt.title('Top 15 Global SHAP Feature Importances (Physiological Track)')
plt.xlabel('Mean |SHAP Value|')
plt.tight_layout()
plt.show()
    """))

    # 8. Clinical Cost
    cells.append(nbf.v4.new_markdown_cell("""
## Asymmetric Clinical Cost & CARES Risk Control
We propose a custom `COST_MATRIX` that heavily penalizes undertriage. Furthermore, CARES estimates selective risk using Clopper-Pearson upper bounds on out-of-fold correctness.
    """))
    
    cells.append(nbf.v4.new_code_cell("""
COST_MATRIX = np.array([
    [  0,  2, 10, 30, 50],
    [  1,  0,  5, 15, 30],
    [  2,  1,  0,  3, 10],
    [  3,  2,  1,  0,  4],
    [  4,  3,  2,  1,  0],
])

y_pred = final_tuned_oof_preds
cm = confusion_matrix(y_train, y_pred)
cm_norm = cm / cm.sum()
total_cost = np.sum(cm_norm * COST_MATRIX)
ut_cost = np.sum(cm_norm * COST_MATRIX * np.triu(np.ones_like(COST_MATRIX), k=1))
print(f"Cost from Undertriage: {ut_cost:.5f} ({(ut_cost/total_cost)*100:.1f}%)\\n")

# CARES Safety on Calibrated Meta-Learner
is_correct = (y_pred == y_train).astype(int)
global_mean = is_correct.mean()
print(f"Base Accuracy: {global_mean*100:.2f}%")

max_prob = np.max(final_oof_probs, axis=1) # use pre-fairness shift probabilities as they are properly calibrated
sorted_idx = np.argsort(-max_prob)
is_correct_sorted = is_correct[sorted_idx]

cum_errors = np.cumsum(1 - is_correct_sorted)
cum_total = np.arange(1, len(is_correct_sorted) + 1)
cp_lower, cp_upper = proportion_confint(cum_errors, cum_total, alpha=0.1, method='beta')

valid_idx = np.where(cp_upper <= 0.05)[0]
if len(valid_idx) > 0:
    max_valid_idx = valid_idx[-1]
    coverage = (max_valid_idx + 1) / len(is_correct_sorted)
    print(f"CARES Selective Risk Control can automate {coverage*100:.2f}% of cases while keeping upper bound error < 5%.")
    """))

    # 9. Submission
    cells.append(nbf.v4.new_code_cell("""
# Fairness: Apply computed shift to test set probabilities
elderly_mask_test = test['age_group'].astype(str).str.strip() == 'elderly'
final_test_probs_fair = final_test_probs.copy()
final_test_probs_fair[elderly_mask_test, 0] += computed_shift * 1.5
final_test_probs_fair[elderly_mask_test, 1] += computed_shift * 1.0
final_test_probs_fair[elderly_mask_test, 2] += computed_shift * 0.5

# Apply Global Bias Tuning to final test predictions
final_test_preds = np.argmax(np.log(np.clip(final_test_probs_fair, 1e-12, 1.0)) + best_bias_global[None, :], axis=1)

try:
    sub = pd.read_csv(PATH + 'sample_submission.csv')
    sub['triage_acuity'] = final_test_preds + 1
    sub.to_csv('submission.csv', index=False)
    print("Submission generated successfully.")
except FileNotFoundError:
    pass
    """))
    
    nb['cells'] = cells
    with open('submissions/final_submission.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Notebook generated successfully.")

if __name__ == "__main__":
    create_notebook()
