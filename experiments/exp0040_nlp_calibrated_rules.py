import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import cohen_kappa_score

def custom_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

print("Loading data...")
train = pd.read_csv('data/train.csv')
cc = pd.read_csv('data/chief_complaints.csv')
train = train.merge(cc, on='patient_id', how='left')

y = train['triage_acuity'].values - 1
nlp_text = train['chief_complaint_raw'].fillna('Missing').astype(str).values
groups = nlp_text.copy()

# Step 2.1: Find exact rules (deterministic templates)
# We calculate the purity of each template in the training set
df = pd.DataFrame({'text': nlp_text, 'y': y})
template_counts = df.groupby('text').size()
template_purity = df.groupby('text')['y'].nunique()
pure_templates = template_purity[template_purity == 1].index

print(f"Total unique templates: {len(template_counts)}")
print(f"Templates perfectly deterministic (pure): {len(pure_templates)}")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_nlp_probs = np.zeros((len(y), 5))

print("Starting Grouped CV Loop for NLP...")
for fold, (train_idx, valid_idx) in enumerate(sgkf.split(nlp_text, y, groups=groups)):
    X_tr_text, X_vl_text = nlp_text[train_idx], nlp_text[valid_idx]
    y_tr, y_vl = y[train_idx], y[valid_idx]
    
    # Step 2.2: Ensemble Text Modeling (Word + Char n-grams)
    tfidf = TfidfVectorizer(max_features=5000, analyzer='word', ngram_range=(1,2), stop_words='english')
    X_tr_tfidf = tfidf.fit_transform(X_tr_text)
    X_vl_tfidf = tfidf.transform(X_vl_text)
    
    # Step 2.3: True Probability Calibration
    # Instead of hacky softmax, we wrap Ridge in CalibratedClassifierCV
    base_ridge = RidgeClassifier(alpha=1.0, class_weight='balanced')
    calibrated_clf = CalibratedClassifierCV(base_ridge, cv=3, method='isotonic')
    calibrated_clf.fit(X_tr_tfidf, y_tr)
    
    oof_nlp_probs[valid_idx] = calibrated_clf.predict_proba(X_vl_tfidf)
    
    fold_qwk = custom_qwk(y_vl, np.argmax(oof_nlp_probs[valid_idx], axis=1))
    print(f"Fold {fold} Calibrated NLP QWK: {fold_qwk:.4f}")

total_qwk = custom_qwk(y, np.argmax(oof_nlp_probs, axis=1))
print(f"\\nGrouped CV Calibrated NLP QWK: {total_qwk:.4f}")
