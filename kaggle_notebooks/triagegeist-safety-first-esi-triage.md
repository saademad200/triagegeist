# Triagegeist — Safety-First ESI Triage Acuity Prediction

**A decision-support model that predicts Emergency Severity Index (ESI 1–5) from intake data, engineered around the error that actually harms patients: _undertriage_.**

> Every minute counts in the emergency department. Triage assigns each arriving patient an acuity level (ESI 1 = most urgent → 5 = least urgent) under time pressure and incomplete information. Inter-rater variability and **systematic undertriage of vulnerable groups** are documented patient-safety problems. This notebook builds a transparent, reproducible model and — crucially — audits it for *clinical safety* and *fairness*, not just accuracy.

### What this notebook delivers
1. **Leakage-controlled** acuity model (drops post-triage outcome columns).
2. **Asymmetric, undertriage-aware** evaluation (a missed ESI-1 is not the same as a missed ESI-5).
3. A **fairness/bias audit** of undertriage across age, sex, language, and insurance.
4. **Explainability** — the clinical features driving each prediction.

_Note on data: this notebook uses the competition's provided structured intake dataset. It is a synthetic/simulated cohort intended as a proof-of-concept; the **methodology** (leakage control, undertriage-aware evaluation, fairness auditing, explainability) is what transfers to real data such as MIMIC-IV-ED or NHAMCS._

## 1. Setup & data


```python
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

CANDS = ['/kaggle/input/triagegeist', '/kaggle/input/competitions/triagegeist', '../data', './data']
D = next(p for p in CANDS if os.path.exists(os.path.join(p, 'train.csv')))
TARGET, ID, SEED = 'triage_acuity', 'patient_id', 42
rd = lambda f: pd.read_csv(os.path.join(D, f))
train, test = rd('train.csv'), rd('test.csv')
cc, ph, sub = rd('chief_complaints.csv'), rd('patient_history.csv'), rd('sample_submission.csv')
print('train', train.shape, '| test', test.shape, '| history', ph.shape, '| complaints', cc.shape)
```

## 2. Exploratory analysis
ESI is **ordinal** and **imbalanced** — most patients are mid-acuity (3–4), while the life-threatening ESI-1 class is rare. This imbalance is exactly why plain accuracy is misleading and why we track undertriage separately.


```python
vc = train[TARGET].value_counts().sort_index()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(vc.index.astype(str), vc.values, color='#3b7dd8'); ax[0].set_title('ESI acuity distribution (train)'); ax[0].set_xlabel('ESI (1=most urgent)')
if 'news2_score' in train: train.boxplot(column='news2_score', by=TARGET, ax=ax[1]); ax[1].set_title('NEWS2 by acuity'); ax[1].set_xlabel('ESI')
plt.suptitle(''); plt.tight_layout(); plt.show()
print(vc.to_dict())
```

## 3. Leakage control — the most important modeling decision
`disposition` and `ed_los_hours` appear only in `train`. They describe what happened **after** the triage decision (admission outcome, length of stay), so using them is circular and clinically invalid — and they are unavailable at inference time. We drop them. Verifying *no* train-only feature leaks into the model is the difference between an honest 0.9 and a fantasy 0.99.


```python
leak = ['disposition', 'ed_los_hours']
print('Train-only columns:', sorted(set(train.columns) - set(test.columns)))
train = train.drop(columns=[c for c in leak if c in train.columns])
# merge chief-complaint text + history
train = train.merge(cc[[ID,'chief_complaint_raw']].drop_duplicates(ID), on=ID, how='left').merge(ph.drop_duplicates(ID), on=ID, how='left')
test  = test.merge(cc[[ID,'chief_complaint_raw']].drop_duplicates(ID), on=ID, how='left').merge(ph.drop_duplicates(ID), on=ID, how='left')
print('after merge:', train.shape, test.shape)
```

## 4. Feature engineering
Clinically motivated: abnormal-vital flags at standard thresholds, comorbidity burden, and chief-complaint signals (keyword flags + TF-IDF→SVD on the free text).


```python
def feats(df):
    for c,(lo,hi) in {'heart_rate':(50,100),'systolic_bp':(90,180),'spo2':(92,101),'respiratory_rate':(8,22),'temperature_c':(35,38)}.items():
        if c in df: df[f'{c}_lo']=(df[c]<lo).astype('int8'); df[f'{c}_hi']=(df[c]>hi).astype('int8')
    hx=[c for c in df.columns if c.startswith('hx_')]
    if hx: df['hx_count']=df[hx].fillna(0).sum(axis=1)
    if 'chief_complaint_raw' in df:
        t=df['chief_complaint_raw'].fillna('').astype(str).str.lower()
        df['kw_crit']=t.str.contains(r'arrest|unrespons|not breathing|anaphylaxis|stroke|seizure|overdose',regex=True).astype('int8')
        df['kw_high']=t.str.contains(r'chest pain|short.*breath|severe|sudden|syncope|altered',regex=True).astype('int8')
        df['kw_low']=t.str.contains(r'\bmild\b|minor|routine|chronic|refill|rash',regex=True).astype('int8')
        df['cc_len']=t.str.len()
    return df
train, test = feats(train), feats(test)
vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=4000,sublinear_tf=True)
Xt=vec.fit_transform(train['chief_complaint_raw'].fillna('').astype(str)); Xe=vec.transform(test['chief_complaint_raw'].fillna('').astype(str))
svd=TruncatedSVD(32,random_state=SEED); St=svd.fit_transform(Xt); Se=svd.transform(Xe)
for i in range(32): train[f'tx_{i}']=St[:,i]; test[f'tx_{i}']=Se[:,i]
drop=[ID,TARGET,'chief_complaint_raw']; fcols=[c for c in train.columns if c not in drop and c in test.columns]
for c in fcols:
    if train[c].dtype=='object':
        cats=pd.concat([train[c],test[c]]).astype('category').cat.categories
        train[c]=pd.Categorical(train[c],categories=cats).codes; test[c]=pd.Categorical(test[c],categories=cats).codes
Xtr=train[fcols].astype('float32'); Xte=test[fcols].astype('float32')
classes=np.sort(train[TARGET].unique()); c2i={c:i for i,c in enumerate(classes)}; i2c={v:k for k,v in c2i.items()}; y=train[TARGET].map(c2i).values
print(len(fcols),'features')
```

## 5. Model & cross-validation
LightGBM multiclass, 5-fold stratified CV, out-of-fold predictions. We report **linear** and **quadratic weighted κ** (the metrics appropriate for ordinal acuity), alongside accuracy and macro-F1.


```python
lwk=lambda a,b: cohen_kappa_score(a,b,weights='linear'); qwk=lambda a,b: cohen_kappa_score(a,b,weights='quadratic')
skf=StratifiedKFold(5,shuffle=True,random_state=SEED); oof=np.zeros((len(y),len(classes))); imp=np.zeros(len(fcols))
for tr,va in skf.split(Xtr,y):
    m=lgb.LGBMClassifier(objective='multiclass',num_class=len(classes),n_estimators=600,learning_rate=0.05,
        num_leaves=127,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=SEED,n_jobs=-1,verbose=-1)
    m.fit(Xtr.iloc[tr],y[tr]); oof[va]=m.predict_proba(Xtr.iloc[va]); imp+=m.booster_.feature_importance('gain')/5
pred=oof.argmax(1)
print(f'Linear weighted kappa : {lwk(y,pred):.4f}')
print(f'Quadratic weighted kappa: {qwk(y,pred):.4f}')
print(f'Accuracy               : {accuracy_score(y,pred):.4f}')
print(f'Macro-F1               : {f1_score(y,pred,average="macro"):.4f}')
```

## 6. Clinical safety — undertriage analysis
**Undertriage** (predicting a *less* urgent level than the truth) is the dangerous error: it can delay life-saving care. We separate undertriage from overtriage and zoom in on **critical undertriage** — true ESI 1–2 patients predicted as less urgent. A safety-grade triage aid must keep this near zero.


```python
under=(pred>y).mean(); over=(pred<y).mean()
crit=np.isin(y,[c2i[1],c2i[2]]); crit_under=(pred[crit]>y[crit]).mean()
print(f'Undertriage rate        : {under*100:.2f}%')
print(f'Overtriage rate         : {over*100:.2f}%')
print(f'Critical undertriage    : {crit_under*100:.2f}%  (true ESI 1-2 predicted less urgent)')
cm=confusion_matrix(y,pred)
plt.figure(figsize=(5,4)); plt.imshow(cm,cmap='Blues'); plt.colorbar(); plt.title('Confusion matrix (OOF)')
plt.xlabel('Predicted ESI idx'); plt.ylabel('True ESI idx');
for i in range(len(classes)):
    for j in range(len(classes)): plt.text(j,i,cm[i,j],ha='center',va='center',fontsize=8)
plt.tight_layout(); plt.show()
```

## 7. Fairness / bias audit
Systematic undertriage of specific populations is an active patient-safety concern in the triage literature. We break down undertriage and accuracy across **age group, sex, language, and insurance type** — a model can be accurate overall yet quietly under-serve a subgroup.


```python
rows=[]
for col in ['age_group','sex','language','insurance_type']:
    if col in train.columns:
        for val,idx in train.groupby(col).groups.items():
            ii=train.index.get_indexer(idx)
            rows.append({'attribute':col,'group':str(val),'n':len(ii),
                         'accuracy':round(accuracy_score(y[ii],pred[ii]),3),
                         'undertriage_%':round((pred[ii]>y[ii]).mean()*100,2)})
fair=pd.DataFrame(rows); print(fair.to_string(index=False))
piv=fair.pivot_table(index='attribute',values='undertriage_%',aggfunc=['min','max'])
print('\nMax undertriage gap within any attribute:', round((fair.groupby('attribute')['undertriage_%'].max()-fair.groupby('attribute')['undertriage_%'].min()).max(),2),'percentage points')
```

## 8. Explainability
Triage clinicians will only trust a tool they can interrogate. The top gain-importance features confirm the model leans on **physiologically meaningful** signals (NEWS2, GCS, SpO₂, vitals, pain), not spurious identifiers.


```python
order=np.argsort(imp)[::-1][:15]
plt.figure(figsize=(7,5)); plt.barh([fcols[i] for i in order][::-1],[imp[i] for i in order][::-1],color='#3b7dd8')
plt.title('Top features (LightGBM gain)'); plt.tight_layout(); plt.show()
print('Top drivers:', ', '.join(fcols[i] for i in order[:10]))
```

## 9. Limitations & reproducibility
- **Synthetic data.** The provided cohort is simulated; the near-ceiling κ reflects a partly rule-generated label process, not a claim of real-world performance. The contribution is the **safety-and-fairness-aware methodology**, designed to transfer to MIMIC-IV-ED / NHAMCS.
- **No temporal validation.** Real deployment needs prospective, site-stratified validation and calibration monitoring.
- **Decision support, not replacement.** Intended to flag likely under-triaged patients for clinician review.
- **Reproducibility.** Fixed seed (42), single notebook, no internet/external data, deterministic 5-fold CV. Runs end-to-end top-to-bottom.

## 10. Submission


```python
mf=lgb.LGBMClassifier(objective='multiclass',num_class=len(classes),n_estimators=700,learning_rate=0.05,
    num_leaves=127,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=SEED,n_jobs=-1,verbose=-1)
mf.fit(Xtr,y); tp=pd.Series(mf.predict(Xte)).map(i2c).astype(int).values
out=sub.copy(); pmap=dict(zip(test[ID].values,tp))
out[TARGET]=out[ID].map(pmap).fillna(int(pd.Series(train[TARGET]).mode().iloc[0])).astype(int)
out[[ID,TARGET]].to_csv('submission.csv',index=False)
print('submission.csv', out.shape, out[TARGET].value_counts().sort_index().to_dict())
```
