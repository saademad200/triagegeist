## **Section 1: Train Data**
---


```python
import warnings

warnings.filterwarnings("ignore")
```


```python
import pandas as pd

df = pd.read_csv("/kaggle/input/competitions/triagegeist/train.csv")
df.head()
```

- **Getting dataset shape**


```python
df.shape
```

# **Data Cleaning**
---

## **Dropping `duplicate` values**


```python
df = df.drop_duplicates()
df.shape
```

## **Handling `NaN` values**


```python
import numpy as np

na_ratio = df.isna().sum() / len(df)
na_ratio = na_ratio[na_ratio > 0]

np.round(na_ratio, 3)
```

- **`NOTE:`** Since we have other features that have missing values that are represented by **`-1`**.


```python
df = df.fillna(-1).reset_index(drop=True)
df.head()
```

# **Feature Selection**
---


```python
df.iloc[0].to_frame()
```

## **Manual Feature Removal**

### **`disposition` and `ed_los_hours`**
- It is often post-triage information so this can highly affect the triage decision and will not be available for triage prediction!


```python
df = df.drop(['disposition', 'ed_los_hours'], axis=1)
```

### **`patient_id`**

- **Is total `patient_id` counts equal to total unique `patient_id`**


```python
if (df['patient_id'].count() == len(df['patient_id'].unique())) == True:
 print("All patient ids are unique!")
else:
 print("All patient ids are not unique!")
```

- **Since all are unique, we just drop them directly as we do not get any important information from it!**


```python
df = df.drop('patient_id', axis=1)
```

### **`nurse_id`**
The triage decision must be made depending upon the condition of the patient and not at all by a nurse. So, we drop it!


```python
df = df.drop('triage_nurse_id', axis=1)
```

### **`insurance_type`**
A model can be biased if the decision is made with consideration to the insurance of the patient, so we drop it!


```python
df = df.drop('insurance_type', axis=1)
```

### **`news2_score`**
The **`NEWS 2`** score is a standardized clinical scoring system used in emergency and acute care to quantify how sick a patient is and how likely they are to deteriorate. This can higly influence the traige decision in a negative direction because, having a single feature making the final prediction while rendering all other features irrelelvant is not good at all! So, we are going to run a few tests on these.

### **Ablation Test**

- **Declaring column types**


```python
cat_cols = [ 'site_id', 'arrival_mode', 'arrival_hour', 'arrival_day', 'arrival_month',
 'arrival_season', 'shift', 'age_group', 'sex', 'language', 'transport_origin', 'pain_location', 
 'mental_status_triage', 'chief_complaint_system'
]

num_cols = ['news2_score', 'age', 'num_prior_ed_visits_12m', 'num_prior_admissions_12m', 
 'num_active_medications', 'num_comorbidities', 'systolic_bp', 'diastolic_bp', 
 'mean_arterial_pressure', 'pulse_pressure', 'heart_rate', 'respiratory_rate', 'temperature_c', 
 'spo2', 'gcs_total', 'pain_score', 'weight_kg', 'height_cm', 'bmi', 'shock_index'
]

target_col = ['triage_acuity']
```

#### **Splitting, Scaling and Encoding**


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# defining features & target
X = df[cat_cols + num_cols].copy()
y = df[target_col[0]].copy()

# performing train-test split first
X_train, X_test, y_train, y_test = train_test_split(
 X, y,
 test_size=0.2,
 random_state=42,
 stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# performing label encoding on training data only
label_encoders = {}

for col in cat_cols:
 le = LabelEncoder()
 
 X_train[col] = X_train[col].astype(str)
 X_test[col] = X_test[col].astype(str)
 
 # fitting on training data
 X_train[col] = le.fit_transform(X_train[col])
 
 # transforming test data while handling unseen values safely
 X_test[col] = X_test[col].map(
 lambda x: le.transform([x])[0] if x in le.classes_ else -1
 )
 
 label_encoders[col] = le

# performing scaling on training data only
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# converting to NumPy
X_train_ = X_train.to_numpy()
X_test_ = X_test.to_numpy()
y_train_ = y_train.to_numpy()
y_test_ = y_test.to_numpy()

# performing sanity checks
print("\nTrain class distribution:\n", y_train.value_counts(normalize=True))
print("\nTest class distribution:\n", y_test.value_counts(normalize=True))
```

- **Accuracy Test with `news2_score`**


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

news2_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with NEWS 2 Score:", news2_accuracy)
```

- **Accuracy test without `news2_score`**


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier(random_state=42)

X_train_no_news = X_train.drop(columns=["news2_score"])
X_test_no_news = X_test.drop(columns=["news2_score"])

model.fit(X_train_no_news, y_train)
y_pred = model.predict(X_test_no_news)

no_news2_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy without NEWS 2 Score:", no_news2_accuracy)
```

**With NEWS 2 Score** : 0.7694375
**Without NEWS 2 Score** : 0.76375
**Difference** : ~0.005 (0.57%)

This clearly means that it is only providing about **`0.57%`** contribution towards accuracy which does not prove to be dominating, but it indeed is useful.

### **Single Feature Baseline**


```python
X_train_news = X_train[["news2_score"]]
X_test_news = X_test[["news2_score"]]

model.fit(X_train_news, y_train)
y_pred = model.predict(X_test_news)

print("Accuracy with only NEWS 2:", accuracy_score(y_test, y_pred))
```

NEWS 2 alone is moderately predictive not not highly dominating. This can be acclaimed by checking the following jump of **`84.06%`** from **`65.28%`**, i.e. **`18.8%`** which has been provided by the other features meaning that my model captures substantial additional information even beyond **`NEWS 2 Score`**.

### **SHAP Dominance Check**


```python
import shap

# creating explainer
explainer = shap.Explainer(model, X_train)

# computing SHAP values
shap_values = explainer(X_test)

# extracting NumPy array
shap_vals = shap_values.values

mean_importance = np.mean(np.abs(shap_vals), axis=(0,2))

importance_df = pd.DataFrame({
 "feature": X_test.columns,
 "importance": mean_importance
}).sort_values(by="importance", ascending=False)

display(importance_df.head(10))
```

**`NEWS 2 Score`** does not dominate importance so we can keep it. But there is a serious issue with this feature. **`news2_score`** is a patient vitals derived feature. Its value is determined by **SpO₂**, **Heart rate**, **Respiratory rate**, **Blood pressure**, **Temperature**, **Consciousness** and **Oxygen support**. Since we have these features already in the input stream, **`NEWS2_score`** is only providing a compressed summary of the patient and not adding any meaningful raw data about the patient! This is why, we drop it.


```python
df = df.drop('news2_score', axis=1)
```

## **Single feature predictive power test for Numerical Features**

- **Re-defninng numerical columns**


```python
num_cols = ['age', 'num_prior_ed_visits_12m', 'num_prior_admissions_12m', 'num_active_medications', 
 'num_comorbidities', 'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure', 'pulse_pressure', 
 'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 'gcs_total', 'pain_score', 'weight_kg', 
 'height_cm', 'bmi', 'shock_index'
]
```


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

results = []

for col in num_cols:
    model = DecisionTreeClassifier(random_state=42)
    
    model.fit(X_train[[col]], y_train)
    acc = accuracy_score(y_test, model.predict(X_test[[col]]))
    
    results.append((col, acc))

# converting to DataFrame
results_df = pd.DataFrame(results, columns=["Feature", "Accuracy"])
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
display(results_df.head(10))
```

This is what we observe:

| **Feature** | **Accuracy** | **Risk Level** |
| -------------------------- | ------------ | --------------- |
| **gcs_total** | 0.5350 | High |
| **respiratory_rate** | 0.5050 | Moderate |
| **pain_score** | 0.5048 | High |
| **spo2** | 0.4946 | Moderate |
| **temperature_c** | 0.4836 | Moderate |
| **shock_index** | 0.4694 | Moderate |
| **mean_arterial_pressure** | 0.4659 | Moderate |

Lets consider **`gcs_total`** and **`pain_score`** for now. They have high predictive power but that does not make it inherently bad! GCS is calculated before triage and pain score is a self desined score so it is okay to keep both of them. Others dont really have a suspicious level of predictive capability so we keep them as is.

## **Categorical Features Relevance Test**


```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway, entropy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
import os

results = []
results_dir = "/kaggle/working/results/"
filename = "categorical_feature_analysis_for_training_data.csv"
os.makedirs(results_dir, exist_ok=True)

# encoding target once
le_target = LabelEncoder()
y = le_target.fit_transform(df[target_col[0]])

for col in tqdm(cat_cols, desc="Categorical Feature Tests (5 Tests)"):
    
    temp = df[[col, target_col[0]]].dropna()
    
    if temp[col].nunique() < 2:
        continue
    
    # encoding feature
    le = LabelEncoder()
    X_encoded = le.fit_transform(temp[col])
    y_encoded = LabelEncoder().fit_transform(temp[target_col[0]])
    
    # computing chi-square
    contingency = pd.crosstab(temp[col], temp[target_col[0]])
    chi2, p_chi2, _, _ = chi2_contingency(contingency)
    chi2_relevant = p_chi2 < 0.05
    
    # computing Cramér’s V
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    cramers_v = np.sqrt(phi2 / (min(k - 1, r - 1))) if min(k - 1, r - 1) > 0 else 0
    cramers_relevant = cramers_v >= 0.1
    
    # computing mutual information
    mi = mutual_info_classif(
        X_encoded.reshape(-1, 1),
        y_encoded,
        discrete_features=True
    )[0]
    
    # computing ANOVA on encoded values
    groups = [X_encoded[y_encoded == cls] for cls in np.unique(y_encoded)]
    try:
        _, p_anova = f_oneway(*groups)
    except:
        p_anova = 1.0
    anova_relevant = p_anova < 0.05
    
    # computing entropy for feature richness
    probs = temp[col].value_counts(normalize=True)
    ent = entropy(probs)
    entropy_relevant = ent > 1.0  
    
    results.append({
        "feature": col,
        
        "chi2_p": p_chi2,
        "chi2_relevant": chi2_relevant,
        
        "cramers_v": cramers_v,
        "cramers_relevant": cramers_relevant,
        
        "mutual_info": mi,
        
        "anova_p": p_anova,
        "anova_relevant": anova_relevant,
        
        "entropy": ent,
        "entropy_relevant": entropy_relevant
    })

cat_results = pd.DataFrame(results)

# applying mutual information threshold
mi_threshold = cat_results["mutual_info"].median()
cat_results["mi_relevant"] = cat_results["mutual_info"] >= mi_threshold

# performing final voting
cat_results["relevance_votes"] = (
    cat_results["chi2_relevant"].astype(int) +
    cat_results["cramers_relevant"].astype(int) +
    cat_results["mi_relevant"].astype(int) +
    cat_results["anova_relevant"].astype(int) +
    cat_results["entropy_relevant"].astype(int)
)

# applying final decision threshold of at least 3 out of 5
cat_results["Overall_Relevance"] = np.where(
    cat_results["relevance_votes"] >= 3,
    "RELEVANT",
    "NOT_RELEVANT"
)

# sorting results
cat_results = cat_results.sort_values(
    by=["relevance_votes", "mutual_info"],
    ascending=False
).reset_index(drop=True)

display(cat_results)

results_save_path = os.path.join(results_dir, filename)
cat_results.to_csv(results_save_path, index=False)
print(f"Results saved to {results_save_path}")
```

- It can be clearly seen that only **`mental_status_triage`** is relevant so we keep it and we can drop the rest!


```python
cat_cols = ['mental_status_triage']
```

## **Numerical Features Relevance Test**


```python
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

# removing leakage columns
leakage_cols = ['disposition', 'ed_los_hours']
num_cols_clean = [col for col in num_cols if col not in leakage_cols]

# encoding target
le = LabelEncoder()
y = le.fit_transform(df[target_col[0]])

# computing mutual information in batch
X_num = df[num_cols_clean].copy()
X_num_filled = X_num.fillna(X_num.median())

mi_scores = mutual_info_classif(X_num_filled, y)
mi_dict = dict(zip(num_cols_clean, mi_scores))

results = []

# running feature selection loop
for col in tqdm(num_cols_clean, desc="Numerical 5-Test Evaluation"):
    
    x = df[col].values
    mask = ~np.isnan(x)
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(np.unique(x_clean)) < 2:
        continue
    
    groups = [x_clean[y_clean == cls] for cls in np.unique(y_clean)]
    
    if len(groups) < 2:
        continue
    
    # computing ANOVA
    try:
        _, p_anova = f_oneway(*groups)
    except:
        p_anova = 1.0
    anova_rel = p_anova < 0.05
    
    # computing Kruskal-Wallis test
    try:
        _, p_kruskal = kruskal(*groups)
    except:
        p_kruskal = 1.0
    kruskal_rel = p_kruskal < 0.05
    
    # computing eta squared
    grand_mean = np.mean(x_clean)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((x_clean - grand_mean) ** 2)
    eta_sq = ss_between / ss_total if ss_total != 0 else 0
    eta_rel = eta_sq >= 0.01
    
    # computing mutual information
    mi = mi_dict[col]
    # applying threshold later
    
    # computing Spearman correlation
    try:
        corr, p_spear = spearmanr(x_clean, y_clean)
    except:
        corr, p_spear = 0, 1.0
    
    spear_rel = (abs(corr) >= 0.1) and (p_spear < 0.05)
    
    results.append({
        "feature": col,
        
        "anova_p": p_anova,
        "anova_relevant": anova_rel,
        
        "kruskal_p": p_kruskal,
        "kruskal_relevant": kruskal_rel,
        
        "eta_squared": eta_sq,
        "eta_relevant": eta_rel,
        
        "mutual_info": mi,
        
        "spearman_corr": corr,
        "spearman_p": p_spear,
        "spearman_relevant": spear_rel
    })

num_results = pd.DataFrame(results)

# applying mutual information threshold
mi_threshold = num_results["mutual_info"].median()
num_results["mi_relevant"] = num_results["mutual_info"] >= mi_threshold

# applying voting
num_results["relevance_votes"] = (
    num_results["anova_relevant"].astype(int) +
    num_results["kruskal_relevant"].astype(int) +
    num_results["eta_relevant"].astype(int) +
    num_results["mi_relevant"].astype(int) +
    num_results["spearman_relevant"].astype(int)
)

# making final decision
num_results["Overall_Relevance"] = np.where(
    num_results["relevance_votes"] >= 3,
    "RELEVANT",
    "NOT_RELEVANT"
)

# sorting results
num_results = num_results.sort_values(
    by=["relevance_votes", "mutual_info"],
    ascending=False
).reset_index(drop=True)

num_results = num_results.round({
    "anova_p": 2,
    "kruskal_p": 2,
    "eta_squared": 2,
    "mutual_info": 2,
    "spearman_corr": 2,
    "spearman_p": 2
})

num_results

filename = "numerical_feature_analysis_for_training_data.csv"
results_save_path = os.path.join(results_dir, filename)
num_results.to_csv(results_save_path, index=False)

print(f"Results saved to {results_save_path}")
```

- **Keeping only `relevant` numerical features**


```python
import os
import pandas as pd

relevant_set = set(
 num_results.loc[num_results["Overall_Relevance"] == "RELEVANT", "feature"]
)

num_cols = [col for col in num_cols if col in relevant_set]

selected_num_cols_df = pd.DataFrame({
 "selected_numerical_features": num_cols
})

results_dir = "/kaggle/working/results/"
os.makedirs(results_dir, exist_ok=True)

filename = "selected_numerical_features_for_training_data.csv"
save_path = os.path.join(results_dir, filename)
selected_num_cols_df.to_csv(save_path, index=False)

print(f"Selected numerical features saved to {save_path}")
display(selected_num_cols_df)
```

## **Final Feature Space**


```python
# combining all selected columns
final_cols = cat_cols + num_cols + target_col

# keeping only columns that actually exist as a safety check
final_cols = [col for col in final_cols if col in df.columns]

# subsetting dataframe
df = df[final_cols].copy()
print(df.columns)

# optionally performing sanity check
print(f"Final shape: {df.shape}")
df.head()
```

## **Checking for duplicated inputs**


```python
df.duplicated().sum()
```

# **Exploratory Data Analysis**
---

## **Target Variable Analysis**


```python
import os
import matplotlib.pyplot as plt

# defining path
plots_dir = "/kaggle/working/plots/training_data/"

# creating directory
os.makedirs(plots_dir, exist_ok=True)

# plotting
plt.figure(figsize=(6, 4))

df["triage_acuity"].value_counts().sort_index().plot(kind="bar", rot=0)

plt.title("Triage Acuity Distribution", fontweight='bold')
plt.xlabel("Triage Acuity", fontweight='bold')
plt.ylabel("Count", fontweight='bold')

filename = "triage_acuity_distribution.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

print(f"Plot saved to {save_path}")
```

## **Categorical Feature Analysis**


```python
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="mental_status_triage", hue="triage_acuity")
plt.title("Mental Status vs Triage", fontweight='bold')
plt.xlabel("Class", fontweight='bold')
plt.ylabel("Mental Status Triage", fontweight='bold')
plt.xticks(rotation=0)

filename = "categorical_feature_analysis.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

print(f"Plot saved to {save_path}")
```

## **Numerical Feature Analysis**


```python
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop("triage_acuity")

fig, axes = plt.subplots(5, 3, figsize=(15, 15))
axes = axes.flatten()

for i, col in enumerate(num_cols):
 axes[i].hist(df[col], bins=30)
 axes[i].set_title(col, fontweight='bold')

for j in range(len(num_cols), len(axes)):
 fig.delaxes(axes[j])

plt.suptitle("Numerical Features Distribution", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.98])

filename = "numerical_feature_analysis.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

print(f"Plot saved to {save_path}")
```

## **Feature vs Target**


```python
fig, axes = plt.subplots(5, 3, figsize=(15, 20))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    
    # robust filtering
    temp_df = df.loc[(df[col] != -1) & (~df[col].isna())]
    
    if temp_df.empty:
        continue
    
    sns.boxplot(x="triage_acuity", y=col, data=temp_df, ax=axes[i])
    axes[i].set_title(col, fontweight='bold')

# removing unused subplots
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Outlier Detection with Boxplots", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.98])

filename = "feature_vs_target.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

print(f"Plot saved to {save_path}")
```

## **Correlation Analysis**


```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# replacing -1 with NaN for numerical columns
df_corr = df[num_cols].replace(-1, np.nan)

# computing correlation with NaNs ignored automatically
corr = df_corr.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 9}
)

plt.title("Correlation Matrix", fontweight='bold')

filename = "correlation_heatmap.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

print(f"Plot saved to {save_path}")
```

**`num_active_medication`** and **`num_comorbidities`** are highly correlated which does make absolute sense, the more disease you have the greater the medications you are on! We drop **`num_active_medication`**. Additionally, **`mean_arterial_pressure`** and **`pulse_pressure`** are calculated with **`systolic_bp`** and **`diastolic_bp`** so they too are highly correlated. We hence drop the derived features!


```python
df = df.drop(['num_comorbidities', 'mean_arterial_pressure', 'pulse_pressure'], axis=1)

num_cols = ['num_prior_ed_visits_12m', 'num_prior_admissions_12m', 'num_active_medications', 
 'systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 
 'gcs_total', 'pain_score', 'shock_index']
```


```python
corr = df[num_cols].corr()

plt.figure(figsize=(10, 7.5))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 8})
plt.title("Correlation Matrix", fontweight='bold')

filename = "cleaned_correlation_heatmap.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

print(f"Plot saved to {save_path}")
```

# **Data Preprocessing**
---

## **Label Encoding and Standard Scaling**


```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# defining X and y
X = df[cat_cols + num_cols].copy()
y = df[target_col[0]].copy()

# performing split first
X_train, X_test, y_train, y_test = train_test_split(
 X, y,
 test_size=0.2,
 random_state=42,
 stratify=y
)

# performing label encoding by fitting on training data only
label_encoders = {}

for col in cat_cols:
 le = LabelEncoder()
 
 # converting to string safely
 X_train[col] = X_train[col].astype(str)
 X_test[col] = X_test[col].astype(str)
 
 # fitting only on training data
 X_train[col] = le.fit_transform(X_train[col])
 
 # transforming test data while handling unseen values safely
 X_test[col] = X_test[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
 
 label_encoders[col] = le

# performing scaling by fitting on training data only
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# performing sanity check
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTrain class distribution:\n", y_train.value_counts(normalize=True))
print("\nTest class distribution:\n", y_test.value_counts(normalize=True))
```

# **Modelling**
---


```python
!pip install lazypredict -q
```


```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score
)
from sklearn.base import clone
from tqdm import tqdm
import pandas as pd
import time
import os
import joblib

# =========================
# MODEL SAVE DIR
# =========================
model_dir = "kaggle/working/temp_models"
os.makedirs(model_dir, exist_ok=True)

# =========================
# defining models
# =========================
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

models_dict = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RidgeClassifier": RidgeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB(),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(),
    "LinearSVC": LinearSVC()
}

results = []

# =========================
# TRAIN LOOP
# =========================
for name, model in tqdm(models_dict.items(), desc="Training Models"):

    start_time = time.time()
    
    try:
        m = clone(model)
        m.fit(X_train, y_train)
        
        y_pred = m.predict(X_test)
        
        # =========================
        # METRICS
        # =========================
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        
        # QWK
        qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
        
        elapsed = time.time() - start_time
        
        # =========================
        # SAVE MODEL
        # =========================
        model_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(m, model_path)
        
        size_kb = os.path.getsize(model_path) / 1024
        
        # =========================
        # STORE RESULTS
        # =========================
        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1_macro": f1,
            "Precision_macro": precision,
            "Recall_macro": recall,
            "QWK": qwk,
            "Model Size (KB)": size_kb,
            "Time (s)": elapsed
        })
        
    except Exception as e:
        results.append({
            "Model": name,
            "Accuracy": None,
            "F1_macro": None,
            "Precision_macro": None,
            "Recall_macro": None,
            "QWK": None,
            "Model Size (KB)": None,
            "Time (s)": None
        })

# =========================
# RESULTS DF
# =========================
results_df = pd.DataFrame(results)

# drop failed
results_df = results_df.dropna()

# sort by F1 (you can change to QWK)
results_df = results_df.sort_values(by="F1_macro", ascending=False).reset_index(drop=True)

results_df.head(5)
```

## **Gradient Boosting Classifier**


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    cohen_kappa_score
)
import matplotlib.pyplot as plt
import numpy as np

import os
import pickle

# training model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# generating predictions
y_pred = model.predict(X_test)

# =========================
# calculating metrics
# =========================
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

# QWK
qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

print(f"Accuracy: {acc:.4f}")
print(f"F1_macro: {f1:.4f}")
print(f"Precision_macro: {precision:.4f}")
print(f"Recall_macro: {recall:.4f}")
print(f"QWK: {qwk:.4f}")

# =========================
# classification report
# =========================
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================
# confusion matrix
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

im = plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix (GBC)", fontweight='bold')
plt.xlabel("Predicted Triage", fontweight='bold')
plt.ylabel("Actual Triage", fontweight='bold')

classes = np.unique(y_test)
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

# annotating values
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
        plt.text(j, i, cm[i, j], ha="center", va="center", color=color)

plt.colorbar(im)
plt.tight_layout()

filename = "gradient_boosting_classifier_confusion_matrix.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Confusion matrix saved to {save_path}")

# =========================
# saving model
# =========================
SAVE_DIR = "/kaggle/working/saved_models/training_data/"
os.makedirs(SAVE_DIR, exist_ok=True)

model_path = os.path.join(SAVE_DIR, "gradient_boosting.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved at: {model_path}")

# =========================
# model size
# =========================
size_kb = os.path.getsize(model_path) / 1024
size_mb = size_kb / 1024

print(f"Model Size: {size_kb:.2f} KB ({size_mb:.2f} MB)")
```

# **Explainability**
---

## **Feature Ranking**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier

# preparing data
X = df.drop(columns=["triage_acuity", "mental_status_triage"]) # remove categorical
y = df["triage_acuity"]

# computing scores

# computing mutual information for continuous features
mi_df = pd.Series(
 mutual_info_classif(X, y, discrete_features=False, random_state=42),
 index=X.columns
)

# computing Spearman correlation for ordinal features
corr_df = X.corrwith(y, method="spearman").abs()

# computing Kendall tau for ordinal features
kendall_df = X.corrwith(y, method="kendall").abs()

# computing permutation importance
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

perm = permutation_importance(
 model, X, y,
 n_repeats=5,
 random_state=42,
 n_jobs=-1
)
perm_df = pd.Series(perm.importances_mean, index=X.columns)

# sorting for plotting
methods = {
 "Mutual Info": mi_df.sort_values(ascending=False),
 "Spearman Corr": corr_df.sort_values(ascending=False),
 "Kendall Tau": kendall_df.sort_values(ascending=False),
 "Permutation": perm_df.sort_values(ascending=False)
}

# plotting raw scores
fig, axes = plt.subplots(2, 2, figsize=(16,8))
axes = axes.flatten()

for i, (name, series) in enumerate(methods.items()):
 top5 = series.head(5)
 
 axes[i].barh(top5.index[::-1], top5.values[::-1])
 axes[i].set_title(name, fontweight='bold')
 axes[i].set_xlabel("Score", fontweight='bold')

plt.tight_layout()

filename = "feature_ranking_tests.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Feature ranking plot saved to {save_path}")

# defining normalization function
def normalize(s):
 return (s - s.min()) / (s.max() - s.min() + 1e-9)

# combining normalized scores
scores_df = pd.DataFrame({
 "mi": normalize(mi_df),
 "spearman": normalize(corr_df),
 "kendall": normalize(kendall_df),
 "perm": normalize(perm_df)
})

scores_df["combined"] = scores_df.mean(axis=1)
scores_df = scores_df.sort_values(by="combined", ascending=False)

# plotting normalized top 5
top5 = scores_df.head(5)

plt.figure(figsize=(10,4))
plt.barh(top5.index[::-1], top5["combined"][::-1])
plt.xlabel("Normalized Combined Score", fontweight='bold')
plt.title("Top 5 Features (Consensus Ranking)", fontweight='bold')
plt.xlim(0, 1.1)

# annotating plot
for i, v in enumerate(top5["combined"][::-1]):
 plt.text(v, i, f"{v:.3f}", va='center')

plt.tight_layout()

filename = "normalized_feature_ranking.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Feature ranking plot saved to {save_path}")
```

## **Class-wise top 5 important features**

### **1. Population-level feature importance**


```python
import shap
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# loading model
MODEL_PATH = os.path.join("/kaggle/working/saved_models/training_data/", "gradient_boosting.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# using KernelExplainer with predict_proba
# Use a small background sample (50–100 rows) to keep computation feasible.
background = shap.sample(X_train, 100, random_state=42)   # or shap.kmeans(X_train, 10) for speed

explainer = shap.KernelExplainer(model.predict_proba, background)

# Computing SHAP values — shap_values will be a list of arrays, one per class
# using a subset of X_test for slower KernelExplainer runs
X_test_sample = X_test[:200]   # adjust as needed

shap_values = explainer.shap_values(X_test_sample)
# shap_values → list of shape [(n_samples, n_features)] * n_classes

n_classes = len(shap_values)
```


```python
import pandas as pd

X_arr = X_test_sample.values if isinstance(X_test_sample, pd.DataFrame) else np.array(X_test_sample)
feature_names = list(X_test_sample.columns) if isinstance(X_test_sample, pd.DataFrame) else None

n_classes = shap_values.shape[2]

for class_idx in range(n_classes):

    actual_triage = model.classes_[class_idx]
    class_shap    = shap_values[:, :, class_idx]   # shape: (200, 13)

    print(f"Triage {actual_triage} → class_shap shape: {class_shap.shape}")

    shap.summary_plot(
        class_shap,
        X_arr,
        feature_names=feature_names,
        plot_type="bar",
        max_display=5,
        plot_size=(12, 4),
        show=False
    )

    plt.suptitle(
        f"Top Features for Triage {actual_triage}",
        fontsize=16,
        fontweight='bold',
        y=0.95
    )
    plt.xlabel("mean(|SHAP|)", fontsize=12, fontweight='bold')
    plt.tight_layout()

    filename  = f"shap_top_features_for_triage_{actual_triage}.png"
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    print(f"Saved → {save_path}")
```

### **2. Peak impact per patient**


```python
import pandas as pd

X_arr = X_test_sample.values if isinstance(X_test_sample, pd.DataFrame) else np.array(X_test_sample)
feature_names = list(X_test_sample.columns) if isinstance(X_test_sample, pd.DataFrame) else None

n_classes = shap_values.shape[2]

for class_idx in range(n_classes):

    actual_triage = model.classes_[class_idx]
    class_shap    = shap_values[:, :, class_idx]   # shape: (200, 13)

    print(f"Triage {actual_triage} → class_shap shape: {class_shap.shape}")

    shap.summary_plot(
        class_shap,
        X_arr,
        feature_names=feature_names,
        plot_type="dot",
        max_display=5,
        plot_size=(12, 4),
        show=False
    )

    plt.suptitle(
        f"Top Features for Triage {actual_triage}",
        fontsize=16,
        fontweight='bold',
        y=0.95
    )
    plt.xlabel("mean(|SHAP|)", fontsize=12, fontweight='bold')
    plt.tight_layout()

    filename  = f"shap_top_features_for_triage_{actual_triage}.png"
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    print(f"Saved → {save_path}")
```

## **Global Feature Importance across all triage**


```python
import pandas as pd

X_arr = X_test_sample.values if isinstance(X_test_sample, pd.DataFrame) else np.array(X_test_sample)
feature_names = list(X_test_sample.columns) if isinstance(X_test_sample, pd.DataFrame) else None

n_classes = shap_values.shape[2]

# mapping SHAP indices to actual triage labels
triage_names = [f"Triage {cls}" for cls in model.classes_]

# creating global SHAP summary
shap.summary_plot(
    shap_values,
    X_arr,
    feature_names=feature_names,
    plot_type="bar",
    class_names=triage_names,
    max_display=15,
    plot_size=(12, 6),
    show=False
)

plt.suptitle(
    "Global Feature Importance Across Triage Classes",
    fontsize=18,
    fontweight='bold',
    y=1.02
)

plt.xlabel("mean(|SHAP|)", fontsize=13, fontweight='bold')
plt.tight_layout()

filename  = "global_XAI_feature_importance_across_all_triage.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

## **Patient-wise Waterfall Plot for each Class**
### **Triage 1**


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import random

X_arr          = X_test_sample.values if isinstance(X_test_sample, pd.DataFrame) else np.array(X_test_sample)
feature_names  = X_test_sample.columns.tolist() if isinstance(X_test_sample, pd.DataFrame) else None

class_id      = 0
actual_triage = model.classes_[class_id]

# keeping indices within the 200-row sample used for shap_values
random.seed(42)
patient_indices = random.sample(range(len(X_test_sample)), 4)

fig, axes = plt.subplots(2, 2, figsize=(40, 18))

fig.suptitle(
    f"Triage {actual_triage} Prediction Explanations",
    fontsize=42,
    fontweight="bold",
    y=1.04
)

for ax, patient_idx in zip(axes.flatten(), patient_indices):

    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        shap.Explanation(
            values      = shap_values[patient_idx, :, class_id],          # (13,)
            base_values = explainer.expected_value[class_id],
            data        = X_test_sample.iloc[patient_idx],                # aligned row
            feature_names = feature_names
        ),
        max_display=4,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=220)
    buf.seek(0)
    plt.close()

    img = mpimg.imread(buf)
    ax.imshow(img)
    ax.axis("off")

    true_label = y_test.iloc[patient_idx]
    pred_label = model.predict(X_test_sample.iloc[[patient_idx]])[0]
    correct    = "correct" if true_label == pred_label else "incorrect"

    ax.set_title(
        f"Patient {patient_idx} | True: Triage {true_label} | Pred: Triage {pred_label} ({correct})",
        fontsize=26,
        fontweight="bold",
        color="green" if correct == "correct" else "red",
        pad=14
    )

plt.tight_layout()

filename  = f"individual_patient_explanations_triage_{actual_triage}.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

### **Triage 2**


```python
class_id      = 1
actual_triage = model.classes_[class_id]

# keeping indices within the 200-row sample used for shap_values
random.seed(42)
patient_indices = random.sample(range(len(X_test_sample)), 4)

fig, axes = plt.subplots(2, 2, figsize=(40, 18))

fig.suptitle(
    f"Triage {actual_triage} Prediction Explanations",
    fontsize=42,
    fontweight="bold",
    y=1.04
)

for ax, patient_idx in zip(axes.flatten(), patient_indices):

    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        shap.Explanation(
            values      = shap_values[patient_idx, :, class_id],          # (13,)
            base_values = explainer.expected_value[class_id],
            data        = X_test_sample.iloc[patient_idx],                # aligned row
            feature_names = feature_names
        ),
        max_display=4,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=220)
    buf.seek(0)
    plt.close()

    img = mpimg.imread(buf)
    ax.imshow(img)
    ax.axis("off")

    true_label = y_test.iloc[patient_idx]
    pred_label = model.predict(X_test_sample.iloc[[patient_idx]])[0]
    correct    = "correct" if true_label == pred_label else "incorrect"

    ax.set_title(
        f"Patient {patient_idx} | True: Triage {true_label} | Pred: Triage {pred_label} ({correct})",
        fontsize=26,
        fontweight="bold",
        color="green" if correct == "correct" else "red",
        pad=14
    )

plt.tight_layout()

filename  = f"individual_patient_explanations_triage_{actual_triage}.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

### **Triage 3**


```python
class_id      = 2
actual_triage = model.classes_[class_id]

# keeping indices within the 200-row sample used for shap_values
random.seed(42)
patient_indices = random.sample(range(len(X_test_sample)), 4)

fig, axes = plt.subplots(2, 2, figsize=(40, 18))

fig.suptitle(
    f"Triage {actual_triage} Prediction Explanations",
    fontsize=42,
    fontweight="bold",
    y=1.04
)

for ax, patient_idx in zip(axes.flatten(), patient_indices):

    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        shap.Explanation(
            values      = shap_values[patient_idx, :, class_id],          # (13,)
            base_values = explainer.expected_value[class_id],
            data        = X_test_sample.iloc[patient_idx],                # aligned row
            feature_names = feature_names
        ),
        max_display=4,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=220)
    buf.seek(0)
    plt.close()

    img = mpimg.imread(buf)
    ax.imshow(img)
    ax.axis("off")

    true_label = y_test.iloc[patient_idx]
    pred_label = model.predict(X_test_sample.iloc[[patient_idx]])[0]
    correct    = "correct" if true_label == pred_label else "incorrect"

    ax.set_title(
        f"Patient {patient_idx} | True: Triage {true_label} | Pred: Triage {pred_label} ({correct})",
        fontsize=26,
        fontweight="bold",
        color="green" if correct == "correct" else "red",
        pad=14
    )

plt.tight_layout()

filename  = f"individual_patient_explanations_triage_{actual_triage}.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

### **Triage 4**


```python
class_id      = 3
actual_triage = model.classes_[class_id]

# keeping indices within the 200-row sample used for shap_values
random.seed(42)
patient_indices = random.sample(range(len(X_test_sample)), 4)

fig, axes = plt.subplots(2, 2, figsize=(40, 18))

fig.suptitle(
    f"Triage {actual_triage} Prediction Explanations",
    fontsize=42,
    fontweight="bold",
    y=1.04
)

for ax, patient_idx in zip(axes.flatten(), patient_indices):

    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        shap.Explanation(
            values      = shap_values[patient_idx, :, class_id],          # (13,)
            base_values = explainer.expected_value[class_id],
            data        = X_test_sample.iloc[patient_idx],                # aligned row
            feature_names = feature_names
        ),
        max_display=4,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=220)
    buf.seek(0)
    plt.close()

    img = mpimg.imread(buf)
    ax.imshow(img)
    ax.axis("off")

    true_label = y_test.iloc[patient_idx]
    pred_label = model.predict(X_test_sample.iloc[[patient_idx]])[0]
    correct    = "correct" if true_label == pred_label else "incorrect"

    ax.set_title(
        f"Patient {patient_idx} | True: Triage {true_label} | Pred: Triage {pred_label} ({correct})",
        fontsize=26,
        fontweight="bold",
        color="green" if correct == "correct" else "red",
        pad=14
    )

plt.tight_layout()

filename  = f"individual_patient_explanations_triage_{actual_triage}.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

### **Triage 5**


```python
class_id      = 4
actual_triage = model.classes_[class_id]

# keeping indices within the 200-row sample used for shap_values
random.seed(42)
patient_indices = random.sample(range(len(X_test_sample)), 4)

fig, axes = plt.subplots(2, 2, figsize=(40, 18))

fig.suptitle(
    f"Triage {actual_triage} Prediction Explanations",
    fontsize=42,
    fontweight="bold",
    y=1.04
)

for ax, patient_idx in zip(axes.flatten(), patient_indices):

    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        shap.Explanation(
            values      = shap_values[patient_idx, :, class_id],          # (13,)
            base_values = explainer.expected_value[class_id],
            data        = X_test_sample.iloc[patient_idx],                # aligned row
            feature_names = feature_names
        ),
        max_display=4,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=220)
    buf.seek(0)
    plt.close()

    img = mpimg.imread(buf)
    ax.imshow(img)
    ax.axis("off")

    true_label = y_test.iloc[patient_idx]
    pred_label = model.predict(X_test_sample.iloc[[patient_idx]])[0]
    correct    = "correct" if true_label == pred_label else "incorrect"

    ax.set_title(
        f"Patient {patient_idx} | True: Triage {true_label} | Pred: Triage {pred_label} ({correct})",
        fontsize=26,
        fontweight="bold",
        color="green" if correct == "correct" else "red",
        pad=14
    )

plt.tight_layout()

filename  = f"individual_patient_explanations_triage_{actual_triage}.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

---
# **`Section 2:` Chief Complaints**
---


```python
import pandas as pd

df = pd.read_csv("/kaggle/input/competitions/triagegeist/chief_complaints.csv")
df.head()
```

- **Getting shape**


```python
df.shape
```

## **Getting `Triage Acuity` data for each patient**


```python
data = pd.read_csv("/kaggle/input/competitions/triagegeist/train.csv")
data = data[['patient_id', 'triage_acuity']]
data.head()
```

## **Merging these data together**


```python
# merging on patient_id with inner join to remove mismatches
df = df.merge(
 data[["patient_id", "triage_acuity"]],
 on="patient_id",
 how="inner"
)

# optionally resetting index
df = df.reset_index(drop=True)
df = df[['chief_complaint_raw', 'triage_acuity']]
df.head()
```

# **Data Cleaning**
---

## **Handling `NaN`**


```python
df.isna().sum()
```

## **Handling `dulicates`**


```python
df.duplicated().sum()
```


```python
df = df.drop_duplicates()
df.shape
```

# **Data Preprocessing**
---

## **Creating Unique list of Complaints**

### **1. Constructing unique list by splitting as per `commas (,)`**


```python
import numpy as np

unique_complaints = (
 df["chief_complaint_raw"]
 .dropna()
 .str.lower()
 .str.split(",") # split by comma
 .explode() # flatten lists
 .str.strip() # remove spaces
 .replace("", np.nan) # remove empty strings
 .dropna()
 .unique()
)

unique_complaints = sorted(unique_complaints)

print(f"Total unique complaint tokens: {len(unique_complaints)}")
print(unique_complaints[:50])
```

### **2. Splitting against `connectors` inside the splits**


```python
import re
import numpy as np
import pandas as pd
import pickle
import os

# defining connector patterns
CONNECTORS = [
    r"\bwith\b",
    r"\bworsening\b"
]

pattern = "|".join(CONNECTORS)

def split_complaint(text):
    if pd.isna(text):
        return []
    
    # normalizing text
    text = text.lower().replace("，", ",")
    
    # removing unwanted phrase
    text = re.sub(r"\bin known patient\b", "", text)
    
    # splitting by comma
    parts = [p.strip() for p in text.split(",")]
    
    final_parts = []
    
    for part in parts:
        sub_parts = re.split(pattern, part)
        sub_parts = [s.strip() for s in sub_parts if s.strip()]
        final_parts.extend(sub_parts)
    
    return final_parts

# generating tokens
tokens = (
    df["chief_complaint_raw"]
    .dropna()
    .apply(split_complaint)
    .explode()
    .str.strip()
    .replace("", np.nan)
    .dropna()
    .unique()
)

tokens = sorted(tokens)

print(f"Total unique refined tokens: {len(tokens)}")
print(tokens[:50])

# saving tokens

# creating mappings
index_to_token = {i: token for i, token in enumerate(tokens)}
token_to_index = {token: i for i, token in enumerate(tokens)}

# saving directory
save_dir = "artifacts"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "complaint_token_mappings.pkl")

# saving both mappings
with open(save_path, "wb") as f:
    pickle.dump({
        "index_to_token": index_to_token,
        "token_to_index": token_to_index
    }, f)

print(f"Token mappings saved to {save_path}")
```

## **Creating Multi-Hot Encoding**

- **Global token space**


```python
token_list = sorted(tokens)

token_to_idx = {token: i for i, token in enumerate(token_list)}
idx_to_token = {i: token for token, i in token_to_idx.items()}

VOCAB_SIZE = len(token_list)
print("Vocab size:", VOCAB_SIZE)
```

- **Tokenizing complaints**


```python
import numpy as np

def encode_multihot(token_list_sample):
    vec = np.zeros(VOCAB_SIZE, dtype=np.uint8)
    
    for token in token_list_sample:
        idx = token_to_idx.get(token)
        if idx is not None:
            vec[idx] = 1
    
    return vec

df["token_list"] = df["chief_complaint_raw"].apply(split_complaint)

X_tokens = np.vstack(
    df["token_list"].apply(encode_multihot).values
)

print("Token feature shape:", X_tokens.shape)
```

- **Converting back to dataframe**


```python
token_columns = [f"{t}" for t in token_list]

X_tokens_df = pd.DataFrame(X_tokens, columns=token_columns)
X_tokens_df.head()
```

- **Combining triage data**


```python
X_tokens_df["triage_acuity"] = df["triage_acuity"].values

df = X_tokens_df.copy()
df.head()
```

## **Checking for duplicate entries**


```python
df.duplicated().sum()
```

- **Dropping duplicates**


```python
df = df.drop_duplicates()
df.shape
```

# **Exploratory Data Analysis**
---

## **Sparsity Analysis**


```python
sparsity = (df.drop(columns=["triage_acuity"]) == 0).mean().mean()
print(f"Sparsity: {sparsity:.4f}")
```

## **Complaint Frequency**


```python
import os
import matplotlib.pyplot as plt

feature_cols = df.columns.drop("triage_acuity")

freq = df[feature_cols].sum().sort_values(ascending=False)

plots_dir = "/kaggle/working/plots/complaints_data/"
os.makedirs(plots_dir, exist_ok=True)

freq.head(10).plot(kind="barh", figsize=(8, 3))
plt.title("Top 10 Most Frequent Complaints", fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()

filename = "top_10_most_frequent_complaints.png"
save_path = os.path.join(plots_dir, filename)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")

```

## **Rare Feature Detection**


```python
rare_features = freq[freq < 20] # threshold adjustable
print(f"Rare features (<20 occurrences): {len(rare_features)}")
```

## **Feature vs Target**


```python
import pandas as pd

target_means = df.groupby("triage_acuity")[feature_cols].mean()

target_means.T.sort_values(by=target_means.index.tolist(), ascending=False).head(5)
```

## **Class-wise Complaint Distribution**


```python
for cls in sorted(df["triage_acuity"].unique()):
 print(f"\nClass {cls}")
 print(df[df["triage_acuity"] == cls][feature_cols].sum().sort_values(ascending=False).head())
```

## **Feature Variance**


```python
variance = df[feature_cols].var().sort_values()

low_var = variance[variance < 0.01]

print(f"Low variance features: {len(low_var)}")
```

# **Feature Ranking**
---


```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier

plots_dir = "/kaggle/working/plots/complaints_data/"
results_dir = "/kaggle/working/results/complaints_data/"
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# preparing data
X = df.drop(columns=["triage_acuity"])
y = df["triage_acuity"]

# computing scores
chi_scores, _ = chi2(X, y)
chi_df = pd.Series(chi_scores, index=X.columns)

mi_df = pd.Series(mutual_info_classif(X, y, discrete_features=True), index=X.columns)

f_scores, _ = f_classif(X, y)
f_df = pd.Series(f_scores, index=X.columns)

var_df = X.var()

corr_df = X.corrwith(y, method="spearman").abs()

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

perm = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
perm_df = pd.Series(perm.importances_mean, index=X.columns)

# sorting for plotting
methods = {
 "Chi-Square": chi_df.sort_values(ascending=False),
 "Mutual Info": mi_df.sort_values(ascending=False),
 "ANOVA F-test": f_df.sort_values(ascending=False),
 "Variance": var_df.sort_values(ascending=False),
 "Spearman Corr": corr_df.sort_values(ascending=False),
 "Permutation": perm_df.sort_values(ascending=False)
}

# plotting raw scores
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
axes = axes.flatten()

for i, (name, series) in enumerate(methods.items()):
 top5 = series.head(5)
 axes[i].barh(top5.index[::-1], top5.values[::-1])
 axes[i].set_title(name, fontweight='bold')
 axes[i].set_xlabel("Score", fontweight='bold')

plt.tight_layout()
raw_plot_path = os.path.join(plots_dir, "feature_ranking_tests.png")
plt.savefig(raw_plot_path, bbox_inches='tight', dpi=300)
plt.show()
print(f"Feature ranking plot saved to {raw_plot_path}")

# defining normalization function
def normalize(s):
 return (s - s.min()) / (s.max() - s.min() + 1e-9)

# combining normalized scores
scores_df = pd.DataFrame({
 "chi2": normalize(chi_df),
 "mi": normalize(mi_df),
 "anova": normalize(f_df),
 "variance": normalize(var_df),
 "corr": normalize(corr_df),
 "perm": normalize(perm_df)
})

scores_df["combined"] = scores_df.mean(axis=1)
scores_df = scores_df.sort_values(by="combined", ascending=False)

scores_save_path = os.path.join(results_dir, "consensus_feature_ranking.csv")
scores_df.to_csv(scores_save_path, index=True)
print(f"Feature ranking results saved to {scores_save_path}")

# plotting normalized top 5
top5 = scores_df.head(5)

plt.figure(figsize=(10, 4))
plt.barh(top5.index[::-1], top5["combined"][::-1])
plt.xlabel("Normalized Combined Score", fontweight='bold')
plt.title("Top 5 Features (Consensus Ranking)", fontweight='bold')
plt.xlim(0, 0.85)

for i, v in enumerate(top5["combined"][::-1]):
 plt.text(v, i, f"{v:.3f}", va='center')

plt.tight_layout()
normalized_plot_path = os.path.join(plots_dir, "normalized_feature_ranking.png")
plt.savefig(normalized_plot_path, bbox_inches='tight', dpi=300)
plt.show()
print(f"Feature ranking plot saved to {normalized_plot_path}")
```

# **Modelling**
---

## **Test-Train Split**


```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["triage_acuity"])
y = df["triage_acuity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

## **Lazy Predict**


```python
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(
 verbose=1,
 ignore_warnings=True,
 custom_metric=None
)

models, predictions = clf.fit(X_train, X_test, y_train, y_test)
models_sorted = models.sort_values(by="F1 Score", ascending=False)
models_sorted.head(10)
```

## **Top Models Performance Comparison**


```python
import time
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    cohen_kappa_score
)
from sklearn.preprocessing import label_binarize

from sklearn.linear_model import Perceptron, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models_dict = {
    "Perceptron": Perceptron(),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True, kernel="rbf", C=1.0),
    "RidgeClassifier": RidgeClassifier(),
    "PassiveAggressive": PassiveAggressiveClassifier()
}

X_train_ = X_train.reset_index(drop=True).to_numpy()
y_train_ = y_train.reset_index(drop=True).to_numpy()

X_test_ = X_test.reset_index(drop=True).to_numpy()
y_test_ = y_test.reset_index(drop=True).to_numpy()

classes = np.unique(y_train_)
y_test_bin = label_binarize(y_test_, classes=classes)

SAVE_DIR = "/kaggle/working/complaints_cleaned_model"
os.makedirs(SAVE_DIR, exist_ok=True)

results = []

for name, model in models_dict.items():
    print(f"\nTraining {name}...")
    start = time.time()
    
    try:
        model.fit(X_train_, y_train_)
        y_pred = model.predict(X_test_)
        
        acc = accuracy_score(y_test_, y_pred)
        bal_acc = balanced_accuracy_score(y_test_, y_pred)
        f1 = f1_score(y_test_, y_pred, average="weighted")
        precision = precision_score(y_test_, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test_, y_pred, average="weighted", zero_division=0)
        qwk = cohen_kappa_score(y_test_, y_pred, weights="quadratic")
        
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_)
                roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
            else:
                roc_auc = np.nan
        except:
            roc_auc = np.nan

    except Exception as e:
        print(f"Failed for {name}: {e}")
        acc = bal_acc = f1 = precision = recall = roc_auc = qwk = np.nan
    
    end = time.time()
    
    model_path = os.path.join(SAVE_DIR, f"{name}.pkl")
    
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        size_kb = os.path.getsize(model_path) / 1024
    except:
        size_kb = np.nan
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "ROC AUC": roc_auc,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "QWK": qwk,
        "Time Taken": end - start,
        "Model Size (KB)": size_kb
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="QWK", ascending=False).reset_index(drop=True)

print("\nFinal Results:")
display(results_df)

results_df.to_csv(
    os.path.join(SAVE_DIR, "model_comparison_with_size.csv"),
    index=False
)
```

- **Visualizing performance**


```python
import os
import matplotlib.pyplot as plt
import numpy as np

plots_dir = "/kaggle/working/plots/complaints_data/"
os.makedirs(plots_dir, exist_ok=True)

selected_models = ["Perceptron", "RidgeClassifier", "PassiveAggressive"]
df_plot = results_df[results_df["Model"].isin(selected_models)].copy()

name_map = {
    "Perceptron": "Percep",
    "RidgeClassifier": "Ridge",
    "PassiveAggressive": "PA"
}

df_plot["Short_Model"] = df_plot["Model"].map(name_map).fillna(df_plot["Model"])

df_plot = df_plot.sort_values(by="QWK", ascending=False)

models = df_plot["Short_Model"]
size = df_plot["Model Size (KB)"].fillna(0)
qwk = df_plot["QWK"].fillna(0)

x = np.arange(len(models))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(x, size)
axes[0].set_title("Model Size (KB)", fontweight='bold')
axes[0].set_xlabel("Models", fontweight='bold')
axes[0].set_ylabel("Size (KB)", fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].set_yscale('log')
axes[0].set_ylim(16, 16.35)

for i, v in enumerate(size):
    if v > 0:
        axes[0].text(i, v, f"{v:.1f}", ha='center', va='bottom', fontsize=9)

axes[1].bar(x, qwk)
axes[1].set_title("QWK", fontweight='bold')
axes[1].set_xlabel("Models", fontweight='bold')
axes[1].set_ylabel("QWK", fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].set_ylim(0.99, 1.01)

for i, v in enumerate(qwk):
    axes[1].text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()

save_path = os.path.join(plots_dir, "model_size_vs_qwk.png")
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

## **Perceptron Classifier**


```python
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score
)

plots_dir = "/kaggle/working/plots/complaints_data/"
results_dir = "/kaggle/working/results/complaints_data/"
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

model = Perceptron()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

labels = model.classes_

qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
print(f"QWK: {qwk:.4f}")

report_dict = classification_report(y_test, y_pred, labels=labels, digits=4, output_dict=True)
print("Classification Report (Perceptron)")
print(classification_report(y_test, y_pred, labels=labels, digits=4))

report_df = pd.DataFrame(report_dict).transpose()
report_df.loc["QWK", :] = [qwk] + [None]*(report_df.shape[1]-1)

report_save_path = os.path.join(results_dir, "perceptron_classification_report.csv")
report_df.to_csv(report_save_path, index=True)
print(f"Classification report saved to {report_save_path}")

cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_save_path = os.path.join(results_dir, "perceptron_confusion_matrix.csv")
cm_df.to_csv(cm_save_path, index=True)
print(f"Confusion matrix values saved to {cm_save_path}")

fig, ax = plt.subplots(figsize=(5, 4))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[l for l in labels]
)

disp.plot(cmap='inferno', ax=ax, colorbar=True)

ax.set_title("Confusion Matrix - Perceptron", fontweight='bold')
ax.set_xlabel("Predicted Triage", fontweight='bold')
ax.set_ylabel("True Triage", fontweight='bold')

plt.tight_layout()
plot_save_path = os.path.join(plots_dir, "perceptron_confusion_matrix.png")
plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {plot_save_path}")
```

### **PC - Cross Validation**


```python
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score
)

plots_dir = "/kaggle/working/plots/complaints_data/"
results_dir = "/kaggle/working/results/complaints_data/"
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

model = Perceptron()

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

y_pred_cv = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

labels = sorted(y.unique())

qwk = cohen_kappa_score(y, y_pred_cv, weights="quadratic")
print(f"QWK (10-Fold CV): {qwk:.4f}")

report_dict = classification_report(y, y_pred_cv, labels=labels, digits=4, output_dict=True)
print("=== 10-Fold CV Classification Report (Perceptron) ===")
print(classification_report(y, y_pred_cv, labels=labels, digits=4))

report_df = pd.DataFrame(report_dict).transpose()
report_df.loc["QWK", :] = [qwk] + [None]*(report_df.shape[1]-1)

report_save_path = os.path.join(results_dir, "perceptron_10_fold_cv_classification_report.csv")
report_df.to_csv(report_save_path, index=True)
print(f"Classification report saved to {report_save_path}")

cm = confusion_matrix(y, y_pred_cv, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_save_path = os.path.join(results_dir, "perceptron_10_fold_cv_confusion_matrix.csv")
cm_df.to_csv(cm_save_path, index=True)
print(f"Confusion matrix values saved to {cm_save_path}")

fig, ax = plt.subplots(figsize=(5, 4))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[l for l in labels]
)

disp.plot(cmap='inferno', ax=ax, colorbar=True)

ax.set_title("Confusion Matrix - Perceptron (10-Fold CV)", fontweight='bold')
ax.set_xlabel("Predicted Triage", fontweight='bold')
ax.set_ylabel("True Triage", fontweight='bold')

plt.tight_layout()
plot_save_path = os.path.join(plots_dir, "perceptron_10_fold_cv_confusion_matrix.png")
plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {plot_save_path}")
```

**`NOTE:` Since the performance is very high, we will check for data leakage or anything similar**

# **Checking for data leakage**
---

## **1. Train/Test Overlap Detection**


```python
import pandas as pd

train_df = pd.DataFrame(X_train).assign(_split="train")
test_df = pd.DataFrame(X_test).assign(_split="test")

dupes = pd.concat([train_df, test_df]).duplicated(
 subset=train_df.columns[:-1], keep=False
).sum()

print("TEST 1 — Train/test overlap")
print(f"Duplicate rows across splits: {dupes}")
print("FAIL — leakage likely\n" if dupes > 0 else "PASS\n")
```

### **Making non-overlapping splits**


```python
# splitting data
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42, stratify=y
)

# converting to DataFrame for safe row-wise comparison
X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
X_test_df = pd.DataFrame(X_test).reset_index(drop=True)

y_train = pd.Series(y_train).reset_index(drop=True)
y_test = pd.Series(y_test).reset_index(drop=True)

# creating set of training rows
train_rows = set(map(tuple, X_train_df.values))

# identifying overlapping rows in test data
overlap_mask = X_test_df.apply(lambda row: tuple(row) in train_rows, axis=1)

num_duplicates = overlap_mask.sum()

# removing duplicates from test set
X_test_clean = X_test_df[~overlap_mask].reset_index(drop=True)
y_test_clean = y_test[~overlap_mask].reset_index(drop=True)

# reporting results
print(f"Duplicates removed from test set: {num_duplicates}")
print(f"Original test size: {len(X_test_df)}")
print(f"New test size: {len(X_test_clean)}")
```

### **Re-training**


```python
import time
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    cohen_kappa_score
)
from sklearn.preprocessing import label_binarize

from sklearn.linear_model import Perceptron, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models_dict = { 
    "Perceptron": Perceptron(),
    "RidgeClassifier": RidgeClassifier(),
    "PassiveAggressive": PassiveAggressiveClassifier(),
    "SVC": SVC(probability=True, kernel="rbf", C=1.0, gamma="scale"),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

X_train_ = X_train.reset_index(drop=True)
y_train_ = y_train.reset_index(drop=True)

X_test_ = X_test_clean.reset_index(drop=True)
y_test_ = y_test_clean.reset_index(drop=True)

classes = np.unique(y_train_)
y_test_bin = label_binarize(y_test_, classes=classes)

SAVE_DIR = "/kaggle/working/complaints_cleaned_model"
os.makedirs(SAVE_DIR, exist_ok=True)

results = []

for name, model in models_dict.items():
    
    print(f"\nTraining {name}...")
    start = time.time()
    
    try:
        model.fit(X_train_, y_train_)
        y_pred = model.predict(X_test_)
        
        acc = accuracy_score(y_test_, y_pred)
        bal_acc = balanced_accuracy_score(y_test_, y_pred)
        f1 = f1_score(y_test_, y_pred, average="weighted")
        precision = precision_score(y_test_, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test_, y_pred, average="weighted", zero_division=0)
        qwk = cohen_kappa_score(y_test_, y_pred, weights="quadratic")
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_)
            roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
        else:
            roc_auc = np.nan
    
    except Exception as e:
        print(f"Failed for {name}: {e}")
        acc = bal_acc = f1 = precision = recall = roc_auc = qwk = np.nan
    
    end = time.time()
    
    model_path = os.path.join(SAVE_DIR, f"{name}.pkl")
    
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        size_kb = os.path.getsize(model_path) / 1024
    except:
        size_kb = np.nan
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "ROC AUC": roc_auc,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "QWK": qwk,
        "Time Taken": end - start,
        "Model Size (KB)": size_kb
    })

results_df = pd.DataFrame(results)

if "Accuracy" in results_df.columns:
    results_df = results_df.sort_values(by="QWK", ascending=False)

print("\nFinal Results:")
display(results_df)

results_df.to_csv(
    os.path.join(SAVE_DIR, "model_comparison_with_size.csv"),
    index=False
)
```

## **Accuracy Gap (Overfitting/Leakage Signal)**


```python
from sklearn.linear_model import Perceptron

# defining and training model
model = Perceptron()
model.fit(X_train, y_train)

# evaluating model
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
gap = train_acc - test_acc

print("TEST 2 — Suspiciously high accuracy")
print(f"Train accuracy : {train_acc:.4f}")
print(f"Test accuracy : {test_acc:.4f}")
print(f"Gap : {gap:.4f}")
print("WARN — gap > 0.05, check for overfitting/leakage\n" if gap > 0.05 else "PASS")
```

## **Feature Importance Dominance Check**


```python
print("TEST 3 — Feature importance dominance")

if hasattr(model, "feature_importances_"):
 importances = model.feature_importances_
 top_share = importances.max()
 top_idx = importances.argmax()
 top_name = X_train.columns[top_idx] if hasattr(X_train, "columns") else top_idx

 print(f"Top feature: '{top_name}' accounts for {top_share:.1%} of importance")
 print("WARN — single feature dominates, investigate\n" if top_share > 0.8 else "PASS\n")
else:
 print("SKIP — model has no feature_importances_ attribute\n")
```

## **Shuffle Test (Sanity Check)**


```python
import numpy as np
from sklearn.base import clone

shuffled_model = clone(model)

y_shuffled = y_train.sample(frac=1, random_state=42).reset_index(drop=True)
np.random.shuffle(y_shuffled)

shuffled_model.fit(X_train, y_shuffled)

shuffle_acc = shuffled_model.score(X_test, y_test)
random_baseline = 1 / len(np.unique(y_test))

print("TEST 4 — Shuffle test (y scrambled)")
print(f"Accuracy on scrambled labels: {shuffle_acc:.4f}")
print(f"Random baseline : {random_baseline:.4f}")

print(
 "FAIL — model still beats random on scrambled data"
 if shuffle_acc > random_baseline + 0.05
 else "PASS"
)
```

# **XAI Explainations**
---

## **Class-wise top 5 important features**

### **1. Population-level feature importance**


```python
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import Perceptron

plots_dir = "/kaggle/working/plots/complaints_data/"
os.makedirs(plots_dir, exist_ok=True)

# training perceptron
model = Perceptron()
model.fit(X_train, y_train)

# creating SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# computing SHAP values
shap_vals = shap_values.values  # shape: (samples, features) OR (samples, features, classes)

# handling both binary and multiclass cases
if len(shap_vals.shape) == 2:
    shap_vals = shap_vals[:, :, np.newaxis]

n_classes = shap_vals.shape[2]

# mapping to actual triage labels
triage_labels = model.classes_

# selecting class-wise top features
for class_idx in range(n_classes):
    
    actual_class = triage_labels[class_idx]

    shap.summary_plot(
        shap_vals[:, :, class_idx],
        X_test,
        plot_type="bar",
        max_display=5,
        plot_size=(12, 4),
        show=False
    )

    plt.suptitle(
        f"Top Features for Triage {actual_class}",
        fontsize=16,
        fontweight='bold',
        y=0.95
    )

    plt.xlabel("mean(|SHAP|)", fontsize=12, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(plots_dir, f"perceptron_shap_top_features_class_{actual_class}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    print(f"Plot saved to {save_path}")
```

### **2. Peak impact per patient**


```python
import os

plots_dir = "/kaggle/working/plots/complaints_data/"
os.makedirs(plots_dir, exist_ok=True)

for class_idx in range(n_classes):
 actual_class = triage_labels[class_idx]

 shap.summary_plot(
 shap_vals[:, :, class_idx],
 X_test,
 plot_type="dot",
 max_display=5,
 plot_size=(12, 4),
 show=False
 )

 plt.suptitle(
 f"Top Features for Triage {actual_class}",
 fontsize=16,
 fontweight='bold',
 y=0.95
 )

 plt.xlabel("mean(|SHAP|)", fontsize=12, fontweight='bold')
 plt.tight_layout()

 save_path = os.path.join(plots_dir, f"beeswarm_top_features_for_class_{actual_class}.png")
 plt.savefig(save_path, bbox_inches='tight', dpi=300)
 plt.show()

 print(f"Plot saved to {save_path}")

```

## **Global Feature Importance across all triage**


```python
import os

plots_dir = "/kaggle/working/plots/complaints_data/"
os.makedirs(plots_dir, exist_ok=True)

class_names = [str(i) for i in range(5)]

shap.summary_plot(
 [shap_vals[:, :, i] for i in range(5)],
 X_test,
 plot_type="bar",
 class_names=class_names,
 max_display=15,
 plot_size=(12, 6),
 show=False
)

plt.suptitle(
 "Global feature importance across triage classes",
 fontsize=16,
 fontweight='bold',
 y=1
)

plt.xlabel("mean(|SHAP|)", fontsize=13, fontweight='bold')
plt.tight_layout()

save_path = os.path.join(plots_dir, "global_XAI_feature_importance_across_all_triage.png")
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

## **Patient-wise Waterfall Plot for each Class**

### **Triage 1**


```python
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import random

plots_dir = "/kaggle/working/plots/complaints_data/"
os.makedirs(plots_dir, exist_ok=True)

class_id = 0
actual_class = model.classes_[class_id]

random.seed(42)
patient_indices = random.sample(range(len(X_test)), 4)

fig, axes = plt.subplots(2, 2, figsize=(40, 18))

fig.suptitle(
    f"Triage {actual_class} Prediction Explanations",
    fontsize=42,
    fontweight="bold",
    y=1.02
)

for ax, patient_idx in zip(axes.flatten(), patient_indices):
    
    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[patient_idx, :, class_id],
            base_values=explainer.expected_value[class_id] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value,
            data=X_test.iloc[patient_idx],
            feature_names=X_test.columns.tolist()
        ),
        max_display=4,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    plt.close()

    img = mpimg.imread(buf)

    ax.imshow(img)
    ax.axis("off")

    true_label = y_test.iloc[patient_idx]
    pred_label = model.predict(X_test.iloc[[patient_idx]].values)[0]
    correct = "correct" if true_label == pred_label else "incorrect"

    ax.set_title(
        f"Patient {patient_idx} | True: {true_label} | Pred: {pred_label} ({correct})",
        fontsize=25,
        fontweight="bold",
        color="green" if correct == "correct" else "red",
        pad=12
    )

plt.tight_layout()

save_path = os.path.join(
    plots_dir,
    f"individual_patient_explanations_triage_{actual_class}.png"
)

plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

### **Triage 2**


```python
class_id = 1
actual_class = model.classes_[class_id]

random.seed(42)
patient_indices = random.sample(range(len(X_test)), 4)

fig, axes = plt.subplots(2, 2, figsize=(40, 18))

fig.suptitle(
    f"Triage {actual_class} Prediction Explanations",
    fontsize=42,
    fontweight="bold",
    y=1.02
)

for ax, patient_idx in zip(axes.flatten(), patient_indices):
    
    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[patient_idx, :, class_id],
            base_values=explainer.expected_value[class_id] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value,
            data=X_test.iloc[patient_idx],
            feature_names=X_test.columns.tolist()
        ),
        max_display=4,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    plt.close()

    img = mpimg.imread(buf)

    ax.imshow(img)
    ax.axis("off")

    true_label = y_test.iloc[patient_idx]
    pred_label = model.predict(X_test.iloc[[patient_idx]].values)[0]
    correct = "correct" if true_label == pred_label else "incorrect"

    ax.set_title(
        f"Patient {patient_idx} | True: {true_label} | Pred: {pred_label} ({correct})",
        fontsize=25,
        fontweight="bold",
        color="green" if correct == "correct" else "red",
        pad=12
    )

plt.tight_layout()

save_path = os.path.join(
    plots_dir,
    f"individual_patient_explanations_triage_{actual_class}.png"
)

plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

### **Triage 3**


```python
class_id = 2
actual_class = model.classes_[class_id]

random.seed(42)
patient_indices = random.sample(range(len(X_test)), 4)

fig, axes = plt.subplots(2, 2, figsize=(40, 18))

fig.suptitle(
    f"Triage {actual_class} Prediction Explanations",
    fontsize=42,
    fontweight="bold",
    y=1.02
)

for ax, patient_idx in zip(axes.flatten(), patient_indices):
    
    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[patient_idx, :, class_id],
            base_values=explainer.expected_value[class_id] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value,
            data=X_test.iloc[patient_idx],
            feature_names=X_test.columns.tolist()
        ),
        max_display=4,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    plt.close()

    img = mpimg.imread(buf)

    ax.imshow(img)
    ax.axis("off")

    true_label = y_test.iloc[patient_idx]
    pred_label = model.predict(X_test.iloc[[patient_idx]].values)[0]
    correct = "correct" if true_label == pred_label else "incorrect"

    ax.set_title(
        f"Patient {patient_idx} | True: {true_label} | Pred: {pred_label} ({correct})",
        fontsize=25,
        fontweight="bold",
        color="green" if correct == "correct" else "red",
        pad=12
    )

plt.tight_layout()

save_path = os.path.join(
    plots_dir,
    f"individual_patient_explanations_triage_{actual_class}.png"
)

plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

### **Triage 4**


```python
class_id = 3
actual_class = model.classes_[class_id]

random.seed(42)
patient_indices = random.sample(range(len(X_test)), 4)

fig, axes = plt.subplots(2, 2, figsize=(40, 18))

fig.suptitle(
    f"Triage {actual_class} Prediction Explanations",
    fontsize=42,
    fontweight="bold",
    y=1.02
)

for ax, patient_idx in zip(axes.flatten(), patient_indices):
    
    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[patient_idx, :, class_id],
            base_values=explainer.expected_value[class_id] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value,
            data=X_test.iloc[patient_idx],
            feature_names=X_test.columns.tolist()
        ),
        max_display=4,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    plt.close()

    img = mpimg.imread(buf)

    ax.imshow(img)
    ax.axis("off")

    true_label = y_test.iloc[patient_idx]
    pred_label = model.predict(X_test.iloc[[patient_idx]].values)[0]
    correct = "correct" if true_label == pred_label else "incorrect"

    ax.set_title(
        f"Patient {patient_idx} | True: {true_label} | Pred: {pred_label} ({correct})",
        fontsize=25,
        fontweight="bold",
        color="green" if correct == "correct" else "red",
        pad=12
    )

plt.tight_layout()

save_path = os.path.join(
    plots_dir,
    f"individual_patient_explanations_triage_{actual_class}.png"
)

plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

### **Triage 5**


```python
class_id = 4
actual_class = model.classes_[class_id]

random.seed(42)
patient_indices = random.sample(range(len(X_test)), 4)

fig, axes = plt.subplots(2, 2, figsize=(40, 18))

fig.suptitle(
    f"Triage {actual_class} Prediction Explanations",
    fontsize=42,
    fontweight="bold",
    y=1.02
)

for ax, patient_idx in zip(axes.flatten(), patient_indices):
    
    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[patient_idx, :, class_id],
            base_values=explainer.expected_value[class_id] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value,
            data=X_test.iloc[patient_idx],
            feature_names=X_test.columns.tolist()
        ),
        max_display=4,
        show=False
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    plt.close()

    img = mpimg.imread(buf)

    ax.imshow(img)
    ax.axis("off")

    true_label = y_test.iloc[patient_idx]
    pred_label = model.predict(X_test.iloc[[patient_idx]].values)[0]
    correct = "correct" if true_label == pred_label else "incorrect"

    ax.set_title(
        f"Patient {patient_idx} | True: {true_label} | Pred: {pred_label} ({correct})",
        fontsize=25,
        fontweight="bold",
        color="green" if correct == "correct" else "red",
        pad=12
    )

plt.tight_layout()

save_path = os.path.join(
    plots_dir,
    f"individual_patient_explanations_triage_{actual_class}.png"
)

plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

print(f"Plot saved to {save_path}")
```

---
# **`Section 3:` Patient History**
---


```python
import pandas as pd

df = pd.read_csv("/kaggle/input/competitions/triagegeist/patient_history.csv")
df.head()
```

# **Getting `Triage Acuity` data for each patient**
---


```python
data = pd.read_csv("/kaggle/input/competitions/triagegeist/train.csv")
data = data[['patient_id', 'triage_acuity']]
data.head()
```

- **Merging them together**


```python
# merging on patient_id with inner join to remove mismatches
df = df.merge(
 data[["patient_id", "triage_acuity"]],
 on="patient_id",
 how="inner"
)

# optionally resetting index
df = df.reset_index(drop=True)
df = df.drop('patient_id', axis=1)
df.head()
```

# **Data Cleaning**
---

## **Handling `NaN`**


```python
df.isna().sum()
```

## **Handling `duplicates`**


```python
df.duplicated().sum()
```


```python
df = df.drop_duplicates()
df.reset_index(drop=True)
df.head()
```

# **Feature Ranking**
---


```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier

plots_dir = "/kaggle/working/plots/patient_history/"
results_dir = "/kaggle/working/results/patient_history/"
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# preparing data
X = df.drop(columns=["triage_acuity"])
y = df["triage_acuity"]

# computing scores
chi_scores, _ = chi2(X, y)
chi_df = pd.Series(chi_scores, index=X.columns)

mi_df = pd.Series(mutual_info_classif(X, y, discrete_features=True), index=X.columns)

f_scores, _ = f_classif(X, y)
f_df = pd.Series(f_scores, index=X.columns)

var_df = X.var()

corr_df = X.corrwith(y, method="spearman").abs()

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

perm = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
perm_df = pd.Series(perm.importances_mean, index=X.columns)

# sorting for plotting
methods = {
 "Chi-Square": chi_df.sort_values(ascending=False),
 "Mutual Info": mi_df.sort_values(ascending=False),
 "ANOVA F-test": f_df.sort_values(ascending=False),
 "Variance": var_df.sort_values(ascending=False),
 "Spearman Corr": corr_df.sort_values(ascending=False),
 "Permutation": perm_df.sort_values(ascending=False)
}

# plotting raw scores
fig, axes = plt.subplots(3, 2, figsize=(16, 10))
axes = axes.flatten()

for i, (name, series) in enumerate(methods.items()):
 top5 = series.head(5)
 axes[i].barh(top5.index[::-1], top5.values[::-1])
 axes[i].set_title(name, fontweight='bold')
 axes[i].set_xlabel("Score", fontweight='bold')

plt.tight_layout()
raw_plot_path = os.path.join(plots_dir, "feature_ranking_tests.png")
plt.savefig(raw_plot_path, bbox_inches='tight', dpi=300)
plt.show()
print(f"Feature ranking plot saved to {raw_plot_path}")

# defining normalization function
def normalize(s):
 return (s - s.min()) / (s.max() - s.min() + 1e-9)

# combining normalized scores
scores_df = pd.DataFrame({
 "chi2": normalize(chi_df),
 "mi": normalize(mi_df),
 "anova": normalize(f_df),
 "variance": normalize(var_df),
 "corr": normalize(corr_df),
 "perm": normalize(perm_df)
})

scores_df["combined"] = scores_df.mean(axis=1)
scores_df = scores_df.sort_values(by="combined", ascending=False)

scores_save_path = os.path.join(results_dir, "consensus_feature_ranking.csv")
scores_df.to_csv(scores_save_path, index=True)
print(f"Feature ranking results saved to {scores_save_path}")

# plotting normalized top 5
top5 = scores_df.head(5)

plt.figure(figsize=(10, 4))
plt.barh(top5.index[::-1], top5["combined"][::-1])
plt.xlabel("Normalized Combined Score", fontweight='bold')
plt.title("Top 5 Features (Consensus Ranking)", fontweight='bold')
plt.xlim(0, 1.1)

for i, v in enumerate(top5["combined"][::-1]):
 plt.text(v, i, f"{v:.3f}", va='center')

plt.tight_layout()
normalized_plot_path = os.path.join(plots_dir, "normalized_feature_ranking.png")
plt.savefig(normalized_plot_path, bbox_inches='tight', dpi=300)
plt.show()
print(f"Feature ranking plot saved to {normalized_plot_path}")

```

# **Modelling**
---


```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["triage_acuity"])
y = df["triage_acuity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```


```python
import os
import pandas as pd
import traceback

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

results_dir = "/kaggle/working/results/patient_history/"
os.makedirs(results_dir, exist_ok=True)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RidgeClassifier": RidgeClassifier(),
    "SGDClassifier": SGDClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "ExtraTree": ExtraTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(),
    "KNN": KNeighborsClassifier(n_jobs=-1),
    "LinearSVC": LinearSVC(),
    "SVC": SVC(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "MLP": MLPClassifier(max_iter=200),
}

results = []

for name, model in models.items():
    print(f"START: {name}")

    try:
        model.fit(X_train, y_train)
        print(f"FIT DONE: {name}")

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

        results.append((name, acc, f1, qwk))
        print(f"DONE: {name} | Acc: {acc:.4f} | F1: {f1:.4f} | QWK: {qwk:.4f}")

    except Exception as e:
        print(f"FAILED: {name}")
        print("Error:", str(e))
        traceback.print_exc()

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1", "QWK"])
results_df = results_df.sort_values(by="QWK", ascending=False)

print("=== FINAL RESULTS ===")
display(results_df)

results_save_path = os.path.join(results_dir, "manual_model_comparison_results.csv")
results_df.to_csv(results_save_path, index=False)
print(f"Model comparison results saved to {results_save_path}")
```

# **XAI Explainations are not at all reliable in here!**
---

# **`SECTION 4:` Unified Dataset**
---

- **Importing datasets**


```python
import pandas as pd

df_train = pd.read_csv("/kaggle/input/competitions/triagegeist/train.csv")
print(f"Training dataset shape: {df_train.shape}")

df_complaints = pd.read_csv("/kaggle/input/competitions/triagegeist/chief_complaints.csv")
df_complaints = df_complaints[['patient_id', 'chief_complaint_raw']]
print(f"Complaints dataset shape: {df_complaints.shape}")

df_history = pd.read_csv("/kaggle/input/competitions/triagegeist/patient_history.csv")
print(f"History dataset shape: {df_history.shape}")
```


```python
print(f"Training Data original columns: {df_train.columns}")
print(f"History Data columns: {df_history.columns}")
```

- **Merging `training` data and `history` data on `patient_id`**


```python
df = df_train.merge(df_history, on="patient_id", how="left")
df.shape
```

# **Selecting only required columns**
---


```python
df = df[['patient_id', 'mental_status_triage', 'num_prior_ed_visits_12m', 'num_prior_admissions_12m', 'num_active_medications', 
 'systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 'gcs_total', 'pain_score', 'shock_index', 'triage_acuity',

 'hx_hypertension', 'hx_diabetes_type2', 'hx_diabetes_type1', 'hx_asthma', 'hx_copd', 'hx_heart_failure', 'hx_atrial_fibrillation', 'hx_ckd', 'hx_liver_disease', 'hx_malignancy', 'hx_obesity', 'hx_depression', 'hx_anxiety', 'hx_dementia', 'hx_epilepsy', 'hx_hypothyroidism', 'hx_hyperthyroidism', 'hx_hiv', 'hx_coagulopathy', 'hx_immunosuppressed', 'hx_pregnant', 'hx_substance_use_disorder', 'hx_coronary_artery_disease', 'hx_stroke_prior', 'hx_peripheral_vascular_disease'
]]

df.shape
```

# **Data Cleaning**
---

## **Handling `NaN` values**


```python
na_counts = df.isna().sum()
na_counts = na_counts[na_counts > 0].sort_values(ascending=False)

print(na_counts)
```

- **Since these are the same `NaN` values from train dataset, we fill them with `-1` as we did earlier.**


```python
df = df.fillna(-1).reset_index(drop=True)
df.head()
```

## **Handling `duplicates`**

- **Removing duplicate entries**


```python
df.duplicated().sum()
```

# **Data Preparation**
---

## **Loading complaint token dictionary**


```python
import os
import pickle

mapping_path = os.path.join("artifacts", "complaint_token_mappings.pkl")

with open(mapping_path, "rb") as f:
 complaint_mappings = pickle.load(f)

index_to_token = complaint_mappings["index_to_token"]
token_to_index = complaint_mappings["token_to_index"]

print(f"Loaded {len(token_to_index)} complaint tokens")
```

## **Creating empty columns**


```python
import numpy as np
import pandas as pd

token_cols = [f"cc_{token}" for token in token_to_index.keys()]
token_cols = [col for col in token_cols if col not in df.columns]

empty_token_df = pd.DataFrame(
 np.nan,
 index=df.index,
 columns=token_cols
)

df = pd.concat([df, empty_token_df], axis=1)

print(f"Added {len(token_cols)} new token columns to df")
df.head()

```


```python
import re
import numpy as np
import pandas as pd

# applying same connector logic as before
CONNECTORS = [
    r"\bwith\b",
    r"\bworsening\b"
]
pattern = "|".join(CONNECTORS)


def split_complaint(text):
    if pd.isna(text):
        return []

    text = str(text).lower().replace("，", ",")
    text = re.sub(r"\bin known patient\b", "", text)

    parts = [p.strip() for p in text.split(",")]
    final_parts = []

    for part in parts:
        sub_parts = re.split(pattern, part)
        sub_parts = [s.strip() for s in sub_parts if s.strip()]
        final_parts.extend(sub_parts)

    return final_parts


def get_token_columns(token_to_index, prefix="cc_"):
    return [f"{prefix}{token}" for token in token_to_index.keys()]


def ensure_token_columns(df, token_to_index, prefix="cc_"):
    token_cols = get_token_columns(token_to_index, prefix=prefix)
    missing_cols = [col for col in token_cols if col not in df.columns]

    if missing_cols:
        empty_df = pd.DataFrame(np.nan, index=df.index, columns=missing_cols)
        df = pd.concat([df, empty_df], axis=1)

    return df


def build_patient_token_lookup(df_complaints, token_to_index):
    patient_token_map = {}

    for patient_id, complaint_text in zip(
        df_complaints["patient_id"],
        df_complaints["chief_complaint_raw"]
    ):
        tokens = split_complaint(complaint_text)
        valid_tokens = [t for t in tokens if t in token_to_index]

        if patient_id not in patient_token_map:
            patient_token_map[patient_id] = set()

        patient_token_map[patient_id].update(valid_tokens)

    return patient_token_map


def populate_complaint_multihot(df, df_complaints, token_to_index, prefix="cc_"):
    df = ensure_token_columns(df, token_to_index, prefix=prefix)

    token_cols = [f"{prefix}{token}" for token in token_to_index.keys()]
    patient_token_map = build_patient_token_lookup(df_complaints, token_to_index)

    # starting with all zeros
    df[token_cols] = 0

    # filling ones for tokens present for each patient
    for idx, patient_id in df["patient_id"].items():
        tokens = patient_token_map.get(patient_id, set())
        if tokens:
            cols_to_fill = [f"{prefix}{token}" for token in tokens]
            df.loc[idx, cols_to_fill] = 1

    return df
```


```python
df = populate_complaint_multihot(
    df=df,
    df_complaints=df_complaints,
    token_to_index=token_to_index,
    prefix="cc_"
)

df.head()
```

## **Getting feature types**


```python
import numpy as np
import pandas as pd

exclude_cols = ["patient_id", "triage_acuity"]

candidate_cols = [col for col in df.columns if col not in exclude_cols]

binary_cols = [
 col for col in candidate_cols
 if set(df[col].dropna().unique()).issubset({0, 1})
]

string_cols = [
 col for col in candidate_cols
 if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col])
]

numeric_cols = [
 col for col in candidate_cols
 if pd.api.types.is_numeric_dtype(df[col]) and col not in binary_cols
]

print("Binary 0/1 columns:")
print(binary_cols)
print("Binary:", len(binary_cols))

print("\nString columns:")
print(string_cols)
print("String:", len(string_cols))

print("\nNumerical columns (excluding patient_id and binary columns):")
print(numeric_cols)
print("Numerical:", len(numeric_cols))
```

## **Splitting Data**


```python
from sklearn.model_selection import train_test_split

# defining target
target_col = "triage_acuity"

# defining column groups
exclude_cols = ["patient_id", target_col]

candidate_cols = [col for col in df.columns if col not in exclude_cols]

binary_cols = [
    col for col in candidate_cols
    if set(df[col].dropna().unique()).issubset({0, 1})
]

string_cols = [
    col for col in candidate_cols
    if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col])
]

numeric_cols = [
    col for col in candidate_cols
    if pd.api.types.is_numeric_dtype(df[col]) and col not in binary_cols
]

# defining X and y
X = df.drop(columns=["patient_id", target_col]).copy()
y = df[target_col].copy()

# performing train-test split first
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

## **One Hot Encoding**


```python
from sklearn.preprocessing import OneHotEncoder

# initializing encoder
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# ensuring string type
X_train[string_cols] = X_train[string_cols].astype(str)
X_test[string_cols] = X_test[string_cols].astype(str)

# fitting on training data and transforming both sets
X_train_encoded = ohe.fit_transform(X_train[string_cols])
X_test_encoded = ohe.transform(X_test[string_cols])

# getting new column names
encoded_cols = ohe.get_feature_names_out(string_cols)

# converting to DataFrames
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)

# dropping original categorical columns
X_train = X_train.drop(columns=string_cols)
X_test = X_test.drop(columns=string_cols)

# concatenating encoded columns
X_train = pd.concat([X_train, X_train_encoded], axis=1)
X_test = pd.concat([X_test, X_test_encoded], axis=1)
```

## **Standard Scaling**


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

if numeric_cols:
 X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
 X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
```

## **Saving `Scalers` and `Encoders`**


```python
import os
import pickle

save_dir = "artifacts/preprocessing"
os.makedirs(save_dir, exist_ok=True)

# saving scaler
with open(os.path.join(save_dir, "standard_scaler.pkl"), "wb") as f:
 pickle.dump(scaler, f)

# saving OneHotEncoder
with open(os.path.join(save_dir, "onehot_encoder.pkl"), "wb") as f:
 pickle.dump(ohe, f)

# saving feature columns
with open(os.path.join(save_dir, "feature_columns.pkl"), "wb") as f:
 pickle.dump(X_train.columns.tolist(), f)

print("Preprocessing artifacts saved successfully")
```

## **Getting final columns**


```python
columns = X_train.columns
columns
```

# **Modelling**
---


```python
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier, Perceptron, RidgeClassifierCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, cohen_kappa_score
import pandas as pd
import numpy as np
import time
import os

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

results = []

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(),
    "CalibratedClassifierCV": CalibratedClassifierCV(), 
    "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1),
    "Perceptron": Perceptron()
}

for name, model in models.items():
    print(f"\nTraining: {name}")
    
    try:
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
        
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                roc = roc_auc_score(y_test, y_prob, multi_class="ovr")
            else:
                roc = None
        except:
            roc = None
        
        end_time = time.time()
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Balanced Accuracy": bal_acc,
            "ROC AUC": roc,
            "F1 Score": f1,
            "QWK": qwk,
            "Time Taken": end_time - start_time
        })
     
        print(f"{name} | Acc: {acc:.4f} | F1: {f1:.4f} | QWK: {qwk:.4f}")
        
    except Exception as e:
        print(f"Failed: {name}")
        print("Error:", str(e))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="QWK", ascending=False)

print("\n--- FINAL RESULTS ---")
display(results_df)
```


```python
import os
import time
import joblib

from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, Perceptron, RidgeClassifierCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, cohen_kappa_score
import pandas as pd

model_dir = "artifacts/models"
os.makedirs(model_dir, exist_ok=True)

results = []

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(),
    "CalibratedClassifierCV": CalibratedClassifierCV(), 
    "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1),
    "Perceptron": Perceptron()
}

for name, model in models.items():
    print(f"\nTraining: {name}")
    
    try:
        start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
        
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                roc = roc_auc_score(y_test, y_prob, multi_class="ovr")
            else:
                roc = None
        except:
            roc = None
        
        end_time = time.time()
        
        model_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        
        size_kb = os.path.getsize(model_path) / 1024
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Balanced Accuracy": bal_acc,
            "ROC AUC": roc,
            "F1 Score": f1,
            "QWK": qwk,
            "Time Taken": end_time - start_time,
            "Model Size (KB)": size_kb
        })
        
        print(f"{name} | Acc: {acc:.4f} | F1: {f1:.4f} | QWK: {qwk:.4f} | Size: {size_kb:.2f} KB")
        
    except Exception as e:
        print(f"Failed: {name}")
        print("Error:", str(e))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="QWK", ascending=False)

print("\n--- FINAL RESULTS ---")
display(results_df)
```

## **Perceptron Classifier**


```python
import os
import time
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score
)

model_dir = "artifacts/models"
os.makedirs(model_dir, exist_ok=True)

model = Perceptron()

print("\nTraining: Perceptron")

start_time = time.time()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

end_time = time.time()

model_path = os.path.join(model_dir, "Perceptron.pkl")
joblib.dump(model, model_path)

size_kb = os.path.getsize(model_path) / 1024

qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

print("\n--- Classification Report ---\n")
print(classification_report(y_test, y_pred))
print(f"QWK: {qwk:.4f}")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", ax=ax, colorbar=True)

ax.set_title("Confusion Matrix - Perceptron", fontweight="bold")
ax.set_xlabel("Predicted", fontweight="bold")
ax.set_ylabel("Actual", fontweight="bold")

plt.tight_layout()
plt.show()

print(f"\nTime Taken: {end_time - start_time:.2f} sec")
print(f"Model Size: {size_kb:.2f} KB")
```

## **10 Fold Cross Validation**


```python
import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score
)

model_dir = "artifacts/models"
os.makedirs(model_dir, exist_ok=True)

model = Perceptron()

print("\nRunning 10-Fold Cross Validation: Perceptron")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

start_time = time.time()

y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)

print("\n--- Classification Report (10-Fold CV) ---\n")
print(classification_report(y_train, y_pred_cv))

qwk = cohen_kappa_score(y_train, y_pred_cv, weights="quadratic")
print(f"QWK (10-Fold CV): {qwk:.4f}")

cm = confusion_matrix(y_train, y_pred_cv)

fig, ax = plt.subplots(figsize=(5, 4))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", ax=ax, colorbar=True)

ax.set_title("Confusion Matrix - Perceptron (10-Fold CV)", fontweight="bold")
ax.set_xlabel("Predicted", fontweight="bold")
ax.set_ylabel("Actual", fontweight="bold")

plt.tight_layout()
plt.show()

acc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"\nMean CV Accuracy: {acc_scores.mean():.4f}")
print(f"Std CV Accuracy: {acc_scores.std():.4f}")

end_time = time.time()

model.fit(X_train, y_train)

model_path = os.path.join(model_dir, "Perceptron_cv.pkl")
joblib.dump(model, model_path)

size_kb = os.path.getsize(model_path) / 1024

print(f"\nTime Taken: {end_time - start_time:.2f} sec")
print(f"Model Size: {size_kb:.2f} KB")
```

# **Final Test for Data Leakage**
---


```python
import numpy as np

y_shuffled = np.random.permutation(y_train)

model.fit(X_train, y_shuffled)
y_pred = model.predict(X_test)

print("Accuracy with shuffled labels:", accuracy_score(y_test, y_pred))
```

# **Model Optimization**
---

## **Save the Model**


```python
import os
import time
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# defining directory to save model (UPDATED PATH)
model_dir = "/kaggle/working/complaints_cleaned_model"
os.makedirs(model_dir, exist_ok=True)

# training model
model = Perceptron()

print("\nTraining: Perceptron")

start_time = time.time()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

end_time = time.time()

# saving model (UPDATED PATH)
model_path = os.path.join(model_dir, "Perceptron.pkl")
joblib.dump(model, model_path)

# calculating model size
size_kb = os.path.getsize(model_path) / 1024

# calculating metrics
print("\n--- Classification Report ---\n")
print(classification_report(y_test, y_pred))

# creating confusion matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", ax=ax, colorbar=True)

ax.set_title("Confusion Matrix - Perceptron", fontweight="bold")
ax.set_xlabel("Predicted", fontweight="bold")
ax.set_ylabel("Actual", fontweight="bold")

plt.tight_layout()
plt.show()

# creating summary
print(f"\nTime Taken: {end_time - start_time:.2f} sec")
print(f"Model Size: {size_kb:.2f} KB")
print(f"Model saved at: {model_path}")
```

## **Hyperparameter Tuning**


```python
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, cohen_kappa_score
import joblib
import os
import time
from tqdm.auto import tqdm

from joblib import parallel_backend
from contextlib import contextmanager

@contextmanager
def tqdm_joblib(tqdm_object):
    from joblib import parallel
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    
    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

param_grid = {
    "penalty": [None, "l2", "l1", "elasticnet"],
    "alpha": [1e-4, 1e-3, 1e-2],
    "max_iter": [1000, 2000],
    "eta0": [0.001, 0.01, 0.1],
    "fit_intercept": [True, False]
}

model = Perceptron()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=0
)

total_combinations = (
    len(param_grid["penalty"]) *
    len(param_grid["alpha"]) *
    len(param_grid["max_iter"]) *
    len(param_grid["eta0"]) *
    len(param_grid["fit_intercept"])
)
total_fits = total_combinations * cv.get_n_splits()

print(f"\n🚀 Starting Grid Search ({total_fits} fits)...")

start_time = time.time()

with tqdm_joblib(tqdm(total=total_fits)):
    grid.fit(X_train, y_train)

end_time = time.time()

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

print(f"\nBest CV Score (F1_macro): {grid.best_score_:.4f}")

y_pred = best_model.predict(X_test)

qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

print("\nTest Classification Report:\n")
print(classification_report(y_test, y_pred))
print(f"QWK: {qwk:.4f}")

save_dir = "/kaggle/working/complaints_cleaned_model"
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "perceptron_tuned.pkl")
joblib.dump(best_model, model_path)

print(f"\nModel saved at: {model_path}")
print(f"\nTime Taken: {end_time - start_time:.2f} sec")
```

## **Soft Optimization**


```python
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, cohen_kappa_score
import joblib
import os
import time
import numpy as np
from tqdm.auto import tqdm
from contextlib import contextmanager

@contextmanager
def tqdm_joblib(tqdm_object):
    from joblib import parallel
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    
    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

param_grid = [
    {
        "penalty": [None],
        "fit_intercept": [True, False],
        "max_iter": [1000]
    },
    {
        "penalty": ["l2"],
        "alpha": [1e-5, 1e-4, 1e-3],
        "eta0": [0.01, 0.02],
        "fit_intercept": [True, False],
        "max_iter": [1000]
    },
    {
        "penalty": ["l1"],
        "alpha": [1e-5, 1e-4],
        "eta0": [0.01, 0.02],
        "fit_intercept": [True, False],
        "max_iter": [1000]
    },
    {
        "penalty": ["elasticnet"],
        "alpha": [1e-5, 1e-4],
        "eta0": [0.01, 0.02],
        "l1_ratio": [0.1, 0.3, 0.5],
        "fit_intercept": [True],
        "max_iter": [1000]
    }
]

model = Perceptron()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=0
)

def count_combinations(grid_list):
    total = 0
    for g in grid_list:
        sizes = [len(v) for v in g.values()]
        total += np.prod(sizes)
    return total

total_combinations = count_combinations(param_grid)
total_fits = total_combinations * cv.get_n_splits()

print(f"\n🚀 Starting Grid Search ({total_fits} fits)...")

start_time = time.time()

with tqdm_joblib(tqdm(total=total_fits)):
    grid.fit(X_train, y_train)

end_time = time.time()

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

print(f"\nBest CV Score (F1_macro): {grid.best_score_:.4f}")

y_pred = best_model.predict(X_test)

qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

print("\nTest Classification Report:\n")
print(classification_report(y_test, y_pred))
print(f"QWK: {qwk:.4f}")

save_dir = "/kaggle/working/complaints_cleaned_model"
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "perceptron_tuned.pkl")
joblib.dump(best_model, model_path)

print(f"\nModel saved at: {model_path}")
print(f"\nTime Taken: {end_time - start_time:.2f} sec")
```

## **Re-training best Model**


```python
import os
import time
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score
)

model = Perceptron(
    alpha=0.00001,
    eta0=0.02,
    fit_intercept=True,
    max_iter=1000,
    penalty="elasticnet",
    l1_ratio=0.3
)

print("\n🚀 Training Final Perceptron...")

start_time = time.time()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

end_time = time.time()

print("\n--- Classification Report ---\n")
print(classification_report(y_test, y_pred))

qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
print(f"QWK: {qwk:.4f}")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", ax=ax, colorbar=True)

ax.set_title("Confusion Matrix - Tuned Perceptron", fontweight="bold")
ax.set_xlabel("Predicted", fontweight="bold")
ax.set_ylabel("Actual", fontweight="bold")

plt.tight_layout()
plt.show()

save_dir = "/kaggle/working/complaints_cleaned_model"
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "perceptron_final_tuned.pkl")
joblib.dump(model, model_path)

size_kb = os.path.getsize(model_path) / 1024
size_mb = size_kb / 1024

print(f"\n⏱ Time Taken: {end_time - start_time:.2f} sec")
print(f"💾 Model Size: {size_kb:.2f} KB ({size_mb:.4f} MB)")
print(f"📂 Saved at: {model_path}")
```

# **Predicting `test.csv`**
---

## **Importing `test` data**


```python
df_test = pd.read_csv("/kaggle/input/competitions/triagegeist/test.csv")
print(f"Test dataset shape: {df_test.shape}")
```

## **Merging with history data**


```python
df = df_test.merge(df_history, on="patient_id", how="left")
df.shape
```

## **Keeping only required columns**


```python
df = df[['patient_id', 'mental_status_triage', 'num_prior_ed_visits_12m',
    'num_prior_admissions_12m', 'num_active_medications', 
    'systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 'gcs_total', 'pain_score', 'shock_index',

    'hx_hypertension', 'hx_diabetes_type2', 'hx_diabetes_type1', 'hx_asthma', 'hx_copd', 'hx_heart_failure', 'hx_atrial_fibrillation', 'hx_ckd', 'hx_liver_disease', 'hx_malignancy', 'hx_obesity', 'hx_depression', 'hx_anxiety', 'hx_dementia', 'hx_epilepsy', 'hx_hypothyroidism', 'hx_hyperthyroidism', 'hx_hiv', 'hx_coagulopathy', 'hx_immunosuppressed', 'hx_pregnant', 'hx_substance_use_disorder', 'hx_coronary_artery_disease', 'hx_stroke_prior', 'hx_peripheral_vascular_disease'
]]

df.shape
```

## **Data Handling**


```python
na_counts = df.isna().sum()
na_counts = na_counts[na_counts > 0].sort_values(ascending=False)

print(na_counts)
```


```python
df = df.fillna(-1).reset_index(drop=True)
df.head()
```

## **Creating empty features as per complaint tokens**


```python
token_cols = [f"cc_{token}" for token in token_to_index.keys()]
token_cols = [col for col in token_cols if col not in df.columns]

empty_token_df = pd.DataFrame(
    np.nan,
    index=df.index,
    columns=token_cols
)

df = pd.concat([df, empty_token_df], axis=1)

print(f"Added {len(token_cols)} new token columns to df")
df.head()
```

## **Populating the features**


```python
df = populate_complaint_multihot(
    df=df,
    df_complaints=df_complaints,
    token_to_index=token_to_index,
    prefix="cc_"
)

df.head()
```

## **Encoding and Scaling**


```python
import os
import pickle
import pandas as pd

# loading artifacts
load_dir = "artifacts/preprocessing"

with open(os.path.join(load_dir, "standard_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(load_dir, "onehot_encoder.pkl"), "rb") as f:
    ohe = pickle.load(f)

with open(os.path.join(load_dir, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

print("Artifacts loaded successfully")

# defining column groups from your file
string_cols = ["mental_status_triage"]

num_cols = [
    'num_prior_ed_visits_12m', 'num_prior_admissions_12m',
    'num_active_medications', 'systolic_bp', 'diastolic_bp',
    'heart_rate', 'respiratory_rate', 'temperature_c',
    'spo2', 'gcs_total', 'pain_score', 'shock_index'
]

# treating remaining columns as binary columns
exclude_cols = ["patient_id"] + string_cols + num_cols
binary_cols = [col for col in df.columns if col not in exclude_cols]

# defining preprocessing function
def preprocess(df_input):
    
    df = df_input.copy()
    
    # processing string features with one-hot encoding
    df[string_cols] = df[string_cols].astype(str)
    
    X_ohe = ohe.transform(df[string_cols])
    ohe_cols = ohe.get_feature_names_out(string_cols)
    
    X_ohe = pd.DataFrame(X_ohe, columns=ohe_cols, index=df.index)
    
    # processing numerical features with scaling
    X_num = df[num_cols].copy()
    X_num = X_num.fillna(X_num.median())  # safety
    X_num_scaled = scaler.transform(X_num)
    
    X_num = pd.DataFrame(X_num_scaled, columns=num_cols, index=df.index)
    
    # processing binary features
    X_bin = df[binary_cols].copy()
    
    # concatenating processed features
    X_final = pd.concat([X_bin, X_num, X_ohe], axis=1)
    
    # aligning with training features
    X_final = X_final.reindex(columns=feature_columns, fill_value=0)
    
    return X_final

# applying preprocessing
X_processed = preprocess(df)

print("Processed shape:", X_processed.shape)
```

## **Restructuring columns**

- **Check if same numbers of columns exist as in training**


```python
X_processed.shape[1] == X_train.shape[1]
```

- **Making sure we send the same stream of features to the pre-trained model**


```python
X_processed = X_processed[columns]
```


```python
import joblib
import pandas as pd

# loading model
MODEL_PATH = "/kaggle/working/complaints_cleaned_model/perceptron_tuned.pkl"
model = joblib.load(MODEL_PATH)

print("Model loaded")

# ensuring patient_id is excluded from model input
if "patient_id" in X_processed.columns:
    X_input = X_processed.drop(columns=["patient_id"])
else:
    X_input = X_processed

# generating predictions
predictions = model.predict(X_input)

# saving output
output_df = pd.DataFrame({
    "patient_id": df["patient_id"],
    "triage_acuity": predictions
})

OUTPUT_PATH = "/kaggle/working/output.csv"
output_df.to_csv(OUTPUT_PATH, index=False)

print(f"Predictions saved to {OUTPUT_PATH}")
```
