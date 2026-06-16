```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv('/kaggle/input/competitions/triagegeist/train.csv')
test = pd.read_csv('/kaggle/input/competitions/triagegeist/test.csv')
```


```python
df.head()
```


```python
df.columns
```


```python
df.isnull().sum()
```


```python
df['site_id'].unique()
```


```python
# encoding side_id
site = {
    'SITE-TMP-01' : 1,
    'SITE-HEL-01' : 2,
    'SITE-HEL-02' : 3,
    'SITE-TUR-01' : 4,
    'SITE-OUL-01' : 5
}

df['site_id'] = df['site_id'].map(site)
```


```python
df['triage_nurse_id'].unique()
```


```python
## frequency encoding the nurses ,  because behavior of nurses is a key feature for acuity
freq = df['triage_nurse_id'].value_counts().to_dict()
df['triage_nurse_id'] = df['triage_nurse_id'].map(freq)
```


```python
df.head()
```


```python
df['arrival_mode'].unique()
```


```python
# manual encoding arrival_mode
arrival = {
    'walk-in':1,
    'police':2,
    'ambulance':3,
    'transfer':4,
    'helicopter':5,
    'brought_by_family':6,
}
df['arrival_mode'] = df['arrival_mode'].map(arrival)
```


```python
df.head()
```


```python
df['arrival_day'].unique()
```


```python
## Cyclic encoding for our week days

day_map = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

df['day_num'] = df['arrival_day'].map(day_map)
```


```python
## Applying sin , cos transformations

df['day_sin'] = np.sin(2 * np.pi * df['day_num'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_num'] / 7)
```


```python
df['day_sin']
```


```python
df['day_cos']
```


```python
### why we did cyclic encoding instead of manual encoding , becoz in that case model thinks sunday=0 is > saturday=6
```


```python
# now we remove df['arrival_day] and day_num

df = df.drop(['arrival_day' , 'day_num'] , axis = 1)
```


```python
df['arrival_season'].unique()
```


```python
# checking behavior of seasons with target (triage_acuity)

df.groupby('arrival_season')['triage_acuity'].mean().sort_values()
```


```python
sns.countplot(x = 'arrival_season' , hue = 'triage_acuity' , data = df)
plt.title('Acuity across Seasons')
plt.show()
```


```python
# we see all seasons are identical , so we drop this feature
df = df.drop('arrival_season' , axis = 1)
```


```python
df.head()
df.columns
```


```python
# checking monthly behavior with target feature
sns.countplot(x = 'arrival_month' , hue = 'triage_acuity' , data = df)
plt.title('Acuity across Month')
plt.show()
```


```python
# so we drop thius feature too
df = df.drop('arrival_month' , axis=1)
```


```python
### encoding done till arrival_mode , 
# have to start from arrival_hour

df.head()
```


```python
sns.countplot(x = 'arrival_hour' , hue = 'triage_acuity' , data = df)
plt.title('Acuity across Shift')
plt.show()
```


```python
df = df.drop('arrival_hour' , axis = 1)
```


```python
# checking arrival shift distribution
sns.countplot(x = 'shift' , hue = 'triage_acuity' , data = df)
plt.title('Acuity across Shift')
plt.show()
```


```python
## shift also follows same pattern eevrywhere , traige = 3 is highest
# so we drop it
df = df.drop('shift' , axis = 1)
```


```python
### observing age group
num_age_groups = df['age_group'].value_counts()
num_age_groups
```


```python
index_age_groups = df['age_group'].value_counts().index
index_age_groups
```


```python
plt.pie(
    num_age_groups , 
    labels = index_age_groups , 
    autopct = '%1.2f%%' , 
    startangle=90
)
plt.title('Age Group Distribution')
plt.show()
```


```python
sns.countplot(x = 'age_group' , hue = 'triage_acuity' , data = df)
plt.title('Acuity across Age Group')
plt.show()
```


```python
## encoding age group
df['age_group'] = df['age_group'].map({
    'pediatric' : 0,
    'young_adult' : 1,
    'middle_aged' : 2,
    'elderly' : 3
})
```


```python
# cHECKING language
df['language'].unique()
```


```python
num_lang = df['language'].value_counts()
num_lang

index_lang = df['language'].value_counts().index
index_lang


plt.pie(
    num_lang , 
    labels = index_lang , 
    autopct = '%1.2f%%' , 
    startangle=90
)
plt.title('Language Distribution of Pie Chart')
plt.show()
```


```python
sns.countplot(x = 'language' , hue = 'triage_acuity' , data = df)
plt.title('Acuity across Language')
plt.show()
```


```python
# %age plot for language column
pd.crosstab(df['language'] , df['triage_acuity'] , normalize = 'index')
```


```python
# removing language column
df = df.drop('language' , axis=1)
df.head()
```


```python
## Boxplot of sex with triage acuity
sns.boxplot(x = 'sex' , y = 'triage_acuity' , data = df)
plt.title('BoxPLot')
plt.show()
```


```python
## Checking the percentage plot of sex column
pd.crosstab(df['sex'] , df['triage_acuity'] , normalize = 'index')
```


```python
# dropping sex column as it's acuity is same for male and female
df = df.drop('sex' , axis = 1)
```


```python
df.head()
```


```python
##### a function of top 3 visualizattttion techniques of a column to save our time

def top3visuals(df , col , target = 'triage_acuity'):
    plt.figure(figsize = (18 , 12))  # width = 18 , height = 12

    # countplot
    plt.subplot(2 , 3 , 1)  # grid = 2 rows * 3 columns ,  postion = 1
    sns.countplot(x = col ,hue = target ,  data = df)
    plt.title(f" countplot distribution of {col} vs {target}")
    plt.xticks(rotation = 45) # for rotaing readability

    # %age plot
    plt.subplot(2 , 3 , 2)   # grid = 2 rows * 3 columns ,  postion = 2
    percent = pd.crosstab(df[col] , df[target] , normalize = 'index')# normalize='index' means we want = row-wise percentage
    # perecnt = is a daatframe of percentages
    percent.plot(kind = 'bar' , stacked = True , ax = plt.gca())  # we are now plotting this dataframe called percent , and kind = 'bar' means vertical bar chart , stacked = 'True' means each bar is on top of other making combined %age as 100% and plt.gca() = “get current axis”
    plt.title(f"Percentage Plot for {col} v/s {target}")
    plt.xticks(rotation = 45)

    # pie chart
    plt.subplot(2 , 3 , 3)   # grid = 2 rows * 3 columns ,  postion = 3
    values = df[col].value_counts()
    index = df[col].value_counts().index
    plt.pie(values , labels = index , startangle=90 , autopct = '%1.2f%%')
    plt.title(f"PIE CHART DISTRIBUTION of {col}")

    plt.tight_layout()
    plt.show()
```


```python
top3visuals(df , 'insurance_type')
```


```python
# in the above figure we see acuity 3 is dominating everwhere , so 'insurance_type' is a weak feature
# dropping it
df = df.drop('insurance_type' , axis = 1)
```


```python
# checking transport orgin column
top3visuals(df , 'transport_origin')
```


```python
df = df.drop('transport_origin' , axis = 1)
```


```python
# checking pain location
top3visuals(df , 'pain_location')
```


```python
df['pain_location'].unique()
```


```python
### as this is not a real dataset , so above distribution follows a similar patter , but still we dont remove it , because pain location is an important factor for triage acuity

# encoding pain location using one hot encoding

# first merging unknown and none
df['pain_location'] = df['pain_location'].replace({
    'unknown': 'other',
    'none': 'other'
})

df = pd.get_dummies(df , columns = ['pain_location'] , drop_first = True)
```


```python
# coming on mental_status_triage
top3visuals(df , 'mental_status_triage')
```


```python
## Mental status triage is a very usewfull feature fpr determining triage acuity , so we keep it.

# alert < drowsy < confused < agitated < unresponsive

# ordinal encoding it
mental_status = {
    'alert' : 0,
    'drowsy' : 1,
    'confused' : 2,
    'agitated' : 3,
    'unresponsive' : 4
}
df['mental_status_triage'] = df['mental_status_triage'].map(mental_status)
```


```python
df.head()
```


```python
# chief complaint system
top3visuals(df , 'chief_complaint_system')

# chief complaint system is an imp. feature irrespective of its distribution
```


```python
# 3 grouping imp ones to avoid noise

df['chief_complaint_system'] = df['chief_complaint_system'].replace({
    'neurological' : 'critical',
    'trauma' : 'critical',
    'respiratory' : 'critical',
    'cardiovascular' : 'crticial',


    'gastrointestinal' : 'moderate',
    'infectious' : 'moderate',
    'endocrine' : 'moderate',

    'dermatological' : 'mild',
    'ophthalmic' : 'mild',
    'ENT' : 'mild',

    'psychiatric' : 'special',
    'genitourunary' : 'special',
    'musculoskeletal' : 'special',
    'other' : 'special'

})
```


```python
df = pd.get_dummies(df , columns = ['chief_complaint_system'] , drop_first = True)
```


```python
df.columns
```


```python
cat_cols = df.select_dtypes(include = ['object' , 'string' , 'category']).columns
cat_cols
```


```python
df['disposition'].unique()
```


```python
top3visuals(df , 'disposition')
```


```python
### So we will encode this feature using target encoding 9 gives the average value of target variable instead of random numbers)

import category_encoders as ce

encoder = ce.TargetEncoder(cols = ['disposition'])
df['disposition'] = encoder.fit_transform(
    df['disposition'],
    df['triage_acuity']
)
```


```python
df[['disposition', 'triage_acuity']].head()
```


```python
pd.set_option('display.max_columns' , None)
```


```python
df.head(10)
```


```python
df.isna().sum()[df.isna().sum() > 0]
```


```python
cols = [
    'systolic_bp',
    'diastolic_bp',
    'mean_arterial_pressure',
    'pulse_pressure',
    'respiratory_rate',
    'temperature_c',
    'shock_index',
    'triage_acuity'
]

df_selected = df[cols]

df_selected.head()
```


```python
for col in cols:
    df[col + '_missing'] = df[col].isna().astype(int)
```


```python
for col in cols:
    df[col] = df[col].fillna(df[col].median())
```


```python
df.isna().sum().sum()
```


```python
df.isnull().sum()
```


```python
df.columns
```


```python
df = df.drop(['triage_acuity_missing' , 'disposition' , 'ed_los_hours'] , axis = True)
```


```python
df.head()
```


```python
df.duplicated().sum()  # means no duplicate value
```


```python
df.to_csv('training_set.csv' , index = False)
```

****Now Coming to Test Data****


```python
test.head()
```


```python
# encoding side_id
site = {
    'SITE-TMP-01' : 1,
    'SITE-HEL-01' : 2,
    'SITE-HEL-02' : 3,
    'SITE-TUR-01' : 4,
    'SITE-OUL-01' : 5
}

test['site_id'] = test['site_id'].map(site)
```


```python
freq = test['triage_nurse_id'].value_counts().to_dict()
test['triage_nurse_id'] = test['triage_nurse_id'].map(freq)
```


```python
arrival_test = {
    'walk-in':1,
    'police':2,
    'ambulance':3,
    'transfer':4,
    'helicopter':5,
    'brought_by_family':6,
}
test['arrival_mode'] = test['arrival_mode'].map(arrival_test)
```


```python
day_map_test = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

test['day_num'] = test['arrival_day'].map(day_map_test)
```


```python

test['day_sin'] = np.sin(2 * np.pi * test['day_num'] / 7)
test['day_cos'] = np.cos(2 * np.pi * test['day_num'] / 7)
```


```python
test = test.drop(['arrival_day' , 'day_num'] , axis = 1)
test = test.drop('arrival_season' , axis = 1)
test = test.drop('arrival_month' , axis=1)
test = test.drop('arrival_hour' , axis = 1)
test = test.drop('shift' , axis = 1)
```


```python
test['age_group'] = test['age_group'].map({
    'pediatric' : 0,
    'young_adult' : 1,
    'middle_aged' : 2,
    'elderly' : 3
})
```


```python
test = test.drop('language' , axis=1)
test = test.drop('sex' , axis = 1)
test = test.drop('insurance_type' , axis = 1)
test = test.drop('transport_origin' , axis = 1)



test['pain_location'] = test['pain_location'].replace({
    'unknown': 'other',
    'none': 'other'
})

test = pd.get_dummies(test , columns = ['pain_location'] , drop_first = True)
```


```python
mental_status = {
    'alert' : 0,
    'drowsy' : 1,
    'confused' : 2,
    'agitated' : 3,
    'unresponsive' : 4
}
test['mental_status_triage'] = test['mental_status_triage'].map(mental_status)
```


```python
test['chief_complaint_system'] = test['chief_complaint_system'].replace({
    'neurological' : 'critical',
    'trauma' : 'critical',
    'respiratory' : 'critical',
    'cardiovascular' : 'crticial',


    'gastrointestinal' : 'moderate',
    'infectious' : 'moderate',
    'endocrine' : 'moderate',

    'dermatological' : 'mild',
    'ophthalmic' : 'mild',
    'ENT' : 'mild',

    'psychiatric' : 'special',
    'genitourunary' : 'special',
    'musculoskeletal' : 'special',
    'other' : 'special'

})
```


```python
test = pd.get_dummies(test , columns = ['chief_complaint_system'] , drop_first = True)
```


```python
cols_test = [
    'systolic_bp',
    'diastolic_bp',
    'mean_arterial_pressure',
    'pulse_pressure',
    'respiratory_rate',
    'temperature_c',
    'shock_index'
]

test_selected = test[cols_test]

test_selected.head()
```


```python
for col in cols_test:
    test[col + '_missing'] = test[col].isna().astype(int)
```


```python
for col in cols_test:
    test[col] = test[col].fillna(test[col].median())
```


```python
test.to_csv('test_set.csv' , index = False)
```


```python
train = pd.read_csv('/kaggle/working/training_set.csv')
train.head()
```


```python
X = train.drop(['patient_id' , 'triage_acuity'] , axis = 1)
y = train['triage_acuity']
```


```python
test = pd.read_csv('/kaggle/working/test_set.csv')
test.head()
```


```python
## Splitting into training and validation data

from sklearn.model_selection import train_test_split
X_train , X_val , y_train , y_val = train_test_split(X , y , test_size = 0.20 ,stratify = y , random_state = 42)
```


```python
print(sorted(y_train.unique()))
```


```python
y_train = y_train - 1
y_val = y_val - 1
```


```python
X_test = test.drop('patient_id' , axis = 1)
```


```python
!pip install lightgbm
```


```python
class_weight_dict = {
    0: 1.0,
    1: 1.0,
    2: 1.2,
    3: 1.6,
    4: 2.0
}

sample_weights = y_train.map(class_weight_dict)
```


```python
import lightgbm as light
```


```python
# Creating Training Dataset
train_data = light.Dataset(X_train , label = y_train , weight = sample_weights)

val_data = light.Dataset(X_val , label =y_val)
```


```python
params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',

    'learning_rate': 0.02,
    'num_leaves': 100,
    'max_depth': 10,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,

    'lambda_l1': 0.5,
    'lambda_l2': 0.5
}
```

*****Now i will train my model using LightGBM Classifier , which is fast and efficient for medical and large data*****


```python
# Training our Model

model = light.train(
    params,
    train_data,
    valid_sets = [val_data] ,
    num_boost_round = 1000 ,
    callbacks = [light.early_stopping(100)]
)
```


```python
y_val_pred = model.predict(X_val)
y_val_pred = y_val_pred.argmax(axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_val, y_val_pred))
```


```python
y_test_pred = model.predict(X_test)
y_test_pred = y_test_pred.argmax(axis=1)
```


```python
y_test_pred = y_test_pred + 1
```


```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_val, y_val_pred))
```


```python
### Now creating submission file

submission = pd.DataFrame({
    'patient_id': test['patient_id'],
    'triage_acuity': y_test_pred
})
```


```python
submission.to_csv("submission.csv", index=False)
```


```python

```


```python

```


```python

```


```python

```
