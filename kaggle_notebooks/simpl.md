```python
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np #
import os
#!pip install openpyxl 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        if filename.find("r ai")>0:
            print('reading train')            
            train=pd.read_excel(os.path.join(dirname, filename),sheet_name='entrenamiento')#,sep='\t' )#,error_bad_lines=False)#,names=['id','body','headline']
        if filename.find("rain")>0:
            print('reading test')
            train=pd.read_csv(os.path.join(dirname, filename) )#,error_bad_lines=False )
        if filename.find(" et")>0:
            print('reading test')            
            test=pd.read_excel(os.path.join(dirname, filename),sheet_name='prueba')#,error_bad_lines=False ,sep='\t',engine='python')#,names=['id','body','headline']
        if filename=="Fake.csv":
            print('reading fake')            
            fake=pd.read_csv(os.path.join(dirname, filename))#,delimiter=',',header=0,names=['class','headline','body'])
        if filename.find("est")>0:
            print('reading prod')            
            test=pd.read_csv(os.path.join(dirname, filename))#,sep=';',error_bad_lines=False )#,delimiter=',',header=0,names=['class','headline','body'])import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
```


```python
hist=pd.read_csv('/kaggle/input/competitions/triagegeist/patient_history.csv')
total=pd.concat([train,test],ignore_index=True)
#total=total.merge(hist,how='left',on='patient_id')
total=total.drop(['disposition','ed_los_hours'],axis=1)


```


```python
target='triage_acuity'
idnr='patient_id'
import h2o
```


```python
import h2o
from h2o.automl import H2OAutoML
h2o.init()

if True:
    h2o_train = h2o.H2OFrame(total[:len(train)].reset_index())#train.reset_index())
    h2o_test = h2o.H2OFrame(total[-len(test):].reset_index())#test.reset_index())
    h2o_train[target] = h2o_train[[target]].asfactor()

atml = H2OAutoML(seed=13, exclude_algos = ['DeepLearning'],
                 balance_classes=True,
                 preprocessing = ["target_encoding"] ,
                 exploitation_ratio = 0.1, 
                 sort_metric = 'logloss')
features = h2o_train.drop([target,idnr]).columns
atml.train(x = features, y= target, training_frame= h2o_train)
board = atml.leaderboard
print( board)

#preds = atml.leader.predict(h2o_test)

if False:
    def vervangkleur(datal,origlabel):
        vervlabel=list([x for x in range(len(origlabel))])
        temp=pd.DataFrame(datal)
        temp=temp.replace(list(origlabel),list(vervlabel))
        return temp

    def H2Oreport(ydata,target,xpredi,tpredi):
        import matplotlib.pyplot as plt
        Xpredi=xpredi.as_data_frame(preds)['predict'].values
        Tpredi=tpredi.as_data_frame(preds)['predict'].values
        #print(ydata)
        from sklearn.metrics import brier_score_loss, precision_score, recall_score,f1_score,classification_report
        grens=0
        print( classification_report(ydata[target][-grens:], Xpredi[-grens:])  )
        
        try:
            #print("\tBrier: %1.3f" % (clf_score))
            print("\tPrecision: %1.3f" % precision_score(ydata[target][-grens:], Xpredi[-grens:]) )
            print("\tRecall: %1.3f" % recall_score(ydata[target][-grens:], Xpredi[-grens:]) )
            print("\tF1: %1.3f\n" % f1_score(ydata[target][-grens:], Xpredi[-grens:]) )
            #str( roc_auc_score(ydata[target][-grens:], Xpredi[-grens:])
        except:
            print('')
        try:
            plt.scatter(x=ydata[target], y=Xpredi[:len(ydata)], marker='.', alpha=1)
            plt.scatter(x=np.mean(ydata[target]), y=np.mean(Xpredi[:len(ydata)]), marker='o', color='green')
            plt.scatter(x=np.mean(ydata[target]), y=np.mean(Tpredi), marker='x', color='red')
            plt.xlabel('Real test'); plt.ylabel('Pred. test')
        except:
            uniek=ydata[target].unique()
            ydata=vervangkleur(ydata,uniek)
            Xpredi=vervangkleur(Xpredi,uniek)
            plt.scatter(x=ydata, y=Xpredi, marker='.', alpha=1)
            plt.scatter(x=np.mean(ydata), y=np.mean(Xpredi), marker='o', color='green')
            plt.scatter(x=np.mean(ydata), y=np.mean(Xpredi), marker='x', color='red')
            plt.xlabel('Real test'); plt.ylabel('Pred. test')
            
        plt.show()
        
        

        try:
                    features=pd.DataFrame( cla.feature_importances_,index=features,columns=['importance'])
                    features=features.sort_values('importance',ascending=False)
                    features[:15].plot(kind='barh')
                    plt.show()

        except:
                    print('')        
        return 
    Xpredi=atml.leader.predict(h2o_train)#.drop([target],axis=1))
    Tpredi=atml.leader.predict(h2o_test)#.drop([target],axis=1))
    H2Oreport(pd.DataFrame(train[target]),target,Xpredi,Tpredi)
    
#subm = pd.DataFrame(test.reset_index()[idnr])
#subm[target]=preds.as_data_frame(preds)['predict'].values
#subm[target]=subm[target].astype('int')
#subm[target]=subm[target].astype('float')
#subm.columns=[idnr,target]
#subm.to_csv('submitH20.csv',index=False)
#subm.groupby(target).count()
```


```python
import pandas as pd 
import numpy as np 
def describepd(data): 
    import numpy as np 
    print(data.head(5)) 
    output=[] 
    for li in data.columns: 
        aantal=len(data) 
        vul=len(data[li].dropna()) 
        vtyp=data[li].dtypes 
        try: 
            uniek=len(data[li].unique()) 
        except: 
            uniek=0 
        if uniek==aantal: 
            veldindex='indexfield' 
        elif uniek==1: 
            veldindex='constant' 
        elif uniek<30: 
            veldindex='categorie' 
        else: 
            veldindex='' 
        output.append([li,vtyp,np.round(100-vul/aantal*100),np.round(uniek/aantal*1000)/10,uniek,veldindex] ) 
    output=pd.DataFrame(output,columns=['label','dtype','%empty','%uniek','aantaluniek','keyparam']) 
    output.index=output['label'] 
    output[['%uniek','%empty']].plot(kind='bar', stacked=False) 
    return output.sort_values('aantaluniek')[-50:]


```


```python
compl=pd.read_csv('/kaggle/input/competitions/triagegeist/chief_complaints.csv')
total=total.merge(compl,how='left',on='patient_id')
```


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from catboost import CatBoostClassifier

def process_and_predict(df, text_col, target_col, id_col):
    # 1. Separate Train and Test (the original "unlabeled" records)
    test_mask = df[target_col].isna()
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    
    # 2. Encode Labels (Labels to Digits)
    le = LabelEncoder()
    y = le.fit_transform(train_df[target_col])
    
    # 3. Create a "Validation Split" from the Train set to check for overfitting
    # 20% for verification
    # ADD 'stratify=y' to ensure all classes appear in both sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_df.drop(columns=[target_col]), y, test_size=0.20, random_state=42,stratify=y  # <--- This is the fix
    )    
    # 4. Define the Feature Extraction Pipeline
    # We use TruncatedSVD to compress the high-dimensional TF-IDF matrix
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, stop_words='english')),
        ('svd', TruncatedSVD(n_components=50))  # Compresses text to 50 features
    ])
    
    # Combine text features with numeric features (like age, weight)
    # Add your numeric columns here
    numeric_cols = ['age', 'weight_kg', 'height_cm', 'bmi'] 
    
    preprocessor = ColumnTransformer([
        ('text', text_pipeline, text_col),
        ('num', StandardScaler(), numeric_cols)
    ])
    
    # 5. The Full Model Pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', CatBoostClassifier(verbose=250, iterations=500))
    ])
    
    # 6. Fit and Verify
    print(f"Training on {len(X_train)} samples, verifying on {len(X_val)} samples...")
    model.fit(X_train, y_train)
    
    val_preds = model.predict(X_val)
    print("--- Validation Results ---")
    print(le.classes_)
    print(classification_report(y_val, val_preds, target_names=[str(xi) for xi in le.classes_]))
    import matplotlib.pyplot as plt
    
    # 1. Get the importance values
    feature_importance = model.named_steps['classifier'].get_feature_importance()
    # 2. Get the feature names (including the ones TF-IDF created)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    
    # 3. Create a DataFrame for easy plotting
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    fi_df = fi_df.sort_values(by='importance', ascending=False)
    
    # 4. Plot the top 20
    plt.figure(figsize=(10, 8))
    plt.barh(fi_df['feature'][:20], fi_df['importance'][:20])
    plt.gca().invert_yaxis()
    plt.title('Top 20 Features Contributing to Triage Acuity Forecast')
    plt.show()    
    # 7. Final Prediction on the actual Test Set (missing labels)
    if len(test_df) > 0:
        test_preds = model.predict(test_df)
        test_df[target_col] = le.inverse_transform(test_preds)
        print(f"Predictions complete for {len(test_df)} test records.")
        return test_df[[id_col, target_col]]
    
    return None

# Usage:
results = process_and_predict(total, 'chief_complaint_raw', target,idnr)
```


```python
# Convert target to numeric for correlation check
corr_df = total[:len(train)].copy()
le = LabelEncoder()
y = le.fit_transform(corr_df[target])
    
corr_df['target_numeric'] = le.transform(corr_df['triage_acuity'])

# Get correlations with the target
correlations = corr_df.select_dtypes(include=[np.number]).corr()['target_numeric'].sort_values(ascending=False)
print(correlations)
```


```python

```
