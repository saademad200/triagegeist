import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import category_encoders as ce

def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    history = pd.read_csv("data/patient_history.csv")
    complaints = pd.read_csv("data/chief_complaints.csv")
    return train, test, history, complaints

def build_features(df, history, complaints, tfidf=None, fit_tfidf=False, tfidf_max_features=200, target_encoder=None, is_train=False):
    df = df.copy()

    # Join history and complaints
    df = df.merge(history, on='patient_id', how='left')
    df = df.merge(complaints[['patient_id','chief_complaint_raw']], on='patient_id', how='left')

    # pain_score: clinical missingness signal
    df['pain_not_recorded'] = (df['pain_score'] == -1).astype(int)
    df.loc[df['pain_score'] == -1, 'pain_score'] = np.nan
    df['pain_score'] = df.groupby('age_group')['pain_score'].transform(
        lambda x: x.fillna(x.median()))

    # Missingness indicators for BP and RR
    num_cols = ['systolic_bp','diastolic_bp','mean_arterial_pressure',
                'pulse_pressure','shock_index','respiratory_rate','temperature_c']
    for col in num_cols:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    # Impute remaining numerics with age_group + shift medians
    for col in num_cols:
        if col in df.columns:
            df[col] = df.groupby(['age_group','shift'])[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    # Derived features
    df['elderly']     = (df['age'] >= 65).astype(int)
    df['pediatric']   = (df['age'] < 16).astype(int)
    df['night_shift'] = (df['shift'] == 'night').astype(int)
    df['weekend']     = df['arrival_day'].isin(['Saturday','Sunday']).astype(int)
    df['high_risk_arrival'] = df['arrival_mode'].isin(['ambulance','helicopter']).astype(int)
    df['altered_ms']  = df['mental_status_triage'].isin(['confused','drowsy','unresponsive','agitated']).astype(int)

    # NOVEL CLINICAL INTERACTION FEATURES
    df['sirs_proxy'] = ((df['heart_rate'] > 90).astype(int) + 
                        ((df['temperature_c'] > 38.0) | (df['temperature_c'] < 36.0)).astype(int) + 
                        (df['respiratory_rate'] > 20).astype(int))
    df['pulse_pressure_index'] = df['pulse_pressure'] / (df['systolic_bp'] + 1e-5)
    
    # Vitals Extremity Score
    df['vitals_extremity'] = (
        (df['systolic_bp'] > 180) | (df['systolic_bp'] < 90) |
        (df['heart_rate'] > 120) | (df['heart_rate'] < 50) |
        (df['respiratory_rate'] > 24) | (df['respiratory_rate'] < 10) |
        (df['spo2'] < 92)
    ).astype(int)

    # TF-IDF on chief complaint text
    df['chief_complaint_raw'] = df['chief_complaint_raw'].fillna('unknown')
    if fit_tfidf:
        tfidf = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1,2),
                                min_df=5, sublinear_tf=True)
        tfidf_matrix = tfidf.fit_transform(df['chief_complaint_raw'])
    else:
        tfidf_matrix = tfidf.transform(df['chief_complaint_raw'])

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'cc_{c}' for c in tfidf.get_feature_names_out()],
        index=df.index
    )
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    # Target Encoding for Nurse Bias and Site/Complaint systems
    te_cols = ['triage_nurse_id', 'chief_complaint_system', 'pain_location', 'site_id']
    if is_train and 'triage_acuity' in df.columns:
        for col in te_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)
                
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        df_encoded = df.copy()
        
        y_te = df['triage_acuity'].astype(float)
        
        target_encoder = ce.TargetEncoder(cols=te_cols, smoothing=10)
        target_encoder.fit(df[te_cols], y_te)
        
        for train_idx, val_idx in kf.split(df):
            X_tr, X_va = df.iloc[train_idx], df.iloc[val_idx]
            y_tr = y_te.iloc[train_idx]
            te = ce.TargetEncoder(cols=te_cols, smoothing=10)
            df_encoded.loc[val_idx, te_cols] = te.fit_transform(X_tr[te_cols], y_tr)
        
        for col in te_cols:
            df[f'{col}_te'] = df_encoded[col].astype(float)
    else:
        for col in te_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str)
        
        if target_encoder is not None:
            te_transformed = target_encoder.transform(df[te_cols])
            for col in te_cols:
                df[f'{col}_te'] = te_transformed[col].astype(float)

    # Label encode other categoricals
    cat_cols = ['arrival_mode','arrival_day','arrival_season','shift','age_group',
                'sex','language','insurance_type','transport_origin',
                'mental_status_triage']
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Drop non-feature columns and guardrails
    drop_cols = ['patient_id','triage_nurse_id','chief_complaint_raw',
                 'disposition','ed_los_hours','triage_acuity',
                 'chief_complaint_system', 'pain_location', 'site_id']
    feat_cols = [c for c in df.columns if c not in drop_cols]

    if is_train:
        return df[feat_cols], tfidf, target_encoder
    else:
        return df[feat_cols], tfidf
