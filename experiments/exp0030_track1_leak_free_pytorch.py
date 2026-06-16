import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.metrics import compute_metric

SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_data():
    print("Loading data for PyTorch MLP (Leak-Free)...")
    train = pd.read_csv("data/train.csv")
    ph = pd.read_csv("data/patient_history.csv")
    df = train.merge(ph, on='patient_id', how='left')
    
    y = df['triage_acuity'].values - 1  
    
    # Advanced FE
    df['historical_admission_rate'] = df['num_prior_admissions_12m'] / df['num_prior_ed_visits_12m'].clip(lower=1)
    df['sirs_tachycardia'] = (df['heart_rate'] > 90).astype(int)
    df['sirs_tachypnea'] = (df['respiratory_rate'] > 20).astype(int)
    df['sirs_temp'] = ((df['temperature_c'] > 38) | (df['temperature_c'] < 36)).astype(int)
    df['sirs_score'] = df['sirs_tachycardia'] + df['sirs_tachypnea'] + df['sirs_temp']
    
    if 'shock_index' not in df.columns:
        df['shock_index'] = df['heart_rate'] / df['systolic_bp'].clip(lower=1)
        
    df['age_adjusted_shock_index'] = df['shock_index'] * df['age']
    df['comorbidity_to_age_ratio'] = df['num_comorbidities'] / df['age'].clip(lower=1)
    df['is_hypoxic'] = (df['spo2'] < 92).astype(int)
    df['is_hypotensive'] = (df['systolic_bp'] < 90).astype(int)
    
    drop_cols = ['patient_id', 'disposition', 'ed_los_hours', 'triage_acuity', 'chief_complaint_system', 'site_id', 'triage_nurse_id']
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
            
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]
    
    # Handle NaNs in numericals (just fillna with 0 here, median will be calculated inside fold if we wanted strictness, but let's just use 0)
    for col in num_cols:
        df[col] = df[col].fillna(0)
        
    for col in cat_cols:
        df[col] = df[col].fillna('Missing').astype(str)
            
    return df, y, num_cols, cat_cols

class TriageDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = torch.FloatTensor(X_num)
        self.X_cat = torch.LongTensor(X_cat) if X_cat is not None else None
        self.y = torch.LongTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X_num)
    
    def __getitem__(self, idx):
        if self.y is not None:
            if self.X_cat is not None:
                return self.X_num[idx], self.X_cat[idx], self.y[idx]
            return self.X_num[idx], self.y[idx]
        else:
            if self.X_cat is not None:
                return self.X_num[idx], self.X_cat[idx]
            return self.X_num[idx]

class TabularMLP(nn.Module):
    def __init__(self, num_features, cat_dims, num_classes=5):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, min(50, (num_classes + 1) // 2))
            for num_classes in cat_dims
        ])
        
        total_emb_dim = sum([emb.embedding_dim for emb in self.embeddings])
        input_dim = num_features + total_emb_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x_num, x_cat):
        emb_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_outputs.append(emb_layer(x_cat[:, i]))
            
        x = torch.cat([x_num] + emb_outputs, dim=1)
        return self.mlp(x)

def main():
    set_seed(SEED)
    df, y, num_cols, cat_cols = prepare_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    oof_preds = np.zeros((len(y), 5))
    qwk_scores = []
    
    epochs = 15
    batch_size = 512
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, y)):
        print(f"Fold {fold+1}")
        df_tr, df_vl = df.iloc[train_idx].copy(), df.iloc[valid_idx].copy()
        y_tr, y_vl = y[train_idx], y[valid_idx]
        
        # STRICT LEAK-FREE PROCESSING INSIDE THE FOLD
        scaler = StandardScaler()
        X_num_tr = scaler.fit_transform(df_tr[num_cols])
        X_num_vl = scaler.transform(df_vl[num_cols])
        
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_cat_tr = encoder.fit_transform(df_tr[cat_cols]) + 1
        X_cat_vl = encoder.transform(df_vl[cat_cols]) + 1
        
        cat_dims = []
        for i, col in enumerate(cat_cols):
            # We assume classes are 1-indexed (due to +1), and 0 is reserved for unknowns
            num_unique = int(max(X_cat_tr[:, i].max(), X_cat_vl[:, i].max())) + 1
            cat_dims.append(num_unique)
        
        # For unknowns (-1+1 = 0), PyTorch Embedding will handle it since 0 is a valid index
        # We ensure unknown entries are set to 0
        X_cat_tr[X_cat_tr == 0] = 0
        X_cat_vl[X_cat_vl == 0] = 0
        
        train_dataset = TriageDataset(X_num_tr, X_cat_tr, y_tr)
        valid_dataset = TriageDataset(X_num_vl, X_cat_vl, y_vl)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        model = TabularMLP(len(num_cols), cat_dims).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_num, batch_cat, batch_y in train_loader:
                batch_num, batch_cat, batch_y = batch_num.to(device), batch_cat.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_num, batch_cat)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_num, batch_cat, batch_y in valid_loader:
                    batch_num, batch_cat, batch_y = batch_num.to(device), batch_cat.to(device), batch_y.to(device)
                    outputs = model(batch_num, batch_cat)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
            val_loss /= len(valid_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_model_fold{fold}.pt')
                
        model.load_state_dict(torch.load(f'best_model_fold{fold}.pt'))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for batch_num, batch_cat, _ in valid_loader:
                batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
                outputs = torch.softmax(model(batch_num, batch_cat), dim=1)
                fold_preds.append(outputs.cpu().numpy())
                
        fold_preds = np.vstack(fold_preds)
        oof_preds[valid_idx] = fold_preds
        
        preds_class = np.argmax(fold_preds, axis=1)
        fold_qwk = compute_metric(y_vl, preds_class)
        qwk_scores.append(fold_qwk)
        print(f'Fold {fold+1} QWK: {fold_qwk:.4f}')

    overall_qwk = compute_metric(y, np.argmax(oof_preds, axis=1))
    print(f"Overall QWK (Leak-Free PyTorch MLP): {overall_qwk:.4f}")
    
    os.makedirs('results/exp0030_track1_leak_free_pytorch', exist_ok=True)
    np.save('results/exp0030_track1_leak_free_pytorch/oof_preds.npy', oof_preds)
    
    with open('results/exp0030_track1_leak_free_pytorch/metrics.json', 'w') as f:
        json.dump({'overall_qwk': overall_qwk, 'fold_qwk': qwk_scores}, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
