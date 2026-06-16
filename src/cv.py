import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from src.metrics import compute_metric

def run_cv(X_train, y_train, params, n_splits=5, seed=42, model_type='lgb'):
    """
    Run StratifiedKFold cross-validation.
    model_type: 'lgb' or 'catboost'
    Returns: cv_score, oof_predictions, models, runtime
    """
    start = time.time()
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros((len(X_train), 5))
    qwk_scores = []
    models = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
        print(f'Fold {fold+1}/{n_splits}')
        X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_vl = y_train[train_idx], y_train[valid_idx]
        
        if model_type == 'lgb':
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dvalid = lgb.Dataset(X_vl, label=y_vl)
            
            callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(stopping_rounds=100, verbose=False)]
            model = lgb.train(
                params, 
                dtrain, 
                valid_sets=[dtrain, dvalid],
                callbacks=callbacks
            )
            preds = model.predict(X_vl)
            
        elif model_type == 'catboost':
            model = CatBoostClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_vl, y_vl),
                early_stopping_rounds=100,
                verbose=100,
                use_best_model=True
            )
            preds = model.predict_proba(X_vl)
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        models.append(model)
        oof_preds[valid_idx] = preds
        
        # Calculate QWK for this fold
        fold_qwk = compute_metric(y_vl, np.argmax(preds, axis=1))
        qwk_scores.append(fold_qwk)
        print(f'Fold {fold+1} QWK: {fold_qwk:.4f}')

    overall_qwk = compute_metric(y_train, np.argmax(oof_preds, axis=1))
    runtime = time.time() - start
    
    return {
        "cv_score": overall_qwk,
        "fold_scores": qwk_scores,
        "oof": oof_preds,
        "models": models,
        "runtime": runtime
    }
