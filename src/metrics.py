import numpy as np
from sklearn.metrics import cohen_kappa_score

def compute_metric(y_true, y_pred):
    """
    Compute Quadratic Weighted Kappa (QWK)
    y_true: 1D array of true labels (0-4)
    y_pred: 1D array of predicted labels (0-4)
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def flat_baseline(y_train):
    """
    Returns the most frequent class (mode).
    """
    return np.bincount(y_train).argmax()
