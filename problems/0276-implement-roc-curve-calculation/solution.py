import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    # Your code here
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)

    thresholds = np.unique(y_scores)[::-1]

    fpr, tpr = [0.0], [0.0]
    for t in thresholds:
        pred = y_scores >= t

        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))

        tpr.append(float(tp / pos) if pos > 0 else 0.0)
        fpr.append(float(fp / neg) if neg > 0 else 0.0)

    return fpr, tpr