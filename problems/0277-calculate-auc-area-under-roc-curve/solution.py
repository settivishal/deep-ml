import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    # Your code here
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)

    if pos == 0.0 or neg == 0.0:
        return 0.0

    tpr, fpr = [0.0], [0.0]
    thresholds = np.unique(y_scores)[::-1]
    for t in thresholds:
        pred = y_scores >= t

        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))

        tpr.append(float(tp / pos) if pos > 0.0 else 0.0)
        fpr.append(float(fp / neg) if neg > 0.0 else 0.0)

    return float(np.trapezoid(tpr, fpr))