import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    # Your code here
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    thresholds = np.unique(y_scores)[::-1]
    total_pos = np.sum(y_true == 1)
    precisions, recalls = [], []

    for t in thresholds:
        pred = y_scores >= t
        tp = np.sum(pred & (y_true == 1))
        pred_pos = np.sum(pred)
        precisions.append(float(tp / pred_pos) if pred_pos > 0 else 1.0)
        recalls.append(float(tp / total_pos) if total_pos > 0 else 0.0)
    return precisions, recalls, [float(t) for t in thresholds]