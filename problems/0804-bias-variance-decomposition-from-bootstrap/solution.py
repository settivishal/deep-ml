import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    preds = np.asarray(predictions, dtype=float)
    y = np.asarray(y_true, dtype=float).reshape(-1)

    mean_pred = preds.mean(axis=0)
    bias_squared = np.mean((mean_pred - y) ** 2)

    variance = np.mean((preds - mean_pred) ** 2)
    mse = np.mean((preds - y) ** 2)
    return {
        "bias_squared": float(bias_squared),
        "variance": float(variance),
        "mse": float(mse),
    }
