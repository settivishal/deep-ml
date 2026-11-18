import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    # Your code here
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = X @ w

    first = np.mean((y_true - y_pred) ** 2)
    second = alpha * np.sum(w ** 2)

    return float(first + second)
