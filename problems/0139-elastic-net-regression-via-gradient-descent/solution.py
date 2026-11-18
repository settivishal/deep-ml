import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        err = X @ w + bias - y

        grad_w = ((X.T @ err) / n_samples) + alpha1 * np.sign(w) + 2 * alpha2 * w
        grad_b = np.sum(err) / n_samples

        w = w - (learning_rate * grad_w)
        bias = bias - (learning_rate * grad_b)

        if np.sum(np.abs(grad_w)) < tol:
            break

    return w, bias