import numpy as np

def permutation_importance(X, y, weights, bias, permutations):
    """
    Compute permutation feature importance for a linear regression model.

    Args:
        X: 2D array-like of shape (n_samples, n_features)
        y: 1D array-like of shape (n_samples,)
        weights: 1D array-like of shape (n_features,)
        bias: float, the intercept
        permutations: list of length n_features; permutations[j] is a list of
                      permutation index arrays to apply to column j

    Returns:
        List of feature importances (length n_features).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    def R2(Xm):
        pred = Xm @ w + bias
        ss_res = np.sum((y - pred) ** 2)
        ss_tol = np.sum((y - y.mean()) ** 2)

        return 1 - (ss_res / ss_tol)

    base = R2(X)
    out = []
    for j, permutation in enumerate(permutations):
        scores = []
        for p in permutation:
            X1 = X.copy()
            X1[:, j] = X[np.asarray(p), j]
            scores.append(R2(X1))
        out.append(float(base - np.mean(scores)))

    return out