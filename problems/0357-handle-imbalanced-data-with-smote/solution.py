import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Note: the random seed is set by the grader before your function runs,
    so you do NOT need to set it. Just use numpy's global RNG directly
    (np.random.randint, np.random.random, ...).

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    
    X = np.asarray(X_minority, dtype=float)
    n_samples, n_features = X.shape

    k_actual = min(k, n_samples - 1)

    if k_actual == 0 or n_synthetic == 0:
        return np.empty((0, n_features))

    out = np.empty((n_synthetic, n_features))
    for s in range(n_synthetic):
        i = np.random.randint(0, n_samples)
        j = np.random.randint(0, k_actual)

        gap = np.random.random()

        xi = X[i]

        dists = np.linalg.norm(X - xi, axis=1)

        order = np.argsort(dists, kind="stable")
        order = order[order != i]
        x_nn = X[order[j]]

        out[s] = xi + gap * (x_nn - xi)

    return out