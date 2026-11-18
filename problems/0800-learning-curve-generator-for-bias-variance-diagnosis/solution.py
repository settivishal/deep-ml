import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    x_tr = np.asarray(X_train, dtype=float).reshape(-1)
    y_tr = np.asarray(y_train, dtype=float).reshape(-1)
    x_va = np.asarray(X_val, dtype=float).reshape(-1)
    y_va = np.asarray(y_val, dtype=float).reshape(-1)

    V_va = np.vander(x_va, degree + 1, increasing=True)

    train_errors, val_errors = [], []
    for n in train_sizes:
        x_n, y_n = x_tr[:n], y_tr[:n]
        V_n = np.vander(x_n, degree + 1, increasing=True)
        w = np.linalg.pinv(V_n) @ y_n

        train_errors.append(float(np.mean((V_n @ w - y_n) ** 2)))
        val_errors.append(float(np.mean((V_va @ w - y_va) ** 2)))

    train_error, val_error = train_errors[-1], val_errors[-1]
    if train_error > bias_threshold:
        diagnosis = "high_bias"
    elif val_error - train_error > variance_threshold:
        diagnosis = "high_variance"
    else:
        diagnosis = "good_fit"

    return {
        "train_errors": train_errors, 
        "val_errors": val_errors, 
        "diagnosis": diagnosis
    }