# --- cell 1 ---
import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: An unregularized approach WILL overfit here.
    - Unregularized OLS: Train R² ≈ 1.0, Val R² ≈ -5.0
    - You need regularization to pass!
    
    Args:
        X_train: numpy array of shape (n_samples, n_features) -- standardized
                 (~250 samples, ~264 features -- more features than samples!)
        y_train: numpy array of shape (n_samples,) -- target values
        X_val:   numpy array of shape (n_val, n_features) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation targets
    
    Returns:
        predict: callable that takes X (n, n_features) and returns y_pred (n,)
    """
    # TODO: implement a training strategy that avoids overfitting
    # Some ideas:
    #   - Ridge regression (L2): add alpha * ||W||^2 to loss
    #   - Gradient descent with L2 penalty
    #   - Feature selection (remove low-variance or uncorrelated features)
    #   - Early stopping on gradient descent
    #   - Any combination of the above!
    
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    
    f_lo, f_hi = X_train.min(axis=0), X_train.max(axis=0)
    y_lo, y_hi = y_train.min(), y_train.max()
    
    y_mean = y_train.mean()
    yc = y_train - y_mean
    d = X_train.shape[1]
    XtX = X_train.T @ X_train
    Xty = X_train.T @ yc
    
    def fit(lam):
        return np.linalg.solve(XtX + lam * np.eye(d), Xty)
        
    def pred(X, w):
        Xc = np.clip(np.asarray(X, dtype=float), f_lo, f_hi)
        return np.clip(Xc @ w + y_mean, y_lo, y_hi)  
        
    def r2(y, pred):
        return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        
    best_lam, best_score = None, -np.inf
    for lam in np.logspace(-2, 4, 25):
        score = r2(y_val, pred(X_val, fit(lam)) + y_mean)
        if score > best_score:
            best_lam, best_score = lam, score
            
    w = fit(best_lam)
    
    lo, hi = y_train.min(), y_train.max()
    def predict(X):
        return pred(X, w)

    return predict