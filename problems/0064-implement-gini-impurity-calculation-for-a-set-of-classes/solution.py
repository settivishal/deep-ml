
import numpy as np

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    pass
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    val = float(1 - np.sum(p ** 2))
    return round(val,3)