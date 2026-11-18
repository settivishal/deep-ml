import numpy as np

def apply_weight_decay(
    parameters: list[list[float]], 
    gradients: list[list[float]], 
    lr: float, 
    weight_decay: float, 
    apply_to_all: list[bool]
) -> list[list[float]]:
    """
    Apply weight decay (L2 regularization) to parameters.
    
    Args:
        parameters: List of parameter arrays
        gradients: List of gradient arrays
        lr: Learning rate
        weight_decay: Weight decay factor
        apply_to_all: Boolean list indicating which parameter groups get weight decay
    
    Returns:
        Updated parameters
    """
    # Your code here
    updated = []
    for p, g, decay in zip(parameters, gradients, apply_to_all):
        p = np.asarray(p, dtype=float)
        g = np.asarray(g, dtype=float)
        if decay:
            new_p = p - lr * (g + weight_decay * p)
        else:
            new_p = p - lr * g
        updated.append(new_p.tolist())
    
    return updated