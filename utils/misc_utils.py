import torch


def calculate_class_weights(indices, num_classes):
    """Calculate class weights for imbalanced classes"""
    # Use reshape to handle non-contiguous tensors
    all_indices = indices.contiguous().reshape(-1).cpu().numpy()
    
    # Count occurrences of each class
    class_counts = np.bincount(all_indices, minlength=num_classes)
    
    # Add small epsilon to avoid division by zero
    class_counts = class_counts + 1e-5
    
    # Calculate weights as inverse of frequency
    weights = 1.0 / class_counts
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return torch.tensor(weights, dtype=torch.float32)


def make_json_serializable(obj):
    """Make objects JSON serializable by converting numpy types"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    else:
        return obj

