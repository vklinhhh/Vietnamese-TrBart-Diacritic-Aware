import torch


def compute_accuracy_metrics(batch, outputs, tokenizer):
    """
    Compute various accuracy metrics from model outputs
    
    Args:
        batch: Input batch dictionary
        outputs: Model output dictionary
        tokenizer: Tokenizer for decoding predictions
        
    Returns:
        Dictionary of accuracy metrics
    """
    # Get labels and predictions
    labels = batch['labels']
    base_char_indices = batch['base_character_indices']
    diacritic_indices = batch['diacritic_indices']
    
    # Token-level accuracy
    predicted_ids = outputs['logits'].argmax(-1)
    word_correct = 0
    char_correct = 0
    char_total = 0
    
    for i, (pred, label) in enumerate(zip(predicted_ids, labels)):
        valid_indices = label != -100
        if valid_indices.any():
            valid_pred = pred[valid_indices]
            valid_label = label[valid_indices]
            word_correct += torch.all(valid_pred == valid_label).item()
            char_correct += (valid_pred == valid_label).sum().item()
            char_total += valid_indices.sum().item()
    
    batch_word_acc = word_correct / len(labels)
    batch_char_acc = char_correct / char_total if char_total > 0 else 0
    
    # Skip character-level metrics if outputs are not available
    if outputs['base_char_logits'] is None or outputs['diacritic_logits'] is None:
        return {
            'word_acc': batch_word_acc,
            'char_acc': batch_char_acc,
            'base_char_acc': 0.0,
            'diacritic_acc': 0.0
        }
    
    # Calculate sequence length for character metrics
    seq_length = min(
        base_char_indices.size(1),
        outputs['base_char_logits'].size(1)
    )
    
    # Base character accuracy
    base_char_preds = outputs['base_char_logits'].argmax(-1)
    base_char_matches = 0
    base_char_total = 0
    
    for pos in range(seq_length):
        # Only count positions with valid labels (not padding)
        valid_indices = base_char_indices[:, pos] != 0  # Assuming 0 is padding
        matches = (base_char_preds[:, pos][valid_indices] == 
                 base_char_indices[:, pos][valid_indices]).sum().item()
        base_char_matches += matches
        base_char_total += valid_indices.sum().item()
    
    batch_base_char_acc = base_char_matches / base_char_total if base_char_total > 0 else 0
    
    # Diacritic accuracy
    diacritic_preds = outputs['diacritic_logits'].argmax(-1)
    diacritic_matches = 0
    diacritic_total = 0
    
    for pos in range(seq_length):
        # Only count positions with valid labels (not padding)
        valid_indices = diacritic_indices[:, pos] != 0  # Assuming 0 is padding
        matches = (diacritic_preds[:, pos][valid_indices] == 
                 diacritic_indices[:, pos][valid_indices]).sum().item()
        diacritic_matches += matches
        diacritic_total += valid_indices.sum().item()
    
    batch_diacritic_acc = diacritic_matches / diacritic_total if diacritic_total > 0 else 0
    
    # Return all metrics
    return {
        'word_acc': batch_word_acc,
        'char_acc': batch_char_acc,
        'base_char_acc': batch_base_char_acc,
        'diacritic_acc': batch_diacritic_acc
    }

