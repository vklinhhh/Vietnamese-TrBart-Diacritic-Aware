import torch
import torch.nn as nn
from tqdm.auto import tqdm
from utils.losses import FocalLoss
from utils.metrics import compute_accuracy_metrics

def compute_validation_metrics(model, val_loader, device, logger=None):
    """
    Compute validation metrics on the full validation set
    
    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        device: Device to run evaluation on
        logger: Optional logger for progress reporting
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    # Initialize metrics
    val_loss = 0
    val_text_loss = 0
    val_base_char_loss = 0
    val_diacritic_loss = 0
    word_acc = 0
    char_acc = 0 
    base_char_acc = 0
    diacritic_acc = 0
    
    # Define loss functions
    ce_loss = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(gamma=2.0)
    
    # Track batch count for averaging
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            try:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)
                # Move data to device
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                base_char_indices = batch['base_character_indices'].to(device)
                diacritic_indices = batch['diacritic_indices'].to(device)
                
                # Forward pass
                outputs = model(pixel_values=pixel_values, labels=labels)
                
                # Calculate losses
                text_loss = outputs['loss'] if outputs['loss'] is not None else torch.tensor(0.0, device=device)
                
                # Skip character losses if outputs are not available
                if outputs['base_char_logits'] is None or outputs['diacritic_logits'] is None:
                    base_char_loss = torch.tensor(0.0, device=device)
                    diacritic_loss = torch.tensor(0.0, device=device)
                    total_loss = text_loss
                else:
                    # Calculate sequence length for character and diacritic losses
                    seq_length = min(
                        base_char_indices.size(1),
                        outputs['base_char_logits'].size(1)
                    )
                    
                    # Initialize losses
                    base_char_loss = torch.zeros(1, device=device)
                    diacritic_loss = torch.zeros(1, device=device)
                    
                    # Calculate for each position
                    for pos in range(seq_length):
                        # Cross entropy for base characters
                        base_char_loss += ce_loss(
                            outputs['base_char_logits'][:, pos, :], 
                            base_char_indices[:, pos]
                        )
                        
                        # Focal loss for diacritics
                        diacritic_loss += focal_loss(
                            outputs['diacritic_logits'][:, pos, :], 
                            diacritic_indices[:, pos]
                        )
                    
                    # Average over sequence length
                    base_char_loss /= seq_length
                    diacritic_loss /= seq_length
                    
                    # Combined loss with equal weights for validation
                    total_loss = text_loss + base_char_loss + diacritic_loss
                
                # Accumulate losses
                val_loss += total_loss.item()
                val_text_loss += text_loss.item()
                val_base_char_loss += base_char_loss.item()
                val_diacritic_loss += diacritic_loss.item()
                
                # Calculate accuracies
                metrics = compute_accuracy_metrics(batch, outputs, model.bartpho_tokenizer)
                
                # Accumulate accuracy metrics
                word_acc += metrics['word_acc']
                char_acc += metrics['char_acc']
                base_char_acc += metrics['base_char_acc']
                diacritic_acc += metrics['diacritic_acc']
                
                # Increment batch count
                batch_count += 1
                
            except Exception as e:
                if logger:
                    logger.error(f"Error in validation batch: {e}")
                # Continue with next batch
                continue
    
    # Calculate averages
    if batch_count > 0:
        val_loss /= batch_count
        val_text_loss /= batch_count
        val_base_char_loss /= batch_count
        val_diacritic_loss /= batch_count
        word_acc /= batch_count
        char_acc /= batch_count
        base_char_acc /= batch_count
        diacritic_acc /= batch_count
    
    # Return all metrics
    return {
        'val_loss': val_loss,
        'val_text_loss': val_text_loss,
        'val_base_char_loss': val_base_char_loss,
        'val_diacritic_loss': val_diacritic_loss,
        'word_acc': word_acc,
        'char_acc': char_acc,
        'base_char_acc': base_char_acc,
        'diacritic_acc': diacritic_acc
    }

