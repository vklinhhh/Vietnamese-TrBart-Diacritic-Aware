# utils/metrics.py
"""
Utility functions for computing accuracy metrics during training and validation.
"""

import torch
import logging # Optional: for logging warnings within the function

# Setup a basic logger if needed within this module
logger = logging.getLogger(__name__)
# Configure logger if running this file directly or if not configured elsewhere
# if not logger.hasHandlers():
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)


def compute_accuracy_metrics(batch, outputs, tokenizer, return_counts=False):
    """
    Compute various accuracy metrics from model outputs.
    Ensures tensors are on the correct device for comparison.
    Optionally returns raw counts instead of averages.

    Args:
        batch (dict): Input batch dictionary potentially containing tensors on CPU or GPU.
                      Expected keys: 'labels', 'base_character_indices', 'diacritic_indices'.
        outputs (dict): Model output dictionary containing tensors on the model's device.
                        Expected keys: 'logits', optionally 'base_char_logits', 'diacritic_logits'.
        tokenizer: Tokenizer instance (used for pad_token_id).
        return_counts (bool): If True, return raw counts (correct, total) instead of averages.

    Returns:
        dict: Dictionary of accuracy metrics (averages or counts). Returns default zero values
              if logits are missing or an error occurs during tensor processing.
    """
    # --- Determine Target Device ---
    # Get device from model outputs (which are guaranteed to be on the target device)
    logits = outputs.get('logits')
    if logits is None:
        logger.warning("No 'logits' found in outputs for metric calculation. Returning zero metrics.")
        # Return default zero values based on return_counts
        if return_counts:
            return {
                'word_correct': 0, 'word_total': 0, 'char_correct': 0, 'char_total': 0,
                'base_char_correct': 0, 'base_char_total': 0, 'diacritic_correct': 0, 'diacritic_total': 0,
            }
        else:
            return {
                'word_acc': 0.0, 'char_acc': 0.0, 'base_char_acc': 0.0, 'diacritic_acc': 0.0
            }

    # Use device from model output logits as the target device
    device = logits.device

    # --- Ensure Batch Tensors are on Target Device ---
    # Access and move tensors needed from the batch dictionary
    try:
        # Explicitly move/confirm device for all tensors used from the batch
        labels = batch['labels'].to(device)
        base_char_indices = batch['base_character_indices'].to(device)
        diacritic_indices = batch['diacritic_indices'].to(device)
    except KeyError as e:
        logger.error(f"Missing expected key '{e}' in batch dictionary during metric calculation.")
        # Return default zero values
        if return_counts: return {'word_correct': 0, 'word_total': 0, 'char_correct': 0, 'char_total': 0, 'base_char_correct': 0, 'base_char_total': 0, 'diacritic_correct': 0, 'diacritic_total': 0}
        else: return {'word_acc': 0.0, 'char_acc': 0.0, 'base_char_acc': 0.0, 'diacritic_acc': 0.0}
    except Exception as e:
        logger.error(f"Error moving batch tensors to device {device} within metrics function: {e}", exc_info=True)
        # Return default zero values
        if return_counts: return {'word_correct': 0, 'word_total': 0, 'char_correct': 0, 'char_total': 0, 'base_char_correct': 0, 'base_char_total': 0, 'diacritic_correct': 0, 'diacritic_total': 0}
        else: return {'word_acc': 0.0, 'char_acc': 0.0, 'base_char_acc': 0.0, 'diacritic_acc': 0.0}

    # Detach tensors to prevent gradient tracking if not already done (e.g., during validation)
    labels = labels.detach()
    base_char_indices = base_char_indices.detach()
    diacritic_indices = diacritic_indices.detach()
    predicted_ids = logits.argmax(-1).detach() # Already on 'device'

    # --- Initialize Counters ---
    word_correct_count = 0
    char_correct_count = 0
    char_total_count = 0
    word_total_count = labels.size(0) # Batch size

    base_char_correct_count = 0
    base_char_total_count = 0
    diacritic_correct_count = 0
    diacritic_total_count = 0

    # Handle case where tokenizer might not have a pad token ID
    pad_token_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_token_id is None:
        logger.debug("Tokenizer does not have a pad_token_id. Comparisons will not exclude padding based on it.")
        # Use a value unlikely to be a real token ID if needed for masking
        pad_token_id = -999

    # --- Token-level accuracy ---
    for i, (pred, label) in enumerate(zip(predicted_ids, labels)): # Both tensors are on 'device'
        # Define valid indices for comparison (exclude padding and ignore_index)
        valid_indices = (label != -100) & (label != pad_token_id) # Comparison happens on 'device'

        if valid_indices.any():
            valid_pred = pred[valid_indices]     # Slice on 'device'
            valid_label = label[valid_indices]   # Slice on 'device'

            # Check if all predictions for this sequence match the labels
            if torch.all(valid_pred == valid_label): # Comparison happens on 'device'
                word_correct_count += 1

            # Accumulate character counts
            char_correct_count += (valid_pred == valid_label).sum().item()
            char_total_count += valid_indices.sum().item() # Count number of valid characters

    # --- Character-level accuracies ---
    base_logits = outputs.get('base_char_logits')
    diac_logits = outputs.get('diacritic_logits')

    # Ensure character logits are also detached and on the correct device
    if base_logits is not None:
         if base_logits.device != device: base_logits = base_logits.to(device)
         base_logits = base_logits.detach()
    if diac_logits is not None:
         if diac_logits.device != device: diac_logits = diac_logits.to(device)
         diac_logits = diac_logits.detach()

    # Proceed only if both character-level logits are available
    if base_logits is not None and diac_logits is not None:
        try:
            # Determine the valid sequence length for comparison (minimum of logits and indices)
            seq_length = min(
                base_char_indices.size(1),
                base_logits.size(1) # Use length from logits tensor
            )

            if seq_length > 0:
                # Get predictions for the aligned sequence length
                base_char_preds = base_logits[:, :seq_length, :].argmax(-1) # Shape: [B, seq_length]
                diacritic_preds = diac_logits[:, :seq_length, :].argmax(-1) # Shape: [B, seq_length]

                # Align ground truth indices length
                base_char_indices_aligned = base_char_indices[:, :seq_length] # Shape: [B, seq_length]
                diacritic_indices_aligned = diacritic_indices[:, :seq_length] # Shape: [B, seq_length]

                # Base character accuracy counts
                # Assuming 0 is padding index for base characters - ADJUST IF DIFFERENT
                valid_base_mask = (base_char_indices_aligned != 0) # Mask happens on 'device'
                if valid_base_mask.any():
                    # Comparison happens on 'device'
                    base_char_correct_count += (base_char_preds[valid_base_mask] == base_char_indices_aligned[valid_base_mask]).sum().item()
                    base_char_total_count += valid_base_mask.sum().item()

                # Diacritic accuracy counts
                # Assuming 0 is padding/no_diacritic index - ADJUST IF DIFFERENT
                valid_diac_mask = (diacritic_indices_aligned != 0) # Mask happens on 'device'
                if valid_diac_mask.any():
                    # Comparison happens on 'device'
                    diacritic_correct_count += (diacritic_preds[valid_diac_mask] == diacritic_indices_aligned[valid_diac_mask]).sum().item()
                    diacritic_total_count += valid_diac_mask.sum().item()
            else:
                 logger.debug(f"Aligned sequence length for char/diacritic metrics is 0.")

        except Exception as char_e:
             logger.error(f"Error calculating character-level metrics: {char_e}", exc_info=True)
             # Reset char counts to 0 if error occurs
             base_char_correct_count = 0
             base_char_total_count = 0
             diacritic_correct_count = 0
             diacritic_total_count = 0


    # --- Return results ---
    if return_counts:
        return {
            'word_correct': word_correct_count,
            'word_total': word_total_count,
            'char_correct': char_correct_count,
            'char_total': char_total_count,
            'base_char_correct': base_char_correct_count,
            'base_char_total': base_char_total_count,
            'diacritic_correct': diacritic_correct_count,
            'diacritic_total': diacritic_total_count,
        }
    else:
        # Calculate averages safely, handling potential division by zero
        avg_word_acc = word_correct_count / word_total_count if word_total_count > 0 else 0.0
        avg_char_acc = char_correct_count / char_total_count if char_total_count > 0 else 0.0
        avg_base_char_acc = base_char_correct_count / base_char_total_count if base_char_total_count > 0 else 0.0
        avg_diacritic_acc = diacritic_correct_count / diacritic_total_count if diacritic_total_count > 0 else 0.0
        return {
            'word_acc': avg_word_acc,
            'char_acc': avg_char_acc,
            'base_char_acc': avg_base_char_acc,
            'diacritic_acc': avg_diacritic_acc
        }