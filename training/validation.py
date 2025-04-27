import torch
import torch.nn as nn
from tqdm.auto import tqdm
import logging

from utils.losses import FocalLoss
from utils.metrics import compute_accuracy_metrics 

EXPECTED_BASE_CLASSES = 96
EXPECTED_DIAC_CLASSES = 25

def compute_validation_metrics(model, val_loader, device, logger=None):
    """
    Compute validation metrics on the full validation set (Single GPU).
    Uses vectorized loss calculation for character-level losses.

    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        device: Device to run evaluation on (e.g., 'cuda:0' or 'cpu')
        logger: Optional logger for progress reporting

    Returns:
        Dictionary of validation metrics
    """
    if logger is None: # Basic fallback logger
        logger = logging.getLogger("Validation")
        logging.basicConfig(level=logging.INFO)

    model.eval() # Set model to evaluation mode

    total_val_loss = 0.0
    total_text_loss = 0.0
    total_base_char_loss = 0.0
    total_diacritic_loss = 0.0
    total_word_acc = 0.0
    total_char_acc = 0.0
    total_base_char_acc = 0.0
    total_diacritic_acc = 0.0

    # --- Define loss functions ---
    # Assuming 0 is the padding index for base/diacritic that should be ignored
    # Using 'mean' reduction for standard validation averaging
    ce_loss_val = nn.CrossEntropyLoss(reduction='mean')
    focal_loss_val = FocalLoss(gamma=2.0, reduction='mean')
    # Text loss calculation might use a different ignore index (-100 or pad_token_id)
    pad_token_id = getattr(model, 'pad_token_id', -100)
    logger.debug(f"pad_token_id: {pad_token_id}")
    text_ce_loss_val = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    batch_count = 0

    with torch.no_grad(): # Disable gradients for validation
        progress_bar = tqdm(val_loader, desc="Validation")
        for i, batch in enumerate(progress_bar): # Add index i for logging
            # --- Data Movement ---
            try:
                pixel_values = batch['pixel_values'].to(device) # Removed non_blocking
                labels = batch['labels'].to(device)
                base_char_indices = batch['base_character_indices'].to(device)
                diacritic_indices = batch['diacritic_indices'].to(device)
            except Exception as move_e:
                 if logger: logger.error(f"Validation batch {i} move error: {move_e}", exc_info=True)
                 continue
            # --- End Data Movement ---

            try:
                # --- Forward pass ---
                logger.debug(f"Batch {i} - Labels Range: min={labels.min().item()}, max={labels.max().item()}")
                logger.debug(f"Batch {i} - Base Idx Range: min={base_char_indices.min().item()}, max={base_char_indices.max().item()}")
                logger.debug(f"Batch {i} - Diac Idx Range: min={diacritic_indices.min().item()}, max={diacritic_indices.max().item()}")
                # Check against expected ranges
                # if labels.max().item() >= model.bartpho_tokenizer.vocab_size:
                #     logger.error(f"!!! Batch {i}: Label index {labels.max().item()} is out of bounds for tokenizer vocab size {model.bartpho_tokenizer.vocab_size} !!!")
                # if base_char_indices.max().item() >= len(model.base_char_vocab):
                #     logger.error(f"!!! Batch {i}: Base index {base_char_indices.max().item()} is out of bounds for base vocab size {len(model.base_char_vocab)} !!!")
                # if diacritic_indices.max().item() >= len(model.diacritic_vocab):
                #     logger.error(f"!!! Batch {i}: Diacritic index {diacritic_indices.max().item()} is out of bounds for diacritic vocab size {len(model.diacritic_vocab)} !!!")
                # # Also check for unexpected negative values (other than -100 if that's your ignore_index for labels)
                # if (labels < 0).any() and (labels != -100).any(): # Check if any label is negative AND not -100
                #     logger.error(f"!!! Batch {i}: Negative Label index found that is not -100: {labels[labels < 0]} !!!")
                # if (base_char_indices < 0).any(): logger.error(f"!!! Batch {i}: Negative Base index found: {base_char_indices[base_char_indices < 0]} !!!")
                # if (diacritic_indices < 0).any(): logger.error(f"!!! Batch {i}: Negative Diacritic index found: {diacritic_indices[diacritic_indices < 0]} !!!")
                outputs = model(pixel_values=pixel_values, labels=labels)

                # --- Calculate Losses (Using Mean Reduction) ---
                # Text Loss

                if outputs.get('logits') is not None and labels is not None:
                    shift_logits = outputs['logits'][..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Calculate text loss - text_ce_loss_val now correctly ignores -100
                    batch_text_loss = text_ce_loss_val(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                     batch_text_loss = torch.tensor(0.0, device=device)
                # --- Vectorized Base Char / Diacritic Loss ---
                batch_base_char_loss = torch.tensor(0.0, device=device)
                batch_diacritic_loss = torch.tensor(0.0, device=device)
                batch_total_loss = batch_text_loss # Start with text loss

                base_logits = outputs.get('base_char_logits')
                diac_logits = outputs.get('diacritic_logits')

                # Proceed only if both logits and corresponding indices are present
                if base_logits is not None and diac_logits is not None and \
                   base_char_indices is not None and diacritic_indices is not None:

                    # Align sequence lengths (use min length)
                    len_logits = base_logits.size(1)
                    len_indices = base_char_indices.size(1) # Assume base/diac indices have same length
                    valid_len = min(len_logits, len_indices)

                    if valid_len > 0:
                        # Slice tensors to aligned length
                        logits_base_aligned = base_logits[:, :valid_len, :].contiguous()
                        indices_base_aligned = base_char_indices[:, :valid_len].contiguous()
                        logits_diac_aligned = diac_logits[:, :valid_len, :].contiguous()
                        indices_diac_aligned = diacritic_indices[:, :valid_len].contiguous()

                        # --- Base Char Loss (Vectorized) ---
                        # Reshape: [B, Seq, Classes] -> [B*Seq, Classes]
                        flat_logits_base = logits_base_aligned.view(-1, logits_base_aligned.size(-1))
                        # Reshape: [B, Seq] -> [B*Seq]
                        flat_indices_base = indices_base_aligned.view(-1)
                        # Loss function handles ignore_index=0 internally
                        batch_base_char_loss = ce_loss_val(flat_logits_base, flat_indices_base)

                        # --- Diacritic Loss (Vectorized) ---
                        flat_logits_diac = logits_diac_aligned.view(-1, logits_diac_aligned.size(-1))
                        flat_indices_diac = indices_diac_aligned.view(-1)
                        # Loss function handles ignore_index=0 internally
                        batch_diacritic_loss = focal_loss_val(flat_logits_diac, flat_indices_diac)

                        # Add to total loss
                        batch_total_loss += batch_base_char_loss + batch_diacritic_loss
                    else:
                        logger.debug(f"Batch {i}: Sequence length for char/diacritic loss is 0 after alignment.")
                else:
                    logger.debug(f"Batch {i}: Skipping char/diacritic loss calculation due to missing tensors.")


                # --- Accumulate metrics ---
                total_val_loss += batch_total_loss.item()
                total_text_loss += batch_text_loss.item()
                total_base_char_loss += batch_base_char_loss.item()
                total_diacritic_loss += batch_diacritic_loss.item()

                # --- Calculate accuracies for the batch ---
                metrics = compute_accuracy_metrics(batch, outputs, model.bartpho_tokenizer) # Ensure this uses the fixed version
                total_word_acc += metrics['word_acc']
                total_char_acc += metrics['char_acc']
                total_base_char_acc += metrics['base_char_acc']
                total_diacritic_acc += metrics['diacritic_acc']

                batch_count += 1

            except RuntimeError as e:
                # Catch runtime errors specifically, log batch index
                if logger:
                    logger.error(f"RuntimeError processing validation batch {i}: {e}", exc_info=True)
                    # Log shapes just before error might be useful if it happens consistently after a specific batch
                    logger.error(f"Batch {i} Shapes: Pixels: {pixel_values.shape}, Labels: {labels.shape}, BaseIdx: {base_char_indices.shape}, DiacIdx: {diacritic_indices.shape}")
                    if outputs and outputs.get('base_char_logits') is not None: logger.error(f"BaseLogits: {outputs['base_char_logits'].shape}")
                    if outputs and outputs.get('diacritic_logits') is not None: logger.error(f"DiacLogits: {outputs['diacritic_logits'].shape}")
                continue # Continue to next batch
            except Exception as e:
                 if logger:
                      logger.error(f"General error processing validation batch {i}: {e}", exc_info=True)
                 continue # Continue to next batch

    # --- Calculate final averages ---
    # (Calculation logic remains the same as previous version)
    final_metrics = {}
    if batch_count > 0:
        final_metrics['val_loss'] = total_val_loss / batch_count
        final_metrics['val_text_loss'] = total_text_loss / batch_count
        final_metrics['val_base_char_loss'] = total_base_char_loss / batch_count
        final_metrics['val_diacritic_loss'] = total_diacritic_loss / batch_count
        final_metrics['word_acc'] = total_word_acc / batch_count
        final_metrics['char_acc'] = total_char_acc / batch_count
        final_metrics['base_char_acc'] = total_base_char_acc / batch_count
        final_metrics['diacritic_acc'] = total_diacritic_acc / batch_count
    else:
        logger.warning("Validation loop completed without processing any batches successfully.")
        keys = ['val_loss', 'val_text_loss', 'val_base_char_loss', 'val_diacritic_loss',
                'word_acc', 'char_acc', 'base_char_acc', 'diacritic_acc']
        final_metrics = {k: 0.0 for k in keys}

    model.train() # Set model back to training mode
    return final_metrics