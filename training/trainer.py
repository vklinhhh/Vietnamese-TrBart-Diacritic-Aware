import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm.auto import tqdm
import logging
import json
import gc

from utils.losses import FocalLoss
from utils.logging_utils import log_curriculum_learning_curves
from utils.misc_utils import make_json_serializable
from data.collation import custom_collate_fn
from training.validation import compute_validation_metrics


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), # Ensure logs go to console
        logging.FileHandler('bartpho_training.log')
    ]
)
logger = logging.getLogger('BartPhoTraining')

# --- Constants for Vocab Sizes (Update if necessary based on logs/model) ---
# These should ideally match len(model.base_char_vocab) etc. after model init
EXPECTED_BASE_CLASSES = 96
EXPECTED_DIAC_CLASSES = 25

# ==============================================================================
# <<< --- START: Full train_bartpho_model_with_curriculum Implementation --- >>>
# ==============================================================================
def train_bartpho_model_with_curriculum(
    # --- Passed-in Components & State ---
    model,                      # Initialized/loaded model
    optimizer,                  # Initialized/loaded optimizer
    lr_scheduler,               # Initialized/loaded scheduler
    scaler,                     # Initialized GradScaler (or None if not using AMP)
    train_curriculum,           # Initialized curriculum object (stage potentially restored)
    val_dataset,                # Validation dataset (potentially filtered by curriculum setup)
    num_curriculum_stages_arg,  # Required (No default value)
    start_epoch=0,              # Epoch to start from (0 for new, N for resume)
    resumed_optimizer_steps=0,  # Optimizer steps completed before resuming
    resumed_best_val_loss=float('inf'), # Best validation loss seen so far
    # --- Core Training Parameters ---
    epochs=10,                  # Total epochs to train for
    batch_size=8,
    device=None,                # Device ('cuda' or 'cpu')
    project_name="vietnamese-ocr-bartpho-curriculum", # For WandB
    run_name=None,              # For WandB
    log_interval=10,            # Log every N optimizer steps
    focal_loss_gamma=2.0,       # For FocalLoss instantiation
    output_dir="vietnamese-ocr-bartpho-model", # Base directory for output

    stage_epochs=None,          # Optional fixed epochs per stage list [e1, e2, ...]
    stage_patience=3,           # Patience for automatic stage advancement
    early_stopping_patience=5,  # Patience for stopping entire training
    use_amp=False,              # Whether AMP is enabled
    grad_accumulation_steps=1,
    num_workers=4,              # For DataLoader
    dynamic_loss_weighting=True, # Use dynamic loss weights based on stage
    eval_steps=None,            # Evaluate every N optimizer steps
):
    """
    Enhanced training function for BartPho VietOCR with curriculum learning and resuming.

    Handles training loop, validation, checkpointing, curriculum advancement,
    and logging based on provided components and starting state.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training function using device: {device}")
    # REMOVED: curriculum_stages = train_curriculum.num_stages (causes error)

    # --- Setup Output Dirs & WandB ---
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    wandb_run = wandb.run # Use existing run if initialized in main()
    if not wandb_run and project_name:
        try:
            # Prepare config dictionary for wandb init
            wandb_config = {
                "total_epochs": epochs, "start_epoch": start_epoch, "batch_size": batch_size,
                "learning_rate": optimizer.param_groups[0]['lr'], "grad_accumulation_steps": grad_accumulation_steps,
                "use_amp": use_amp, "device": str(device),
                "curriculum_strategy": train_curriculum.strategy if hasattr(train_curriculum, 'strategy') else 'unknown',
                "curriculum_stages": num_curriculum_stages_arg, # Use argument passed in
                "stage_epochs_schedule": stage_epochs, "stage_patience": stage_patience,
                "early_stopping_patience": early_stopping_patience, "focal_loss_gamma": focal_loss_gamma,
                "dynamic_loss_weighting": dynamic_loss_weighting, "num_workers": num_workers,
                "eval_steps": eval_steps, "log_interval": log_interval,
                "weight_decay": optimizer.defaults.get('weight_decay', None), # Log weight decay if possible
            }
            wandb_run = wandb.init(
                project=project_name, name=run_name, resume="allow", config=wandb_config
            )
            logger.info(f"Initialized new WandB run: {wandb_run.id if wandb_run else 'Failed'}")
        except Exception as e:
            logger.error(f"Failed to initialize WandB in training func: {e}")
            wandb_run = None

    # --- Setup Validation Loader ---
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn,
        num_workers=num_workers, pin_memory=False
    )

    # --- Define Loss Functions ---
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)
    focal_loss = FocalLoss(gamma=focal_loss_gamma, ignore_index=0)

    # --- Initialize Training State ---
    best_val_loss = resumed_best_val_loss
    no_improvement_count = 0
    optimizer_steps = resumed_optimizer_steps

    # --- History Tracking ---
    stage_histories = []
    current_history = {
        "stage": train_curriculum.current_stage, "epoch_losses": [], "val_losses": [],
        "word_accs": [], "char_accs": [], "base_char_accs": [], "diacritic_accs": []
    }

    # --- Stage Epoch Schedule ---
    stage_start_epochs = [0]
    if stage_epochs is not None:
        for i in range(len(stage_epochs) - 1):
            stage_start_epochs.append(stage_start_epochs[-1] + stage_epochs[i])

    # ==========================================================
    # --- Main Training Loop ---
    # ==========================================================
    logger.info(f"--- Starting Training Loop from Epoch {start_epoch + 1} ---")
    training_exception = None
    best_model_state = None # Initialize here
    try:
        for epoch in range(start_epoch, epochs):
            current_epoch_num = epoch + 1

            # --- Curriculum Stage Management (Fixed Schedule) ---
            if stage_epochs is not None:
                 target_epoch_stage = 0
                 for i in range(1, len(stage_start_epochs)):
                      if epoch >= stage_start_epochs[i]: target_epoch_stage = i
                 if train_curriculum.current_stage < target_epoch_stage:
                      logger.info(f"Advancing curriculum stage at start of epoch {current_epoch_num} (target: {target_epoch_stage+1})")
                      if current_history["epoch_losses"]: stage_histories.append(current_history)
                      while train_curriculum.current_stage < target_epoch_stage: train_curriculum.advance_stage()
                      current_history = { # Reset history
                          "stage": train_curriculum.current_stage, "epoch_losses": [], "val_losses": [],
                          "word_accs": [], "char_accs": [], "base_char_accs": [], "diacritic_accs": []
                      }

            # Get current stage dataset and loader
            current_train_dataset = train_curriculum.get_current_stage_dataset()
            train_loader = DataLoader(
                current_train_dataset, batch_size=batch_size, shuffle=True,
                collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=False
            )
            logger.info(f"Starting Epoch {current_epoch_num}, Stage {train_curriculum.current_stage+1}/{num_curriculum_stages_arg}, Training on {len(train_loader.dataset)} examples.") # Use arg

            # --- Epoch Training ---
            model.train()
            epoch_train_loss, epoch_text_loss, epoch_base_loss, epoch_diacritic_loss = 0.0, 0.0, 0.0, 0.0
            batches_processed_in_epoch = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {current_epoch_num}, Stage {train_curriculum.current_stage+1}")

            for step, batch in enumerate(progress_bar):
                try:
                    # Move Batch to Device
                    try:
                        pixel_values = batch['pixel_values'].to(device, non_blocking=True)
                        labels = batch['labels'].to(device, non_blocking=True)
                        base_char_indices = batch['base_character_indices'].to(device, non_blocking=True)
                        diacritic_indices = batch['diacritic_indices'].to(device, non_blocking=True)
                    except Exception as move_e:
                        logger.error(f"Error moving batch {step} to device: {move_e}", exc_info=False)
                        logger.warning(f"Skipping batch {step}.")
                        continue

                    # --- Forward Pass & Loss Calculation ---
                    if use_amp:
                        with autocast():
                            outputs = model(pixel_values=pixel_values, labels=labels)
                            # Vectorized Loss Calc & Weighting (AMP)
                            text_loss = outputs.get('loss', torch.tensor(0.0, device=device))
                            base_logits, diac_logits = outputs.get('base_char_logits'), outputs.get('diacritic_logits')
                            base_char_loss, diacritic_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                            if base_logits is not None:
                                valid_len = base_char_indices.size(1); logits_l = base_logits[:, :valid_len, :]
                                if logits_l.shape[1] == valid_len:
                                    targets_r = base_char_indices[:,:valid_len].reshape(-1); logits_r = logits_l.reshape(-1, EXPECTED_BASE_CLASSES)
                                    if targets_r.numel() > 0: base_char_loss = ce_loss(logits_r, targets_r)
                            if diac_logits is not None:
                                valid_len = diacritic_indices.size(1); logits_l = diac_logits[:, :valid_len, :]
                                if logits_l.shape[1] == valid_len:
                                    targets_r = diacritic_indices[:,:valid_len].reshape(-1); logits_r = logits_l.reshape(-1, EXPECTED_DIAC_CLASSES)
                                    if targets_r.numel() > 0: diacritic_loss = focal_loss(logits_r, targets_r)
                            # Weighting (Use num_curriculum_stages_arg)
                            if dynamic_loss_weighting and num_curriculum_stages_arg > 1:
                                ratio = train_curriculum.current_stage / max(1, num_curriculum_stages_arg - 1)
                                tw, bw, dw = 0.7 + 0.3*ratio, 0.8 + 0.2*ratio, 0.5 + 1.5*ratio
                            else: tw, bw, dw = 1.0, 1.0, 1.0
                            total_loss = (tw*text_loss + bw*base_char_loss + dw*diacritic_loss) / grad_accumulation_steps
                        scaler.scale(total_loss).backward() # Backward AMP
                    else: # FP32
                        outputs = model(pixel_values=pixel_values, labels=labels)
                        # Vectorized Loss Calc & Weighting (FP32)
                        text_loss = outputs.get('loss', torch.tensor(0.0, device=device))
                        base_logits, diac_logits = outputs.get('base_char_logits'), outputs.get('diacritic_logits')
                        base_char_loss, diacritic_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                        if base_logits is not None:
                            valid_len = base_char_indices.size(1); logits_l = base_logits[:, :valid_len, :]
                            if logits_l.shape[1] == valid_len:
                                targets_r = base_char_indices[:,:valid_len].reshape(-1); logits_r = logits_l.reshape(-1, EXPECTED_BASE_CLASSES)
                                if targets_r.numel() > 0: base_char_loss = ce_loss(logits_r, targets_r)
                        if diac_logits is not None:
                            valid_len = diacritic_indices.size(1); logits_l = diac_logits[:, :valid_len, :]
                            if logits_l.shape[1] == valid_len:
                                targets_r = diacritic_indices[:,:valid_len].reshape(-1); logits_r = logits_l.reshape(-1, EXPECTED_DIAC_CLASSES)
                                if targets_r.numel() > 0: diacritic_loss = focal_loss(logits_r, targets_r)
                        # Weighting (Use num_curriculum_stages_arg)
                        if dynamic_loss_weighting and num_curriculum_stages_arg > 1:
                            ratio = train_curriculum.current_stage / max(1, num_curriculum_stages_arg - 1)
                            tw, bw, dw = 0.7 + 0.3*ratio, 0.8 + 0.2*ratio, 0.5 + 1.5*ratio
                        else: tw, bw, dw = 1.0, 1.0, 1.0
                        total_loss = (tw*text_loss + bw*base_char_loss + dw*diacritic_loss) / grad_accumulation_steps
                        total_loss.backward() # Backward FP32

                    # Accumulate Epoch Losses
                    epoch_train_loss += total_loss.item() * grad_accumulation_steps
                    epoch_text_loss += text_loss.item()
                    epoch_base_loss += base_char_loss.item()
                    epoch_diacritic_loss += diacritic_loss.item()
                    batches_processed_in_epoch += 1

                    # --- Optimizer Step ---
                    if (step + 1) % grad_accumulation_steps == 0:
                        if use_amp: scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        if use_amp: scaler.step(optimizer); scaler.update()
                        else: optimizer.step()
                        if lr_scheduler is not None: lr_scheduler.step()
                        optimizer.zero_grad()
                        optimizer_steps += 1

                        # Log Batch Metrics
                        if optimizer_steps % log_interval == 0 and wandb_run:
                             current_lr = optimizer.param_groups[0]['lr']
                             wandb_run.log({
                                 "train/batch_loss": total_loss.item() * grad_accumulation_steps,
                                 "train/batch_text_loss": text_loss.item(), "train/batch_base_char_loss": base_char_loss.item(),
                                 "train/batch_diacritic_loss": diacritic_loss.item(), "train/learning_rate": current_lr,
                                 "curriculum/current_stage": train_curriculum.current_stage + 1,
                                 "train/step": optimizer_steps
                             })

                    # --- Run Evaluation Step ---
                    if eval_steps is not None and optimizer_steps > 0 and optimizer_steps % eval_steps == 0:
                        logger.info(f"Running evaluation at step {optimizer_steps}...")
                        val_metrics = compute_validation_metrics(model, val_loader, device, logger)
                        if wandb_run: wandb_run.log({f"val_step/{k}": v for k, v in val_metrics.items()}, step=optimizer_steps)

                        # Check/Save Best Step Checkpoint
                        if val_metrics['val_loss'] < best_val_loss:
                             best_val_loss = val_metrics['val_loss']
                             step_state = {
                                 'epoch': epoch, 'step': optimizer_steps, 'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(), 'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                                 'scaler_state_dict': scaler.state_dict() if scaler else None, 'curriculum_stage': train_curriculum.current_stage,
                                 'best_val_loss': best_val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}
                             }
                             step_chkpt_path = os.path.join(checkpoints_dir, f"best_step_{optimizer_steps}.pt")
                             save_checkpoint(step_state, step_chkpt_path) # Use helper
                             logger.info(f"Saved new best step checkpoint to {step_chkpt_path} (Val Loss: {best_val_loss:.4f})")
                             no_improvement_count = 0
                             # Store best state for potential later restoration
                             best_model_state = step_state
                        else: no_improvement_count += 1
                        model.train() # Back to training mode

                # --- Handle Batch Error ---
                except Exception as batch_e:
                    logger.error(f"Error processing batch {step} in epoch {current_epoch_num}: {batch_e}", exc_info=True)
                    if 'CUDA' in str(batch_e): logger.warning("CUDA error detected...")
                    logger.warning(f"Skipping batch {step}.")
                    if (step + 1) % grad_accumulation_steps != 0: optimizer.zero_grad()
                    continue

            # --- End of Epoch ---
            avg_train_loss = epoch_train_loss / batches_processed_in_epoch if batches_processed_in_epoch > 0 else 0
            avg_text_loss = epoch_text_loss / batches_processed_in_epoch if batches_processed_in_epoch > 0 else 0
            avg_base_loss = epoch_base_loss / batches_processed_in_epoch if batches_processed_in_epoch > 0 else 0
            avg_diac_loss = epoch_diacritic_loss / batches_processed_in_epoch if batches_processed_in_epoch > 0 else 0
            current_history["epoch_losses"].append(avg_train_loss)

            # Validation
            logger.info(f"Running end-of-epoch {current_epoch_num} validation...")
            val_metrics = compute_validation_metrics(model, val_loader, device, logger)
            current_history["val_losses"].append(val_metrics['val_loss'])
            current_history["word_accs"].append(val_metrics['word_acc'])
            current_history["char_accs"].append(val_metrics['char_acc'])
            current_history["base_char_accs"].append(val_metrics['base_char_acc'])
            current_history["diacritic_accs"].append(val_metrics['diacritic_acc'])

            # Log Epoch Metrics
            if wandb_run:
                log_data_epoch = {
                    "epoch": current_epoch_num, "train/loss_epoch": avg_train_loss, "train/text_loss_epoch": avg_text_loss,
                    "train/base_char_loss_epoch": avg_base_loss, "train/diacritic_loss_epoch": avg_diac_loss,
                    **{f"val/{k}": v for k, v in val_metrics.items()}, # Log all validation metrics
                    "curriculum/stage": train_curriculum.current_stage + 1,
                    "curriculum/total_stages": num_curriculum_stages_arg, # Use arg
                    "progress/no_improvement_count": no_improvement_count
                }
                wandb_run.log(log_data_epoch)

            # Print Epoch Summary
            logger.info(f'--- Epoch {current_epoch_num}/{epochs} Summary ---')
            logger.info(f'Curriculum Stage: {train_curriculum.current_stage + 1}/{num_curriculum_stages_arg}') # Use arg
            logger.info(f'Avg Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics["val_loss"]:.4f}')
            logger.info(f'Val Word Acc: {val_metrics["word_acc"]:.4f} | Val Char Acc: {val_metrics["char_acc"]:.4f}')
            logger.info(f'Val Base Acc: {val_metrics["base_char_acc"]:.4f} | Val Diac Acc: {val_metrics["diacritic_acc"]:.4f}')
            logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.7f}')
            logger.info('---------------------')

            # Prepare State for Checkpointing
            current_epoch_state = {
                'epoch': epoch, # Save epoch that just finished (0-based)
                'step': optimizer_steps,
                'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'curriculum_stage': train_curriculum.current_stage,
                'best_val_loss': best_val_loss, # Track best loss seen so far
                **{f"val_{k}": v for k, v in val_metrics.items()} # Save current validation metrics
            }

            # Save Best Epoch Checkpoint
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                no_improvement_count = 0
                current_epoch_state['best_val_loss'] = best_val_loss # Update state with new best loss
                best_chkpt_path = os.path.join(checkpoints_dir, f"best_epoch_{current_epoch_num}.pt")
                save_checkpoint(current_epoch_state, best_chkpt_path) # Save as best epoch
                logger.info(f"Saved new best epoch checkpoint to {best_chkpt_path} (Val Loss: {best_val_loss:.4f})")
                # Store the best state separately if needed for final restoration
                best_model_state = current_epoch_state
                # Optionally save HF format
                best_model_hf_path = os.path.join(output_dir, "best_model_hf")
                try: model.save_pretrained(best_model_hf_path)
                except Exception as e: logger.error(f"Could not save best HF model: {e}")
            else:
                no_improvement_count += 1

            # Save Latest Checkpoint (Always save at end of epoch)
            latest_chkpt_path = os.path.join(checkpoints_dir, "latest_checkpoint.pt")
            save_checkpoint(current_epoch_state, latest_chkpt_path)

            # Early Stopping Check
            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered after epoch {current_epoch_num}.")
                break

            # Automatic Curriculum Advancement
            if stage_epochs is None and no_improvement_count >= stage_patience:
                 if train_curriculum.current_stage < num_curriculum_stages_arg - 1: # Use arg
                      logger.info(f"Validation loss plateaued for {stage_patience} epochs. Advancing curriculum stage.")
                      if current_history["epoch_losses"]: stage_histories.append(current_history)
                      train_curriculum.advance_stage(); next_stage = train_curriculum.current_stage
                      no_improvement_count = 0
                      current_history = { # Reset history
                          "stage": next_stage, "epoch_losses": [], "val_losses": [],
                          "word_accs": [], "char_accs": [], "base_char_accs": [], "diacritic_accs": []
                      }
                      if wandb_run: wandb_run.log({"curriculum/stage_transition": next_stage + 1, "epoch": current_epoch_num})
                 else: logger.info("Reached final curriculum stage and validation loss plateaued.")

            # End of Epoch Cleanup
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- End of Training Loop (while epochs) ---

    # --- Handle Exceptions from Outer Loop ---
    except KeyboardInterrupt:
        training_exception = "KeyboardInterrupt"
        logger.warning("--- Training interrupted by user (KeyboardInterrupt) ---")
    except Exception as e:
        training_exception = e
        logger.error(f"--- Training failed with unexpected error: {e} ---", exc_info=True)

    # --- Post-Training / Exception Handling ---
    finally:
        final_epoch = epoch if 'epoch' in locals() else start_epoch - 1
        logger.info(f"Training loop ended. Final epoch completed: {final_epoch}. Optimizer steps: {optimizer_steps}")

        # Save final/interrupt/emergency checkpoint state
        if training_exception or True: # Always save final state here
            checkpoint_type = "latest" # Default to latest if loop finished
            if training_exception:
                 checkpoint_type = "interrupt" if isinstance(training_exception, KeyboardInterrupt) else "emergency"
            logger.info(f"Attempting to save final state as {checkpoint_type}_checkpoint.pt...")
            try:
                final_state = {
                    'epoch': final_epoch, # Save the last completed epoch
                    'step': optimizer_steps, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if scaler else None, 'curriculum_stage': train_curriculum.current_stage,
                    'best_val_loss': best_val_loss,
                }
                final_chkpt_path = os.path.join(checkpoints_dir, f"{checkpoint_type}_checkpoint.pt")
                # Avoid overwriting latest if saving interrupt/emergency
                if checkpoint_type != "latest" or not os.path.exists(os.path.join(checkpoints_dir, "latest_checkpoint.pt")):
                     save_checkpoint(final_state, final_chkpt_path)
                elif checkpoint_type == "latest":
                      save_checkpoint(final_state, final_chkpt_path) # Ensure latest is saved if loop finishes

            except Exception as final_save_e:
                logger.error(f"Could not save {checkpoint_type} checkpoint: {final_save_e}")

        # Add final stage history
        if current_history["epoch_losses"]: stage_histories.append(current_history)
        if stage_histories:
            try: log_curriculum_learning_curves(stage_histories, wandb_run)
            except Exception as curve_e: logger.error(f"Failed to log learning curves: {curve_e}")

        # Save Final Model & Metadata (if loop completed normally)
        if not training_exception:
            logger.info("Training completed normally.")
            # Restore best model before final save?
            if best_model_state is not None:
                 logger.info(f"Restoring best model state (Epoch {best_model_state['epoch'] + 1}) before final save.")
                 try: model.load_state_dict(best_model_state['model_state_dict'])
                 except Exception as load_e: logger.error(f"Could not load best model state for final save: {load_e}")

            final_model_hf_path = os.path.join(output_dir, "final_model_hf")
            try:
                 model.save_pretrained(final_model_hf_path)
                 logger.info(f"Saved final model in HF format to {final_model_hf_path}")
            except Exception as save_e: logger.error(f"Could not save final HF model: {save_e}")

            # Save curriculum metadata
            try:
                final_meta = {
                    "curriculum_strategy": train_curriculum.strategy if hasattr(train_curriculum, 'strategy') else 'unknown',
                    "curriculum_stages": num_curriculum_stages_arg, # Use arg
                    "thresholds": [float(t) if t != float('inf') else "inf" for t in train_curriculum.thresholds] if hasattr(train_curriculum, 'thresholds') else [],
                    "final_stage": train_curriculum.current_stage,
                    "best_val_loss": best_val_loss if best_val_loss != float('inf') else None,
                    "best_epoch": best_model_state['epoch'] + 1 if best_model_state else None,
                    "best_step": best_model_state['step'] if best_model_state else None,
                    "total_epochs_completed": final_epoch + 1, # Use final epoch value
                    "stage_histories": make_json_serializable(stage_histories)
                }
                with open(os.path.join(output_dir, "curriculum_metadata.json"), "w", encoding='utf-8') as f:
                    json.dump(final_meta, f, indent=2, ensure_ascii=False)
            except Exception as meta_e: logger.error(f"Could not save curriculum metadata: {meta_e}")

        # Finish WandB Run
        if wandb_run:
            exit_code = 1 if training_exception else 0
            logger.info(f"Finishing wandb run (exit code: {exit_code})...")
            try:
                 wandb_run.save(os.path.join(output_dir,"*.json"))
                 wandb_run.save(os.path.join(output_dir,"*.log"))
                 wandb_run.finish(exit_code=exit_code)
                 logger.info("Wandb run finished.")
            except Exception as wandb_e: logger.error(f"Error finishing wandb run: {wandb_e}")

        # Re-raise exception if training failed unexpectedly
        if training_exception and not isinstance(training_exception, KeyboardInterrupt):
             raise training_exception

    return model
# ==============================================================================
# <<< --- END: Full train_bartpho_model_with_curriculum Implementation --- >>>
# ==============================================================================

def save_checkpoint(state, filepath):
    """Saves training state to a checkpoint file."""
    try:
        # Use the global logger if available, otherwise print
        try:
            logger.info(f"Saving checkpoint to {filepath}...")
        except NameError: # logger might not be defined if this file is imported
             print(f"Saving checkpoint to {filepath}...")

        torch.save(state, filepath)

        try:
            logger.info(f"Checkpoint saved successfully.")
        except NameError:
             print(f"Checkpoint saved successfully.")

    except Exception as e:
        try:
            logger.error(f"Error saving checkpoint to {filepath}: {e}", exc_info=True)
        except NameError:
            print(f"Error saving checkpoint to {filepath}: {e}")
