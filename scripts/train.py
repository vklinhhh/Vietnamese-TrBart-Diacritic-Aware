import os
import sys
import argparse
import torch
from datasets import load_dataset
import logging
import math
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp

# Import custom modules
from model.bartpho_ocr import BartPhoVietOCR
from data.dataset import ImprovedBartPhoDataset
from data.curriculum import create_improved_curriculum_datasets
from utils.schedulers import CosineWarmupScheduler
from utils.optimizers import create_optimizer
from training.trainer import train_bartpho_model_with_curriculum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), # Ensure logs go to console
        logging.FileHandler('bartpho_training.log')
    ]
)
logger = logging.getLogger('BartPhoTraining')

def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train BartPho Vietnamese OCR model with curriculum learning")

    # --- Arguments ---
    # Dataset and model parameters
    parser.add_argument('--dataset_name', type=str, default='vklinhhh/vietnamese_character_diacritic_cwl_v2', help='HuggingFace dataset name')
    parser.add_argument('--vision_encoder', type=str, default='microsoft/trocr-base-handwritten', help='Vision encoder model name')
    parser.add_argument('--bartpho_model', type=str, default='vinai/bartpho-syllable-base', help='BartPho model name')
    parser.add_argument('--output_dir', type=str, default='vietnamese-ocr-bartpho-curriculum', help='Directory to save the model and checkpoints')
    # Resuming
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint file to resume training. If None, checks for "latest_checkpoint.pt" in output_dir/checkpoints.')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=15, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Base learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio if dataset needs splitting')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length for model')
    parser.add_argument('--rank', type=int, default=16, help='Rank parameter for Rethinking module')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Portion of training steps for LR warmup')
    parser.add_argument('--grad_accumulation', type=int, default=1, help='Gradient accumulation steps')
    # Loss parameters
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    parser.add_argument('--dynamic_loss_weighting', action='store_true', help='Dynamically adjust loss component weights')
    # Curriculum learning parameters
    parser.add_argument('--curriculum_strategy', type=str, default='combined', choices=['length', 'complexity', 'combined'], help='Curriculum difficulty strategy')
    parser.add_argument('--curriculum_stages', type=int, default=3, help='Number of curriculum stages')
    parser.add_argument('--stage_epochs', type=str, default=None, help='Comma-separated epochs per stage (e.g., "5,5,5"). Overrides auto progression.')
    parser.add_argument('--stage_patience', type=int, default=5, help='Epochs to wait before auto-advancing stage')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Epochs to wait before early stopping')
    # Logging parameters
    parser.add_argument('--wandb_project', type=str, default='vietnamese-ocr-bartpho-curriculum', help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval in optimizer steps')
    parser.add_argument('--eval_steps', type=int, default=None, help='Evaluate every N optimizer steps')
    # Advanced options
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision training')
    parser.add_argument('--discriminative_lr', action='store_true', help='Use different learning rates for model parts')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers (0 to disable multiprocessing)')

    args = parser.parse_args()

    # --- Setup Multiprocessing and Device ---
    if args.num_workers > 0:
        try:
            current_method = mp.get_start_method(allow_none=True)
            if current_method != 'spawn':
                logger.info(f"Setting multiprocessing start method to 'spawn' (currently {current_method})")
                mp.set_start_method('spawn', force=True)
            else:
                 logger.info("Multiprocessing start method already set to 'spawn'.")
        except Exception as e:
            logger.warning(f"Error setting multiprocessing start method: {e}. Using default: {mp.get_start_method()}. If CUDA errors occur, try setting num_workers=0.")
    else:
         logger.info("num_workers set to 0, multiprocessing disabled.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Selected device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")


    # --- Parse Stage Epochs ---
    stage_epochs_list = None
    if args.stage_epochs:
        try:
            stage_epochs_list = [int(e) for e in args.stage_epochs.split(',')]
            assert len(stage_epochs_list) == args.curriculum_stages, "Number of stage_epochs must match curriculum_stages"
            logger.info(f"Using fixed stage epochs schedule: {stage_epochs_list}")
        except Exception as e:
            logger.error(f"Error parsing stage_epochs '{args.stage_epochs}': {e}. Falling back to automatic stage progression.")
            args.stage_epochs = None

    # --- Initialize Components (Define before loading checkpoint) ---
    model = None
    optimizer = None
    lr_scheduler = None
    scaler = GradScaler() if args.use_amp else None
    train_curriculum = None

    # --- Initialize Resume State Variables ---
    start_epoch = 0
    resumed_optimizer_steps = 0
    resumed_best_val_loss = float('inf')
    resumed_curriculum_stage = 0

    # --- Initialize Model Structure ---
    # Needed before loading weights and potentially before loading datasets/curriculum
    try:
        logger.info("Initializing model structure...")
        model = BartPhoVietOCR(
            vision_encoder_name=args.vision_encoder,
            bartpho_name=args.bartpho_model,
            max_seq_len=args.max_seq_len,
            rank=args.rank
        )
        # Log vocab sizes immediately after init
        logger.info(f"BartPho Tokenizer Vocab Size: {model.bartpho_tokenizer.vocab_size}")
        logger.info(f"Base Character Vocab Size: {len(model.base_char_vocab)}")
        logger.info(f"Diacritic Vocab Size: {len(model.diacritic_vocab)}")
        logger.info("Model structure initialized.")
    except Exception as model_init_e:
        logger.error(f"FATAL: Failed to initialize model structure: {model_init_e}", exc_info=True)
        return 1

    # --- Load Datasets (Needed for Scheduler estimation) ---
    try:
        logger.info(f"Loading dataset: {args.dataset_name}")
        hf_dataset = load_dataset(args.dataset_name)
        if 'validation' not in hf_dataset or 'train' not in hf_dataset:
            logger.warning(f"Dataset {args.dataset_name} missing 'train' or 'validation' split. Attempting to split 'train'.")
            if 'train' not in hf_dataset: raise ValueError("Dataset must have a 'train' split.")
            # Use full dataset for estimation, split later
            full_train_dataset = hf_dataset['train']
            dataset_dict = full_train_dataset.select(range(20)).train_test_split(test_size=args.val_split, seed=42)
            train_hf_split = dataset_dict['train']
            val_hf_split = dataset_dict['test']
        else:
            full_train_dataset = hf_dataset['train'] # Use for estimation
            train_hf_split = hf_dataset['train']
            val_hf_split = hf_dataset['validation']
        logger.info(f"Full Training set size (for estimation): {len(full_train_dataset)}")
    except Exception as dataset_load_e:
        logger.error(f"FATAL: Failed to load dataset: {dataset_load_e}", exc_info=True)
        return 1

    # --- Create Optimizer and Scheduler (Needed before loading state) ---
    try:
        optimizer = create_optimizer(
            model, args.learning_rate, args.weight_decay, args.discriminative_lr
        )

        # Estimate total steps for scheduler using full dataset size
        # This estimation is primarily for setting up the scheduler correctly
        if stage_epochs_list:
            stage_total_steps = []
            approx_total_train_samples = len(full_train_dataset)
            for stage_num in range(args.curriculum_stages):
                 # Distribute samples evenly for estimation if stages not size-based
                 est_stage_size = approx_total_train_samples / args.curriculum_stages
                 stage_steps = math.ceil(est_stage_size / args.batch_size / args.grad_accumulation) * stage_epochs_list[stage_num]
                 stage_total_steps.append(stage_steps)
            total_steps_for_scheduler = sum(stage_total_steps)
        else:
            estimated_total_samples = len(full_train_dataset) * args.epochs
            total_steps_for_scheduler = math.ceil(estimated_total_samples / args.batch_size / args.grad_accumulation)

        warmup_steps_for_scheduler = int(total_steps_for_scheduler * args.warmup_ratio)
        logger.info(f"Scheduler Setup: Estimated total steps={total_steps_for_scheduler}, warmup steps={warmup_steps_for_scheduler}")

        lr_scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps_for_scheduler,
            max_steps=total_steps_for_scheduler
        )
        logger.info("Optimizer and LR Scheduler created.")
    except Exception as opt_sched_e:
        logger.error(f"FATAL: Failed to create optimizer/scheduler: {opt_sched_e}", exc_info=True)
        return 1

    # --- Determine Checkpoint Path and Load State ---
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    resume_checkpoint_path = args.resume_from_checkpoint
    load_successful = False # Flag to track if loading worked

    if resume_checkpoint_path is None:
        potential_path = os.path.join(checkpoints_dir, "latest_checkpoint.pt")
        if os.path.isfile(potential_path):
            logger.info(f"Auto-detected latest checkpoint: {potential_path}")
            resume_checkpoint_path = potential_path
        else:
             logger.info("No specific checkpoint provided and 'latest_checkpoint.pt' not found. Starting fresh.")
             resume_checkpoint_path = None # Ensure it's None if not found

    if resume_checkpoint_path and os.path.isfile(resume_checkpoint_path):
        logger.info(f"--- Attempting to load checkpoint: {resume_checkpoint_path} ---")
        try:
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)
            logger.info(f"Checkpoint loaded. Keys available: {list(checkpoint.keys())}")

            # Load Model state (already initialized)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("-> Model state loaded.")

            # Load Optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("-> Optimizer state loaded.")

            # Load Scheduler state
            if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                logger.info("-> LR Scheduler state loaded.")
            else: logger.warning("-> LR Scheduler state not found/loaded.")

            # Load Scaler state
            if args.use_amp and scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("-> AMP GradScaler state loaded.")
            elif args.use_amp: logger.warning("-> AMP GradScaler state not found/loaded.")

            # Load scalar training variables
            start_epoch = checkpoint.get('epoch', 0) + 1 # Resume AFTER the saved epoch
            resumed_optimizer_steps = checkpoint.get('step', 0)
            # Load best val loss correctly (use value from checkpoint)
            resumed_best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            resumed_curriculum_stage = checkpoint.get('curriculum_stage', 0)

            logger.info(f"-> Resuming from Epoch: {start_epoch} (Step: {resumed_optimizer_steps})")
            logger.info(f"-> Resumed Best Val Loss: {resumed_best_val_loss:.4f}")
            logger.info(f"-> Resumed Curriculum Stage: {resumed_curriculum_stage}")
            load_successful = True # Mark loading as successful

        except Exception as e:
            logger.error(f"ERROR loading checkpoint from {resume_checkpoint_path}: {e}", exc_info=True)
            logger.warning("Could not fully load checkpoint state. Training will start from scratch.")
            # Reset variables to default start values
            start_epoch = 0
            resumed_optimizer_steps = 0
            resumed_best_val_loss = float('inf')
            resumed_curriculum_stage = 0
            load_successful = False
            # Consider reloading initial model weights if load fails badly?
            # model = BartPhoVietOCR(...) # Re-init model if needed
    else:
        if args.resume_from_checkpoint: # If a path was given but not found
             logger.warning(f"Specified checkpoint not found: {args.resume_from_checkpoint}. Starting fresh.")
        logger.info("Starting training from scratch (epoch 0).")

    # --- Create Dataset Wrappers and Curriculum (After model init/load) ---
    try:
        logger.info("Initializing dataset wrappers...")
        train_wrapper = ImprovedBartPhoDataset(
            train_hf_split, model.vision_processor, model.base_char_vocab, model.diacritic_vocab
        )
        val_wrapper = ImprovedBartPhoDataset(
            val_hf_split, model.vision_processor, model.base_char_vocab, model.diacritic_vocab
        )

        logger.info("Setting up curriculum learning...")
        train_curriculum, val_dataset_curriculum = create_improved_curriculum_datasets(
            train_wrapper, val_wrapper, args.curriculum_strategy, args.curriculum_stages,
            min_examples_per_stage=args.batch_size * 5, logger=logger, wandb_run=None # Wandb setup later
        )
        # Restore curriculum stage if loading was successful
        if load_successful:
             logger.info(f"Setting curriculum stage to loaded stage: {resumed_curriculum_stage}")
             train_curriculum.set_stage(resumed_curriculum_stage)
        logger.info("Curriculum setup complete.")

    except Exception as dataset_setup_e:
        logger.error(f"FATAL: Failed to setup datasets/curriculum: {dataset_setup_e}", exc_info=True)
        return 1


    # --- Move model to device (ensure it's on device after potential loading) ---
    model.to(device)

    # --- Start Training ---
    logger.info("============ Starting Training Phase ============")
    trained_model = train_bartpho_model_with_curriculum(
        # Pass initialized/loaded components and state
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        train_curriculum=train_curriculum,
        val_dataset=val_dataset_curriculum,
        start_epoch=start_epoch,
        resumed_optimizer_steps=resumed_optimizer_steps,
        resumed_best_val_loss=resumed_best_val_loss,
        # Pass args needed inside the training function
        num_curriculum_stages_arg=args.curriculum_stages,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        project_name=args.wandb_project,
        run_name=args.wandb_run_name,
        log_interval=args.log_interval,
        focal_loss_gamma=args.focal_loss_gamma,
        output_dir=args.output_dir,
        stage_epochs=stage_epochs_list,
        stage_patience=args.stage_patience,
        early_stopping_patience=args.early_stopping_patience,
        use_amp=args.use_amp,
        grad_accumulation_steps=args.grad_accumulation,
        num_workers=args.num_workers,
        dynamic_loss_weighting=args.dynamic_loss_weighting,
        eval_steps=args.eval_steps,
    )

    logger.info(f"============ Training process finished ============")
    logger.info(f"Final model artifacts should be in {args.output_dir}")
    return 0 # Indicate success

if __name__ == "__main__":
    status = main()
    sys.exit(status)