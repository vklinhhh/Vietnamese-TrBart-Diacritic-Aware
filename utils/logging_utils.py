import tempfile
import matplotlib.pyplot as plt
import wandb


def log_curriculum_learning_curves(stage_histories, wandb_run=None):
    """
    Log learning curves for each curriculum stage to wandb and as plots.
    
    Args:
        stage_histories: List of dictionaries with stage training history
        wandb_run: Optional wandb run for logging
    """
    # Create a figure showing how metrics evolved across stages
    plt.figure(figsize=(15, 10))
    
    # Create 2x2 subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Colors for different stages
    stage_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot training loss
    ax = axes[0, 0]
    for i, history in enumerate(stage_histories):
        color = stage_colors[i % len(stage_colors)]
        epochs = range(1, len(history["epoch_losses"]) + 1)
        ax.plot(epochs, history["epoch_losses"], '-', color=color, label=f"Stage {i+1}")
    ax.set_title('Training Loss by Curriculum Stage')
    ax.set_xlabel('Epochs within Stage')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot validation loss
    ax = axes[0, 1]
    for i, history in enumerate(stage_histories):
        color = stage_colors[i % len(stage_colors)]
        epochs = range(1, len(history["val_losses"]) + 1)
        ax.plot(epochs, history["val_losses"], '-', color=color, label=f"Stage {i+1}")
    ax.set_title('Validation Loss by Curriculum Stage')
    ax.set_xlabel('Epochs within Stage')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot word accuracy
    ax = axes[1, 0]
    for i, history in enumerate(stage_histories):
        color = stage_colors[i % len(stage_colors)]
        epochs = range(1, len(history["word_accs"]) + 1)
        ax.plot(epochs, history["word_accs"], '-', color=color, label=f"Stage {i+1}")
    ax.set_title('Word Accuracy by Curriculum Stage')
    ax.set_xlabel('Epochs within Stage')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Plot character-level accuracies
    ax = axes[1, 1]
    for i, history in enumerate(stage_histories):
        color = stage_colors[i % len(stage_colors)]
        epochs = range(1, len(history["char_accs"]) + 1)
        ax.plot(epochs, history["char_accs"], '-', color=color, label=f"Char Stage {i+1}")
        
        if "base_char_accs" in history and len(history["base_char_accs"]) > 0:
            ax.plot(epochs, history["base_char_accs"], '--', color=color, label=f"Base Stage {i+1}")
            
        if "diacritic_accs" in history and len(history["diacritic_accs"]) > 0:
            ax.plot(epochs, history["diacritic_accs"], ':', color=color, label=f"Diac Stage {i+1}")
    
    ax.set_title('Character-Level Accuracies by Curriculum Stage')
    ax.set_xlabel('Epochs within Stage')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure or log to wandb
    if wandb_run:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name)
            wandb_run.log({"curriculum/learning_curves": wandb.Image(tmp.name)})
    else:
        plt.savefig("curriculum_learning_curves.png")
        
    plt.close()
    
    # Also create a consolidated view showing progression across all epochs
    plt.figure(figsize=(15, 10))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prepare consolidated data
    all_train_losses = []
    all_val_losses = []
    all_word_accs = []
    all_char_accs = []
    all_base_char_accs = []
    all_diacritic_accs = []
    stage_boundaries = [0]  # Start with epoch 0
    
    for history in stage_histories:
        all_train_losses.extend(history["epoch_losses"])
        all_val_losses.extend(history["val_losses"])
        all_word_accs.extend(history["word_accs"])
        all_char_accs.extend(history["char_accs"])
        
        if "base_char_accs" in history:
            all_base_char_accs.extend(history["base_char_accs"])
            
        if "diacritic_accs" in history:
            all_diacritic_accs.extend(history["diacritic_accs"])
            
        # Add boundary for next stage
        stage_boundaries.append(stage_boundaries[-1] + len(history["epoch_losses"]))
    
    # Create x-axis values (epoch numbers)
    epochs = range(1, len(all_train_losses) + 1)
    
    # Plot training and validation loss
    ax = axes[0, 0]
    ax.plot(epochs, all_train_losses, 'b-', label='Training Loss')
    ax.plot(epochs, all_val_losses, 'r-', label='Validation Loss')
    # Add stage boundary lines
    for i, boundary in enumerate(stage_boundaries[1:-1], 1):
        ax.axvline(x=boundary, color='k', linestyle='--', alpha=0.7)
        ax.text(boundary, max(all_train_losses) * 0.9, f'Stage {i+1}', 
                rotation=90, verticalalignment='top')
    ax.set_title('Loss Across All Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot word accuracy
    ax = axes[0, 1]
    ax.plot(epochs, all_word_accs, 'g-', label='Word Accuracy')
    # Add stage boundary lines
    for i, boundary in enumerate(stage_boundaries[1:-1], 1):
        ax.axvline(x=boundary, color='k', linestyle='--', alpha=0.7)
        ax.text(boundary, min(all_word_accs) * 1.1, f'Stage {i+1}', 
                rotation=90, verticalalignment='bottom')
    ax.set_title('Word Accuracy Across All Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Plot character accuracies
    ax = axes[1, 0]
    ax.plot(epochs, all_char_accs, 'c-', label='Character Accuracy')
    
    if all_base_char_accs:
        ax.plot(epochs, all_base_char_accs, 'm--', label='Base Character Accuracy')
        
    if all_diacritic_accs:
        ax.plot(epochs, all_diacritic_accs, 'y:', label='Diacritic Accuracy')
        
    # Add stage boundary lines
    for i, boundary in enumerate(stage_boundaries[1:-1], 1):
        ax.axvline(x=boundary, color='k', linestyle='--', alpha=0.7)
        min_acc = min(all_char_accs) if all_char_accs else 0
        ax.text(boundary, min_acc * 1.1, f'Stage {i+1}', 
                rotation=90, verticalalignment='bottom')
    ax.set_title('Character-Level Accuracies Across All Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Plot curriculum summary
    ax = axes[1, 1]
    ax.text(0.5, 0.5, "Curriculum Summary", 
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=14)
    ax.set_title('Curriculum Learning Progress')
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure or log to wandb
    if wandb_run:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name)
            wandb_run.log({"curriculum/consolidated_curves": wandb.Image(tmp.name)})
    else:
        plt.savefig("curriculum_consolidated_curves.png")
        
    plt.close()