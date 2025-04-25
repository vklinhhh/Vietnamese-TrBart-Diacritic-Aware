# improved_bartpho_curriculum.py
"""
Enhanced curriculum learning implementation for BartPho Vietnamese OCR model
with improved error handling and visualization.
"""

import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import logging
import tempfile
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb
import unicodedata

class ImprovedCurriculumDataset(Dataset):
    """Enhanced dataset wrapper that implements curriculum learning strategy for BartPho"""
    
    def __init__(self, dataset, curriculum_strategy="combined", curriculum_stages=3, 
                 min_examples_per_stage=100, logger=None):
        """
        Initialize curriculum dataset wrapper.
        
        Args:
            dataset: Base dataset to wrap
            curriculum_strategy: Strategy for determining difficulty ('length', 'complexity', or 'combined')
            curriculum_stages: Number of curriculum stages
            min_examples_per_stage: Minimum number of examples required per stage
            logger: Optional logger instance
        """
        self.dataset = dataset
        self.curriculum_strategy = curriculum_strategy
        self.curriculum_stages = curriculum_stages
        self.min_examples_per_stage = min_examples_per_stage
        
        # Set up logging
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("ImprovedCurriculumDataset")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)
        
        # Current curriculum stage (0-indexed)
        self.current_stage = 0
        
        # Cache for score calculation
        self.difficulty_cache = {}
        
        # Calculate difficulty scores for all examples
        self.logger.info(f"Calculating difficulty scores using '{curriculum_strategy}' strategy...")
        self.difficulty_scores = self._calculate_difficulty_scores()
        
        # Compute stage thresholds and distribution
        self._compute_stage_thresholds()
        
        # Get indices for current stage
        self.current_indices = self._get_indices_for_stage(self.current_stage)
        
        self.logger.info(f"Initialized curriculum with {curriculum_stages} stages.")
        self.logger.info(f"Stage 1 (current): {len(self.current_indices)} examples")

    def _calculate_difficulty_scores(self):
        """Calculate difficulty score for each example based on the chosen strategy."""
        difficulty_scores = []
        
        for idx in tqdm(range(len(self.dataset)), desc="Calculating difficulty scores"):
            try:
                # Check cache first
                if idx in self.difficulty_cache:
                    difficulty_scores.append(self.difficulty_cache[idx])
                    continue
                
                # Get sample directly through the dataset's __getitem__
                sample = self.dataset[idx]
                
                # Get the word and diacritic information
                word, diacritic_count = self._extract_difficulty_factors(sample)
                
                # Calculate score based on strategy
                if self.curriculum_strategy == 'length':
                    # Simple length-based scoring
                    score = len(word)
                
                elif self.curriculum_strategy == 'complexity':
                    # Diacritic complexity only
                    score = diacritic_count
                
                elif self.curriculum_strategy == 'combined':
                    # Combine length and diacritic complexity
                    text_length = len(word)
                    
                    # Calculate complexity score
                    complexity_factor = self._calculate_complexity_factor(word, diacritic_count)
                    
                    # Combined score with appropriate weighting
                    score = text_length + complexity_factor
                
                else:
                    raise ValueError(f"Unknown curriculum strategy: {self.curriculum_strategy}")
                
                # Cache the result
                self.difficulty_cache[idx] = score
                difficulty_scores.append(score)
            
            except Exception as e:
                self.logger.warning(f"Error processing sample {idx} for curriculum: {e}")
                # Assign medium difficulty to problematic samples
                difficulty_scores.append(5)  # Medium difficulty as fallback
        
        return np.array(difficulty_scores)
    
    def _extract_difficulty_factors(self, sample):
        """Extract word and diacritic information from sample"""
        # Default values
        word = ""
        diacritic_count = 0
        
        try:
            # Extract word
            if 'word' in sample:
                word = sample['word']
            elif 'full_characters' in sample:
                if isinstance(sample['full_characters'], list):
                    word = ''.join(sample['full_characters'])
                else:
                    word = sample['full_characters']
            elif 'label' in sample:
                word = sample['label']
            
            # Extract diacritics
            if 'diacritic_indices' in sample:
                diacritic_indices = sample['diacritic_indices']
                # Count non-zero diacritics (assuming 0 is no_diacritic)
                diacritic_count = (diacritic_indices > 0).sum().item()
            else:
                # Count diacritics by analyzing the word
                diacritic_count = self._count_diacritics_in_word(word)
                
        except Exception as e:
            self.logger.warning(f"Error extracting difficulty factors: {e}")
        
        return word, diacritic_count
    
    def _count_diacritics_in_word(self, word):
        """Count diacritics in a word by analyzing its characters"""
        diacritic_count = 0
        for char in word:
            # Normalize the character to decomposed form
            norm_char = unicodedata.normalize('NFD', char)
            # If the normalization produces more characters, it has diacritics
            if len(norm_char) > 1:
                diacritic_count += 1
        return diacritic_count
    
    def _calculate_complexity_factor(self, word, diacritic_count):
        """Calculate complexity factor based on word characteristics"""
        # Base factor from diacritic count
        complexity = diacritic_count * 2
        
        # Additional factors:
        # 1. Percentage of characters with diacritics
        if len(word) > 0:
            diacritic_percentage = diacritic_count / len(word)
            # Add up to 2 points for high diacritic density
            complexity += diacritic_percentage * 2
        
        # 2. Check for special Vietnamese characters like ơ, ư which are harder
        special_chars = ['ơ', 'ư', 'ă', 'â', 'ê', 'ô', 'Ơ', 'Ư', 'Ă', 'Â', 'Ê', 'Ô']
        special_count = sum(1 for c in word if c in special_chars)
        complexity += special_count
        
        return complexity

    def _compute_stage_thresholds(self):
        """Compute difficulty thresholds for each curriculum stage."""
        # Get unique difficulty scores and their counts
        unique_scores, counts = np.unique(self.difficulty_scores, return_counts=True)
        
        # Sort scores
        sorted_indices = np.argsort(unique_scores)
        unique_scores = unique_scores[sorted_indices]
        counts = counts[sorted_indices]
        
        # Calculate cumulative counts
        cumulative_counts = np.cumsum(counts)
        total_examples = cumulative_counts[-1]
        
        # Calculate target examples per stage for even distribution
        target_per_stage = total_examples / self.curriculum_stages
        
        # Find thresholds that divide examples most evenly
        thresholds = []
        for stage in range(self.curriculum_stages - 1):
            target_count = (stage + 1) * target_per_stage
            idx = np.argmin(np.abs(cumulative_counts - target_count))
            thresholds.append(unique_scores[idx])
        
        # Add maximum threshold
        thresholds.append(float('inf'))
        
        self.thresholds = thresholds
        
        # Calculate and log stage distribution
        prev_count = 0
        self.stage_distributions = []
        
        for i, thresh in enumerate(thresholds):
            # Count examples below this threshold
            count = np.sum(self.difficulty_scores <= thresh)
            stage_count = count - prev_count
            prev_count = count
            
            self.stage_distributions.append({
                "stage": i+1,
                "count": int(stage_count),
                "threshold": float(thresh) if thresh != float('inf') else None,
                "percent": float(stage_count / len(self.difficulty_scores) * 100)
            })
            
            self.logger.info(f"Stage {i+1}: {stage_count} examples (difficulty <= {thresh})")
            
            # Check if any stage has too few examples
            if stage_count < self.min_examples_per_stage:
                self.logger.warning(f"Stage {i+1} has only {stage_count} examples, "
                                    f"which is less than minimum {self.min_examples_per_stage}")

    def _get_indices_for_stage(self, stage):
        """Get dataset indices for a specific curriculum stage."""
        assert 0 <= stage < self.curriculum_stages, f"Invalid stage: {stage}"
        
        lower_threshold = 0 if stage == 0 else self.thresholds[stage - 1]
        upper_threshold = self.thresholds[stage]
        
        # Get indices where difficulty score is in the appropriate range
        indices = np.where(
            (self.difficulty_scores > lower_threshold) & 
            (self.difficulty_scores <= upper_threshold)
        )[0]
        
        # Convert numpy int64 to python int
        indices = [int(idx) for idx in indices]
        
        return indices

    def set_stage(self, stage):
        """Set curriculum to a specific stage."""
        assert 0 <= stage < self.curriculum_stages, f"Invalid stage: {stage}"
        
        self.current_stage = stage
        self.current_indices = self._get_indices_for_stage(stage)
        
        # Log the stage change
        self.logger.info(f"Switched to curriculum stage {stage + 1}/{self.curriculum_stages}")
        self.logger.info(f"Stage {stage + 1}: {len(self.current_indices)} examples")
        
        return len(self.current_indices)

    def advance_stage(self):
        """Advance to the next curriculum stage if possible."""
        if self.current_stage < self.curriculum_stages - 1:
            return self.set_stage(self.current_stage + 1)
        else:
            self.logger.info("Already at final curriculum stage.")
            return len(self.current_indices)

    def __len__(self):
        """Return the number of examples in the current stage."""
        return len(self.current_indices)

    def __getitem__(self, idx):
        """Get item from the current stage."""
        try:
            # Map the local index to the correct dataset index
            dataset_idx = self.current_indices[idx]
            # Ensure we're using a Python int, not numpy.int64
            dataset_idx = int(dataset_idx)
            # Return the actual item
            return self.dataset[dataset_idx]
        except Exception as e:
            self.logger.error(f"Error retrieving item {idx} (maps to dataset index {dataset_idx}): {e}")
            # Return a dummy item that won't break training
            if hasattr(self.dataset, '_create_dummy_sample'):
                return self.dataset._create_dummy_sample()
            elif hasattr(self.dataset, '__getitem__'):
                # Try to get any valid item as a fallback
                for i in range(min(100, len(self.dataset))):
                    try:
                        return self.dataset[i]
                    except:
                        continue
            
            # Last resort fallback
            return {"error": "Could not retrieve item"}

    def get_all_data(self):
        """Return a dataset with all examples (for evaluation)."""
        return self.dataset

    def get_current_stage_dataset(self):
        """Return a Subset dataset for the current stage (useful for DataLoader)."""
        return Subset(self.dataset, self.current_indices)

    def log_curriculum_stats(self, wandb_run=None):
        """Log curriculum statistics to wandb and console."""
        try:
            # Create histogram of difficulty scores
            plt.figure(figsize=(10, 6))
            
            # Plot histogram
            plt.hist(self.difficulty_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add threshold lines
            for i, thresh in enumerate(self.thresholds[:-1]):  # Skip the last threshold (inf)
                plt.axvline(x=thresh, color='r', linestyle='--', 
                         label=f'Stage {i+1}/{i+2} threshold' if i == 0 else None)
            
            # Add labels
            plt.xlabel('Difficulty Score')
            plt.ylabel('Number of Examples')
            plt.title(f'Curriculum Learning Difficulty Distribution ({self.curriculum_strategy} strategy)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Log to wandb if available
            if wandb_run:
                # Save figure to a temporary file that wandb can handle
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    plt.savefig(tmp.name)
                    wandb_run.log({"curriculum/difficulty_distribution": wandb.Image(tmp.name)})
                
                # Log stage distribution as a table
                columns = ["Stage", "Example Count", "Max Difficulty", "% of Total"]
                data = [[s["stage"], s["count"], 
                       "inf" if s["threshold"] is None else f"{s['threshold']:.1f}", 
                       f"{s['percent']:.1f}%"] 
                      for s in self.stage_distributions]
                
                wandb_run.log({"curriculum/stage_statistics": wandb.Table(
                    columns=columns, data=data
                )})
            
            # Show the plot if not using wandb
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error logging curriculum stats: {e}")
            self.logger.info("Training will continue, but visualization may be incomplete.")


def create_improved_curriculum_datasets(
    train_dataset, 
    val_dataset, 
    curriculum_strategy="combined", 
    curriculum_stages=3, 
    min_examples_per_stage=100,
    logger=None,
    wandb_run=None
):
    """
    Create enhanced curriculum learning wrapper datasets for training and validation.
    
    Args:
        train_dataset: Training dataset to wrap
        val_dataset: Validation dataset to wrap (will not be curriculum filtered)
        curriculum_strategy: Strategy for determining difficulty
        curriculum_stages: Number of curriculum stages
        min_examples_per_stage: Minimum number of examples required per stage
        logger: Optional logger instance
        wandb_run: Optional wandb run instance for logging
    
    Returns:
        train_curriculum: ImprovedCurriculumDataset for training
        val_dataset: Original validation dataset (unchanged)
    """
    # Create logger if not provided
    if not logger:
        logger = logging.getLogger("CurriculumCreator")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
    
    # Create curriculum wrapper for training data
    logger.info(f"Creating curriculum with {curriculum_strategy} strategy and {curriculum_stages} stages")
    train_curriculum = ImprovedCurriculumDataset(
        train_dataset,
        curriculum_strategy=curriculum_strategy,
        curriculum_stages=curriculum_stages,
        min_examples_per_stage=min_examples_per_stage,
        logger=logger
    )
    
    # Log curriculum statistics
    train_curriculum.log_curriculum_stats(wandb_run)
    
    return train_curriculum, val_dataset