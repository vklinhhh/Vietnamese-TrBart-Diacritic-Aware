# improved_bartpho_curriculum.py
"""
Enhanced curriculum learning implementation for BartPho Vietnamese OCR model
with improved error handling and visualization. Includes optimized difficulty
score calculation by accessing raw dataset fields directly.
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
import ast  # For parsing string representations of lists


class ImprovedCurriculumDataset(Dataset):
    """
    Enhanced dataset wrapper that implements curriculum learning strategy for BartPho.
    Optimized difficulty score calculation accesses raw dataset fields directly.
    """

    def __init__(
        self,
        dataset,
        curriculum_strategy='combined',
        curriculum_stages=3,
        min_examples_per_stage=100,
        logger=None,
    ):
        """
        Initialize curriculum dataset wrapper.

        Args:
            dataset: Base dataset (e.g., ImprovedBartPhoDataset instance) to wrap.
                     This dataset instance MUST have an attribute (e.g., 'dataset')
                     that provides access to the underlying raw dataset (like a Hugging Face Dataset)
                     for efficient difficulty calculation.
            curriculum_strategy: Strategy for determining difficulty ('length', 'complexity', or 'combined')
            curriculum_stages: Number of curriculum stages
            min_examples_per_stage: Minimum number of examples required per stage
            logger: Optional logger instance
        """
        self.dataset = dataset  # This is the ImprovedBartPhoDataset instance
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

        # --- Calculate difficulty scores EFFICIENTLY ---
        self.logger.info(f"Calculating difficulty scores using '{curriculum_strategy}' strategy...")
        try:
            self.difficulty_scores = self._calculate_difficulty_scores_efficiently()
        except AttributeError as e:
            self.logger.error(f'Error accessing raw dataset for efficient calculation: {e}')
            self.logger.warning(
                'Falling back to SLOW difficulty calculation (processing each item). This may take a very long time!'
            )
            # Fallback to the slow method if needed (though ideally the efficient one should work)
            self.difficulty_scores = (
                self._calculate_difficulty_scores_slow()
            )  # Keep the slow one as a fallback for clarity
        except Exception as e:
            self.logger.error(f'Unexpected error during difficulty calculation: {e}', exc_info=True)
            raise  # Re-raise unexpected errors

        # Compute stage thresholds and distribution
        self._compute_stage_thresholds()

        # Get indices for current stage
        self.current_indices = self._get_indices_for_stage(self.current_stage)

        self.logger.info(f"Initialized curriculum with {curriculum_stages} stages.")
        self.logger.info(f"Stage 1 (current): {len(self.current_indices)} examples")

    # --- Efficient Difficulty Score Calculation ---
    def _calculate_difficulty_scores_efficiently(self):
        """
        Calculate difficulty score EFFICIENTLY by accessing only raw text fields
        from the underlying dataset, avoiding full item processing.
        """
        difficulty_scores = []

        # --- Access the RAW underlying dataset ---
        # Assumes self.dataset (ImprovedBartPhoDataset) has attribute 'dataset' pointing to raw HF dataset
        if not hasattr(self.dataset, 'dataset'):
            raise AttributeError(
                "The provided 'dataset' object does not have a 'dataset' attribute to access raw data."
            )
        raw_dataset = self.dataset.dataset

        self.logger.info(
            f'Using efficient calculation on raw dataset ({len(raw_dataset)} items)...'
        )

        for idx in tqdm(range(len(raw_dataset)), desc='Calculating difficulty scores (Efficient)'):
            try:
                if idx in self.difficulty_cache:
                    difficulty_scores.append(self.difficulty_cache[idx])
                    continue

                # Get RAW data dictionary directly
                example_dict = raw_dataset[idx]

                # Extract factors needed for scoring from the RAW dictionary
                word, diacritic_count = self._extract_difficulty_factors_from_raw(example_dict)

                # Calculate score based on strategy
                score = self._compute_score(word, diacritic_count)

                self.difficulty_cache[idx] = score
                difficulty_scores.append(score)

            except Exception as e:
                self.logger.warning(f'Error processing raw sample {idx} for curriculum: {e}')
                difficulty_scores.append(5)  # Assign medium difficulty as fallback

        return np.array(difficulty_scores)

    def _parse_if_string_list(self, value):
        """Safely parse string representations of lists (e.g., "['a','b']") into actual lists."""
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                # Use ast.literal_eval for safe evaluation of Python literals
                return ast.literal_eval(value)
            except (SyntaxError, ValueError, TypeError):
                # If parsing fails, return the original string or handle as needed
                self.logger.debug(f'Could not parse string list: {value}')
                return value  # Or potentially an empty list: return []
        return value

    def _extract_difficulty_factors_from_raw(self, example_dict):
        """
        Extracts word and diacritic count directly from a raw dataset sample dictionary.
        Avoids calling image processing or tokenization.
        """
        word = ""
        diacritic_count = 0
        full_chars = []

        # 1. Extract Word String
        # Prioritize keys that likely contain the full word/label directly
        if 'word' in example_dict and isinstance(example_dict['word'], str):
            word = example_dict['word']
        elif 'label' in example_dict and isinstance(example_dict['label'], str):
            word = example_dict['label']
        elif 'full_character' in example_dict:  # May contain list or string representation of list
            raw_chars_data = self._parse_if_string_list(example_dict['full_character'])
            if isinstance(raw_chars_data, list):
                # Filter out non-strings just in case
                valid_chars = [c for c in raw_chars_data if isinstance(c, str)]
                word = ''.join(valid_chars)
                full_chars = valid_chars  # Keep for potential diacritic counting
            elif isinstance(raw_chars_data, str):  # If it's just a single string
                word = raw_chars_data
                full_chars = list(word)  # Treat each char in string as list item
        # Add more fallbacks if your dataset uses different keys (e.g., 'text')
        elif 'text' in example_dict and isinstance(example_dict['text'], str):
            word = example_dict['text']

        # 2. Determine Diacritic Count
        # Option A: Use pre-computed diacritic type information if available
        if 'diacritic_type' in example_dict:
            raw_diac_data = self._parse_if_string_list(example_dict['diacritic_type'])
            if isinstance(raw_diac_data, list):
                # Count valid diacritic entries (assuming 'no_diacritic' or empty string means none)
                diacritic_count = sum(
                    1
                    for d in raw_diac_data
                    if isinstance(d, str) and d.strip() and d != 'no_diacritic'
                )
            elif (
                isinstance(raw_diac_data, str)
                and raw_diac_data.strip()
                and raw_diac_data != 'no_diacritic'
            ):
                # Handle case where it might be a single string for a single character word
                diacritic_count = 1
        # Option B: Calculate from the extracted word if diacritic types aren't present
        elif word:
            diacritic_count = self._count_diacritics_in_word(word)
        # Option C: Fallback using full_chars list if word wasn't formed but chars exist
        elif full_chars:
            # Reconstruct word or count directly from full_chars
            temp_word = ''.join(full_chars)
            diacritic_count = self._count_diacritics_in_word(temp_word)

        return word, diacritic_count

    def _count_diacritics_in_word(self, word):
        """Counts diacritics in a word using Unicode decomposition. Efficient."""
        diacritic_count = 0
        if not isinstance(word, str):  # Handle potential non-string input
            return 0
        for char in word:
            try:
                norm_char = unicodedata.normalize('NFD', char)
                # Check if normalization resulted in more than one character AND
                # if any of the subsequent characters are combining marks.
                if len(norm_char) > 1:
                    has_combining = any(unicodedata.combining(c) > 0 for c in norm_char[1:])
                    if has_combining:
                        diacritic_count += 1
            except TypeError:
                # Handle potential errors if char is not a valid character for normalize
                continue
        return diacritic_count

    def _compute_score(self, word, diacritic_count):
        """Computes the difficulty score based on the configured strategy."""
        if self.curriculum_strategy == 'length':
            score = len(word)
        elif self.curriculum_strategy == 'complexity':
            score = diacritic_count
        elif self.curriculum_strategy == 'combined':
            text_length = len(word)
            complexity_factor = self._calculate_complexity_factor(word, diacritic_count)
            score = text_length + complexity_factor  # Simple addition, weights can be added
        else:
            raise ValueError(f'Unknown curriculum strategy: {self.curriculum_strategy}')
        return score

    def _calculate_complexity_factor(self, word, diacritic_count):
        """Calculate complexity factor based on word characteristics (same as before)."""
        complexity = diacritic_count * 2  # Base weight for diacritics
        word_len = len(word)
        if word_len > 0:
            diacritic_density = diacritic_count / word_len
            complexity += diacritic_density * 2  # Add points for density

        # Check for specific harder Vietnamese characters
        special_chars = ['ơ', 'ư', 'ă', 'â', 'ê', 'ô', 'đ', 'Ơ', 'Ư', 'Ă', 'Â', 'Ê', 'Ô', 'Đ']
        special_count = sum(1 for c in word if c in special_chars)
        complexity += special_count  # Add points for specific chars

        return complexity

    # --- Fallback Slow Difficulty Calculation (Kept for reference/emergency) ---
    def _calculate_difficulty_scores_slow(self):
        """
        Original SLOW difficulty calculation. Processes each item fully.
        Use only as a fallback if efficient method fails unexpectedly.
        """
        difficulty_scores = []
        self.logger.warning('Executing SLOW difficulty calculation method!')

        # Iterate through the main dataset wrapper, triggering __getitem__
        for idx in tqdm(range(len(self.dataset)), desc='Calculating difficulty scores (SLOW)'):
            try:
                if idx in self.difficulty_cache:
                    difficulty_scores.append(self.difficulty_cache[idx])
                    continue

                # This calls ImprovedBartPhoDataset.__getitem__, doing full processing
                sample = self.dataset[idx]

                # This helper needs the PROCESSED sample format
                word, diacritic_count = self._extract_difficulty_factors_from_processed(sample)

                # Calculate score based on strategy
                score = self._compute_score(word, diacritic_count)

                self.difficulty_cache[idx] = score
                difficulty_scores.append(score)

            except Exception as e:
                self.logger.warning(f'Error processing sample {idx} with slow method: {e}')
                difficulty_scores.append(5)

        return np.array(difficulty_scores)

    def _extract_difficulty_factors_from_processed(self, processed_sample):
        """Extracts word and diacritic count from a PROCESSED sample dictionary."""
        word = processed_sample.get('word', '')
        diacritic_indices = processed_sample.get('diacritic_indices', torch.tensor([]))

        # Assuming 0 index in diacritic_vocab is 'no_diacritic'
        diacritic_count = 0
        if isinstance(diacritic_indices, torch.Tensor) and diacritic_indices.numel() > 0:
            diacritic_count = (diacritic_indices > 0).sum().item()
        elif isinstance(diacritic_indices, list):  # Handle list case if necessary
            diacritic_count = sum(1 for d_idx in diacritic_indices if d_idx > 0)

        # Fallback if diacritic indices missing but word present
        if diacritic_count == 0 and word:
            diacritic_count = self._count_diacritics_in_word(word)

        return word, diacritic_count

    # --- Remaining Methods (Unchanged) ---

    def _compute_stage_thresholds(self):
        """Compute difficulty thresholds for each curriculum stage."""
        if len(self.difficulty_scores) == 0:
            self.logger.warning('No difficulty scores calculated, cannot compute thresholds.')
            self.thresholds = [float('inf')] * self.curriculum_stages
            self.stage_distributions = [
                {'stage': i + 1, 'count': 0, 'threshold': None, 'percent': 0.0}
                for i in range(self.curriculum_stages)
            ]
            return

        unique_scores, counts = np.unique(self.difficulty_scores, return_counts=True)
        sorted_indices = np.argsort(unique_scores)
        unique_scores = unique_scores[sorted_indices]
        counts = counts[sorted_indices]
        cumulative_counts = np.cumsum(counts)
        total_examples = cumulative_counts[-1]

        if total_examples == 0:
            self.logger.warning('Total examples count is zero after difficulty scoring.')
            # Handle appropriately, maybe assign all to stage 1
            self.thresholds = [float('inf')] * self.curriculum_stages
            self.stage_distributions = [
                {'stage': i + 1, 'count': 0, 'threshold': None, 'percent': 0.0}
                for i in range(self.curriculum_stages)
            ]
            return

        target_per_stage = total_examples / self.curriculum_stages
        thresholds = []
        last_score_idx = -1  # Track the index of the last threshold score

        for stage in range(self.curriculum_stages - 1):
            target_count = (stage + 1) * target_per_stage
            # Find the index in unique_scores where cumulative count is closest to target
            # Ensure we select an index greater than the previous stage's threshold index
            possible_indices = np.where(np.arange(len(unique_scores)) > last_score_idx)[0]
            if len(possible_indices) == 0:
                # If no scores left, repeat last threshold or use infinity
                thresholds.append(thresholds[-1] if thresholds else unique_scores[-1])
                continue

            valid_cumulative_counts = cumulative_counts[possible_indices]
            # Find the index within possible_indices closest to the target count
            relative_idx = np.argmin(np.abs(valid_cumulative_counts - target_count))
            absolute_idx = possible_indices[relative_idx]

            thresholds.append(unique_scores[absolute_idx])
            last_score_idx = absolute_idx

        thresholds.append(float('inf'))  # Add final threshold

        self.thresholds = thresholds

        # Calculate and log stage distribution
        prev_thresh = -float('inf')  # Start below the lowest possible score
        self.stage_distributions = []
        for i, thresh in enumerate(self.thresholds):
            # Count examples within the threshold range for this stage
            count = np.sum(
                (self.difficulty_scores > prev_thresh) & (self.difficulty_scores <= thresh)
            )

            self.stage_distributions.append(
                {
                    'stage': i + 1,
                    'count': int(count),
                    'threshold': float(thresh) if thresh != float('inf') else 'inf',
                    'percent': float(count / total_examples * 100) if total_examples > 0 else 0.0,
                }
            )

            self.logger.info(
                f'Stage {i + 1}: {count} examples (difficulty range ({prev_thresh:.2f} < score <= {thresh:.2f}])'
            )

            if (
                count < self.min_examples_per_stage and total_examples > 0
            ):  # Check only if dataset not empty
                self.logger.warning(
                    f'Stage {i + 1} has only {count} examples, '
                    f'which is less than minimum {self.min_examples_per_stage}'
                )
            prev_thresh = thresh  # Update lower bound for next stage

    def _get_indices_for_stage(self, stage):
        """Get dataset indices for a specific curriculum stage."""
        if not hasattr(self, 'thresholds') or len(self.thresholds) <= stage:
            self.logger.error(
                f'Thresholds not computed or invalid stage ({stage}). Returning empty indices.'
            )
            return []
        assert 0 <= stage < self.curriculum_stages, f"Invalid stage: {stage}"

        lower_threshold = -float('inf') if stage == 0 else self.thresholds[stage - 1]
        # Ensure lower threshold is not 'inf' if previous stage threshold ended up being inf
        if lower_threshold == float('inf'):
            lower_threshold = -float('inf')

        upper_threshold = self.thresholds[stage]

        # Handle potential edge case where lower >= upper if thresholds didn't separate well
        if lower_threshold >= upper_threshold and upper_threshold != float('inf'):
            self.logger.warning(
                f'Thresholds for stage {stage + 1} are overlapping or inverted ({lower_threshold} >= {upper_threshold}). Selecting only scores equal to upper threshold if possible.'
            )
            # Select only scores equal to the upper threshold in this case
            indices = np.where(self.difficulty_scores == upper_threshold)[0]
        elif upper_threshold == float('inf'):
            # For the last stage, ensure we include scores greater than the previous threshold
            indices = np.where(self.difficulty_scores > lower_threshold)[0]
        else:
            # Standard case: scores between lower (exclusive) and upper (inclusive)
            indices = np.where(
                (self.difficulty_scores > lower_threshold)
                & (self.difficulty_scores <= upper_threshold)
            )[0]

        # Convert numpy int64 to python int
        indices = [int(idx) for idx in indices]
        return indices

    def set_stage(self, stage):
        """Set curriculum to a specific stage."""
        if not 0 <= stage < self.curriculum_stages:
            self.logger.error(
                f'Attempted to set invalid stage: {stage}. Keeping current stage {self.current_stage}.'
            )
            return len(self.current_indices)  # Return current length

        self.current_stage = stage
        self.current_indices = self._get_indices_for_stage(stage)

        self.logger.info(f"Switched to curriculum stage {stage + 1}/{self.curriculum_stages}")
        self.logger.info(f"Stage {stage + 1}: {len(self.current_indices)} examples")

        # Add safety check
        if len(self.current_indices) == 0 and len(self.difficulty_scores) > 0:
            self.logger.warning(
                f'Stage {stage + 1} is empty. Check difficulty score distribution and thresholds.'
            )

        return len(self.current_indices)

    def advance_stage(self):
        """Advance to the next curriculum stage if possible."""
        if self.current_stage < self.curriculum_stages - 1:
            return self.set_stage(self.current_stage + 1)
        else:
            self.logger.info("Already at final curriculum stage.")
            return len(self.current_indices)  # Return current length

    def __len__(self):
        """Return the number of examples in the current stage."""
        return len(self.current_indices)

    def __getitem__(self, idx):
        """Get item from the current stage's indices, accessing the main dataset."""
        if idx >= len(self.current_indices):
            raise IndexError(
                f'Index {idx} out of bounds for current stage size {len(self.current_indices)}'
            )

        try:
            # Map the local index to the correct dataset index
            dataset_idx = self.current_indices[idx]
            dataset_idx = int(dataset_idx)  # Ensure python int

            # Return the actual PROCESSED item using the main dataset's __getitem__
            return self.dataset[dataset_idx]
        except Exception as e:
            # Log specific index causing error if possible
            dataset_idx_str = (
                str(self.current_indices[idx]) if idx < len(self.current_indices) else 'unknown'
            )
            self.logger.error(
                f'Error retrieving item {idx} (maps to dataset index {dataset_idx_str}): {e}',
                exc_info=True,
            )

            # Attempt to return a dummy sample if the base dataset supports it
            if hasattr(self.dataset, '_create_dummy_sample') and callable(
                self.dataset._create_dummy_sample
            ):
                self.logger.warning('Returning dummy sample due to error.')
                return self.dataset._create_dummy_sample()
            else:
                # If no dummy sample method, raise the error to halt potentially bad batches
                raise RuntimeError(
                    f'Failed to get item {idx} and no dummy sample method available.'
                ) from e

    def get_all_data(self):
        """Return the underlying base dataset with all examples."""
        return self.dataset

    def get_current_stage_dataset(self):
        """Return a Subset dataset for the current stage (useful for DataLoader)."""
        # Check if indices are valid before creating Subset
        if not self.current_indices:
            self.logger.warning('Current stage has no indices. Returning empty Subset.')
            return Subset(self.dataset, [])
        return Subset(self.dataset, self.current_indices)

    def log_curriculum_stats(self, wandb_run=None):
        """Log curriculum statistics to wandb and console."""
        if len(self.difficulty_scores) == 0:
            self.logger.warning('Cannot log curriculum stats: No difficulty scores available.')
            return
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(
                self.difficulty_scores,
                bins=min(50, len(np.unique(self.difficulty_scores))),
                alpha=0.75,
                color='skyblue',
                edgecolor='black',
            )

            # Add threshold lines more robustly
            plotted_labels = set()
            for i, thresh in enumerate(self.thresholds[:-1]):  # Skip final 'inf'
                if thresh != float('inf'):
                    label = f'Stage {i + 1}/{i + 2} Threshold ({thresh:.1f})'
                    if label not in plotted_labels:  # Avoid duplicate legend entries
                        plt.axvline(x=thresh, color='r', linestyle='--', label=label)
                        plotted_labels.add(label)
                    else:
                        plt.axvline(x=thresh, color='r', linestyle='--')

            plt.xlabel('Difficulty Score')
            plt.ylabel('Number of Examples')
            plt.title(f'Curriculum Difficulty Distribution ({self.curriculum_strategy} strategy)')
            if plotted_labels:  # Only show legend if thresholds were plotted
                plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()  # Adjust layout

            if wandb_run and hasattr(wandb, 'Image') and hasattr(wandb, 'Table'):
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix='.png', delete=True
                    ) as tmp:  # Use delete=True
                        plt.savefig(tmp.name)
                        wandb_run.log({'curriculum/difficulty_distribution': wandb.Image(tmp.name)})

                    columns = ['Stage', 'Example Count', 'Max Difficulty', '% of Total']
                    data = [
                        [s['stage'], s['count'], str(s['threshold']), f'{s["percent"]:.1f}%']
                        for s in self.stage_distributions
                    ]
                    wandb_run.log(
                        {'curriculum/stage_statistics': wandb.Table(columns=columns, data=data)}
                    )
                except Exception as wandb_log_e:
                    self.logger.error(f'Error logging curriculum stats to WandB: {wandb_log_e}')
            else:
                # Save locally if not using wandb
                plt.savefig('curriculum_difficulty_distribution.png')
                self.logger.info(
                    'Saved curriculum difficulty distribution plot to curriculum_difficulty_distribution.png'
                )

            plt.close()  # Close the plot figure

        except Exception as e:
            self.logger.error(
                f'Error generating or logging curriculum stats plot: {e}', exc_info=True
            )


# --- Factory Function ---

def create_improved_curriculum_datasets(
    train_dataset,  # Should be ImprovedBartPhoDataset instance
    val_dataset,  # Can be any dataset instance
    curriculum_strategy='combined',
    curriculum_stages=3,
    min_examples_per_stage=100,
    logger=None,
    wandb_run=None,
):
    """
    Create enhanced curriculum learning wrapper dataset for training.
    Validation dataset is returned unchanged.

    Args:
        train_dataset: Training dataset (e.g., ImprovedBartPhoDataset) to wrap.
        val_dataset: Validation dataset (returned as is).
        curriculum_strategy: Strategy for determining difficulty.
        curriculum_stages: Number of curriculum stages.
        min_examples_per_stage: Minimum number of examples required per stage.
        logger: Optional logger instance.
        wandb_run: Optional wandb run instance for logging.

    Returns:
        train_curriculum: ImprovedCurriculumDataset for training.
        val_dataset: Original validation dataset.
    """
    if not logger:
        logger = logging.getLogger("CurriculumCreator")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)

    logger.info(
        f'Creating curriculum wrapper for training data with {curriculum_strategy} strategy and {curriculum_stages} stages'
    )
    try:
        train_curriculum = ImprovedCurriculumDataset(
            train_dataset,
            curriculum_strategy=curriculum_strategy,
            curriculum_stages=curriculum_stages,
            min_examples_per_stage=min_examples_per_stage,
            logger=logger,
        )

        # Log curriculum statistics if calculation was successful
        train_curriculum.log_curriculum_stats(wandb_run)

    except Exception as e:
        logger.error(f'Failed to create curriculum dataset: {e}', exc_info=True)
        logger.error('Curriculum learning disabled. Returning the original training dataset.')
        # Fallback: Return the original dataset if curriculum creation fails
        return train_dataset, val_dataset

    # Validation dataset is typically NOT wrapped in curriculum learning
    return train_curriculum, val_dataset