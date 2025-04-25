# Vietnamese-TrBart-Diacritic-Aware OCR

This project implements an Optical Character Recognition (OCR) model specifically designed for Vietnamese text. It leverages a hybrid architecture combining the vision capabilities of TrOCR with the Vietnamese language understanding of BartPho. A key focus is **diacritic awareness**, incorporating mechanisms to improve the recognition of Vietnamese diacritical marks, which are often challenging for standard OCR models. The project also includes features like curriculum learning and character-level processing to enhance training and accuracy.

## Key Features

*   **Hybrid Architecture:** Utilizes `microsoft/trocr-base-handwritten` as the vision encoder and `vinai/bartpho-syllable-base` as the text decoder.
*   **Diacritic-Aware Processing:** Includes specific components (character-level heads, feature extractors) aimed at improving the prediction of base characters and their associated diacritics.
*   **Character-Level Refinement:** Integrates character embeddings and processing alongside BartPho's token-level generation.
*   **Rethinking Module:** Incorporates a rethinking mechanism (inspired by VNHTR) to refine token predictions.
*   **Curriculum Learning:** Implements strategies (length, complexity, combined) to present training data from easy to hard, potentially improving convergence and final performance.
*   **Modular Code Structure:** Organized into distinct modules for data handling, model definition, training logic, and utilities.
*   **WandB Integration:** Supports logging metrics and visualizations to Weights & Biases for experiment tracking.

## Project Structure
```bash
./
├── data/                     # Datasets, preprocessing, curriculum logic
│   ├── __init__.py
│   ├── dataset.py            # ImprovedBartPhoDataset class
│   ├── curriculum.py         # ImprovedCurriculumDataset, create_improved_curriculum_datasets
│   └── collation.py          # custom_collate_fn
├── model/                    # Model definition and sub-modules
│   ├── __init__.py
│   ├── bartpho_ocr.py        # BartPhoVietOCR class definition
│   └── submodules.py         # AttentionModule, Rethinking, DiacriticFeatureExtractor, etc.
├── training/                 # Training loop, validation, checkpointing
│   ├── __init__.py
│   ├── trainer.py            # train_bartpho_model_with_curriculum function
│   ├── validation.py         # compute_validation_metrics function
│   └── checkpointing.py      # save_checkpoint
├── utils/                    # Utility functions (losses, optimizers, schedulers, etc.)
│   ├── __init__.py
│   ├── losses.py             # FocalLoss class
│   ├── optimizers.py         # create_optimizer function
│   ├── schedulers.py         # CosineWarmupScheduler class
│   ├── metrics.py            # compute_accuracy_metrics function
│   ├── logging_utils.py      # log_curriculum_learning_curves function
│   └── misc_utils.py         # calculate_class_weights, make_json_serializable
├── scripts/                  # Executable scripts
│   └── train.py              # Main execution logic, argument parsing
├── logs/                     # Directory for log files
│   └── bartpho_training.log
├── outputs/                  # Directory for saved models, checkpoints
│   └── vietnamese-ocr-bartpho-curriculum/
│       ├── checkpoints/
│       └── final_model_hf/
├── README.md                 # Project description, setup, usage
├── requirements.txt          # Python package dependencies
└── .gitignore                # Files/directories to ignore in git

```


## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vklinhhh/Vietnamese-TrBart-Diacritic-Aware.git
    cd Vietnamese-TrBart-Diacritic-Aware
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv hwt_venv
    source hwt_venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have PyTorch installed according to your CUDA version if using GPU. See [PyTorch installation guide](https://pytorch.org/get-started/locally/))*

## Usage

### Training

The main training script is `scripts/train.py`. Use the `-m` flag to run it as a module from the project root directory.

```bash
python -m scripts.train \
    --dataset_name vklinhhh/vietnamese_character_diacritic_cwl_v2 \
    --output_dir outputs/vietnamese-ocr-bartpho-curriculum \
    --vision_encoder microsoft/trocr-base-handwritten \
    --bartpho_model vinai/bartpho-syllable-base \
    --epochs 18 \
    --batch_size 8 \
    --learning_rate 5e-6 \
    --curriculum_strategy combined \
    --curriculum_stages 3 \
    --stage_epochs 4,6,8 \
    --discriminative_lr \
    --dynamic_loss_weighting \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --num_workers 4 \
    --grad_accumulation 2

```