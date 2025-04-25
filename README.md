# Vietnamese-TrBart-Diacritic-Aware

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
├── scripts/                  # Executable scripts (like main training script)
│   └── train.py              # Main execution logic, argument parsing
├── logs/                     # Directory for log files (add to .gitignore)
│   └── bartpho_training.log
├── outputs/                  # Directory for saved models, checkpoints (add to .gitignore)
│   └── vietnamese-ocr-bartpho-curriculum/
│       ├── checkpoints/
│       └── final_model_hf/
├── README.md                 # Project description, setup, usage
├── requirements.txt          # Python package dependencies
└── .gitignore                # Files/directories to ignore in git