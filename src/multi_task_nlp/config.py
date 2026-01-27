"""Pydantic models for configuration of the multi_task_nlp CLI and Python API."""

from pydantic import BaseModel


class EncoderConfig(BaseModel):
    """Configuration for the encoder."""

    model_name: str
    """Name of the pre-trained model to use."""

    freeze_encoder: bool = True
    """Whether to freeze the encoder during training for this task."""


class DataProcessingConfig(BaseModel):
    """Configuration for the processing of the train, validation, and test datasets."""

    dataset_name: str
    """Name of the HuggingFace dataset to use for this task."""

    batch_size: int = 32
    """Batch size for training."""

    max_length: int = 128
    """Maximum length of the input sequences."""

    val_split: float = 0.15
    """Fraction of the training data to use for validation."""

    num_proc: int = 10
    """Number of processes to use for data loading and processing."""


class TrainerConfig(BaseModel):
    """Configuration for training."""

    max_epochs: int = 5
    """Maximum number of training epochs."""

    max_steps: int = -1
    """Maximum number of training steps. -1 means no limit."""

    learning_rate: float = 2e-5
    """Learning rate for the optimizer."""


class Config(BaseModel):
    """Configuration for the multi_task_nlp CLI and Python API."""

    project_name: str
    """Name of the project to use for logging and checkpoint saving."""

    encoder: EncoderConfig
    """Configuration for the encoder."""

    data_processing: DataProcessingConfig
    """Configuration for dataset processing for training, validation, and testing."""

    trainer: TrainerConfig = TrainerConfig()
    """Configuration for training."""

    seed: int = None
    """Random seed for reproducibility."""

    wandb: bool = False
    """Whether to use Weights & Biases for experiment tracking."""
