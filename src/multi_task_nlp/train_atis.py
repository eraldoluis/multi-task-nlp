"""Training script for multi-task NLP using PyTorch Lightning."""

from typing import Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from multi_task_nlp.config import Config
from multi_task_nlp.data_module_atis import AtisIntentClassificationAndSlotFillingDataModule
from multi_task_nlp.lightning_module import NlpModule


def train_atis_multi_task(config: Config, task: Literal["intent", "slots", "multi"] = "multi"):
    """Training function for multi-task NLP using the ATIS dataset.

    Args:
        config: Configuration for training.
        task: The type of task to run. Either "intent" for intent classification, "slots" for slot filling,
            or "multi" for multi-task learning.
    """

    if config.seed is not None:
        pl.seed_everything(config.seed)

    if config.wandb:
        wandb.login()

    # Initialize the data module
    data_module = AtisIntentClassificationAndSlotFillingDataModule(
        dataset_name=config.data_processing.dataset_name,
        model_name=config.encoder.model_name,
        task=task,
        batch_size=config.data_processing.batch_size,
        max_length=config.data_processing.max_length,
        val_split=config.data_processing.val_split,
        num_proc=config.data_processing.num_proc,
    )

    # Initialize model
    model = NlpModule(
        data_module=data_module,
        learning_rate=config.trainer.learning_rate,
        freeze_encoder=config.encoder.freeze_encoder,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoints",
        filename=f"{config.project_name}-{task}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        mode="min",
    )

    early_stop_callback = EarlyStopping(monitor="val/loss", patience=3, mode="min", verbose=True)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.trainer.max_epochs,
        max_steps=config.trainer.max_steps,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=WandbLogger(project=config.project_name) if config.wandb else None,
        val_check_interval=config.trainer.val_check_interval,
    )

    # Train model
    print("Starting training...")
    trainer.fit(model, data_module)

    # Test model
    print("\nTesting model on test set...")
    trainer.test(model, data_module)

    print("\nTraining complete!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


# if __name__ == "__main__":
#     # Example config
#     config = Config(
#         project_name="atis-intent-slots",
#         encoder=EncoderConfig(model_name="distilbert/distilbert-base-uncased", freeze_encoder=True),
#         data_processing=DataProcessingConfig(dataset_name="tuetschek/atis"),
#         seed=42,
#         wandb=True,
#     )
#     train_multi_task_nlp(config)
