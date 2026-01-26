"""PyTorch Lightning code for training a slot filling model on the ATIS dataset."""

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

import wandb


class SlotFillingDataModule(pl.LightningDataModule):
    """Data module for loading and preparing the ATIS dataset with slot filling labels.

    The ATIS dataset contains four columns:
    1. id: numeric identifier for each example
    2. text: the utterance text
    3. intent: the intent label for the utterance (it may contain multiple intents separated by '+')
    4. slots: the slot filling labels for each token in the utterance based on the BIO scheme

    The slot filling labels use the following scheme:
    - B/I-<slot_name>.<entity_type>: we will ignore the entity type for simplicity

    Example:
    ```
    {
        'id': 719,
        'intent': 'flight+airfare',
        'text': 'first flights and fares from pittsburgh to atlanta on a thursday',
        'slots': 'B-flight_mod O O O O B-fromloc.city_name O B-toloc.city_name O O B-depart_date.day_name'
    }
    ```
    """

    KEEP_COLUMNS = ["input_ids", "attention_mask", "label_ids"]
    """Columns to keep for training and evaluation (for the DataLoaders)."""

    def __init__(
        self, model_name: str, batch_size: int = 32, max_length: int = 128, val_split: float = 0.15, num_proc: int = 10
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.val_split = val_split
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.num_proc = num_proc
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _prepare_input(self, example: dict) -> dict:
        """Prepares a single example for the model.

        Args:
            example: A single example from the dataset.
        """
        return {
            # take only the last intent
            "intent": example["intent"].split("+")[-1],
            # pre-tokenized words
            "words": example["text"].split(),
            # token-level BIO labels without entity types
            "slots": [s.split(".")[0] for s in example["slots"].split()],
        }

    def _tokenize_and_align_labels(self, examples: dict) -> dict:
        """Tokenizes the input and aligns the labels with the tokens.

        Source: adapted from https://huggingface.co/learn/llm-course/chapter7/2

        Args:
            examples: A batch of examples from the dataset.

        Returns:
            The tokenized input and aligned labels.
        """
        tokenized_inputs = self.tokenizer(
            examples["words"], truncation=True, is_split_into_words=True, max_length=self.max_length
        )

        all_labels = examples["slots"]
        all_label_ids = examples["label_ids"]
        new_labels = []
        new_label_ids = []
        for i, (labels, label_ids) in enumerate(zip(all_labels, all_label_ids, strict=True)):
            word_idxs = tokenized_inputs.word_ids(i)
            aligned_labels, assigned_label_ids = self._align_labels_with_tokens(labels, label_ids, word_idxs)
            new_labels.append(aligned_labels)
            new_label_ids.append(assigned_label_ids)
        tokenized_inputs["slots"] = new_labels
        tokenized_inputs["label_ids"] = new_label_ids
        tokenized_inputs["tokens"] = [tokenized_inputs.tokens(i) for i in range(len(all_labels))]
        return tokenized_inputs

    def _align_labels_with_tokens(
        self, labels: list[str], label_ids: list[int], word_idxs: list[int]
    ) -> tuple[list[str], list[int]]:
        """Aligns the wordwise (pre-tokenized) BIO-labels with the tokenized input.

        Source: adapted from https://huggingface.co/learn/llm-course/chapter7/2

        Args:
            labels: The word-level labels.
            label_ids: The word-level label IDs.
            word_idxs: The word IDs for each token.

        Returns:
            The token-level labels.
        """
        new_labels = []
        new_label_ids = []
        current_word = None
        for word_id in word_idxs:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = None if word_id is None else labels[word_id]
                label_id = -100 if word_id is None else label_ids[word_id]
                new_labels.append(label)
                new_label_ids.append(label_id)
            elif word_id is None:
                # Special token
                new_labels.append(None)
                new_label_ids.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                label_id = label_ids[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label.startswith("B-"):
                    label = "I-" + label[2:]
                    label_id = int(self.label_encoder.transform([label])[0])
                new_labels.append(label)
                new_label_ids.append(label_id)

        return new_labels, new_label_ids

    def prepare_data(self):
        # Download dataset
        load_dataset("tuetschek/atis")

    def setup(self, stage=None):
        # Load dataset
        dataset = load_dataset("tuetschek/atis")

        # Prepare each input
        dataset = dataset.map(self._prepare_input, num_proc=self.num_proc)

        # All labels without B or I
        all_labels = sorted(
            {label[2:] for split in dataset.values() for labels in split["slots"] for label in labels if label != "O"}
        )
        # Add O, B- and I- labels
        all_labels = ["O"] + [f"B-{label}" for label in all_labels] + [f"I-{label}" for label in all_labels]
        # Fit the encoder
        self.label_encoder.fit(all_labels)
        # Number of classes
        self.num_classes = len(self.label_encoder.classes_)

        # Encode labels
        dataset = dataset.map(
            lambda ex: {"label_ids": self.label_encoder.transform(ex["slots"]).tolist()}, num_proc=self.num_proc
        )

        # Tokenize input and align labels with tokens
        dataset = dataset.map(
            self._tokenize_and_align_labels,
            batched=True,
            num_proc=self.num_proc,
        )

        dataset = dataset.remove_columns(set(dataset["train"].column_names) - set(self.KEEP_COLUMNS))

        # Split train into train and validation
        train_data = dataset["train"]
        train_size = int((1 - self.val_split) * len(train_data))
        indices = np.random.permutation(len(train_data))
        # train_indices = indices[:int(train_size*0.1)]
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create train dataset
        self.train_dataset = train_data.select(train_indices)
        self.val_dataset = train_data.select(val_indices)
        self.test_dataset = dataset["test"]

        print(f"Number of classes: {self.num_classes}")
        print(f"Train size: {len(self.train_dataset)}")
        print(f"Validation size: {len(self.val_dataset)}")
        print(f"Test size: {len(self.test_dataset)}")

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # remove label_ids from batch to avoid warning from DataCollatorWithPadding
        batch_no_label = [{k: v for k, v in example.items() if k != "label_ids"} for example in batch]
        batch_collated = self.data_collator(batch_no_label)
        # pad label sequences and add them to the batches
        batch_collated["label_ids"] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(example["label_ids"]) for example in batch],
            batch_first=True,
            padding_value=-100,
        )
        return batch_collated

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self._collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self._collate_fn, num_workers=4)


class TokenClassifier(nn.Module):
    """Token classification model using BERT-like encoder."""

    def __init__(self, model_name: str, num_classes: int, freeze_bert: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            # FIXME (eraldoluis): if we freeze the whole BERT encoder, the model can't learn because it is using the
            # hidden representation of the first token ([CLS]) as input for the classifier. This representation is not
            # meaningful for the classification task. One alternative is to unfree the last layer of BERT and freeze all
            # the rest. Another option is to add an attention pooling layer on top of BERT so that the hidden vectors of
            # all tokens are used to create a more meaningful representation for the classification task.

            # freeze all BERT params but the ones in the last layer
            idx_last_layer = self.bert.config.n_layers - 1
            for name, param in self.bert.named_parameters():
                if not name.startswith(f"transformer.layer.{idx_last_layer}."):
                    param.requires_grad = False

        # self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the hidden state of each token
        cls_output = outputs.last_hidden_state
        logits = self.classifier(cls_output)
        return logits


class TokenClassifierLightning(pl.LightningModule):
    """PyTorch Lightning module for token classification."""

    def __init__(self, model_name: str, num_classes: int, learning_rate: float = 2e-5, freeze_bert: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.model = TokenClassifier(model_name, num_classes, freeze_bert=freeze_bert)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def _pred(self, batch: dict, step: str) -> dict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        # flatten label ids
        label_ids = batch["label_ids"].flatten()

        logits = self(input_ids, attention_mask).flatten(0, 1)
        loss = self.loss_fn(logits, label_ids)

        preds = torch.argmax(logits, dim=1)
        valid_labels = label_ids != self.loss_fn.ignore_index
        num_correct = (valid_labels & (preds == label_ids)).float().sum()
        acc = num_correct / valid_labels.float().sum()

        self.log(f"{step}/loss", loss, prog_bar=True)
        self.log(f"{step}/acc", acc, prog_bar=True)

        return {"loss": loss, "acc": acc}

    def training_step(self, batch, batch_idx):
        return self._pred(batch, step="train")

    def validation_step(self, batch, batch_idx):
        return self._pred(batch, step="val")

    def test_step(self, batch, batch_idx):
        return self._pred(batch, step="test")

    def configure_optimizers(self):
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.5, patience=2, verbose=True
        # )
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def train_token_classifier():
    """Main training function."""

    # TODO convert this to a function so that we can easily create a cyclopts CLI.

    # TODO add yaml config support so that we can have recipes for different experiments.

    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Login to W&B
    wandb.login()

    # Parameters
    MODEL_NAME = "distilbert/distilbert-base-uncased"
    BATCH_SIZE = 32
    MAX_EPOCHS = 3
    MAX_STEPS = -1  # 30
    LEARNING_RATE = 2e-5

    # Initialize data module
    data_module = SlotFillingDataModule(model_name=MODEL_NAME, batch_size=BATCH_SIZE, max_length=128, val_split=0.15)
    data_module.setup()

    # Initialize model
    model = TokenClassifierLightning(
        model_name=MODEL_NAME, num_classes=data_module.num_classes, learning_rate=LEARNING_RATE, freeze_bert=True
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoints",
        filename="voize-slot-filling-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stop_callback = EarlyStopping(monitor="val/loss", patience=3, mode="min", verbose=True)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        max_steps=MAX_STEPS,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        # devices=1,
        log_every_n_steps=1,
        logger=WandbLogger(project="voyze-slot-filling"),
        val_check_interval=10,
    )

    # Train model
    print("Starting training...")
    trainer.fit(model, data_module)

    # Test model
    print("\nTesting model on test set...")
    trainer.test(model, data_module)

    print("\nTraining complete!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train_token_classifier()
