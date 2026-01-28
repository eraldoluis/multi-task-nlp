"""PyTorch Lightning module for multi-task NLP."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel

from multi_task_nlp.data_module import NlpDataModule


class TextClassificationHead(nn.Module):
    """Head for text classification tasks."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, encoder_outputs):
        # Use the hidden state of the first token
        # TODO (eraldoluis): consider using the last token instead
        logits = self.classifier(encoder_outputs[:, 0, :])
        return logits


class TokenClassifierHead(nn.Module):
    """Head for token classification tasks."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, encoder_outputs):
        # We flatten the batch and sequence dimensions
        logits = self.classifier(encoder_outputs).flatten(0, 1)
        return logits


class NlpModule(pl.LightningModule):
    """PyTorch Lightning module for multi-task NLP.

    The given data module defines the tasks to be performed. Currently supports text and token classification tasks but
    it is easy to add more.
    """

    def __init__(self, data_module: NlpDataModule, learning_rate: float = 2e-5, freeze_encoder: bool = False):
        super().__init__()
        self.save_hyperparameters({"model_name": data_module.model_name, "ignore": "data_module"})

        self.data_module = data_module

        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(data_module.model_name)

        hidden_size = self.encoder.config.hidden_size
        self.heads = {
            # Heads for each text classification task
            **{
                task: TextClassificationHead(hidden_size, data_module.num_classes[task])
                for task in data_module.text_class_tasks
            },
            # Heads for token classification task
            **{
                task: TokenClassifierHead(hidden_size, data_module.num_classes[task])
                for task in data_module.token_class_tasks
            },
        }

        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        # TODO (eraldoluis): if we freeze the whole encoder, the model can't learn because it is using the
        # hidden representation of the first token ([CLS]) as input for the classifier. This representation is not
        # meaningful for the classification task. One alternative is to unfree the last layer of BERT and freeze all
        # the rest. Another option is to add an attention pooling layer on top of BERT so that the hidden vectors of
        # all tokens are used to create a more meaningful representation for the classification task.

        # freeze all encoder params but the ones in the last layer
        idx_last_layer = self.encoder.config.n_layers - 1
        for name, param in self.encoder.named_parameters():
            if not name.startswith(f"transformer.layer.{idx_last_layer}."):
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_outputs = outputs.last_hidden_state

        # return logits for each task in a dictionary keyed by task name
        logits = {task: head(encoder_outputs) for task, head in self.heads.items()}
        return logits

    def _pred(self, batch: dict, step: str) -> dict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # get logits for each task
        logits = self(input_ids, attention_mask)

        # compute losses and metrics for the text classification tasks
        losses = []
        for task in self.data_module.text_class_tasks:
            label_id = batch[f"{task}_label_id"]
            loss = self.loss_fn(logits[task], label_id)
            losses.append(loss)

            preds = torch.argmax(logits[task], dim=1)
            num_correct = (preds == label_id).float().sum()
            acc = num_correct / label_id.size(0)

            self.log(f"{step}/{task}/loss", loss, prog_bar=True)
            self.log(f"{step}/{task}/acc", acc, prog_bar=True)

        # compute losses and metrics for the token classification tasks
        for task in self.data_module.token_class_tasks:
            label_ids = batch[f"{task}_label_ids"].flatten()
            loss = self.loss_fn(logits[task], label_ids)
            losses.append(loss)

            preds = torch.argmax(logits[task], dim=1)
            valid_labels = label_ids != self.loss_fn.ignore_index
            num_correct = (valid_labels & (preds == label_ids)).float().sum()
            acc = num_correct / valid_labels.float().sum()

            self.log(f"{step}/{task}/loss", loss, prog_bar=True)
            self.log(f"{step}/{task}/acc", acc, prog_bar=True)

        # TODO (eraldoluis): add more metrics (precision, recall, F1, entity-level)

        # aggregate losses
        loss = torch.stack(losses).mean()
        self.log(f"{step}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._pred(batch, step="train")

    def validation_step(self, batch, batch_idx):
        return self._pred(batch, step="val")

    def test_step(self, batch, batch_idx):
        return self._pred(batch, step="test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
