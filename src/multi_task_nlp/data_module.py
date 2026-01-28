"""Pytorch Lightning DataModule for multi-task NLP datasets using HuggingFace datasets and transformers."""

from abc import abstractmethod
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from multi_task_nlp.util import LabelEncoder


class NlpDataModule(pl.LightningDataModule):
    """Data module for loading and preparing a HuggingFace dataset for different NLP tasks.

    This is an abstract class and should be extended for specific datasets. It handles data pre-processing,
    tokenization, label encoding, and data loading for training/validation/testing.
    """

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        labels: dict[str, list[str]],
        batch_size: int = 32,
        max_length: int = 128,
        val_split: float = 0.15,
        num_proc: int = 10,
    ):
        """Initializes the data module.

        Args:
            dataset_name: Name of the HuggingFace dataset to use.
            model_name: Name of the pre-trained model to use for tokenization.
            labels: Dictionary mapping task names to lists of string labels for each task.
            batch_size: Batch size for training.
            max_length: Maximum length of the input sequences.
            val_split: Fraction of the training split to use for validation.
            num_proc: Number of processes to use for data loading and processing.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.val_split = val_split
        self.num_proc = num_proc

        if labels.keys() != set(self.all_tasks):
            raise ValueError(f"Labels ({labels.keys()}) and task names ({set(self.all_tasks)}) must match.")

        self.label_encoder = {task_name: LabelEncoder(labels) for task_name, labels in labels.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Collator used to pad inputs to the same length in a batch. Labels are padded in the collate_fn method
        # depending on the task type.
        self.input_data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    @property
    def num_classes(self) -> dict[str, int]:
        """Number of classes for each task.

        Returns:
            A dictionary mapping task names to number of classes.
        """
        return {task_name: len(encoder.labels) for task_name, encoder in self.label_encoder.items()}

    @property
    def token_class_tasks(self) -> list[str]:
        """List of task names that are token classification tasks."""
        return []

    @property
    def text_class_tasks(self) -> list[str]:
        """List of task names that are text classification tasks."""
        return []

    @property
    def all_tasks(self) -> list[str]:
        """List of all task names."""
        return self.token_class_tasks + self.text_class_tasks

    @abstractmethod
    def preprocess_example(self, example: dict) -> dict:
        """Pre-process a single example from the dataset for the model with multi-task labels.

        It must return a dictionary with at least the list of pre-tokenized words under the key "words". Additionally,
        it must return training labels for each task following the structure below:
        ```
        {
            "words": list[str],  # list of pre-tokenized words
            "<task1>_labels": list[str],  # list of labels for each word (token classification task)
            "<task2>_label": str,  # single label for the whole input (text classification task)
            ...
        }
        ```

        In the example above, `<task1>` and `<task2>` are placeholders for the actual task names.

        Args:
            example: A single example from the dataset.

        Returns:
            A dict with the preprocessed example.
        """

    def encode_labels(self, example: dict) -> dict:
        """Encodes the string labels for each task to integer IDs in the given pre-processed example.

        For text classification tasks, the encoded label ID is stored under the key `<task>_label_id`.
        For token classification tasks, the list of encoded label IDs is stored under the key `<task>_label_ids`.

        Args:
            example: A single pre-processed example.

        Returns:
            The example with encoded labels.
        """
        for task in self.text_class_tasks:
            example[f"{task}_label_id"] = self.label_encoder[task].transform(example[f"{task}_label"])
        for task in self.token_class_tasks:
            example[f"{task}_label_ids"] = self.label_encoder[task].transform(example[f"{task}_labels"])
        return example

    def tokenize_batch_and_align_labels(self, examples: dict) -> dict:
        """Tokenizes the input words for the examples in the batch and aligns word-level labels with the tokens.

        The alignment algorithm is adapted from https://huggingface.co/learn/llm-course/chapter7/2

        Args:
            examples: A batch of examples from the dataset.

        Returns:
            The tokenized input and aligned labels.
        """
        tokenized_inputs = self.tokenizer(
            examples["words"], truncation=True, is_split_into_words=True, max_length=self.max_length
        )

        for task in self.token_class_tasks:
            all_labels = examples[f"{task}_labels"]
            all_label_ids = examples[f"{task}_label_ids"]
            new_labels = []
            new_label_ids = []
            for i, (labels, label_ids) in enumerate(zip(all_labels, all_label_ids, strict=True)):
                word_idxs = tokenized_inputs.word_ids(i)
                aligned_labels, assigned_label_ids = self._align_labels_with_tokens(
                    labels=labels, label_ids=label_ids, word_idxs=word_idxs, label_encoder=self.label_encoder[task]
                )
                new_labels.append(aligned_labels)
                new_label_ids.append(assigned_label_ids)

            tokenized_inputs[f"{task}_labels"] = new_labels
            tokenized_inputs[f"{task}_label_ids"] = new_label_ids

        tokenized_inputs["tokens"] = [tokenized_inputs.tokens(i) for i in range(len(examples["words"]))]
        return tokenized_inputs

    def _align_labels_with_tokens(
        self, labels: list[str], label_ids: list[int], word_idxs: list[int], label_encoder: LabelEncoder = None
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
                    label_id = label_encoder.transform(label)
                new_labels.append(label)
                new_label_ids.append(label_id)

        return new_labels, new_label_ids

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate function to be used in the DataLoader for batching examples.

        Input tokens are padded using the self.input_data_collator. Labels are padded depending on the task type:
        - For text classification tasks, labels are stacked into a tensor of shape (batch_size,).
        - For token classification tasks, labels are padded to the maximum sequence length in the batch using
          -100 as the padding value.

        Args:
            batch: A list of examples.

        Returns:
            A batch dictionary with padded inputs and labels.
        """
        # remove labels from batch for input collation
        batch_no_label = [
            {k: v for k, v in example.items() if not k.endswith("_label_ids") and not k.endswith("_label_id")}
            for example in batch
        ]
        batch_collated = self.input_data_collator(batch_no_label)

        # add text labels
        for task in self.text_class_tasks:
            batch_collated[f"{task}_label_id"] = torch.tensor([example[f"{task}_label_id"] for example in batch])

        # pad token-level labels
        for task in self.token_class_tasks:
            batch_collated[f"{task}_label_ids"] = nn.utils.rnn.pad_sequence(
                [torch.tensor(example[f"{task}_label_ids"]) for example in batch],
                batch_first=True,
                padding_value=-100,
            )

        return batch_collated

    def prepare_data(self):
        # Download dataset from the Hub
        load_dataset(self.dataset_name)

    def setup(self, stage=None):
        # Load dataset split
        dataset = load_dataset(self.dataset_name, split="test" if stage == "test" else "train", num_proc=self.num_proc)

        # Prepare each input
        dataset = dataset.map(self.preprocess_example, num_proc=self.num_proc, desc="Preprocessing examples")

        # Encode labels
        dataset = dataset.map(self.encode_labels, num_proc=self.num_proc, desc="Encoding labels")

        # Tokenize input and align labels with tokens
        dataset = dataset.map(
            self.tokenize_batch_and_align_labels,
            batched=True,
            num_proc=self.num_proc,
            desc="Tokenizing inputs",
        )

        # keep only necessary columns
        dataset = dataset.remove_columns(
            set(dataset.column_names)
            - {"input_ids", "attention_mask"}
            - {f"{task}_label_ids" for task in self.token_class_tasks}
            - {f"{task}_label_id" for task in self.text_class_tasks}
        )

        print("Number of classes:")
        lst_num_labels = "\n\t".join(
            [f"{task_name}: {len(encoder.labels)}" for task_name, encoder in self.label_encoder.items()]
        )
        print(lst_num_labels)
        print()

        if stage == "test":
            self.test_dataset = dataset
            print(f"Test size: {len(self.test_dataset)}")
        else:
            # Split train into train and validation
            train_val_datasets = dataset.train_test_split(test_size=self.val_split)
            self.train_dataset = train_val_datasets["train"]
            self.val_dataset = train_val_datasets["test"]

            print(f"Train size: {len(self.train_dataset)}")
            print(f"Validation size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4)
