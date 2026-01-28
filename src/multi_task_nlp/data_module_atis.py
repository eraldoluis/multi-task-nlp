"""Data module for intent classification and slot filling tasks using the ATIS dataset."""

from typing import Literal

from multi_task_nlp.data_module import NlpDataModule


class AtisIntentClassificationAndSlotFillingDataModule(NlpDataModule):
    """Data module for intent classification and slot filling tasks with the ATIS dataset.

    The ATIS dataset contains four columns:
    1. id: numeric identifier for each example
    2. text: the utterance text
    3. intent: the intent label for the utterance (it may contain multiple intents separated by '+')
    4. slots: the slot filling labels for each token in the utterance based on the BIO scheme

    The slot filling labels use the scheme "B/I-<slot_name>.<entity_type>". We will ignore the entity type for
    simplicity.

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

    INTENTS = [
        "flight",
        "flight_time",
        "airfare",
        "aircraft",
        "ground_service",
        "airport",
        "airline",
        "distance",
        "abbreviation",
        "ground_fare",
        "quantity",
        "city",
        "flight_no",
        "capacity",
        "meal",
        "restriction",
        "cheapest",
        "day_name",
    ]
    """List of possible intents in the ATIS dataset."""

    SLOT_ENTITIES = [
        "fromloc",
        "depart_time",
        "toloc",
        "arrive_time",
        "depart_date",
        "flight_time",
        "cost_relative",
        "round_trip",
        "fare_amount",
        "city_name",
        "stoploc",
        "class_type",
        "airline_name",
        "mod",
        "fare_basis_code",
        "transport_type",
        "flight_mod",
        "arrive_date",
        "meal",
        "meal_description",
        "return_date",
        "airline_code",
        "flight_stop",
        "time",
        "or",
        "economy",
        "flight_number",
        "flight_days",
        "state_code",
        "airport_code",
        "aircraft_code",
        "connect",
        "restriction_code",
        "airport_name",
        "days_code",
        "day_name",
        "period_of_day",
        "today_relative",
        "meal_code",
        "state_name",
        "time_relative",
        "return_time",
        "month_name",
        "day_number",
        "compartment",
        "booking_class",
        "flight",
    ]
    """List of possible slot entities in the ATIS dataset."""

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        task: Literal["intent", "slots", "multi"] = "multi",
        batch_size: int = 32,
        max_length: int = 128,
        val_split: float = 0.15,
        num_proc: int = 10,
    ):
        """Initialize the ATIS data module supporting for multi-task learning (intent classification and slot filling).

        Args:
            dataset_name: Name of the dataset to load from HuggingFace datasets following the ATIS format.
            model_name: Name of the pre-trained model to use for tokenization.
            task: The type of task to run. Either "intent" for intent classification, "slots" for slot filling,
                or "multi" for multi-task learning.
            batch_size: Batch size for training and evaluation.
            max_length: Maximum sequence length for tokenization.
            val_split: Proportion of the training set to use as validation set.
            num_proc: Number of processes to use for data preprocessing.
        """

        if task not in ["intent", "slots", "multi"]:
            raise ValueError(f"Unknown task: {task}. Must be one of 'intent', 'slots', or 'multi'.")

        super().__init__(
            dataset_name=dataset_name,
            model_name=model_name,
            labels={
                "intent": self.INTENTS,
                "slots": ["O"] + [f"{pref}-{entity}" for entity in self.SLOT_ENTITIES for pref in ["B", "I"]],
            },
            batch_size=batch_size,
            max_length=max_length,
            val_split=val_split,
            num_proc=num_proc,
        )

        self.task = task
        self._text_class_tasks = ["intent"] if task in ["intent", "multi"] else []
        self._token_class_tasks = ["slots"] if task in ["slots", "multi"] else []

    @property
    def text_class_tasks(self) -> list[str]:
        """List of task names that are text classification tasks."""
        return self._text_class_tasks

    @property
    def token_class_tasks(self) -> list[str]:
        """List of task names that are token classification tasks."""
        return self._token_class_tasks

    def preprocess_example(self, example: dict) -> dict:
        proc_example = {
            # pre-tokenized words
            "words": example["text"].split(),
        }

        if self.text_class_tasks:
            # take only the last intent
            proc_example["intent_label"] = example["intent"].split("+")[-1]

        if self.token_class_tasks:
            # token-level BIO labels without entity types
            proc_example["slots_labels"] = [s.split(".")[0] for s in example["slots"].split()]

        return proc_example
