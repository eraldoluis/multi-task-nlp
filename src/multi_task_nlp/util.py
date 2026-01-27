"""Utility functions for multi-task NLP."""


class LabelEncoder:
    """Simple label encoder for converting string labels to integer IDs and vice versa."""

    def __init__(self, labels: list[str]):
        if not labels:
            raise ValueError("Classes list cannot be empty.")

        if len(set(labels)) != len(labels):
            raise ValueError("Classes must be unique.")

        self.labels = labels
        self.label_to_index = {cls: idx for idx, cls in enumerate(labels)}

    def transform(self, labels: str | list[str]) -> int | list[int]:
        """Convert string labels to integer IDs.

        Args:
            labels: A single label or a list of labels.

        Returns:
            The corresponding integer ID(s).
        """
        if isinstance(labels, str):
            return self.label_to_index[labels]
        return [self.label_to_index[label] for label in labels]

    def inverse_transform(self, indices: int | list[int]) -> str | list[str]:
        """Convert integer IDs back to string labels.

        Args:
            indices: A single integer ID or a list of integer IDs.

        Returns:
            The corresponding string label(s).
        """
        if isinstance(indices, int):
            return self.labels[indices]
        return [self.labels[idx] for idx in indices]
