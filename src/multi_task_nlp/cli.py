"""Console script for multi_task_nlp."""

from typing import Annotated, Literal

from cyclopts import App, Parameter

from multi_task_nlp.config import Config

app = App()


@app.default()
def main(*, task: Literal["text", "token", "multi"], config: Annotated[Config, Parameter(name="*")]):
    """Console script for multi_task_nlp.

    Args:
        task: The type of task to run. Either "text" for text classification, "token" for token classification,
            or "multi" for multi-task learning.
        config: The configuration for the task.

    Raises:
        ValueError: If the task is not recognized.
    """
    if task == "text":
        from multi_task_nlp.intent_classification import train_text_classifier

        train_text_classifier()
    elif task == "token":
        from multi_task_nlp.token_classification import train_token_classifier

        train_token_classifier(config)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    app()
