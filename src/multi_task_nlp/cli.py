"""Console script for multi_task_nlp."""

from typing import Annotated, Literal

from cyclopts import App, Parameter

from multi_task_nlp.config import Config
from multi_task_nlp.train_atis import train_atis_multi_task

app = App()


@app.default()
def main(*, config: Annotated[Config, Parameter(name="*")], task: Literal["text", "token", "multi"] = "multi"):
    """Console script for multi_task_nlp for the ATIS dataset supporting intent classification and slot filling.

    Args:
        config: The configuration for the task.
        task: The type of task to run. Either "text" for text classification, "token" for token classification,
            or "multi" for multi-task learning. Default is "multi".
    """
    train_atis_multi_task(config, task=task)


if __name__ == "__main__":
    app()
