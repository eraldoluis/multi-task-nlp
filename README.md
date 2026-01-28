# multi-task-nlp

![PyPI version](https://img.shields.io/pypi/v/multi-task-nlp.svg)

Multi-task NLP Python package comprising data loading, training and evaluation

* PyPI package: https://pypi.org/project/multi-task-nlp/
* Free software: MIT License

## Credits

### Package template

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.

### Claude AI

Claude AI was used to generate an initial version of the code to train an intent classifier using PyTorch Lightning
because I've never used this framework before. The generated code was then greatly edited in order to simplify some
parts, optimize other parts, and generalize the code. The following prompt was used to generate this initial script:
> Create a code to train an intent classifier using PyTorch Lightning. The code has to include validation and test
> procedures. It should use the HuggingFace dataset "tuetschek/atis" which comprises two splits: train and test. The
> train split should be divided into train and validation. This dataset includes three columns: "intent", "text" and
> "slots". You must ignore the "slots" column and use the "intent" column as the label and the "text" column as the
> input for the model. The model should be based on the "distilbert/distilbert-base-uncased" model from HuggingFace. A
> classification layer should be added on top of this model to predict the intent. The input for the classification
> layer should be the hidden representation of the first token of the input.

### HuggingFace LLM Course

Parts of the code for token classification were copied from the [HF LLM Course](https://huggingface.co/learn/llm-course/chapter7/2).

## Features

* Load any dataset from HuggingFace for training, validation and testing
* Support multiple NLP tasks with shared encoder
* Train using PyTorch Lightning
    * Uni- or multi-task training
    * Arbitrary combination of tasks
    * *TODO* Train resumption from a checkpoint
* Evaluation #TODO
    * Precision, Recall, F-score for multi-class and entity detection tasks (intent and slot filling, for instance)
* *TODO* Tests
    * Docker
    * Justfile
    * Github Actions
