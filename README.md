# multi-task-nlp

![PyPI version](https://img.shields.io/pypi/v/multi-task-nlp.svg)

Multi-task NLP Python package comprising data loading, training and evaluation

* PyPI package: https://pypi.org/project/multi-task-nlp/
* Free software: MIT License

## Credits

### Package template

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.

### Claude AI

Claude AI was used to generate an initial version of the code to train an intent classifier using PyTorch Lightning.
The generated code was then edited by Eraldo in order to both simplify some parts and optimize others.

### HuggingFace LLM Course

Parts of the code for token classification were copied from the [HF LLM Course](https://huggingface.co/learn/llm-course/chapter7/2).

## Features

* Load any dataset from HuggingFace for training, validation and testing
* Support multiple NLP tasks with shared encoder
* Train using PyTorch Lightning
    * Uni- or multi-task training
    * Arbitrary combination of tasks
    * *TODO* Train resumption
* Evaluation #TODO
    * Precision, Recall, F-score for multi-class and entity detection tasks (intent and slot filling, for instance)
* Tests #TODO
    * Docker
    * Justfile
    * Github Actions
