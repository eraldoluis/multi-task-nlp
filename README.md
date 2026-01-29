# multi-task-nlp

Multi-task NLP Python package comprising data loading, training and evaluation.
It is based on PyTorch Lightning and HuggingFace libs.
It currently supports the ATIS dataset and two tasks:
- intent classification, and
- slot filling (via token classification using BIO tags).

It supports training independent models for each individual task as well as a multi-task model that shares the same
encoder for both tasks. In the multi-task mode, the encoder is trained simultaneously using examples from both tasks.
When training a multi-task model, each example must include labels for both tasks.

## Installation

For now, the only supported installation is from source:
```
python3 -m venv venv
source venv/bin/activate
git clone git@github.com:eraldoluis/multi-task-nlp.git
pip install -e ./multi-task-nlp/
```

## CLI

After installation, you have access to the command `multi-task-nlp`.
You can train a multi-task model for the ATIS dataset (intent classification and slot filling) using the following command:
```
multi_task_nlp \
    --project-name atis-multi-task \
    --encoder.model-name distilbert/distilbert-base-uncased \
    --data-processing.dataset-name tuetschek/atis \
    --seed 42
```

You can see all the available options by running:
```
multi_task_nlp --help
```

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

### GitHub Copilot

I've used GitHub Copilot all the time while developing this project, mainly for inline suggestion.

### HuggingFace LLM Course

Parts of the code for token classification were copied from the [HF LLM Course](https://huggingface.co/learn/llm-course/chapter7/2).

## Features

* Load any dataset from HuggingFace for training, validation and testing (following ATIS format)
* Support multi-task NLP tasks with shared encoder
* Support seamlessly adding new dataset formats
* Easy to add new tasks (beyond text and token classification)
* Support any token-based encoder from HF (freezing may not work for some)
* Train using PyTorch Lightning
    * Mono- or multi-task training
    * Arbitrary combination of tasks
* Evaluation
    * Loss and token-level accuracy for each task and combined
    * Testing after training

### Upcoming features and known issues

* Examples with incomplete task labels (currently all examples must be labeled with all tasks)
* Train resumption from a checkpoint
* Tests, linting and deployment
    * Docker and Justfile for homogeneous tasks both locally and remotely
    * CI/CD
* Evaluation
    * Precision, Recall, F-score for multi-class tasks
    * Same for entity recognition/classification tasks
    * Submit table to W&B with final test metrics
* Datasets caching (`map` method)
    * It seems the caching mechanism of the `Dataset.map()` method is not working properly. This is probably due to the
      use of object methods (which include the `self` argument), making them not serializable. I need to investigate
      further on how to change that while still supporting the features of the framework.

## More general ideas related to multi-task NLP

* Knowledge distillation
    * Implement KD on the multi-task scenario so that a multi-task teacher can be used to distill a multi-task student
    * Zero-shot LLM teacher: LLMs present a strong performance on several NLP tasks even on a zero-shot scenario.
        This idea involves using a LLM with a zero-shot prompt for specific tasks in order to distill a student model using
        the LLM logits for the task(s) at hand.
* ASR features: incorporate outputs of the ASR encoder to improve dowstream tasks accuracy
* ASR/NLP joint training: going deeper and integrating ASR and NLP training
* Quantization
