# `Task` Guide

The `Task` class is the foundation of all natural language tasks in the `lm-evaluation-harness` (harness). It encompasses everything you’d need to perform few-shot evaluation of an autoregressive language model. Here we’ll provide a step-by-step guide on how to subclass `Task` to create your very own task/s.

## Setup

If you haven't already, go ahead and fork the main repo, clone it, create a branch with the name of your task, and install the project requirements in your environment:

```sh
# After forking...
git clone https://github.com/<YOUR-USERNAME>/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout -b <task-name>
pip install -r requirements.txt
```

## Creating Your Task File

The first step in creating a task is to create a Python file in `lm_eval/tasks/`  with the task's name:

```sh
cd lm_eval/tasks
touch <task-name>.py
```

Then open the file and create a multiline docstring on the first line with the following contents:

```python
"""
<Paper title>
<Paper PDF URL>

<Short description of task>

Homepage: <URL to task's homepage>
"""
```

For example, take the QuAC dataset. We have:

```python
"""
QuAC: Question Answering in Context
https://arxiv.org/abs/1808.07036

Question Answering in Context (QuAC) is a dataset for modeling, understanding, and 
participating in information seeking dialog. Data instances consist of an interactive
dialog between two crowd workers: (1) a student who poses a sequence of freeform
questions to learn as much as possible about a hidden Wikipedia text, and (2)
a teacher who answers the questions by providing short excerpts (spans) from the text.

Homepage: https://quac.ai/
"""
```

Next, at the module-level, create a constant variable named
`_CITATION` that contains the citation information for your task in BibTeX format.

Now let's walk through the actual implementation - from data handling to evaluation.

## Data Handling

### Downloading your Data

There are 2 standard approaches we follow for downloading data:

1. Firstly, you should always check to see if your task's dataset is already provided by HuggingFace (__HF__); check their `datasets` catalog [here](https://huggingface.co/datasets). Is it in there? If yes, continue reading here, else go to 2. In the case that it’s there, things are a bit easier.  You can inherit from the `HFTask` class as so:

    ```python
    from . common import HFTask

    class TaskName(HFTask):
        DATASET_PATH = "..."
        DATASET_NAME = "..."
    ```
	where `DATASET_PATH` is the name of the benchmark/task dataset as listed by HF and `DATASET_NAME` is the name of, what HF calls, a “data instance” of the benchmark. If your task is not a benchmark containing any data instances just set `DATASET_NAME = None`.

2. Your task's dataset is not in HF's catalog, so you'll have to override a few abstract methods of the `Task` base class. First let's define our benchmark/task and inherit from `Task`.

    ```python
    from lm_eval.base import Task
    from pathlib import Path

    class TaskName(Task):
        DATASET_PATH = Path("data/<task-name>")
    ```
    where `DATASET_PATH` is the local directory we'll download into.
    Now we need to override the following methods:

    ```python
    def download(self):
    ```
    This should download the dataset into the relative path specified by `DATASET_PATH`. The preferred approach is to use EleutherAI's [best-download](https://github.com/EleutherAI/best-download) package which provides a `download_file` function that lets you validate complete data transmission through a checksum argument.  The overall logic should be something like: If the `DATASET_PATH` already exists then don’t download anything and exit the method, otherwise create the `DATASET_PATH` directory and actually download into it.  See this [task](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/logiqa.py#L9-L21) for an example.

   Next up, we have to set some “flags”:

    ```python
    def has_training_docs(self):
        return # True/False
    def has_validation_docs(self):
        return # True/False
    def has_test_docs(self):
        return # True/False
    ```
   These methods return `True`/`False` whether or not your task dataset provides documents for each split type. __Note__: if the test set doesn't have publicly available labels, please do not put it down as having a test set.

	Lastly, we need to load the documents. In our terminology, a document (`doc`) is a single natural language data example stored in a Python `dict`. E.g.: `{“question”: “What is the capital of France?”, “answer”: “Paris”}`. Override the following methods to load your data splits from their storage location in `DATASET_PATH`:
    ```python
    def training_docs(self):
        return #...
    def validation_docs(self):
        return #...
    def test_docs(self):
        return #...
    ```
	These should return a Python iterable (`list` or `generator`) of `dict`s that can be queried for individual `doc` examples. __NOTE__: If your task doesn't have a train/validation/test set, remember to raise a `NotImplementedError` for that specific split.

### Formatting your Few-Shot Examples

The harness is designed to facilitate task evaluations under the few-shot setting. Here we’ll format such examples.

<br>

⚠️  **Multiple-Choice Formatting**

If your task is **multiple-choice**, just inherit from the `MultipleChoiceTask` class we provide.

```python
from lm_eval.base import MultipleChoiceTask

class TaskName(..., MultipleChoiceTask):
```

This will require you to format your documents such that they contain `gold` and `choices` fields. They can also have other fields, but those will be ignored by `MultipleChoiceTask`. `choices` should be a list of possible continuations, and `gold` should be an integer specifying the index of the correct completion.

See [this task](https://github.com/EleutherAI/lm-evaluation-harness/blob/105fa9741ff660f6a62c2eef0d2facfde36dda41/lm_eval/tasks/sat.py#L56) for an example. When used in combination with `HFTask`, it may be useful to override [`_convert_standard`](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/common.py#L28), which will be applied to every document in the HF dataset. See [this task](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/headqa.py) for an example of this.

You can now skip ahead to <a href="#Registering-Your-Task">registering your task</a>.

⚠️  **End Multiple-Choice Formatting**

<br>

In the case your task is _not_ multiple-choice, override the following methods for your task class:

Format your document into a single query prompt __without the answer__ here. This method takes a single `doc` example of type `dict` with `str` key-value members. You should concatenate these `doc` item values together into a neatly formatted prompt.

```python
def doc_to_text(self, doc):
    return ""
```

Put the target answer of the prompt here, in the form: `" " + <answer>`.

```python
def doc_to_target(self, doc):
    return ""
```

Understand that the strings from `doc_to_text` and `doc_to_target` will be concatenated together to build up labeled examples in the k-shot setting where k > 0. Design with that in mind 👍.

### Registering Your Task

Now's a good time to register your task to expose it for usage. All you'll need to do is import your task module in `lm_eval/tasks/__init__.py` and provide an entry in the `TASK_REGISTRY`  dictionary with the key as the name of your benchmark task (in the form it'll be referred to in the command line) and the value as the task class. See how it's done for other tasks in the [file](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/__init__.py).

### Checking the Data

After registering your task, you can now check on your data downloading and verify that the few-shot samples look as intended. Run the following command with your desired args:

```bash
python -m scripts.write_out \
    --output_base_path <path> \
    --tasks <your-task> \
    --sets <train | val | test> \
    --num_fewshot K \
    --num_examples N \ 
    --description_dict_path <path>
```

Open the file specified at the `--output_base_path <path>` and ensure it passes
a simple eye test.

## Evaluation

**🛑**  If your task is a single-true multiple-choice task and you've correctly inherited from `MultipleChoiceTask` then your job here is done; <a href="#Checking-the-Task-Performance">go ‘head and check on the task performance!</a> 🛑

Now comes evaluation. The methods you'll need to implement are:

```python
def construct_requests(self, doc, ctx):
    """ Uses RequestFactory to construct Requests and returns an iterable of
    Requests which will be sent to the LM.

    :param doc:
        The document as returned from training_docs, validation_docs, or test_docs.
    :param ctx: str
        The context string, generated by fewshot_context. This includes the natural
        language description, as well as the few shot examples, and the question
        part of the document for `doc`.
    """
    return ...
```
If your task requires generating text you'll need to return a `rf.greedy_until` request otherwise an `rf.loglikelihood` across all labels in a classification tasks will do.

```python
def process_results(self, doc, results):
    """Take a single document and the LM results and evaluates, returning a
    dict where keys are the names of submetrics and values are the values of
    the metric for that one document

    :param doc:
        The document as returned from training_docs, validation_docs, or test_docs.
    :param results:
        The results of the requests created in construct_requests.
    """
    return {}
```

```python
def aggregation(self):
    """
    :returns: {str: [float] -> float}
        A dictionary where keys are the names of submetrics and values are
        functions that aggregate a list of metrics
    """
    return {}
```

See `lm_eval/metrics.py` for a few "built-in" aggregate metrics you can easily import.

```python
def higher_is_better(self):
    """
    :returns: {str: bool}
        A dictionary where keys are the names of submetrics and values are
        whether a higher value of the submetric is better
    """
    return {}
```

Some tasks that are good examples of various ways evaluation can be implemented can be found here: [LAMBADA](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/lambada.py), [TriviaQA](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/triviaqa.py), [SQuAD](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/squad.py).

Tip: Feel free to create your own helper-methods for your task!

### Checking the Task Performance

```sh
python main.py \
	--model gpt2 \
	--model_args device=<device-name> \
	--tasks <task-name> \
	--num_fewshot K
```

Set the limit size, `N`, to a smallish number (e.g. 10) and try out the task under different `K`-shot settings. If you have an Nvidia GPU at your disposal, add the argument
`--model_args device=cuda:0`. If you have access to an OpenAI API key, you can also evaluate GPT-3 on various tasks with the following command:

```sh
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
	--model gpt3 \
	--tasks <task-name> \
	--num_fewshot K
```

### Running Unit Tests

To run the entire test suite, use:

```sh
pytest
```

This is usually overkill; to run only the tests for your task, do:
```sh
pytest -k <task name>
```

## Versioning

Lastly, we need to "version control". Tasks in the harness can always evolve. Metrics get updated, data sources change, etc. It’s important to mark each task with a version attribute so users can document which implementation version was used to obtain their results. Add a `VERSION` attribute to your task right below the class name and set it to `0` (this is the first version/implementation of your task):

```python
class TaskName(...):
	VERSION = 0
```

## Submitting your Task

Although we currently do not work behind a specific style guide, we'd appreciate if you tidy up your file/s with the `black` formatter (which should've been install through the `requirements.txt`). Keep things clean…ish 🙂.

Now push your work and make a pull request! Thanks for the contribution 👍. If there are any questions, leave a message in the `#lm-thunderdome` channel on the EAI discord.
