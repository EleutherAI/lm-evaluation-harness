# `Task` Guide

The `Task` class is the foundation of all natural language tasks in the `lm-evaluation-harness` (harness). It encompasses everything you’d need to perform few-shot evaluation of an autoregressive language model. Here we’ll provide a step-by-step guide on how to subclass `Task` to create your very own task/s.

## Setup

If you haven't already, go ahead and fork the main repo, clone it, create a branch with the name of your task, and install the project requirements in your environment:

```sh
# After forking...
git clone https://github.com/<YOUR-USERNAME>/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout -b <task-name>
pip install -e ".[dev]"
```

## Creating Your Task File

From the `lm-evaluation-harness` project root, copy over the `new_task.py` template to `lm_eval/datasets`.

```sh
cp templates/new_task.py lm_eval/tasks/<task-name>.py
```

or if your task is **multiple-choice**, the `new_multiple_choice_task.py`:

```sh
cp templates/new_multiple_choice_task.py lm_eval/tasks/<task-name>.py
```

This will set you up with a few `TODO`s to fill-in which we'll now go over in detail.

## Task Heading

Open the file you've just created and add a multiline docstring on the first line with the following contents:

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

All data downloading and management is handled through the HuggingFace (**HF**) [`datasets`](https://github.com/huggingface/datasets) API. So, the first thing you should do is check to see if your task's dataset is already provided in their catalog [here](https://huggingface.co/datasets). If it's not in there, please consider adding it to their Hub to make it accessible to a wider user base by following their [new dataset guide](https://github.com/huggingface/datasets/blob/master/ADD_NEW_DATASET.md)
.
Now, that you have your HF dataset, you need to assign its path and name to your `Task` in the following fields:

```python
class TaskName(...):
    DATASET_PATH = "..."
    DATASET_NAME = "..."
```

where `DATASET_PATH` is the name of the dataset as listed by HF in the `datasets` Hub and `DATASET_NAME` is the name of, what HF calls, a “data instance” or sub-task of the benchmark. If your task does not contain any data instances, just set `DATASET_NAME = None`.
(If you're familiar with the HF `datasets.load_dataset` function, these are just the first 2 arguments to it.)

Next up, we have to set some “flags”:

```python
    def has_training_docs(self):
        return # True/False

    def has_validation_docs(self):
        return # True/False

    def has_test_docs(self):
        return # True/False
```

These methods return `True`/`False` whether or not your task dataset provides documents for each split type. __Note__: if the test set does not have publicly available answer labels, please do not put it down as having a test set - return False.

Lastly, we need to load the documents. In our terminology, a document (`doc`) is a single natural language data example stored in a Python `dict`. E.g.: `{“question”: “What is the capital of France?”, “answer”: “Paris”}`. Override the following methods to load your data splits from their storage location in `DATASET_PATH`:

```python
    def training_docs(self):
        return #...

    def validation_docs(self):
        return #...

    def test_docs(self):
        return #...
```

These should return a Python iterable (`list` or `generator`) of `dict`s that can be queried for individual `doc` examples.

#### Processing Documents

At this point, you can also process each individual document to, for example, strip whitespace or "detokenize" its fields. Put the processing logic into `_process_doc` and map the functions across training/validation/test docs inside of the respective functions.
🔠 If your task is **multiple-choice**, we require you to format your documents such that they contain `gold` and `choices` fields. They can also have other fields, but those will be ignored by `MultipleChoiceTask`. `choices` should be a list of possible continuations, and `gold` should be an integer specifying the index of the correct completion.
See [this task](https://github.com/EleutherAI/lm-evaluation-harness/blob/6caa0afd96a7a7efb2ec4c1f24ad1756e48f3aa7/lm_eval/tasks/sat.py#L60) for an example. 🔠

### Formatting your Few-Shot Examples

The harness is designed to facilitate task evaluations under the few-shot setting. Here we’ll format such examples.

Format your document into a single query prompt __without the answer__ here. This method takes a single `doc` example of type `dict` with `str` key-value members. You should concatenate these `doc` item values together into a neatly formatted prompt.

```python
def doc_to_text(self, doc):
    return ""
```

<br>

️🔠 **Multiple-Choice Formatting**

If your task is multiple-choice, you can now skip ahead to <a href="#Registering-Your-Task">registering your task</a>.

️️🔠 **End Multiple-Choice Formatting**

<br>

Format the target answer from the contents of `doc`. Note that the prepended `" "` is required to space out the `doc_to_text` and `doc_to_target` strings.

```python
def doc_to_target(self, doc):
    target = ""
    return " " + target
```

Finally, be aware that the strings from `doc_to_text` and `doc_to_target` will be concatenated together to build up labeled examples in the k-shot setting where k > 0. Design with that in mind 👍.

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
```

Open the file specified at the `--output_base_path <path>` and ensure it passes
a simple eye test.

## Evaluation

**🛑**  If your task is a single-true multiple-choice task and you've correctly inherited from `MultipleChoiceTask` then your job here is done; <a href="#Checking-the-Task-Performance">go ‘head and check on the task performance!</a> 🛑

Now comes evaluation. The methods you'll need to implement are:

```python
def construct_requests(self, doc, ctx):
    """Uses RequestFactory to construct Requests and returns an iterable of
    Requests which will be sent to the LM.

    Args:
        doc (dict):
            The document as returned from training_docs, validation_docs, or
            test_docs.
        ctx (str):
            The context string, generated by fewshot_context. This includes
            the natural language description, as well as the few shot examples,
            and the question part of the document for `doc`.
        args (dict):
            The specifics of the context, including number of few shots.

    Returns:
        An iterable of `Request` objects.
    """
    return ...
```
If your task requires generating text you'll need to return a `rf.greedy_until` request otherwise an `rf.loglikelihood` across all labels in a classification tasks will do.

```python
def process_results(self, doc, results):
    """Take a single document and the LM results and evaluates, returning a
    dict where keys are the names of sub-metrics and values are the values of
    the metric for that one document.

    NOTE: This function automates processing by using the `promptsource`
    metadata to determine the metric.

    Args:
        doc (dict):
            The document as returned from training_docs, validation_docs, or
            test_docs.
        results (list):
            The results of the requests created in construct_requests.

    Returns:
        A dict of metric results.
    """
    return {}
```

```python
def aggregation(self):
    """
    Returns:
        A dictionary where keys are the names of sub-metrics and values are
        functions that aggregate a list of metric scores.
        {str: [metric_score] -> float}
    """
    return {}
```

See `lm_eval/metrics.py` for a few "built-in" aggregate metrics you can easily import.

```python
def higher_is_better(self):
    """
    Returns:
        A dictionary where keys are the names of sub-metrics and values are
        whether a higher value of the sub-metric is better.
        {str: bool}
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
