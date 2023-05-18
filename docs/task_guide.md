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

### Decontamination
For background on decontamination please see [this](./decontamination.md).

If you wish to support decontamination studies for your task simply override the "should_decontaminate" method and return true.

You also need to override "doc_to_decontamination_query" and return the data you wish to compare against the training set. This doesn't necessarily need to be the full document or request, and we leave this up to the implementor. For a multi-choice evaluation you could for example just return the question.

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
#### What's a `Request`? What's a `doc`?
To reiterate, a `doc` is just a `Dict` object that contains information about a document from your corpus. It can contain things like a prompt, question type information, answers and anything else you think will be needed in order to assess your model for a given task. Keep in mind that the fields of this can be basically whatever you want (you can sort this out in `training_docs` \ `validation_docs` \ `test_docs` if you need to customise things - see above), just remember to be consistent with them throughout the rest of the `Task` you write up.
A `Request` is an object that takes the text prompt you want to present to a model and computes one of a few different types of response. These are evaluated lazily (meaning, only when the result is actually needed). If your task requires generating text you'll need to return a `rf.greedy_until` request otherwise an `rf.loglikelihood` across all labels in a classification tasks will do.
The function `construct_requests` can return a list of `Request`s or an iterable; it's perfectly fine to `yield` them from something or other. This is particularly handy if you are creating more than one request per `doc` (usually because you're up to something like multi-task learning). The objects this function returns then get consumed one by one and turned into result objects.


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
This is the next step in the chain after `construct_requests`. In between this function and the one above, the request is evaluated. The results of that request are returned in the `results` arg to this function. By processing results, what is meant is calculating the metric or metrics of interest for your dataset using the result and associated ground truth given to this function. It's possible to calculate and return multiple metrics in this function and the logic for it can be whatever you want - as long as you've made sure the ground truth was included in the `doc` object. The dict returned from this function should be of the format `{'metric_name': value}`. It is not necessary to have the same keys for every doc processed using `process_results`; this sort of thing can be handled in the next function, `aggregation`.


```python
def aggregation(self):
    """
    :returns: {str: [float] -> float}
        A dictionary where keys are the names of submetrics and values are
        functions that aggregate a list of metrics
    """
    return {}
```
In `process_results`, model outputs are converted into metrics. These metrics are per document metrics, however; the `aggregation` function is used to work out what to do with them to create a corpus-level metric. Imagine you have a bunch of documents, for each of which you have calculated an F1 score. What should that mean overall? Should they be summed, averaged, the min/max found? This function handles that problem.

The contents of the function itself are pretty straightforward; it should simply return a dict that maps from each metric label that could be returned by `process_results` to a function that can be used to aggregate that metric. That is to say, if the metrics that `process_results` could return are given by `{'a', 'b', 'c'}`, then all of these keys should be present in the dict returned by `aggregation`.
__NOTE__: See `lm_eval/metrics.py` for a few "built-in" aggregate metrics you can easily import. The standard metrics available in this package are generally based on `sklearn` functions, so if you are in any doubt for how to set things up the documentation over there can be of assistance. If you need to write a custom metric for some reason, start by looking at the existing ones in `lm_eval/metrics.py` for an idea about what the function signature needs to be.

```python
def higher_is_better(self):
    """
    :returns: {str: bool}
        A dictionary where keys are the names of submetrics and values are
        whether a higher value of the submetric is better
    """
    return {}
```
Finally, this function returns a dict with the same keys as `aggregation` and as it says in the description, simply tells us whether higher scores are better.

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

You can format your changes and perform flake8 standard checks by running the following commands:

```sh
pre-commit install
pre-commit run --all-files
```

Now push your work and make a pull request! Thanks for the contribution 👍. If there are any questions, leave a message in the `#lm-thunderdome` channel on the EAI discord.
