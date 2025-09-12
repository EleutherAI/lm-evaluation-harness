# New Task Guide

`lm-evaluation-harness` is a framework that strives to support a wide range of zero- and few-shot evaluation tasks on autoregressive language models (LMs).

This documentation page provides a walkthrough to get started creating your own task, in `lm-eval` versions v0.4.0 and later.

A more interactive tutorial is available as a Jupyter notebook [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb).

## Setup

If you haven't already, go ahead and fork the main repo, clone it, create a branch with the name of your task, and install the project requirements in your environment:

```sh
# After forking...
git clone https://github.com/<YOUR-USERNAME>/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout -b <task-name>
pip install -e ".[dev]"
```

In this document, we'll walk through the basics of implementing a static benchmark evaluation in two formats: a *generative* task which requires sampling text from a model, such as [`gsm8k`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml), and a *discriminative*, or *multiple choice*, task where the model picks the most likely of several fixed answer choices, such as [`sciq`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/sciq/sciq.yaml).

## Creating a YAML file

To implement a new standard task, we'll need to write a YAML file which configures our task logic. We start by making a new empty YAML file. This file can have any name, but we recommend placing it in a subfolder of `lm_eval/tasks` titled by the dataset or task's shorthand name: for example,

```sh
touch lm_eval/tasks/<dataset_name>/<my_new_task_name>.yaml
```

Or, copy the template subfolder we provide from `templates/new_yaml_task`:

```sh
cp -r templates/new_yaml_task lm_eval/tasks/
```

and rename the folders and YAML file(s) as desired.

### Selecting and configuring a dataset

All data downloading and management is handled through the HuggingFace (**HF**) [`datasets`](https://github.com/huggingface/datasets) API. So, the first thing you should do is check to see if your task's dataset is already provided in their catalog [here](https://huggingface.co/datasets). If it's not in there, please consider adding it to their Hub to make it accessible to a wider user base by following their [new dataset guide](https://github.com/huggingface/datasets/blob/main/ADD_NEW_DATASET.md)
.
> [!TIP]
> To test your task, we recommend using verbose logging using `export LOGLEVEL = DEBUG` in your shell before running the evaluation script. This will help you debug any issues that may arise.
Once you have a HuggingFace dataset prepared for your task, we want to assign our new YAML to use this dataset:

```yaml
dataset_path: ... # the name of the dataset on the HF Hub.
dataset_name: ... # the dataset configuration to use. Leave `null` if your dataset does not require a config to be passed. See https://huggingface.co/docs/datasets/load_hub#configurations for more info.
dataset_kwargs: null # any extra keyword arguments that should be passed to the dataset constructor, e.g. `data_dir`.
```

Next, we'd like to tell our task what the dataset's train, validation, and test splits are named, if they exist:

```yaml
training_split: <split name of training set, or `null`>
validation_split: <split name of val. set, or `null`>
test_split: <split name of test set, or `null`>
```

Tests will run on the `test_split` if it is available, and otherwise evaluate on the `validation_split`.

We can also specify from which split the task should retrieve few-shot examples via:

```yaml
fewshot_split: <split name to draw fewshot examples from, or `null`>
```

or by hardcoding them, either using the following in the yaml file:

```yaml
fewshot_config:
  sampler: first_n
  samples: [
    {<sample 1>},
    {<sample 2>},
  ]
```

or by adding the function `list_fewshot_samples` in the associated utils.py file:

```python
def list_fewshot_samples() -> list[dict]:
  return [{<sample 1>}, {<sample 2>}]
```

See `lm_eval/tasks/minerva_math/minerva_math_algebra.yaml` for an example of the latter, and `lm_eval/tasks/gsm8k/gsm8k-cot.yaml` for an example of the former.

In this case, each sample must contain the same fields as the samples in the above sets--for example, if `doc_to_text` expects an `input` field when rendering input prompts, these provided samples must include an `input` key.

If neither above options are not set, we will default to train/validation/test sets, in that order.

Finally, our dataset may not be already in the exact format we want. Maybe we have to strip whitespace and special characters via a regex from our dataset's "question" field! Or maybe we just want to rename its columns to match a convention we'll be using for our prompts.

Let's create a python file in the directory where we're writing our YAML file:

```bash
touch lm_eval/tasks/<dataset_name>/utils.py
```

Now, in `utils.py` we'll write a function to process each split of our dataset (the following example is drawn from [the `hellaswag` task](../lm_eval/tasks/hellaswag/utils.py)):

```python
def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)
```

Now, in our YAML config file we'll use the `!function` constructor, and tell the config where our imported Python function will come from. At runtime, before doing anything else we will preprocess our dataset according to this function!

```yaml
process_docs: !function utils.process_docs
```

### Using Local Datasets

To load a local dataset for evaluation, you can specify data files in the `dataset_kwargs` field, such as the following for JSON files:

```yaml
dataset_path: json
dataset_name: null
dataset_kwargs:
  data_files: /path/to/my/json
```

Or with files already split into separate directories:

```yaml
dataset_path: arrow
dataset_kwargs:
  data_files:
    train: /path/to/arrow/train/data-00000-of-00001.arrow
    validation: /path/to/arrow/validation/data-00000-of-00001.arrow
```

Alternatively, if you have previously downloaded a dataset from huggingface hub (using `save_to_disk()`) and wish to use the local files, you will need to use `data_dir` under `dataset_kwargs` to point to where the directory is.

```yaml
dataset_path: hellaswag
dataset_kwargs:
  data_dir: hellaswag_local/
```

You can also set `dataset_path` as a directory path in your local system. This will assume that there is a loading script with the same name as the directory. [See datasets docs](https://huggingface.co/docs/datasets/loading#local-loading-script).

## Writing a Prompt Template

The next thing we need to do is decide what format to use when presenting the data to the LM. This is our **prompt**, where we'll define both an input and output format.

To write a prompt, users will use `doc_to_text`, `doc_to_target`, and `doc_to_choice` (Optional when certain conditions are met).

`doc_to_text` defines the input string a model will be given while `doc_to_target` and `doc_to_choice` will be used to generate the target text. `doc_to_target` can be either a text string that refers to the target string or an integer that refers to the index of the correct label. When it is set as an index, `doc_to_choice` must also be set with the appropriate list of possible choice strings.

### Basic prompts

If a dataset is straightforward enough, users can enter the feature name directly. This assumes that no preprocessing is required. For example in [Swag](https://github.com/EleutherAI/lm-evaluation-harness/blob/1710b42d52d0f327cb0eb3cb1bfbbeca992836ca/lm_eval/tasks/swag/swag.yaml#L10-L11), `doc_to_text` and `doc_to_target` given the name of one of the feature each.

```yaml
doc_to_text: startphrase
doc_to_target: label
```

Hard-coding is also possible as is the case in [SciQ](https://github.com/EleutherAI/lm-evaluation-harness/blob/1710b42d52d0f327cb0eb3cb1bfbbeca992836ca/lm_eval/tasks/sciq/sciq.yaml#L11).

```yaml
doc_to_target: 3
```

`doc_to_choice` can be directly given a list of text as option (See [Toxigen](https://github.com/EleutherAI/lm-evaluation-harness/blob/1710b42d52d0f327cb0eb3cb1bfbbeca992836ca/lm_eval/tasks/toxigen/toxigen.yaml#L11))

```yaml
doc_to_choice: ['No', 'Yes']
```

if a dataset feature is already a list, you can set the name of the feature as `doc_to_choice` (See [Hellaswag](https://github.com/EleutherAI/lm-evaluation-harness/blob/e0eda4d3ffa10e5f65e0976161cd134bec61983a/lm_eval/tasks/hellaswag/hellaswag.yaml#L13))

```yaml
doc_to_choice: choices
```

### Writing a prompt with Jinja 2

We support the [Jinja 2](https://jinja.palletsprojects.com/en/3.1.x/) templating language for writing prompts. In practice, this means you can take your dataset's columns and do many basic string manipulations to place each document into prompted format.

Take for example the dataset `super_glue/boolq`. As input, we'd like to use the features `passage` and `question` and string them together so that for a sample line `doc`, the model sees something in the format of:

```text
doc["passage"]
Question: doc["question"]?
Answer:
```

We do this by [writing](https://github.com/EleutherAI/lm-evaluation-harness/blob/1710b42d52d0f327cb0eb3cb1bfbbeca992836ca/lm_eval/tasks/super_glue/boolq/default.yaml#L9C1-L9C61)

```yaml
doc_to_text: "{{passage}}\nQuestion: {{question}}?\nAnswer:"
```

Such that `{{passage}}` will be replaced by `doc["passage"]` and `{{question}}` with `doc["question"]` when rendering the prompt template.

Our intended output is for the model to predict a single whitespace, and then the answer to the question. We do this via:

```yaml
doc_to_target: "{{answer}}"
```

#### Multiple choice format

For tasks which are multiple choice (a fixed, finite set of label words per each document) and evaluated via comparing loglikelihoods of all label words (the `multiple_choice` task output type) we enforce a particular convention on prompt format.

> [!WARNING]
> We add `target_delimiter` between input and target which defaults to " ", such that the full input-output string is `doc_to_text(doc) + target_delimiter + doc_to_target(doc)`. `doc_to_text` and `doc_to_target` should not contain trailing right or left whitespace, respectively. For multiple choice the target will be each choice index concatenated with the delimiter.

An annotated example in the case of SciQ is as follows:

```yaml
doc_to_text: "{{support.lstrip()}}\nQuestion: {{question}}\nAnswer:" # This is the input portion of the prompt for this doc. It will have " {{choice}}" appended to it as target for each choice in answer_choices.
doc_to_target: 3 # this contains the index into the answer choice list of the correct answer.
doc_to_choice: "{{[distractor1, distractor2, distractor3, correct_answer]}}"
```

Task implementers are thus able to decide what the answer choices should be for a document, and what prompt format to use.

The label index can also be sourced from a feature directly. For example in `superglue/boolq`, the label index if defined in the feature `label`. We can set `doc_to_target` as simply `label`. The options or verbalizers can be written in the form of a list `["no", "yes"]` that will correspond to the label index.

```yaml
doc_to_text: "{{passage}}\nQuestion: {{question}}?\nAnswer:"
doc_to_target: label
doc_to_choice: ["no", "yes"]
```

### Using Python Functions for Prompts

There may be cases where the prompt we want to implement is easier expressed in Python instead of Jinja 2. For this, we can use Python helper functions that are defined in the YAML config. It should be noted that the function script must be in the same directory as the yaml.

A good example is WikiText that requires a lot of regex rules to clean the samples.

```python
def wikitext_detokenizer(doc):
    string = doc["page"]
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    ...
    string = string.replace(" 's", "'s")

    return string
```

We can load this function in `doc_to_target` by using a `!function` operator after `doc_to_target` and followed by `<file name>.<function name>`. In the file [wikitext.yaml](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/wikitext.yaml) we write:

```yaml
doc_to_target: !function preprocess_wikitext.wikitext_detokenizer
```

### Importing a Prompt from Promptsource

[Promptsource](https://github.com/bigscience-workshop/promptsource/tree/main/promptsource) is a great repository for crowdsourced prompts for many datasets. We can load these prompts easily by using the `use_prompt` argument and filling it with the format `"promptsource:<name of prompt template>"`. To use this, `doc_to_text` and `doc_to_target` should be left undefined. This will fetch the template of the dataset defined in the YAML file.

For example, For Super Glue BoolQ, if we want to use the prompt template `GPT-3 Style` we can add this to the YAML file.

```yaml
use_prompt: "promptsource:GPT-3 Style"
```

If you would like to run evaluation on all prompt templates, you can simply call it this way.

```yaml
use_prompt: "promptsource:*"
```

### Setting metrics

You're almost done! Now we need to choose how to score our task.

- *If this is a multiple choice task:* do you just want to check your model's accuracy in choosing the correct answer choice?
- *If this is a generation task:* do you just want to check how often your model outputs *exactly the ground-truth output string provided*?

If the answer to the above is no: you'll need to record what scoring metrics to use! Metrics can be listed in the following format:

```yaml
metric_list:
  - metric: <name of the metric here>
    aggregation: <name of the aggregation fn here>
    higher_is_better: <true or false>
  - metric: !function script.function
    aggregation: ...
    higher_is_better: ...
```

`aggregation` and `higher_is_better` can optionally be left out to default to the manually-set defaults if using a natively supported metric, otherwise it must be defined explicitly (for example, when using a custom metric implemented as a function).

For a full list of natively supported metrics and aggregation functions see [`docs/task_guide.md`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md). All metrics supported in [HuggingFace Evaluate](https://github.com/huggingface/evaluate/tree/main/metrics) can also be used, and will be loaded if a given metric name is not one natively supported in `lm-eval` or `hf_evaluate` is set to `true`.

### Optional, More Advanced Setup

Some tasks may require more advanced processing logic than is described in this guide.

As a heuristic check:

- Does your task require generating multiple free-form outputs per input document?
- Does your task require complex, multi-step post-processing of generated model outputs?
- Does your task require subsetting documents on the fly based on their content?
- Do you expect to compute metrics after applying multiple such processing steps on your model outputs?
- Does your task rely on metrics that need a custom implementation?

For more detail on the task system and advanced features, see [`docs/task_guide.md`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md). If none of the above sounds like they apply to your task, it's time to continue onto checking your task performance!

### Task name + tags (registering a task)

To test a task conveniently, it helps to *register* the task--that is, to give it a name and make the `lm-eval` library aware it exists!

If you're writing your YAML file inside the `lm_eval/tasks` folder, you just need to give your task a name! You can do this inside your YAML file:

```yaml
task: <name of the task>
```

Including a task name is mandatory.

It is often also convenient to label your task with several `tag` values, though this field is optional:

```yaml
tag:
  - tag1
  - tag2
```

This will add your task to the `tag1` and `tag2` tags, enabling people to know how to categorize your task, and if desired run all tasks in one of these groups at once, your task along with them.

If your task is not in the `lm_eval/tasks` folder, you'll need to tell the Eval Harness where to look for YAML files.

You can do this via the `--include_path` argument in `__main__.py`. This command will be used to initialize the `TaskManager` object which you can also use for your custom scripts.

```python
task_manager = TaskManager(args.verbosity, include_path=args.include_path)
```

Passing `--tasks /path/to/yaml/file` is also accepted.

### Advanced Group Configs

While `tag` values are helpful when you want to be able to quickly and conveniently run a set of related tasks via `--tasks my_tag_name`, often, we wish to implement more complex logic. For example, the MMLU benchmark contains 57 *subtasks* that must all be *averaged* together in order to report a final 'MMLU score'.

Groupings of tasks might also use particular variants of a task--for example, we might want to default to evaluating a task as 5-shot when called as part of a given grouping, but not have a preference for number of shots when evaluating it as a standalone.

We implement this via **groups**, which are distinct from tags. Groups can be implemented via *group config* YAML files, which are laid out similarly but slightly differently to tasks' YAML configs.

The most basic form of group can be defined via a YAML config similar to the following:

```yaml
group: nli_tasks
task:
  - cb
  - anli_r1
  - rte
metadata:
  version: 1.0
```

This will behave almost identically to a `tag` that includes these 3 tasks, but with one key distinction: we'll print the `nli_tasks` group as a row (with no associated metrics) in our table of outputs, and visually show that these 3 tasks appear under its subheader.

Now, let's assume we actually want to report an aggregate score for `nli_tasks`. We would instead use a YAML config like the following:

```yaml
group: nli_tasks
task:
  - cb
  - anli_r1
  - rte
aggregate_metric_list:
  - metric: acc
    aggregation: mean
    weight_by_size: true # defaults to `true`. Set this to `false` to do a "macro" average (taking each subtask's average accuracy, and summing those accuracies and dividing by 3)--by default we do a "micro" average (retain all subtasks' per-document accuracies, and take the mean over all documents' accuracies to get our aggregate mean).
metadata:
  version: 1.0
```

Similar to our `metric_list` for listing out the metrics we want to calculate for a given task, we use an `aggregate_metric_list` field to specify which metric name to aggregate across subtasks, what aggregation function to use, and whether we should micro- or macro- average these metrics. See [./task_guide.md](./task_guide.md) for a full list of related sub-keys.

**[!Tip]: currently, we predominantly only support the aggregation of group metrics that use `mean` (either micro- or macro- averaged) over their subtasks. If you require even more complex aggregation rules, you may want to perform aggregation offline.**

Group configs can be fairly complex! We can do various operations, such as defining new subtask(s) inline in our group YAML, overriding an existing task's specific config value, or nesting existing groups within our

For example, let's build a config for evaluating MMLU and a few natural language inference tasks. For MMLU, we can write the name for the benchmark as a subtask written under `task`. You can configure the parameters such as `num_fewshot`. If the task being configured is a group such as `mmlu` or `super_glue`, the parameter set will be applied to all of the subtasks.

```yaml
group: nli_and_mmlu
task:
  - group: nli_tasks
    task:
      - cb
      - anli_r1
      - rte
    aggregate_metric_list:
      - metric: acc
        aggregation: mean
        higher_is_better: true
  - task: mmlu
    num_fewshot: 2
```

### Configuring python classes

There can be occasions when yaml-based tasks cannot accommodate how a task is handled. LM-Eval supports the manually implementing tasks as was previously done before `0.4.x`. To register the task, you can simply make a yaml with the name of the task in `task` and the class object in `class` using the `!function` prefix.

```yaml
task: squadv2
class: !function task.SQuAD2
```

This also applies to building group configurations with subtasks that are python classes.

```yaml
group: scrolls
task:
  - task: scrolls_qasper
    class: !function task.Qasper
  - task: scrolls_quality
    class: !function task.QuALITY
  - task: scrolls_narrativeqa
    class: !function task.NarrativeQA
  ...
```

You can also pass a custom argument to your class by accepting `config` in the custom class constructor.
Here's how to do it:

```yaml
task: 20_newsgroups
class: !function task.Unitxt
recipe: card=cards.20_newsgroups,template=templates.classification.multi_class.title
```

In this example, `recipe` is the custom argument for the `Unitxt` class.

## Beautifying Table Display

To avoid conflict, each task needs to be registered with a unique name. Because of this, slight variations of task are still counted as unique tasks and need to be named uniquely. This could be done by appending an additional naming that may refer to the variation such as in MMLU where the template used to evaluated for flan are differentiated from the default by the prefix `mmlu_flan_*`. Printing the full task names can easily clutter the results table at the end of the evaluation especially when you have a long list of tasks or are using a benchmark that comprises of many tasks. To make it more legible, you can use `task_alias` and `group_alias` to provide an alternative task name and group name that will be printed. For example in `mmlu_abstract_algebra.yaml` we set `task_alias` to `abstract_algebra`. In group configs, a `group_alias` for a group can also be set.

```yaml
"dataset_name": "abstract_algebra"
"description": "The following are multiple choice questions (with answers) about abstract\
  \ algebra.\n\n"
"include": "_default_template_yaml"
"task": "mmlu_abstract_algebra"
"task_alias": "abstract_algebra"
```

## Checking validity

After registering your task, you can now check on your data downloading and verify that the few-shot samples look as intended. Run the following command with your desired args:

```bash
python -m scripts.write_out \
    --output_base_path <path> \
    --tasks <your-task-name> \
    --sets <train | val | test> \
    --num_fewshot K \
    --num_examples N \
```

Open the file specified at the `--output_base_path <path>` and ensure it passes
a simple eye test.

## Versioning

One key feature in LM Evaluation Harness is the ability to version tasks and groups--that is, mark them with a specific version number that can be bumped whenever a breaking change is made.

This version info can be provided by adding the following to your new task or group config file:

```yaml
metadata:
  version: 0
```

Now, whenever a change needs to be made to your task in the future, please increase the version number by 1 so that users can differentiate the different task iterations and versions.

If you are incrementing a task's version, please also consider adding a changelog to the task's README.md noting the date, PR number, what version you have updated to, and a one-liner describing the change.

for example,

- \[Dec 25, 2023\] (PR #999) Version 0.0 -> 1.0: Fixed a bug with answer extraction that led to underestimated performance.

## Checking performance + equivalence

It's now time to check models' performance on your task! In the evaluation harness, we intend to support a wide range of evaluation tasks and setups, but prioritize the inclusion of already-proven benchmarks following the precise evaluation setups in the literature where possible.

To enable this, we provide a checklist that should be completed when contributing a new task, to enable accurate book-keeping and to ensure that tasks added to the library are well-tested and, where applicable, precedented.

### Task Validity Checklist

The checklist is the following:

For adding novel benchmarks/datasets to the library:

- [ ] Is the task an existing benchmark in the literature?
  - [ ] Have you referenced the original paper that introduced the task?
  - [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

- [ ] Is the "Main" variant of this task clearly denoted?
- [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
- [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

It is recommended to include a filled-out copy of this checklist in the README.md for the subfolder you are creating, if you have created a new subfolder in `lm_eval/tasks`.

**Finally, please add a short description of your task(s), along with a link to its subfolder in lm_eval/tasks, to [`lm_eval/tasks/README.md`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md) so that users can discover your task in the library, and follow the link to your README for more information about the variants supported, their task names, and the original source of the dataset and/or evaluation setup.**

## Submitting your task

You're all set! Now push your work and make a pull request to the `main` branch! Thanks for the contribution :). If there are any questions, please leave a message in the `#lm-thunderdome` channel on the EAI discord!
