# Task Configuration

The `lm-evaluation-harness` is meant to be an extensible and flexible framework within which many different evaluation tasks can be defined. All tasks in the new version of the harness are built around a YAML configuration file format.

These YAML configuration files, along with the current codebase commit hash, are intended to be shareable such that providing the YAML config enables another researcher to precisely replicate the evaluation setup used by another, in the case that the prompt or setup differs from standard `lm-eval` task implementations.

While adding a standard evaluation task on a new dataset can be occasionally as simple as swapping out a Hugging Face dataset path in an existing file, more specialized evaluation setups. Here we'll provide a crash course on the more advanced logic implementable in YAML form available to users.

If your intended task relies on features beyond what are described in this guide, we'd love to hear about it! Feel free to open an issue describing the scenario on Github, create a PR to the project with a proposed implementation, or ask in the `#lm-thunderdome` channel on the EleutherAI discord.

## Configurations

Tasks are configured via the `TaskConfig` object. Below, we describe all fields usable within the object, and their role in defining a task.

### Parameters

Task naming + registration:
- **task** (`str`, defaults to None) — name of the task.
- **group** (`str`, *optional*) — name of the task group(s) a task belongs to. Enables one to run all tasks with a specified tag or group name at once.

Dataset configuration options:
- **dataset_path** (`str`) — The name of the dataset as listed by HF in the datasets Hub.
- **dataset_name**  (`str`, *optional*, defaults to None) — The name of what HF calls a “data instance” or sub-task of the benchmark. If your task does not contain any data instances, just leave this to default to None. (If you're familiar with the HF `datasets.load_dataset` function, these are just the first 2 arguments to it.)
- **dataset_kwargs** (`dict`, *optional*) — Auxiliary arguments that `datasets.load_dataset` accepts. This can be used to specify arguments such as `data_files` or `data_dir` if you want to use local datafiles such as json or csv.
- **training_split** (`str`, *optional*) — Split in the dataset to use as the training split.
- **validation_split** (`str`, *optional*) — Split in the dataset to use as the validation split.
- **test_split** (`str`, *optional*) — Split in the dataset to use as the test split.
- **fewshot_split** (`str`, *optional*) — Split in the dataset to draw few-shot exemplars from. assert that this not None if num_fewshot > 0.
- **process_docs** (`Callable`, *optional*) — Optionally define a function to apply to each HF dataset split, to preprocess all documents before being fed into prompt template rendering or other evaluation steps. Can be used to rename dataset columns, or to process documents into a format closer to the expected format expected by a prompt template.

Prompting / in-context formatting options:
- **use_prompt** (`str`, *optional*) — Name of prompt in promptsource to use. if defined, will overwrite doc_to_text, doc_to_target, and doc_to_choice.
- **doc_to_text** (`Union[Callable, str]`, *optional*) — Jinja2, f-string, or function to process a sample into the appropriate input for the model
- **doc_to_target** (`Union[Callable, str]`, *optional*) — Jinja2, f-string, or function to process a sample into the appropriate target output for the model. For multiple choice tasks, this should return an index into
- **doc_to_choice** (`Union[Callable, str]`, *optional*) — Jinja2, f-string, or function to process a sample into a list of possible string choices for `multiple_choice` tasks. Left undefined for `generate_until` tasks.
- **fewshot_delimiter** (`str`, *optional*, defaults to "\n\n") — String to insert between few-shot examples.
- **target_delimiter** (`str`, *optional*, defaults to `" "`) — String to insert between input and target output for the datapoint being tested.

Runtime configuration options:
- **num_fewshot** (`int`, *optional*, defaults to 0) — Number of few-shot examples before the input.
- **batch_size** (`int`, *optional*, defaults to 1) — Batch size.

Scoring details:
- **metric_list** (`str`, *optional*, defaults to None) — A list of metrics to use for evaluation. See docs for expected format.
- **output_type** (`str`, *optional*, defaults to "generate_until") — Selects the type of model output for the given task. Options are `generate_until`, `loglikelihood`, `loglikelihood_rolling`, and `multiple_choice`.
- **generation_kwargs** (`dict`, *optional*) — Auxiliary arguments for the `generate` function from HF transformers library. Advanced keyword arguments may not be supported for non-HF LM classes.
- **repeats** (`int`, *optional*, defaults to 1) — Number of repeated runs through model for each sample. can be used for cases such as self-consistency.
- **filter_list** (`Union[str, list]`, *optional*) — List of filters to postprocess model outputs. See below for further detail on the filter API.
- **should_decontaminate** (`bool`, *optional*, defaults to False) - Whether to decontaminate or not.
- **doc_to_decontamination_query** (`str`, *optional*) — Query for decontamination if `should_decontaminate` is True. If `should_decontaminate` is True but `doc_to_decontamination_query` is `None`, `doc_to_decontamination_query` will follow `doc_to_text`.

Other:
- **metadata** (`Union[str, list]`, *optional*) — An optional field where arbitrary metadata can be passed. A good example would be `version` that is used to denote the version of the yaml config.

## Filters

Explain: What are filters? What is their place in the pipeline?

A key component of the `lm-evaluation-harness` library is the `Filter` object. In a typical evaluation run of the harness, we take the formatted inputs and run them through our LM, with the appropriate output type (greedy or free-form generation, or loglikelihood-based comparative scoring).

After getting scores or output text from our LM on each `Instance` or document in the dataset, we then need to feed these responses into a metric or scoring function to return scores to a user.

However, certain tasks may require more complex behavior than directly turning over model outputs to a metric function. For example, we may want to post-process our output text by truncating it or extracting a model's answer, we may want to ensemble over multiple "takes" on a different document, et cetera.

**Detailed Aside**:
We do such post-processing by operating on *responses*, which are stored after running an LM on an `Instance` from the task in `Instance.resps`.

`resps` is a `List[str]` for each instance, and we pass a `List[List[<expected return type from model>]]` to our filters that is a list of `[instance.resps for instance in instances]`.

Our filters, after completing a pipeline, must return a `List[<expected return type from model>]` which we then unpack and store each element of in `Instance.filtered_resps` for the corresponding instance. Thus, we take as input a list of returns from our model for each doc, and must return a return from our model *without it being wrapped in a list* for each doc.

**End Aside**


A full list of supported filter operations can be found in `lm_eval/filters/__init__.py`. Contributions of new filter types are welcome!

### Multiple Filter Pipelines

Tasks need not be limited to a single filter pipeline. We enable users to run multiple, distinct, filter pipelines on *the same model outputs* generated in one run on a task.

As a case study, let's look at an implementation of solving the Gsm8k math word problem benchmark in `lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml`. Here, we are emulating the setup used by [Self-Consistency Improves Chain of Thought Prompting](https://arxiv.org/abs/2203.11171), in which evaluation is performed by generating N chain-of-thought outputs from a model via temperature-based sampling, then selecting the answers output by the model at the end of the chains of thought, then majority voting across all those numeric answers.

Within our YAML file:

```yaml
...
repeats: 64
filter_list:
  - name: "score-first"
    filter:
      - function: "regex"
        regex_pattern: "The answer is (\\-?[0-9\\.\\,]*[0-9]+)"
      - function: "take_first"
  - name: "maj@64"
    filter:
      - function: "regex"
        regex_pattern: "The answer is (\\-?[0-9\\.\\,]*[0-9]+)"
      - function: "majority_vote"
      - function: "take_first"
  - name: "maj@8"
    filter:
      - function: "take_first_k"
        k: 8
      - function: "regex"
        regex_pattern: "The answer is (\\-?[0-9\\.\\,]*[0-9]+)"
      - function: "majority_vote"
      - function: "take_first"
```

We are able to provide multiple different filter pipelines, each with their own name and list of filters to apply in sequence.

Our first filter pipeline implements
- applying a regex to the model generations (extracting the number within the phrase "The answer is (number)")
- selecting only the first out of the 64 model answers

Then scoring this single answer.

```yaml
- name: "score-first"
  filter:
    - function: "regex"
      regex_pattern: "The answer is (\\-?[0-9\\.\\,]*[0-9]+)"
    - function: "take_first"
```

Our second filter pipeline, "maj@64", does majority voting across all 64 answers via:
- applying the same regex to all responses, to get the numerical answer from the model for each of the 64 responses per problem
- applying majority voting to all responses, which then returns a length-1 `[<majority answer>]` list for each
- taking the first element of this length-1 list, to then score the sole response `<majority answer>` for each document.

```yaml
- name: "maj@64"
  filter:
    - function: "regex"
      regex_pattern: "The answer is (\\-?[0-9\\.\\,]*[0-9]+)"
    - function: "majority_vote"
    - function: "take_first"
```

Our final filter pipeline, "maj@8", does majority voting across the first 8 of the model's responses per document via:
- subsetting the len-64 list of responses `[answer1, answer2, ..., answer64]` to `[answer1, answer2, ..., answer8]` for each document
- performing the same sequence of filters on these new sets of 8 responses, for each document.
```yaml
- name: "maj@8"
  filter:
    - function: "take_first_k"
      k: 8
    - function: "regex"
      regex_pattern: "The answer is (\\-?[0-9\\.\\,]*[0-9]+)"
    - function: "majority_vote"
    - function: "take_first"
```

Thus, given the 64 responses from our LM on each document, we can report metrics on these responses in these 3 different ways, as defined by our filter pipelines.


## Embedded Python Code

Use can use python functions for certain arguments by using the `!function` operator after the argument name followed by `<filename>.<pythonfunctionname>`. This feature can be used for the following arguments:
1. `doc_to_text`
2. `doc_to_target`
3. `doc_to_choice`
4. `aggregation` for a `metric` in `metric_list`

## (No Longer Recommended) Direct `Task` Subclassing

The prior implementation method of new tasks was to subclass `Task`. While we intend to migrate all tasks to the new YAML implementation option going forward, it remains possible to subclass the Task class and implement custom logic. For more information, see `docs/task_guide.md` in v0.3.0 of the `lm-evaluation-harness`.


## Including a Base YAML

You can base a YAML on another YAML file as a template. This can be handy when you need to just change the prompt for `doc_to_text` but keep the rest the same or change `filters` to compare which is better. Simply use `include` in the YAML file and write the name of the template you want to base from. This assumes that the base temeplate is in the same directory. Otherwise, You will need to define the full path.
```
include: <YAML filename or with full path>
...
```
You can find an example of how to use this feature at [gsm8k-cot-self-consistency.yaml](https://github.com/EleutherAI/lm-evaluation-harness/blob/3c07cc04a92fc467d7c9a94894aeddd58c93a5da/lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml) where it is based off [gsm8k-cot.yaml](https://github.com/EleutherAI/lm-evaluation-harness/blob/3c07cc04a92fc467d7c9a94894aeddd58c93a5da/lm_eval/tasks/gsm8k/gsm8k-cot.yaml)


## Passing Arguments to Metrics

Metrics can be defined in the `metric_list` argument when building the YAML config. Multiple metrics can be listed along with any auxiliary arguments. For example, setting the [`exact_match` metric](https://github.com/huggingface/evaluate/tree/main/metrics/exact_match), auxiliary arguments such as `ignore_case`, `ignore_punctuation`, `regexes_to_ignore` can be listed as well. They will be added to the metric function as `kwargs`. Some metrics have predefined values for `aggregation` and `higher_is_better` so listing the metric name only can be sufficient.

```
metric_list:
  - metric: acc
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - "\\$"
```

### Natively Supported Metrics

Here we list all metrics currently supported natively in `lm-eval`:

Metrics:
* `acc` (accuracy)
* `acc_norm` (length-normalized accuracy)
* `acc_mutual_info` (baseline loglikelihood - normalized accuracy)
* `perplexity`
* `word_perplexity` (perplexity per word)
* `byte_perplexity` (perplexity per byte)
* `bits_per_byte`
* `matthews_corrcoef` (Matthews correlation coefficient)
* `f1` (F1 score)
* `bleu`
* `chrf`
* `ter`

Aggregation functions:
* `mean`
* `median`
* `perplexity`
* `weighted_perplexity`
* `bits_per_byte`


## Good Reference Tasks

Contributing a new task can be daunting! Luckily, much of the work has often been done for you in a different, similarly evaluated task. Good examples of task implementations to study include:

Multiple choice tasks:
- SciQ (`lm_eval/tasks/sciq/sciq.yaml`)

Corpus perplexity evaluations:
- Wikitext (`lm_eval/tasks/wikitext/wikitext.yaml`)

Generative tasks:
- GSM8k (`lm_eval/tasks/gsm8k/gsm8k.yaml`)

Tasks using complex filtering:
- GSM8k with CoT (+ with Self-Consistency): (`lm_eval/tasks/gsm8k/gsm8k-cot.yaml` ; `lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml`)


## Benchmarks

When evaluating a language model, it's is not unusual to test across a number of tasks that may not be related to one another in order to assess a variety of capabilities. To this end, it may be combursome to have to list the set of tasks or add a new group name to each yaml of each individual task.

To solve this, we can create a benchmark yaml config. This is a config that contains the names of the tasks that should be included in a particular benchmark. The config consists of two main keys `group` which denotes the name of the benchmark and `task` which is where we can list the tasks. The tasks listed in `task` are the task names that have been registered. A good example would be the list of tasks used to evaluate the Pythia Suite.

```yaml
group: pythia
task:
  - lambada_openai
  - wikitext
  - piqa
  - sciq
  - wsc
  - winogrande
  - arc
  - logiqa
  - blimp
  - hendrycksTest*
```

Alternatively, benchmarks can have tasks that are customizable for each task. They can be defined like how a yaml task is usually set.

```yaml
group: t0_eval
task:
  # Coreference Resolution
  - dataset_path: super_glue
    dataset_name: wsc.fixed
    use_prompt: promptsource:*
    training_split: train
    validation_split: validation
    metric_list:
      - metric: exact_match
        aggregation: mean
        higher_is_better: true
        ignore_case: true
        ignore_punctuation: true
  # Coreference Resolution
  - dataset_path: winogrande
    dataset_name: winogrande_xl
    use_prompt: promptsource:*
    training_split: train
    validation_split: validation
    metric_list:
      - metric: exact_match
        aggregation: mean
        higher_is_better: true
        ignore_case: true
        ignore_punctuation: true
  ...
```

If the benchmark contains the same dataset but with different configurations, use `task` to differentiate between them. For example, T0-Eval evaluates on 3 versions of ANLI but the huggingface dataset collects them in one dataset.

```YAML
group: t0_eval
task:
  ...
  - task: anli_r1
    dataset_path: anli
    use_prompt: promptsource:*
    training_split: train_r1
    validation_split: dev_r1
    metric_list:
      - metric: exact_match
        aggregation: mean
        higher_is_better: true
        ignore_case: true
        ignore_punctuation: true
  - task: anli_r2
    dataset_path: anli
    use_prompt: promptsource:*
    training_split: train_r2
    validation_split: dev_r2
    metric_list:
      - metric: exact_match
        aggregation: mean
        higher_is_better: true
        ignore_case: true
        ignore_punctuation: true
```

Calling the benchmark is done the same way we would call any task with `--tasks`. Benchmarks can be added in `lm_eval/benchmarks/`
