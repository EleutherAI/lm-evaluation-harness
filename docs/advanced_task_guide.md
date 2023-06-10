# Advanced Task Configuration

While adding a standard evaluation task on a new dataset can be occasionally as simple as swapping out a dataset path, with more specialized evaluation setups

## Configurations


### Parameters

- **task** (`str`, defaults to None) — name of the task.
- **group** (`str`, *optional*) —
- **reference** (`str`, *optional*) —
- **task_name** (`str`, *optional*) —
- **dataset_path** (`str`) — The name of the dataset as listed by HF in the datasets Hub. 
- **dataset_name**  (`str`, *optional*, defaults to None) — The name of, what HF calls, a “data instance” or sub-task of the benchmark. If your task does not contain any data instances, just leave this to default to None. (If you're familiar with the HF `datasets.load_dataset` function, these are just the first 2 arguments to it.)
- **dataset_kwargs** (`dict`, *optional*) — Auxillary arguments that `datasets.load_dataset` accepts. This can be used to specify arguments such as `data_files` or `data_dir` if you want to use local datafiles such as json or csv.
- **training_split** (`str`, *optional*) — Split in the dataset to use as the training split.
- **validation_split** (`str`, *optional*) — Split in the dataset to use as the validation split.
- **test_split** (`str`, *optional*) — Split in the dataset to use as the test split.
- **fewshot_split** (`str`, *optional*) — assert that this not None if num_fewshot > 0. (?) assert if this is same split as one evaling (?)
- **template_aliases** (`str`, *optional*) — 
- **aliases**: (`Union[str, list]`, *optional*) —
- **doc_to_text** (`Union[Callable, str]`, *optional*) — Jinja2, f-string, or function to process a sample into the appropriate input for the model
- **doc_to_target** (`Union[Callable, str]`, *optional*) — Jinja2, f-string, or function to process a sample into the appropriate target output for the model
- **num_fewshot** (`int`, *optional*, defaults to 0) — Number of few-shot examples before the input.
- **batch_size** (`int`, *optional*, defaults to 1) — Batch size.
- **repeats** (`int`, *optional*, defaults to 1) — Number of repeated runs for a sample, can be used for cases such as self-consistency.
- **metric_list** (`str`, *optional*, defaults to None) — A list of metrics to use for evaluation.
- **gold_alias** (`str`, *optional*, defaults to None) — 
- **output_type** (`str`, *optional*, defaults to "greedy_until") — Selects the type of model output for the given task. Options are `greedy_until`, `loglikelihood`, `loglikelihood_rolling`, and `multiple_choice`.
- **generation_kwargs** (`dict`, *optional*) — Auxillary arguments for the `generate` function from HF transformers library.
- **delimiter** (`str`, *optional*, defaults to "\n\n") — String to insert between few-shot examples.
- **filter_list** (`Union[str, list]`, *optional*) — List of filters to postprocess model outputs. 
- **normalization** (`str`, *optional*) —
- **should_decontaminate** (`bool`, *optional*, defaults to False)
- **doc_to_decontamination_query** (`str`, *optional*) —
- **use_prompt** (`str`, *optional*) — Name of prompt in promptsource to use, if defined will overwrite doc_to_text.
- **metadata** (`str`, *optional*) —

## Filters

Explain: What are filters? What is their place in the pipeline?

Format of the `resps` object, and what needs to happen to yield proper scorable results
TODO: triviaqa is implementable if we don't use `take_first` and implement a multi-alias exact_match_any metric
TODO: Filters might warrant a separate doc.

### Multiple Filter Pipelines

On the same model outputs, we can perform multiple distinct filtering setups in parallel

Case study: gsm8k-CoT-self-consistency

### "Splitting" Pipelines

TODO: either allow for pipelines that "split" and report multiple keys, or something different. We in particular want to support not re-running reward /scoring models on every different filter pipeline if can be shared.

## Embedded Python Code

There could be cases where Jinja 2 or simple f-string format won't cut it. For tasks like these, we additionally support the importing of Python helper functions that can be injected directly to the yaml. It should be noted that the function script must be in the same directory as the yaml.

TODO: document the `!function filename.pythonfunctionname` syntax here.

TODO: add permannent link to wikitext.yaml and super_glue_cb.yml
```
wikitext.yaml and helper fn go here
```

## (No Longer Recommended) Direct `Task` Subclassing

The prior implementation method of new tasks was to subclass `Task`. While we intend to migrate all tasks to the new YAML implementation option going forward, it remains possible to subclass

{Insert a sample custom `Task` subclass code block here}

## Configuring Tasks with YAMLs

You can easily make a task evaluation using yamls, this is to allow faster and easier experience.

Doc to text
Jinja,
You can use Jinja or f-strings to make a prompt template.
To set a mapping of verbalizer to label, you can define that in the jinja string dorectly.


## Including a Base YAML

You can base a yaml based on a template, if you like the original yaml but just want to change the prompt, you can do that. To do that, use !include and write the name of the template you want to base from. This assumes that the base temeplate is in the same directiry. Otherwise. You will need to define the full path

```
!include: <name or full path to yaml file>

```

## Listing Metrics

For example, setting a `exact_match` (TODO: Add url to metric), auxilarry arguments such as `ignore_case`, `ignore_punctuation`, `regexes_to_ignore` can be listed as well. They will be added to the metric function as `kwargs`.
```
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - "\\$"
```

## Using Promptsource

- load prompt from promptsource
