# Advanced Task Configuration

While adding a standard evaluation task on a new dataset can be occasionally as simple as swapping out a dataset path, with more specialized evaluation setups

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
