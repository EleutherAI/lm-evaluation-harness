# Advanced Task Configuration

While adding a standard evaluation task on a new dataset can be occasionally as simple as swapping out a dataset path, with more specialized evaluation setups  

## Filters

Explain: What are filters? What is their place in the pipeline?

Format of the `resps` object, and what needs to happen to yield proper scorable results 
TODO: triviaqa is implementable if we don't use `take_first` and implement a multi-alias exact_match_any metric

### Multiple Filter Pipelines

On the same model outputs, we can perform multiple distinct filtering setups in parallel

Case study: gsm8k-CoT-self-consistency

### "Splitting" Pipelines

TODO: either allow for pipelines that "split" and report multiple keys, or something different. We in particular want to support not re-running reward /scoring models on every different filter pipeline if can be shared.

## Embedded Python Code

For tasks requiring preprocessing of the HuggingFace dataset columns that are beyond the complexity of Jinja 2 prompt templating language we use, we additionally support the importing of Python helper functions. 

TODO: document the `!function filename.pythonfunctionname` syntax here.

```
wikitext.yaml and helper fn go here
```

## (No Longer Recommended) Direct `Task` Subclassing

The prior implementation method of new tasks was to subclass `Task`. While we intend to migrate all tasks to the new YAML implementation option going forward, it remains possible to subclass 

{Insert a sample custom `Task` subclass code block here}