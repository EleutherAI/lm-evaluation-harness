# User Guide

This document details the interface exposed by `lm-eval` and provides details on what flags are available to users.

## Command-line Interface

A majority of users run the library by cloning it from Github, installing the package as editable, and running the `python -m lm_eval` script.

Equivalently, running the library can be done via the `lm-eval` entrypoint at the command line.

This mode supports a number of command-line arguments, the details of which can be also be seen via running with `-h` or `--help`:

* `--model` : Selects which model type or provider is evaluated. Must be a string corresponding to the name of the model type/provider being used. See [the main README](https://github.com/EleutherAI/lm-evaluation-harness/tree/main#commercial-apis) for a full list of enabled model names and supported libraries or APIs.

* `--model_args` : Controls parameters passed to the model constructor. Accepts a string containing comma-separated keyword arguments to the model class of the format `"arg1=val1,arg2=val2,..."`, such as, for example `--model_args pretrained=EleutherAI/pythia-160m,dtype=float32`. For a full list of what keyword arguments, see the initialization of the `lm_eval.api.model.LM` subclass, e.g. [`HFLM`](https://github.com/EleutherAI/lm-evaluation-harness/blob/365fcda9b85bbb6e0572d91976b8daf409164500/lm_eval/models/huggingface.py#L66)

* `--tasks` : Determines which tasks or task groups are evaluated. Accepts a comma-separated list of task names or task group names. Must be solely comprised of valid tasks/groups.

* `--num_fewshot` : Sets the number of few-shot examples to place in context. Must be an integer.

* `--gen_kwargs` : takes an arg string in same format as `--model_args` and creates a dictionary of keyword arguments. These will be passed to the models for all called `generate_until` (free-form or greedy generation task) tasks, to set options such as the sampling temperature or `top_p` / `top_k`. For a list of what args are supported for each model type, reference the respective library's documentation (for example, the documentation for `transformers.AutoModelForCausalLM.generate()`.) These kwargs will be applied to all `generate_until` tasks called--we do not currently support unique gen_kwargs or batch_size values per task in a single run of the library. To control these on a per-task level, set them in that task's YAML file.

* `--batch_size` : Sets the batch size used for evaluation. Can be a positive integer or `"auto"` to automatically select the largest batch size that will fit in memory, speeding up evaluation. One can pass `--batch_size auto:N` to re-select the maximum batch size `N` times during evaluation. This can help accelerate evaluation further, since `lm-eval` sorts documents in descending order of context length.

* `--max_batch_size` : Sets the maximum batch size to try to fit in memory, if `--batch_size auto` is passed.

* `--device` : Sets which device to place the model onto. Must be a string, for example, `"cuda", "cuda:0", "cpu", "mps"`. Defaults to "cuda", and can be ignored if running multi-GPU or running a non-local model type.

* `--output_path` : A string of the form `dir/file.jsonl` or `dir/`. Provides a path where high-level results will be saved, either into the file named or into the directory named. If `--log_samples` is passed as well, then per-document outputs and metrics will be saved into the directory as well.

* `--log_samples` : If this flag is passed, then the model's outputs, and the text fed into the model, will be saved at per-document granularity. Must be used with `--output_path`.

* `--limit` : Accepts an integer, or a float between 0.0 and 1.0 . If passed, will limit the number of documents to evaluate to the first X documents (if an integer) per task or first X% of documents per task. Useful for debugging, especially on costly API models.

* `--use_cache` : Should be a path where a sqlite db file can be written to. Takes a string of format `/path/to/sqlite_cache_` in order to create a cache db at `/path/to/sqlite_cache_rank{i}.db` for each process (0-NUM_GPUS). This allows results of prior runs to be cached, so that there is no need to re-run results in order to re-score or re-run a given (model, task) pair again.

* `--decontamination_ngrams_path` : Deprecated, see (this commit)[https://github.com/EleutherAI/lm-evaluation-harness/commit/00209e10f6e27edf5d766145afaf894079b5fe10] or older for a working decontamination-checker tool.

* `--check_integrity` : If this flag is used, the library tests for each task selected are run to confirm task integrity.

* `--write_out` : Used for diagnostic purposes to observe the format of task documents passed to a model. If this flag is used, then prints the prompt and gold target string for the first document of each task.

* `--show_config` : If used, prints the full `lm_eval.api.task.TaskConfig` contents (non-default settings the task YAML file) for each task which was run, at the completion of an evaluation. Useful for when one is modifying a task's configuration YAML locally to transmit the exact configurations used for debugging or for reproducibility purposes.

* `--include_path` : Accepts a path to a folder. If passed, then all YAML files containing `lm-eval`` compatible task configurations will be added to the task registry as available tasks. Used for when one is writing config files for their own task in a folder other than `lm_eval/tasks/`

## External Library Usage

We also support using the library's external API for use within model training loops or other scripts.

`lm_eval` supplies two functions for external import and use: `lm_eval.evaluate()` and `lm_eval.simple_evaluate()`.


`simple_evaluate()` can be used by simply creating an `lm_eval.api.model.LM` subclass that implements the methods described in the [Model Guide](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs/model_guide.md), and wrapping your custom model in that class as follows:

```python
import lm_eval
...

my_model = initialize_my_model() # create your model (could be running finetuning with some custom modeling code)
...
lm_obj = Your_LM(model=my_model, batch_size=16) # instantiate an LM subclass that takes your initialized model and can run `Your_LM.loglikelihood()`, `Your_LM.loglikelihood_rolling()`, `Your_LM.generate_until()`

lm_eval.tasks.initialize_tasks() # register all tasks from the `lm_eval/tasks` subdirectory. Alternatively, can call `lm_eval.tasks.include_path("path/to/my/custom/task/configs")` to only register a set of tasks in a separate directory.

results = lm_eval.simple_evaluate( # call simple_evaluate
    model=lm_obj,
    tasks=["taskname1", "taskname2"],
    num_fewshot=0,
    ...
)
```


See https://github.com/EleutherAI/lm-evaluation-harness/blob/365fcda9b85bbb6e0572d91976b8daf409164500/lm_eval/evaluator.py#L35 for a full description of all arguments available. All keyword arguments to simple_evaluate share the same role as the command-line flags described previously.

Additionally, the `evaluate()` function offers the core evaluation functionality provided by the library, but without some of the special handling and simplification + abstraction provided by `simple_evaluate()`.

See https://github.com/EleutherAI/lm-evaluation-harness/blob/365fcda9b85bbb6e0572d91976b8daf409164500/lm_eval/evaluator.py#L173 for more details.

As a brief example usage of `evaluate()`:
```python
import lm_eval

from my_tasks import MyTask1 # suppose you've defined a custom lm_eval.api.Task subclass in your own external codebase
...

my_model = initialize_my_model() # create your model (could be running finetuning with some custom modeling code)
...
lm_obj = Your_LM(model=my_model, batch_size=16) # instantiate an LM subclass that takes your initialized model and can run `Your_LM.loglikelihood()`, `Your_LM.loglikelihood_rolling()`, `Your_LM.generate_until()`

lm_eval.tasks.initialize_tasks() # register all tasks from the `lm_eval/tasks` subdirectory. Alternatively, can call `lm_eval.tasks.include_path("path/to/my/custom/task/configs")` to only register a set of tasks in a separate directory.

def evaluate(
    lm=lm_obj,
    task_dict={"mytask1": MyTask1},
    ...
):
```
