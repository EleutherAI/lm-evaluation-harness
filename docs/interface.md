# User Guide

This document details the interface exposed by `lm-eval` and provides details on what flags are available to users.

## Command-line Interface

A majority of users run the library by cloning it from Github, installing the package as editable, and running the `python -m lm_eval` script.

Equivalently, running the library can be done via the `lm-eval` entrypoint at the command line.

### Subcommand Structure

The CLI now uses a subcommand structure for better organization:

- `lm-eval run` - Execute evaluations (default behavior)
- `lm-eval list` - List available tasks, models, etc.
- `lm-eval validate` - Validate task configurations

For backward compatibility, if no subcommand is specified, `run` is automatically inserted. So `lm-eval --model hf --tasks hellaswag` is equivalent to `lm-eval run --model hf --tasks hellaswag`.

### Run Command Arguments

The `run` command supports a number of command-line arguments. Details can also be seen via running with `-h` or `--help`:

#### Configuration

- `--config` **[path: str]** : Set initial arguments from a YAML configuration file. Takes a path to a YAML file that contains argument values. This allows you to specify complex configurations in a file rather than on the command line. Further CLI arguments can override values from the configuration file.

  For the complete list of available configuration fields and their types, see [`EvaluatorConfig` in the source code](../lm_eval/config/evaluate_config.py).

#### Model and Tasks

- `--model` **[str, default: "hf"]** : Selects which model type or provider is evaluated. Must be a string corresponding to the name of the model type/provider being used. See [the main README](https://github.com/EleutherAI/lm-evaluation-harness/tree/main#model-apis-and-inference-servers) for a full list of enabled model names and supported libraries or APIs.

- `--model_args` **[comma-sep str | json str → dict]** : Controls parameters passed to the model constructor. Can be provided as:
  - Comma-separated string: `pretrained=EleutherAI/pythia-160m,dtype=float32`
  - JSON string: `'{"pretrained": "EleutherAI/pythia-160m", "dtype": "float32"}'`

  For a full list of supported arguments, see the initialization of the `lm_eval.api.model.LM` subclass, e.g. [`HFLM`](https://github.com/EleutherAI/lm-evaluation-harness/blob/365fcda9b85bbb6e0572d91976b8daf409164500/lm_eval/models/huggingface.py#L66)

- `--tasks` **[comma-sep str → list[str]]** : Determines which tasks or task groups are evaluated. Accepts a comma-separated list of task names or task group names. Must be solely comprised of valid tasks/groups. A list of supported tasks can be viewed with `lm-eval list tasks`.

#### Evaluation Settings

- `--num_fewshot` **[int]** : Sets the number of few-shot examples to place in context. Must be an integer.

- `--batch_size` **[int | "auto" | "auto:N", default: 1]** : Sets the batch size used for evaluation. Options:
  - Integer: Fixed batch size (e.g., `8`)
  - `"auto"`: Automatically select the largest batch size that fits in memory
  - `"auto:N"`: Re-select maximum batch size N times during evaluation

  Auto mode is useful since `lm-eval` sorts documents in descending order of context length.

- `--max_batch_size` **[int]** : Sets the maximum batch size to try when using `--batch_size auto`.

- `--device` **[str]** : Sets which device to place the model onto. Examples: `"cuda"`, `"cuda:0"`, `"cpu"`, `"mps"`. Can be ignored if running multi-GPU or non-local model types.

- `--gen_kwargs` **[comma-sep str | json str → dict]** : Generation arguments for `generate_until` tasks. Same format as `--model_args`:
  - Comma-separated: `temperature=0.8,top_p=0.95`
  - JSON: `'{"temperature": 0.8, "top_p": 0.95}'`

  See model documentation (e.g., `transformers.AutoModelForCausalLM.generate()`) for supported arguments. Applied to all generation tasks - use task YAML files for per-task control.

#### Data and Output

- `--output_path` **[path: str]** : Output location for results. Format options:
  - Directory: `results/` - saves as `results/<model_name>_<timestamp>.json`
  - File: `results/output.jsonl` - saves to specific file

  When used with `--log_samples`, per-document outputs are saved in the directory.

- `--log_samples` **[flag, default: False]** : Save model outputs and inputs at per-document granularity. Requires `--output_path`. Automatically enabled when using `--predict_only`.

- `--limit` **[int | float]** : Limit evaluation examples per task. **WARNING: Only for testing!**
  - Integer: First N documents (e.g., `100`)
  - Float (0.0-1.0): Percentage of documents (e.g., `0.1` for 10%)

- `--samples` **[path | json str | dict → dict]** : Evaluate specific sample indices only. Input formats:
  - JSON file path: `samples.json`
  - JSON string: `'{"hellaswag": [0, 1, 2], "arc_easy": [10, 20]}'`
  - Dictionary (programmatic use)

  Format: `{"task_name": [indices], ...}`. Incompatible with `--limit`.

#### Caching and Performance

- `--use_cache` **[path: str]** : SQLite cache database path prefix. Creates per-process cache files:
  - Single GPU: `/path/to/cache.db`
  - Multi-GPU: `/path/to/cache_rank0.db`, `/path/to/cache_rank1.db`, etc.

  Caches model outputs to avoid re-running the same (model, task) evaluations.

- `--cache_requests` **["true" | "refresh" | "delete"]** : Dataset request caching control:
  - `"true"`: Use existing cache
  - `"refresh"`: Regenerate cache (use after changing task configs)
  - `"delete"`: Delete cache

  Cache location: `lm_eval/cache/.cache` or `$LM_HARNESS_CACHE_PATH` if set.

- `--check_integrity` **[flag, default: False]** : Run task integrity tests to validate configurations.

#### Instruct Formatting

- `--system_instruction` **[str]** : Custom system instruction to prepend to prompts. Used with instruction-following models.

- `--apply_chat_template` **[bool | str, default: False]** : Apply chat template formatting. Usage:
  - No argument: Apply default/only available template
  - Template name: Apply specific template (e.g., `"chatml"`)

  For HuggingFace models, uses the tokenizer's chat template. Default template defined in [`transformers` documentation](https://github.com/huggingface/transformers/blob/fc35907f95459d7a6c5281dfadd680b6f7b620e3/src/transformers/tokenization_utils_base.py#L1912).

- `--fewshot_as_multiturn` **[flag, default: False]** : Format few-shot examples as multi-turn conversation:
  - Questions → User messages
  - Answers → Assistant responses

  Requires: `--num_fewshot > 0` and `--apply_chat_template` enabled.

#### Task Management

- `--include_path` **[path: str]** : Directory containing custom task YAML files. All `.yaml` files in this directory will be registered as available tasks. Use for custom tasks outside of `lm_eval/tasks/`.

#### Logging and Tracking

- `--verbosity` **[str]** : **DEPRECATED** - Use `LOGLEVEL` environment variable instead.

- `--write_out` **[flag, default: False]** : Print first document's prompt and target for each task. Useful for debugging prompt formatting.

- `--show_config` **[flag, default: False]** : Display full task configurations after evaluation. Shows all non-default settings from task YAML files.

- `--wandb_args` **[comma-sep str → dict]** : Weights & Biases integration. Arguments for `wandb.init()`:
  - Example: `project=my-project,name=run-1,tags=test`
  - Special: `step=123` sets logging step
  - See [W&B docs](https://docs.wandb.ai/ref/python/init) for all options

- `--wandb_config_args` **[comma-sep str → dict]** : Additional W&B config arguments, same format as `--wandb_args`.

- `--hf_hub_log_args` **[comma-sep str → dict]** : Hugging Face Hub logging configuration. Format: `key1=value1,key2=value2`. Options:
  - `hub_results_org`: Organization name (default: token owner)
  - `details_repo_name`: Repository for detailed results
  - `results_repo_name`: Repository for aggregated results
  - `push_results_to_hub`: Enable pushing (`True`/`False`)
  - `push_samples_to_hub`: Push samples (`True`/`False`, requires `--log_samples`)
  - `public_repo`: Make repo public (`True`/`False`)
  - `leaderboard_url`: Associated leaderboard URL
  - `point_of_contact`: Contact email
  - `gated`: Gate the dataset (`True`/`False`)
  - ~~`hub_repo_name`~~: Deprecated, use `details_repo_name` and `results_repo_name`

#### Advanced Options

- `--predict_only` **[flag, default: False]** : Generate outputs without computing metrics. Automatically enables `--log_samples`. Use to get raw model outputs.

- `--seed` **[int | comma-sep str → list[int], default: [0,1234,1234,1234]]** : Set random seeds for reproducibility:
  - Single integer: Same seed for all (e.g., `42`)
  - Four values: `python,numpy,torch,fewshot` seeds (e.g., `0,1234,8,52`)
  - Use `None` to skip setting a seed (e.g., `0,None,8,52`)

  Default preserves backward compatibility.

- `--trust_remote_code` **[flag, default: False]** : Allow executing remote code from Hugging Face Hub. **Security Risk**: Required for some models with custom code.

- `--confirm_run_unsafe_code` **[flag, default: False]** : Acknowledge risks when running tasks that execute arbitrary Python code (e.g., code generation tasks).

- `--metadata` **[json str → dict]** : Additional metadata for specific tasks. Format: `'{"key": "value"}'`. Required by tasks like RULER that need extra configuration.

## External Library Usage

We also support using the library's external API for use within model training loops or other scripts.

`lm_eval` supplies two functions for external import and use: `lm_eval.evaluate()` and `lm_eval.simple_evaluate()`.

`simple_evaluate()` can be used by simply creating an `lm_eval.api.model.LM` subclass that implements the methods described in the [Model Guide](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs/model_guide.md), and wrapping your custom model in that class as follows:

```python
import lm_eval
from lm_eval.utils import setup_logging
...
# initialize logging
setup_logging("DEBUG") # optional, but recommended; or you can set up logging yourself
my_model = initialize_my_model() # create your model (could be running finetuning with some custom modeling code)
...
# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`
lm_obj = Your_LM(model=my_model, batch_size=16)

# indexes all tasks from the `lm_eval/tasks` subdirectory.
# Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
# to include a set of tasks in a separate directory.
task_manager = lm_eval.tasks.TaskManager()

# Setting `task_manager` to the one above is optional and should generally be done
# if you want to include tasks from paths other than ones in `lm_eval/tasks`.
# `simple_evaluate` will instantiate its own task_manager if it is set to None here.
results = lm_eval.simple_evaluate( # call simple_evaluate
    model=lm_obj,
    tasks=["taskname1", "taskname2"],
    num_fewshot=0,
    task_manager=task_manager,
    ...
)
```

See the `simple_evaluate()` and `evaluate()` functions in [lm_eval/evaluator.py](../lm_eval/evaluator.py#:~:text=simple_evaluate) for a full description of all arguments available. All keyword arguments to simple_evaluate share the same role as the command-line flags described previously.

Additionally, the `evaluate()` function offers the core evaluation functionality provided by the library, but without some of the special handling and simplification + abstraction provided by `simple_evaluate()`.

As a brief example usage of `evaluate()`:

```python
import lm_eval

# suppose you've defined a custom lm_eval.api.Task subclass in your own external codebase
from my_tasks import MyTask1
...

# create your model (could be running finetuning with some custom modeling code)
my_model = initialize_my_model()
...

# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`
lm_obj = Your_LM(model=my_model, batch_size=16)

# optional: the task_manager indexes tasks including ones
# specified by the user through `include_path`.
task_manager = lm_eval.tasks.TaskManager(
    include_path="/path/to/custom/yaml"
    )

# To get a task dict for `evaluate`
task_dict = lm_eval.tasks.get_task_dict(
    [
        "mmlu", # A stock task
        "my_custom_task", # A custom task
        {
            "task": ..., # A dict that configures a task
            "doc_to_text": ...,
            },
        MyTask1 # A task object from `lm_eval.task.Task`
        ],
    task_manager # A task manager that allows lm_eval to
                 # load the task during evaluation.
                 # If none is provided, `get_task_dict`
                 # will instantiate one itself, but this
                 # only includes the stock tasks so users
                 # will need to set this if including
                 # custom paths is required.
    )

results = evaluate(
    lm=lm_obj,
    task_dict=task_dict,
    ...
)
```
