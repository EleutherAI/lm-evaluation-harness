# Interactive multi-turn tasks

`output_type: multi_turn_generate` evaluates sequential episodes through the
existing model backend's `generate_until` method. Each wave batches the next
requests of active episodes; subsequent prompts can depend on actual earlier
responses. This differs from `fewshot_as_multiturn`, which only formats examples.

Implement a Python `Task`/`ConfigurableTask` subclass with these hooks:

| Hook | Contract |
| --- | --- |
| `init_multiturn_state(doc, ctx, gen_kwargs, apply_chat_template=False, chat_template=None)` | Return fresh episode state. The driver deep-copies generation kwargs for every document/repeat. |
| `multiturn_next_request(state)` | Return `(prompt, gen_kwargs)` for one generation, or `None` after marking the episode done. |
| `multiturn_consume_response(state, response)` | Update history/state with generated text and advance the episode. |
| `multiturn_is_done(state)` | Return whether the episode is complete. |
| `multiturn_result(state)` | Return the completed episode object consumed by the task's `process_results`. |

The supplied `chat_template` is the model's renderer when `--apply_chat_template`
is enabled. Tasks decide how to retain system messages and build history. Tasks
that construct all prompts themselves may override `fewshot_context` to return
an empty initial context, as POLAR does. Declaring this output type alone does
not implement a conversation; the hooks must be supplied by the task class.

One placeholder `Instance` per document is built and cached by the normal task
machinery. For interactive tasks the request-cache key also includes the task
version, exported configuration and sample selection, so changed generation
options do not reuse stale placeholders. Tasks loading variable/local data must
record its content digest in task metadata before building requests; POLAR does
this with the input JSON file's SHA256. Input documents must be treated as
read-only; evolving conversation state belongs in the episode.

The driver runs each repeat independently, then puts completed episode
results into that instance's `resps`. Standard filters, task scoring, logging,
aggregation, limits, and sample selection continue through the existing pipeline.
Do not include clients, renderers, credentials, or mutable episode state in the
placeholder or exported task configuration.

Set `MAX_MULTITURN_STEPS` (default 64) on the task to bound the number of model
backend calls per episode. External tools/attacker calls are task-owned and are
not counted by this driver limit. The cap is per task, including mixed tasks in a
run. Reaching it before completion raises; returning `None` without marking the
episode done also raises. Model exceptions and mismatched response counts are
propagated, so interrupted conversations are not silently scored.

The first implementation requires `world_size=1` and text generation. Model
tensor parallelism is supported. Distributed evaluation fails before request
construction or collectives. No model backend changes are required. The driver
returns the task-defined episode result, so task-specific filters/scorers must
accept that format. Response caching of interactive episodes is not validated.

See [POLAR-Bench](../lm_eval/tasks/polar_bench/README.md) and
`tests/test_multiturn.py` for an implementation and offline examples.
