# Basic Skills

This task family ports the six-domain Basic Skills benchmark from
[`allenai/olmes@5a51f502`](https://github.com/allenai/olmes/blob/5a51f502d463b8cdc4a2dcad7d7096c41ff1197e/oe_eval/tasks/oe_eval_tasks/basic_skills.py).
It loads each raw JSON file directly from `allenai/basic-skills` at revision
`faf63e9719e124d7741519a024719c8992622630`. Direct raw-file URLs preserve the
source snapshot while remaining compatible with `datasets>=4`, which no longer
executes the repository's `basic-skills.py` loader script. The port provides
the two OLMES prompt forms for every domain:

- `*_rc` scores the shuffled answer strings as continuations.
- `*_mc` writes the shuffled answers into the prompt and scores their letter labels.

Both forms use five examples sampled from the validation data with the OLMES
default seed (`1234` unless overridden at evaluation time). Choice order is
independently deterministic for each example because it is seeded by the
dataset `id`.

The `acc_per_token` metric is the OLMES primary metric. `acc` and lm-eval's
character-normalized `acc_norm` are retained as secondary diagnostics; they are
not aliases for `acc_per_token`.

The pinned OLMES `BasicSkillsMC` class has an `acc_raw` default, but its
registered `:mc::olmes` task configuration overrides that default to
`acc_per_token`. This port follows the effective registered configuration for
both RC and MC.

Run all configurations with `--tasks basic_skills`, or select only
`basic_skills_rc` or `basic_skills_mc`.
