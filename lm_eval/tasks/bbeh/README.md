# BBEH

This task port loads the 23 full BIG-Bench Extra Hard task files and the
official 460-example mini set directly from
[`google-deepmind/bbeh@80d12ca`](https://github.com/google-deepmind/bbeh/tree/80d12ca916b7158f22293fcf3144f4d3d854d4be).
It preserves the source answer extraction, normalization, and fuzzy matcher.

Selectors:

- `bbeh`: all 23 full tasks, with per-task rows and micro accuracy;
- `bbeh_mini`: the official combined mini release and its micro accuracy;
- `bbeh_<task>`: one full task.

The official `BBEH` headline is the adjusted harmonic mean of the 23 full-task
accuracies: add one percentage point to each accuracy, then take their harmonic
mean. The current lm-eval group API only supports
arithmetic aggregation, so compute that value from the emitted per-task rows with
`utils.harmonic_mean`; the group-level `bbeh_acc` is the official micro result.

The BBEH software is Apache-2.0 and the benchmark materials are CC-BY-4.0.
BBEH derives some tasks from earlier datasets; follow the constituent-task
attribution and citation requirements in the pinned upstream README.
