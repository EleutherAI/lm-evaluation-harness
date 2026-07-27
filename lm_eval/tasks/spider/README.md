# Spider

### Paper

Title: `Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task`

Abstract: https://arxiv.org/abs/1809.08887

Spider is a large-scale, cross-domain text-to-SQL benchmark: 200 databases spanning
138 domains, annotated with 10,181 natural-language questions and their corresponding
SQL queries. Unlike single-database benchmarks, Spider requires generalizing to unseen
databases and schemas at evaluation time.

Homepage: https://yale-lily.github.io/spider

### Data

The official Spider release (train/dev splits, `tables.json` schema definitions, and
per-database SQLite files) is distributed via a Google Drive link on the task
homepage rather than a script-loadable location, and the reference execution-accuracy
evaluator (`taoyds/test-suite-sql-eval`) is not pip-installable. To make this task
runnable through `dataset_path` like any other `lm-eval` task, the official archive was
repackaged into an HF Hub dataset repo (`<hf-username>/spider-eval`) containing:

- `data/train.jsonl`, `data/validation.jsonl`: `db_id`, `question`, `query` (gold SQL)
- `schema.json`: `db_id` -> `CREATE TABLE` schema text, derived from `tables.json`
- `database/<db_id>/<db_id>.sqlite`: the per-database SQLite files used for execution

See `scripts/prepare_spider_hf_dataset.py` for the extraction/repackaging script. To
point the task at your own copy of the dataset, set the `SPIDER_HF_REPO` environment
variable (defaults to the repo used during development).

### Task

The model is prompted with the database schema (as `CREATE TABLE` statements) and the
natural-language question, and generates a single SQL query. The query is executed
against the real SQLite database and its result set is compared against the result set
of the gold query (`execution accuracy`), rather than doing a string/exact match against
the gold SQL text. This follows the standard evaluation protocol used in the Spider
leaderboard and related text-to-SQL literature. Comparison is order-sensitive only when
the gold query contains `ORDER BY`, matching the convention used by the official Spider
evaluation scripts.

- `dev` (`validation` split): 1,034 examples, used for scoring.
- `train`: 7,000 examples (available for few-shot use; not used by default).

### Task validity checklist

For adding novel benchmarks/datasets to the library:

- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task?
  - [x] If yes, does the original paper provide a reference implementation? If so, have
        you checked against the reference implementation and documented how to run such
        a test? -- The official reference evaluator is `taoyds/test-suite-sql-eval`
        (execution accuracy with test-suite databases). This task implements execution
        accuracy against the single provided database per example (not the expanded
        test-suite databases), which is a lighter-weight approximation of the same
        metric; scores are expected to correlate with, but not exactly reproduce, the
        official leaderboard numbers.

### Groups, Tags, and Tasks

#### Tags

* `text_to_sql`: text-to-SQL semantic parsing tasks.

#### Tasks

* `spider`: execution-accuracy scored Spider evaluation on the `dev` split.

### Changelog

- \[2026-07-27\] Version 0.0: Initial implementation.
