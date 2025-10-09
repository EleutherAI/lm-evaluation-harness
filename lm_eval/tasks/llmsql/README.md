# LLMSQL

## Paper

Title: `LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL`

Abstract: `https://arxiv.org/abs/2510.02350`

`LLMSQL benchmark is a reviesed version of the WikiSQL benchmark. The benchmark is suited for modern LLMs and for Text2SQL task. The Q&A pairs are single table only, which means that there are no JOINs.`

Homepage: `https://llmsql.github.io/llmsql-benchmark/`

### Citation

```text
@inproceedings{llmsql_bench,
  title={LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL},
  author={Pihulski, Dzmitry and  Charchut, Karol and Novogrodskaia, Viktoria and Koco{'n}, Jan},
  booktitle={2025 IEEE International Conference on Data Mining Workshops (ICDMW)},
  year={2025},
  organization={IEEE}
}
```

### Evaluation Nuances

The evaluation process in this benchmark differs from the standard benchmarks. Since multiple SQL query formulations can yield identical execution results, for example:

```sql
SELECT Guests FROM Table WHERE Guests = "Tom" AND Age = "30";
```

is equivalent to:

```sql
SELECT Guests FROM Table WHERE Age = "30" AND Guests = "Tom";
```

the goal of this evaluation is **not** to compare SQL string similarity, but rather to **execute the model-generated queries** (up to 10 candidates extracted via regex) against the database and compare their outputs with the ground truth.

The resulting metric, **Execution Accuracy**, reflects the proportion of predictions whose executed results exactly match the reference outputs.

A helper SQLite database (`sqlite_tables.db`) required for execution-based evaluation will be automatically downloaded to the working directory at the start of the evaluation process.
