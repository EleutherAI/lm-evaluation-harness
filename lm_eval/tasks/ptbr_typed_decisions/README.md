# PT-BR typed decisions

### Paper

No paper for the compilation yet. Dataset card with method, licenses and citations:
https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench

Seven closed-question tasks on real Brazilian Portuguese text, built from openly licensed sources with human labels:

| Task | Question | Source |
|---|---|---|
| `ptbr_decisions_olid_ofensivo` | Is the comment offensive? | OLID-BR (Trajano et al., Language Resources and Evaluation, 2024) |
| `ptbr_decisions_olid_alvo` | Who is the target of the offense? | OLID-BR |
| `ptbr_decisions_factck_veracidade` | Fact-check verdict of a claim | FACTCK.BR (Moreno and Bressan, WebMedia 2019) |
| `ptbr_decisions_faquad_resposta` | Does the passage answer the question? | FaQuAD (Sayama et al., BRACIS 2019), NLI version |
| `ptbr_decisions_scielo_area` | Scientific area of an abstract | SciELO Brasil open-access abstracts |
| `ptbr_decisions_juristcu_area` | Area of an audit-court decision excerpt | JurisTCU (Fernandes et al., Language Resources and Evaluation, 2026) |
| `ptbr_decisions_camara_tema` | Theme of a bill | Câmara dos Deputados open data; LegiSubjects-Br |

Each item lists the options with letters and the model scores each letter (loglikelihood), zero-shot. The headline
metric is balanced accuracy (mean per-class recall), because several tasks are imbalanced; `acc` is also reported.

### Citation

```
@misc{felhen2026ptbrtypeddecisions,
  title  = {PT-BR typed decisions: a benchmark of closed decisions on Brazilian Portuguese text},
  author = {Felhen},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench}}
}
```

Please also cite the source datasets listed on the dataset card.

### Groups, Tags, and Tasks

#### Groups

* `ptbr_typed_decisions`: unweighted mean over the seven tasks.

#### Tags

* `ptbr_typed_decisions_tasks`: the seven tasks.

#### Tasks

* `ptbr_decisions_camara_tema`, `ptbr_decisions_factck_veracidade`, `ptbr_decisions_faquad_resposta`,
  `ptbr_decisions_juristcu_area`, `ptbr_decisions_olid_alvo`, `ptbr_decisions_olid_ofensivo`,
  `ptbr_decisions_scielo_area`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature? It is a new compilation of existing datasets; each source is cited above and on the dataset card.
  * [x] Have you referenced the original paper that introduced the task? The source datasets are referenced.
  * [ ] If yes, does the original paper provide a reference implementation? The dataset card's results use a separate evaluator for decision-model servers, with per-item option shuffling; this harness version keeps the dataset's option order and is not directly comparable with that table.

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted? The group `ptbr_typed_decisions`.
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant? Not applicable.
