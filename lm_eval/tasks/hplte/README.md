# HPLT-e

## Paper

Title: `HPLT 3.0: Very Large-Scale Multilingual Resources for LLMs and MT. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models`

Abstract: https://aclanthology.org/2026.lrec-1.110/

HPLT-e is a framework for multilingual and multi-prompt *k*-shot evaluation of language models, developed in the [HPLT project](https://hplt-project.org) to compare pretraining corpora. It covers 124 tasks in nine typologically diverse languages, with an emphasis on natively created, human-curated datasets. Every task comes with several prompts written by native speakers, to account for prompt sensitivity.

Homepage: https://github.com/hplt-project/hplt-e

### Languages

This folder contains the HPLT-e tasks for six languages and the Czech Global-MMLU prompts. Every task has three prompts.

| Language | Folder | Datasets | Based on |
|:---|:---|:---|:---|
| Basque | [`eus_Latn`](eus_Latn/README.md) | 16 | BasqueBench ([`basque_bench`](../basque_bench/README.md)) |
| Catalan | [`cat_Latn`](cat_Latn/README.md) | 23 | CatalanBench ([`catalan_bench`](../catalan_bench/README.md)) |
| Czech | [`ces_Latn`](ces_Latn/README.md) | 1 | Global-MMLU only ([`global_mmlu`](../global_mmlu/README.md); see below) |
| French | [`fra_Latn`](fra_Latn/README.md) | 9 | FrenchBench ([`french_bench`](../french_bench/README.md)) |
| Galician | [`glg_Latn`](glg_Latn/README.md) | 10 | GalicianBench ([`galician_bench`](../galician_bench/README.md)) |
| Spanish | [`spa_Latn`](spa_Latn/README.md) | 16 | SpanishBench ([`spanish_bench`](../spanish_bench/README.md)) |
| Ukrainian | [`ukr_Cyrl`](ukr_Cyrl/README.md) | 7 | UkrainianBench, created as part of HPLT-e |

The other HPLT-e languages are not part of this folder:

* **Norwegian** (Bokmål and Nynorsk): HPLT-e uses [NorEval](https://aclanthology.org/2025.findings-acl.181/) ([v1.0 release](https://github.com/ltgoslo/noreval/releases/tag/v1.0)), which is already in LM Evaluation Harness with five prompts per task: [`noreval`](../noreval/README.md).
* **Czech**: HPLT-e uses [BenCzechMark](https://github.com/DCGM/lm-evaluation-harness/tree/main/lm_eval/tasks/benczechmark), which is not yet in LM Evaluation Harness. The version used in HPLT-e can be found in our [HPLT-e repository](https://github.com/hplt-project/hplt-e/tree/main/ces_Latn). We updated it for recent LM Evaluation Harness releases. Only the Global-MMLU prompts written for HPLT-e are included here.
* **Finnish**: HPLT-e uses [FinBench v2](https://github.com/LumiOpen/lm-evaluation-harness/tree/finbench_v2/lm_eval/tasks/finbench_v2) ([paper](https://arxiv.org/abs/2512.13330)), which is not yet in LM Evaluation Harness.

### Task names

Tasks are named `hplte_<language>_<task>_p<N>`, where `<language>` is the ISO 639-3 code (`cat`, `ces`, `eus`, `fra`, `glg`, `spa`, `ukr`) and `p<N>` is the prompt (`p0`–`p2`). Each prompt is treated as a separate task. The language READMEs map every task to its name in the HPLT-e repository and paper.

To evaluate all prompts of a task, use its tag, which is the task name without the prompt suffix (`hplte_<language>_<task>`):

```bash
lm-eval run \
  --model hf \
  --model_args pretrained=<model> \
  --tasks hplte_spa_xnli \
  --num_fewshot 0 \
  --batch_size auto
```

Global-MMLU and EusExams have a group per prompt instead of a tag (see below); use a wildcard such as `--tasks "hplte_spa_global_mmlu_p*"` to run all of their prompts.

### Aggregation

There are no HPLT-e-wide or per-language groups. HPLT-e does not average over prompts: it takes the maximum score across prompts, rescales it against a random baseline (min-max normalisation), averages within task categories, and then averages the category scores into a language score. These steps cannot be expressed as LM Evaluation Harness group aggregations. Compute them from the per-prompt results with the code in the [HPLT-e repository](https://github.com/hplt-project/hplt-e#-multilingual-evaluation-recipe).

Groups are used only where a single prompt is evaluated over several subsets: Global-MMLU (`hplte_<language>_global_mmlu_p<N>`, six subject categories) and EusExams (`hplte_eus_exams_p<N>`, 29 exams).

### HPLT-e task selection

Based on the evaluation of 2.15B-parameter models trained on 100B tokens, HPLT-e recommends the HPLT-e tasks that are representative for multilingual evaluation based on our 100BT model evaluation results (each with all of its prompts):

| Language | Tasks |
|:---|:---|
| Catalan | `hplte_cat_arc_easy`, `hplte_cat_copa`, `hplte_cat_coqcat`, `hplte_cat_piqa`, `hplte_cat_siqa`, `hplte_cat_teca`, `hplte_cat_xstorycloze` |
| French | `hplte_fra_fquad`, `hplte_fra_vocabulary` |
| Spanish | `hplte_spa_copa`, `hplte_spa_openbookqa`, `hplte_spa_xnli`, `hplte_spa_xstorycloze` |
| Ukrainian | `hplte_ukr_ua_squad`, `hplte_ukr_zno` |

These tasks are marked with 😎 in the language READMEs. See the [HPLT-e repository](https://github.com/hplt-project/hplt-e#-hplt-e-tasks) and our paper for the selection criteria and other details.

### Datasets

Every dataset is pinned with `dataset_kwargs.revision` to the data used for the HPLT-e evaluations: the last commit before 2025-11-18, when the [HPLT-e results](https://huggingface.co/datasets/HPLT/2508-datasets-evals) were published. FLORES and MGSM-es use a later commit with identical data, because the earlier one only provides a loading script, which `datasets` v4 and later cannot run.

Two datasets are gated, and you need to accept their conditions on the Hub before use: [`facebook/flores`](https://huggingface.co/datasets/facebook/flores) (FLORES tasks in Basque, Catalan, Galician and Spanish) and [`gplsi/cocoteros_va`](https://huggingface.co/datasets/gplsi/cocoteros_va) (`hplte_cat_cocoteros_va`).

### Differences from the HPLT-e repository

The prompts, few-shot settings and metrics are the ones used for the published HPLT-e results. The following changes were made for LM Evaluation Harness:

* Task names have the `hplte_<language>_` prefix. The language READMEs and the `task_mapping.json` file in each language folder map every task to its name in the HPLT-e repository.
* Each task family is a single folder with its own `utils.py`.
* EusExams has separate subtasks for each prompt (`hplte_eus_exams_<exam>_p<N>`), instead of three groups that share subtask names.
* The SQuAD-style metrics (`compute_exact`, `compute_f1`) are included in `utils.py` instead of being imported from `transformers.data.metrics`.

### Verification

Every task was run with the configurations from the HPLT-e repository (LM Evaluation Harness v0.4.9.1, at the same dataset revisions) and compared with the task here: the evaluation documents, all zero-shot requests and the metric outputs are identical. The prompts also match the ones stored with the [published results](https://huggingface.co/datasets/HPLT/2508-datasets-evals).

To compare a task with its counterpart in the HPLT-e repository (the names are listed in each language folder's `task_mapping.json`):

```python
from lm_eval.tasks import TaskManager

def requests(task_name, include_path=None):
    task = TaskManager(include_path=include_path).load(task_name)["tasks"][task_name]
    task.build_all_requests(limit=None, rank=0, world_size=1)
    return [instance.arguments for instance in task.instances]

assert requests("hplte_spa_xnli_p0") == requests("xnli_es_p0", include_path="hplt-e/spa_Latn")
```

### Citation

```text
@inproceedings{oepen-etal-2026-hplt,
    title = "{HPLT} 3.0: Very Large-Scale Multilingual Resources for {LLM}s and {MT}. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models",
    author = "Oepen, Stephan  and
      Arefyev, Nikolay  and
      Aulamo, Mikko  and
      Ba{\~n}{\'o}n, Marta  and
      Buljan, Maja  and
      Burchell, Laurie V.  and
      Charpentier, Lucas Georges Gabriel  and
      Chen, Pinzhen  and
      Fedorova, Mariia  and
      de Gibert, Ona  and
      Haddow, Barry  and
      Haji{\v{c}}, Jan  and
      Helcl, Jindrich  and
      Kutuzov, Andrey  and
      Laippala, Veronika  and
      Li, Zihao  and
      Malik, Bhavitvya  and
      Mikhailov, Vladislav  and
      Myntti, Amanda  and
      O{'}Brien, Dayy{\'a}n  and
      Polakova, Lucie  and
      Ram{\'i}rez-S{\'a}nchez, Gema  and
      Siewert, Janine  and
      Stepachev, Pavel  and
      Tiedemann, Joerg  and
      Vahtola, Teemu  and
      Varis, Dusan  and
      Vitiugin, Fedor  and
      Zaragoza, Jaume",
    editor = "Piperidis, Stelios  and
      Bel, N{\'u}ria  and
      van den Heuvel, Henk  and
      Ide, Nancy  and
      Krek, Simon  and
      Toral, Antonio",
    booktitle = "Proceedings of the Fifteenth Language Resources and Evaluation Conference",
    month = may,
    year = "2026",
    address = "Palma de Mallorca, Spain",
    publisher = "ELRA Language Resource Association",
    url = "https://aclanthology.org/2026.lrec-1.110/",
    doi = "10.63317/25xbdofco9od",
    pages = "1409--1434",
    abstract = "We present an ongoing initiative to provide open, very large, high-quality, and richly annotated textual datasets for almost 200 languages. At 30 trillion tokens, this is likely the largest generally available multilingual collection of LLM pre-training data. These datasets are derived from web crawls from different sources and accompanied with a complete, open-source pipeline for document selection from web archives, text extraction from HTML, language identification for noisy texts, exact and near-deduplication, annotation with, among others, register labels, text quality estimates, and personally identifiable information; and final selection and filtering. We report on data quality probes through contrastive and analytical statistics, through manual inspection of samples for some 20 languages, and through end-to-end evaluation of various language model architectures trained on this data. For multilingual LLM evaluation, we provide a comprehensive collection of benchmarks for nine European languages, with special emphasis on natively created tasks, mechanisms to mitigate prompt sensitivity, and refined normalization and aggregation of scores. Additionally, we train and evaluate a family of 57 monolingual encoder{--}decoder models, as well as about 30 ``smallish'' monolingual GPT-like reference models. Besides the monolingual data and models, we also present a very large collection of parallel texts automatically mined from this data, together with a novel parallel corpus synthesized via machine translation."
}
```

Please also cite the original benchmarks and datasets (the BibTeX entries for the benchmarks in this folder are in the language READMEs):

* [IberoBench](https://aclanthology.org/2025.coling-main.699/) (Baucells et al., COLING 2025): CatalanBench, SpanishBench, GalicianBench and BasqueBench;
* [CroissantLLM](https://openreview.net/forum?id=uA19Xo1o31) (Faysse et al., TMLR 2025): FrenchBench;
* [Belebele](https://aclanthology.org/2024.acl-long.44/) (Bandarkar et al., ACL 2024): all languages in this folder except Czech;
* [Global-MMLU](https://aclanthology.org/2025.acl-long.919/) (Singh et al., ACL 2025): Czech, French, Spanish and Ukrainian;
* [INCLUDE](https://openreview.net/forum?id=k3gCieTXeY) (Romanou et al., ICLR 2025): Basque, French, Spanish and Ukrainian;
* [BenCzechMark](https://aclanthology.org/2025.tacl-1.50/) (Fajcik et al., TACL 2025): the Czech part of HPLT-e;
* [NorEval](https://aclanthology.org/2025.findings-acl.181/) (Mikhailov et al., Findings of ACL 2025): the Norwegian part of HPLT-e;
* [FIN-bench-v2](https://arxiv.org/abs/2512.13330) (Kytöniemi et al., 2025): the Finnish part of HPLT-e;
* the papers or dataset cards of the individual datasets, linked in the language READMEs (for Ukrainian: ZNO, UA-SQuAD, TextDetox and WMT24++).

### Groups, Tags, and Tasks

See the language READMEs: [Basque](eus_Latn/README.md), [Catalan](cat_Latn/README.md), [Czech](ces_Latn/README.md), [Finnish](https://github.com/LumiOpen/lm-evaluation-harness/blob/finbench_v2/lm_eval/tasks/finbench_v2/README.md), [French](fra_Latn/README.md), [Galician](glg_Latn/README.md), [Norwegian](../noreval/README.md), [Spanish](spa_Latn/README.md), [Ukrainian](ukr_Cyrl/README.md).

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial release.
