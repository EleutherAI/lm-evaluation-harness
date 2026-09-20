# Document-Level NTREX Translation Tasks

This directory provides YAML-based tasks for evaluating document-level machine translation on **NTREX**.

The task family is built from the official NTREX GitHub repository:

* `NTREX-128/newstest2019-src.eng.txt`
* `NTREX-128/newstest2019-ref.*.txt`
* `NTREX-additional/newstest2019-ref.*.txt`
* `DOCUMENT_IDS.tsv`
* `LANGUAGES.tsv`

NTREX provides English source text and human reference translations for many target languages, together with line-aligned document IDs. This makes it suitable for reconstructing document-level examples from sentence-aligned files.

This task family supports:

* **English → X** tasks using the official NTREX direction
* **X → English** tasks derived by swapping the aligned NTREX files
* group tasks for:
  * **all English → X**
  * **all X → English**
  * **all bidirectional tasks**

Each language pair is exposed as a separate task, using consistent prompt formatting and WMT-style metrics.

## Dataset

* **Source repo**: `MicrosoftTranslator/NTREX`
* **Core files**:
  * `NTREX-128/newstest2019-src.eng.txt`
  * `NTREX-128/newstest2019-ref.*.txt`
  * `NTREX-additional/newstest2019-ref.*.txt`
  * `DOCUMENT_IDS.tsv`
  * `LANGUAGES.tsv`
* **Evaluation split**: `newstest2019`

NTREX is described as “News Test References for MT Evaluation from English into a total of 128 target languages with document-level information.”

In this task family, we:

* reconstruct **document-level** examples by grouping lines with the same `DOCUMENT_IDS.tsv` entry
* expose:
  * **official** `eng -> X` tasks
  * **derived** `X -> eng` tasks by swapping aligned source/reference files

So the reverse-direction tasks are convenient and aligned, but they are **not official direct NTREX directions**.

## Task layout

The task directory is organized as follows:

```text
lm_eval/tasks/ntrex/
├── utils.py
├── ntrex_doc_common.yaml
├── generate_tasks.py
├── groups/
│   ├── ntrex_doc_en_all.yaml
│   ├── ntrex_doc_all_en.yaml
│   └── ntrex_doc_bidirectional_all.yaml
├── en_to_x/
│   ├── ntrex_doc_eng_deu.yaml
│   ├── ntrex_doc_eng_fra.yaml
│   ├── ntrex_doc_eng_spa.yaml
│   └── ...
└── x_to_en/
    ├── ntrex_doc_deu_eng.yaml
    ├── ntrex_doc_fra_eng.yaml
    ├── ntrex_doc_spa_eng.yaml
    └── ...
```

The thin task YAMLs are generated automatically by `generate_tasks.py`.

## Local data setup

These tasks expect a local NTREX checkout, unless your `utils.py` is configured to clone it automatically.

The simplest setup is:

```bash
git clone https://github.com/MicrosoftTranslator/NTREX.git
export NTREX_PATH=$PWD/NTREX
```

If `NTREX_PATH` is not set, the loader will look for `./NTREX` relative to the current working directory, or use its configured cache location if automatic cloning is enabled.

## Tasks

Common configuration is defined in `ntrex_doc_common.yaml`:

* `output_type: generate_until`
* `doc_to_text: !function utils.doc_to_text`
* `doc_to_target: !function utils.doc_to_target`
* `custom_dataset: !function utils.load_ntrex_dataset`
* `test_split: test`

Each pair-specific YAML includes the common config and passes:

* `src_lang`
* `tgt_lang`

through task metadata.

Example: **English → German**

```yaml
include: ../ntrex_doc_common.yaml

task: ntrex-doc-eng-deu

tag:
  - translation
  - ntrex_doc

metadata:
  version: 1.0
  src_lang: "eng"
  tgt_lang: "deu"
```

Example: **German → English**

```yaml
include: ../ntrex_doc_common.yaml

task: ntrex-doc-deu-eng

tag:
  - translation
  - ntrex_doc
  - reverse_derived

metadata:
  version: 1.0
  src_lang: "deu"
  tgt_lang: "eng"
```

## Supported task families

This task family supports:

* **specific pair**

  * `ntrex-doc-eng-deu`
  * `ntrex-doc-deu-eng`

* **all English → X tasks**

  * `ntrex-doc-en-all`

* **all X → English tasks**

  * `ntrex-doc-all-en`

* **all bidirectional tasks**

  * `ntrex-doc-bidirectional-all`

## Running the tasks

Run all **English → X** tasks:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks ntrex-doc-en-all
```

Run all **X → English** tasks:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks ntrex-doc-all-en
```

Run all **bidirectional** tasks:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks ntrex-doc-bidirectional-all
```

Run a specific subset explicitly:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks ntrex-doc-eng-deu,ntrex-doc-spa-eng
```

You can also provide a chat template if needed:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks ntrex-doc-eng-deu \
  --apply_chat_template
```

## Metrics

This task family exposes:

* **BLEU** (`bleu`) – via SacreBLEU
* **ChrF** (`chrf`) – character n-gram F-score

All metrics are implemented via `lm_eval.api.metrics` and use SacreBLEU under the hood.

## Task Validity Checklist

For adding novel benchmarks/datasets to the library:

* [x] **Is the task an existing benchmark in the literature?**
  Yes. NTREX is an existing benchmark/resource in the literature, however, here we implement document-level evaluation.

* [x] **Have you referenced the original paper that introduced the task?**
  Yes. The citation for NTREX is provided below.

* [ ] **If yes, does the original paper provide a reference implementation?**
  This implementation is aligned with the public NTREX file release, but it does not attempt to reproduce every detail of any original evaluation pipeline. It adds document-level reconstruction and the prompt from Boquet.

If other tasks on this dataset are already supported:

* [x] **Is the "Main" variant of this task clearly denoted?**
  Yes. Task names explicitly indicate the source and target directions, and this README makes clear that evaluation is document-level.

* [x] **Have you provided a short sentence on what each new variant adds / evaluates?**
  Yes. This README explains that the task family reconstructs document-level examples using `DOCUMENT_IDS.tsv` and adds derived reverse-direction variants.

* [x] **Have you noted which published evaluation setups are matched by this variant?**
  Yes. The task uses the public NTREX files, `newstest2019`, line-aligned document IDs, and standard MT metrics.

## Citation

Please cite the original NTREX paper and the lm-evaluation-harness project as appropriate when using these tasks in publications.

```bibtex
@inproceedings{federmann-etal-2022-ntrex,
    title = "{NTREX}-128 -- News Test References for {MT} Evaluation of 128 Languages",
    author = "Federmann, Christian and Kocmi, Tom and Xin, Ying",
    booktitle = "Proceedings of the First Workshop on Scaling Up Multilingual Evaluation",
    year = "2022",
    pages = "21--24",
    url = "https://aclanthology.org/2022.sumeval-1.4",
}
```
