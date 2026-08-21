# Document-Level WMT GeneralMT Testset Translation Tasks

This directory provides YAML-based tasks for evaluating document-level machine translation on the **WMT GeneralMT test sets**.

The task family is built automatically from locally downloaded WMT XML files. During setup, the task code can:

* download the official WMT GeneralMT archive
* extract the XML files
* convert them into document-level plain-text files
* generate task YAMLs from the discovered testsets and language pairs

It supports any source-target direction present in the processed WMT test files.

## Dataset

The raw data comes from the official WMT GeneralMT archive:

* `https://data.statmt.org/wmt24/general-mt/wmt24_GeneralMT-devsets.zip`

The task loader processes XML files into document-level `.SRC` / `.TGT` files.

Expected processed layout:

```text id="d0x3z7"
data/
  raw/
    wmt_generalmt/
      xml/
        ...
      processed/
        newstest2010.en-de.SRC
        newstest2010.en-de.TGT
        wmttest2022.cs-en.SRC
        wmttest2022.cs-en.TGT
        ...
```

The processed files are expected to use:

* one sentence per line
* blank lines between documents

## Scope

This task family includes:

* **WMT testsets only**
* multilingual source-target directions
* document-level reconstruction from XML

This task family intentionally:

* **groups by language pair**
* **groups by year**, where `newstestYYYY` and `wmttestYYYY` are treated as the same year
* provides special groups for:
  * all tasks
  * all `en-xx`
  * all `xx-en`

This task family intentionally **ignores**:

* **FLORES** testsets such as `florestest2021`
* **WMT24** testsets (for WMT24 document-level evaluation, use **`wmt24pp_doc`** instead)

## Task layout

The task directory is organized as follows:

```text id="9p5mxm"
lm_eval/tasks/wmt_generalmt_doc/
├── utils.py
├── wmt_generalmt_doc_common.yaml
├── generate_tasks.py
├── groups/
│   ├── wmt_generalmt_doc_all.yaml
│   ├── wmt_generalmt_doc_en_all.yaml
│   ├── wmt_generalmt_doc_all_en.yaml
│   ├── wmt_generalmt_doc_2010.yaml
│   ├── wmt_generalmt_doc_2011.yaml
│   ├── ...
│   ├── wmt_generalmt_doc_en_de.yaml
│   ├── wmt_generalmt_doc_cs_en.yaml
│   └── ...
└── tasks/
    ├── wmt_generalmt_doc_newstest2010_en_de.yaml
    ├── wmt_generalmt_doc_newstest2011_en_de.yaml
    ├── wmt_generalmt_doc_wmttest2022_cs_en.yaml
    └── ...
```

The thin task YAMLs are generated automatically by `generate_tasks.py`.

## Tasks

Common configuration is defined in `wmt_generalmt_doc_common.yaml`:

* `training_split: null`
* `validation_split: null`
* `test_split: test`
* `output_type: generate_until`
* `doc_to_text: !function utils.doc_to_text`
* `doc_to_target: !function utils.doc_to_target`
* `custom_dataset: !function utils.load_wmt_generalmt_dataset`

Each pair-specific YAML includes the common config and passes:

* `testset`
* `src_lang`
* `tgt_lang`

through task metadata.

Example:

```yaml id="z1wq7g"
include: ../wmt_generalmt_doc_common.yaml

task: wmt-generalmt-doc-newstest2010-en-de

tag:
  - translation
  - wmt_generalmt_doc

metadata:
  version: 1.0
  testset: "newstest2010"
  src_lang: "en"
  tgt_lang: "de"
```

## How tasks are created

Tasks are created from the processed file inventory.

For every matching pair of files:

* `<testset>.<src>-<tgt>.SRC`
* `<testset>.<src>-<tgt>.TGT`

the generator creates one task:

* `wmt-generalmt-doc-<testset>-<src>-<tgt>`

Examples:

* `newstest2010.en-de` → `wmt-generalmt-doc-newstest2010-en-de`
* `wmttest2022.cs-en` → `wmt-generalmt-doc-wmttest2022-cs-en`

So each task corresponds to one exact combination of:

* testset
* source language
* target language

## How groups are created

Groups are created in three ways.

### 1. Group by language pair

All tasks sharing the same `src-tgt` direction are grouped together.

Example:

* `wmt-generalmt-doc-newstest2010-en-de`
* `wmt-generalmt-doc-newstest2011-en-de`
* `wmt-generalmt-doc-wmttest2022-en-de`

go into:

* `wmt-generalmt-doc-en-de`

This lets you evaluate one language pair across all available years/testsets.

### 2. Group by year

All tasks whose testset ends in the same year are grouped together.

Examples:

* `newstest2019`
* `wmttest2019`

are both grouped under:

* `wmt-generalmt-doc-2019`

This treats `newstest` and `wmttest` equally for year-based evaluation.

### 3. Global groups

Three special global groups are created:

* `wmt-generalmt-doc-all`
* `wmt-generalmt-doc-en-all`
* `wmt-generalmt-doc-all-en`

These mean:

* all discovered tasks
* all tasks where the source is English
* all tasks where the target is English

## Running the tasks

Generate the task YAMLs after downloading and processing the data:

```bash id="fa3id2"
python lm_eval/tasks/wmt_generalmt_doc/generate_tasks.py
```

Run all discovered tasks:

```bash id="0v7k44"
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt-generalmt-doc-all
```

Run all `en -> xx` tasks:

```bash id="96g6hr"
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt-generalmt-doc-en-all
```

Run all `xx -> en` tasks:

```bash id="v22r5r"
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt-generalmt-doc-all-en
```

Run all tasks for a given year:

```bash id="qjlwm1"
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt-generalmt-doc-2022
```

Run all tasks for one language pair across years:

```bash id="yjlwm2"
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt-generalmt-doc-en-de
```

Run one exact task:

```bash id="ljlwm3"
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt-generalmt-doc-newstest2010-en-de
```

You can also provide a chat template if needed:

```bash id="jlwm45"
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt-generalmt-doc-newstest2010-en-de \
  --apply_chat_template
```

## Metrics

This task family exposes:

* **BLEU** (`bleu`) – via SacreBLEU
* **ChrF** (`chrf`) – character n-gram F-score

TER is intentionally **not** included.

All metrics are implemented via `lm_eval.api.metrics` and use SacreBLEU under the hood.

## Task Validity Checklist

For adding novel benchmarks/datasets to the library:

* [x] **Is the task an existing benchmark in the literature?**
  Yes. This task family is based on public WMT GeneralMT test sets.

* [x] **Have you referenced the original source introducing the task/data?**
  Yes. The source archive and WMT testset naming are documented above.

* [ ] **If yes, does the original source provide a reference implementation?**
  This implementation is aligned with the public XML release, but it does not attempt to reproduce every detail of any original evaluation pipeline. It adds document-level reconstruction inside `lm-evaluation-harness`. It adds document-level reconstruction and the prompt from Bouquet.

If other tasks on this dataset are already supported:

* [x] **Is the "Main" variant of this task clearly denoted?**
  Yes. Task names explicitly indicate the testset, source language, and target language, and this README makes clear that evaluation is document-level.

* [x] **Have you provided a short sentence on what each new variant adds / evaluates?**
  Yes. This README explains that the task family reconstructs document-level examples from XML and groups them by language pair and by year.

* [x] **Have you noted which published evaluation setups are matched by this variant?**
  Yes. The task uses the public WMT GeneralMT XML files, excludes FLORES and WMT24 from this family, and evaluates with BLEU and ChrF.

## Citation

Please cite the relevant WMT source materials and the lm-evaluation-harness project as appropriate when using these tasks in publications.
