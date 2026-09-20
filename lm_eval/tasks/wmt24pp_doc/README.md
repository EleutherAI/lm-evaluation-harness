# Document-Level WMT24++ Translation Tasks

This directory provides YAML-based tasks for evaluating document-level machine translation on the **WMT24++** benchmark hosted on the Hugging Face Hub as [`google/wmt24pp`](https://huggingface.co/datasets/google/wmt24pp).

It supports:

* **English → X** tasks using the official WMT24++ language-pair configs
* **X → English** tasks derived by swapping the aligned `en->X` pairs
* group tasks for:
  * **all English → X**
  * **all X → English**
  * **all bidirectional tasks**

Each language pair is exposed as a separate task, using consistent prompt formatting and WMT-style metrics.

This task family is adapted from [https://github.com/EleutherAI/lm-evaluation-harness/pull/3480](https://github.com/EleutherAI/lm-evaluation-harness/pull/3480)

## Dataset

* **HF ID**: `google/wmt24pp`
* **Configs**: one per official language pair (e.g. `en-de_DE`, `en-pl_PL`, `en-pt_BR`, ...)
* **Split**: a single split (`train`), used here as the evaluation split
* **Fields (per segment-level example)**:
  * `lp`: language pair, e.g. `"en-de_DE"`
  * `domain`: text domain (`canary`, `news`, `social`, `speech`, `literary`)
  * `document_id`: document identifier
  * `segment_id`: global segment identifier
  * `is_bad_source`: boolean flag for low-quality sources
  * `source`: English source sentence
  * `target`: post-edit of `original_target` (recommended reference)
  * `original_target`: original reference translation

In this task family, we:
* **drop all examples with `is_bad_source == True`**
* **use all domains** (no filtering on `domain`)
* **evaluate at document level** by reconstructing each document from its segments using `document_id` and `segment_id`

For **English → X**, the task uses the official WMT24++ direction directly.

For **X → English**, the task is **derived** from the aligned `en->X` config by swapping:
* the translated `target` text into the source side
* the original English `source` text into the target side

## Task layout

The task directory is organized as follows:

```text
lm_eval/tasks/wmt24pp_doc/
├── utils.py
├── wmt24pp_doc_common.yaml
├── generate_tasks.py
├── groups/
│   ├── wmt24pp_doc_en-all.yaml
│   ├── wmt24pp_doc_all-en.yaml
│   └── wmt24pp_doc_all.yaml
├── en_to_x/
│   ├── wmt24pp_doc_en-ar_EG.yaml
│   ├── wmt24pp_doc_en-ca_ES.yaml
│   ├── wmt24pp_doc_en-de_DE.yaml
│   └── ...
└── x_to_en/
    ├── wmt24pp_doc_ar_EG-en.yaml
    ├── wmt24pp_doc_ca_ES-en.yaml
    ├── wmt24pp_doc_de_DE-en.yaml
    └── ...
```

The files are generated automatically by `generate_tasks.py`, so there is no need to hand-maintain one YAML per language pair.

## Tasks

Common configuration is defined in `wmt24pp_common.yaml`:

* `dataset_path: google/wmt24pp`
* `test_split: train`
* `output_type: generate_until`
* `doc_to_text: !function utils.doc_to_text`
* `doc_to_target: !function utils.doc_to_target`
* `custom_dataset: !function utils.load_wmt24pp_dataset`

The common config also sets document-level generation defaults and WMT-style metrics.

Each language pair has its own thin YAML that includes the common config.

Example: **English → German**

```yaml
include: ../wmt24pp_doc_common.yaml

task: wmt24pp_doc_en-de_DE

tag:
  - translation
  - wmt24pp_doc

metadata:
  version: 1.0
  src_lang: "en"
  tgt_lang: "de_DE"
```

Example: **German → English**

```yaml
include: ../wmt24pp_doc_common.yaml

task: wmt24pp_doc_de_DE-en

tag:
  - translation
  - wmt24pp_doc
  - reverse_derived

metadata:
  version: 1.0
  src_lang: "de_DE"
  tgt_lang: "en"
```

The `src_lang` and `tgt_lang` values in `metadata` are passed to `utils.load_wmt24pp_dataset`, which loads the corresponding WMT24++ config, filters out bad sources, and reconstructs document-level examples.

## Supported task families

This task family supports three main entry points:

* **specific pair**

  * `wmt24pp_doc_en-de_DE`
  * `wmt24pp_doc_de_DE-en`

* **all English → X tasks**

  * `wmt24pp_doc_en-all`

* **all X → English tasks**

  * `wmt24pp_doc_all-en`

* **all bidirectional tasks**

  * `wmt24pp_doc_all`

## Running the tasks

Run all **English → X** tasks:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt24pp_doc_en-all
```

Run all **X → English** tasks:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt24pp_doc_all-en
```

Run **all bidirectional** tasks:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt24pp_doc_all
```

Run a specific subset explicitly:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt24pp_doc_en-de_DE,wmt24pp_doc_pl_PL-en
```

You can also provide a chat template if needed:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt24pp_doc_en-de_DE \
  --apply_chat_template
```

## Metrics

We follow the same general metric setup as the other WMT translation tasks in this repository.

This task family exposes:

* **BLEU** (`bleu`) – via SacreBLEU
* **ChrF** (`chrf`) – character n-gram F-score

All metrics are implemented via `lm_eval.api.metrics` and use SacreBLEU under the hood.

## Document statistics

At the document level, the reconstructed task family has the following overall shape:

| Dataset | Pairs | Files | Docs/file | Sents/file | Words/file | Avg sents/doc | Avg words/doc | Avg words/sent |
| ------- | ----: | ----: | --------: | ---------: | ---------: | ------------: | ------------: | -------------: |
| wmt24pp |    55 |   110 |    170.01 |     960.24 |   30364.10 |          5.65 |        178.60 |          31.62 |

Here, “Files” counts both directions:

* 55 official `en->X` files
* 55 derived `X->en` files

## Task Validity Checklist

For adding novel benchmarks/datasets to the library:

* [x] **Is the task an existing benchmark in the literature?**
  Yes. WMT24++ extends the official WMT24 benchmark to 55 languages and dialects as described by Deutsch et al. (2025).

* [x] **Have you referenced the original paper that introduced the task?**
  The citation for the WMT24++ paper is provided below.

* [ ] **If yes, does the original paper provide a reference implementation?**
  Prompt template and dataset filtering match the reference release, but this task family does not attempt to exactly replicate every detail of the original evaluation pipeline.

If other tasks on this dataset are already supported:

* [x] **Is the "Main" variant of this task clearly denoted?**
  Yes. Task names explicitly indicate the source and target directions, and this README makes clear that evaluation is document-level.

* [x] **Have you provided a short sentence on what each new variant adds / evaluates?**
  Yes. This README explains the distinction between official `en->X` tasks and derived `X->en` tasks.

* [x] **Have you noted which published evaluation setups are matched by this variant?**
  Yes. The task uses the WMT24++ dataset split (`train`), applies bad-source filtering, uses the recommended post-edited `target` references for `en->X`, and evaluates using standard MT metrics.

## Citation

Please cite the original WMT24++ paper and the lm-evaluation-harness project as appropriate when using these tasks in publications.

```bibtex
@misc{deutsch2025wmt24expandinglanguagecoverage,
      title={{WMT24++: Expanding the Language Coverage of WMT24 to 55 Languages & Dialects}},
      author={Daniel Deutsch and Eleftheria Briakou and Isaac Caswell and Mara Finkelstein and Rebecca Galor and Juraj Juraska and Geza Kovacs and Alison Lui and Ricardo Rei and Jason Riesa and Shruti Rijhwani and Parker Riley and Elizabeth Salesky and Firas Trabelsi and Stephanie Winkler and Biao Zhang and Markus Freitag},
      year={2025},
      eprint={2502.12404},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12404},
}
```
