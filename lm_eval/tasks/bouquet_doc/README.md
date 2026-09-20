# Document-Level BOUQuET Translation Tasks

This directory provides YAML-based tasks for evaluating **document-level** machine translation on the **BOUQuET** benchmark hosted on the Hugging Face Hub as [`facebook/bouquet`](https://huggingface.co/datasets/facebook/bouquet).

Although BOUQuET is a **multi-way, multi-centric** benchmark, this task family intentionally exposes an **English-centric** view for simplicity and consistency inside `lm-evaluation-harness`:

- **English → X** tasks
- **X → English** tasks
- group tasks for:
  - **all English → X**
  - **all X → English**
  - **all bidirectional tasks**

Each language pair is exposed as a separate task, using consistent prompt formatting and WMT-style metrics.

## Dataset

- **HF ID**: `facebook/bouquet`
- **Splits**: `test`
- **Underlying structure**: sentence-level rows grouped into paragraph/document-like units
- **Relevant fields used here**:
  - `level`
  - `src_lang`
  - `tgt_lang`
  - `src_text`
  - `tgt_text`
  - `par_id`
  - `uniq_id`
  - optionally `sent_id`

BOUQuET is a multi-way, multi-centric, and multi-register/domain benchmark. It is explicitly designed to go beyond English-centric evaluation and is organized in paragraphs rather than isolated sentences.

In this task family, we:

- use only rows with `level == "sentence_level"`
- reconstruct paragraph/document-level examples by grouping rows by `par_id`
- order sentences within each paragraph using:
  - `sent_id` if present
  - otherwise a best-effort fallback from `uniq_id`
- expose only **English-centric** directions:
  - `eng_Latn -> X`
  - `X -> eng_Latn`

This means the task family is a **restricted English-centric projection** of BOUQuET, not a full representation of the benchmark.

## Task layout

The task directory is organized as follows:

```text
lm_eval/tasks/bouquet/
├── utils.py
├── bouquet_doc_common.yaml
├── bouquet_doc_generic.yaml
├── generate_tasks.py
├── run_bouquet_doc.py
├── groups/
│   ├── bouquet_doc_en_all.yaml
│   ├── bouquet_doc_all_en.yaml
│   └── bouquet_doc_bidirectional_all.yaml
├── en_to_x/
│   ├── bouquet_doc_en_fra_Latn.yaml
│   ├── bouquet_doc_en_deu_Latn.yaml
│   ├── bouquet_doc_en_spa_Latn.yaml
│   └── ...
└── x_to_en/
    ├── bouquet_doc_fra_Latn_en.yaml
    ├── bouquet_doc_deu_Latn_en.yaml
    ├── bouquet_doc_spa_Latn_en.yaml
    └── ...
````

The thin task YAMLs are generated automatically by `generate_tasks.py`.

## Tasks

Common configuration is defined in `bouquet_doc_common.yaml`:

* `dataset_path: facebook/bouquet`
* `test_split: test`
* `validation_split: null`
* `output_type: generate_until`
* `doc_to_text: !function utils.doc_to_text`
* `doc_to_target: !function utils.doc_to_target`
* `custom_dataset: !function utils.load_bouquet_dataset`

Each pair-specific YAML includes the common config and passes:

* `src_lang`
* `tgt_lang`

through task metadata.

Example: **English → French**

```yaml
include: ../bouquet_doc_common.yaml

task: bouquet-doc-en-fra_Latn

tag:
  - translation
  - bouquet_doc

metadata:
  version: 1.0
  src_lang: "eng_Latn"
  tgt_lang: "fra_Latn"
```

Example: **French → English**

```yaml
include: ../bouquet_doc_common.yaml

task: bouquet-doc-fra_Latn-en

tag:
  - translation
  - bouquet_doc

metadata:
  version: 1.0
  src_lang: "fra_Latn"
  tgt_lang: "eng_Latn"
```

## Supported task families

This task family supports:

* **specific pair**

  * `bouquet-doc-en-fra_Latn`
  * `bouquet-doc-fra_Latn-en`

* **all English → X tasks**

  * `bouquet-doc-en-all`

* **all X → English tasks**

  * `bouquet-doc-all-en`

* **all bidirectional tasks**

  * `bouquet-doc-bidirectional-all`

## Running the tasks

Run all **English → X** tasks:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks bouquet-doc-en-all
```

Run all **X → English** tasks:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks bouquet-doc-all-en
```

Run all **bidirectional** tasks:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks bouquet-doc-bidirectional-all
```

Run a specific subset explicitly:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks bouquet-doc-en-fra_Latn,bouquet-doc-deu_Latn-en
```

You can also run any non-English language pair with the convenience wrapper `run_bouquet_doc.py`. This uses the generic BOUQuET task and passes the pair through task metadata at runtime.

Example:

```bash
python run_bouquet_doc.py hin_Deva-spa_Latn \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
```

The wrapper expects the pair in **official dataset language-code format**, for example:

* `spa_Latn-fra_Latn`
* `deu_Latn-por_Latn`
* `hin_Deva-ukr_Cyrl`

You can also provide a chat template if needed:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks bouquet-doc-en-fra_Latn \
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
  Yes. BOUQuET is an existing benchmark introduced in the literature.

* [x] **Have you referenced the original paper that introduced the task?**
  Yes. Citations for the BOUQuET paper and the Omnilingual MT paper are provided below.

* [ ] **If yes, does the original paper provide a reference implementation?**
  This implementation is aligned with the public dataset release, but it does not attempt to exactly reproduce every detail of the original evaluation pipeline. In particular, this task family implements **document-level evaluation** inside `lm-evaluation-harness`, which is a task-specific adaptation.

If other tasks on this dataset are already supported:

* [x] **Is the "Main" variant of this task clearly denoted?**
  Yes. Task names explicitly indicate source and target direction, and this README makes clear that the implementation is document-level.

* [x] **Have you provided a short sentence on what each new variant adds / evaluates?**
  Yes. This README explains that the task family reconstructs paragraph/document-level examples from sentence-level rows and restricts evaluation to English-centric directions.

* [x] **Have you noted which published evaluation setups are matched by this variant?**
  Yes. This task family uses the public BOUQuET dataset splits and fields, but adapts them to document-level evaluation within `lm-evaluation-harness`.

## Citation

Please cite the original BOUQuET paper and the lm-evaluation-harness project as appropriate when using these tasks in publications.

```bibtex
@inproceedings{andrews-etal-2025-bouquet,
    title = "{BOUQ}u{ET} : dataset, Benchmark and Open initiative for Universal Quality Evaluation in Translation",
    author = "Andrews, Pierre and Artetxe, Mikel and Meglioli, Mariano Coria and Costa-juss{\`a}, Marta R. and Chuang, Joe and Dale, David and Duppenthaler, Mark and Ekberg, Nathanial Paul and Gao, Cynthia and Licht, Daniel Edward and Maillard, Jean and Mourachko, Alexandre and Ropers, Christophe and Saleem, Safiyyah and S{\'a}nchez, Eduardo and Tsiamas, Ioannis and Turkatenko, Arina and Ventayol-Boada, Albert and Yates, Shireen",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year = "2025",
}

@article{alastruey2026omnilingual,
  title={Omnilingual MT: Machine Translation for 1,600 Languages},
  author={Alastruey, Belen and Bafna, Niyati and Caciolai, Andrea and Heffernan, Kevin and Kozhevnikov, Artyom and Ropers, Christophe and S{\'a}nchez, Eduardo and Saint-James, Charles-Eric and Tsiamas, Ioannis and Cheng, Chierh and others},
  journal={arXiv preprint arXiv:2603.16309},
  year={2026}
}
```