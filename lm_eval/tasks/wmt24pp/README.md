# WMT24++ Translation Tasks

This directory provides YAML-based tasks for evaluating English→X machine
translation on the **WMT24++** benchmark hosted on the Hugging Face Hub as
[`google/wmt24pp`](https://huggingface.co/datasets/google/wmt24pp).

Each language pair is exposed as a separate task, using consistent
WMT-style generation and metrics.

## Dataset

- **HF ID**: `google/wmt24pp`
- **Configs**: one per language pair (e.g. `en-de_DE`, `en-pl_PL`, `en-pt_BR`, ...)
- **Split**: single split (`train`), used here as the evaluation split
- **Fields (per example)**:
  - `lp`: language pair, e.g. `"en-de_DE"`
  - `domain`: text domain (canary, news, social, speech, literary)
  - `document_id`: document identifier
  - `segment_id`: global segment identifier
  - `is_bad_source`: boolean flag for low-quality sources
  - `source`: English source sentence
  - `target`: post-edit of `original_target` (recommended reference)
  - `original_target`: original reference translation

In this task family, we:
- **always evaluate English→X** using `source` as input and `target` as reference
- **drop all examples with `is_bad_source == True`**
- **use all domains** (no filtering on `domain`).

## Tasks

Common configuration is defined in `wmt24pp_common.yaml` (note the missing file
extension; this is the file referenced by `include: wmt24pp_common.yaml` in every
per-language YAML):

- `dataset_path: google/wmt24pp`
- `test_split: train`
- `output_type: generate_until`
- `doc_to_text: !function utils.doc_to_text`
- `doc_to_target: "{{target}}"`
- `custom_dataset: !function utils.load_wmt24pp_dataset`
- Metrics: **BLEU**, **TER**, **ChrF** (same triple as classic WMT tasks)

The `lang_pair` in `metadata` is passed to `utils.load_wmt24pp_dataset`, which
loads the corresponding HF config and filters out bad sources.

Each language pair has its own YAML including the common config, e.g.:

```yaml
include: wmt24pp_common.yaml

task: wmt24pp-en-de_DE

tag:
  - translation
  - wmt24pp

metadata:
  version: 1.0
  lang_pair: "en-de_DE"
```

The `lang_pair` in `metadata` is passed to `utils.load_wmt24pp_dataset`, which
loads the corresponding HF config and filters out bad sources.

All available language pairs are listed in the dataset card; in this repo they
are instantiated as tasks named `wmt24pp-<lp>`, where `<lp>` matches the HF
config (e.g. `wmt24pp-en-pt_BR`).

### Group

`wmt24pp_group.yaml` defines a group:

- `group: wmt24pp`
- `group_alias: WMT24++`
- `task: [wmt24pp-en-de_DE, wmt24pp-en-pl_PL, ...]`
- `aggregate_metric_list` aggregating **ChrF** across all subtasks using
  `mean` (weighted by dataset size).

You can run all WMT24++ tasks via:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt24pp
```

or select any subset of language pairs explicitly:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt24pp-en-de_DE wmt24pp-en-pl_PL
```

You can also provide a chat template:

```bash
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks wmt24pp-en-de_DE wmt24pp-en-pl_PL \
  --apply_chat_template ...
```

## Example evaluation config

You can run a subset of language pairs using a YAML config.

```yaml
model: hf
model_args:
  pretrained: Qwen/Qwen2.5-7B-Instruct
  dtype: float16

tasks:
  - wmt24pp-en-pl_PL

num_fewshot: 0
batch_size: 1
max_batch_size: 1
# device: cuda
limit: 10

gen_kwargs:
  temperature: 0.0
  max_gen_toks: 1400

output_path: ./results/
log_samples: true

wandb_args: {}
hf_hub_log_args: {}
```

With the configuration in the YAML file, you can run an experiment with the following command:

```bash
lm_eval run \
  --config my-tasks-config.yaml \
  --apply_chat_template ... \
```

## Metrics

We follow the same metric setup as the other WMT translation tasks in this
repository, exposing three standard MT metrics:

- **BLEU** (`bleu`) – via SacreBLEU
- **TER** (`ter`) – Translation Error Rate
- **ChrF++** (`chrf`) – primary metric of interest for WMT24++ (character n‑gram
  F-score), matching common reporting practices (e.g. Nemotron-3 Nano 30B).

All metrics are implemented via `lm_eval.api.metrics` and use SacreBLEU under
the hood.

## Task Validity Checklist

For adding novel benchmarks/datasets to the library:

- [x] **Is the task an existing benchmark in the literature?**  
  Yes. WMT24++ extends the official WMT24 benchmark to 55 languages/dialects as
described by Deutsch et al. (2025).
- [x] **Have you referenced the original paper that introduced the task?**  
  The citation for the WMT24++ paper is provided in the section below.
- [ ] **If yes, does the original paper provide a reference implementation?**  
  Prompt template and dataset filtering match the reference release. But we didn't replicate full original implementation. 

If other tasks on this dataset are already supported:

- [x] **Is the "Main" variant of this task clearly denoted?**  
  Yes. Every YAML task is `wmt24pp-en-<target>` to emphasize the English→X
setup, and the group config exposes the complete benchmark as `wmt24pp`.
- [x] **Have you provided a short sentence on what each new variant adds / evaluates?**  
  The README explains that each YAML corresponds to a single HF config / language
pair; they all evaluate the same translation direction with identical metrics.
- [x] **Have you noted which published evaluation setups are matched by this variant?**  
  Yes. See the section above for the specific alignment with the WMT24++ dataset
card: same split (`train`), same bad-source filtering, same post-edited reference,
and the BLEU/TER/ChrF++ metric trio used in the paper/MTME release.

## Citation

Please cite the original WMT24++ paper and the lm-evaluation-harness project
as appropriate when using these tasks in publications.

```
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
