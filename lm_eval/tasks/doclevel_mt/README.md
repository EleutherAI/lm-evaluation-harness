# Document-Level Machine Translation Task Collection

This directory provides **meta-groups** over the document-level machine translation task families implemented in this repository.

It does **not** introduce a new benchmark or dataset. Instead, it groups together the existing document-level MT task families so they can be run more conveniently from a single entry point.

Currently, this collection covers:

* **WMT24++** document-level tasks
* **BOUQuET** document-level tasks
* **NTREX** document-level tasks
* **WMT GeneralMT** document-level tasks

## Purpose

The goal of this collection is to make it easy to run:

* all English-centric bidirectional document-level MT tasks together
* all **English → X** document-level tasks together
* all **X → English** document-level tasks together

This is especially useful for broad evaluation sweeps across multiple benchmarks with a single `--tasks` argument.

## Included task families

### WMT24++

This family provides:

* official **English → X** tasks
* derived **X → English** tasks
* document-level evaluation using reconstructed documents from WMT24++

Typical groups include:

* `wmt24pp-en-all`
* `wmt24pp-all-en`
* `wmt24pp-bidirectional-all`

### BOUQuET

This family provides:

* English-centric document-level tasks:
  * `bouquet-doc-en-all`
  * `bouquet-doc-all-en`
  * `bouquet-doc-bidirectional-all`
* a generic runtime wrapper for arbitrary non-English language pairs

For the purposes of this collection, only the **named static groups** are included. Runtime one-off pairs launched through the wrapper are not part of the meta-groups.

### NTREX

This family provides:

* official **eng → X** tasks
* derived **X → eng** tasks
* document-level reconstruction using `DOCUMENT_IDS.tsv`

Typical groups include:

* `ntrex-doc-en-all`
* `ntrex-doc-all-en`
* `ntrex-doc-bidirectional-all`

### WMT GeneralMT

This family provides:

* multilingual document-level tasks reconstructed from processed WMT XML files
* grouping by language pair
* grouping by year
* special global groups for:
  * `wmt-generalmt-doc-all`
  * `wmt-generalmt-doc-en-all`
  * `wmt-generalmt-doc-all-en`

This family intentionally excludes:

* **FLORES** testsets
* **WMT24** testsets

For WMT24 document-level evaluation, use **`wmt24pp_doc`** instead.

## Meta-groups

This collection is intended to provide groups such as:

* `doc-mt-all`
* `doc-mt-en-all`
* `doc-mt-all-en`

Their purpose is:

* **`doc-mt-all`**
  Run all included English-centric bidirectional document-level MT tasks together.

* **`doc-mt-en-all`**
  Run all included **English → X** document-level MT tasks.

* **`doc-mt-all-en`**
  Run all included **X → English** document-level MT tasks.


## Running the collection

Run all English-centric document-level MT tasks:

```bash id="v4nb1b"
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks doc-mt-all
```

Run all **English → X** document-level MT tasks:

```bash id="wcff7p"
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks doc-mt-en-all
```

Run all **X → English** document-level MT tasks:

```bash id="z8aw30"
python -m lm_eval run \
  --model hf --model_args pretrained=... \
  --tasks doc-mt-all-en
```

## Notes

* This directory is a **task aggregator**, not a standalone benchmark.
* The actual data loading, prompt rendering, and metric definitions remain in the underlying task families.
* Any dataset-specific caveats still apply:

  * WMT24++ reverse directions are derived
  * BOUQuET generic multilingual pairs are runtime-generated and not statically grouped here
  * NTREX reverse directions are derived
  * WMT GeneralMT is built from locally processed XML files

## Citation

Please cite the original datasets and the lm-evaluation-harness project as appropriate when using these tasks in publications.
