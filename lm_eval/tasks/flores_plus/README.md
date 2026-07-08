# FLORES+

### Paper

Title: `No Language Left Behind: Scaling Human-Centered Machine Translation`

Paper Link: https://arxiv.org/abs/2207.04672

HomePage: https://huggingface.co/datasets/openlanguagedata/flores_plus

### Dataset access

FLORES+ is a gated dataset. Before running or generating these tasks:

1. Accept the dataset terms on Hugging Face.
2. Log in locally, e.g. `huggingface-cli login` or `hf auth login`.

Generation (`generate_tasks.py`) also requires HF Hub access to list languages and
resolve display names from the dataset README.

### Citation

```
@article{nllb2022,
  author    = {NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi,  Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Jeff Wang},
  title     = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  year      = {2022}
}
```

### Groups and Tasks

#### Groups

* `flores_plus` (alias: **FLORES+**)

#### Main variant

The checked-in **main variant** is English-centric translation in both directions
(`X → English` and `English → X`) for all FLORES+ language varieties, using a
single prompt template (“You are a translation expert…”) and the public
`dev` / `devtest` splits.

Other language pairs can be generated locally with `generate_tasks.py` but are
gitignored.

Task naming: `flores_{src}-{tgt}` (alias: `{src-short}->{tgt-short}`)

Examples:

* `flores_fra_Latn-eng_Latn` (alias: `fra_Latn->eng_Latn`)
* `flores_eng_Latn-fra_Latn` (alias: `eng_Latn->fra_Latn`)

### Evaluation setup

This implementation matches the standard harness FLORES translation setup used
elsewhere in the repo (e.g. Galician/Basque FLORES):

* **Splits:** `dev` (validation / few-shot), `devtest` (test)
* **Metrics:** BLEU, chrF, TER
* **Decoding:** greedy (`do_sample: false`, `temperature: 0.0`), stop at newline
* **Prompt:** one-turn translation instruction with source sentence appended

Reference implementations: [SacreBLEU](https://github.com/mjpost/sacrebleu) metrics
via the harness; dataset alignment follows FLORES+ `id` keys across per-language
JSONL files.

### Generating configs

```bash
cd lm_eval/tasks/flores_plus

# Default: all languages, English <-> X, both directions (checked in)
python generate_tasks.py --overwrite

# Any explicit language pairs (local only; use ':' separator)
python generate_tasks.py --overwrite --pairs fra_Latn:deu_Latn zho_Hans:jpn_Jpan

# Only the ordered direction(s) you specify
python generate_tasks.py --overwrite --pairs fra_Latn:deu_Latn --ordered-pairs

# Subset of languages (English-centric)
python generate_tasks.py --overwrite --languages fra_Latn deu_Latn zho_Hans

# All ordered language pairs (very large; local only)
python generate_tasks.py --overwrite --all-pairs
```

### Validating prompts

Inspect rendered prompts before running a full eval:

```bash
python -m scripts.write_out \
  --tasks flores_fra_Latn-eng_Latn \
  --sets test \
  --num_fewshot 0 \
  --num_examples 5 \
  --output_base_path /tmp/flores_plus_write_out
```

### Changelog

- Initial release (v1.0): English-centric FLORES+ translation tasks for 226
  language varieties; BLEU/chrF/TER; configs generated via `generate_tasks.py`.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? SacreBLEU
    metrics are used via the harness; dataset format follows FLORES+ JSONL alignment
    on `id`. Full paper replication may additionally require matching the exact
    tokenizer/preprocessing used in NLLB/FLORES-200 evaluations.

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted? English-centric, both
  directions, single prompt — see **Main variant** above.
* [x] Have you provided a short sentence in a README on what each new variant adds /
  evaluates? Non-English pairs via `--pairs` / `--all-pairs` are local-only.
* [x] Have you noted which, if any, published evaluation setups are matched by this
  variant? See **Evaluation setup** above.
