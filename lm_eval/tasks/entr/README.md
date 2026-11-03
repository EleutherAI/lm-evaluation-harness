# ENTR: Turkish Corpus and Synthetic Factual Evaluation Tasks

## Paper

There is no standalone paper for this Harness task collection. Cite the dataset repositories used
by each task.

## Homepage

- [Turkish Wikipedia validation data](https://huggingface.co/datasets/umutyesildal/entr-eval-trwiki-validation)
- [Held-out source IDs](https://huggingface.co/datasets/umutyesildal/entr-eval-heldout-ids)
- [Synthetic factual panels](https://huggingface.co/datasets/umutyesildal/entr-eval-factual-panels)

## Groups

- `entr_corpus_tr`: Turkish Wikipedia validation and OSCAR Turkish held-out tasks.
- `entr_factual`: fixed five-candidate and observed-error exploratory tasks.

Neither group defines an aggregate metric.

## Tasks

- `entr_trwiki_tr`: rolling log-likelihood on the pinned Turkish Wikipedia validation split.
- `entr_oscar_tr_heldout`: rolling log-likelihood on text reconstructed from the pinned VNGRS
  source shards and the published OSCAR held-out locator.
- `entr_mc4_tr_heldout`: rolling log-likelihood on text reconstructed from the pinned VNGRS
  source shards and the mC4 locator mapped from the historical held-out ID file.
- `entr_factual_five_choice_fixed`: fixed five-candidate synthetic-world panel.
- `entr_factual_five_choice_observed_error`: outcome-informed exploratory challenge pool; it is not
  held-out validation.

The generic VNGRS loader accepts both `oscar` and `mc4` corpus identifiers. The mC4 locator is
published at the pinned held-out-ID dataset revision; its IDs were mapped from the historical
held-out file, not recomputed.

## Metrics and interpretation

The rolling corpus tasks return weighted pairs for `word_perplexity`, `byte_perplexity`, and
`bits_per_byte`. The word count follows upstream WikiText's raw-text definition:
`len(re.split(r"\s+", text))`. Text is not detokenized. Word/byte perplexity from these tasks is
not token perplexity.

The factual tasks expose Harness `acc` and `acc_norm`. Harness `acc` is summed-log-likelihood
argmax with Harness tie handling; `acc_norm` divides by character length. Neither is the thesis
primary metric. The `entr-eval` package postprocessor computes the historical mean answer-token
log-likelihood with pessimistic ties.

The candidate facts describe a synthetic world, not real biographies. An unexposed model is not
expected to know arbitrary bindings. The model states are:

- **M0:** frozen base.
- **M1:** M0 after controlled English factual teaching.
- **M2-A:** M1 after general Turkish adaptation.
- **M2-B:** the same M1 after matched Turkish adaptation with factual re-exposure.

M2-A and M2-B are siblings from M1; B never starts from A and is **not pure transfer**. Exposure
subset, model state, question language, candidate language, and accepted generated-answer language
are separate dimensions. There is no composite or overall Turkish score.

## Data and licensing

- `entr_trwiki_tr`: `umutyesildal/entr-eval-trwiki-validation` at revision
  `635c45b9f1b9ae0f7e76ed2f7f411bd2c63307c0`, CC BY-SA 4.0; attribute Turkish Wikipedia and
  Wikimedia contributors.
- `entr_oscar_tr_heldout` and `entr_mc4_tr_heldout`: locator rows from
  `umutyesildal/entr-eval-heldout-ids` at revision `fe6bd502be0f6154ca9a6e4fdba06c4c5447c495`,
  CC BY-NC-SA 4.0. The repository hosts IDs only;
  evaluation text is rebuilt from `vngrs-ai/vngrs-web-corpus` at revision
  `ee5c6201ee84457a18182bfc483a7d8a7f3655ba`. The OSCAR locator references only 2 of the 32
  source shards (`data/train-00004-of-00284.parquet` and `data/train-00013-of-00284.parquet`,
  895,992,220 bytes in total), so the first cache fill downloads about 0.9 GB. The loader verifies
  every selected shard's size and SHA-256 before parsing.
- `entr_mc4_tr_heldout` maps its 10,000 IDs from the historical held-out ID file; the split was not
  recomputed. Its locator references 30 of 32 shards (8,606,323,208 bytes), so its first cache
  fill downloads about 8.6 GB of source shards; plan HF cache storage accordingly. The source
  license is CC BY-NC-SA 4.0.
- Factual panels: `umutyesildal/entr-eval-factual-panels` at revision
  `3d07437e2fb73ad1819a29c00165dbcb2822ccf5`, MIT.

## Checklist

- [x] Public source revisions, identity hashes, and licensing notes are recorded.
- [x] Corpus and factual groups preserve separate metrics without aggregation.
- [x] The observed-error pool is labelled exploratory and outcome-informed.
- [ ] Native model inference and full benchmark evaluation have not been performed here.

## Citation

Cite the relevant Hugging Face dataset repository and pinned revision listed above. For VNGRS
source data, also cite the references listed by its upstream dataset card:
[arXiv:2403.01308](https://arxiv.org/abs/2403.01308) and
[arXiv:2401.14373](https://arxiv.org/abs/2401.14373).
