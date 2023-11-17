"""
SCROLLS: Standardized CompaRison Over Long Language Sequences
https://arxiv.org/abs/2201.03533

SCROLLS is a suite of datasets that require synthesizing information over long texts.
The benchmark includes seven natural language tasks across multiple domains,
including summarization, question answering, and natural language inference.

Homepage: https://www.scrolls-benchmark.com/

Since SCROLLS tasks are generally longer than the maximum sequence length of many models,
it is possible to create "subset" tasks that contain only those samples whose tokenized length
is less than some pre-defined limit. For example, to create a subset of "Qasper" that would
be suitable for a model using the GPTNeoX tokenizer and a 4K maximium sequence length:

```
class QasperGPTNeoX4K(Qasper):
    PRUNE_TOKENIZERS = ["EleutherAI/pythia-410m-deduped"]
    PRUNE_MAX_TOKENS = 4096
    PRUNE_NUM_PROC = _num_cpu_cores() # optional, to speed up pruning of large datasets like NarrativeQA
```

`PRUNE_TOKENIZERS` can contain more than one tokenizer; this will include only samples that are
less than `PRUNE_MAX_TOKENS` for ALL of the tokenizers. This can be useful to comparing models
that use different tokenizers but the same maximum sequence length.

Once the subset task class has been defined in this file, it can be used by adding the class
to `lm_eval/tasks/__init__.py`.

NOTE: GovReport may need `max_gen_toks` set larger for causal models.
"""
