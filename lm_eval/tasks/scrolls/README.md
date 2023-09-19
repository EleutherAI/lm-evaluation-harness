# SCROLLS

### Paper

Title: `SCROLLS: Standardized CompaRison Over Long Language Sequences`

Abstract: https://arxiv.org/abs/2201.03533

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

### Citation

```
@inproceedings{shaham-etal-2022-scrolls,
    title = "{SCROLLS}: Standardized {C}ompa{R}ison Over Long Language Sequences",
    author = "Shaham, Uri  and 
      Segal, Elad  and
      Ivgi, Maor  and
      Efrat, Avia  and
      Yoran, Ori  and
      Haviv, Adi  and
      Gupta, Ankit  and
      Xiong, Wenhan  and
      Geva, Mor  and
      Berant, Jonathan  and
      Levy, Omer",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.823",
    pages = "12007--12021"
}
```

### Groups and Tasks

#### Groups

* `qasper`: executes both `qasper_bool` and `qasper_freeform`

#### Tasks

* `qasper_bool`: Multiple choice task that evaluates the task with `answer_type="bool"` 
* `qasper_freeform`: Greedy generation task that evaluates the samples from the task with `answer_type="free form answer"`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
