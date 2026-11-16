# Colossal Clean Crawled Corpus(C4)

### Paper

[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

A colossal, cleaned version of Common Crawl's web crawl corpus. Based on [Common Crawl dataset](https://commoncrawl.org).

This is the processed version of Google's C4 dataset.

[Homepage](https://huggingface.co/datasets/allenai/c4)

### Citation

```text
@misc{raffel2023exploringlimitstransferlearning,
      title={Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
      author={Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
      year={2023},
      eprint={1910.10683},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1910.10683},
}
```

### Groups, Tags, and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `c4`: measure perplexity on the C4 dataset, via rolling loglikelihoods.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
