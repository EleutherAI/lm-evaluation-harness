# XNLIeu

### Paper

Title: XNLIeu: a dataset for cross-lingual NLI in Basque

Abstract: https://arxiv.org/abs/2404.06996

XNLI is a popular Natural Language Inference (NLI) benchmark widely used to evaluate cross-lingual Natural Language Understanding (NLU) capabilities across languages. In this paper, we expand XNLI to include Basque, a low-resource language that can greatly benefit from transfer-learning approaches. The new dataset, dubbed XNLIeu, has been developed by first machine-translating the English XNLI corpus into Basque, followed by a manual post-edition step. We have conducted a series of experiments using mono- and multilingual LLMs to assess a) the effect of professional post-edition on the MT system; b) the best cross-lingual strategy for NLI in Basque; and c) whether the choice of the best cross-lingual strategy is influenced by the fact that the dataset is built by translation. The results show that post-edition is necessary and that the translate-train cross-lingual strategy obtains better results overall, although the gain is lower when tested in a dataset that has been built natively from scratch. Our code and datasets are publicly available under open licenses at https://github.com/hitz-zentroa/xnli-eu.

Homepage: https://github.com/hitz-zentroa/xnli-eu


### Citation

```bibtex
@misc{heredia2024xnlieu,
    title={XNLIeu: a dataset for cross-lingual NLI in Basque},
    author={Maite Heredia and Julen Etxaniz and Muitze Zulaika and Xabier Saralegi and Jeremy Barnes and Aitor Soroa},
    year={2024},
    eprint={2404.06996},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Groups, Tags, and Tasks

#### Tags

* `xnli_eu_mt_native`: Includes MT and Native variants of the XNLIeu dataset.

#### Tasks

* `xnli_eu`: XNLI in Basque postedited from MT.
* `xnli_eu_mt`: XNLI in Basque machine translated from English.
* `xnli_eu_native`: XNLI in Basque natively created.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
