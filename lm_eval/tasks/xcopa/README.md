# XCOPA

### Paper

Title: `XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning`

Abstract: https://ducdauge.github.io/files/xcopa.pdf

The Cross-lingual Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across languages.
The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around the globe.
The dataset is challenging as it requires both the command of world knowledge and the ability to generalise to new languages.
All the details about the creation of XCOPA and the implementation of the baselines are available in the paper.

Homepage: https://github.com/cambridgeltl/xcopa

### Citation

```
@inproceedings{ponti2020xcopa,
  title={{XCOPA: A} Multilingual Dataset for Causal Commonsense Reasoning},
  author={Edoardo M. Ponti, Goran Glava\v{s}, Olga Majewska, Qianchu Liu, Ivan Vuli\'{c} and Anna Korhonen},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020},
  url={https://ducdauge.github.io/files/xcopa.pdf}
}
```

### Groups and Tasks

#### Groups

* `xcopa`

#### Tasks

* `xcopa_et`: Estonian
* `xcopa_ht`: Haitian Creole
* `xcopa_id`: Indonesian
* `xcopa_it`: Italian
* `xcopa_qu`: Cusco-Collao Quechua
* `xcopa_sw`: Kiswahili
* `xcopa_ta`: Tamil
* `xcopa_th`: Thai
* `xcopa_tr`: Turkish
* `xcopa_vi`: Vietnamese
* `xcopa_zh`: Mandarin Chinese


### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
