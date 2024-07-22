# BasqueGLUE

### Paper

Title: `BasqueGLUE: A Natural Language Understanding Benchmark for Basque`

Abstract: `https://aclanthology.org/2022.lrec-1.172/`

Natural Language Understanding (NLU) technology has improved significantly over the last few years and multitask benchmarks such as GLUE are key to evaluate this improvement in a robust and general way. These benchmarks take into account a wide and diverse set of NLU tasks that require some form of language understanding, beyond the detection of superficial, textual clues. However, they are costly to develop and language-dependent, and therefore they are only available for a small number of languages. In this paper, we present BasqueGLUE, the first NLU benchmark for Basque, a less-resourced language, which has been elaborated from previously existing datasets and following similar criteria to those used for the construction of GLUE and SuperGLUE. We also report the evaluation of two state-of-the-art language models for Basque on BasqueGLUE, thus providing a strong baseline to compare upon. BasqueGLUE is freely available under an open license.

Homepage: `https://github.com/orai-nlp/BasqueGLUE`

Title: `Latxa: An Open Language Model and Evaluation Suite for Basque`

Abstract: `https://arxiv.org/abs/2403.20266`

The use of BasqueGLUE for evaluating the performance of decoder models in Basque is presented in this paper.

Homepage: `https://github.com/hitz-zentroa/latxa`

### Citation

```
@InProceedings{urbizu2022basqueglue,
  author    = {Urbizu, Gorka  and  San Vicente, Iñaki  and  Saralegi, Xabier  and  Agerri, Rodrigo  and  Soroa, Aitor},
  title     = {BasqueGLUE: A Natural Language Understanding Benchmark for Basque},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {1603--1612},
  url       = {https://aclanthology.org/2022.lrec-1.172}
}

@misc{etxaniz2024latxa,
      title={Latxa: An Open Language Model and Evaluation Suite for Basque},
      author={Julen Etxaniz and Oscar Sainz and Naiara Perez and Itziar Aldabe and German Rigau and Eneko Agirre and Aitor Ormazabal and Mikel Artetxe and Aitor Soroa},
      year={2024},
      eprint={2403.20266},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* `basque-glue`: First version of the implementation

#### Tasks

* `bhtc_v2`: Topic classification of news extracts with 12 categories.
* `bec2016eu`: Sentiment analysis on tweets about the campaign for the 2016 Basque elections.
* `vaxx_stance`: Stance detection on tweets around the anti-vaccine movement.
* `qnlieu`: Q&A NLI as in [glue/qnli](../glue/qnli).
* `wiceu`: Word-in-Context as in [super_glue/wic](../super_glue/wic).
* `epec_koref_bin`: Correference detection as in [super_glue/wsc](../super_glue/wsc).

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
