# MLQA

### Paper

Title: `MLQA: Evaluating Cross-lingual Extractive Question Answering`

Abstract: `https://arxiv.org/abs/1910.07475`

MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
4 different languages on average

Homepage: `https://github.com/facebookresearch/MLQA`


### Citation

```
@misc{lewis2020mlqaevaluatingcrosslingualextractive,
      title={MLQA: Evaluating Cross-lingual Extractive Question Answering},
      author={Patrick Lewis and Barlas Oğuz and Ruty Rinott and Sebastian Riedel and Holger Schwenk},
      year={2020},
      eprint={1910.07475},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1910.07475},
}
```

### Groups, Tags, and Tasks

#### Groups

* Not part of a group yet

#### Tasks

Tasks of the form `mlqa_context-lang_question-lang`
* `mlqa_ar_ar`
* `mlqa_ar_de`
* `mlqa_ar_vi`
* `mlqa_ar_zh`
* `mlqa_ar_en`
* `mlqa_ar_es`
* `mlqa_ar_hi`
* `mlqa_de_ar`
* `mlqa_de_de`
* `mlqa_de_vi`
* `mlqa_de_zh`
* `mlqa_de_en`
* `mlqa_de_es`
* `mlqa_de_hi`
* `mlqa_vi_ar`
* `mlqa_vi_de`
* `mlqa_vi_vi`
* `mlqa_vi_zh`
* `mlqa_vi_en`
* `mlqa_vi_es`
* `mlqa_vi_hi`
* `mlqa_zh_ar`
* `mlqa_zh_de`
* `mlqa_zh_vi`
* `mlqa_zh_zh`
* `mlqa_zh_en`
* `mlqa_zh_es`
* `mlqa_zh_hi`
* `mlqa_en_ar`
* `mlqa_en_de`
* `mlqa_en_vi`
* `mlqa_en_zh`
* `mlqa_en_en`
* `mlqa_en_es`
* `mlqa_en_hi`
* `mlqa_es_ar`
* `mlqa_es_de`
* `mlqa_es_vi`
* `mlqa_es_zh`
* `mlqa_es_en`
* `mlqa_es_es`
* `mlqa_es_hi`
* `mlqa_hi_ar`
* `mlqa_hi_de`
* `mlqa_hi_vi`
* `mlqa_hi_zh`
* `mlqa_hi_en`
* `mlqa_hi_es`
* `mlqa_hi_hi`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
