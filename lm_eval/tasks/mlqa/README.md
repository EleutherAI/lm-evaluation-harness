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

Tasks of the form `mlqa_context-lang_question-lang.yaml`
* `mlqa_ar_ar.yaml`
* `mlqa_ar_de.yaml`
* `mlqa_ar_vi.yaml`
* `mlqa_ar_zh.yaml`
* `mlqa_ar_en.yaml`
* `mlqa_ar_es.yaml`
* `mlqa_ar_hi.yaml`
* `mlqa_de_ar.yaml`
* `mlqa_de_de.yaml`
* `mlqa_de_vi.yaml`
* `mlqa_de_zh.yaml`
* `mlqa_de_en.yaml`
* `mlqa_de_es.yaml`
* `mlqa_de_hi.yaml`
* `mlqa_vi_ar.yaml`
* `mlqa_vi_de.yaml`
* `mlqa_vi_vi.yaml`
* `mlqa_vi_zh.yaml`
* `mlqa_vi_en.yaml`
* `mlqa_vi_es.yaml`
* `mlqa_vi_hi.yaml`
* `mlqa_zh_ar.yaml`
* `mlqa_zh_de.yaml`
* `mlqa_zh_vi.yaml`
* `mlqa_zh_zh.yaml`
* `mlqa_zh_en.yaml`
* `mlqa_zh_es.yaml`
* `mlqa_zh_hi.yaml`
* `mlqa_en_ar.yaml`
* `mlqa_en_de.yaml`
* `mlqa_en_vi.yaml`
* `mlqa_en_zh.yaml`
* `mlqa_en_en.yaml`
* `mlqa_en_es.yaml`
* `mlqa_en_hi.yaml`
* `mlqa_es_ar.yaml`
* `mlqa_es_de.yaml`
* `mlqa_es_vi.yaml`
* `mlqa_es_zh.yaml`
* `mlqa_es_en.yaml`
* `mlqa_es_es.yaml`
* `mlqa_es_hi.yaml`
* `mlqa_hi_ar.yaml`
* `mlqa_hi_de.yaml`
* `mlqa_hi_vi.yaml`
* `mlqa_hi_zh.yaml`
* `mlqa_hi_en.yaml`
* `mlqa_hi_es.yaml`
* `mlqa_hi_hi.yaml`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
