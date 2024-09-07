# DROP

### Paper

Title: `TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages`

Abstract: https://arxiv.org/pdf/2003.05002

Confidently making progress on multilingual modeling requires challenging, trustworthy evaluations. We present TyDi QA---a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs. The languages of TyDi QA are diverse with regard to their typology---the set of linguistic features each language expresses---such that we expect models performing well on this set to generalize across a large number of the world's languages. We present a quantitative analysis of the data quality and example-level qualitative linguistic analyses of observed language phenomena that would not be found in English-only corpora. To provide a realistic information-seeking task and avoid priming effects, questions are written by people who want to know the answer, but don't know the answer yet, and the data is collected directly in each language without the use of translation.

Homepage: https://ai.google.com/research/tydiqa/

Acknowledgement: This implementation is based on the official evaluation for `TyDiQA Gold`:
https://github.com/google-research-datasets/tydiqa/blob/master/tydi_eval.py

### Citation

```
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
year    = {2020},
journal = {Transactions of the Association for Computational Linguistics}
}
```

### Groups and Tasks

#### Groups

* 

#### Tasks

* `tydiqa`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
