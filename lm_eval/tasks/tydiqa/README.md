# tydiqa

### Paper

Title: `TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages`

Abstract: `Boldly venturing beyond English-only machine comprehension, we present TyDi QA---a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs. The languages of TyDi QA are diverse with regard to their typology---the set of linguistic features each language expresses---such that we expect models performing well on this set to generalize across a large number of the world's languages.`

Homepage: https://ai.google.com/research/tydiqa

### Citation

```
@article{clark-etal-2020-tydi,
    title   = {{TyDi QA}: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
    author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki},
    journal = {Transactions of the Association for Computational Linguistics},
    year    = {2020},
    volume  = {8},
    pages   = {454--470},
}
```

### Groups, Tags, and Tasks

#### Tags

* `tydiqa_goldp`: All 11 languages of the TyDiQA Gold Passage task

#### Tasks

This implements the **Gold Passage (secondary) task**: given a passage guaranteed to contain the answer, extract the answer span. Uses the SQuAD 1.1 extractive QA format with language-aware F1 and Exact Match metrics.

* `tydiqa_goldp_ar` - Arabic
* `tydiqa_goldp_bn` - Bengali
* `tydiqa_goldp_en` - English
* `tydiqa_goldp_fi` - Finnish
* `tydiqa_goldp_id` - Indonesian
* `tydiqa_goldp_ja` - Japanese
* `tydiqa_goldp_ko` - Korean
* `tydiqa_goldp_ru` - Russian
* `tydiqa_goldp_sw` - Swahili
* `tydiqa_goldp_te` - Telugu
* `tydiqa_goldp_th` - Thai

### Checklist

For adding novel benchmarks/datasets to the library:
* [X] Is the task an existing benchmark in the literature?
  * [X] Have you referenced the original paper that introduced the task?
  * [X] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [X] Is the "Main" variant of this task clearly denoted?
* [X] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
