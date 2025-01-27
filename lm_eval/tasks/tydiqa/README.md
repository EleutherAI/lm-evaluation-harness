# TyDi QA

### Paper

Title: `TYDI QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages`

Abstract: `https://arxiv.org/abs/2003.05002`

`TyDi QA is a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs. The languages of TyDi QA are diverse with regard to their typology -- the set of linguistic features that each language expresses -- such that we expect models performing well on this set to generalize across a large number of the languages in the world. It contains language phenomena that would not be found in English-only corpora. To provide a realistic information-seeking task and avoid priming effects, questions are written by people who want to know the answer, but don’t know the answer yet, (unlike SQuAD and its descendents) and the data is collected directly in each language without the use of translation (unlike MLQA and XQuAD).`

Homepage: `https://github.com/google-research-datasets/tydiqa` , `https://ai.google.com/research/tydiqa`


### Citation

```
@misc{clark2020tydiqabenchmarkinformationseeking,
      title={TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
      author={Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki},
      year={2020},
      eprint={2003.05002},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2003.05002},
}
```

### Groups, Tags, and Tasks

#### Groups

* `SelectP`:  Given a list of the passages in the article, return either (a) the index of the passage that answers the question or (b) NULL if no such passage exists.
* `MinSpan`: Given the full text of an article, return one of (a) the start and end byte indices of the minimal span that completely answers the question; (b) YES or NO if the question requires a yes/no answer and we can draw a conclusion from the passage; (c) NULL if it is not possible to produce a minimal answer for this question.
* `Gold Passage Task`:  Given a passage that is guaranteed to contain the answer, predict the single contiguous span of characters that answers the question.

#### Tags

* `tag_name`: `Short description`

#### Tasks

* `task_name`: `1-sentence description of what this particular task does`
* `task_name2`: ...

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
