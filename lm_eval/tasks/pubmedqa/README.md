# PubMedQA

### Paper

Title: `PubMedQA: A Dataset for Biomedical Research Question Answering`

Abstract: https://arxiv.org/abs/1909.06146

PubMedQA is a novel biomedical question answering (QA) dataset collected from
PubMed abstracts. The task of PubMedQA is to answer research questions with
yes/no/maybe (e.g.: Do preoperative statins reduce atrial fibrillation after
coronary artery bypass grafting?) using the corresponding abstracts. PubMedQA
has 1k expert-annotated, 61.2k unlabeled and 211.3k artificially generated QA
instances. Each PubMedQA instance is composed of (1) a question which is either
an existing research article title or derived from one, (2) a context which is
the corresponding abstract without its conclusion, (3) a long answer, which is
the conclusion of the abstract and, presumably, answers the research question,
and (4) a yes/no/maybe answer which summarizes the conclusion.

Homepage: https://pubmedqa.github.io/


### Citation

```
@inproceedings{jin2019pubmedqa,
    title={PubMedQA: A Dataset for Biomedical Research Question Answering},
    author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
    booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
    pages={2567--2577},
    year={2019}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `pubmed_qa`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
