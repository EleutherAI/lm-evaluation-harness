# CaseHOLD

### Paper

Title: `When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset`

Abstract: `https://arxiv.org/abs/2104.08671`

CaseHOLD (Case Holdings On Legal Decisions) is a law dataset comprised of over 53,000+ multiple choice questions to identify the relevant holding of a cited case. This dataset presents a fundamental task to lawyers and is both legally meaningful and difficult from an NLP perspective (F1 of 0.4 with a BiLSTM baseline). The citing context from the judicial decision serves as the prompt for the question. The answer choices are holding statements derived from citations following text in a legal decision. There are five answer choices for each citing text. The correct answer is the holding statement that corresponds to the citing text. The four incorrect answers are other holding statements.


Homepage: `https://github.com/reglab/casehold`

### Citation

```
@inproceedings{zhengguha2021,
	title={When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset},
	author={Lucia Zheng and Neel Guha and Brandon R. Anderson and Peter Henderson and Daniel E. Ho},
	year={2021},
	eprint={2104.08671},
	archivePrefix={arXiv},
	primaryClass={cs.CL},
	booktitle={Proceedings of the 18th International Conference on Artificial Intelligence and Law},
	publisher={Association for Computing Machinery}
}
```

### Groups, Tags, and Tasks

#### Groups

None.

#### Tags

None.

#### Tasks

* `casehold`: Complete the following excerpt from a US court opinion. Multiple choice with 5 options.

### Checklist

For adding novel benchmarks/datasets to the library:
  * [X] Is the task an existing benchmark in the literature?
  * [X] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    - Note: I am running with some problems while reproducing the results. I will update the checklist after solving the problem.

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
