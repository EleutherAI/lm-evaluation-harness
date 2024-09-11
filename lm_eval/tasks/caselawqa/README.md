# Task-name

CaselawQA

### Paper

Title: Lawma: The Power of Specialization for Legal Tasks

Abstract: https://arxiv.org/abs/2407.16615

CaselawQA is a benchmark designed for legal classification tasks, drawing from the Supreme Court and Songer Court of Appeals databases.
The majority of its 10,000 questions are multiple-choice, with 5,000 sourced from each database. 
The questions are randomly selected from the test sets of the [Lawma tasks](https://huggingface.co/datasets/ricdomolm/lawma-tasks).\
From a technical machine learning perspective, these tasks provide highly non-trivial classification problems where even the best models leave much room for improvement. 
From a substantive legal perspective, efficient solutions to such classification problems have rich and important applications in legal research.
CaselawQA also includes two additional subsets: CaselawQA Tiny and CaselawQA Hard. 
CaselawQA Tiny consists of 49 Lawma tasks with fewer than 150 training examples. 
CaselawQA Hard comprises tasks where [Lawma 70B](https://huggingface.co/ricdomolm/lawma-70b) achieves less than 70% accuracy.

Homepage: https://github.com/socialfoundations/lawma


### Citation

```
@misc{dominguezolmedo2024lawmapowerspecializationlegal,
      title={Lawma: The Power of Specialization for Legal Tasks}, 
      author={Ricardo Dominguez-Olmedo and Vedant Nanda and Rediet Abebe and Stefan Bechtold and Christoph Engel and Jens Frankenreiter and Krishna Gummadi and Moritz Hardt and Michael Livermore},
      year={2024},
      eprint={2407.16615},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.16615}, 
}
```

### Groups, Tags, and Tasks

#### Groups

* `caselawqa`: average accuracy of `caselawqa_sc` and `caselaw_songer`

#### Tags

* `caselawqa_lb`: tasks of the caselaw leaderboard

#### Tasks

* `caselawqa_sc`: 5,000 questions derived from the Supreme Court database
* `caselaw_songer`: 5,000 questions derived from the Songer Court of Appeals database
* `caselaw_tiny`: questions derived from Lawma tasks with fewer than 150 training examples
* `caselaw_hard`: questions derived from tasks for which Lawma 70B achieves less than 70% accuracy

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
