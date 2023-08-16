# bAbI

### Paper

Title: Towards ai-complete question answering: A set of prerequisite toy tasks
Abstract: https://arxiv.org/abs/1502.05698

One long-term goal of machine learning research is to produce methods that are applicable to reasoning and natural language, in particular building an intelligent dialogue agent. To measure progress towards that goal, we argue for the usefulness of a set of proxy tasks that evaluate reading comprehension via question answering. Our tasks measure understanding in several ways: whether a system is able to answer questions via chaining facts, simple induction, deduction and many more. The tasks are designed to be prerequisites for any system that aims to be capable of conversing with a human. We believe many existing learning systems can currently not solve them, and hence our aim is to classify these tasks into skill sets, so that researchers can identify (and then rectify) the failings of their systems. We also extend and improve the recently introduced Memory Networks model, and show it is able to solve some, but not all, of the tasks.

Homepage: https://github.com/facebookarchive/bAbI-tasks


### Citation

```
@article{weston2015towards,
  title={Towards ai-complete question answering: A set of prerequisite toy tasks},
  author={Weston, Jason and Bordes, Antoine and Chopra, Sumit and Rush, Alexander M and Van Merri{\"e}nboer, Bart and Joulin, Armand and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1502.05698},
  year={2015}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `babi`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
