# Task-name

### Paper

Title: `Know What You Don’t Know: Unanswerable Questions for SQuAD`
Abstract: https://arxiv.org/abs/1806.03822

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset,
consisting of questions posed by crowdworkers on a set of Wikipedia articles,
where the answer to every question is a segment of text, or span, from the
corresponding reading passage, or the question might be unanswerable.
SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable
questions written adversarially by crowdworkers to look similar to answerable ones.
To do well on SQuAD2.0, systems must not only answer questions when possible, but
also determine when no answer is supported by the paragraph and abstain from answering.

Homepage: https://rajpurkar.github.io/SQuAD-explorer/


### Citation

```
@misc{rajpurkar2018know,
    title={Know What You Don't Know: Unanswerable Questions for SQuAD},
    author={Pranav Rajpurkar and Robin Jia and Percy Liang},
    year={2018},
    eprint={1806.03822},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `squadv2`: `Default squadv2 task`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
