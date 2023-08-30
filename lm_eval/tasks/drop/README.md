# DROP

### Paper

Title: `DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs`

Abstract: https://aclanthology.org/attachments/N19-1246.Supplementary.pdf

DROP is a QA dataset which tests comprehensive understanding of paragraphs. In
this crowdsourced, adversarially-created, 96k question-answering benchmark, a
system must resolve multiple references in a question, map them onto a paragraph,
and perform discrete operations over them (such as addition, counting, or sorting).

Homepage: https://allenai.org/data/drop

Acknowledgement: This implementation is based on the official evaluation for `DROP`:
https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py

### Citation

```
@misc{dua2019drop,
    title={DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs},
    author={Dheeru Dua and Yizhong Wang and Pradeep Dasigi and Gabriel Stanovsky and Sameer Singh and Matt Gardner},
    year={2019},
    eprint={1903.00161},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `drop`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
