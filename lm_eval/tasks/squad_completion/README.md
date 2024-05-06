# Squad-completion

### Paper

Title: Simple Linear Attention Language Models Balance The Recall-Throughput Tradeoff

A Variant of the SQuAD question answering task, as implemented by Based. See [https://github.com/EleutherAI/lm-evaluation-harness/lm_eval/tasks/squadv2/README.md] for more info.

Homepage: https://github.com/HazyResearch/based-evaluation-harness




### Citation

```
@misc{arora2024simple,
      title={Simple linear attention language models balance the recall-throughput tradeoff},
      author={Simran Arora and Sabri Eyuboglu and Michael Zhang and Aman Timalsina and Silas Alberti and Dylan Zinsley and James Zou and Atri Rudra and Christopher Ré},
      year={2024},
      eprint={2402.18668},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

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

#### Tasks

* `squad_completion`: the SQuAD task as implemented in the paper "Simple linear attention language models balance the recall-throughput tradeoff". Designed for zero-shot evaluation of small LMs.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
