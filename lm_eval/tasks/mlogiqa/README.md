# MLogiQA

## Paper

Title: P-MMEval: A Parallel Multilingual Multitask Benchmark for Consistent Evaluation of LLMs

Abstract: https://arxiv.org/abs/2411.09116

Homepage: https://huggingface.co/datasets/Qwen/P-MMEval/viewer/mlogiqa

### Citation

```text
@misc{zhang2024pmmevalparallelmultilingualmultitask,
      title={P-MMEval: A Parallel Multilingual Multitask Benchmark for Consistent Evaluation of LLMs}, 
      author={Yidan Zhang and Yu Wan and Boyi Deng and Baosong Yang and Haoran Wei and Fei Huang and Bowen Yu and Junyang Lin and Fei Huang and Jingren Zhou},
      year={2024},
      eprint={2411.09116},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.09116}, 
}
```

### Implementation

Figure 12 from the original paper shows the prompt used:

> "Passage: {context}\nQuestion: {question}\nChoices:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD.{option_d}\nPlease choose the most suitable one among A, B, C and Das the answer to this question, and return it in the following JSON format:\n{'answer': '[choice]'}\nwhere [choice] must be one of A, B, C and D."

In this MCQA implementation, we remove the last part of the original prompt:

> "Passage: {context}\nQuestion: {question}\nChoices:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD.{option_d}\nPlease choose the most suitable one among A, B, C and Das the answer to this question."

### Groups, Tags, and Tasks

#### Groups

* `mlogiqa`: MLogiQA problems in all languages

#### Tasks

* `mlogiqa_<language>`: MLogiQA problems in `<language>`
* Languages: ar, en, es, fr, ja, ko, pt, th, vi, zh

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
