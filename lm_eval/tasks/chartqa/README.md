# Task-name

### Paper

Title: `ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning`

Abstract: `In this work, we present a large-scale benchmark covering 9.6K human-written questions as well as 23.1K questions generated from human-written chart summaries.`

`Short description of paper / benchmark goes here:`

Homepage: `https://github.com/vis-nlp/ChartQA`


### Citation

```
@misc{masry2022chartqabenchmarkquestionanswering,
      title={ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning},
      author={Ahmed Masry and Do Xuan Long and Jia Qing Tan and Shafiq Joty and Enamul Hoque},
      year={2022},
      eprint={2203.10244},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2203.10244},
}
```

### Groups, Tags, and Tasks

#### Tasks

* `chartqa`: `Prompt taken from on mistral-evals: https://github.com/mistralai/mistral-evals/blob/main/eval/tasks/chartqa.py`
* `chartqa_llama`: `variant as implemented in https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/eval_details.md`
* `chartqa_llama_90`: `similar to chartqa_llama but specific to the 90B models of llama 3.2`


### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
