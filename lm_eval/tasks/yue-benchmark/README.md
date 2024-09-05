# Yue-Benchmark

### Paper

Title: `How Far Can Cantonese NLP Go? Benchmarking Cantonese Capabilities of Large Language Models`

Abstract: https://arxiv.org/abs/2408.16756

This paper introduces benchmarks for evaluating large language models (LLMs) in Cantonese, including Yue-TruthfulQA, Yue-GSM8K, Yue-ARC-C, Yue-MMLU, and Yue-TRANS. Each benchmark targets different aspects of language understanding and generation in Cantonese, offering a comprehensive assessment of LLMs' capabilities in this language.

Homepage: [GitHub](https://github.com/jiangjyjy/Yue-Benchmark), [HuggingFace](https://huggingface.co/datasets/BillBao/Yue-Benchmark)

### Citation

```
@misc{jiang2024farcantonesenlpgo,
      title={How Far Can Cantonese NLP Go? Benchmarking Cantonese Capabilities of Large Language Models},
      author={Jiyue Jiang and Liheng Chen and Pengan Chen and Sheng Wang and Qinghang Bao and Lingpeng Kong and Yu Li and Chuan Wu},
      year={2024},
      eprint={2408.16756},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.16756},
}
```

### Groups and Tasks

#### Groups

* `yue_benchmark`: Benchmarks for evaluating large language models (LLMs) in Cantonese, including multiple tasks such as Yue-TruthfulQA, Yue-GSM8K, Yue-ARC-C, Yue-MMLU, and Yue-TRANS.

#### Tasks

* `Yue-TruthfulQA`: `Evaluates LLMs on truthfulness and accuracy in Cantonese.`
* `Yue-GSM8K`: `Tests mathematical problem-solving abilities in Cantonese.`
* `Yue-ARC-C`: `Assesses LLMs on Cantonese multiple-choice questions.`
* `Yue-MMLU`: `Evaluates multi-task learning and understanding in Cantonese.`
* `Yue-TRANS`: `Measures translation accuracy between English/Mandarin and Cantonese.`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
    * [x] Have you referenced the original paper that introduced the task?
    * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds/evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### License

The Yue-Benchmark dataset is licensed under the MIT License.
