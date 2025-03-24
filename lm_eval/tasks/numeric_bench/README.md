# Task-name

### Paper

Title: **Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models**

<p align="center">
  <img src="https://github.com/TreeAI-Lab/NumericBench/blob/main/figure/NumericBench.png?raw=true" width=1000>
</p>

<p align="center">
    •🔎 <a href="https://github.com/TreeAI-Lab/NumericBench" target="_blank">Github Homepage</a> •📖 <a href="https://arxiv.org/abs/2502.11075" target="_blank">NumericBench Paper</a>  •🤗 <a href="https://huggingface.co/datasets/TreeAILab/NumericBench" target="_blank">NumericBench Dataset</a>
</p>


Abstract: `NumericBench is a comprehensive benchmark designed to evaluate the numerical reasoning capabilities of Large Language Models, addressing their limitations in tasks like arithmetic, number recognition, contextual retrieval, comparison, summarization, and logical reasoning. By incorporating diverse datasets ranging from synthetic number lists to real-world domains like stock trends and weather patterns, NumericBench systematically tests LLMs in both structured and noisy contexts. Experiments on models such as GPT-4o and DeepSeek-V3 reveal significant weaknesses, emphasizing the need for numerically-aware modeling to enhance LLMs' real-world applicability.`

### Limitation

Due to the limitations of the lm-eval framework, we have only integrated most of the datasets, all of which are single-turn. If you need multi-turn datasets, please refer to our [GitHub Homepage](https://github.com/TreeAI-Lab/NumericBench).

### Citation

```
@misc{li2025exposingnumeracygapsbenchmark,
      title={Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models}, 
      author={Haoyang Li and Xuejia Chen and Zhanchao XU and Darian Li and Nicole Hu and Fei Teng and Yiming Li and Luyu Qiu and Chen Jason Zhang and Qing Li and Lei Chen},
      year={2025},
      eprint={2502.11075},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11075}, 
}
```

### Tags, and Tasks

#### Tags

* **NumericBench**: Benchmark for non-CoT models using 6 tasks to evaluate long-context numerical capabilities.
* **NumericBench-cot**: Benchmark for CoT models using 6 tasks to evaluate long-context numerical capabilities.

#### Tasks without CoT

- arithmetic_operation
- mixed_number_string
- num_list
- sequence
- stock_single_turn
- weather_single_turn

#### Tasks with CoT
- arithmetic_operation-cot
- mixed_number_string-cot
- num_list-cot
- sequence-cot
- stock_single_turn-cot
- weather_single_turn-cot


### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
