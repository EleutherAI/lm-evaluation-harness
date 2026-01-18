# MATH-500

### Paper

Let’s Verify Step by Step  
https://arxiv.org/abs/2305.20050  

This dataset contains a subset of 500 problems from the MATH benchmark that OpenAI created in their Let's Verify Step by Step paper. It evaluates models on mathematical reasoning tasks requiring multi-step solutions.  

Homepage: https://github.com/openai/prm800k/tree/main#math-splits

### Citation

```
@misc{lightman2023letsverifystepstep,
      title={Let's Verify Step by Step},
      author={Hunter Lightman and Vineet Kosaraju and Yura Burda and Harri Edwards and Bowen Baker and Teddy Lee and Jan Leike and John Schulman and Ilya Sutskever and Karl Cobbe},
      year={2023},
      eprint={2305.20050},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.20050},
}
```

### Groups and Tasks

#### Groups

* `math_word_problems`

#### Tasks

* `math_500`: The math_500 original dataset with instructions from https://www.vals.ai/benchmarks/math500.
* `math_500_ru`: A Russian translation of math_500 from the [AvitoTech/ru_math500](https://huggingface.co/datasets/AvitoTech/ru_math500) HuggingFace dataset, used to evaluate models' mathematical reasoning in Russian.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * The original dataset and evaluation scripts are available in the [OpenAI prm800k repository](https://github.com/openai/prm800k/tree/main?tab=readme-ov-file#math-splits). The current implementation follows the same prompt format and scoring logic as described in the repository and [vals.ai benchmark page](https://www.vals.ai/benchmarks/math500).

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
