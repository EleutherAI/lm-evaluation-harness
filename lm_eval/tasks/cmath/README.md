# CMATH

### Paper

Title: CMATH: Can Your Language Model Pass Chinese Elementary School Math Test?

Abstract: We introduce CMATH, a Chinese elementary school math word problem dataset, consisting of 1498 problems from real-world elementary school exams in China, accompanied by detailed annotations. These problems span six difficulty levels (grades 1–6) and cover a wide range of topics, including addition, subtraction, multiplication, division, fractions, and their combinations. This dataset can be used to evaluate the reasoning abilities of LLMs on Chinese math word problems.

- Dataset: `weitianwen/cmath`
- License: Apache 2.0
- Paper: https://arxiv.org/abs/2306.16636

### Citation

```bibtex
@misc{wei2023cmath,
      title={CMATH: Can Your Language Model Pass Chinese Elementary School Math Test?},
      author={Tianwen Wei and Jian Luan and Wei Liu and Shuang Dong and Bin Wang},
      year={2023},
      eprint={2306.16636},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

- `cmath`: All CMATH tasks (direct + cot)

#### Tasks

- `cmath_direct`: Direct answer (no chain-of-thought)
- `cmath_cot`: Answer with chain-of-thought reasoning in Chinese

### Checklist

* [x] Is the task an existing benchmark in the literature?
* [x] Have you referenced the original paper that introduced the task?
* [x] If available, does your implementation match the original evaluation protocol from the paper?
* [x] Does the task use a `validation` and `test` split?
* [x] Have you verified the task metrics against the original paper?
