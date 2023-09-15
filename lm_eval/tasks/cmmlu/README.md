# CMMLU

### Paper

CMMLU: Measuring massive multitask language understanding in Chinese
https://arxiv.org/abs/2306.09212

CMMLU is a comprehensive evaluation benchmark specifically designed to evaluate the knowledge and reasoning abilities of LLMs within the context of Chinese language and culture.
CMMLU covers a wide range of subjects, comprising 67 topics that span from elementary to advanced professional levels.

Homepage: https://github.com/haonan-li/CMMLU

### Citation

```bibtex
@misc{li2023cmmlu,
      title={CMMLU: Measuring massive multitask language understanding in Chinese},
      author={Haonan Li and Yixuan Zhang and Fajri Koto and Yifei Yang and Hai Zhao and Yeyun Gong and Nan Duan and Timothy Baldwin},
      year={2023},
      eprint={2306.09212},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

- `cmmlu`: All 67 subjects of the CMMLU dataset, evaluated following the methodology in MMLU's original implementation.

#### Tasks


The following tasks evaluate subjects in the CMMLU dataset using loglikelihood-based multiple-choice scoring:
- `cmmlu_{subject_english}`

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [x] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
