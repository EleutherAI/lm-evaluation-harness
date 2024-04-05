# TMMLU+

### Paper

Title: `An Improved Traditional Chinese Evaluation Suite for Foundation Model`

Abstract: `We present TMMLU+, a comprehensive dataset designed for the Traditional Chinese massive multitask language understanding dataset. TMMLU+ is a multiple-choice question-answering dataset with 66 subjects from elementary to professional level. Compared to its predecessor, TMMLU, TMMLU+ is six times larger and boasts a more balanced subject distribution. We included benchmark results in TMMLU+ from closed-source models and 24 open-weight Chinese large language models of parameters ranging from 1.8B to 72B. Our findings reveal that Traditional Chinese models still trail behind their Simplified Chinese counterparts. Additionally, current large language models have yet to outperform human performance in average scores. We publicly release our dataset and the corresponding benchmark source code.`


Homepage: [https://huggingface.co/datasets/ikala/tmmluplus](https://huggingface.co/datasets/ikala/tmmluplus)


### Citation

```
@article{ikala2024improved,
  title={An Improved Traditional Chinese Evaluation Suite for Foundation Model},
  author={Tam, Zhi-Rui and Pai, Ya-Ting and Lee, Yen-Wei and Cheng, Sega and Shuai, Hong-Han},
  journal={arXiv preprint arXiv:2403.01858},
  year={2024}
}
```

### Groups and Tasks

#### Groups

* `tmmluplus`: `The dataset comprises 22,690 multiple-choice questions from 66 subjects ranging from primary to professional level. `

#### Tasks

The following tasks evaluate subjects in the TMMLU+ dataset using loglikelihood-based multiple-choice scoring:

* `tmmluplus_{subject_english}`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
