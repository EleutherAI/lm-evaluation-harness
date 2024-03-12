# KorMedMCQA

### Paper

Title: `KorMedMCQA: Multi-Choice Question Answering Benchmark for Korean Healthcare Professional Licensing Examinations`

Abstract: `We introduce KorMedMCQA, the first Korean multiple-choice question answering (MCQA) benchmark derived from Korean healthcare professional licensing examinations, covering from the year 2012 to year 2023. This dataset consists of a selection of questions from the license examinations for doctors, nurses, and pharmacists, featuring a diverse array of subjects. We conduct baseline experiments on various large language models, including proprietary/open-source, multilingual/Korean-additional pretrained, and clinical context pretrained models, highlighting the potential for further enhancements. We make our data publicly available on HuggingFace and provide a evaluation script via LM-Harness, inviting further exploration and advancement in Korean healthcare environments.`


Paper : https://arxiv.org/abs/2403.01469

Homepage: https://huggingface.co/datasets/sean0042/KorMedMCQA


### Citation

```
@article{kweon2024kormedmcqa,
      title={KorMedMCQA: Multi-Choice Question Answering Benchmark for Korean Healthcare Professional Licensing Examinations},
      author={Sunjun Kweon and Byungjin Choi and Minkyu Kim and Rae Woong Park and Edward Choi},
      journal={arXiv preprint arXiv:2403.01469},
      year={2024}
}
```

### Groups and Tasks

* `kormedmcqa`: Runs `kormedmcqa_doctor`, `kormedmcqa_nurse`, and `kormedmcqa_pharm`.

#### Tasks

* `kormedmcqa_doctor`: `Official Korean Doctor Examination`
* `kormedmcqa_nurse`: `Official Korean Nurse Examination`
* `kormedmcqa_pharm`: `Official Korean Pharmacist Examination`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
