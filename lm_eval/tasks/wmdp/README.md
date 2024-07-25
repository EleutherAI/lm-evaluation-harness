# WMDP

### Paper

Title: `The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning`

Abstract: `https://arxiv.org/abs/2403.03218`

`The Weapons of Mass Destruction Proxy (WMDP) benchmark is a dataset of 4,157 multiple-choice questions surrounding hazardous knowledge in biosecurity cybersecurity, and chemical security. WMDP serves as both a proxy evaluation for hazardous knowledge in large language models (LLMs) and a benchmark for unlearning methods to remove such knowledge.`

Homepage: https://wmdp.ai


### Citation

```
@misc{li2024wmdp,
      title={The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning},
      author={Nathaniel Li and Alexander Pan and Anjali Gopal and Summer Yue and Daniel Berrios and Alice Gatti and Justin D. Li and Ann-Kathrin Dombrowski and Shashwat Goel and Long Phan and Gabriel Mukobi and Nathan Helm-Burger and Rassin Lababidi and Lennart Justen and Andrew B. Liu and Michael Chen and Isabelle Barrass and Oliver Zhang and Xiaoyuan Zhu and Rishub Tamirisa and Bhrugu Bharathi and Adam Khoja and Zhenqi Zhao and Ariel Herbert-Voss and Cort B. Breuer and Andy Zou and Mantas Mazeika and Zifan Wang and Palash Oswal and Weiran Liu and Adam A. Hunt and Justin Tienken-Harder and Kevin Y. Shih and Kemper Talley and John Guan and Russell Kaplan and Ian Steneker and David Campbell and Brad Jokubaitis and Alex Levinson and Jean Wang and William Qian and Kallol Krishna Karmakar and Steven Basart and Stephen Fitz and Mindy Levine and Ponnurangam Kumaraguru and Uday Tupakula and Vijay Varadharajan and Yan Shoshitaishvili and Jimmy Ba and Kevin M. Esvelt and Alexandr Wang and Dan Hendrycks},
      year={2024},
      eprint={2403.03218},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Groups and Tasks

#### Groups

* `wmdp`: All 4,157 multiple-choice questions in biosecurity, cybersecurity, and chemical security

#### Tasks

* `wmdp_bio`: 1,520 multiple-choice questions in biosecurity
* `wmdp_cyber`: 2,225 multiple-choice questions in cybersecurity
* `wmdp_chemistry`: 412 multiple-choice questions in chemical security

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
