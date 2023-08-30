# AGIEval

## Paper
AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models

https://arxiv.org/abs/2304.06364

AGIEval is a human-centric benchmark specifically designed to evaluate the general abilities of foundation models in tasks pertinent to human cognition and problem-solving. This benchmark is derived from 20 official, public, and high-standard admission and qualification exams intended for general human test-takers, such as general college admission tests (e.g., Chinese College Entrance Exam (Gaokao) and American SAT), law school admission tests, math competitions, lawyer qualification tests, and national civil service exams.

NOTE: Only the english datasets are added for now.

Homepage: https://github.com/microsoft/AGIEval


## Citation
```
@misc{zhong2023agieval,
      title={AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models},
      author={Wanjun Zhong and Ruixiang Cui and Yiduo Guo and Yaobo Liang and Shuai Lu and Yanlin Wang and Amin Saied and Weizhu Chen and Nan Duan},
      year={2023},
      eprint={2304.06364},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
Citations for individual datasets:
```
@inproceedings{ling-etal-2017-program,
    title = "Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems",
    author = "Ling, Wang  and
      Yogatama, Dani  and
      Dyer, Chris  and
      Blunsom, Phil",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1015",
    doi = "10.18653/v1/P17-1015",
    pages = "158--167",
    abstract = "Solving algebraic word problems requires executing a series of arithmetic operations{---}a program{---}to obtain a final answer. However, since programs can be arbitrarily complicated, inducing them directly from question-answer pairs is a formidable challenge. To make this task more feasible, we solve these problems by generating answer rationales, sequences of natural language and human-readable mathematical expressions that derive the final answer through a series of small steps. Although rationales do not explicitly specify programs, they provide a scaffolding for their structure via intermediate milestones. To evaluate our approach, we have created a new 100,000-sample dataset of questions, answers and rationales. Experimental results show that indirect supervision of program learning via answer rationales is a promising strategy for inducing arithmetic programs.",
}

@inproceedings{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}

@inproceedings{Liu2020LogiQAAC,
  title={LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning},
  author={Jian Liu and Leyang Cui and Hanmeng Liu and Dandan Huang and Yile Wang and Yue Zhang},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2020}
}

@inproceedings{zhong2019jec,
  title={JEC-QA: A Legal-Domain Question Answering Dataset},
  author={Zhong, Haoxi and Xiao, Chaojun and Tu, Cunchao and Zhang, Tianyang and Liu, Zhiyuan and Sun, Maosong},
  booktitle={Proceedings of AAAI},
  year={2020},
}

@article{Wang2021FromLT,
  title={From LSAT: The Progress and Challenges of Complex Reasoning},
  author={Siyuan Wang and Zhongkun Liu and Wanjun Zhong and Ming Zhou and Zhongyu Wei and Zhumin Chen and Nan Duan},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2021},
  volume={30},
  pages={2201-2216}
}
```
### Groups and Tasks

#### Benchmarks
- agieval

#### Groups

- `multiple_choice`

#### Tasks
All tasks have been standardized to a common format.

- `agieval_aquarat`: 254 test examples; 5 few-shot
- `agieval_logiqa`: 651 test examples; 3 few-shot
- `agieval_lsatar`: 230 test examples; 3 few-shot
- `agieval_lsatlr`: 510 test examples; 3 few-shot
- `agieval_lsatrc`: 269 test examples; 3 few-shot
- `agieval_math`: 1000 test examples; 4 few-shot (not included in benchmark)
- `agieval_saten`: 206 test examples; 3 few-shot
- `agieval_saten_wop`: 206 test examples; 3 few-shot
- `agieval_satm`: 220 test examples; 5 few-shot

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * Note: I implemented the `few-shot` variant from the original repo. The repo also had `zero-shot` and `few-shot-COT` variants.


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
  * The few-shot variant is implemented for now.
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
  * The mean aggregate accuracy of the benchmark is quite close to the [LLama 2 paper](https://arxiv.org/abs/2307.09288). However, there is alot of variance in the accuracy for the individual tasks.
