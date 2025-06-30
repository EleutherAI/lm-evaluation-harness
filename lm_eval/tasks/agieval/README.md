# AGIEval

### Paper

Title: AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models

Abstract: https://arxiv.org/abs/2304.06364.pdf

AGIEval is a human-centric benchmark specifically designed to evaluate the general abilities of foundation models in tasks pertinent to human cognition and problem-solving.
This benchmark is derived from 20 official, public, and high-standard admission and qualification exams intended for general human test-takers, such as general college admission tests (e.g., Chinese College Entrance Exam (Gaokao) and American SAT), law school admission tests, math competitions, lawyer qualification tests, and national civil service exams.

Homepage: https://github.com/ruixiangcui/AGIEval

### Citation

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

Please make sure to cite all the individual datasets in your paper when you use them. We provide the relevant citation information below:

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

### Groups, Tags, and Tasks

#### Groups

- `agieval`: Evaluates all tasks listed below.

- `agieval_en`: Evaluates all English subtasks: `agieval_aqua_rat`, `agieval_gaokao_english`, `agieval_logiqa_en`, `agieval_lsat_*`, `agieval_sat_*`, `agieval_math`

- `agieval_cn`: Evaluates all Chinese subtasks:
`agieval_gaokao_biology`, `agieval_gaokao_chemistry`, `agieval_gaokao_chinese`, `agieval_gaokao_geography`,
`agieval_gaokao_history`, `agieval_gaokao_mathqa`, `agieval_gaokao_mathcloze`, `agieval_gaokao_physics`, `agieval_jec_qa_ca`, `agieval_jec_qa_kd`, `agieval_logiqa_zh`

- `agieval_nous`: Evaluates a specific subset of AGIEval tasks (multiple-choice and english-only), namely those in https://github.com/teknium1/LLM-Benchmark-Logs/blob/main/benchmark-logs/Mistral-7B-Base.md

#### Tags

None.

#### Tasks

- `agieval_aqua_rat`
- `agieval_gaokao_biology`
- `agieval_gaokao_chemistry`
- `agieval_gaokao_chinese`
- `agieval_gaokao_english`
- `agieval_gaokao_geography`
- `agieval_gaokao_history`
- `agieval_gaokao_mathqa`
- `agieval_gaokao_mathcloze`
- `agieval_gaokao_physics`
- `agieval_jec_qa_ca`
- `agieval_jec_qa_kd`
- `agieval_logiqa_en`
- `agieval_logiqa_zh`
- `agieval_lsat_ar`
- `agieval_lsat_lr`
- `agieval_lsat_rc`
- `agieval_sat_en`
- `agieval_sat_en_without_passage`
- `agieval_sat_math`
- `agieval_math`
