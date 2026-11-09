# Task-name

### Paper

Title: `CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation`

Abstract: https://arxiv.org/abs/2102.04664

CodeXGLUE provides benchmark datasets for multiple code understanding and generation tasks, including generating docstrings in natural language from code snippets (code2text).

### Citation

```
@inproceedings{DBLP:conf/nips/LuGRHSBCDJTLZSZ21,
  author       = {Shuai Lu and
                  Daya Guo and
                  Shuo Ren and
                  Junjie Huang and
                  Alexey Svyatkovskiy and
                  Ambrosio Blanco and
                  Colin B. Clement and
                  Dawn Drain and
                  Daxin Jiang and
                  Duyu Tang and
                  Ge Li and
                  Lidong Zhou and
                  Linjun Shou and
                  Long Zhou and
                  Michele Tufano and
                  Ming Gong and
                  Ming Zhou and
                  Nan Duan and
                  Neel Sundaresan and
                  Shao Kun Deng and
                  Shengyu Fu and
                  Shujie Liu},
  editor       = {Joaquin Vanschoren and
                  Sai{-}Kit Yeung},
  title        = {CodeXGLUE: {A} Machine Learning Benchmark Dataset for Code Understanding
                  and Generation},
  booktitle    = {Proceedings of the Neural Information Processing Systems Track on
                  Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December
                  2021, virtual},
  year         = {2021},
  url          = {https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c16a5320fa475530d9583c34fd356ef5-Abstract-round1.html},
  timestamp    = {Thu, 19 Dec 2024 22:07:31 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/LuGRHSBCDJTLZSZ21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

### Groups and Tasks

#### Groups

* code2text

#### Tasks

* `code2text_go`: Generate docstring in natural language from Go code snippets.
* `code2text_java`: Generate docstring in natural language from Java code snippets.
* `code2text_javascript`: Generate docstring in natural language from JavaScript code snippets.
* `code2text_php`: Generate docstring in natural language from PHP code snippets.
* `code2text_python`: Generate docstring in natural language from Python code snippets.
* `code2text_ruby`: Generate docstring in natural language from Ruby code snippets.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
