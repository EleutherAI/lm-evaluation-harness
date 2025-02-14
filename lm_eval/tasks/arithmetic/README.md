# Arithmetic

### Paper

Title: `Language Models are Few-Shot Learners`
Abstract: https://arxiv.org/abs/2005.14165

A small battery of 10 tests that involve asking language models a simple arithmetic
problem in natural language.

Homepage: https://github.com/openai/gpt-3/tree/master/data


### Citation

```
@inproceedings{NEURIPS2020_1457c0d6,
    author = {Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and Agarwal, Sandhini and Herbert-Voss, Ariel and Krueger, Gretchen and Henighan, Tom and Child, Rewon and Ramesh, Aditya and Ziegler, Daniel and Wu, Jeffrey and Winter, Clemens and Hesse, Chris and Chen, Mark and Sigler, Eric and Litwin, Mateusz and Gray, Scott and Chess, Benjamin and Clark, Jack and Berner, Christopher and McCandlish, Sam and Radford, Alec and Sutskever, Ilya and Amodei, Dario},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
    pages = {1877--1901},
    publisher = {Curran Associates, Inc.},
    title = {Language Models are Few-Shot Learners},
    url = {https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf},
    volume = {33},
    year = {2020}
}
```

### Groups, Tags, and Tasks

#### Tags

* `arithmetic`: Evaluates `1dc` to `5ds`

#### Tasks

* `arithmetic_1dc`
* `arithmetic_2da`
* `arithmetic_2dm`
* `arithmetic_2ds`
* `arithmetic_3da`
* `arithmetic_3ds`
* `arithmetic_4da`
* `arithmetic_4ds`
* `arithmetic_5da`
* `arithmetic_5ds`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
version 2.0: (2025-Feb-14) set target delimiter to "" as the targets already start with a space.
