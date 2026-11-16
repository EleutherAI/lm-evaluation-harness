# SuperGLUE

### Paper

Title: `SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems`
Abstract: `https://w4ngatang.github.io/static/papers/superglue.pdf`

SuperGLUE is a benchmark styled after GLUE with a new set of more difficult language
understanding tasks.

Homepage: https://super.gluebenchmark.com/

### Citation

```
@inproceedings{NEURIPS2019_4496bf24,
    author = {Wang, Alex and Pruksachatkun, Yada and Nangia, Nikita and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
    pages = {},
    publisher = {Curran Associates, Inc.},
    title = {SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems},
    url = {https://proceedings.neurips.cc/paper/2019/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf},
    volume = {32},
    year = {2019}
}
```

### Groups, Tags, and Tasks

#### Groups

None.

#### Tags

* `super-glue-lm-eval-v1`: SuperGLUE eval adapted from LM Eval V1
* `super-glue-t5-prompt`: SuperGLUE prompt and evaluation that matches the T5 paper (if using accelerate, will error if record is included.)

#### Tasks

Comparison between validation split score on T5x and LM-Eval (T5x models converted to HF)
| T5V1.1 Base | SGLUE | BoolQ | CB        | Copa | MultiRC | ReCoRD | RTE | WiC | WSC |
| ----------- | ------| ----- | --------- | ---- | ------- | ------ | --- | --- | --- |
| T5x | 69.47 | 78.47(acc) | 83.93(f1) 87.5(acc) | 50(acc) | 73.81(f1) 33.26(em) | 70.09(em) 71.34(f1) | 78.7(acc) | 63.64(acc) | 75(acc) |
| LM-Eval | 71.35 | 79.36(acc) | 83.63(f1) 87.5(acc) | 63(acc) | 73.45(f1) 33.26(em) | 69.85(em) 68.86(f1) | 78.34(acc) | 65.83(acc) | 75.96(acc) |



* `super-glue-lm-eval-v1`
    -  `boolq`
    - `cb`
    - `copa`
    - `multirc`
    - `record`
    - `rte`
    - `wic`
    - `wsc`

* `super-glue-t5-prompt`
    - `super_glue-boolq-t5-prompt`
    - `super_glue-cb-t5-prompt`
    - `super_glue-copa-t5-prompt`
    - `super_glue-multirc-t5-prompt`
    - `super_glue-record-t5-prompt`
    - `super_glue-rte-t5-prompt`
    - `super_glue-wic-t5-prompt`
    - `super_glue-wsc-t5-prompt`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
