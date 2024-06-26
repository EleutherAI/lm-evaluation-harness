# FLD

### Paper

Title: Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic

Abstract: https://arxiv.org/abs/2308.07336

**FLD** (**F**ormal **L**ogic **D**eduction) is a deductive reasoning benchmark.
Given a set of facts and a hypothesis, an LLM is required to generate (i) proof steps to (dis-)prove the hypothesis, and (ii) an answer ("proved", "disproved" or unknown").

Unique features of FLD are:
* It assesses the model's logical reasoning ability *isolated from knowledge*, as the facts are randomly constructed so that referring to existing knowledge never helps solve the task.
* It assesses diverse reasoning patterns (i.e., deduction rules), as it is based on formal logic theory.
* As a result, it is highly challenging. Indeed, even GPT-4 can solve only about half of the problems.

Homepage: https://github.com/hitachi-nlp/FLD


### Citation

```
@InProceedings{pmlr-v202-morishita23a,
  title = 	 {Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic},
  author =       {Morishita, Terufumi and Morio, Gaku and Yamaguchi, Atsuki and Sogawa, Yasuhiro},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {25254--25274},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/morishita23a/morishita23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/morishita23a.html},
}
```

### Groups and Tasks

This release is the simplified version of FLD where a model is required to predict only an answer.
This setting is described by "answer accuracy" in the original paper.


#### Tasks in Group `fld`
* `fld_default` is a basic task based on [FLD.v2](https://huggingface.co/datasets/hitachi-nlp/FLD.v2/viewer/star)
* `fld_star`: is a more challenging version based on [FLD.v2-star](https://huggingface.co/datasets/hitachi-nlp/FLD.v2/viewer/star)

The output format is `generate_until`, which requires a model to generate, rather than choose, a label from the specific set: "PROVED", "DISPROVED", or "UNKNOWN".
Therefor, if a model fails to follow this format, its answer is evaluated as incorrect.


#### Tasks in Group `fld_logical_formula`
Further, we have "logical formula" versions of the benchmarks, which evaluate LLMs' pure logical reasoning capabilities within the domain of logical formulas, rather than natural language:
* `fld_logical_formula_default`
* `fld_logical_formula_fld_star`


#### **(RECOMMENDED)** Tasks in Group `fld_mcq`
* `fld_mcq_default`
* `fld_mcq_star`

The multiple-choice question version of `fld` group.
We recommend using this variant to avoid the issue of models generating answers in unexpected formats, as stated above.


### **(RECOMMENDED)** Tasks in Group `fld_mcq_logical_formula`
The multiple-choice question version of `fld_logical_formula` group.
* `fld_mcq_logical_formula_default`
* `fld_mcq_logical_formula_star`
