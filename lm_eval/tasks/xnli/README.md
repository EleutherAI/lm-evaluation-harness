# XNLI

### Paper

Title: `XNLI: Evaluating Cross-lingual Sentence Representations`

Abstract: https://arxiv.org/abs/1809.05053

Based on the implementation of @yongzx (see https://github.com/EleutherAI/lm-evaluation-harness/pull/258)

Prompt format (same as XGLM and mGPT):

sentence1 + ", right? " + mask = (Yes|Also|No) + ", " + sentence2

Predicition is the full sequence with the highest likelihood.

Language specific prompts are translated word-by-word with Google Translate
and may differ from the ones used by mGPT and XGLM (they do not provide their prompts).

Homepage: https://github.com/facebookresearch/XNLI


### Citation

"""
@InProceedings{conneau2018xnli,
  author = "Conneau, Alexis
        and Rinott, Ruty
        and Lample, Guillaume
        and Williams, Adina
        and Bowman, Samuel R.
        and Schwenk, Holger
        and Stoyanov, Veselin",
  title = "XNLI: Evaluating Cross-lingual Sentence Representations",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  location = "Brussels, Belgium",
}
"""

### Groups and Tasks

#### Groups

* `xnli`

#### Tasks

* `xnli_ar`: Arabic
* `xnli_bg`: Bulgarian
* `xnli_de`: German
* `xnli_el`: Greek
* `xnli_en`: English
* `xnli_es`: Spanish
* `xnli_fr`: French
* `xnli_hi`: Hindi
* `xnli_ru`: Russian
* `xnli_sw`: Swahili
* `xnli_th`: Thai
* `xnli_tr`: Turkish
* `xnli_ur`: Urdu
* `xnli_vi`: Vietnamese
* `xnli_zh`: Chinese

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
