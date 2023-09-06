# MGSM

### Paper

Title: `Language Models are Multilingual Chain-of-Thought Reasoners`

Abstract: https://arxiv.org/abs/2210.03057

Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems, proposed in the paper [Language models are multilingual chain-of-thought reasoners](http://arxiv.org/abs/2210.03057).

The same 250 problems from [GSM8K](https://arxiv.org/abs/2110.14168) are each translated via human annotators in 10 languages. The 10 languages are:
- Spanish
- French
- German
- Russian
- Chinese
- Japanese
- Thai
- Swahili
- Bengali
- Telugu

GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.

You can find the input and targets for each of the ten languages (and English) as `.tsv` files.
We also include few-shot exemplars that are also manually translated from each language in `exemplars.py`.

Homepage: https://github.com/google-research/url-nlp/tree/main/mgsm


### Citation

```
@misc{cobbe2021training,
    title={Training Verifiers to Solve Math Word Problems},
    author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
    year={2021},
    eprint={2110.14168},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
@misc{shi2022language,
    title={Language Models are Multilingual Chain-of-Thought Reasoners},
    author={Freda Shi and Mirac Suzgun and Markus Freitag and Xuezhi Wang and Suraj Srivats and Soroush Vosoughi and Hyung Won Chung and Yi Tay and Sebastian Ruder and Denny Zhou and Dipanjan Das and Jason Wei},
    year={2022},
    eprint={2210.03057},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* `mgsm_direct`: Direct question
  * `mgsm_direct_bn`: Bengali
  * `mgsm_direct_de`: German
  * `mgsm_direct_en`: English
  * `mgsm_direct_es`: Spanish
  * `mgsm_direct_fr`: French
  * `mgsm_direct_ja`: Japanese
  * `mgsm_direct_ru`: Russian
  * `mgsm_direct_sw`: Swahili
  * `mgsm_direct_te`: Telugu
  * `mgsm_direct_th`: Thai
  * `mgsm_direct_zh`: Chinese
* `mgsm_cot_native`: Question with Answer followed by CoT prompt in the same language as the dataset.
  * `mgsm_cot_native_bn`: Bengali
  * `mgsm_cot_native_de`: German
  * `mgsm_cot_native_en`: English
  * `mgsm_cot_native_es`: Spanish
  * `mgsm_cot_native_fr`: French
  * `mgsm_cot_native_ja`: Japanese
  * `mgsm_cot_native_ru`: Russian
  * `mgsm_cot_native_sw`: Swahili
  * `mgsm_cot_native_te`: Telugu
  * `mgsm_cot_native_th`: Thai
  * `mgsm_cot_native_zh`: Chinese

Examplar Samples: https://github.com/google-research/url-nlp/blob/main/mgsm/exemplars.py

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
