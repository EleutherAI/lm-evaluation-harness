# Task-name

### Paper

Title: `It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning`
Abstract: `https://arxiv.org/abs/2106.12066`

Multilingual winograd schema challenge that includes English, French, Japanese, Portuguese, Russian and Chinese. Winograd schema challenges come from the XWinograd dataset introduced in Tikhonov et al. As it only contains 16 Chinese schemas, we add 488 Chinese schemas from clue/cluewsc2020.

Homepage: `https://huggingface.co/datasets/Muennighoff/xwinograd`


### Citation

```
@misc{muennighoff2022crosslingual,
      title={Crosslingual Generalization through Multitask Finetuning},
      author={Niklas Muennighoff and Thomas Wang and Lintang Sutawika and Adam Roberts and Stella Biderman and Teven Le Scao and M Saiful Bari and Sheng Shen and Zheng-Xin Yong and Hailey Schoelkopf and Xiangru Tang and Dragomir Radev and Alham Fikri Aji and Khalid Almubarak and Samuel Albanie and Zaid Alyafeai and Albert Webson and Edward Raff and Colin Raffel},
      year={2022},
      eprint={2211.01786},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{tikhonov2021heads,
    title={It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning},
    author={Alexey Tikhonov and Max Ryabinin},
    year={2021},
    eprint={2106.12066},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* `xwinograd`

#### Tasks

List or describe tasks defined in this folder, and their names here:
* `xwinograd_en`: Winograd schema challenges in English.
* `xwinograd_fr`: Winograd schema challenges in French.
* `xwinograd_jp`: Winograd schema challenges in Japanese.
* `xwinograd_pt`: Winograd schema challenges in Portuguese.
* `xwinograd_ru`: Winograd schema challenges in Russian.
* `xwinograd_zh`: Winograd schema challenges in Chinese.

### Checklist

For adding novel benchmarks/datasets to the library:
  * [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
