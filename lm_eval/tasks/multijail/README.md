# Multijail

## Paper

Title: Multilingual Jailbreak Challenges in Large Language Models

Abstract: https://arxiv.org/abs/2310.06474

Homepage: https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail

315 English unsafe prompts annotated into nine non-English languages.

The languages were categorized based on resource availability, as shown below:

- High-resource languages: Chinese (zh), Italian (it), Vietnamese (vi)
- Medium-resource languages: Arabic (ar), Korean (ko), Thai (th)
- Low-resource languages: Bengali (bn), Swahili (sw), Javanese (jv)

### Citation

```text
@misc{deng2023multilingual,
title={Multilingual Jailbreak Challenges in Large Language Models},
author={Yue Deng and Wenxuan Zhang and Sinno Jialin Pan and Lidong Bing},
year={2023},
eprint={2310.06474},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
```

### Groups, Tags, and Tasks

#### Groups

* `multijail`: `All languages`

#### Tasks

* `multijail_en`: `English`
* `multijail_ar`: `Arabic`
* `multijail_ko`: `Korean`
* `multijail_th`: `Thai`
* `multijail_bn`: `Bengali`
* `multijail_sw`: `Swahili`
* `multijail_jv`: `Javanese`
* `multijail_it`: `Italian`
* `multijail_vi`: `Vietnamese`

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
