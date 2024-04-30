# DecodingTrust

### Paper

DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models

arxiv link: https://arxiv.org/pdf/2306.11698.pdf

This repo contains the source code of DecodingTrust. This research endeavor is designed to help researchers better understand the capabilities, limitations, and potential risks associated with deploying these state-of-the-art Large Language Models (LLMs). See our paper for details.

Homepage: https://decodingtrust.github.io

### Citation

```
@article{wang2023decodingtrust,
  title={DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models},
  author={Wang, Boxin and Chen, Weixin and Pei, Hengzhi and Xie, Chulin and Kang, Mintong and Zhang, Chenhui and Xu, Chejian and Xiong, Zidi and Dutta, Ritik and Schaeffer, Rylan and others},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```

### Groups and Tasks

#### Groups

- `decoding_trust`
- `dt_adv_glue_plus_plus`
- `dt_adv_demonstrations`
- `dt_fairness`
- `dt_ood`
- `dt_privacy`
- `dt_stereotype`
- `dt_toxicity`

TODOs

- add [privacy understanding](https://github.com/AI-secure/DecodingTrust/blob/30839b483f71a35eb8bf36cda443d0d941a5095e/src/dt/perspectives/privacy/utils.py#L261) (constructs scenarios where you're told a secret and asked to divulge it)
- add [pii messages](https://github.com/AI-secure/DecodingTrust/blob/30839b483f71a35eb8bf36cda443d0d941a5095e/src/dt/perspectives/privacy/utils.py#L169) (asks to get SSN given it in the prompt)

#### Tasks

* TODO fill this in

Note this skips the `machine_ethics` portion of decoding trust.  If you wish to run it run it from `hendrycks_ethics.` Might add that task as a subtask later.

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
  * [ ] Matches v0.3.0 of Eval Harness
