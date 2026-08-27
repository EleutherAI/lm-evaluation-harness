# ZhoBLiMP: A Systematic Assessment of Language Models with Linguistic Minimal Pairs in Chinese

## Paper

Title: `A Systematic Assessment of Language Models with Linguistic Minimal Pairs in Chinese`

Paper: https://arxiv.org/pdf/2411.06096

> Whether and how language models (LMs) acquire the syntax of natural languages has been widely evaluated under the minimal pair paradigm. However, a lack of wide-coverage benchmarks in languages other than English has constrained systematic investigations into the issue. Addressing it, we first introduce ZhoBLiMP, the most comprehensive benchmark of linguistic minimal pairs for Chinese to date, with 118 paradigms, covering 15 linguistic phenomena.

Homepage: https://github.com/sjtu-compling/ZhoBLiMP

### Citation

```
@article{liu2024zhoblimp,
  title={Zhoblimp: a systematic assessment of language models with linguistic minimal pairs in chinese},
  author={Liu, Yikang and Shen, Yeting and Zhu, Hongao and Xu, Lilong and Qian, Zhiheng and Song, Siyuan and Zhang, Kejia and Tang, Jialong and Zhang, Pei and Yang, Baosong and others},
  journal={arXiv preprint arXiv:2411.06096},
  year={2024}
}
```

### Groups, Tags, and Tasks

* `zhoblimp`: Runs all ZhoBLiMP subtasks and calculates mean performance.

#### Implementation notes

* **Length normalization:** The [original implementation](https://github.com/sjtu-compling/ZhoBLiMP) normalizes sentence length using a custom function which is not supported by the Language Model Evaluation Harness. For this reason, the implementation provided here includes both un-normalized accuracy (`acc`) and byte-length-normalized accuracy (`acc_norm`).

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

### Changelog
