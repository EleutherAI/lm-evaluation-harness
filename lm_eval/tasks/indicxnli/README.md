# IndicXNLI (Gujarati)

### Paper

Title: `IndicXNLI: Evaluating Multilingual Inference for Indian Languages`

Abstract: https://arxiv.org/abs/2204.08776

INDICXNLI extends the XNLI natural language inference benchmark to eleven major
Indic languages via machine translation, verified against the original English
XNLI dataset. This task adds evaluation for **Gujarati**, which is not covered
by the original `xnli` task group in this repository (`facebook/xnli` does not
include Gujarati).

Homepage: https://huggingface.co/datasets/Divyanshu/indicxnli

### Citation

@misc{https://doi.org/10.48550/arxiv.2204.08776,
  doi = {10.48550/ARXIV.2204.08776},
  url = {https://arxiv.org/abs/2204.08776},
  author = {Aggarwal, Divyanshu and Gupta, Vivek and Kunchukuttan, Anoop},
  title = {IndicXNLI: Evaluating Multilingual Inference for Indian Languages},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

### Groups, Tags, and Tasks

#### Tasks

* `indicxnli_gu`: Gujarati natural language inference (entailment/neutral/contradiction), scored via loglikelihood over the three label continuations, following the same format as the existing `xnli` task family.

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] Have you checked if there is a reference implementation, and matched its setup where applicable? The scoring format matches the existing xnli_* tasks in this repo, the closest available reference implementation.
* [x] Is this a new variant of an existing task?
  * This is not part of the existing xnli group, since that group is built on facebook/xnli, which does not include Gujarati. This task instead uses Divyanshu/indicxnli, the dataset from the IndicXNLI paper, mirroring the xnli task structure for consistency.
