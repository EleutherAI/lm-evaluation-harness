# SLR-Bench

SLR-Bench is a benchmark for scalable logical reasoning with language models. The tasks require generating Prolog rules that correctly classify trains based on their compositions.

### Paper

Title: `SLR: Automated Synthesis for Scalable Logical Reasoning`

Abstract: [https://arxiv.org/abs/2506.15787](https://arxiv.org/abs/2506.15787)

### Dataset

The complete dataset can be found at:
[https://huggingface.co/datasets/AIML-TUDA/SLR-Bench](https://huggingface.co/datasets/AIML-TUDA/SLR-Bench)

### Verifier

- Generated rules are evaluated using a symbolic verifier (`AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning`).

###  Automated Synthesis Framework to Generate new Tasks

[https://github.com/ml-research/ScalableLogicalReasoning](https://github.com/ml-research/ScalableLogicalReasoning)

## Prerequisites

- **SWI-Prolog**: Required for evaluating generated rules. Make sure `swipl` is installed and available in your system PATH.

#### Tasks

The following variants are available:

- `slr_bench_all` &mdash; Full dataset (`v1-All`)
- `slr_bench_basic` &mdash; Basic subset (`v1-Basic`)
- `slr_bench_easy` &mdash; Easy subset (`v1-Easy`)
- `slr_bench_medium` &mdash; Medium subset (`v1-Medium`)
- `slr_bench_hard` &mdash; Hard subset (`v1-Hard`)


## Citation
```
@misc{helff2025slrautomatedsynthesisframework,
      title={SLR: An Automated Synthesis Framework for Scalable Logical Reasoning},
      author={Lukas Helff and Ahmad Omar and Felix Friedrich and Wolfgang Stammer and Antonia Wüst and Tim Woydt and Rupert Mitchell and Patrick Schramowski and Kristian Kersting},
      year={2025},
      eprint={2506.15787},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.15787},
}
```
