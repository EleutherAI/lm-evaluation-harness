# SAS-Bench

Paper: [SAS-Bench: A Fine-Grained Benchmark for Evaluating Short Answer
Scoring with Large Language Models](https://arxiv.org/abs/2505.07247)
(Lai et al., Peking University, 2025)

Dataset: `aleversn/SAS-Bench`

## What it tests

Given a physics exam question, a reference answer, and a student's
step-by-step response, the model must assign an integer score matching
expert human annotation.

- 4,109 student responses across 1,030 questions
- Expert-annotated step-wise scores and error labels
- Domain: Chinese high school / undergraduate Physics
- Only a `test` split is available

## Metric

Primary: **QWK** (quadratic weighted kappa) between the model-predicted
score and the human `manual_label`. Scoring is ordinal, so QWK — the
standard metric for short-answer scoring — is more appropriate than exact
agreement. Exact match is also reported as a secondary metric.

**Known limitation:** QWK is currently computed over raw scores pooled
across questions with differing maximum scores. Scale-normalized or
per-question QWK is left as future work.

## Run it

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-70m \
    --tasks sas_bench \
    --limit 10
```

## Citation

```bibtex
@article{lai2025sasbench,
  title={SAS-Bench: A Fine-Grained Benchmark for Evaluating Short Answer
         Scoring with Large Language Models},
  author={Peichao Lai and others},
  year={2025},
  journal={arXiv preprint arXiv:2505.07247}
}
```