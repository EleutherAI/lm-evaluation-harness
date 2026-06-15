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

## Metric
Exact match between model-predicted score and human `manual_label`.

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