# PerspectiveGap

PerspectiveGap ([arXiv:2606.08878](https://arxiv.org/abs/2606.08878)) evaluates LLMs' ability to compose orchestration prompts for multi-agent systems, testing whether a model can decide what each sub-agent needs to know without leaking irrelevant context.

- Dataset: [sun1245/PerspectiveGap](https://huggingface.co/datasets/sun1245/PerspectiveGap)
- 220 evaluations (110 scenarios x 2 shuffle seeds)
- Scoring: [WhymustIhaveaname/PerspectiveGap](https://github.com/WhymustIhaveaname/PerspectiveGap)

## Tasks

- `perspective_gap_role_assignment` — output a JSON object mapping roles to fragment IDs
- `perspective_gap_prompt_writing` — write one markdown prompt per role with relevant fragments verbatim

## Metrics

| Metric | Description |
|--------|-------------|
| strict_pass | 1 if all roles correct, 0 otherwise |
| net_match_score | (TP - FP - FN) / expected, clipped to [0, 1] |
| required_coverage | TP / (TP + FN) |
| boundary_precision | TP / (TP + FP) |
| distractor_leakage | times the distractor was incorrectly included |

## Dependencies

Requires the `perspective-gap` package for scoring:

```bash
pip install git+https://github.com/WhymustIhaveaname/PerspectiveGap.git
```
