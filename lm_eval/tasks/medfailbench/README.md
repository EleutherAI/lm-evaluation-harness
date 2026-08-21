# MedFailBench

MedFailBench is a clinician-authored medical AI safety benchmark for evaluating unsafe clinical response patterns in LLM outputs.

This task uses the public Hugging Face prompt dataset:

- `goktugozkanmd/medfailbench-v02-prompts`

The upstream benchmark repository and result artifacts are available at:

- GitHub: https://github.com/goktugozkanmd/medical-ai-failure-atlas
- Results dataset: https://huggingface.co/datasets/goktugozkanmd/medfailbench-v02-results
- DOI: https://doi.org/10.5281/zenodo.21205535

The native MedFailBench workflow uses SafetyGuard post-processing for safety, accuracy, source transparency, refusal appropriateness, and clinical grounding. The lm-evaluation-harness task exposes the prompt set as a zero-shot `generate_until` task so model samples can be reproduced and scored downstream.
