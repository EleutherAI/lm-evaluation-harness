# Evalita-LLM

### Paper

Evalita-LLM, a new benchmark designed to evaluate Large Language
Models (LLMs) on Italian tasks. The distinguishing and innovative features of
Evalita-LLM are the following: (i) all tasks are native Italian, avoiding issues of
translating from Italian and potential cultural biases; (ii) in addition to well established multiple-choice tasks, the benchmark includes generative tasks, enabling more natural interaction with LLMs; (iii) all tasks are evaluated against multiple prompts, this way mitigating the model sensitivity to specific prompts and allowing a fairer and objective evaluation.

### Citation

```bibtex
@misc{magnini2025evalitallmbenchmarkinglargelanguage,
      title={Evalita-LLM: Benchmarking Large Language Models on Italian},
      author={Bernardo Magnini and Roberto Zanoli and Michele Resta and Martin Cimmino and Paolo Albano and Marco Madeddu and Viviana Patti},
      year={2025},
      eprint={2502.02289},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.02289},
}
```

### Groups

- `evalita-mp`: All tasks (perplexity and non-perplexity based).
- `evalita-mp_gen`: Only generative tasks.
- `evalita-mp_mc`: Only perplexity-based tasks.

#### Tasks

The following Evalita-LLM tasks can also be evaluated in isolation:
  - `evalita-mp_te`: Textual Entailment
  - `evalita-mp_sa`: Sentiment Analysis
  - `evalita-mp_wic`: Word in Context
  - `evalita-mp_hs`: Hate Speech Detection
  - `evalita-mp_at`: Admission Tests
  - `evalita-mp_faq`: FAQ
  - `evalita-mp_sum_fp`:  Summarization
  - `evalita-mp_ls`: Lexical Substitution
  - `evalita-mp_ner_group`: Named Entity Recognition
  - `evalita-mp_re`: Relation Extraction


### Usage

```bash

lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf --tasks evalita-mp --device cuda:0 --batch_size auto
```

### Changelog

- **v0.1**: Renamed `evalita-mp_sum_fp` small variant group to `evalita-mp_sum_fp_small` to resolve duplicate group name conflict with the full variant.

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [x] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
