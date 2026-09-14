# EconLogicQA

## Paper

Title: `EconLogicQA: A Question-Answering Benchmark for Evaluating Large Language Models in Economic Sequential Reasoning`

Abstract: https://arxiv.org/abs/2405.07938

EconLogicQA asks a model to order four interconnected events drawn from a real
economic, business, or supply-chain narrative. Unlike benchmarks that ask for
the *next* event, it requires the model to sequence all four at once, and the
ordering is **logical rather than merely chronological** — the model has to
reason about what causes or enables what. Questions were generated from business
news articles with GPT-4 and then filtered by human review.

Homepage: https://huggingface.co/datasets/yinzhu-quan/econ_logic_qa

## Citation

```bibtex
@misc{quan2024econlogicqa,
      title={EconLogicQA: A Question-Answering Benchmark for Evaluating Large Language Models in Economic Sequential Reasoning},
      author={Yinzhu Quan and Zefang Liu},
      year={2024},
      eprint={2405.07938},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Groups and Tasks

### Tasks

- `econ_logic_qa`: order four economic events by logical precedence. 130 test
  questions; the 390-question train split supplies the few-shot examples.

## Evaluation setup

The task follows the paper's setup: greedy decoding, the permutation extracted
from the generation with a regular expression, and accuracy over exact matches
of that permutation against the gold order.

The default is **1-shot**, the paper's best-performing setting. The paper
deliberately excludes 0-shot ("the results are unsatisfactory due to the task's
complexity") and recommends a few-shot approach for sorting problems, so the
default here is 1 rather than 0. To reproduce the paper's other column:

```bash
lm_eval --model hf --model_args pretrained=<model> --tasks econ_logic_qa --num_fewshot 5
```

### Metrics

| Metric | Description | Chance |
|---|---|---|
| `exact_match` | The extracted permutation equals the gold order. This is the paper's accuracy. | 1/24 ≈ 4.2% |
| `pairwise_accuracy` | Fraction of the six event pairs placed in the correct relative order. A generation with no recoverable permutation scores 0. | 50% |

`pairwise_accuracy` is not in the paper; it is reported alongside because
`exact_match` cannot distinguish a model that ordered the events wrongly from
one that never emitted an ordering at all. That distinction matters on this
benchmark: several models in the paper's Table 2 score *below* the 4.2% random-
permutation floor (Llama-2-7B at 0.77%, Gemma-1.1-7B-IT at 0.77%,
Gemma-7B-IT at 2.31%), which is a signature of failing the output format rather
than of reasoning. A model that cannot produce a permutation scores 0 on
`pairwise_accuracy`; one that produces a wrong ordering scores near 0.5.

### Reference results

Accuracy (`exact_match`) reported in Table 2 of the paper:

| Model | 1-shot | 5-shot |
|---|---|---|
| GPT-4-Turbo | 56.92% | 56.15% |
| GPT-4 | 55.38% | 53.85% |
| GPT-3.5-Turbo | 37.69% | 38.46% |
| Llama-3-8B-Instruct | 34.62% | 37.69% |
| Mistral-7B-Instruct-v0.2 | 31.54% | 32.31% |
| Llama-3-8B | 23.85% | 23.85% |
| Llama-2-7B | 0.77% | 1.54% |

The paper states it used lm-evaluation-harness with a YAML config for the open
models, but no configuration or reference implementation was released, so the
prompt template here is reconstructed from the dataset's own answer format and
the setup described in Section 4.1. Absolute numbers are therefore expected to
track the paper's ranking rather than match it exactly.

### Results from this implementation

`Qwen/Qwen2.5-0.5B-Instruct`, full 130-question test split, greedy, `dtype=float32`:

| n-shot | `exact_match` | `pairwise_accuracy` | permutation recovered |
|---|---|---|---|
| 1 | 6.92% ± 2.23 | 54.74% ± 2.19 | 128/130 |
| 5 | 4.62% ± 1.85 | 54.87% ± 2.06 | 129/130 |

```bash
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=float32 \
    --tasks econ_logic_qa --num_fewshot 1 --batch_size 8
```

A 0.5B model lands at roughly the 4.2% random-permutation floor on
`exact_match`, below every model in the paper's table except the Llama-2 and
Gemma runs. The two metrics together say why: it emits a well-formed
permutation on 98% of questions but orders the pairs barely better than chance,
so this is an absence of sequencing ability rather than a format failure.

## Checklist

For adding novel benchmarks/datasets to the library:

- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task?
  - [x] If yes, does the original paper provide a reference implementation? **No.**
        The paper reports using lm-evaluation-harness but did not release its
        YAML or any evaluation code, and the dataset repository contains only
        the CSV splits. The setup here reproduces Section 4.1 as described
        (greedy decoding, regex extraction, exact match, few-shot) and the
        published numbers are quoted above for comparison.

If other tasks on this dataset are already supported:

- [x] Is the "Main" variant of this task clearly denoted? — `econ_logic_qa` is
      the only variant.
- [x] Have you provided a short sentence in a README on what each new variant
      adds / evaluates?
- [x] Have you noted which, if any, published evaluation setups are matched by
      this variant? — See "Evaluation setup" above.
