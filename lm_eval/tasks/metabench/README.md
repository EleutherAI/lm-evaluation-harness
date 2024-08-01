# Metabench

### Paper

Title: `metabench` -- A Sparse Benchmark to Measure General Ability in Large Language Models

Abstract: https://arxiv.org/abs/2407.12844

Large Language Models (LLMs) vary in their abilities on a range of tasks. Initiatives such as the 𝙾𝚙𝚎𝚗 𝙻𝙻𝙼 𝙻𝚎𝚊𝚍𝚎𝚛𝚋𝚘𝚊𝚛𝚍 aim to quantify these differences with several large benchmarks (sets of test items to which an LLM can respond either correctly or incorrectly). However, high correlations within and between benchmark scores suggest that (1) there exists a small set of common underlying abilities that these benchmarks measure, and (2) items tap into redundant information and the benchmarks may thus be considerably compressed. We use data from $n> 5000$ LLMs to identify the most informative items of six benchmarks, ARC, GSM8K, HellaSwag, MMLU, TruthfulQA and WinoGrande (with d=28,632 items in total). From them we distill a sparse benchmark, `metabench`, that has less than $3%$ of the original size of all six benchmarks combined. This new sparse benchmark goes beyond point scores by yielding estimators of the underlying benchmark-specific abilities. We show that these estimators (1) can be used to reconstruct each original individual benchmark score with, on average, $1.5%$ root mean square error (RMSE), (2) reconstruct the original total score with $0.8%$ RMSE, and (3) have a single underlying common factor whose Spearman correlation with the total score is $r=0.93$.

Homepage: https://github.com/adkipnis/metabench


### Citation

```bibtex
@article{metabench,
  author  = {Alex Kipnis and Konstantinos Voudouris and Luca M. Schulze Buschoff and Eric Schulz},
  title   = {metabench - A Sparse Benchmark to Measure General Ability in Large Language Models},
  journal = {arXiv preprint arXiv:2407.12844},
  year    = {2024},
}
```

### Groups and Tasks

#### Groups

There are no groups.

#### Tasks

* `metabench_arc` -- a subset of the [ARC benchmark](https://huggingface.co/datasets/allenai/ai2_arc) containing the 100 most informative items.
* `metabench_gsm8k` -- a subset of the [GSM8K benchmark](https://huggingface.co/datasets/openai/gsm8k) containing the 237 most informative items.
* `metabench_hellaswag` -- a subset of the [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) benchmark containing the 58 most informative items.
* `metabench_mmlu` -- a subset of the [MMLU benchmark](https://huggingface.co/datasets/cais/mmlu) containing the 102 most informative items (strictly, a subset of [hails/mmmlu_no_train](https://huggingface.co/datasets/hails/mmlu_no_train)).
* `metabench_truthfulqa` -- a subset of the [TruthfulQA benchmark](https://huggingface.co/datasets/truthfulqa/truthful_qa) containing the 136 most informative items.
* `metabench_winogrande` -- a subset of the [Winogrande benchmark](https://huggingface.co/datasets/allenai/winogrande) containing the 106 most informative items.

### Checklist

For adding novel benchmarks/datasets to the library:
* [X] Is the task an existing benchmark in the literature?
  * [X] Have you referenced the original paper that introduced the task?
  * [X] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [X] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?