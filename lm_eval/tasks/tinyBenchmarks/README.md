# tinyBenchmarks

### Paper

Title: `tinyBenchmarks: evaluating LLMs with fewer examples`

Abstract: https://arxiv.org/abs/2402.14992

The versatility of large language models (LLMs) led to the creation of diverse benchmarks that thoroughly test a variety of language models' abilities. These benchmarks consist of tens of thousands of examples making evaluation of LLMs very expensive. In this paper, we investigate strategies to reduce the number of evaluations needed to assess the performance of an LLM on several key benchmarks. For example, we show that to accurately estimate the performance of an LLM on MMLU, a popular multiple-choice QA benchmark consisting of 14K examples, it is sufficient to evaluate this LLM on 100 curated examples. We release evaluation tools and tiny versions of popular benchmarks: Open LLM Leaderboard, MMLU, HELM, and AlpacaEval 2.0. Our empirical analysis demonstrates that these tools and tiny benchmarks are sufficient to reliably and efficiently reproduce the original evaluation results. 

Homepage: -

### Groups and Tasks

#### Groups

* `tinyBenchmarks`

#### Tasks

* `tinyArc`, `tinyGSM8k`, `tinyHellaswag`, `tinyMMLU`, `tinyTruthfulQA`, `tinyWinogrande`,

### Usage

*tinyBenchmarks* can evaluate different benchmarks with a fraction of their examples.
To obtain accurate results, the score vectors obtained with this task require post-processing using the *tinyBenchmarks*-package (see [here](https://github.com/felipemaiapolo/tinyBenchmarks/blob/main/README.md?plain=1)).

You can install our package by running the following commands on the terminal

``` :sh
$ pip install git+https://github.com/felipemaiapolo/tinyBenchmarks
```

Through the package, the ability parameter $\theta$ from the IRT model will be estimated using all the available data. For `benchmark='lb'` or `benchmark='helm_lite'`, the dimension of `y` should be 600 and 1000, respectively, where the correctness values must obey the following order 
- For the Open LLM Leaderboard: TruthfulQA, GSM8K, Winogrande, ARC, HellaSwag, and MMLU;

For all other, benchmarks the dimension of `y` should be 100.

```python
import json
import tinyBenchmarks as tb
import numpy as np

# Choose benchmark (e.g. hellaswag)
benchmark = 'hellaswag' # possible benchmarks:
                        # ['lb','mmlu','alpaca','helm_lite','truthfulqa',
                        #  'gsm8k', 'winogrande', 'arc', 'hellaswag']

# Get score vector from output-file (the metric `acc_norm` depends on the benchmark)
file_path = '<your-output-file.jsonl>'
with open(file_path, 'r') as file:
    outputs = json.load(file)
y = np.array([float(item['acc_norm']) for item in outputs])

### Evaluation
tb.evaluate(y, benchmark)
```

### Performance 

We report in the following tables the average estimation error in the test set (using data from the paper) and standard deviation across LLMs.

#### Open LLM Leaderboard

Estimating performance for each scenario separately
|| IRT | p-IRT | gp-IRT |
|--|--|--|--|
| TruthfulQA | 0.013 (0.010) | 0.010 (0.009) | 0.011 (0.009) |
| GSM8K | 0.022 (0.017) | 0.029 (0.022) | 0.020 (0.017) |
| Winogrande | 0.022 (0.017) | 0.016 (0.014) | 0.015 (0.013) |
| ARC | 0.022 (0.018) | 0.017 (0.014) | 0.017 (0.013) |
| HellaSwag | 0.013 (0.016) | 0.015 (0.012) | 0.015 (0.012) |
| MMLU | 0.024 (0.017) | 0.016 (0.015) | 0.016 (0.015) |

Estimating performance for each scenario all at once
|| IRT | p-IRT | gp-IRT |
|--|--|--|--|
| TruthfulQA  | 0.013 (0.010) | 0.016 (0.013) | 0.011 (0.009) |
| GSM8K | 0.022 (0.017) | 0.022 (0.017) | 0.020 (0.015) |
| Winogrande | 0.022 (0.017) | 0.011 (0.013) | 0.011 (0.011) |
| ARC | 0.022 (0.018) | 0.012 (0.010) | 0.010 (0.009) |
| HellaSwag | 0.013 (0.016) | 0.011 (0.020) | 0.011 (0.018) |
| MMLU | 0.024 (0.018) | 0.017 (0.017) | 0.015 (0.015) |



### Citation

```
@article{polo2024tinybenchmarks,
      title={tinyBenchmarks: evaluating LLMs with fewer examples},
      author={Maia Polo, Felipe and Weber, Lucas and Choshen, Leshem and Sun, Yuekai and Xu, Gongjun and Yurochkin, Mikhail},
      journal={arXiv preprint arXiv:2402.14992},
      year={2024}
    }
```


### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
