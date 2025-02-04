```
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
````
# SCORE: Systematic COnsistency and Robustness Evaluation for Large Language Models


## Citation
```bib
[Citation placeholder]
```

## Groups

- `score_robustness_mmlu_pro`: two 0-shot robutstness tasks on MMLU-PRO dataset [[1](#mmlu_pro)]

- `score_robustness_agieval`: two 0-shot robutstness tasks on the AGIEVAL datasets [[2](#agi_eval)] multiple choice questions subsets:  `'agieval-sat-math'`, `'agieval-lsat-lr'`, `'agieval-lsat-rc'`, `'agieval-logiqa-en'`, `'agieval-aqua-rat'`, `'agieval-sat-en'`, `'agieval-lsat-ar'`

- `score_robustness_math`: one 0-shot robutstness tasks on Hendryk's MATH dataset [[3](#math)]

## Tasks

Both `score_robustness_mmlu_pro` and `score_robustness_agieval` contain the following 3 tasks:

* Option order robustness:
`score_option_order_robustness_mmlu_pro`,
`score_option_order_robustness_agieval`

* Prompt robustness:
`score_prompt_robustness_mmlu_pro`,
`score_prompt_robustness_agieval`,

* Non greedy robustness
`score_non_greedy_robustness_mmlu_pro`,
`score_non_greedy_robustness_agieval`,

Whereas math contains the following 2:
* Prompt robustness:
`score_prompt_robustness_math`
`score_non_greedy_robustness_math`,

### Option order robustness

Measures the model's robustness to the placement of the correct answer in the options list by swapping the correct answer with all the other possible options.

### Prompt robustness

Measures the model's robustness to 10 different prompts. list of the prompts can be found in the `./prompt_templates.json` file under the key `prompt_robustness`.


### Non greedy robustness

Measures the model's robustness to 5 different seeds: seeds = \[1-5\]. For evaluating on the non greedy task, please, refer to [NON_GREEDY.md](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/score/NON_GREEDY.md)

## Metrics

All robustness tasks calculate 2 metrics: *Accuracy* and *Consistency Rate(CR)* [[4](#cr)].

$CR = \frac{1}{|Q|} \sum_{Q_k \in Q} \sum_{y_i \in Y_k} \sum_{\substack{y_j \in Y_k \\ j \neq i}}\frac{\text{sim}(y_i, y_j)}{\binom{|Y_k|}{2}}$

## Notes

- All tasks are designed for **Instruct** models for which we recommend to pass "`--apply_chat_template`" flag.


## References
<a name=mmlu_pro></a>[1] Wang, et al. "Mmlu-pro: A more robust and challenging multi-task language understanding benchmark." arXiv preprint arXiv:2406.01574 (2024).

<a name=agi_eval></a>[2] Zhong, et al. "Agieval: A human-centric benchmark for evaluating foundation models." arXiv preprint arXiv:2304.06364 (2023).

<a name=math></a>[3] Hendrycks et al. "Measuring Mathematical Problem Solving With the MATH Dataset." arXiv:2103.03874 (2021).

<a name=cr></a>[4] Yukun et al. "Improving the robustness of large language models via consistency alignment." arXiv:2403.14221 (2024).

## Checklist

For adding novel benchmarks/datasets to the library:
* [-] Is the task an existing benchmark in the literature?
  * [-] Have you referenced the original paper that introduced the task? - Will be referenced as soon as the paper is published
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
