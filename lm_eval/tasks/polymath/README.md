# PolyMath

## Paper

Title: PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts

Abstract: https://arxiv.org/abs/2504.18428

Homepage: https://huggingface.co/datasets/Qwen/PolyMath

PolyMath is a multilingual mathematical reasoning benchmark covering 18 languages and 4 easy-to-hard difficulty levels. Our benchmark ensures difficulty comprehensiveness, language diversity, and high-quality translation, making it a highly discriminative multilingual mathematical benchmark in the era of reasoning LLMs.

### Citation

```text
@article{wang2025polymath,
  title={PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts},
  author={Yiming Wang and Pei Zhang and Jialong Tang and Haoran Wei and Baosong Yang and Rui Wang and Chenshu Sun and Feitong Sun and Jiran Zhang and Junxuan Wu and Qiqian Cang and Yichang Zhang and Fei Huang and Junyang Lin and Fei Huang and Jingren Zhou},
  journal={arXiv preprint arXiv:2504.18428},
  year={2025},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2504.18428}, 
}
```

### Implementation

With respect to the evaluation prompt, the authors state:

> Prompts. In addition to the original input problem Q, we append the instruction “Note: Please put the final answer in $\boxed{}$.” after it to help extract the final answer. Each language uses its own version of this instruction, as detailed in Appendix C.2.

In this implementation, the evaluation prompt follows the format shown in appendix examples of the original paper (## Problem: {{question}}\n## Answer:\n) and the instruction for the answer format in each language is taken from Figure 8 of the original paper.

With respect to the evaluation metric, the authors state:

> At each level, we obtain an accuracy (ACC) result for each model and language, corresponding to the pass@1 metric. 

We use the `exact_match` metric to evaluate the accuracy of the model's output.

Finally, for the aggregation, the authors introduce the Difficulty-Weighted Accuracy (DWACC):

> This metric assigns level-specific weights w1, w2, w3, w4 to each problem from the low/medium/high/top level, respectively. The weights double at each ascending level: By default, we set w1 = 1, leading to w2 = 2, w3 = 4, w4 = 8. This means that solving eight low-level problems is equivalent to solving a single top-level problem in terms of contribution to the final score.

The DWACC formula is defined as:

```
DWACC = (w₁a₁ + w₂a₂ + w₃a₃ + w₄a₄) / (w₁ + w₂ + w₃ + w₄)
```

Where:
- a₁, a₂, a₃, a₄ are the accuracy scores at each difficulty level (low, medium, high, top)
- w₁, w₂, w₃, w₄ are the corresponding weights (1, 2, 4, 8)

Since the weights sum to 15 (1 + 2 + 4 + 8 = 15), this can also be written as:

```
DWACC = (a₁ + 2a₂ + 4a₃ + 8a₄) / 15
```

### Groups, Tags, and Tasks

#### Groups

* `polymath` - Main PolyMath benchmark (all languages and difficulties)
* `polymath_low` - Low difficulty level (all 18 languages)
* `polymath_medium` - Medium difficulty level (all 18 languages)
* `polymath_high` - High difficulty level (all 18 languages)
* `polymath_top` - Top difficulty level (all 18 languages)
* `polymath_ar` - Arabic language (all 4 difficulties)
* `polymath_bn` - Bengali language (all 4 difficulties)
* `polymath_de` - German language (all 4 difficulties)
* `polymath_en` - English language (all 4 difficulties)
* `polymath_es` - Spanish language (all 4 difficulties)
* `polymath_fr` - French language (all 4 difficulties)
* `polymath_id` - Indonesian language (all 4 difficulties)
* `polymath_it` - Italian language (all 4 difficulties)
* `polymath_ja` - Japanese language (all 4 difficulties)
* `polymath_ko` - Korean language (all 4 difficulties)
* `polymath_ms` - Malay language (all 4 difficulties)
* `polymath_pt` - Portuguese language (all 4 difficulties)
* `polymath_ru` - Russian language (all 4 difficulties)
* `polymath_sw` - Swahili language (all 4 difficulties)
* `polymath_te` - Telugu language (all 4 difficulties)
* `polymath_th` - Thai language (all 4 difficulties)
* `polymath_vi` - Vietnamese language (all 4 difficulties)
* `polymath_zh` - Chinese language (all 4 difficulties)

#### Tasks

* `polymath_{language}_{difficulty}` - Individual language-difficulty pairs (72 total tasks)
  * Languages: ar, bn, de, en, es, fr, id, it, ja, ko, ms, pt, ru, sw, te, th, vi, zh
  * Difficulties: low, medium, high, top

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
