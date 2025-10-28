# HRM8K

### Paper

Title: [Understand, Solve and Translate: Bridging the Multilingual Mathematical Reasoning Gap](https://www.arxiv.org/abs/2501.02448)

Large language models (LLMs) demonstrate exceptional performance on complex reasoning tasks. However, despite their strong reasoning capabilities in high-resource languages (e.g., English and Chinese), a significant performance gap persists in other languages. To investigate this gap in Korean, we introduce HRM8K, a benchmark comprising 8,011 English-Korean parallel bilingual math problems. Through systematic analysis of model behaviors, we identify a key finding: these performance disparities stem primarily from difficulties in comprehending non-English inputs, rather than limitations in reasoning capabilities. Based on these findings, we propose UST (Understand, Solve, and Translate), a method that strategically uses English as an anchor for reasoning and solution generation. By fine-tuning the model on 130k synthetically generated data points, UST achieves a 10.91% improvement on the HRM8K benchmark and reduces the multilingual performance gap from 11.6% to 0.7%. Additionally, we show that improvements from UST generalize effectively to different Korean domains, demonstrating that capabilities acquired from machine-verifiable content can be generalized to other areas. We publicly release the benchmark, training dataset, and models.

Homepage: https://huggingface.co/datasets/HAERAE-HUB/HRM8K

### Evaluation Settings

The authors suggest (Appendix B) using:

* **Sampling temperature:** `0.7`  
* **Top‑p:** `0.95`  
* **Output length:** *min* `8` tokens, *max* `2048` tokens (`max_gen_toks`)

We default to greedy decoding.

### Citation

```
@article{ko2025understand,
  title={Understand, Solve and Translate: Bridging the Multilingual Mathematical Reasoning Gap},
  author={Ko, Hyunwoo and Son, Guijin and Choi, Dasol},
  journal={arXiv preprint arXiv:2501.02448},
  year={2025}
}
```

### Groups and Tasks

#### Groups

* `hrm8k`: HRM8K comprises 8,011 instances for evaluation, sourced through a combination of translations from established English benchmarks (e.g., GSM8K, MATH, OmniMath, MMMLU) and original problems curated from existing Korean math exams. This benchmark consists of Korean instruction and question.
* `hrm8k_en`: English version of `hrm8k`. This benchmark consists of English instruction and question.

#### Tasks

* `hrm8k_{gsm8k|ksm|math|mmmlu|omni_math}`
* `hrm8k_en_{gsm8k|ksm|math|mmmlu|omni_math}`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
- 2025-07-22: v1.1: increased `max_gen_toks` to 2048
