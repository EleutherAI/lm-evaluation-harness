# Belebele

### Paper

The Belebele Benchmark for Massively Multilingual NLU Evaluation
https://arxiv.org/abs/2308.16884

Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants. This dataset enables the evaluation of mono- and multi-lingual models in high-, medium-, and low-resource languages. Each question has four multiple-choice answers and is linked to a short passage from the FLORES-200 dataset. The human annotation procedure was carefully curated to create questions that discriminate between different levels of generalizable language comprehension and is reinforced by extensive quality checks. While all questions directly relate to the passage, the English dataset on its own proves difficult enough to challenge state-of-the-art language models. Being fully parallel, this dataset enables direct comparison of model performance across all languages. Belebele opens up new avenues for evaluating and analyzing the multilingual abilities of language models and NLP systems.

Homepage: https://github.com/facebookresearch/belebele

### Citation

```bibtex
@misc{bandarkar2023belebele,
      title={The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants},
      author={Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
      year={2023},
      eprint={2308.16884},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

- `belebele`: All 122 languages of the Belebele dataset, evaluated following the methodology in MMLU's original implementation.

#### Tasks


The following tasks evaluate languages in the Belebele dataset using loglikelihood-based multiple-choice scoring:
- `belebele_{language}`

The variant evaluated here is the 0-shot or few-shot evaluation with English Instructions.

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [ ] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
