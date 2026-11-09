# Social IQA

### Paper

Title: Social IQA: Commonsense Reasoning about Social Interactions

Abstract: https://arxiv.org/abs/1904.09728

> We introduce Social IQa, the first largescale benchmark for commonsense reasoning about social situations. Social IQa contains 38,000 multiple choice questions for probing emotional and social intelligence in a variety of everyday situations (e.g., Q: "Jordan wanted to tell Tracy a secret, so Jordan leaned towards Tracy. Why did Jordan do this?" A: "Make sure no one else could hear"). Through crowdsourcing, we collect commonsense questions along with correct and incorrect answers about social interactions, using a new framework that mitigates stylistic artifacts in incorrect answers by asking workers to provide the right answer to a different but related question. Empirical results show that our benchmark is challenging for existing question-answering models based on pretrained language models, compared to human performance (>20% gap). Notably, we further establish Social IQa as a resource for transfer learning of commonsense knowledge, achieving state-of-the-art performance on multiple commonsense reasoning tasks (Winograd Schemas, COPA).

Homepage: https://allenai.org/data/socialiqa


### Citation

```
@inproceedings{sap2019social,
  title={Social IQa: Commonsense Reasoning about Social Interactions},
  author={Sap, Maarten and Rashkin, Hannah and Chen, Derek and Le Bras, Ronan and Choi, Yejin},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={4463--4473},
  year={2019}
}
```

### Checklist

For adding novel benchmarks/datasets to the library:
* [X] Is the task an existing benchmark in the literature?
  * [X] Have you referenced the original paper that introduced the task?
  * [X] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test? The original paper doesn't have an associated implementation, but there is an official entry in [BigBench](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/social_iqa). I use the same prompting format as BigBench.


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
