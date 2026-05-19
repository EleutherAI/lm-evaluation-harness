# NEREL-bench

## Paper

Title: **NEREL-bench: A Benchmark for Evaluating LLMs on Russian Knowledge Graph Construction Tasks**

NEREL-bench is a benchmark dataset designed to evaluate the capabilities of Large Language Models (LLMs) in performing knowledge graph construction tasks on Russian-language texts. The benchmark comprises four distinct configurations targeting specific aspects of knowledge graph construction: entity recognition, entity definition generation, relation extraction, and relation definition generation. All tasks are critical for evaluating whether an LLM can effectively construct a knowledge graph suitable for GraphRAG and similar knowledge-intensive approaches.

Homepage: [https://huggingface.co/datasets/bond005/NEREL_bench](https://huggingface.co/datasets/bond005/NEREL_bench)

### Citation

```text
@dataset{bondarenko2026nerelbench,
  title={NEREL-bench: A Benchmark for Evaluating LLMs on Russian Knowledge Graph Construction Tasks},
  author={Bondarenko, Ivan},
  year={2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Datasets},
  howpublished = {\url{https://huggingface.co/datasets/bond005/NEREL_bench}}
}

@inproceedings{loukachevitch2021nerel,
  title={{NEREL: A Russian} Dataset with Nested Named Entities, Relations and Events},
  author={Loukachevitch, Natalia and Artemova, Ekaterina and Batura, Tatiana and Braslavski, Pavel and Denisov, Ilia and Ivanov, Vladimir and Manandhar, Suresh and Pugachev, Alexander and Tutubalina, Elena},
  booktitle={Proceedings of RANLP},
  pages={876--885},
  year={2021}
}
```

### Groups, Tags, and Tasks

#### Groups

No groups defined.

#### Tags

* `nerel-bench`: Tasks for evaluating LLMs on Russian knowledge graph construction (NER, NED, RE, RD)

#### Tasks

* NER `ru_entity_recognition`: Evaluates the model's ability to identify and list all named entities in Russian text in normalized form (metric: F1)
* NED `ru_entity_definition`: Evaluates generation of contextually appropriate definitions for named entities (metric: chrF++)
* RE `ru_relation_extraction`: Evaluates identification and extraction of binary relationships between named entities (metric: F1)
* RD `ru_relation_definition`: Evaluates generation of meaningful explanations of relationships between entity pairs (metric: chrF++)

#### Metrics

**F1 Score (for `ru_entity_recognition` and `ru_relation_extraction`)**

For entity recognition and relation extraction tasks, we use a set-based F1 score computed over normalized entity lists. Both predictions and reference answers are split by newlines, with each entity undergoing normalization (lowercasing, tokenization, filtering to alphanumeric tokens, and optional Russian stemming). The normalization step is essential for Russian, where the same entity may appear in different grammatical cases (e.g., «Новосибирск», «Новосибирска», «Новосибирску» should be recognized as the same entity).

Precision is calculated as the ratio of correctly predicted entities to all predicted entities, while recall measures the ratio of correctly predicted entities to all reference entities. The final F1 score is the harmonic mean of precision and recall, aggregated across all examples in the test set. This approach accounts for both false positives (hallucinated entities) and false negatives (missed entities), providing a balanced evaluation of extraction quality.

**chrF++ (for `ru_entity_definition` and `ru_relation_definition`)**

For definition generation tasks, we use chrF++ (character n-gram F-score) from the sacrebleu library. This metric was chosen over exact match or token-level metrics due to the morphological complexity of Russian. A semantically correct definition may differ from the reference in word forms, word order, or synonyms while remaining valid. chrF++ captures these variations by operating at the character level, making it robust to morphological variations that are pervasive in Russian (case endings, gender agreement, verbal conjugations).

Before computing chrF++, both predictions and references undergo the same normalization procedure used for F1 calculation (lowercasing, alphanumeric filtering, and Russian stemming). The final score is averaged across all examples in the test set. Note that chrF++ in sacrebleu includes word n-grams (up to bigrams by default), which helps capture multi-word entity names and relation phrases that should be preserved across variations.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * The original NEREL dataset provides annotation guidelines. This benchmark extends NEREL with instruction-tuning format for LLM evaluation. Reference implementation available at: https://huggingface.co/datasets/bond005/NEREL_bench

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
