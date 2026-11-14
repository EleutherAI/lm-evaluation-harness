# ShawarmaChats

## Paper

Title: `Shawarma Chats: A Benchmark Exact Dialogue & Evaluation Platter in Egyptian, Maghrebi & Modern Standard Arabic—A Triple-Dialect Feast for Hungry Language Models`

Paper: https://aclanthology.org/2025.arabicnlp-main.39/

ShawarmaChats has 30,000 six-turn dialogues grounded in Arabic Wikipedia, split evenly across
Modern Standard Arabic, Egyptian Arabic and Maghrebi Arabic and checked by native speakers.
The test set has 504 dialogues per dialect.

Homepage: https://github.com/KamyarZeinalipour/Shawarma-Chats

Data: https://huggingface.co/datasets/Kamyar-zeinalipour/arabic_multi_dialect_dialogue (`dialogues` config)

### Citation

```text
@inproceedings{zeinalipour-etal-2025-shawarma,
    title = "Shawarma Chats: A Benchmark Exact Dialogue {\&} Evaluation Platter in {E}gyptian, {M}aghrebi {\&} {M}odern {S}tandard {A}rabic{---}{A} Triple-Dialect Feast for Hungry Language Models",
    author = "Zeinalipour, Kamyar and Saad, Mohamed Zaky and Attafi, Oumaima and Maggini, Marco and Gori, Marco",
    booktitle = "Proceedings of The Third {A}rabic Natural Language Processing Conference",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.arabicnlp-main.39/",
    doi = "10.18653/v1/2025.arabicnlp-main.39",
    pages = "472--524"
}
```

### Groups, Tags, and Tasks

#### Groups

* `shawarma_chats`: the three dialects.

#### Tasks

* `shawarma_chats_msa`: write a six-turn dialogue in Modern Standard Arabic about a Wikipedia passage; chrF against the reference.
* `shawarma_chats_egyptian`: the same in Egyptian Arabic.
* `shawarma_chats_maghrebi`: the same in Maghrebi Arabic.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

These tasks report chrF only. The paper's own evaluation scripts are in the GitHub repository;
numbers from the two setups are not directly comparable.

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version.
