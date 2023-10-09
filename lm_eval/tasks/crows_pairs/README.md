# CrowS-Pairs

### Paper

CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models
https://aclanthology.org/2020.emnlp-main.154/
French CrowS-Pairs: Extending a challenge dataset for measuring social bias in masked
language models to a language other than English
https://aclanthology.org/2022.acl-long.583/

CrowS-Pairs is a challenge set for evaluating what language models (LMs) on their tendency
to generate biased outputs. CrowS-Pairs comes in 2 languages and the English subset has
a newer version which fixes some of the issues with the original version.

Homepage: https://github.com/nyu-mll/crows-pairs, https://gitlab.inria.fr/french-crows-pairs

### Citation

```bibtex
@inproceedings{nangia-etal-2020-crows,
    title = "{C}row{S}-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models",
    author = "Nangia, Nikita  and
      Vania, Clara  and
      Bhalerao, Rasika  and
      Bowman, Samuel R.",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.154",
    doi = "10.18653/v1/2020.emnlp-main.154",
    pages = "1953--1967",
    abstract = "Pretrained language models, especially masked language models (MLMs) have seen success across many NLP tasks. However, there is ample evidence that they use the cultural biases that are undoubtedly present in the corpora they are trained on, implicitly creating harm with biased representations. To measure some forms of social bias in language models against protected demographic groups in the US, we introduce the Crowdsourced Stereotype Pairs benchmark (CrowS-Pairs). CrowS-Pairs has 1508 examples that cover stereotypes dealing with nine types of bias, like race, religion, and age. In CrowS-Pairs a model is presented with two sentences: one that is more stereotyping and another that is less stereotyping. The data focuses on stereotypes about historically disadvantaged groups and contrasts them with advantaged groups. We find that all three of the widely-used MLMs we evaluate substantially favor sentences that express stereotypes in every category in CrowS-Pairs. As work on building less biased models advances, this dataset can be used as a benchmark to evaluate progress.",
}

@inproceedings{neveol-etal-2022-french,
    title = "{F}rench {C}row{S}-Pairs: Extending a challenge dataset for measuring social bias in masked language models to a language other than {E}nglish",
    author = {N{\'e}v{\'e}ol, Aur{\'e}lie  and
      Dupont, Yoann  and
      Bezan{\c{c}}on, Julien  and
      Fort, Kar{\"e}n},
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.583",
    doi = "10.18653/v1/2022.acl-long.583",
    pages = "8521--8531",
    abstract = "Warning: This paper contains explicit statements of offensive stereotypes which may be upsetting.Much work on biases in natural language processing has addressed biases linked to the social and cultural experience of English speaking individuals in the United States. We seek to widen the scope of bias studies by creating material to measure social bias in language models (LMs) against specific demographic groups in France. We build on the US-centered CrowS-pairs dataset to create a multilingual stereotypes dataset that allows for comparability across languages while also characterizing biases that are specific to each country and language. We introduce 1,679 sentence pairs in French that cover stereotypes in ten types of bias like gender and age. 1,467 sentence pairs are translated from CrowS-pairs and 212 are newly crowdsourced. The sentence pairs contrast stereotypes concerning underadvantaged groups with the same sentence concerning advantaged groups. We find that four widely used language models (three French, one multilingual) favor sentences that express stereotypes in most bias categories. We report on the translation process from English into French, which led to a characterization of stereotypes in CrowS-pairs including the identification of US-centric cultural traits. We offer guidelines to further extend the dataset to other languages and cultural environments.",
}
```

### Groups and Tasks

#### Groups

- `crows_pairs_english`: The entire English subset of the CrowS-Pairs dataset.
- `crows_pairs_french`: The entire French subset of the CrowS-Pairs dataset.

#### Tasks


The following tasks evaluate sub-areas of bias in the English CrowS-Pairs dataset:
- `crows_pairs_english_age`
- `crows_pairs_english_autre`
- `crows_pairs_english_disability`
- `crows_pairs_english_gender`
- `crows_pairs_english_nationality`
- `crows_pairs_english_physical_appearance`
- `crows_pairs_english_race_color`
- `crows_pairs_english_religion`
- `crows_pairs_english_sexual_orientation`
- `crows_pairs_english_socioeconomic`

The following tasks evaluate sub-areas of bias in the French CrowS-Pairs dataset:
- `crows_pairs_french_age`
- `crows_pairs_french_autre`
- `crows_pairs_french_disability`
- `crows_pairs_french_gender`
- `crows_pairs_french_nationality`
- `crows_pairs_french_physical_appearance`
- `crows_pairs_french_race_color`
- `crows_pairs_french_religion`
- `crows_pairs_french_sexual_orientation`
- `crows_pairs_french_socioeconomic`

All tasks evaluate the percentage of more-stereotypical sentences that are rated as more likely by a model than the non-stereotypical sentences (`pct_stereotype`), as well as the average absolute difference of loglikelihoods between the sentences in the pairs.

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [x] The original paper does not for causal language models, so this is a novel formulation of the task for autoregressive LMs.

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
