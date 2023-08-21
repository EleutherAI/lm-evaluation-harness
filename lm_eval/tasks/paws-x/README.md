# PAWS-X

### Paper

Title: `PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification`
Abstract: https://arxiv.org/abs/1908.11828

The dataset consists of 23,659 human translated PAWS evaluation pairs and
296,406 machine translated training pairs in 6 typologically distinct languages.

Examples are adapted from  PAWS-Wiki

Prompt format (same as in mGPT):

"<s>" + sentence1 + ", right? " + mask + ", " + sentence2 + "</s>",

where mask is the string that matches the label:

Yes, No.

Example:

<s> The Tabaci River is a tributary of the River Leurda in Romania, right? No, The Leurda River is a tributary of the River Tabaci in Romania.</s>

Language specific prompts are translated word-by-word with Google Translate
and may differ from the ones used by mGPT and XGLM (they do not provide their prompts).

Homepage: https://github.com/google-research-datasets/paws/tree/master/pawsx


### Citation

```
@inproceedings{yang-etal-2019-paws,
    title = "{PAWS}-{X}: A Cross-lingual Adversarial Dataset for Paraphrase Identification",
    author = "Yang, Yinfei  and
      Zhang, Yuan  and
      Tar, Chris  and
      Baldridge, Jason",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1382",
    doi = "10.18653/v1/D19-1382",
    pages = "3687--3692",
}
```

### Groups and Tasks

#### Groups

* `pawsx`

#### Tasks

* `paws_de`: German
* `paws_en`: English
* `paws_es`: Spanish
* `paws_fr`: French
* `paws_ja`: Japanese
* `paws_ko`: Korean
* `paws_zh`: Chinese


### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
