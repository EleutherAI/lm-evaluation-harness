# Icelandic WinoGrande

### Paper

Title: `A Warm Start and a Clean Crawled Corpus - A Recipe for Good Language Models`

Link: https://aclanthology.org/2022.lrec-1.464/

Dataset: https://huggingface.co/datasets/mideind/icelandic-winogrande

Icelandic WinoGrande is a manually translated and localized version of the English-language WinoGrande dataset, designed to be 'a new and challenging benchmark for commonsense reasoning and natural language understanding' in Icelandic [(Snæbjarnarson et al., 2022)](https://aclanthology.org/2022.lrec-1.464/).

**Implementation Note:** The original dataset is designed for evaluation on a BERT model. Following the evaluation method used for the original (English-language) WinoGrande on the Harness (see information [here](../winogrande/README.md)), this evaluation uses partial scoring as described by [Trinh & Le (2018)](https://arxiv.org/abs/1806.02847) to allow evaluation on autoregressive models.

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `icelandic_winogrande`

### Citation

```
@inproceedings{snaebjarnarson-etal-2022-warm,
    title = "A Warm Start and a Clean Crawled Corpus - A Recipe for Good Language Models",
    author = "Sn{\ae}bjarnarson, V{\'e}steinn  and
      S{\'i}monarson, Haukur Barri  and
      Ragnarsson, P{\'e}tur Orri  and
      Ing{\'o}lfsd{\'o}ttir, Svanhv{\'i}t Lilja  and
      J{\'o}nsson, Haukur  and
      Thorsteinsson, Vilhjalmur  and
      Einarsson, Hafsteinn",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.464/",
    pages = "4356--4366"
}
```

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
