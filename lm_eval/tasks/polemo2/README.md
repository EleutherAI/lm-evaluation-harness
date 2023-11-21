# PolEmo 2.0

### Paper

Title: `Multi-Level Sentiment Analysis of PolEmo 2.0: Extended Corpus of Multi-Domain Consumer Reviews`

Abstract: https://aclanthology.org/K19-1092/

The PolEmo 2.0 is a dataset of online consumer reviews in Polish from four domains: medicine, hotels, products, and university. It is human-annotated on a level of full reviews and individual sentences. It comprises over 8000 reviews, about 85% from the medicine and hotel domains.
The goal is to predict the sentiment of a review. There are two separate test sets, to allow for in-domain (medicine and hotels) as well as out-of-domain (products and university) validation.

Homepage: https://clarin-pl.eu/dspace/handle/11321/710


### Citation

```
@inproceedings{kocon-etal-2019-multi,
    title = "Multi-Level Sentiment Analysis of {P}ol{E}mo 2.0: Extended Corpus of Multi-Domain Consumer Reviews",
    author = "Koco{\'n}, Jan  and
      Mi{\l}kowski, Piotr  and
      Za{\'s}ko-Zieli{\'n}ska, Monika",
    booktitle = "Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/K19-1092",
    doi = "10.18653/v1/K19-1092",
    pages = "980--991",
    abstract = "In this article we present an extended version of PolEmo {--} a corpus of consumer reviews from 4 domains: medicine, hotels, products and school. Current version (PolEmo 2.0) contains 8,216 reviews having 57,466 sentences. Each text and sentence was manually annotated with sentiment in 2+1 scheme, which gives a total of 197,046 annotations. We obtained a high value of Positive Specific Agreement, which is 0.91 for texts and 0.88 for sentences. PolEmo 2.0 is publicly available under a Creative Commons copyright license. We explored recent deep learning approaches for the recognition of sentiment, such as Bi-directional Long Short-Term Memory (BiLSTM) and Bidirectional Encoder Representations from Transformers (BERT).",
}
```

### Groups and Tasks

#### Groups

* `polemo2`: Evaluates `polemo2_in` and `polemo2_out`

#### Tasks

* `polemo2_in`: evaluates sentiment predictions of in-domain (medicine and hotels) reviews
* `polemo2_out`: evaluates sentiment predictions of out-of-domain (products and university) reviews

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
