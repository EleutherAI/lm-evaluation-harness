# The Pile

### Paper
Title: The Pile: An 800GB Dataset of Diverse Text for Language Modeling

Abstract: https://arxiv.org/abs/2101.00027

The Pile is a 825 GiB diverse, open source language modelling data set that consists
of 22 smaller, high-quality datasets combined together. To score well on Pile
BPB (bits per byte), a model must be able to understand many disparate domains
including books, github repositories, webpages, chat logs, and medical, physics,
math, computer science, and philosophy papers.

Homepage: https://pile.eleuther.ai/

### Citation
```
@article{pile,
  title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
```

### Groups and Tasks

#### Groups

* `pile`

#### Tasks

* `pile_arxiv`
* `pile_bookcorpus2`
* `pile_books3`
* `pile_dm-mathematics`
* `pile_enron`
* `pile_europarl`
* `pile_freelaw`
* `pile_github`
* `pile_gutenberg`
* `pile_hackernews`
* `pile_nih-exporter`
* `pile_opensubtitles`
* `pile_openwebtext2`
* `pile_philpapers`
* `pile_pile-cc`
* `pile_pubmed-abstracts`
* `pile_pubmed-central`
* `pile_stackexchange`
* `pile_ubuntu-irc`
* `pile_uspto`
* `pile_wikipedia`
* `pile_youtubesubtitles`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
