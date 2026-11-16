# Wikitext

### Paper

Pointer Sentinel Mixture Models
https://arxiv.org/pdf/1609.07843.pdf

The WikiText language modeling dataset is a collection of over 100 million tokens
extracted from the set of verified Good and Featured articles on Wikipedia.

NOTE: This `Task` is based on WikiText-2.

Homepage: https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/


### Citation

```
@misc{merity2016pointer,
    title={Pointer Sentinel Mixture Models},
    author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
    year={2016},
    eprint={1609.07843},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `wikitext`: measure perplexity on the Wikitext dataset, via rolling loglikelihoods.

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
