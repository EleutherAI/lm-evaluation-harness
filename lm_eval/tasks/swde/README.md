# SWDE

### Paper

Title: Language Models Enable Simple Systems For
Generating Structured Views Of Heterogenous Data
Lakes

Abstract: A long standing goal of the data management community is to develop general, automated systems
that ingest semi-structured documents and output queryable tables without human effort or domain
specific customization. Given the sheer variety of potential documents, state-of-the art systems make
simplifying assumptions and use domain specific training. In this work, we ask whether we can
maintain generality by using large language models (LLMs). LLMs, which are pretrained on broad
data, can perform diverse downstream tasks simply conditioned on natural language task descriptions.
We propose and evaluate EVAPORATE, a simple, prototype system powered by LLMs. We identify
two fundamentally different strategies for implementing this system: prompt the LLM to directly
extract values from documents or prompt the LLM to synthesize code that performs the extraction.
Our evaluations show a cost-quality tradeoff between these two approaches. Code synthesis is cheap,
but far less accurate than directly processing each document with the LLM. To improve quality while
maintaining low cost, we propose an extended code synthesis implementation, EVAPORATE-CODE+,
which achieves better quality than direct extraction. Our key insight is to generate many candidate
functions and ensemble their extractions using weak supervision. EVAPORATE-CODE+ not only
outperforms the state-of-the art systems, but does so using a sublinear pass over the documents with
the LLM. This equates to a 110× reduction in the number of tokens the LLM needs to process,
averaged across 16 real-world evaluation settings of 10k documents each.


A task for LMs to perform Information Extraction, as implemented by Based.

Homepage: https://github.com/HazyResearch/based-evaluation-harness


Description:
> SWDE (Information Extraction). The task in the SWDE benchmark is to extract semi-structured relations from raw HTML websites. For example, given an IMBD page for a movie (e.g. Harry Potter and the Sorcerer’s Stone) and a relation key (e.g. release date), the model must extract the correct relation value (e.g. 2001). The SWDE benchmark was originally curated by Lockard et al. for the task of open information extraction from the semi-structured web. Because we are evaluating the zero-shot capabilities of relatively small language models, we adapt the task to make it slightly easier. Our task setup is similar after to that used in Arora et al.

### Citation

```
@misc{arora2024simple,
      title={Simple linear attention language models balance the recall-throughput tradeoff},
      author={Simran Arora and Sabri Eyuboglu and Michael Zhang and Aman Timalsina and Silas Alberti and Dylan Zinsley and James Zou and Atri Rudra and Christopher Ré},
      year={2024},
      eprint={2402.18668},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{arora2023language,
      title={Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes},
      author={Simran Arora and Brandon Yang and Sabri Eyuboglu and Avanika Narayan and Andrew Hojel and Immanuel Trummer and Christopher Ré},
      year={2023},
      eprint={2304.09433},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{lockard-etal-2019-openceres,
    title = "{O}pen{C}eres: {W}hen Open Information Extraction Meets the Semi-Structured Web",
    author = "Lockard, Colin  and
      Shiralkar, Prashant  and
      Dong, Xin Luna",
    editor = "Burstein, Jill  and
      Doran, Christy  and
      Solorio, Thamar",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1309",
    doi = "10.18653/v1/N19-1309",
    pages = "3047--3056",
    abstract = "Open Information Extraction (OpenIE), the problem of harvesting triples from natural language text whose predicate relations are not aligned to any pre-defined ontology, has been a popular subject of research for the last decade. However, this research has largely ignored the vast quantity of facts available in semi-structured webpages. In this paper, we define the problem of OpenIE from semi-structured websites to extract such facts, and present an approach for solving it. We also introduce a labeled evaluation dataset to motivate research in this area. Given a semi-structured website and a set of seed facts for some relations existing on its pages, we employ a semi-supervised label propagation technique to automatically create training data for the relations present on the site. We then use this training data to learn a classifier for relation extraction. Experimental results of this method on our new benchmark dataset obtained a precision of over 70{\%}. A larger scale extraction experiment on 31 websites in the movie vertical resulted in the extraction of over 2 million triples.",
}
```

### Groups and Tasks

#### Tasks

* `swde`: the SWDE task as implemented in the paper "Simple linear attention language models balance the recall-throughput tradeoff". Designed for zero-shot evaluation of small LMs.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
