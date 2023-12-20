# GLUE
**NOTE**: GLUE benchmark tasks do not provide publicly accessible labels for their test sets, so we default to the validation sets for all sub-tasks.

### Paper

Title: `GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding`

Abstract: https://openreview.net/pdf?id=rJ4km2R5t7

The General Language Understanding Evaluation (GLUE) benchmark is a collection of
resources for training, evaluating, and analyzing natural language understanding
systems. GLUE consists of:
- A benchmark of nine sentence- or sentence-pair language understanding tasks built
on established existing datasets and selected to cover a diverse range of dataset
sizes, text genres, and degrees of difficulty, and
- A diagnostic dataset designed to evaluate and analyze model performance with
respect to a wide range of linguistic phenomena found in natural language.

Homepage: https://gluebenchmark.com/

### Citation

```
@inproceedings{wang-etal-2018-glue,
    title = "{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding",
    author = "Wang, Alex  and
      Singh, Amanpreet  and
      Michael, Julian  and
      Hill, Felix  and
      Levy, Omer  and
      Bowman, Samuel",
    booktitle = "Proceedings of the 2018 {EMNLP} Workshop {B}lackbox{NLP}: Analyzing and Interpreting Neural Networks for {NLP}",
    month = nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-5446",
    doi = "10.18653/v1/W18-5446",
    pages = "353--355",
    abstract = "Human ability to understand language is \textit{general, flexible, and robust}. In contrast, most NLU models above the word level are designed for a specific task and struggle with out-of-domain data. If we aspire to develop models with understanding beyond the detection of superficial correspondences between inputs and outputs, then it is critical to develop a unified model that can execute a range of linguistic tasks across different domains. To facilitate research in this direction, we present the General Language Understanding Evaluation (GLUE, gluebenchmark.com): a benchmark of nine diverse NLU tasks, an auxiliary dataset for probing models for understanding of specific linguistic phenomena, and an online platform for evaluating and comparing models. For some benchmark tasks, training data is plentiful, but for others it is limited or does not match the genre of the test set. GLUE thus favors models that can represent linguistic knowledge in a way that facilitates sample-efficient learning and effective knowledge-transfer across tasks. While none of the datasets in GLUE were created from scratch for the benchmark, four of them feature privately-held test data, which is used to ensure that the benchmark is used fairly. We evaluate baselines that use ELMo (Peters et al., 2018), a powerful transfer learning technique, as well as state-of-the-art sentence representation models. The best models still achieve fairly low absolute scores. Analysis with our diagnostic dataset yields similarly weak performance over all phenomena tested, with some exceptions.",
}
```

### Groups and Tasks

#### Groups

* `glue`: Run all Glue subtasks.

#### Tasks

* `cola`
* `mnli`
* `mrpc`
* `qnli`
* `qqp`
* `rte`
* `sst`
* `wnli`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
