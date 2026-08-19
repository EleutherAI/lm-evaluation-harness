# BEAR

Dataset to evaluate common factual knowledge in language models. This dataset was created as part of the [paper "BEAR: A Unified Framework for Evaluating Relational Knowledge in Causal and Masked Language Models"](https://aclanthology.org/2024.findings-naacl.155/).

For more information visit the [LM Pub Quiz website](https://lm-pub-quiz.github.io/).

## Paper

Title: BEAR: A Unified Framework for Evaluating Relational Knowledge in Causal and Masked Language Models

Abstract: https://aclanthology.org/2024.findings-naacl.155/

Knowledge probing assesses to which degree a language model (LM) has successfully learned relational knowledge during pre-training. Probing is an inexpensive way to compare LMs of different sizes and training configurations. However, previous approaches rely on the objective function used in pre-training LMs and are thus applicable only to masked or causal LMs. As a result, comparing different types of LMs becomes impossible. To address this, we propose an approach that uses an LM’s inherent ability to estimate the log-likelihood of any given textual statement. We carefully design an evaluation dataset of 7,731 instances (40,916 in a larger variant) from which we produce alternative statements for each relational fact, one of which is correct. We then evaluate whether an LM correctly assigns the highest log-likelihood to the correct statement. Our experimental evaluation of 22 common LMs shows that our proposed framework, BEAR, can effectively probe for knowledge across different LM types. We release the BEAR datasets and an open-source framework that implements the probing approach to the research community to facilitate the evaluation and development of LMs.

Homepage: https://lm-pub-quiz.github.io/

### Citation

```text
@inproceedings{wiland-etal-2024-bear,
    title = "{BEAR}: A Unified Framework for Evaluating Relational Knowledge in Causal and Masked Language Models",
    author = "Wiland, Jacek  and
      Ploner, Max  and
      Akbik, Alan",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.155/",
    doi = "10.18653/v1/2024.findings-naacl.155",
    pages = "2393--2411",
    abstract = "Knowledge probing assesses to which degree a language model (LM) has successfully learned relational knowledge during pre-training. Probing is an inexpensive way to compare LMs of different sizes and training configurations. However, previous approaches rely on the objective function used in pre-training LMs and are thus applicable only to masked or causal LMs. As a result, comparing different types of LMs becomes impossible. To address this, we propose an approach that uses an LM{'}s inherent ability to estimate the log-likelihood of any given textual statement. We carefully design an evaluation dataset of 7,731 instances (40,916 in a larger variant) from which we produce alternative statements for each relational fact, one of which is correct. We then evaluate whether an LM correctly assigns the highest log-likelihood to the correct statement. Our experimental evaluation of 22 common LMs shows that our proposed framework, BEAR, can effectively probe for knowledge across different LM types. We release the BEAR datasets and an open-source framework that implements the probing approach to the research community to facilitate the evaluation and development of LMs."
}
```

### Groups, Tags, and Tasks

#### Groups

* Not part of a group yet.

#### Tags

* No specific tags yet

#### Tasks

* `bear`: BEAR knowledge probe (factual multiple choice task)
* `bear_big`: Big variant of BEAR (more questions, larger answer spaces)



### Changelog
