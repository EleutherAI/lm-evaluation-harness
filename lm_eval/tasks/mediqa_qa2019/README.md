# MEDIQA_QA 2019

### Paper

Title: `Overview of the MEDIQA 2019 Shared Task on Textual Inference, Question Entailment and Question Answering`

Abstract: [https://aclanthology.org/W19-5039/](https://aclanthology.org/W19-5039/)

Open-ended medical Question-Answering stemming from the MEDIQA 2019 open challenge.

Homepage: \
[https://sites.google.com/view/mediqa2019](https://sites.google.com/view/mediqa2019)


#### Tasks

* `mediqa_qa2019`: Open-ended QA in english.
* `mediqa_qa2019_perplexity`: Open-Ended QA in english, evaluated with perplexity.

### Dataset loading

Both tasks read the source-schema Parquet export published in `bigbio/mediqa_qa`
at revision `789036735c8a87ae9dd2621153e0e4a7d9b0b898`. Selecting the four
`mediqa_qa_source` files explicitly preserves the original script's default
schema and split names without executing a dataset loading script on
`datasets>=4`.

The ordered records match the original MEDIQA 2019 XML files: 104 in
`train_live_qa_med`, 104 in `train_alexa`, 25 in `validation`, and 150 in `test`.
Training/few-shot selection remains `train_live_qa_med`; prompts, first-answer
references, scoring, and task version 1.0 are unchanged.

### Citation

```bibtex
@inproceedings{ben-abacha-etal-2019-overview,
    title = "Overview of the {MEDIQA} 2019 Shared Task on Textual Inference, Question Entailment and Question Answering",
    author = "Ben Abacha, Asma  and
      Shivade, Chaitanya  and
      Demner-Fushman, Dina",
    editor = "Demner-Fushman, Dina  and
      Cohen, Kevin Bretonnel  and
      Ananiadou, Sophia  and
      Tsujii, Junichi",
    booktitle = "Proceedings of the 18th BioNLP Workshop and Shared Task",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-5039/",
    doi = "10.18653/v1/W19-5039",
    pages = "370--379",
    abstract = "This paper presents the MEDIQA 2019 shared task organized at the ACL-BioNLP workshop. The shared task is motivated by a need to develop relevant methods, techniques and gold standards for inference and entailment in the medical domain, and their application to improve domain specific information retrieval and question answering systems. MEDIQA 2019 includes three tasks: Natural Language Inference (NLI), Recognizing Question Entailment (RQE), and Question Answering (QA) in the medical domain. 72 teams participated in the challenge, achieving an accuracy of 98{\%} in the NLI task, 74.9{\%} in the RQE task, and 78.3{\%} in the QA task. In this paper, we describe the tasks, the datasets, and the participants' approaches and results. We hope that this shared task will attract further research efforts in textual inference, question entailment, and question answering in the medical domain."
}
```
