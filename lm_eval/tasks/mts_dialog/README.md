# MTS-Dialog

### Paper

Title: `An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters`

Abstract: [https://aclanthology.org/2023.eacl-main.168/](https://aclanthology.org/2023.eacl-main.168/)

MTS-Dialog is a collection of 1,700 doctor-patient dialogues and corresponding clinical notes.
This task implements open-ended Question Answering (QA) on MTS-Dialog.


#### Tasks

* `mts_dialog`: Open-Ended QA in english.
* `mts_dialog_perplexity`: Open-Ended QA in english, evaluated with perplexity.

### Citation

```bibtex
@inproceedings{ben-abacha-etal-2023-empirical,
    title = "An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters",
    author = "Ben Abacha, Asma  and
      Yim, Wen-wai  and
      Fan, Yadan  and
      Lin, Thomas",
    editor = "Vlachos, Andreas  and
      Augenstein, Isabelle",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.168/",
    doi = "10.18653/v1/2023.eacl-main.168",
    pages = "2291--2302",
    abstract = "Medical doctors spend on average 52 to 102 minutes per day writing clinical notes from their patient encounters (Hripcsak et al., 2011). Reducing this workload calls for relevant and efficient summarization methods. In this paper, we introduce new resources and empirical investigations for the automatic summarization of doctor-patient conversations in a clinical setting. In particular, we introduce the MTS-Dialog dataset; a new collection of 1,700 doctor-patient dialogues and corresponding clinical notes. We use this new dataset to investigate the feasibility of this task and the relevance of existing language models, data augmentation, and guided summarization techniques. We compare standard evaluation metrics based on n-gram matching, contextual embeddings, and Fact Extraction to assess the accuracy and the factual consistency of the generated summaries. To ground these results, we perform an expert-based evaluation using relevant natural language generation criteria and task-specific criteria such as critical omissions, and study the correlation between the automatic metrics and expert judgments. To the best of our knowledge, this study is the first attempt to introduce an open dataset of doctor-patient conversations and clinical notes, with detailed automated and manual evaluations of clinical note generation."
}
```
