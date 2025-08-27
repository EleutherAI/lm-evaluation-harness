#  BHS: Controlled Evaluation of Syntactic Knowledge in Basque, Hindi, and Swahili

## Paper

Title: Controlled Evaluation of Syntactic Knowledge in Multilingual Language Models

Abstract:

> Language models (LMs) are capable of acquiring elements of human-like syntactic knowledge. Targeted syntactic evaluation tests have been employed to measure how well they form generalizations about syntactic phenomena in high-resource languages such as English. However, we still lack a thorough understanding of LMs' capacity for syntactic generalizations in low-resource languages, which are responsible for much of the diversity of syntactic patterns worldwide. In this study, we develop targeted syntactic evaluation tests for three low-resource languages (Basque, Hindi, and Swahili) and use them to evaluate five families of open-access multilingual Transformer LMs. We find that some syntactic tasks prove relatively easy for LMs while others (agreement in sentences containing indirect objects in Basque, agreement across a prepositional phrase in Swahili) are challenging. We additionally uncover issues with publicly available Transformers, including a bias toward the habitual aspect in Hindi in multilingual BERT and underperformance compared to similar-sized models in XGLM-4.5B. ([Kryvosheieva & Levy, 2025](https://aclanthology.org/2025.loreslm-1.30/))


Homepage: https://github.com/dariakryvosheieva/syntactic_generalization_multilingual

### Citation

```
@inproceedings{kryvosheieva-levy-2025-controlled,
    title = "Controlled Evaluation of Syntactic Knowledge in Multilingual Language Models",
    author = "Kryvosheieva, Daria and Levy, Roger",
    editor = "Hettiarachchi, Hansi and Ranasinghe, Tharindu and Rayson, Paul and Mitkov, Ruslan and Gaber, Mohamed and Premasiri, Damith and Tan, Fiona Anting and Uyangodage, Lasitha",
    booktitle = "Proceedings of the First Workshop on Language Models for Low-Resource Languages",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.loreslm-1.30/",
    pages = "402--413"
}
```

### Groups, Tags, and Tasks

* `bhs_basque`: Run all Basque tasks (listed below) and calculate mean performance. In all tasks, the goal is for the model to predict the auxiliary verb (AUX) that correctly agrees with the subject (S), direct object (DO), and indirect object (IO). Each task manipulates a different one of these, e.g., for `bhs__basque__DO__S_IO_DO_V_AUX`, the two presented sentences (with `S_IO_DO_V_AUX` structure) have auxiliary verbs that agree with the subject and indirect object, and the task is to correctly assign the one that also agrees with the direct object (DO) a higher probability than the one that does not. For specific examples, see [Kryvosheieva & Levy (2025)](https://aclanthology.org/2025.loreslm-1.30/).
    * `bhs__basque__DO__S_DO_V_AUX`
    * `bhs__basque__DO__S_IO_DO_V_AUX`
    * `bhs__basque__IO__IO_S_V_AUX`
    * `bhs__basque__IO__S_IO_DO_V_AUX`
    * `bhs__basque__S__IO_S_V_AUX`
    * `bhs__basque__S__S_DO_V_AUX`
    * `bhs__basque__S__S_IO_DO_V_AUX`
    * `bhs__basque__S__S_V_AUX`

* `bhs_hindi`: Run all Hindi tasks (listed below) and calculate mean performance. In all tasks, the goal is for the model to predict that in a sentence with the 'ne' clitic, the final verb should be in a perfective form, and in sentences without, it should be in a non-perfective form (in this case, habitual or progressive) by assigning a higher probability to the correct verb. For specific examples, see [Kryvosheieva & Levy (2025)](https://aclanthology.org/2025.loreslm-1.30/).
    * `bhs__hindi__S_O_V`
    * `bhs__hindi__S_PossPRN_O_V`
    * `bhs__hindi__S_PossPRN_PossN_O_V`
    * `bhs__hindi__S_ne_O_V`
    * `bhs__hindi__S_ne_PossPRN_O_V`
    * `bhs__hindi__S_ne_PossPRN_PossN_O_V`

* `bhs_swahili`:  Run all Swahili tasks (listed below) and calculate mean performance. In all tasks, the goal is for the model to assign the final word - a verb (V) or adjective (A/AN) a higher probability if it correctly agrees with the initial noun (in terms of noun class) than if it does not. For specific examples, see [Kryvosheieva & Levy (2025)](https://aclanthology.org/2025.loreslm-1.30/).
    * `bhs__swahili__N_of_Poss_D_AP_V_ni_AN`
    * `bhs__swahili__N_of_Poss_D_AP_ni_AN`
    * `bhs__swahili__N_of_Poss_D_A_V`
    * `bhs__swahili__N_of_Poss_D_A_V1_V2`
    * `bhs__swahili__N_of_Poss_D_V`
    * `bhs__swahili__N_of_Poss_D_ni_A`
    * `bhs__swahili__N_of_Poss_V`
    * `bhs__swahili__N_of_Poss_ni_A`


**Implementation Note:**  The [original implementation](https://github.com/dariakryvosheieva/syntactic_generalization_multilingual) normalizes the log-probability of the final word by its length in number of tokens, which is not supported by the Language Model Evaluation Harness (see [[1](https://blog.eleuther.ai/multiple-choice-normalization/)], [[2](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)], [[3](https://github.com/EleutherAI/lm-evaluation-harness/issues/1396)]). For this reason, the implementation provided here includes both the `acc` (accuracy based on comparing the unnormalized log-probability of the correct and incorrect versions of each sentence) and `acc_norm` (the same as `acc` but with sentence log-probability normalized by number of bytes) metrics.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


### Changelog
