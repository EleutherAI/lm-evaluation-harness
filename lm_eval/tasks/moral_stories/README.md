# Moral Stories

### Paper

Title: `Moral Stories: Situated Reasoning about Norms, Intents, Actions, and their Consequences`

Abstract: `https://aclanthology.org/2021.emnlp-main.54/`

Moral Stories is a crowd-sourced dataset of structured narratives that describe normative and norm-divergent actions taken by individuals to accomplish certain intentions in concrete situations, and their respective consequences. All stories in the dataset consist of seven sentences, belonging to the following categories:

- Norm: A guideline for social conduct generally observed by most people in everyday situations.
- Situation: Setting of the story that introduces story participants and describes their environment.
- Intention: Reasonable goal that one of the story participants (the actor), wants to fulfill.
- Normative action: An action by the actor that fulfills the intention and observes the norm.
- Normative consequence: Possible effect of the normative action on the actor's environment.
- Divergent action: An action by the actor that fulfills the intention and diverges from the norm.
- Divergent consequence: Possible effect of the divergent action on the actor's environment.


Homepage: `https://github.com/demelin/moral_stories`

The implementation is based on the paper "Histoires Morales: A French Dataset for Assessing Moral Alignment." The source code is available at: `https://github.com/upunaprosk/histoires-morales`.

### Citation

```
@inproceedings{emelin-etal-2021-moral,
    title = "Moral Stories: Situated Reasoning about Norms, Intents, Actions, and their Consequences",
    author = "Emelin, Denis  and
      Le Bras, Ronan  and
      Hwang, Jena D.  and
      Forbes, Maxwell  and
      Choi, Yejin",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.54",
    doi = "10.18653/v1/2021.emnlp-main.54",
    pages = "698--718",
    abstract = "In social settings, much of human behavior is governed by unspoken rules of conduct rooted in societal norms. For artificial systems to be fully integrated into social environments, adherence to such norms is a central prerequisite. To investigate whether language generation models can serve as behavioral priors for systems deployed in social settings, we evaluate their ability to generate action descriptions that achieve predefined goals under normative constraints. Moreover, we examine if models can anticipate likely consequences of actions that either observe or violate known norms, or explain why certain actions are preferable by generating relevant norm hypotheses. For this purpose, we introduce Moral Stories, a crowd-sourced dataset of structured, branching narratives for the study of grounded, goal-oriented social reasoning. Finally, we propose decoding strategies that combine multiple expert models to significantly improve the quality of generated actions, consequences, and norms compared to strong baselines.",
}
```

### Groups, Tags, and Tasks

#### Groups

* Not part of a group yet

#### Tags

* `moral_stories`: `Evaluation of the likelihoods of moral actions versus immoral actions. Accuracy is computed as the ratio of preferred moral actions based on their likelihood.`

#### Tasks

* `moral_stories.yaml`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
