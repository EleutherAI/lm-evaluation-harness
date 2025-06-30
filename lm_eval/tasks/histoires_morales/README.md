# Histoires Morales

### Paper

Title: `Histoires Morales: A French Dataset for Assessing Moral Alignment`

Abstract: `https://arxiv.org/pdf/2501.17117`

⚖ Histoires Morales is the first dataset for moral model alignment evaluation in French. It consists of narratives describing normative and norm-divergent actions taken by individuals to achieve certain intentions in concrete situations, along with their respective consequences.
Each of the 12,000 stories (histoires) follows the same seven-sentence structure as the Moral Stories dataset:

Context:

1. Norm: A guideline for social conduct generally observed by most people in everyday situations.
2. Situation: The setting of the story, introducing participants and describing their environment.
3. Intention: A reasonable goal that one of the story participants (the actor) wants to achieve.

Normative path:
4. Normative action: An action by the actor that fulfills the intention while observing the norm.
5. Normative consequence: A possible effect of the normative action on the actor’s environment.

Norm-divergent path:
6. Divergent action: An action by the actor that fulfills the intention but diverges from the norm.
7. Divergent consequence: A possible effect of the divergent action on the actor’s environment.

Histoires Morales is adapted to French from the widely used Moral Stories dataset.
We translated the Moral Stories dataset and refined these translations through manual annotations.
See paper for more details.

Homepage: `https://huggingface.co/datasets/LabHC/histoires_morales`


### Citation

Coming soon (accepted to NAACL 2025)

### Groups, Tags, and Tasks

#### Groups

* Not part of a group yet

#### Tags

No tags, since there is a single task.

#### Tasks

* `histoires_morales.yaml`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
