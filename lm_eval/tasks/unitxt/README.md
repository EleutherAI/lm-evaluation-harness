# Unitxt

### Paper

Title: `Unitxt: Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI`
Abstract: `https://arxiv.org/abs/2401.14019`

Unitxt is a library for customizable textual data preparation and evaluation tailored to generative language models. Unitxt natively integrates with common libraries like HuggingFace and LM-eval-harness and deconstructs processing flows into modular components, enabling easy customization and sharing between practitioners. These components encompass model-specific formats, task prompts, and many other comprehensive dataset processing definitions. These components are centralized in the Unitxt-Catalog, thus fostering collaboration and exploration in modern textual data workflows.

The full Unitxt catalog can be viewed in an online explorer. `https://unitxt.readthedocs.io/en/latest/docs/demo.html`

Homepage: https://unitxt.readthedocs.io/en/latest/index.html

### Citation

```
@misc{unitxt,
      title={Unitxt: Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI},
      author={Elron Bandel and Yotam Perlitz and Elad Venezian and Roni Friedman-Melamed and Ofir Arviv and Matan Orbach and Shachar Don-Yehyia and Dafna Sheinwald and Ariel Gera and Leshem Choshen and Michal Shmueli-Scheuer and Yoav Katz},
      year={2024},
      eprint={2401.14019},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* `unitxt`:  Subset of unitxt tasks that were not in LM-Eval Harness task catalog, including new types of tasks like multi-label classification, grammatical error correction, named entity extraction.

#### Tasks

The full list of tasks can be seen under tasks/unitxt directory.
 

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
