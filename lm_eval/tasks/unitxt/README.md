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

* `unitxt`:  Subset of Unitxt tasks that were not in LM-Eval Harness task catalog, including new types of tasks like multi-label classification, grammatical error correction, named entity extraction.

#### Tasks

The full list of Unitxt tasks currently supported can be seen under `tasks/unitxt` directory.

### Adding tasks

You can add additional tasks from the Unitxt catalog by generating new LM-Eval yaml files for these datasets.

The Unitxt task yaml files are generated via the `generate_yamls.py` script in the `tasks/unitxt` directory.

To add a yaml file for an existing dataset Unitxt which is not yet in LM-Eval:
1. Add the card name to the `unitxt_datasets`  file in the `tasks/unitxt` directory.  
2. The generate_yaml.py contains the default Unitxt [template](https://unitxt.readthedocs.io/en/latest/docs/adding_template.html) used for each kind of NLP task in the `default_template_per_task` dictionary.  If the dataset is of a Unitxt task type, previously not used in LM-Eval, you will need to add a default template for it in the dictionary.  

```
default_template_per_task = {
     "tasks.classification.multi_label" : "templates.classification.multi_label.title" ,
     "tasks.classification.multi_class" : "templates.classification.multi_class.title" ,
     "tasks.summarization.abstractive" :  "templates.summarization.abstractive.full",
     "tasks.regression.two_texts" : "templates.regression.two_texts.simple",
     "tasks.qa.with_context.extractive" : "templates.qa.with_context.simple",
     "tasks.grammatical_error_correction" : "templates.grammatical_error_correction.simple",
     "tasks.span_labeling.extraction" : "templates.span_labeling.extraction.title"
}
```
3. Run `python generate_yaml.py` (this will generate all the datasets listed in the `unitxt_datasets`)

If you want to add a new dataset to the Unitxt catalog, see the Unitxt documentation:

https://unitxt.readthedocs.io/en/latest/docs/adding_dataset.html



### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
