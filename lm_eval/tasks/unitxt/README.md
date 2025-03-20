# Unitxt

Unitxt is a library for customizable textual data preparation and evaluation tailored to generative language models. Unitxt natively integrates with common libraries like HuggingFace and LM-eval-harness and deconstructs processing flows into modular components, enabling easy customization and sharing between practitioners. These components encompass model-specific formats, task prompts, and many other comprehensive dataset processing definitions. These components are centralized in the Unitxt-Catalog, thus fostering collaboration and exploration in modern textual data workflows.

The full Unitxt catalog can be viewed in an [online explorer](https://unitxt.readthedocs.io/en/latest/docs/demo.html).

Read more about Unitxt at [www.unitxt.ai](https://www.unitxt.ai/).

To use Unitxt dataset with lm-eval, you should first install unitxt via 'pip install unitxt'.

### Paper

Title: `Unitxt: Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI`
Abstract: [link](https://arxiv.org/abs/2401.14019)



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

See the [adding tasks guide](https://www.unitxt.ai/en/latest/docs/lm_eval.html#).
