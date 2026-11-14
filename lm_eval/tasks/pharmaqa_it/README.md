# PharmaQA.IT

## Paper

Title: `PharmaQA.IT: an Italian dataset for Q&A in the pharmaceutical domain`

Paper: https://aclanthology.org/2026.eacl-industry.70/

PharmaQA.IT is an Italian extractive question-answering benchmark of 861 expert-checked
question-answer pairs over official medicine leaflets (Summary of Product Characteristics, RCP)
published by the Italian Medicines Agency (AIFA).

Homepage: https://github.com/KamyarZeinalipour/PharmaQA.IT

Data: https://huggingface.co/datasets/Kamyar-zeinalipour/PharmaQA.IT

### Citation

```text
@inproceedings{zeinalipour-etal-2026-pharmaqa,
    title = "{P}harma{QA}.{IT}: an {I}talian dataset for {Q}{\&}{A} in the pharmaceutical domain",
    author = "Zeinalipour, Kamyar and Zugarini, Andrea and Zanollo, Asya and Rigutini, Leonardo",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 5: Industry Track)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-industry.70/",
    doi = "10.18653/v1/2026.eacl-industry.70",
    pages = "937--947"
}
```

### Groups, Tags, and Tasks

#### Tasks

* `pharmaqa_it`: the model reads the full leaflet and answers the question; SQuAD-style exact match and token F1 with Italian article removal.

The leaflets are long (about 49,000 characters on average, up to 250,000), so this is a
long-context task: use a model and `max_length` that fit the documents, otherwise the start of the
document is truncated.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version.
