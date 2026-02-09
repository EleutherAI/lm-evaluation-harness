# E3C_V3.0_RE_NER

### Paper

We introduce E3C-3.0, a multilingual dataset in the medical domain featuring clinical cases annotated with diseases and test-result relations. The dataset consists of texts translated and projected from an English source into five target languages: Greek, Italian, Polish, Slovak, and Slovenian. A semi-automatic approach was adopted, combining automatic annotation projection using large language models with human revision.

### Citation

```bibtex
@misc{ghosh2025lowresourceinformationextraction,
      title={Low-resource Information Extraction with the European Clinical Case Corpus},
      author={Soumitra Ghosh, Begona Altuna, Saeed Farzi, Pietro Ferrazzi1, Alberto Lavelli, Giulia Mezzanotte, Manuela Speranza, Bernardo Magnini},
      year={2025},
      eprint={2503.20568},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.20568},
}
```

### Groups

- `e3c3.0_re`: All relation extraction tasks for English, Greek, Italian, Polish, Slovak, and Slovenian.
- `e3c3.0_ner`: All Named Entity Recognition  tasks for English, Greek, Italian, Polish, Slovak, and Slovenian

#### Tasks

The following tasks can also be evaluated :
  - `_en_ner_e3c_p1_task`: Engish Named Entity Recognition task
  - `_it_ner_e3c_p1_task`: Italian Named Entity Recognition task
  - `_gr_ner_e3c_p1_task`: Greek Named Entity Recognition task
  - `_sl_ner_e3c_p1_task`: Slovenian Named Entity Recognition task
  - `_pl_ner_e3c_p1_task`: Polish Named Entity Recognition task
  - `_sk_ner_e3c_p1_task`: Slovak Named Entity Recognition task
  - `e3c3.0_re_en_task`: Engish Relation Extraction task
  - `e3c3.0_re_it_task`: Italian Relation Extraction task
  - `e3c3.0_re_gr_task`: Greek Relation Extraction task
  - `e3c3.0_re_sl_task`: Slovenian Relation Extraction task
  - `e3c3.0_re_pl_task`: Polish Relation Extraction task
  - `e3c3.0_re_sk_task`: Slovak Relation Extraction task
 


### Usage

```bash
lm_eval   --model hf --model_args pretrained=microsoft/Phi-3.5-mini-instruct  --tasks _en_ner_e3c_p1_task --device cuda:6 --batch_size auto --trust_remote_code
```

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [x] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
