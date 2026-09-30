# Canine Geroscience Questions

## Dataset

Title: `canine-geroscience-questions` (v0.1)

Homepage: https://huggingface.co/datasets/w0lph/canine-geroscience-questions

Source and tooling: https://github.com/w0lph/k9/tree/main/questions

A short-answer question set on the biology of aging in companion dogs, an emerging model
for human geroscience (shared environment, naturally occurring age-related disease, and
lifespan outcomes measurable in years rather than decades). 133 questions across ten
categories: lifespan epidemiology, body size and genetics, interventions, biomarkers and
epigenetic clocks, cognition, frailty and quality of life, Dog Aging Project methods, the
translational model, immunity and microbiome, and disease and mortality.

Every question has a concise gold answer and at least one piece of evidence: a record key in
a versioned corpus of the canine aging literature (Europe PMC) plus a verbatim quote that a
validator checks against that record's abstract or full text. Items were drafted by model
agents under fixed authoring rules, machine-validated for grounding, then reviewed
item-by-item by an independent model pass. No domain expert has reviewed the set; treat it as
a grounded, non-adversarial factual probe of a niche scientific field rather than an expert
benchmark.

Answer types: numeric (58), short_text (36), list (28), categorical (9), boolean (2).
Difficulty: 1 = stated in an abstract (42), 2 = needs a results section or two facts (60),
3 = synthesis or a non-obvious detail (31).

Licence: CC BY 4.0 (questions and answers); quotes remain under the licence of their source
article and are limited to short spans.

### Citation

```text
@misc{canine_geroscience_questions_2026,
  title        = {Canine Geroscience Questions: a verbatim-grounded question set on aging in companion dogs},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/w0lph/canine-geroscience-questions}},
  note         = {Version 0.1}
}
```

### Groups, Tags, and Tasks

#### Tags

* `canine_geroscience_tasks`: both tasks below.

#### Tasks

* `canine_geroscience`: all 133 questions, free-form generation, scored with normalised
  exact match and SQuAD-style token F1 against the gold answer. Zero-shot by default; three
  hand-written few-shot examples (not drawn from the evaluation set) are available via
  `--num_fewshot 3`.
* `canine_geroscience_numeric`: the 58 numeric questions, scored with `numeric_acc` (the gold
  answer's primary figure, i.e. its first number outside parentheses, appears in the model's
  answer; parentheticals hold confidence intervals), `numeric_acc_all` (every number outside
  parentheses appears, for multi-part answers such as a dose plus a frequency) and exact match.

Gold answers are short spans rather than single tokens, so token F1 is the more informative
headline metric for the full set; `numeric_acc` is the strict metric where an unambiguous
one exists.

### Checklist

For adding novel benchmarks/datasets to the library:

* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

The dataset has no paper; its construction, validator and review process are documented in
the dataset card and the repository linked above. There is no reference implementation.

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted? (`canine_geroscience`)
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant? (none)

### Changelog

* v1.0 (2026-09-30): initial implementation against dataset v0.1.
