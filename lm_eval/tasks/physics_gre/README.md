# Physics GRE

### Paper

Title: Inflection-2.5 / Inflection-Benchmarks

Homepage: https://github.com/InflectionAI/Inflection-Benchmarks#physics-gre

The Physics GRE benchmark consists of processed Physics GRE exams, a common
graduate-school entrance exam for physics students, released by Inflection AI.
Each question is a five-way (A–E) multiple-choice problem. The dataset exposes
three fields per item: `input` (the question, with answer options inlined),
`target_scores` (a mapping where the correct option has a score of 1), and
`has_image` (whether the question depends on a diagram).

Following the original methodology, only image-free questions are scored
("we include only questions without an image in our scoring"), so items with
`has_image` are filtered out by `process_docs`.

The data is loaded from the [`shayekh/physics_gre`](https://huggingface.co/datasets/shayekh/physics_gre)
Hub mirror, which was verified to match the original Inflection-Benchmarks
release exactly (`input` / `target_scores` / `has_image`).


### Citation

```
@misc{inflection2024benchmarks,
  title  = {Inflection-Benchmarks},
  author = {Inflection AI},
  year   = {2024},
  howpublished = {\url{https://github.com/InflectionAI/Inflection-Benchmarks}}
}
```

### Groups, Tags, and Tasks

#### Groups

None.

#### Tasks

* `physics_gre`: Exam GR8677, the split Inflection reports results on (image-free items).
* `physics_gre_additional`: Three additional exams (GR9277, GR9677, GR0177).
* `physics_gre_all`: The union of all four exams.

The "Main" variant is `physics_gre` (GR8677), matching the split used in the
original report.

### Scoring note

The original benchmark reports a percentile derived from a GRE-style raw score
(+1 per correct answer, −0.25 per incorrect answer). These tasks report standard
`acc` over image-free items; the GRE penalty scoring is not applied.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

The original release provides data and a scoring formula but no runnable
reference implementation; the methodology (image-free scoring) is matched here.

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
