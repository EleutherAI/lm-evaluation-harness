# k_mmlu

### Paper

Title: `K-MMLU(work in progress) `
* ongoing project at publishing help/ non-english benchmark

Abstract: `The K-MMLU (Korean-MMLU) is a comprehensive suite designed to evaluate the advanced knowledge and reasoning abilities of large language models (LLMs) within the Korean language and cultural context. This suite encompasses 45 topics, primarily focusing on expert-level subjects. It includes general subjects like Physics and Ecology, and law and political science, alongside specialized fields such as Non-Destructive Training and Maritime Engineering. The datasets are derived from Korean licensing exams, with about 90% of the questions including human accuracy based on the performance of human test-takers in these exams. K-MMLU is segmented into training, testing, and development subsets, with the test subset ranging from a minimum of 100 to a maximum of 1000 questions, totaling 35,000 questions. Additionally, a set of 10 questions is provided as a development set for few-shot exemplar development. At total, K-MMLU consists of 254,334 instances.`

Homepage: https://huggingface.co/datasets/HAERAE-HUB/K-MMLU-Preview

### Citation

```
We'll be updating this section soon.
```

### Groups and Tasks

#### Groups

* `kmmlu`: 'All 45 subjects of the KMMLU dataset, evaluated following the methodology in MMLU's original implementation'

#### Tasks

The following tasks evaluate subjects in the KMMLU dataset
- `kmmlu_{subject_english}`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
