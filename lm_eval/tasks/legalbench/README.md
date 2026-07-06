# LegalBench

### Paper

Title: LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models

Abstract: https://arxiv.org/abs/2308.11462

Homepage: https://hazyresearch.stanford.edu/legalbench/

LegalBench is a collaboratively built benchmark of legal reasoning tasks. The
full benchmark contains 162 tasks; this implementation covers the five-task
subset used by HELM, selected to represent distinct legal reasoning patterns.
Prompts (instructions, field ordering, and output nouns) follow the official
`HazyResearch/legalbench` HELM prompt settings, and data is loaded from the
[`nguha/legalbench`](https://huggingface.co/datasets/nguha/legalbench) Hub
dataset. Tasks are evaluated zero-/few-shot with `generate_until` and
case-/punctuation-insensitive exact match; the `train` split provides the
few-shot examples.

### Citation

```
@misc{guha2023legalbench,
  title  = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  year   = {2023},
  eprint = {2308.11462},
  archivePrefix = {arXiv},
}
```

### Groups, Tags, and Tasks

#### Groups

* `legalbench`: The five-task HELM subset, aggregated by mean exact match.

#### Tasks

* `legalbench_abercrombie`: Classify trademark distinctiveness (generic / descriptive / suggestive / arbitrary / fanciful).
* `legalbench_corporate_lobbying`: Judge whether a Congressional bill is relevant to a company (Yes / No).
* `legalbench_function_of_decision_section`: Classify the function of a judicial opinion paragraph (Facts / Procedural History / Issue / Rule / Analysis / Conclusion / Decree).
* `legalbench_international_citizenship_questions`: Answer questions about international citizenship law (Yes / No).
* `legalbench_proa`: Decide whether a statutory clause grants a private right of action (Yes / No).

### Implementation notes

This covers the HELM-lite subset, not all 162 LegalBench tasks. The official
benchmark reports per-task balanced accuracy; here we report exact match
(case- and punctuation-insensitive) over the `test` split, matching the HELM
generation setup. The broader paper categories can be added later.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? The prompts follow the official `HazyResearch/legalbench` HELM settings and HELM's task selection.

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted? (the `legalbench` group)
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant? (HELM-lite)
