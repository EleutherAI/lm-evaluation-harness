# TurBLiMP: A Turkish Benchmark of Linguistic Minimal Pairs

## Paper

Title: TurBLiMP: A Turkish Benchmark of Linguistic Minimal Pairs

Abstract:

> TurBLiMP is the first Turkish benchmark of linguistic minimal pairs, designed to evaluate the linguistic abilities of monolingual and multilingual language models. The dataset covers 16 core grammatical phenomena in Turkish, with 1,000 minimal pairs per phenomenon.

Homepage: https://github.com/ezgibasar/TurBLiMP

### Citation

```
bibtex
@misc{basar2025turblimpturkishbenchmarklinguistic,
  title={TurBLiMP: A Turkish Benchmark of Linguistic Minimal Pairs},
  author={Ezgi Ba{\c{s}}ar and Francesca Padovani and Jaap Jumelet and Arianna Bisazza},
  year={2025},
  eprint={2506.13487},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2506.13487}
}
```

### Groups, Tags, and Tasks

#### Groups

* `turblimp_core`: Runs all 16 grammatical 'core' subtasks of TurBLiMP (additional experimental paradigms which have no correct answer are included in the original release; these are not included here).

#### Tasks

* `turblimp_anaphor_agreement`: Reflexive pronoun agreement violations
* `turblimp_argument_structure_transitive`: Case marking errors with transitive verbs
* `turblimp_argument_structure_ditransitive`: Case marking errors with ditransitive verbs
* `turblimp_binding`: Principle B violations in binding theory
* `turblimp_determiners`: Obligatory use of the indefinite article
* `turblimp_ellipsis`: Backward gapping with non-parallel word orders
* `turblimp_irregular_forms`: Incorrect aorist allomorph usage
* `turblimp_island_effects`: Wh-adjunct extraction from complex NPs
* `turblimp_nominalization`: Incorrect nominalization suffix selection
* `turblimp_npi_licensing`: Negative polarity items in non-negative contexts
* `turblimp_passives`: Unlicensed use of by-phrases in impersonal passives
* `turblimp_quantifiers`: Quantifier usage with bare nouns
* `turblimp_relative_clauses`: Incorrect case marking in relative clauses
* `turblimp_scrambling`: Illicit postverbal scrambling from embedded clauses
* `turblimp_subject_agreement`: Person/number agreement violations
* `turblimp_suspended_affixation`: Improper tense suffix suspension

**Implementation Note:**  The [original implementation](https://github.com/ezgibasar/TurBLiMP) normalizes length by number of tokens, which is not supported by the Language Model Evaluation Harness (see [[1](https://blog.eleuther.ai/multiple-choice-normalization/)], [[2](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)], [[3](https://github.com/EleutherAI/lm-evaluation-harness/issues/1396)]). For this reason, the implementation provided here includes both the `acc` (accuracy based on comparing the unnormalized log-probability of the correct and incorrect versions of each sentence) and `acc_norm` (the same as `acc` but with sentence log-probability normalized by number of bytes) metrics.


### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


### Changelog
