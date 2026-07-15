---
language:
- it
license: cc-by-sa-4.0
task_categories:
- text-classification
- text-ranking
- question-answering
pretty_name: BLiMP-IT
size_categories:
- 10K<n<100K
tags:
- linguistics
- syntax
- morphology
- acceptability
- minimal-pairs
- language-model-evaluation
- italian
---

# BLiMP-IT

# Paper

Title: Language models assessment through linguistically motivated contrasts: A benchmark for Italian (BLiMP-IT)

Abstract:

> BLiMP-IT is a linguistically-informed benchmark to assess the performance of Italian Language Models (LMs). Inspired by state-of-the-art tools for LM evaluation and informed both by generative theorizing and psycholinguistic metrics, this benchmark tests a rich variety of structures using minimal pair contrasts, i.e., a grammatical sentence and an ungrammatical one minimally differing with respect to a single morphosyntactic property. Prompting the model to assign a probability value to the sentences within each pair, BLiMP-IT tests LMs accuracy, as well as their ability to reach linguistically meaningful generalizations, ultimately offering insights on human-machine comparability and the validity of the Poverty of Stimulus hypothesis. ungrammatical sentence. The corpus covers 84 paradigms, classified into 22 syntactic phenomena. Ten sentence pairs of each paradigm were created by hand, while the remaining 90 were generated semi-automatically and manually validated afterwards.
([Bressan et al., 2025](https://doi.org/10.11576/glow-1242))

# Citation


```bibtex
@inproceedings{bressan2026blimpit,
  title={Language models assessment through linguistically motivated contrasts: A benchmark for Italian (BLiMP-IT)},
  author={Bressan, Veronica and Barbini, Matilde and Fusco, Achille and Neri, Sofia and Piccini Bianchessi, Maria Letizia and Rossi, Sarah and Chesi, Cristiano},
  booktitle={Proceedings of GLOW 47},
  year={2026}
}
```

## Dataset Summary

BLiMP-IT is a linguistically motivated benchmark for evaluating Italian language models through **minimal pairs**. 
Each example consists of a grammatical sentence paired with a minimally different ungrammatical counterpart that isolates a single morphosyntactic contrast. 
The benchmark is designed to evaluate whether language models assign higher probability to the grammatical sentence than to the ungrammatical one.

The benchmark is inspired by the [English BLiMP](https://aclanthology.org/2020.tacl-1.25/) while extending and adapting it to Italian morphosyntax. 
It combines phenomena from the original BLiMP with additional contrasts derived from Italian linguistic literature and the [COnVERSA psycholinguistic assessment battery](https://www.hogrefe.com/it/shop/comprensione-delle-opposizioni-morfosintattiche-verbali-attraverso-la-scrittura.html).

Rather than measuring downstream task performance, BLiMP-IT measures the acquisition of abstract morphosyntactic generalizations.

## Supported Tasks

The dataset is intended for:

- Language model evaluation through sentence probability comparison
- Minimal-pair acceptability evaluation
- Psycholinguistically motivated syntactic evaluation
- Cross-model comparison
- Cross-linguistic comparison with BLiMP and related benchmarks

Typical evaluation consists of assigning higher probability (or lower perplexity) to the grammatical member of each minimal pair.

---

# Dataset Structure

The repository contains one YAML configuration per linguistic phenomenon.

## Macro-phenomena

The benchmark covers four major linguistic domains.

| Macro-phenomenon | Number of phenomena |
|------------------|--------------------:|
| Agreement & Inflection | 31 |
| Verbal Class & Argument Structure | 10 |
| Pronouns | 25 |
| Non-local Dependencies | 64 |

Overall the benchmark contains **130 phenomena**.

Some phenomena originate from the original BLiMP benchmark, while others are adapted from the COnVERSA battery or specifically developed for Italian.

---

# Phenomena

## Agreement & Inflection

### Original BLiMP

- DP agreement
  - `blimp_it_original_agreement_and_inflection_dp`

### COnVERSA

- Determiner–noun agreement
  - `conversa_a_agreement_dp`
  - `conversa_a_agreement_in_dp`

- Subject–verb agreement
  - `conversa_a_agreement_subject_verb_trans`
  - `conversa_a_agreement_subject_verb_unacc`
  - `conversa_a_agreement_subject_verb_unerg`
  - `conversa_a_agreement_subject_verb_unerg_trans`
  - `conversa_a_agreement_subject_verb_attraction`
  - `conversa_a_agreement_subject_verb_cumulative`
  - `conversa_a_agreement_verb_subject_unacc`

- Predicate agreement
  - `conversa_a_agreement_subject_nominal_predicate`
  - `conversa_a_agreement_subject_nominal_predicate_attraction`

- Past participle agreement
  - `conversa_a_agreement_past_particple_pre_v_cl`
  - `conversa_a_agreement_past_particple_unacc`
  - `conversa_a_past_particple_pre_v_cl`
  - `conversa_a_past_particple_unacc`

- Psych verbs
  - `conversa_a_agreement_psych_verbs_piacere`
  - `conversa_a_agreement_psych_verbs_preoccupare`
  - `conversa_a_psych_verbs_piacere`
  - `conversa_a_psych_verbs_preoccupare`

---

## Verbal Class & Argument Structure

- Theta-role assignment
  - `conversa_b_theta_roles`
  - `conversa_b_argument_structure_theta_roles`
  - `conversa_b_argument_structure_b_theta_roles`

- Auxiliary selection
  - `conversa_b_auxiliary_selection_transitives`
  - `conversa_b_auxiliary_selection_unaccusatives`
  - `conversa_b_auxiliary_selection_unergatives`
  - `conversa_b_auxiliary_selection_ditranstitives`
  - `conversa_b_auxiliary_selection_passive_active_diathesis`

- Argument structure variants
  - `conversa_b_argument_structure_b_auxiliary_selection_transitives`
  - `conversa_b_argument_structure_b_auxiliary_selection_unaccusatives`
  - `conversa_b_argument_structure_b_auxiliary_selection_unergatives`
  - `conversa_b_argument_structure_b_auxiliary_selection_ditranstitives`
  - `conversa_b_argument_structure_b_auxiliary_selection_passive_active_diathesis`

---

## Pronouns

### Personal pronouns

- `conversa_c_1_2_person_in_answers`
- `conversa_c_1_2_person_in_elicited_declaratives`
- `conversa_c_pronouns_1_2_person_answers`
- `conversa_c_pronouns_1_2_person_elicited_declaratives`

### Clitics

- `conversa_c_clitics_answering_with_acc_cl`
- `conversa_c_clitics_coordinating_with_acc_cl`
- `conversa_c_clitics_coordinating_with_dat_cl`

- `conversa_c_pronouns_clitics_answering_with_acc_cl`
- `conversa_c_pronouns_clitics_coordinating_with_acc_cl`
- `conversa_c_pronouns_clitics_coordinating_with_dat_cl`

### Reflexives

- `conversa_c_reflexives_other`
- `conversa_c_reflexives_psych_verbs`
- `conversa_c_reflexives_unacc`

- `conversa_c_pronouns_reflexives_other`
- `conversa_c_pronouns_reflexives_psych_verbs`
- `conversa_c_pronouns_reflexives_unacc`

---

## Question Formation and Comprehension

- Yes/no questions
- Why questions
- WH-questions
- Relative clause questions
- Subject position

Configurations include

- `conversa_d_answering_yes_no_questions`
- `conversa_d_answering_why_questions`
- `conversa_d_answering_wh_adjunct_questions`
- `conversa_d_answering_wh_argument_questions`
- `conversa_d_answering_wh_argument_questions_number_disambiguation`
- `conversa_d_answering_questions_with_rc_subj`
- `conversa_d_answering_questions_with_rc_obj`
- `conversa_d_question_formation_subject_position`

and their corresponding `conversa_d_questions_*` variants.

---

## Non-local Dependencies

The benchmark contains extensive coverage of filler–gap dependencies and syntactic islands.

### Relative clauses

- subject relatives
- object relatives
- gap vs no-gap conditions

### WH extraction

- root extraction
- embedded extraction
- two-level embedded extraction
- clitic fillers
- NP fillers
- demonstrative fillers

### Across-the-board movement

- affirmative
- interrogative
- relative clause
- gap/no-gap manipulations

### Islands

- adjunct island
- complex NP island
- sentential subject island
- WH island
- left branch island
- coordinate structure constraint

### Parasitic gaps

- adjunct island variants

Both the original BLiMP-derived (`blimp_it_original_non_local_dependencies_*`) and the manually refined (`non_local_dependencies_*`) versions are included.

---

# Data Fields

Each dataset configuration consists of sentence pairs containing:

- `sentence_good`
- `sentence_bad`
- grammaticality label
- linguistic phenomenon identifier
- minimal-pair metadata

(Field names may vary slightly depending on the implementation.)

---

# Data Creation

## Source

BLiMP-IT combines:

- manually designed minimal pairs
- automatic lexical generation using grammar templates
- manually annotated Italian lexicon
- human validation

The lexicon was derived from a 2.4M-token corpus of Italian child-directed and naturalistic language, annotated with Universal POS tags, morphological features, and animacy information. Automatic generation produces lexical variants while preserving the targeted morphosyntactic contrast.

---

# Dataset Characteristics

The benchmark follows two design principles:

1. **Minimality of contrasts**

   Each grammatical and ungrammatical sentence differs only in the manipulation responsible for (un)grammaticality.

2. **Linguistically meaningful errors**

   Ungrammatical sentences contain theoretically motivated morphosyntactic violations rather than random perturbations or word permutations.

---

# Intended Uses

BLiMP-IT is intended for

- evaluating Italian language models
- probing syntactic generalization
- comparing language models
- comparing humans and language models
- psycholinguistic research
- computational linguistics research

---

# Limitations

- The benchmark focuses exclusively on morphosyntax rather than semantics.
- Some generated sentences may be semantically unusual while remaining syntactically well formed.
- Human acceptability judgments currently exist only for subsets overlapping with COnVERSA.
- Automatic lexical generation is still expanding coverage across all phenomena.


---

# License

CC BY-SA 4.0

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
  
### Changelog