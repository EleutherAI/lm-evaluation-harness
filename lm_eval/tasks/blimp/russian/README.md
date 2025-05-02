# RuBLiMP

### Paper

* Title: `RuBLiMP: Russian Benchmark of Linguistic Minimal Pairs`
* Abstract: [aclanthology.org/2024.emnlp-main.522](https://aclanthology.org/2024.emnlp-main.522/)
* Homepage: [github.com/RussianNLP/RuBLiMP](https://github.com/RussianNLP/RuBLiMP/tree/main)

The Russian Benchmark of Linguistic Minimal Pairs (RuBLiMP) includes 45,000 pairs of sentences that differ in grammaticality and isolate a morphological, syntactic, or semantic phenomenon. In contrast to existing benchmarks of linguistic minimal pairs, RuBLiMP is created by applying linguistic perturbations to automatically annotated sentences from open text corpora and decontaminating test data across 25 open-source language models that support Russian.


### Tasks

`rublimp` can be used to run the zero-shot evaluation on all 45 benchmark datasets. The dataset-specific task names are provided below.

Refer to more details about the phenomena in [the GitHub repository](https://github.com/RussianNLP/RuBLiMP/tree/main/src/phenomena).

### Morphology

<details>
    <summary><b>Word Formation</b></summary>

- **Addition of Extra Morphemes: Uninterpretable Suffix Combinations** (`rublimp_add_new_suffix`) \
     Adding a new suffix to the noun or adjective to create a non-existing word

- **Addition of Extra Morphemes: Verb Prefixes** (`rublimp_add_verb_prefix`) \
    Adding a prefix to a verb to create a violation of prefix stacking rules.

- **Morpheme Permutation: Verb Prefixes** (`rublimp_change_verb_prefixes_order`) \
    Changing the order of the verb's prefixes to create a violation of prefix stacking rules.

</details>

<details>
    <summary><b>Word Inflection</b></summary>

- **Replacement of Inflectional Affixes: Noun Declensions (Simple)** (`rublimp_change_declension_ending`) \
    Changing the inflectional suffixes of a noun to the suffixes of another declension

- **Replacement of Inflectional Affixes: Declensions of Nouns With Agreeing Dependents** (`rublimp_change_declension_ending_has_dep`) \
    Changing the inflectional suffixes of a noun to the suffixes of another declension in the presence of an agreeing noun modifier

- **Inflectional Affixes: Verbal Conjugation Swap** (`rublimp_change_verb_conjugation`) \
    Replacing the verb’s inflection with inflection of the opposite conjugation


</details>


### Syntax
<details>
    <summary><b>Government</b></summary>

- **Prepositional Government** (`rublimp_adp_government_case`) \
    Changing the case of a noun, governed by a preposition

- **Verbal Government: Direct Object** (`rublimp_verb_acc_object`) \
    Changing the case of a direct verb object

- **Verbal Government: Genitive Object** (`rublimp_verb_gen_object`) \
    Changing the case of an indirect verb object in Genitive case

- **Verbal Government: Object in Instrumental Case** (`rublimp_verb_ins_object`) \
    Changing the case of an indirect verb object in Instrumental case

- **Verbal Government: Nominalizations** (`rublimp_nominalization_cas`) \
    Changing the case of a dependent of a nominalization

</details>


<details>
    <summary><b>Subject-Predicate Agreement</b></summary>

- **Subject-Predicate Agreement (Number)** (`rublimp_noun_subj_predicate_agreement_number`) \
    Changing the number of the predicate to be distinct from its subject's (or, sometimes, changing number of the subject to be distinct from its predicate's)

- **Genitive Subject-Predicate Agreement (Number)** (`rublimp_genitive_subj_predicate_agreement_number`) \
    Changing the number of the predicate to plural, when subject is genitive and the agreement must be the default singular neuter

- **Clausal Subject-Predicate Agreement (Number)** (`rublimp_clause_subj_predicate_agreement_number`) \
    Changing the number of the predicate to plural, when subject is a clause and the agreement must be the default singular neuter

- **Subject-Predicate Agreement in Presence of an Attractor (Number)** (`rublimp_subj_predicate_agreement_number_attractor`) \
    Changing the number of the verb to that, which is different from the subject, but the same as subject's dependent, or the attractor  

- **Subject-Predicate Agreement (Gender)** (`rublimp_noun_subj_predicate_agreement_gender`) \
    Changing the gender of the predicate to be distinct from its subject's (or, sometimes, changing number of the subject to be distinct from its predicate's)

- **Genitive Subject-Predicate Agreement (Gender)** (`rublimp_genitive_subj_predicate_agreement_gender`)
    Changing the gender of the predicate to feminine or masculine, when subject is genitive and the agreement must be the default singular neuter

- **Clausal Subject-Predicate Agreement (Gender)** (`rublimp_clause_subj_predicate_agreement_gender`) \
    Changing the gender of the predicate to feminine or masculine, when subject is a clause and the agreement must be the default singular neuter

- **Subject-Predicate Agreement in Presence of an Attractor (Gender)** (`rublimp_subj_predicate_agreement_gender_attractor`) \
    Changing the gender of the verb to that, which is different from the subject, but the same as subject's dependent, or the attractor

- **Subject-Predicate Agreement (Person)** (`rublimp_noun_subj_predicate_agreement_person`) \
    Changing the person of the predicate to be distinct from its subject's

- **Genitive Subject-Predicate Agreement (Person)** (`rublimp_genitive_subj_predicate_agreement_person`) \
    Changing the person of the predicate to first or second person, when subject is genitive and the agreement must be the default third person singular

- **Clausal Subject-Predicate Agreement (Person)** (`rublimp_clause_subj_predicate_agreement_person`) \
    Changing the person of the predicate to first or second person, when subject is a clause and the agreement must be the default third person singular

</details>


<details>
    <summary><b>Anaphor Agreement</b></summary>

- **Anaphor Agreement (Number)** (`rublimp_anaphor_agreement_number`) \
    Changing the number of the relative pronoun or of its head noun

- **Anaphor Agreement (Gender)** (`rublimp_anaphor_agreement_gender`) \
    Changing the gender of the relative pronoun  

</details>

<details>
    <summary><b>Noun Phrase Agreement</b></summary>

- **Noun Phrase Agreement (Number)** (`rublimp_np_agreement_number`) \
    Changing the number of an agreeing adjective

- **Noun Phrase Agreement (Gender)** (`rublimp_np_agreement_gender`) \
    Changing the gender of an agreeing adjective

- **Noun Phrase Agreement (Case)** (`rublimp_np_agreement_case`) \
    Changing the case of an agreeing adjective

</details>

<details>
    <summary><b>Floating Quantifier Agreement</b></summary>

- **Floating Quantifier Agreement (Number)** (`rublimp_floating_quantifier_agreement_number`) \
    Changing the number of the quantifier or of the controller

- **Floating Quantifier Agreement (Gender)** (`rublimp_floating_quantifier_agreement_gender`) \
    Changing the gender of the quantifier or of the controller

- **Floating Quantifier Agreement (Case)** (`rublimp_floating_quantifier_agreement_case`) \
    Changing the case of the quantifier or of the controller

</details>



<details>
    <summary><b>Reflexives</b></summary>

- **External Possessor** (`rublimp_external_possessor`) \
    Change a noun phrase or a pronoun to a reflexive pronoun sebya ‘self’ in a *u*-phrase inside the existential *be*-possessive construction.

</details>

<details>
    <summary><b>Negation</b></summary>

- **Negative Concord** (`rublimp_negative_concord`) \
    Shifting the negative particle *ne* from a negated verb to another word in the sentence to violate negative concord rules.

- **Replacement of a Negative Pronoun with an Indefinite One** (`rublimp_negative_pronoun_to_indefinite`) \
    Replacing an negative pronoun in the construction without a negated verb to an indefinite pronoun

- **Replacement of an Indefinite Pronoun with a Negative One** (`rublimp_indefinite_pronoun_to_negative`) \
    Replacing an indefinite pronoun in the construction with a negated verb to a negative pronoun

</details>


### Semantics

<details>
    <summary><b>Argument Structure</b></summary>

- **Transitivity** (`rublimp_transitive_verb`) \
    Replacing a transitive verb with an intransitive one

- **Animate Subject of a Transitive Verb** (`rublimp_transitive_verb_subject`) \
    Swapping the subject and the direct object of a transitive verb or replacing the subject with a random inanimate word

- **Animate Subject of a Passive Verb** (`rublimp_transitive_verb_passive`) \
    Swapping the subject and the direct object of a transitive verb in a passive construction or replacing the subject with a random inanimate word

- **Animate Direct Object of a Transitive Verb** (`rublimp_transitive_verb_object`) \
    Replacing the direct object with a random inanimate word

- **Animate Indirect Object of a Transitive Verb** (`rublimp_transitive_verb_iobject`) \
    Swapping the subject and the indirect object of a transitive verb or replacing the indirect subject of a transitive verb with a random inanimate word


</details>


<details>
    <summary><b>Aspect</b></summary>

- **Incompatibility of the Perfective with the Semantics of Duration** (`rublimp_change_duration_aspect`) \
Replacing an imperfective verb with a perfective one in contexts with semantics of duration

- **Impossibility of the Perfective in Repetitive Situations** (`rublimp_change_repetition_aspect`) \
Replacing an imperfective verb with a perfective one in contexts with semantics of repetition

- **Impossibility of the Perfective Under Negated Strong Deontic Verbs** (`rublimp_deontic_imperative_aspect`) \
Replacing an imperfective verb with a perfective one in contexts with a negated deontic verb

</details>

<details>
    <summary><b>Tense</b></summary>

- **Tense** (`rublimp_single_verb_tense`) \
    Changing verb tense in the presence of a temporal adverbial

- **Tense (coordination)** (`rublimp_conj_verb_tense`) \
    Changing the tense of a conjoined verb in the presence of a temporal adverbial

- **Tense Markers** (`rublimp_tense_marker`) \
    Changing a temporal adverbial in a sentence with a tense-marked verb

</details>

### Citation

```
@inproceedings{taktasheva-etal-2024-rublimp,
    title = "{R}u{BL}i{MP}: {R}ussian Benchmark of Linguistic Minimal Pairs",
    author = "Taktasheva, Ekaterina  and
      Bazhukov, Maxim  and
      Koncha, Kirill  and
      Fenogenova, Alena  and
      Artemova, Ekaterina  and
      Mikhailov, Vladislav",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.522/",
    doi = "10.18653/v1/2024.emnlp-main.522",
    pages = "9268--9299",
    abstract = "Minimal pairs are a well-established approach to evaluating the grammatical knowledge of language models. However, existing resources for minimal pairs address a limited number of languages and lack diversity of language-specific grammatical phenomena. This paper introduces the Russian Benchmark of Linguistic Minimal Pairs (RuBLiMP), which includes 45k pairs of sentences that differ in grammaticality and isolate a morphological, syntactic, or semantic phenomenon. In contrast to existing benchmarks of linguistic minimal pairs, RuBLiMP is created by applying linguistic perturbations to automatically annotated sentences from open text corpora and decontaminating test data. We describe the data collection protocol and present the results of evaluating 25 language models in various scenarios. We find that the widely used LMs for Russian are sensitive to morphological and agreement-oriented contrasts, but fall behind humans on phenomena requiring the understanding of structural relations, negation, transitivity, and tense. RuBLiMP, the codebase, and other materials are publicly available."
}
```

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
