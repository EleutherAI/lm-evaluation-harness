# BLiMP-NL: A Corpus of Dutch Minimal Pairs and Acceptability Judgments for Language Model Evaluation

## Paper

Title: BLiMP-NL: A Corpus of Dutch Minimal Pairs and Acceptability Judgments for Language Model Evaluation

Abstract:

> [A] corpus of 8400 Dutch sentence pairs, intended primarily for the grammatical evaluation of language models. Each pair consists of a grammatical sentence and a minimally different ungrammatical sentence. The corpus covers 84 paradigms, classified into 22 syntactic phenomena. Ten sentence pairs of each paradigm were created by hand, while the remaining 90 were generated semi-automatically and manually validated afterwards.
([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559))


Homepage: https://data.ru.nl/collections/ru/cls/blimp-nl_dsc_550

### Citation

```
@article{10.1162/coli_a_00559,
    author = {Suijkerbuijk, Michelle and Prins, Zo{\"e} and de Heer Kloots, Marianne and Zuidema, Willem and Frank, Stefan L.},
    title = {BLiMP-NL: A Corpus of Dutch Minimal Pairs and Acceptability Judgments for Language Model Evaluation},
    journal = {Computational Linguistics},
    pages = {1-35},
    year = {2025},
    month = {05},
    issn = {0891-2017},
    doi = {10.1162/coli_a_00559},
    url = {https://doi.org/10.1162/coli\_a\_00559},
}
```

### Groups, Tags, and Tasks

#### Groups

* `blimp_nl`: Runs all tasks of the large BLiMP-NL benchmark

**Phenomena** (runs all paradigms within each phenomenon and calculates the mean across all of them):

* `blimp_nl__adpositional_phrases`: "This covers the characteristics of different types of adpositional phrases, such as the PP-complement of a noun phrase or containing an R-word." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__adverbial_modification`: "This covers the position of adverbs in the sentence." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__anaphor_agreement`: "This covers the requirement that reflexive pronouns such as _mezelf_ ('myself') agree with their antecedents in person and number." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__argument_structure`: This covers the different verb types and their characteristics, such as the number of arguments (in-/di-)transitive verbs take and the specific auxiliary (a)telic unaccusative and NOM-DAT verbs select." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__auxiliaries`: "This covers the different types of auxiliary verbs and their behavior." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__binding_principle_a`: " This covers the structural relationship between the reflexive pronoun and its antecedent." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__complementive`: "This covers the possibility of having secondary predication on (in-/di)transitive verbs and the position of that predication." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__crossing_dependencies`: "This covers the specific feature that verbs and arguments are ordered cross-serially." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__determiners`: "This covers the special determiner _geen_ ('no') and its characteristics." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__extraposition`: " This covers the possibility of extraposing nouns and adverbs" ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__finite_argument_clause`: "This covers the argument clause that is finite, and specifically the obligatory complementizer, the position of the clause, and the verbs that select this clause." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__infinitival_argument_clause`: " This covers the argument clause that is infinitival, and specifically the verbs that select this clause and the differences between the infinitival markers _te_ and _om te_." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__nominalization`: "This covers the ways in which words from different categories can be turned into nouns." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__parasitic_gaps`: "This covers the characteristics of parasitic gap formation." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__passive`: "This covers the formation of the impersonal and regular passive construction." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__quantifiers`: " This covers the behavior of quantifiers, specifically their agreement with nouns and verbs." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__r_words`: "This covers the formation and extraction of R-words (e.g., _daar_ and _er_)." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__relativization`: "This covers the characteristics of relativization and the restrictions thereon." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__topicalization`: "This covers the characteristics of topicalization and the restrictions thereon." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__verb_second`: "This covers the different word order restrictions in main and embedded clauses." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__wh_movement`: "This covers the requirements for wh-movement and the related phenomenon stranding." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).
* `blimp_nl__wh_movement_restrictions`: "This covers the restrictions that exist on wh-movement, such as island and superiority constraints." ([Suijkerbuijk et al., 2025](https://doi.org/10.1162/coli_a_00559)).

Each of these is further divided into specific experimental paradigms (which here are represented as individual tasks; 100 items each), which are described in the [Suijkerbuijk et al., (2025)](https://doi.org/10.1162/coli_a_00559).

**Implementation note**: The original implementation as discussed in the paper uses masked language models and compares syntactic log-odds ratios (SLOG; [Pauls & Klein, 2012](https://aclanthology.org/P12-1101/)) between sentences, which normalizes for word frequency. Neither masked langauge models nor SLOG are currently supported by the Harness, and so the implementation provided here includes both un-normalized accuracy (`acc`) and byte-length-normalized accuracy (`acc_norm`).

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


### Changelog
