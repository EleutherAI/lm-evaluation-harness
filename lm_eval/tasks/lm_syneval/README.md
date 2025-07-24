# Targeted Syntactic Evaluation of Language Models (LM-SynEval)

## Paper

**Title:** Targeted Syntactic Evaluation of Language Models

**Authors:**: Rebecca Marvin and Tal Linzen

**Link:** https://doi.org/10.18653/v1/D18-1151

**Abstract:**
> We present a data set for evaluating the grammaticality of the predictions of a language model. We automatically construct a large number of minimally different pairs of English sentences, each consisting of a grammatical and an ungrammatical sentence. The sentence pairs represent different variations of structure-sensitive phenomena: subject-verb agreement, reflexive anaphora and negative polarity items. We expect a language model to assign a higher probability to the grammatical sentence than the ungrammatical one. In an experiment using this data set, an LSTM language model performed poorly on many of the constructions. Multi-task training with a syntactic objective (CCG supertagging) improved the LSTM's accuracy, but a large gap remained between its performance and the accuracy of human participants recruited online. This suggests that there is considerable room for improvement over LSTMs in capturing syntax in a language model.

**Homepage:** https://github.com/BeckyMarvin/LM_syneval

**Language(s):** English

**License:** MIT License

### Citation

```
@inproceedings{marvin-linzen-2018-targeted,
    title = "Targeted Syntactic Evaluation of Language Models",
    author = "Marvin, Rebecca  and
      Linzen, Tal",
    editor = "Riloff, Ellen  and
      Chiang, David  and
      Hockenmaier, Julia  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1151/",
    doi = "10.18653/v1/D18-1151",
    pages = "1192--1202"
}
```

## Groups, Tags, and Tasks

The tasks are structured hierarchically as listed below. For more detailed explanations, see original paper and repository (linked above). In this implementation, group means are unweighted.

* `lm_syneval`
    * `lm_syneval__reflexives`
        * `lm_syneval__reflexives__simple_reflexives`
            * `lm_syneval__reflexives__simple_reflexives__sing_MS_ANPHR`
            * `lm_syneval__reflexives__simple_reflexives__plur_MS_ANPHR`
        * `lm_syneval__reflexives__reflexive_sent_comp`
            * `lm_syneval__reflexives__reflexive_sent_comp__sing_MS_ANPHR_sing_BS`
            * `lm_syneval__reflexives__reflexive_sent_comp__sing_MS_ANPHR_plur_BS`
            * `lm_syneval__reflexives__reflexive_sent_comp__plur_MS_ANPHR_sing_BS`
            * `lm_syneval__reflexives__reflexive_sent_comp__plur_MS_ANPHR_plur_BS`
        * `lm_syneval__reflexives__reflexives_across`
            * `lm_syneval__reflexives__reflexives_across__sing_MS_ANPHR_sing_ES_EV`
            * `lm_syneval__reflexives__reflexives_across__sing_MS_ANPHR_plur_ES_EV`
            * `lm_syneval__reflexives__reflexives_across__plur_MS_ANPHR_sing_ES_EV`
            * `lm_syneval__reflexives__reflexives_across__plur_MS_ANPHR_plur_ES_EV`
    * `lm_syneval__agreement`
        * `lm_syneval__agreement__obj_rel_within_inanim`
            * `lm_syneval__agreement__obj_rel_within_inanim__sing_ES_EV_sing_IS_IV`
            * `lm_syneval__agreement__obj_rel_within_inanim__sing_ES_EV_plur_IS_IV`
            * `lm_syneval__agreement__obj_rel_within_inanim__plur_ES_EV_sing_IS_IV`
            * `lm_syneval__agreement__obj_rel_within_inanim__plur_ES_EV_plur_IS_IV`
        * `lm_syneval__agreement__vp_coord`
            * `lm_syneval__agreement__vp_coord__sing_MS_MV_MV`
            * `lm_syneval__agreement__vp_coord__plur_MS_MV_MV`
        * `lm_syneval__agreement__sent_comp`
            * `lm_syneval__agreement__sent_comp__sing_MS_MV_sing_BS`
            * `lm_syneval__agreement__sent_comp__sing_MS_MV_plur_BS`
            * `lm_syneval__agreement__sent_comp__plur_MS_MV_sing_BS`
            * `lm_syneval__agreement__sent_comp__plur_MS_MV_plur_BS`
        * `lm_syneval__agreement__obj_rel_no_comp_within_inanim`
            * `lm_syneval__agreement__obj_rel_no_comp_within_inanim__sing_ES_EV_sing_IS_IV`
            * `lm_syneval__agreement__obj_rel_no_comp_within_inanim__sing_ES_EV_plur_IS_IV`
            * `lm_syneval__agreement__obj_rel_no_comp_within_inanim__plur_ES_EV_sing_IS_IV`
            * `lm_syneval__agreement__obj_rel_no_comp_within_inanim__plur_ES_EV_plur_IS_IV`
        * `lm_syneval__agreement__obj_rel_within_anim`
            * `lm_syneval__agreement__obj_rel_within_anim__sing_ES_EV_sing_MS_MV`
            * `lm_syneval__agreement__obj_rel_within_anim__sing_ES_EV_plur_MS_MV`
            * `lm_syneval__agreement__obj_rel_within_anim__plur_ES_EV_sing_MS_MV`
            * `lm_syneval__agreement__obj_rel_within_anim__plur_ES_EV_plur_MS_MV`
        * `lm_syneval__agreement__subj_rel`
            * `lm_syneval__agreement__subj_rel__sing_MS_EV_MV_sing_ES`
            * `lm_syneval__agreement__subj_rel__sing_MS_EV_MV_plur_ES`
            * `lm_syneval__agreement__subj_rel__plur_MS_EV_MV_sing_ES`
            * `lm_syneval__agreement__subj_rel__plur_MS_EV_MV_plur_ES`
        * `lm_syneval__agreement__prep_inanim`
            * `lm_syneval__agreement__prep_inanim__sing_IS_IV_sing_ES`
            * `lm_syneval__agreement__prep_inanim__sing_IS_IV_plur_ES`
            * `lm_syneval__agreement__prep_inanim__plur_IS_IV_sing_ES`
            * `lm_syneval__agreement__prep_inanim__plur_IS_IV_plur_ES`
        * `lm_syneval__agreement__long_vp_coord`
            * `lm_syneval__agreement__long_vp_coord__sing_MS_LMV_LMV`
            * `lm_syneval__agreement__long_vp_coord__plur_MS_LMV_LMV`
        * `lm_syneval__agreement__obj_rel_across_anim`
            * `lm_syneval__agreement__obj_rel_across_anim__sing_MS_MV_sing_ES_EV`
            * `lm_syneval__agreement__obj_rel_across_anim__sing_MS_MV_plur_ES_EV`
            * `lm_syneval__agreement__obj_rel_across_anim__plur_MS_MV_sing_ES_EV`
            * `lm_syneval__agreement__obj_rel_across_anim__plur_MS_MV_plur_ES_EV`
        * `lm_syneval__agreement__obj_rel_across_inanim`
            * `lm_syneval__agreement__obj_rel_across_inanim__sing_IS_IV_sing_ES_EV`
            * `lm_syneval__agreement__obj_rel_across_inanim__sing_IS_IV_plur_ES_EV`
            * `lm_syneval__agreement__obj_rel_across_inanim__plur_IS_IV_sing_ES_EV`
            * `lm_syneval__agreement__obj_rel_across_inanim__plur_IS_IV_plur_ES_EV`
        * `lm_syneval__agreement__obj_rel_no_comp_across_anim`
            * `lm_syneval__agreement__obj_rel_no_comp_across_anim__sing_MS_MV_sing_ES_EV`
            * `lm_syneval__agreement__obj_rel_no_comp_across_anim__sing_MS_MV_plur_ES_EV`
            * `lm_syneval__agreement__obj_rel_no_comp_across_anim__plur_MS_MV_sing_ES_EV`
            * `lm_syneval__agreement__obj_rel_no_comp_across_anim__plur_MS_MV_plur_ES_EV`
        * `lm_syneval__agreement__obj_rel_no_comp_across_inanim`
            * `lm_syneval__agreement__obj_rel_no_comp_across_inanim__sing_IS_IV_sing_ES_EV`
            * `lm_syneval__agreement__obj_rel_no_comp_across_inanim__sing_IS_IV_plur_ES_EV`
            * `lm_syneval__agreement__obj_rel_no_comp_across_inanim__plur_IS_IV_sing_ES_EV`
            * `lm_syneval__agreement__obj_rel_no_comp_across_inanim__plur_IS_IV_plur_ES_EV`
        * `lm_syneval__agreement__simple_agrmt`
            * `lm_syneval__agreement__simple_agrmt__sing_MS_MV`
            * `lm_syneval__agreement__simple_agrmt__plur_MS_MV`
        * `lm_syneval__agreement__prep_anim`
            * `lm_syneval__agreement__prep_anim__sing_MS_MV_sing_ES`
            * `lm_syneval__agreement__prep_anim__sing_MS_MV_plur_ES`
            * `lm_syneval__agreement__prep_anim__plur_MS_MV_sing_ES`
            * `lm_syneval__agreement__prep_anim__plur_MS_MV_plur_ES`
        * `lm_syneval__agreement__obj_rel_no_comp_within_anim`
            * `lm_syneval__agreement__obj_rel_no_comp_within_anim__sing_ES_EV_sing_MS_MV`
            * `lm_syneval__agreement__obj_rel_no_comp_within_anim__sing_ES_EV_plur_MS_MV`
            * `lm_syneval__agreement__obj_rel_no_comp_within_anim__plur_ES_EV_sing_MS_MV`
            * `lm_syneval__agreement__obj_rel_no_comp_within_anim__plur_ES_EV_plur_MS_MV`
    * `lm_syneval__npi`
        * `lm_syneval__npi__npi_across_anim`
            * `lm_syneval__npi__npi_across_anim__past`
            * `lm_syneval__npi__npi_across_anim__future`
        * `lm_syneval__npi__npi_across_inanim`
            * `lm_syneval__npi__npi_across_inanim__past`
            * `lm_syneval__npi__npi_across_inanim__future`
        * `lm_syneval__npi__simple_npi_anim`
            * `lm_syneval__npi__simple_npi_anim__past`
            * `lm_syneval__npi__simple_npi_anim__future`
        * `lm_syneval__npi__simple_npi_inanim`
            * `lm_syneval__npi__simple_npi_inanim__past`
            * `lm_syneval__npi__simple_npi_inanim__future`

## Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
      * The original paper evaluates traditional RNN models, which require a very different pipeline to analyze.

## Changelog
