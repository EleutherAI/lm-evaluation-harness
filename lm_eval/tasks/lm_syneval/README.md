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

* `lm_syneval`: Targeted Syntactic Evaluation of Language Models
    * `lm_syneval__agreement`: Agreement
        * `lm_syneval__agreement__simple_agrmt`: Simple agreement
            * `lm_syneval__agreement__simple_agrmt__sing_MS_MV`:
                * Example: 'The author laughs.' (correct) vs. 'The author laugh.' (incorrect)
            * `lm_syneval__agreement__simple_agrmt__plur_MS_MV`:
                * Example: 'The authors laugh.' (correct) vs. 'The authors laughs.' (incorrect)
        * `lm_syneval__agreement__prep_anim`: Agreement across a prepositional phrase with animate subject
            * `lm_syneval__agreement__prep_anim__sing_MS_MV_sing_ES`:
                * Example: 'The author next to the guard laughs.' (correct) vs. 'The author next to the guard laugh.' (incorrect)
            * `lm_syneval__agreement__prep_anim__sing_MS_MV_plur_ES`:
                * Example: 'The author next to the guards laughs.' (correct) vs. 'The author next to the guards laugh.' (incorrect)
            * `lm_syneval__agreement__prep_anim__plur_MS_MV_sing_ES`:
                * Example: 'The authors next to the guard laugh.' (correct) vs. 'The authors next to the guard laughs.' (incorrect)
            * `lm_syneval__agreement__prep_anim__plur_MS_MV_plur_ES`:
                * Example: 'The authors next to the guards laugh.' (correct) vs. 'The authors next to the guards laughs.' (incorrect)
        * `lm_syneval__agreement__prep_inanim`: Agreement across a prepositional phrase with inanimate subject
            * `lm_syneval__agreement__prep_inanim__sing_IS_IV_sing_ES`:
                * Example: 'The movie from the guard is good.' (correct) vs. 'The movie from the guard are good.' (incorrect)
            * `lm_syneval__agreement__prep_inanim__sing_IS_IV_plur_ES`:
                * Example: 'The movie from the guards is good.' (correct) vs. 'The movie from the guards are good.' (incorrect)
            * `lm_syneval__agreement__prep_inanim__plur_IS_IV_sing_ES`:
                * Example: 'The movies from the guard are good.' (correct) vs. 'The movies from the guard is good.' (incorrect)
            * `lm_syneval__agreement__prep_inanim__plur_IS_IV_plur_ES`:
                * Example: 'The movies from the guards are good.' (correct) vs. 'The movies from the guards is good.' (incorrect)
        * `lm_syneval__agreement__sent_comp`: Agreement in a sentential complement
            * `lm_syneval__agreement__sent_comp__sing_MS_MV_sing_BS`:
                * Example: 'The mechanic said the author laughs.' (correct) vs. 'The mechanic said the author laugh.' (incorrect)
            * `lm_syneval__agreement__sent_comp__sing_MS_MV_plur_BS`:
                * Example: 'The mechanics said the author laughs.' (correct) vs. 'The mechanics said the author laugh.' (incorrect)
            * `lm_syneval__agreement__sent_comp__plur_MS_MV_sing_BS`:
                * Example: 'The mechanic said the authors laugh.' (correct) vs. 'The mechanic said the authors laughs.' (incorrect)
            * `lm_syneval__agreement__sent_comp__plur_MS_MV_plur_BS`:
                * Example: 'The mechanics said the authors laugh.' (correct) vs. 'The mechanics said the authors laughs.' (incorrect)
        * `lm_syneval__agreement__subj_rel`: Agreement across a subject relative clause
            * `lm_syneval__agreement__subj_rel__sing_MS_EV_MV_sing_ES`:
                * Example: 'The author that likes the guard laughs.' (correct) vs. 'The author that likes the guard laugh.' (incorrect)
            * `lm_syneval__agreement__subj_rel__sing_MS_EV_MV_plur_ES`:
                * Example: 'The author that likes the guards laughs.' (correct) vs. 'The author that likes the guards laugh.' (incorrect)
            * `lm_syneval__agreement__subj_rel__plur_MS_EV_MV_sing_ES`:
                * Example: 'The authors that like the guard laugh.' (correct) vs. 'The authors that like the guard laughs.' (incorrect)
            * `lm_syneval__agreement__subj_rel__plur_MS_EV_MV_plur_ES`:
                * Example: 'The authors that like the guards laugh.' (correct) vs. 'The authors that like the guards laughs.' (incorrect)
        * `lm_syneval__agreement__vp_coord`: Short verb phrase coordination
            * `lm_syneval__agreement__vp_coord__sing_MS_MV_MV`:
                * Example: 'The author laughs and swims.' (correct) vs. 'The author laughs and swim.' (incorrect)
            * `lm_syneval__agreement__vp_coord__plur_MS_MV_MV`:
                * Example: 'The authors laugh and swim.' (correct) vs. 'The authors laugh and swims.' (incorrect)
        * `lm_syneval__agreement__long_vp_coord`: Long verb phrase coordination
            * `lm_syneval__agreement__long_vp_coord__sing_MS_LMV_LMV`:
                * Example: 'The author knows many different foreign languages and likes to watch television shows.' (correct) vs. 'The author knows many different foreign languages and like to watch television shows.' (incorrect)
            * `lm_syneval__agreement__long_vp_coord__plur_MS_LMV_LMV`:
                * Example: 'The authors know many different foreign languages and like to watch television shows.' (correct) vs. 'The authors know many different foreign languages and likes to watch television shows.' (incorrect)
        * `lm_syneval__agreement__obj_rel_within_anim`: Agreement in an object relative clause with animate external subject
            * `lm_syneval__agreement__obj_rel_within_anim__sing_ES_EV_sing_MS_MV`:
                * Example: 'The author that the guard likes laughs.' (correct) vs. 'The author that the guard like laughs.' (incorrect)
            * `lm_syneval__agreement__obj_rel_within_anim__sing_ES_EV_plur_MS_MV`:
                * Example: 'The authors that the guard likes laugh.' (correct) vs. 'The authors that the guard like laugh.' (incorrect)
            * `lm_syneval__agreement__obj_rel_within_anim__plur_ES_EV_sing_MS_MV`:
                * Example: 'The author that the guards like laughs.' (correct) vs. 'The author that the guards likes laughs.' (incorrect)
            * `lm_syneval__agreement__obj_rel_within_anim__plur_ES_EV_plur_MS_MV`:
                * Example: 'The authors that the guards like laugh.' (correct) vs. 'The authors that the guards likes laugh.' (incorrect)
        * `lm_syneval__agreement__obj_rel_within_inanim`: Agreement in an object relative clause with inanimate external subject
            * `lm_syneval__agreement__obj_rel_within_inanim__sing_ES_EV_sing_IS_IV`:
                * Example: 'The movie that the guard likes is good.' (correct) vs. 'The movie that the guard like is good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_within_inanim__sing_ES_EV_plur_IS_IV`:
                * Example: 'The movies that the guard likes are good.' (correct) vs. 'The movies that the guard like are good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_within_inanim__plur_ES_EV_sing_IS_IV`:
                * Example: 'The movie that the guards like is good.' (correct) vs. 'The movie that the guards likes is good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_within_inanim__plur_ES_EV_plur_IS_IV`:
                * Example: 'The movies that the guards like are good.' (correct) vs. 'The movies that the guards likes are good.' (incorrect)
        * `lm_syneval__agreement__obj_rel_across_anim`: Agreement across an object relative clause with animate external subject
            * `lm_syneval__agreement__obj_rel_across_anim__sing_MS_MV_sing_ES_EV`:
                * Example: 'The author that the guard likes laughs.' (correct) vs. 'The author that the guard likes laugh.' (incorrect)
            * `lm_syneval__agreement__obj_rel_across_anim__sing_MS_MV_plur_ES_EV`:
                * Example: 'The author that the guards like laughs.' (correct) vs. 'The author that the guards like laugh.' (incorrect)
            * `lm_syneval__agreement__obj_rel_across_anim__plur_MS_MV_sing_ES_EV`:
                * Example: 'The authors that the guard likes laugh.' (correct) vs. 'The authors that the guard likes laughs.' (incorrect)
            * `lm_syneval__agreement__obj_rel_across_anim__plur_MS_MV_plur_ES_EV`:
                * Example: 'The authors that the guards like laugh.' (correct) vs. 'The authors that the guards like laughs.' (incorrect)
        * `lm_syneval__agreement__obj_rel_across_inanim`: Agreement across an object relative clause with inanimate external subject
            * `lm_syneval__agreement__obj_rel_across_inanim__sing_IS_IV_sing_ES_EV`:
                * Example: 'The movie that the guard likes is good.' (correct) vs. 'The movie that the guard likes are good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_across_inanim__sing_IS_IV_plur_ES_EV`:
                * Example: 'The movie that the guards like is good.' (correct) vs. 'The movie that the guards like are good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_across_inanim__plur_IS_IV_sing_ES_EV`:
                * Example: 'The movies that the guard likes are good.' (correct) vs. 'The movies that the guard likes is good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_across_inanim__plur_IS_IV_plur_ES_EV`:
                * Example: 'The movies that the guards like are good.' (correct) vs. 'The movies that the guards like is good.' (incorrect)
        * `lm_syneval__agreement__obj_rel_no_comp_within_anim`: Agreement in an object relative clause (no _that_) with animate external subject
            * `lm_syneval__agreement__obj_rel_no_comp_within_anim__sing_ES_EV_sing_MS_MV`:
                * Example: 'The author the guard likes laughs.' (correct) vs. 'The author the guard like laughs.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_within_anim__sing_ES_EV_plur_MS_MV`:
                * Example: 'The authors the guard likes laugh.' (correct) vs. 'The authors the guard like laugh.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_within_anim__plur_ES_EV_sing_MS_MV`:
                * Example: 'The author the guards like laughs.' (correct) vs. 'The author the guards likes laughs.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_within_anim__plur_ES_EV_plur_MS_MV`:
                * Example: 'The authors the guards like laugh.' (correct) vs. 'The authors the guards likes laugh.' (incorrect)
        * `lm_syneval__agreement__obj_rel_no_comp_within_inanim`: Agreement in an object relative clause (no _that_) with inanimate external subject
            * `lm_syneval__agreement__obj_rel_no_comp_within_inanim__sing_ES_EV_sing_IS_IV`:
                * Example: 'The movie the guard likes is good.' (correct) vs. 'The movie the guard like is good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_within_inanim__sing_ES_EV_plur_IS_IV`:
                * Example: 'The movies the guard likes are good.' (correct) vs. 'The movies the guard like are good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_within_inanim__plur_ES_EV_sing_IS_IV`:
                * Example: 'The movie the guards like is good.' (correct) vs. 'The movie the guards likes is good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_within_inanim__plur_ES_EV_plur_IS_IV`:
                * Example: 'The movies the guards like are good.' (correct) vs. 'The movies the guards likes are good.' (incorrect)
        * `lm_syneval__agreement__obj_rel_no_comp_across_anim`: Agreement across an object relative clause (no _that_) with animate external subject
            * `lm_syneval__agreement__obj_rel_no_comp_across_anim__sing_MS_MV_sing_ES_EV`:
                * Example: 'The author the guard likes laughs.' (correct) vs. 'The author the guard like laughs.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_across_anim__sing_MS_MV_plur_ES_EV`:
                * Example: 'The authors the guard likes laugh.' (correct) vs. 'The authors the guard like laugh.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_across_anim__plur_MS_MV_sing_ES_EV`:
                * Example: 'The author the guards like laughs.' (correct) vs. 'The author the guards likes laughs.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_across_anim__plur_MS_MV_plur_ES_EV`:
                * Example: 'The authors the guards like laugh.' (correct) vs. 'The authors the guards likes laugh.' (incorrect)
        * `lm_syneval__agreement__obj_rel_no_comp_across_inanim`: Agreement across an object relative clause (no _that_) with inanimate external subject
            * `lm_syneval__agreement__obj_rel_no_comp_across_inanim__sing_IS_IV_sing_ES_EV`:
                * Example: 'The movie the guard likes is good.' (correct) vs. 'The movie the guard likes are good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_across_inanim__sing_IS_IV_plur_ES_EV`:
                * Example: 'The movie the guards like is good.' (correct) vs. 'The movie the guards like are good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_across_inanim__plur_IS_IV_sing_ES_EV`:
                * Example: 'The movies the guard likes are good.' (correct) vs. 'The movies the guard likes is good.' (incorrect)
            * `lm_syneval__agreement__obj_rel_no_comp_across_inanim__plur_IS_IV_plur_ES_EV`:
                * Example: 'The movies the guards like are good.' (correct) vs. 'The movies the guards like is good.' (incorrect)
    * `lm_syneval__reflexives`: Reflexive anaphora
        * `lm_syneval__reflexives__simple_reflexives`: Simple Reflexives
            * `lm_syneval__reflexives__simple_reflexives__sing_MS_ANPHR`:
                * Example: 'The author hurt himself.' (correct) vs 'The author hurt themselves.' (incorrect)
            * `lm_syneval__reflexives__simple_reflexives__plur_MS_ANPHR`:
                * Example: 'The authors hurt themselves.' (correct) vs. 'The authors hurt himself.' (incorrect)
        * `lm_syneval__reflexives__reflexive_sent_comp`: Reflexives in a sentential complement
            * `lm_syneval__reflexives__reflexive_sent_comp__sing_MS_ANPHR_sing_BS`:
                * Example: 'The mechanic said the author hurt himself.' (correct) vs. 'The mechanic said the author hurt themselves.' (incorrect)
            * `lm_syneval__reflexives__reflexive_sent_comp__sing_MS_ANPHR_plur_BS`:
                * Example: 'The mechanics said the author hurt himself.' (correct) vs. 'The mechanics said the author hurt themselves.' (incorrect)
            * `lm_syneval__reflexives__reflexive_sent_comp__plur_MS_ANPHR_sing_BS`:
                * Example: 'The mechanic said the authors hurt themselves.' (correct) vs. 'The mechanic said the authors hurt himself.' (incorrect)
            * `lm_syneval__reflexives__reflexive_sent_comp__plur_MS_ANPHR_plur_BS`:
                * Example: 'The mechanics said the authors hurt themselves.' (correct) vs. 'The mechanics said the authors hurt himself.' (incorrect)
        * `lm_syneval__reflexives__reflexives_across`: Reflexive across an object relative clause
            * `lm_syneval__reflexives__reflexives_across__sing_MS_ANPHR_sing_ES_EV`:
                * Example: 'The author that the guard likes hurt himself.' (correct) vs. 'The author that the guard likes hurt themselves.' (incorrect)
            * `lm_syneval__reflexives__reflexives_across__sing_MS_ANPHR_plur_ES_EV`:
                * Example: 'The author that the guards like hurt himself.' (correct) vs. 'The author that the guards like hurt themselves.' (incorrect)
            * `lm_syneval__reflexives__reflexives_across__plur_MS_ANPHR_sing_ES_EV`:
                * Example: 'The authors that the guard likes hurt themselves.' (correct) vs. 'The authors that the guard likes hurt himself.' (incorrect)
            * `lm_syneval__reflexives__reflexives_across__plur_MS_ANPHR_plur_ES_EV`:
                * Example: 'The authors that the guards like hurt themselves.' (correct) vs. 'The authors that the guards like hurt himself.' (incorrect)
    * `lm_syneval__npi`: Negative polarity items
        * `lm_syneval__npi__simple_npi_anim`: Simple NPI with animate subject
            * `lm_syneval__npi__simple_npi_anim__past`:
                * Example: 'No authors have ever been popular.' (correct) vs. 'The authors have ever been popular.' (incorrect)
            * `lm_syneval__npi__simple_npi_anim__future`:
                * Example: 'No authors will ever be popular.' (correct) vs. 'The authors will ever be popular.' (incorrect)
        * `lm_syneval__npi__simple_npi_inanim`: Simple NPI with imanimate subject
            * `lm_syneval__npi__simple_npi_inanim__past`:
                * Example: 'No movies have ever been seen.' (correct) vs. 'The movies have ever been seen.' (incorrect)
            * `lm_syneval__npi__simple_npi_inanim__future`:
                * Example: 'No movies will ever be seen.' (correct) vs. 'The movies will ever be seen.' (incorrect)
        * `lm_syneval__npi__npi_across_anim`: NPI across a relative clause with animate subject
            * `lm_syneval__npi__npi_across_anim__past`:
                * Example: 'No authors that the guards like have ever been popular.' (correct) vs. 'The authors that no guards like have ever been popular.' (incorrect)
            * `lm_syneval__npi__npi_across_anim__future`:
                * Example: 'No authors that the guards like will ever be popular.' (correct) vs. 'The authors that no guards like will ever be popular.' (incorrect)
        * `lm_syneval__npi__npi_across_inanim`: NPI across a relative clause with imanimate subject
            * `lm_syneval__npi__npi_across_inanim__past`:
                * Example: 'No movies that the guards like have ever been seen.' (correct) vs. 'The movies that no guards like have ever been seen.' (incorrect)
            * `lm_syneval__npi__npi_across_inanim__future`:
                * Example: 'No movies that the guards like will ever be seen.' (correct) vs. 'The movies that no guards like will ever be seen.' (incorrect)



## Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
      * The original paper evaluates traditional RNN models, which require a very different pipeline to analyze.

## Changelog
