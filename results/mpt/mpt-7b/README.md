# mpt-7b

## mpt-7b_anli_0-shot.json
| Task  |Version|Metric|Value|   |Stderr|
|-------|------:|------|----:|---|-----:|
|anli_r1|      0|acc   | 33.2|±  |  1.49|
|anli_r2|      0|acc   | 33.6|±  |  1.49|
|anli_r3|      0|acc   | 34.5|±  |  1.37|

## mpt-7b_arithmetic_5-shot.json
|     Task     |Version|Metric|Value|   |Stderr|
|--------------|------:|------|----:|---|-----:|
|arithmetic_1dc|      0|acc   | 8.10|±  |  0.61|
|arithmetic_2da|      0|acc   |91.80|±  |  0.61|
|arithmetic_2dm|      0|acc   |25.60|±  |  0.98|
|arithmetic_2ds|      0|acc   |78.75|±  |  0.91|
|arithmetic_3da|      0|acc   |29.15|±  |  1.02|
|arithmetic_3ds|      0|acc   |42.80|±  |  1.11|
|arithmetic_4da|      0|acc   | 2.60|±  |  0.36|
|arithmetic_4ds|      0|acc   | 2.60|±  |  0.36|
|arithmetic_5da|      0|acc   | 0.45|±  |  0.15|
|arithmetic_5ds|      0|acc   | 0.20|±  |  0.10|

## mpt-7b_bbh_3-shot.json
|                      Task                      |Version|       Metric        |Value|   |Stderr|
|------------------------------------------------|------:|---------------------|----:|---|-----:|
|bigbench_causal_judgement                       |      0|multiple_choice_grade|56.32|±  |  3.61|
|bigbench_date_understanding                     |      0|multiple_choice_grade|58.27|±  |  2.57|
|bigbench_disambiguation_qa                      |      0|multiple_choice_grade|36.43|±  |  3.00|
|bigbench_dyck_languages                         |      0|multiple_choice_grade|12.30|±  |  1.04|
|bigbench_formal_fallacies_syllogisms_negation   |      0|multiple_choice_grade|49.92|±  |  0.42|
|bigbench_geometric_shapes                       |      0|multiple_choice_grade|20.33|±  |  2.13|
|                                                |       |exact_str_match      |12.26|±  |  1.73|
|bigbench_hyperbaton                             |      0|multiple_choice_grade|49.36|±  |  0.22|
|bigbench_logical_deduction_five_objects         |      0|multiple_choice_grade|24.00|±  |  1.91|
|bigbench_logical_deduction_seven_objects        |      0|multiple_choice_grade|16.57|±  |  1.41|
|bigbench_logical_deduction_three_objects        |      0|multiple_choice_grade|38.67|±  |  2.82|
|bigbench_movie_recommendation                   |      0|multiple_choice_grade|43.80|±  |  2.22|
|bigbench_navigate                               |      0|multiple_choice_grade|48.60|±  |  1.58|
|bigbench_reasoning_about_colored_objects        |      0|multiple_choice_grade|29.85|±  |  1.02|
|bigbench_ruin_names                             |      0|multiple_choice_grade|29.69|±  |  2.16|
|bigbench_salient_translation_error_detection    |      0|multiple_choice_grade|17.94|±  |  1.22|
|bigbench_snarks                                 |      0|multiple_choice_grade|53.04|±  |  3.72|
|bigbench_sports_understanding                   |      0|multiple_choice_grade|49.49|±  |  1.59|
|bigbench_temporal_sequences                     |      0|multiple_choice_grade|29.60|±  |  1.44|
|bigbench_tracking_shuffled_objects_five_objects |      0|multiple_choice_grade|19.44|±  |  1.12|
|bigbench_tracking_shuffled_objects_seven_objects|      0|multiple_choice_grade|13.43|±  |  0.82|
|bigbench_tracking_shuffled_objects_three_objects|      0|multiple_choice_grade|38.67|±  |  2.82|

## mpt-7b_blimp_0-shot.json
|                          Task                           |Version|Metric|Value|   |Stderr|
|---------------------------------------------------------|------:|------|----:|---|-----:|
|blimp_adjunct_island                                     |      0|acc   | 87.8|±  |  1.04|
|blimp_anaphor_gender_agreement                           |      0|acc   | 99.5|±  |  0.22|
|blimp_anaphor_number_agreement                           |      0|acc   | 99.5|±  |  0.22|
|blimp_animate_subject_passive                            |      0|acc   | 77.5|±  |  1.32|
|blimp_animate_subject_trans                              |      0|acc   | 88.4|±  |  1.01|
|blimp_causative                                          |      0|acc   | 74.7|±  |  1.38|
|blimp_complex_NP_island                                  |      0|acc   | 52.8|±  |  1.58|
|blimp_coordinate_structure_constraint_complex_left_branch|      0|acc   | 77.9|±  |  1.31|
|blimp_coordinate_structure_constraint_object_extraction  |      0|acc   | 84.8|±  |  1.14|
|blimp_determiner_noun_agreement_1                        |      0|acc   | 99.2|±  |  0.28|
|blimp_determiner_noun_agreement_2                        |      0|acc   | 97.4|±  |  0.50|
|blimp_determiner_noun_agreement_irregular_1              |      0|acc   | 93.6|±  |  0.77|
|blimp_determiner_noun_agreement_irregular_2              |      0|acc   | 92.8|±  |  0.82|
|blimp_determiner_noun_agreement_with_adj_2               |      0|acc   | 93.5|±  |  0.78|
|blimp_determiner_noun_agreement_with_adj_irregular_1     |      0|acc   | 88.3|±  |  1.02|
|blimp_determiner_noun_agreement_with_adj_irregular_2     |      0|acc   | 91.9|±  |  0.86|
|blimp_determiner_noun_agreement_with_adjective_1         |      0|acc   | 97.2|±  |  0.52|
|blimp_distractor_agreement_relational_noun               |      0|acc   | 88.9|±  |  0.99|
|blimp_distractor_agreement_relative_clause               |      0|acc   | 74.3|±  |  1.38|
|blimp_drop_argument                                      |      0|acc   | 78.7|±  |  1.30|
|blimp_ellipsis_n_bar_1                                   |      0|acc   | 79.0|±  |  1.29|
|blimp_ellipsis_n_bar_2                                   |      0|acc   | 92.8|±  |  0.82|
|blimp_existential_there_object_raising                   |      0|acc   | 83.4|±  |  1.18|
|blimp_existential_there_quantifiers_1                    |      0|acc   | 98.8|±  |  0.34|
|blimp_existential_there_quantifiers_2                    |      0|acc   | 27.0|±  |  1.40|
|blimp_existential_there_subject_raising                  |      0|acc   | 88.8|±  |  1.00|
|blimp_expletive_it_object_raising                        |      0|acc   | 80.0|±  |  1.27|
|blimp_inchoative                                         |      0|acc   | 67.3|±  |  1.48|
|blimp_intransitive                                       |      0|acc   | 83.2|±  |  1.18|
|blimp_irregular_past_participle_adjectives               |      0|acc   | 97.2|±  |  0.52|
|blimp_irregular_past_participle_verbs                    |      0|acc   | 88.5|±  |  1.01|
|blimp_irregular_plural_subject_verb_agreement_1          |      0|acc   | 92.0|±  |  0.86|
|blimp_irregular_plural_subject_verb_agreement_2          |      0|acc   | 90.8|±  |  0.91|
|blimp_left_branch_island_echo_question                   |      0|acc   | 43.2|±  |  1.57|
|blimp_left_branch_island_simple_question                 |      0|acc   | 89.7|±  |  0.96|
|blimp_matrix_question_npi_licensor_present               |      0|acc   | 70.5|±  |  1.44|
|blimp_npi_present_1                                      |      0|acc   | 57.9|±  |  1.56|
|blimp_npi_present_2                                      |      0|acc   | 68.8|±  |  1.47|
|blimp_only_npi_licensor_present                          |      0|acc   | 91.6|±  |  0.88|
|blimp_only_npi_scope                                     |      0|acc   | 73.2|±  |  1.40|
|blimp_passive_1                                          |      0|acc   | 88.7|±  |  1.00|
|blimp_passive_2                                          |      0|acc   | 89.5|±  |  0.97|
|blimp_principle_A_c_command                              |      0|acc   | 75.0|±  |  1.37|
|blimp_principle_A_case_1                                 |      0|acc   |100.0|±  |  0.00|
|blimp_principle_A_case_2                                 |      0|acc   | 94.0|±  |  0.75|
|blimp_principle_A_domain_1                               |      0|acc   | 99.7|±  |  0.17|
|blimp_principle_A_domain_2                               |      0|acc   | 82.8|±  |  1.19|
|blimp_principle_A_domain_3                               |      0|acc   | 76.1|±  |  1.35|
|blimp_principle_A_reconstruction                         |      0|acc   | 41.0|±  |  1.56|
|blimp_regular_plural_subject_verb_agreement_1            |      0|acc   | 97.1|±  |  0.53|
|blimp_regular_plural_subject_verb_agreement_2            |      0|acc   | 90.7|±  |  0.92|
|blimp_sentential_negation_npi_licensor_present           |      0|acc   | 98.9|±  |  0.33|
|blimp_sentential_negation_npi_scope                      |      0|acc   | 73.3|±  |  1.40|
|blimp_sentential_subject_island                          |      0|acc   | 39.9|±  |  1.55|
|blimp_superlative_quantifiers_1                          |      0|acc   | 82.2|±  |  1.21|
|blimp_superlative_quantifiers_2                          |      0|acc   | 89.7|±  |  0.96|
|blimp_tough_vs_raising_1                                 |      0|acc   | 69.0|±  |  1.46|
|blimp_tough_vs_raising_2                                 |      0|acc   | 82.9|±  |  1.19|
|blimp_transitive                                         |      0|acc   | 87.2|±  |  1.06|
|blimp_wh_island                                          |      0|acc   | 81.3|±  |  1.23|
|blimp_wh_questions_object_gap                            |      0|acc   | 76.3|±  |  1.35|
|blimp_wh_questions_subject_gap                           |      0|acc   | 89.0|±  |  0.99|
|blimp_wh_questions_subject_gap_long_distance             |      0|acc   | 89.3|±  |  0.98|
|blimp_wh_vs_that_no_gap                                  |      0|acc   | 94.6|±  |  0.72|
|blimp_wh_vs_that_no_gap_long_distance                    |      0|acc   | 95.1|±  |  0.68|
|blimp_wh_vs_that_with_gap                                |      0|acc   | 32.1|±  |  1.48|
|blimp_wh_vs_that_with_gap_long_distance                  |      0|acc   | 29.2|±  |  1.44|

## mpt-7b_common_sense_reasoning_0-shot.json
|    Task     |Version| Metric |Value|   |Stderr|
|-------------|------:|--------|----:|---|-----:|
|arc_challenge|      0|acc     |40.61|±  |  1.44|
|             |       |acc_norm|41.81|±  |  1.44|
|arc_easy     |      0|acc     |74.87|±  |  0.89|
|             |       |acc_norm|70.29|±  |  0.94|
|boolq        |      1|acc     |73.52|±  |  0.77|
|copa         |      0|acc     |85.00|±  |  3.59|
|hellaswag    |      0|acc     |57.24|±  |  0.49|
|             |       |acc_norm|76.12|±  |  0.43|
|mc_taco      |      0|em      |13.51|   |      |
|             |       |f1      |45.48|   |      |
|openbookqa   |      0|acc     |32.00|±  |  2.09|
|             |       |acc_norm|42.60|±  |  2.21|
|piqa         |      0|acc     |79.16|±  |  0.95|
|             |       |acc_norm|80.41|±  |  0.93|
|prost        |      0|acc     |25.73|±  |  0.32|
|             |       |acc_norm|30.12|±  |  0.34|
|swag         |      0|acc     |56.17|±  |  0.35|
|             |       |acc_norm|75.80|±  |  0.30|
|winogrande   |      0|acc     |68.67|±  |  1.30|
|wsc273       |      0|acc     |85.71|±  |  2.12|

## mpt-7b_glue_0-shot.json
|     Task      |Version|Metric|Value|   |Stderr|
|---------------|------:|------|----:|---|-----:|
|cola           |      0|mcc   |-4.41|±  |  3.12|
|mnli           |      0|acc   |37.83|±  |  0.49|
|mnli_mismatched|      0|acc   |37.49|±  |  0.49|
|mrpc           |      0|acc   |62.99|±  |  2.39|
|               |       |f1    |75.61|±  |  1.93|
|qnli           |      0|acc   |51.35|±  |  0.68|
|qqp            |      0|acc   |50.36|±  |  0.25|
|               |       |f1    |54.14|±  |  0.29|
|rte            |      0|acc   |63.90|±  |  2.89|
|sst            |      0|acc   |76.83|±  |  1.43|
|wnli           |      1|acc   |47.89|±  |  5.97|

## mpt-7b_human_alignment_0-shot.json
|                 Task                  |Version|       Metric        |Value |   |Stderr|
|---------------------------------------|------:|---------------------|-----:|---|-----:|
|crows_pairs_english_age                |      0|likelihood_difference|415.11|±  | 38.32|
|                                       |       |pct_stereotype       | 73.63|±  |  4.64|
|crows_pairs_english_autre              |      0|likelihood_difference|505.68|±  |177.03|
|                                       |       |pct_stereotype       | 72.73|±  | 14.08|
|crows_pairs_english_disability         |      0|likelihood_difference|601.92|±  | 63.31|
|                                       |       |pct_stereotype       | 76.92|±  |  5.27|
|crows_pairs_english_gender             |      0|likelihood_difference|268.24|±  | 17.01|
|                                       |       |pct_stereotype       | 63.75|±  |  2.69|
|crows_pairs_english_nationality        |      0|likelihood_difference|349.83|±  | 21.51|
|                                       |       |pct_stereotype       | 61.57|±  |  3.32|
|crows_pairs_english_physical_appearance|      0|likelihood_difference|373.78|±  | 33.85|
|                                       |       |pct_stereotype       | 72.22|±  |  5.32|
|crows_pairs_english_race_color         |      0|likelihood_difference|336.20|±  | 14.10|
|                                       |       |pct_stereotype       | 57.28|±  |  2.20|
|crows_pairs_english_religion           |      0|likelihood_difference|366.44|±  | 33.86|
|                                       |       |pct_stereotype       | 72.97|±  |  4.23|
|crows_pairs_english_sexual_orientation |      0|likelihood_difference|463.04|±  | 45.75|
|                                       |       |pct_stereotype       | 82.80|±  |  3.93|
|crows_pairs_english_socioeconomic      |      0|likelihood_difference|406.51|±  | 23.52|
|                                       |       |pct_stereotype       | 67.89|±  |  3.40|
|crows_pairs_french_age                 |      0|likelihood_difference|360.97|±  | 36.15|
|                                       |       |pct_stereotype       | 42.22|±  |  5.24|
|crows_pairs_french_autre               |      0|likelihood_difference|269.23|±  | 92.30|
|                                       |       |pct_stereotype       | 61.54|±  | 14.04|
|crows_pairs_french_disability          |      0|likelihood_difference|495.83|±  | 42.69|
|                                       |       |pct_stereotype       | 63.64|±  |  5.97|
|crows_pairs_french_gender              |      0|likelihood_difference|321.38|±  | 17.59|
|                                       |       |pct_stereotype       | 51.09|±  |  2.79|
|crows_pairs_french_nationality         |      0|likelihood_difference|388.34|±  | 21.84|
|                                       |       |pct_stereotype       | 34.39|±  |  2.99|
|crows_pairs_french_physical_appearance |      0|likelihood_difference|322.74|±  | 43.29|
|                                       |       |pct_stereotype       | 59.72|±  |  5.82|
|crows_pairs_french_race_color          |      0|likelihood_difference|316.14|±  | 16.56|
|                                       |       |pct_stereotype       | 43.70|±  |  2.32|
|crows_pairs_french_religion            |      0|likelihood_difference|356.74|±  | 33.68|
|                                       |       |pct_stereotype       | 62.61|±  |  4.53|
|crows_pairs_french_sexual_orientation  |      0|likelihood_difference|479.12|±  | 40.10|
|                                       |       |pct_stereotype       | 78.02|±  |  4.36|
|crows_pairs_french_socioeconomic       |      0|likelihood_difference|399.39|±  | 26.31|
|                                       |       |pct_stereotype       | 65.82|±  |  3.40|
|ethics_cm                              |      0|acc                  | 54.59|±  |  0.80|
|ethics_deontology                      |      0|acc                  | 50.25|±  |  0.83|
|                                       |       |em                   |  0.44|   |      |
|ethics_justice                         |      0|acc                  | 51.96|±  |  0.96|
|                                       |       |em                   |  1.18|   |      |
|ethics_utilitarianism                  |      0|acc                  | 57.49|±  |  0.71|
|ethics_utilitarianism_original         |      0|acc                  | 99.56|±  |  0.10|
|ethics_virtue                          |      0|acc                  | 80.40|±  |  0.56|
|                                       |       |em                   | 12.56|   |      |
|toxigen                                |      0|acc                  | 43.19|±  |  1.62|
|                                       |       |acc_norm             | 43.19|±  |  1.62|

## mpt-7b_lambada_0-shot.json
|         Task         |Version|Metric|Value |   |Stderr|
|----------------------|------:|------|-----:|---|-----:|
|lambada_openai        |      0|ppl   |  3.87|±  |  0.08|
|                      |       |acc   | 68.35|±  |  0.65|
|lambada_openai_cloze  |      0|ppl   | 26.56|±  |  0.70|
|                      |       |acc   | 39.65|±  |  0.68|
|lambada_openai_mt_de  |      0|ppl   | 70.12|±  |  4.04|
|                      |       |acc   | 33.77|±  |  0.66|
|lambada_openai_mt_en  |      0|ppl   |  3.87|±  |  0.08|
|                      |       |acc   | 68.35|±  |  0.65|
|lambada_openai_mt_es  |      0|ppl   | 67.23|±  |  3.69|
|                      |       |acc   | 36.95|±  |  0.67|
|lambada_openai_mt_fr  |      0|ppl   | 42.93|±  |  2.37|
|                      |       |acc   | 43.02|±  |  0.69|
|lambada_openai_mt_it  |      0|ppl   | 65.76|±  |  3.87|
|                      |       |acc   | 39.20|±  |  0.68|
|lambada_standard      |      0|ppl   |  4.92|±  |  0.11|
|                      |       |acc   | 61.91|±  |  0.68|
|lambada_standard_cloze|      0|ppl   |109.10|±  |  3.04|
|                      |       |acc   | 16.75|±  |  0.52|

## mpt-7b_mmlu_5-shot.json
|                      Task                       |Version| Metric |Value|   |Stderr|
|-------------------------------------------------|------:|--------|----:|---|-----:|
|hendrycksTest-abstract_algebra                   |      0|acc     |18.00|±  |  3.86|
|                                                 |       |acc_norm|21.00|±  |  4.09|
|hendrycksTest-anatomy                            |      0|acc     |38.52|±  |  4.20|
|                                                 |       |acc_norm|37.78|±  |  4.19|
|hendrycksTest-astronomy                          |      0|acc     |39.47|±  |  3.98|
|                                                 |       |acc_norm|42.11|±  |  4.02|
|hendrycksTest-business_ethics                    |      0|acc     |49.00|±  |  5.02|
|                                                 |       |acc_norm|48.00|±  |  5.02|
|hendrycksTest-clinical_knowledge                 |      0|acc     |33.21|±  |  2.90|
|                                                 |       |acc_norm|37.74|±  |  2.98|
|hendrycksTest-college_biology                    |      0|acc     |38.19|±  |  4.06|
|                                                 |       |acc_norm|35.42|±  |  4.00|
|hendrycksTest-college_chemistry                  |      0|acc     |39.00|±  |  4.90|
|                                                 |       |acc_norm|41.00|±  |  4.94|
|hendrycksTest-college_computer_science           |      0|acc     |34.00|±  |  4.76|
|                                                 |       |acc_norm|32.00|±  |  4.69|
|hendrycksTest-college_mathematics                |      0|acc     |27.00|±  |  4.46|
|                                                 |       |acc_norm|33.00|±  |  4.73|
|hendrycksTest-college_medicine                   |      0|acc     |36.42|±  |  3.67|
|                                                 |       |acc_norm|34.68|±  |  3.63|
|hendrycksTest-college_physics                    |      0|acc     |30.39|±  |  4.58|
|                                                 |       |acc_norm|33.33|±  |  4.69|
|hendrycksTest-computer_security                  |      0|acc     |41.00|±  |  4.94|
|                                                 |       |acc_norm|41.00|±  |  4.94|
|hendrycksTest-conceptual_physics                 |      0|acc     |32.77|±  |  3.07|
|                                                 |       |acc_norm|25.53|±  |  2.85|
|hendrycksTest-econometrics                       |      0|acc     |27.19|±  |  4.19|
|                                                 |       |acc_norm|23.68|±  |  4.00|
|hendrycksTest-electrical_engineering             |      0|acc     |36.55|±  |  4.01|
|                                                 |       |acc_norm|33.79|±  |  3.94|
|hendrycksTest-elementary_mathematics             |      0|acc     |29.89|±  |  2.36|
|                                                 |       |acc_norm|28.84|±  |  2.33|
|hendrycksTest-formal_logic                       |      0|acc     |30.95|±  |  4.13|
|                                                 |       |acc_norm|28.57|±  |  4.04|
|hendrycksTest-global_facts                       |      0|acc     |35.00|±  |  4.79|
|                                                 |       |acc_norm|33.00|±  |  4.73|
|hendrycksTest-high_school_biology                |      0|acc     |36.45|±  |  2.74|
|                                                 |       |acc_norm|39.03|±  |  2.78|
|hendrycksTest-high_school_chemistry              |      0|acc     |21.18|±  |  2.87|
|                                                 |       |acc_norm|21.67|±  |  2.90|
|hendrycksTest-high_school_computer_science       |      0|acc     |43.00|±  |  4.98|
|                                                 |       |acc_norm|41.00|±  |  4.94|
|hendrycksTest-high_school_european_history       |      0|acc     |38.18|±  |  3.79|
|                                                 |       |acc_norm|37.58|±  |  3.78|
|hendrycksTest-high_school_geography              |      0|acc     |38.38|±  |  3.46|
|                                                 |       |acc_norm|40.40|±  |  3.50|
|hendrycksTest-high_school_government_and_politics|      0|acc     |41.45|±  |  3.56|
|                                                 |       |acc_norm|41.45|±  |  3.56|
|hendrycksTest-high_school_macroeconomics         |      0|acc     |34.87|±  |  2.42|
|                                                 |       |acc_norm|29.74|±  |  2.32|
|hendrycksTest-high_school_mathematics            |      0|acc     |29.26|±  |  2.77|
|                                                 |       |acc_norm|30.37|±  |  2.80|
|hendrycksTest-high_school_microeconomics         |      0|acc     |33.61|±  |  3.07|
|                                                 |       |acc_norm|36.97|±  |  3.14|
|hendrycksTest-high_school_physics                |      0|acc     |27.81|±  |  3.66|
|                                                 |       |acc_norm|27.81|±  |  3.66|
|hendrycksTest-high_school_psychology             |      0|acc     |46.97|±  |  2.14|
|                                                 |       |acc_norm|44.59|±  |  2.13|
|hendrycksTest-high_school_statistics             |      0|acc     |32.87|±  |  3.20|
|                                                 |       |acc_norm|32.41|±  |  3.19|
|hendrycksTest-high_school_us_history             |      0|acc     |34.31|±  |  3.33|
|                                                 |       |acc_norm|31.37|±  |  3.26|
|hendrycksTest-high_school_world_history          |      0|acc     |29.54|±  |  2.97|
|                                                 |       |acc_norm|28.69|±  |  2.94|
|hendrycksTest-human_aging                        |      0|acc     |33.63|±  |  3.17|
|                                                 |       |acc_norm|32.74|±  |  3.15|
|hendrycksTest-human_sexuality                    |      0|acc     |27.48|±  |  3.92|
|                                                 |       |acc_norm|32.82|±  |  4.12|
|hendrycksTest-international_law                  |      0|acc     |37.19|±  |  4.41|
|                                                 |       |acc_norm|49.59|±  |  4.56|
|hendrycksTest-jurisprudence                      |      0|acc     |34.26|±  |  4.59|
|                                                 |       |acc_norm|39.81|±  |  4.73|
|hendrycksTest-logical_fallacies                  |      0|acc     |38.04|±  |  3.81|
|                                                 |       |acc_norm|36.81|±  |  3.79|
|hendrycksTest-machine_learning                   |      0|acc     |26.79|±  |  4.20|
|                                                 |       |acc_norm|24.11|±  |  4.06|
|hendrycksTest-management                         |      0|acc     |42.72|±  |  4.90|
|                                                 |       |acc_norm|39.81|±  |  4.85|
|hendrycksTest-marketing                          |      0|acc     |55.13|±  |  3.26|
|                                                 |       |acc_norm|55.13|±  |  3.26|
|hendrycksTest-medical_genetics                   |      0|acc     |39.00|±  |  4.90|
|                                                 |       |acc_norm|38.00|±  |  4.88|
|hendrycksTest-miscellaneous                      |      0|acc     |55.56|±  |  1.78|
|                                                 |       |acc_norm|55.68|±  |  1.78|
|hendrycksTest-moral_disputes                     |      0|acc     |32.08|±  |  2.51|
|                                                 |       |acc_norm|30.06|±  |  2.47|
|hendrycksTest-moral_scenarios                    |      0|acc     |26.03|±  |  1.47|
|                                                 |       |acc_norm|27.26|±  |  1.49|
|hendrycksTest-nutrition                          |      0|acc     |34.31|±  |  2.72|
|                                                 |       |acc_norm|40.20|±  |  2.81|
|hendrycksTest-philosophy                         |      0|acc     |37.62|±  |  2.75|
|                                                 |       |acc_norm|36.98|±  |  2.74|
|hendrycksTest-prehistory                         |      0|acc     |33.64|±  |  2.63|
|                                                 |       |acc_norm|30.56|±  |  2.56|
|hendrycksTest-professional_accounting            |      0|acc     |30.50|±  |  2.75|
|                                                 |       |acc_norm|29.08|±  |  2.71|
|hendrycksTest-professional_law                   |      0|acc     |25.95|±  |  1.12|
|                                                 |       |acc_norm|28.42|±  |  1.15|
|hendrycksTest-professional_medicine              |      0|acc     |29.41|±  |  2.77|
|                                                 |       |acc_norm|31.62|±  |  2.82|
|hendrycksTest-professional_psychology            |      0|acc     |31.54|±  |  1.88|
|                                                 |       |acc_norm|30.23|±  |  1.86|
|hendrycksTest-public_relations                   |      0|acc     |41.82|±  |  4.72|
|                                                 |       |acc_norm|42.73|±  |  4.74|
|hendrycksTest-security_studies                   |      0|acc     |28.16|±  |  2.88|
|                                                 |       |acc_norm|24.08|±  |  2.74|
|hendrycksTest-sociology                          |      0|acc     |34.33|±  |  3.36|
|                                                 |       |acc_norm|36.82|±  |  3.41|
|hendrycksTest-us_foreign_policy                  |      0|acc     |38.00|±  |  4.88|
|                                                 |       |acc_norm|39.00|±  |  4.90|
|hendrycksTest-virology                           |      0|acc     |32.53|±  |  3.65|
|                                                 |       |acc_norm|32.53|±  |  3.65|
|hendrycksTest-world_religions                    |      0|acc     |54.39|±  |  3.82|
|                                                 |       |acc_norm|57.89|±  |  3.79|

## mpt-7b_pawsx_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|pawsx_de|      0|acc   |61.40|±  |  1.09|
|pawsx_en|      0|acc   |70.35|±  |  1.02|
|pawsx_es|      0|acc   |64.95|±  |  1.07|
|pawsx_fr|      0|acc   |62.85|±  |  1.08|
|pawsx_ja|      0|acc   |49.30|±  |  1.12|
|pawsx_ko|      0|acc   |53.65|±  |  1.12|
|pawsx_zh|      0|acc   |56.25|±  |  1.11|

## mpt-7b_reading_comprehension_0-shot.json
|Task|Version|Metric|Value|   |Stderr|
|----|------:|------|----:|---|-----:|
|coqa|      1|f1    |76.51|±  |  1.48|
|    |       |em    |63.02|±  |  1.87|
|drop|      1|em    | 3.43|±  |  0.19|
|    |       |f1    |13.39|±  |  0.25|
|race|      1|acc   |38.66|±  |  1.51|

## mpt-7b_superglue_0-shot.json
| Task  |Version|Metric|Value|   |Stderr|
|-------|------:|------|----:|---|-----:|
|boolq  |      1|acc   |73.82|±  |  0.77|
|cb     |      1|acc   |41.07|±  |  6.63|
|       |       |f1    |21.27|   |      |
|copa   |      0|acc   |84.00|±  |  3.68|
|multirc|      1|acc   | 0.84|±  |  0.30|
|record |      0|f1    |90.10|±  |  0.29|
|       |       |em    |89.30|±  |  0.31|
|wic    |      0|acc   |48.43|±  |  1.98|
|wsc    |      0|acc   |63.46|±  |  4.74|

## mpt-7b_unscramble_0-shot.json
|      Task      |Version|Metric|Value|   |Stderr|
|----------------|------:|------|----:|---|-----:|
|anagrams1       |      0|acc   | 0.00|±  |  0.00|
|anagrams2       |      0|acc   | 0.01|±  |  0.01|
|cycle_letters   |      0|acc   | 0.00|±  |  0.00|
|random_insertion|      0|acc   | 0.04|±  |  0.02|
|reversed_words  |      0|acc   | 0.00|±  |  0.00|

## mpt-7b_xcopa_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|xcopa_et|      0|acc   | 47.4|±  |  2.24|
|xcopa_ht|      0|acc   | 49.8|±  |  2.24|
|xcopa_id|      0|acc   | 56.8|±  |  2.22|
|xcopa_it|      0|acc   | 59.4|±  |  2.20|
|xcopa_qu|      0|acc   | 48.4|±  |  2.24|
|xcopa_sw|      0|acc   | 51.6|±  |  2.24|
|xcopa_ta|      0|acc   | 54.0|±  |  2.23|
|xcopa_th|      0|acc   | 54.2|±  |  2.23|
|xcopa_tr|      0|acc   | 51.6|±  |  2.24|
|xcopa_vi|      0|acc   | 53.6|±  |  2.23|
|xcopa_zh|      0|acc   | 63.2|±  |  2.16|

## mpt-7b_xnli_0-shot.json
| Task  |Version|Metric|Value|   |Stderr|
|-------|------:|------|----:|---|-----:|
|xnli_ar|      0|acc   |33.31|±  |  0.67|
|xnli_bg|      0|acc   |36.83|±  |  0.68|
|xnli_de|      0|acc   |46.45|±  |  0.70|
|xnli_el|      0|acc   |36.19|±  |  0.68|
|xnli_en|      0|acc   |54.33|±  |  0.70|
|xnli_es|      0|acc   |45.65|±  |  0.70|
|xnli_fr|      0|acc   |48.80|±  |  0.71|
|xnli_hi|      0|acc   |34.73|±  |  0.67|
|xnli_ru|      0|acc   |44.43|±  |  0.70|
|xnli_sw|      0|acc   |33.41|±  |  0.67|
|xnli_th|      0|acc   |36.13|±  |  0.68|
|xnli_tr|      0|acc   |37.68|±  |  0.68|
|xnli_ur|      0|acc   |33.63|±  |  0.67|
|xnli_vi|      0|acc   |37.33|±  |  0.68|
|xnli_zh|      0|acc   |35.35|±  |  0.68|

## mpt-7b_xstory_cloze_0-shot.json
|     Task      |Version|Metric|Value|   |Stderr|
|---------------|------:|------|----:|---|-----:|
|xstory_cloze_ar|      0|acc   |48.51|±  |  1.29|
|xstory_cloze_en|      0|acc   |77.90|±  |  1.07|
|xstory_cloze_es|      0|acc   |66.05|±  |  1.22|
|xstory_cloze_eu|      0|acc   |51.09|±  |  1.29|
|xstory_cloze_hi|      0|acc   |51.69|±  |  1.29|
|xstory_cloze_id|      0|acc   |55.20|±  |  1.28|
|xstory_cloze_my|      0|acc   |48.38|±  |  1.29|
|xstory_cloze_ru|      0|acc   |57.25|±  |  1.27|
|xstory_cloze_sw|      0|acc   |49.90|±  |  1.29|
|xstory_cloze_te|      0|acc   |52.95|±  |  1.28|
|xstory_cloze_zh|      0|acc   |59.56|±  |  1.26|

## mpt-7b_xwinograd_0-shot.json
|    Task    |Version|Metric|Value|   |Stderr|
|------------|------:|------|----:|---|-----:|
|xwinograd_en|      0|acc   |86.67|±  |  0.71|
|xwinograd_fr|      0|acc   |66.27|±  |  5.22|
|xwinograd_jp|      0|acc   |60.27|±  |  1.58|
|xwinograd_pt|      0|acc   |66.92|±  |  2.91|
|xwinograd_ru|      0|acc   |69.52|±  |  2.60|
|xwinograd_zh|      0|acc   |71.63|±  |  2.01|
