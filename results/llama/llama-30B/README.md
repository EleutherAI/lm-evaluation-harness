# llama-30B

## llama-30B_bbh_3-shot.json
|                      Task                      |Version|       Metric        |Value|   |Stderr|
|------------------------------------------------|------:|---------------------|----:|---|-----:|
|bigbench_causal_judgement                       |      0|multiple_choice_grade|57.37|±  |  3.60|
|bigbench_date_understanding                     |      0|multiple_choice_grade|69.92|±  |  2.39|
|bigbench_disambiguation_qa                      |      0|multiple_choice_grade|54.26|±  |  3.11|
|bigbench_dyck_languages                         |      0|multiple_choice_grade|21.20|±  |  1.29|
|bigbench_formal_fallacies_syllogisms_negation   |      0|multiple_choice_grade|50.58|±  |  0.42|
|bigbench_geometric_shapes                       |      0|multiple_choice_grade|27.86|±  |  2.37|
|                                                |       |exact_str_match      | 0.00|±  |  0.00|
|bigbench_hyperbaton                             |      0|multiple_choice_grade|51.52|±  |  0.22|
|bigbench_logical_deduction_five_objects         |      0|multiple_choice_grade|36.80|±  |  2.16|
|bigbench_logical_deduction_seven_objects        |      0|multiple_choice_grade|25.29|±  |  1.64|
|bigbench_logical_deduction_three_objects        |      0|multiple_choice_grade|53.00|±  |  2.89|
|bigbench_movie_recommendation                   |      0|multiple_choice_grade|63.20|±  |  2.16|
|bigbench_navigate                               |      0|multiple_choice_grade|49.00|±  |  1.58|
|bigbench_reasoning_about_colored_objects        |      0|multiple_choice_grade|55.65|±  |  1.11|
|bigbench_ruin_names                             |      0|multiple_choice_grade|39.73|±  |  2.31|
|bigbench_salient_translation_error_detection    |      0|multiple_choice_grade|19.84|±  |  1.26|
|bigbench_snarks                                 |      0|multiple_choice_grade|46.96|±  |  3.72|
|bigbench_sports_understanding                   |      0|multiple_choice_grade|62.37|±  |  1.54|
|bigbench_temporal_sequences                     |      0|multiple_choice_grade|14.60|±  |  1.12|
|bigbench_tracking_shuffled_objects_five_objects |      0|multiple_choice_grade|21.28|±  |  1.16|
|bigbench_tracking_shuffled_objects_seven_objects|      0|multiple_choice_grade|15.49|±  |  0.87|
|bigbench_tracking_shuffled_objects_three_objects|      0|multiple_choice_grade|53.00|±  |  2.89|

## llama-30B_common_sense_reasoning_0-shot.json
|    Task     |Version| Metric |Value|   |Stderr|
|-------------|------:|--------|----:|---|-----:|
|arc_challenge|      0|acc     |46.76|±  |  1.46|
|             |       |acc_norm|45.48|±  |  1.46|
|arc_easy     |      0|acc     |75.34|±  |  0.88|
|             |       |acc_norm|58.96|±  |  1.01|
|boolq        |      1|acc     |68.41|±  |  0.81|
|copa         |      0|acc     |90.00|±  |  3.02|
|hellaswag    |      0|acc     |62.65|±  |  0.48|
|             |       |acc_norm|79.24|±  |  0.40|
|mc_taco      |      0|em      |11.41|   |      |
|             |       |f1      |48.36|   |      |
|openbookqa   |      0|acc     |29.40|±  |  2.04|
|             |       |acc_norm|42.00|±  |  2.21|
|piqa         |      0|acc     |80.96|±  |  0.92|
|             |       |acc_norm|80.09|±  |  0.93|
|prost        |      0|acc     |25.99|±  |  0.32|
|             |       |acc_norm|29.11|±  |  0.33|
|swag         |      0|acc     |58.61|±  |  0.35|
|             |       |acc_norm|70.36|±  |  0.32|
|winogrande   |      0|acc     |72.77|±  |  1.25|
|wsc273       |      0|acc     |86.81|±  |  2.05|

## llama-30B_gsm8k_8-shot.json
|Task |Version|Metric|Value|   |Stderr|
|-----|------:|------|----:|---|-----:|
|gsm8k|      0|acc   |30.48|±  |  1.27|

## llama-30B_human_alignment_0-shot.json
|                 Task                  |Version|       Metric        | Value |   |Stderr|
|---------------------------------------|------:|---------------------|------:|---|-----:|
|crows_pairs_english_age                |      0|likelihood_difference| 512.91|±  | 58.13|
|                                       |       |pct_stereotype       |  58.24|±  |  5.20|
|crows_pairs_english_autre              |      0|likelihood_difference|1138.07|±  |348.77|
|                                       |       |pct_stereotype       |  63.64|±  | 15.21|
|crows_pairs_english_disability         |      0|likelihood_difference| 888.65|±  |103.42|
|                                       |       |pct_stereotype       |  53.85|±  |  6.23|
|crows_pairs_english_gender             |      0|likelihood_difference| 666.15|±  | 42.85|
|                                       |       |pct_stereotype       |  54.06|±  |  2.79|
|crows_pairs_english_nationality        |      0|likelihood_difference| 587.28|±  | 39.94|
|                                       |       |pct_stereotype       |  53.24|±  |  3.40|
|crows_pairs_english_physical_appearance|      0|likelihood_difference| 540.10|±  | 59.14|
|                                       |       |pct_stereotype       |  52.78|±  |  5.92|
|crows_pairs_english_race_color         |      0|likelihood_difference| 768.21|±  | 39.14|
|                                       |       |pct_stereotype       |  56.10|±  |  2.20|
|crows_pairs_english_religion           |      0|likelihood_difference| 807.57|±  | 94.38|
|                                       |       |pct_stereotype       |  62.16|±  |  4.62|
|crows_pairs_english_sexual_orientation |      0|likelihood_difference| 754.77|±  | 76.83|
|                                       |       |pct_stereotype       |  63.44|±  |  5.02|
|crows_pairs_english_socioeconomic      |      0|likelihood_difference| 730.39|±  | 54.63|
|                                       |       |pct_stereotype       |  53.68|±  |  3.63|
|crows_pairs_french_age                 |      0|likelihood_difference| 892.50|±  |101.09|
|                                       |       |pct_stereotype       |  40.00|±  |  5.19|
|crows_pairs_french_autre               |      0|likelihood_difference| 637.98|±  |165.68|
|                                       |       |pct_stereotype       |  61.54|±  | 14.04|
|crows_pairs_french_disability          |      0|likelihood_difference|1020.27|±  |126.17|
|                                       |       |pct_stereotype       |  56.06|±  |  6.16|
|crows_pairs_french_gender              |      0|likelihood_difference|1373.28|±  |110.30|
|                                       |       |pct_stereotype       |  50.16|±  |  2.80|
|crows_pairs_french_nationality         |      0|likelihood_difference| 985.10|±  | 89.08|
|                                       |       |pct_stereotype       |  38.74|±  |  3.07|
|crows_pairs_french_physical_appearance |      0|likelihood_difference| 821.79|±  |132.68|
|                                       |       |pct_stereotype       |  56.94|±  |  5.88|
|crows_pairs_french_race_color          |      0|likelihood_difference|1061.17|±  | 76.68|
|                                       |       |pct_stereotype       |  41.74|±  |  2.30|
|crows_pairs_french_religion            |      0|likelihood_difference| 794.02|±  | 93.89|
|                                       |       |pct_stereotype       |  56.52|±  |  4.64|
|crows_pairs_french_sexual_orientation  |      0|likelihood_difference| 989.08|±  |161.13|
|                                       |       |pct_stereotype       |  71.43|±  |  4.76|
|crows_pairs_french_socioeconomic       |      0|likelihood_difference| 831.29|±  | 87.37|
|                                       |       |pct_stereotype       |  52.55|±  |  3.58|
|ethics_cm                              |      0|acc                  |  57.50|±  |  0.79|
|ethics_deontology                      |      0|acc                  |  54.17|±  |  0.83|
|                                       |       |em                   |   6.12|   |      |
|ethics_justice                         |      0|acc                  |  51.70|±  |  0.96|
|                                       |       |em                   |   1.33|   |      |
|ethics_utilitarianism                  |      0|acc                  |  50.12|±  |  0.72|
|ethics_utilitarianism_original         |      0|acc                  |  93.97|±  |  0.34|
|ethics_virtue                          |      0|acc                  |  51.82|±  |  0.71|
|                                       |       |em                   |   8.14|   |      |
|toxigen                                |      0|acc                  |  42.66|±  |  1.61|
|                                       |       |acc_norm             |  43.19|±  |  1.62|

## llama-30B_mathematical_reasoning_0-shot.json
|          Task           |Version| Metric |Value|   |Stderr|
|-------------------------|------:|--------|----:|---|-----:|
|drop                     |      1|em      | 3.83|±  |  0.20|
|                         |       |f1      |13.91|±  |  0.25|
|gsm8k                    |      0|acc     | 0.00|±  |  0.00|
|math_algebra             |      1|acc     | 2.95|±  |  0.49|
|math_asdiv               |      0|acc     | 0.00|±  |  0.00|
|math_counting_and_prob   |      1|acc     | 4.01|±  |  0.90|
|math_geometry            |      1|acc     | 1.46|±  |  0.55|
|math_intermediate_algebra|      1|acc     | 0.89|±  |  0.31|
|math_num_theory          |      1|acc     | 2.96|±  |  0.73|
|math_prealgebra          |      1|acc     | 4.13|±  |  0.67|
|math_precalc             |      1|acc     | 1.83|±  |  0.57|
|mathqa                   |      0|acc     |30.59|±  |  0.84|
|                         |       |acc_norm|30.89|±  |  0.85|

## llama-30B_mathematical_reasoning_few_shot_5-shot.json
|          Task           |Version| Metric |Value|   |Stderr|
|-------------------------|------:|--------|----:|---|-----:|
|drop                     |      1|em      | 0.84|±  |  0.09|
|                         |       |f1      | 1.65|±  |  0.10|
|gsm8k                    |      0|acc     | 0.00|±  |  0.00|
|math_algebra             |      1|acc     | 0.00|±  |  0.00|
|math_counting_and_prob   |      1|acc     | 0.00|±  |  0.00|
|math_geometry            |      1|acc     | 0.00|±  |  0.00|
|math_intermediate_algebra|      1|acc     | 0.00|±  |  0.00|
|math_num_theory          |      1|acc     | 0.00|±  |  0.00|
|math_prealgebra          |      1|acc     | 0.11|±  |  0.11|
|math_precalc             |      1|acc     | 0.00|±  |  0.00|
|mathqa                   |      0|acc     |34.74|±  |  0.87|
|                         |       |acc_norm|34.54|±  |  0.87|

## llama-30B_mmlu_5-shot.json
|                      Task                       |Version| Metric |Value|   |Stderr|
|-------------------------------------------------|------:|--------|----:|---|-----:|
|hendrycksTest-abstract_algebra                   |      0|acc     |26.00|±  |  4.41|
|                                                 |       |acc_norm|29.00|±  |  4.56|
|hendrycksTest-anatomy                            |      0|acc     |51.85|±  |  4.32|
|                                                 |       |acc_norm|40.74|±  |  4.24|
|hendrycksTest-astronomy                          |      0|acc     |57.24|±  |  4.03|
|                                                 |       |acc_norm|56.58|±  |  4.03|
|hendrycksTest-business_ethics                    |      0|acc     |67.00|±  |  4.73|
|                                                 |       |acc_norm|48.00|±  |  5.02|
|hendrycksTest-clinical_knowledge                 |      0|acc     |53.21|±  |  3.07|
|                                                 |       |acc_norm|46.42|±  |  3.07|
|hendrycksTest-college_biology                    |      0|acc     |61.11|±  |  4.08|
|                                                 |       |acc_norm|42.36|±  |  4.13|
|hendrycksTest-college_chemistry                  |      0|acc     |31.00|±  |  4.65|
|                                                 |       |acc_norm|32.00|±  |  4.69|
|hendrycksTest-college_computer_science           |      0|acc     |43.00|±  |  4.98|
|                                                 |       |acc_norm|34.00|±  |  4.76|
|hendrycksTest-college_mathematics                |      0|acc     |37.00|±  |  4.85|
|                                                 |       |acc_norm|30.00|±  |  4.61|
|hendrycksTest-college_medicine                   |      0|acc     |51.45|±  |  3.81|
|                                                 |       |acc_norm|43.35|±  |  3.78|
|hendrycksTest-college_physics                    |      0|acc     |23.53|±  |  4.22|
|                                                 |       |acc_norm|29.41|±  |  4.53|
|hendrycksTest-computer_security                  |      0|acc     |66.00|±  |  4.76|
|                                                 |       |acc_norm|58.00|±  |  4.96|
|hendrycksTest-conceptual_physics                 |      0|acc     |51.06|±  |  3.27|
|                                                 |       |acc_norm|32.77|±  |  3.07|
|hendrycksTest-econometrics                       |      0|acc     |35.09|±  |  4.49|
|                                                 |       |acc_norm|31.58|±  |  4.37|
|hendrycksTest-electrical_engineering             |      0|acc     |51.72|±  |  4.16|
|                                                 |       |acc_norm|38.62|±  |  4.06|
|hendrycksTest-elementary_mathematics             |      0|acc     |44.18|±  |  2.56|
|                                                 |       |acc_norm|37.04|±  |  2.49|
|hendrycksTest-formal_logic                       |      0|acc     |42.06|±  |  4.42|
|                                                 |       |acc_norm|39.68|±  |  4.38|
|hendrycksTest-global_facts                       |      0|acc     |47.00|±  |  5.02|
|                                                 |       |acc_norm|37.00|±  |  4.85|
|hendrycksTest-high_school_biology                |      0|acc     |67.10|±  |  2.67|
|                                                 |       |acc_norm|54.52|±  |  2.83|
|hendrycksTest-high_school_chemistry              |      0|acc     |39.90|±  |  3.45|
|                                                 |       |acc_norm|36.95|±  |  3.40|
|hendrycksTest-high_school_computer_science       |      0|acc     |61.00|±  |  4.90|
|                                                 |       |acc_norm|47.00|±  |  5.02|
|hendrycksTest-high_school_european_history       |      0|acc     |69.70|±  |  3.59|
|                                                 |       |acc_norm|56.36|±  |  3.87|
|hendrycksTest-high_school_geography              |      0|acc     |75.76|±  |  3.05|
|                                                 |       |acc_norm|55.05|±  |  3.54|
|hendrycksTest-high_school_government_and_politics|      0|acc     |80.83|±  |  2.84|
|                                                 |       |acc_norm|61.14|±  |  3.52|
|hendrycksTest-high_school_macroeconomics         |      0|acc     |51.54|±  |  2.53|
|                                                 |       |acc_norm|41.54|±  |  2.50|
|hendrycksTest-high_school_mathematics            |      0|acc     |25.93|±  |  2.67|
|                                                 |       |acc_norm|31.48|±  |  2.83|
|hendrycksTest-high_school_microeconomics         |      0|acc     |58.40|±  |  3.20|
|                                                 |       |acc_norm|48.32|±  |  3.25|
|hendrycksTest-high_school_physics                |      0|acc     |31.79|±  |  3.80|
|                                                 |       |acc_norm|31.13|±  |  3.78|
|hendrycksTest-high_school_psychology             |      0|acc     |77.06|±  |  1.80|
|                                                 |       |acc_norm|55.41|±  |  2.13|
|hendrycksTest-high_school_statistics             |      0|acc     |43.52|±  |  3.38|
|                                                 |       |acc_norm|35.65|±  |  3.27|
|hendrycksTest-high_school_us_history             |      0|acc     |72.06|±  |  3.15|
|                                                 |       |acc_norm|55.39|±  |  3.49|
|hendrycksTest-high_school_world_history          |      0|acc     |69.62|±  |  2.99|
|                                                 |       |acc_norm|56.96|±  |  3.22|
|hendrycksTest-human_aging                        |      0|acc     |67.26|±  |  3.15|
|                                                 |       |acc_norm|36.32|±  |  3.23|
|hendrycksTest-human_sexuality                    |      0|acc     |70.23|±  |  4.01|
|                                                 |       |acc_norm|46.56|±  |  4.37|
|hendrycksTest-international_law                  |      0|acc     |70.25|±  |  4.17|
|                                                 |       |acc_norm|76.86|±  |  3.85|
|hendrycksTest-jurisprudence                      |      0|acc     |66.67|±  |  4.56|
|                                                 |       |acc_norm|55.56|±  |  4.80|
|hendrycksTest-logical_fallacies                  |      0|acc     |69.94|±  |  3.60|
|                                                 |       |acc_norm|53.99|±  |  3.92|
|hendrycksTest-machine_learning                   |      0|acc     |40.18|±  |  4.65|
|                                                 |       |acc_norm|30.36|±  |  4.36|
|hendrycksTest-management                         |      0|acc     |71.84|±  |  4.45|
|                                                 |       |acc_norm|55.34|±  |  4.92|
|hendrycksTest-marketing                          |      0|acc     |84.62|±  |  2.36|
|                                                 |       |acc_norm|76.50|±  |  2.78|
|hendrycksTest-medical_genetics                   |      0|acc     |60.00|±  |  4.92|
|                                                 |       |acc_norm|54.00|±  |  5.01|
|hendrycksTest-miscellaneous                      |      0|acc     |81.86|±  |  1.38|
|                                                 |       |acc_norm|61.43|±  |  1.74|
|hendrycksTest-moral_disputes                     |      0|acc     |61.85|±  |  2.62|
|                                                 |       |acc_norm|45.95|±  |  2.68|
|hendrycksTest-moral_scenarios                    |      0|acc     |34.30|±  |  1.59|
|                                                 |       |acc_norm|27.26|±  |  1.49|
|hendrycksTest-nutrition                          |      0|acc     |61.11|±  |  2.79|
|                                                 |       |acc_norm|50.33|±  |  2.86|
|hendrycksTest-philosophy                         |      0|acc     |67.52|±  |  2.66|
|                                                 |       |acc_norm|50.16|±  |  2.84|
|hendrycksTest-prehistory                         |      0|acc     |66.36|±  |  2.63|
|                                                 |       |acc_norm|42.90|±  |  2.75|
|hendrycksTest-professional_accounting            |      0|acc     |39.72|±  |  2.92|
|                                                 |       |acc_norm|33.69|±  |  2.82|
|hendrycksTest-professional_law                   |      0|acc     |40.03|±  |  1.25|
|                                                 |       |acc_norm|34.35|±  |  1.21|
|hendrycksTest-professional_medicine              |      0|acc     |55.51|±  |  3.02|
|                                                 |       |acc_norm|35.66|±  |  2.91|
|hendrycksTest-professional_psychology            |      0|acc     |58.82|±  |  1.99|
|                                                 |       |acc_norm|43.30|±  |  2.00|
|hendrycksTest-public_relations                   |      0|acc     |64.55|±  |  4.58|
|                                                 |       |acc_norm|40.91|±  |  4.71|
|hendrycksTest-security_studies                   |      0|acc     |57.14|±  |  3.17|
|                                                 |       |acc_norm|40.41|±  |  3.14|
|hendrycksTest-sociology                          |      0|acc     |76.12|±  |  3.01|
|                                                 |       |acc_norm|66.17|±  |  3.35|
|hendrycksTest-us_foreign_policy                  |      0|acc     |79.00|±  |  4.09|
|                                                 |       |acc_norm|59.00|±  |  4.94|
|hendrycksTest-virology                           |      0|acc     |49.40|±  |  3.89|
|                                                 |       |acc_norm|34.34|±  |  3.70|
|hendrycksTest-world_religions                    |      0|acc     |81.29|±  |  2.99|
|                                                 |       |acc_norm|76.61|±  |  3.25|

## llama-30B_pawsx_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|pawsx_de|      0|acc   |58.20|±  |  1.10|
|pawsx_en|      0|acc   |58.75|±  |  1.10|
|pawsx_es|      0|acc   |55.80|±  |  1.11|
|pawsx_fr|      0|acc   |52.85|±  |  1.12|
|pawsx_ja|      0|acc   |46.75|±  |  1.12|
|pawsx_ko|      0|acc   |45.70|±  |  1.11|
|pawsx_zh|      0|acc   |45.90|±  |  1.11|

## llama-30B_xcopa_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|xcopa_et|      0|acc   | 47.2|±  |  2.23|
|xcopa_ht|      0|acc   | 51.8|±  |  2.24|
|xcopa_id|      0|acc   | 60.6|±  |  2.19|
|xcopa_it|      0|acc   | 71.4|±  |  2.02|
|xcopa_qu|      0|acc   | 49.4|±  |  2.24|
|xcopa_sw|      0|acc   | 52.4|±  |  2.24|
|xcopa_ta|      0|acc   | 53.2|±  |  2.23|
|xcopa_th|      0|acc   | 54.6|±  |  2.23|
|xcopa_tr|      0|acc   | 52.2|±  |  2.24|
|xcopa_vi|      0|acc   | 52.4|±  |  2.24|
|xcopa_zh|      0|acc   | 62.2|±  |  2.17|

## llama-30B_xnli_0-shot.json
| Task  |Version|Metric|Value|   |Stderr|
|-------|------:|------|----:|---|-----:|
|xnli_ar|      0|acc   |34.49|±  |  0.67|
|xnli_bg|      0|acc   |38.52|±  |  0.69|
|xnli_de|      0|acc   |43.87|±  |  0.70|
|xnli_el|      0|acc   |34.91|±  |  0.67|
|xnli_en|      0|acc   |48.18|±  |  0.71|
|xnli_es|      0|acc   |40.24|±  |  0.69|
|xnli_fr|      0|acc   |42.95|±  |  0.70|
|xnli_hi|      0|acc   |36.47|±  |  0.68|
|xnli_ru|      0|acc   |38.12|±  |  0.69|
|xnli_sw|      0|acc   |34.09|±  |  0.67|
|xnli_th|      0|acc   |33.97|±  |  0.67|
|xnli_tr|      0|acc   |36.53|±  |  0.68|
|xnli_ur|      0|acc   |34.31|±  |  0.67|
|xnli_vi|      0|acc   |35.67|±  |  0.68|
|xnli_zh|      0|acc   |33.51|±  |  0.67|

## llama-30B_xstory_cloze_0-shot.json
|     Task      |Version|Metric|Value|   |Stderr|
|---------------|------:|------|----:|---|-----:|
|xstory_cloze_ar|      0|acc   |50.89|±  |  1.29|
|xstory_cloze_en|      0|acc   |78.16|±  |  1.06|
|xstory_cloze_es|      0|acc   |70.81|±  |  1.17|
|xstory_cloze_eu|      0|acc   |51.36|±  |  1.29|
|xstory_cloze_hi|      0|acc   |56.65|±  |  1.28|
|xstory_cloze_id|      0|acc   |59.23|±  |  1.26|
|xstory_cloze_my|      0|acc   |48.78|±  |  1.29|
|xstory_cloze_ru|      0|acc   |66.71|±  |  1.21|
|xstory_cloze_sw|      0|acc   |50.63|±  |  1.29|
|xstory_cloze_te|      0|acc   |53.21|±  |  1.28|
|xstory_cloze_zh|      0|acc   |58.57|±  |  1.27|

## llama-30B_xwinograd_0-shot.json
|    Task    |Version|Metric|Value|   |Stderr|
|------------|------:|------|----:|---|-----:|
|xwinograd_en|      0|acc   |87.40|±  |  0.69|
|xwinograd_fr|      0|acc   |73.49|±  |  4.87|
|xwinograd_jp|      0|acc   |67.36|±  |  1.51|
|xwinograd_pt|      0|acc   |76.81|±  |  2.61|
|xwinograd_ru|      0|acc   |66.98|±  |  2.65|
|xwinograd_zh|      0|acc   |71.23|±  |  2.02|
