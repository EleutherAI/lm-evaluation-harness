# mpt-7b

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
