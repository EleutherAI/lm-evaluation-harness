# bloom-7b1

## bloom-7b1_bbh_3-shot.json
|                      Task                      |Version|       Metric        |Value|   |Stderr|
|------------------------------------------------|------:|---------------------|----:|---|-----:|
|bigbench_causal_judgement                       |      0|multiple_choice_grade|52.11|±  |  3.63|
|bigbench_date_understanding                     |      0|multiple_choice_grade|36.59|±  |  2.51|
|bigbench_disambiguation_qa                      |      0|multiple_choice_grade|26.36|±  |  2.75|
|bigbench_dyck_languages                         |      0|multiple_choice_grade|14.40|±  |  1.11|
|bigbench_formal_fallacies_syllogisms_negation   |      0|multiple_choice_grade|50.06|±  |  0.42|
|bigbench_geometric_shapes                       |      0|multiple_choice_grade|20.06|±  |  2.12|
|                                                |       |exact_str_match      | 0.00|±  |  0.00|
|bigbench_hyperbaton                             |      0|multiple_choice_grade|48.62|±  |  0.22|
|bigbench_logical_deduction_five_objects         |      0|multiple_choice_grade|26.00|±  |  1.96|
|bigbench_logical_deduction_seven_objects        |      0|multiple_choice_grade|19.14|±  |  1.49|
|bigbench_logical_deduction_three_objects        |      0|multiple_choice_grade|37.00|±  |  2.79|
|bigbench_movie_recommendation                   |      0|multiple_choice_grade|26.40|±  |  1.97|
|bigbench_navigate                               |      0|multiple_choice_grade|49.90|±  |  1.58|
|bigbench_reasoning_about_colored_objects        |      0|multiple_choice_grade|24.85|±  |  0.97|
|bigbench_ruin_names                             |      0|multiple_choice_grade|34.38|±  |  2.25|
|bigbench_salient_translation_error_detection    |      0|multiple_choice_grade|19.14|±  |  1.25|
|bigbench_snarks                                 |      0|multiple_choice_grade|49.72|±  |  3.73|
|bigbench_sports_understanding                   |      0|multiple_choice_grade|50.30|±  |  1.59|
|bigbench_temporal_sequences                     |      0|multiple_choice_grade|24.80|±  |  1.37|
|bigbench_tracking_shuffled_objects_five_objects |      0|multiple_choice_grade|18.40|±  |  1.10|
|bigbench_tracking_shuffled_objects_seven_objects|      0|multiple_choice_grade|14.00|±  |  0.83|
|bigbench_tracking_shuffled_objects_three_objects|      0|multiple_choice_grade|37.00|±  |  2.79|

## bloom-7b1_common_sense_reasoning_0-shot.json
|    Task     |Version| Metric |Value|   |Stderr|
|-------------|------:|--------|----:|---|-----:|
|arc_challenge|      0|acc     |30.38|±  |  1.34|
|             |       |acc_norm|33.53|±  |  1.38|
|arc_easy     |      0|acc     |64.94|±  |  0.98|
|             |       |acc_norm|57.32|±  |  1.01|
|boolq        |      1|acc     |62.87|±  |  0.85|
|copa         |      0|acc     |72.00|±  |  4.51|
|hellaswag    |      0|acc     |46.24|±  |  0.50|
|             |       |acc_norm|59.68|±  |  0.49|
|mc_taco      |      0|em      |13.59|   |      |
|             |       |f1      |50.53|   |      |
|openbookqa   |      0|acc     |25.20|±  |  1.94|
|             |       |acc_norm|35.80|±  |  2.15|
|piqa         |      0|acc     |72.74|±  |  1.04|
|             |       |acc_norm|73.67|±  |  1.03|
|prost        |      0|acc     |26.18|±  |  0.32|
|             |       |acc_norm|30.57|±  |  0.34|
|swag         |      0|acc     |50.25|±  |  0.35|
|             |       |acc_norm|68.26|±  |  0.33|
|winogrande   |      0|acc     |64.33|±  |  1.35|
|wsc273       |      0|acc     |81.32|±  |  2.36|

## bloom-7b1_gsm8k_8-shot.json
|Task |Version|Metric|Value|   |Stderr|
|-----|------:|------|----:|---|-----:|
|gsm8k|      0|acc   |  1.9|±  |  0.38|

## bloom-7b1_mathematical_reasoning_few_shot_5-shot.json
|          Task           |Version| Metric |Value|   |Stderr|
|-------------------------|------:|--------|----:|---|-----:|
|drop                     |      1|em      | 2.51|±  |  0.16|
|                         |       |f1      | 5.09|±  |  0.18|
|gsm8k                    |      0|acc     | 0.00|±  |  0.00|
|math_algebra             |      1|acc     | 0.00|±  |  0.00|
|math_counting_and_prob   |      1|acc     | 0.00|±  |  0.00|
|math_geometry            |      1|acc     | 0.00|±  |  0.00|
|math_intermediate_algebra|      1|acc     | 0.00|±  |  0.00|
|math_num_theory          |      1|acc     | 0.00|±  |  0.00|
|math_prealgebra          |      1|acc     | 0.00|±  |  0.00|
|math_precalc             |      1|acc     | 0.00|±  |  0.00|
|mathqa                   |      0|acc     |26.57|±  |  0.81|
|                         |       |acc_norm|26.53|±  |  0.81|

## bloom-7b1_pawsx_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|pawsx_de|      0|acc   |52.85|±  |  1.12|
|pawsx_en|      0|acc   |61.30|±  |  1.09|
|pawsx_es|      0|acc   |59.35|±  |  1.10|
|pawsx_fr|      0|acc   |50.90|±  |  1.12|
|pawsx_ja|      0|acc   |45.45|±  |  1.11|
|pawsx_ko|      0|acc   |45.10|±  |  1.11|
|pawsx_zh|      0|acc   |47.35|±  |  1.12|

## bloom-7b1_question_answering_0-shot.json
|    Task     |Version|   Metric   |Value|   |Stderr|
|-------------|------:|------------|----:|---|-----:|
|headqa_en    |      0|acc         |31.18|±  |  0.88|
|             |       |acc_norm    |35.56|±  |  0.91|
|headqa_es    |      0|acc         |29.54|±  |  0.87|
|             |       |acc_norm    |34.32|±  |  0.91|
|logiqa       |      0|acc         |20.28|±  |  1.58|
|             |       |acc_norm    |28.11|±  |  1.76|
|squad2       |      1|exact       | 7.82|   |      |
|             |       |f1          |12.64|   |      |
|             |       |HasAns_exact|14.84|   |      |
|             |       |HasAns_f1   |24.51|   |      |
|             |       |NoAns_exact | 0.81|   |      |
|             |       |NoAns_f1    | 0.81|   |      |
|             |       |best_exact  |50.07|   |      |
|             |       |best_f1     |50.07|   |      |
|triviaqa     |      1|acc         | 5.52|±  |  0.21|
|truthfulqa_mc|      1|mc1         |22.40|±  |  1.46|
|             |       |mc2         |38.90|±  |  1.40|
|webqs        |      0|acc         | 2.26|±  |  0.33|

## bloom-7b1_reading_comprehension_0-shot.json
|Task|Version|Metric|Value|   |Stderr|
|----|------:|------|----:|---|-----:|
|coqa|      1|f1    |68.83|±  |  1.63|
|    |       |em    |53.87|±  |  2.00|
|drop|      1|em    | 2.57|±  |  0.16|
|    |       |f1    | 9.85|±  |  0.21|
|race|      1|acc   |36.56|±  |  1.49|

## bloom-7b1_xcopa_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|xcopa_et|      0|acc   | 48.2|±  |  2.24|
|xcopa_ht|      0|acc   | 50.8|±  |  2.24|
|xcopa_id|      0|acc   | 69.8|±  |  2.06|
|xcopa_it|      0|acc   | 52.8|±  |  2.23|
|xcopa_qu|      0|acc   | 50.8|±  |  2.24|
|xcopa_sw|      0|acc   | 51.6|±  |  2.24|
|xcopa_ta|      0|acc   | 59.2|±  |  2.20|
|xcopa_th|      0|acc   | 55.4|±  |  2.23|
|xcopa_tr|      0|acc   | 51.2|±  |  2.24|
|xcopa_vi|      0|acc   | 70.8|±  |  2.04|
|xcopa_zh|      0|acc   | 65.2|±  |  2.13|

## bloom-7b1_xnli_0-shot.json
| Task  |Version|Metric|Value|   |Stderr|
|-------|------:|------|----:|---|-----:|
|xnli_ar|      0|acc   |33.83|±  |  0.67|
|xnli_bg|      0|acc   |39.70|±  |  0.69|
|xnli_de|      0|acc   |39.86|±  |  0.69|
|xnli_el|      0|acc   |35.75|±  |  0.68|
|xnli_en|      0|acc   |53.91|±  |  0.70|
|xnli_es|      0|acc   |48.70|±  |  0.71|
|xnli_fr|      0|acc   |49.68|±  |  0.71|
|xnli_hi|      0|acc   |46.51|±  |  0.70|
|xnli_ru|      0|acc   |43.05|±  |  0.70|
|xnli_sw|      0|acc   |37.92|±  |  0.69|
|xnli_th|      0|acc   |34.99|±  |  0.67|
|xnli_tr|      0|acc   |35.09|±  |  0.67|
|xnli_ur|      0|acc   |42.10|±  |  0.70|
|xnli_vi|      0|acc   |47.05|±  |  0.71|
|xnli_zh|      0|acc   |35.43|±  |  0.68|

## bloom-7b1_xstory_cloze_0-shot.json
|     Task      |Version|Metric|Value|   |Stderr|
|---------------|------:|------|----:|---|-----:|
|xstory_cloze_ar|      0|acc   |58.57|±  |  1.27|
|xstory_cloze_en|      0|acc   |70.75|±  |  1.17|
|xstory_cloze_es|      0|acc   |66.12|±  |  1.22|
|xstory_cloze_eu|      0|acc   |57.18|±  |  1.27|
|xstory_cloze_hi|      0|acc   |60.56|±  |  1.26|
|xstory_cloze_id|      0|acc   |64.46|±  |  1.23|
|xstory_cloze_my|      0|acc   |48.97|±  |  1.29|
|xstory_cloze_ru|      0|acc   |52.75|±  |  1.28|
|xstory_cloze_sw|      0|acc   |53.94|±  |  1.28|
|xstory_cloze_te|      0|acc   |57.45|±  |  1.27|
|xstory_cloze_zh|      0|acc   |61.88|±  |  1.25|

## bloom-7b1_xwinograd_0-shot.json
|    Task    |Version|Metric|Value|   |Stderr|
|------------|------:|------|----:|---|-----:|
|xwinograd_en|      0|acc   |82.15|±  |  0.79|
|xwinograd_fr|      0|acc   |71.08|±  |  5.01|
|xwinograd_jp|      0|acc   |58.50|±  |  1.59|
|xwinograd_pt|      0|acc   |76.81|±  |  2.61|
|xwinograd_ru|      0|acc   |56.83|±  |  2.80|
|xwinograd_zh|      0|acc   |74.40|±  |  1.95|
