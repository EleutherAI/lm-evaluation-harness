# bloom-1b1

## bloom-1b1_common_sense_reasoning_0-shot.json
|    Task     |Version| Metric |Value|   |Stderr|
|-------------|------:|--------|----:|---|-----:|
|arc_challenge|      0|acc     |23.63|±  |  1.24|
|             |       |acc_norm|25.68|±  |  1.28|
|arc_easy     |      0|acc     |51.47|±  |  1.03|
|             |       |acc_norm|45.45|±  |  1.02|
|boolq        |      1|acc     |59.08|±  |  0.86|
|copa         |      0|acc     |68.00|±  |  4.69|
|hellaswag    |      0|acc     |34.63|±  |  0.47|
|             |       |acc_norm|41.77|±  |  0.49|
|mc_taco      |      0|em      |14.49|   |      |
|             |       |f1      |32.43|   |      |
|openbookqa   |      0|acc     |19.60|±  |  1.78|
|             |       |acc_norm|29.40|±  |  2.04|
|piqa         |      0|acc     |67.14|±  |  1.10|
|             |       |acc_norm|67.14|±  |  1.10|
|prost        |      0|acc     |23.41|±  |  0.31|
|             |       |acc_norm|30.50|±  |  0.34|
|swag         |      0|acc     |43.43|±  |  0.35|
|             |       |acc_norm|58.28|±  |  0.35|
|winogrande   |      0|acc     |54.93|±  |  1.40|
|wsc273       |      0|acc     |68.50|±  |  2.82|

## bloom-1b1_gsm8k_8-shot.json
|Task |Version|Metric|Value|   |Stderr|
|-----|------:|------|----:|---|-----:|
|gsm8k|      0|acc   | 0.83|±  |  0.25|

## bloom-1b1_mathematical_reasoning_few_shot_5-shot.json
|          Task           |Version| Metric |Value|   |Stderr|
|-------------------------|------:|--------|----:|---|-----:|
|drop                     |      1|em      | 1.38|±  |  0.12|
|                         |       |f1      | 4.01|±  |  0.15|
|gsm8k                    |      0|acc     | 0.00|±  |  0.00|
|math_algebra             |      1|acc     | 0.00|±  |  0.00|
|math_counting_and_prob   |      1|acc     | 0.21|±  |  0.21|
|math_geometry            |      1|acc     | 0.21|±  |  0.21|
|math_intermediate_algebra|      1|acc     | 0.00|±  |  0.00|
|math_num_theory          |      1|acc     | 0.19|±  |  0.19|
|math_prealgebra          |      1|acc     | 0.11|±  |  0.11|
|math_precalc             |      1|acc     | 0.00|±  |  0.00|
|mathqa                   |      0|acc     |23.55|±  |  0.78|
|                         |       |acc_norm|23.62|±  |  0.78|

## bloom-1b1_pawsx_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|pawsx_de|      0|acc   |46.95|±  |  1.12|
|pawsx_en|      0|acc   |52.45|±  |  1.12|
|pawsx_es|      0|acc   |51.50|±  |  1.12|
|pawsx_fr|      0|acc   |46.15|±  |  1.11|
|pawsx_ja|      0|acc   |48.40|±  |  1.12|
|pawsx_ko|      0|acc   |49.90|±  |  1.12|
|pawsx_zh|      0|acc   |48.95|±  |  1.12|

## bloom-1b1_question_answering_0-shot.json
|    Task     |Version|   Metric   |Value|   |Stderr|
|-------------|------:|------------|----:|---|-----:|
|headqa_en    |      0|acc         |26.44|±  |  0.84|
|             |       |acc_norm    |30.49|±  |  0.88|
|headqa_es    |      0|acc         |24.43|±  |  0.82|
|             |       |acc_norm    |28.30|±  |  0.86|
|logiqa       |      0|acc         |18.89|±  |  1.54|
|             |       |acc_norm    |25.65|±  |  1.71|
|squad2       |      1|exact       | 4.17|   |      |
|             |       |f1          | 6.60|   |      |
|             |       |HasAns_exact| 2.19|   |      |
|             |       |HasAns_f1   | 7.05|   |      |
|             |       |NoAns_exact | 6.14|   |      |
|             |       |NoAns_f1    | 6.14|   |      |
|             |       |best_exact  |50.07|   |      |
|             |       |best_f1     |50.07|   |      |
|triviaqa     |      1|acc         | 2.68|±  |  0.15|
|truthfulqa_mc|      1|mc1         |25.34|±  |  1.52|
|             |       |mc2         |41.80|±  |  1.46|
|webqs        |      0|acc         | 1.38|±  |  0.26|

## bloom-1b1_reading_comprehension_0-shot.json
|Task|Version|Metric|Value|   |Stderr|
|----|------:|------|----:|---|-----:|
|coqa|      1|f1    |45.57|±  |  1.88|
|    |       |em    |32.98|±  |  1.95|
|drop|      1|em    | 3.31|±  |  0.18|
|    |       |f1    | 8.63|±  |  0.22|
|race|      1|acc   |32.63|±  |  1.45|

## bloom-1b1_xcopa_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|xcopa_et|      0|acc   | 50.6|±  |  2.24|
|xcopa_ht|      0|acc   | 53.0|±  |  2.23|
|xcopa_id|      0|acc   | 64.8|±  |  2.14|
|xcopa_it|      0|acc   | 50.8|±  |  2.24|
|xcopa_qu|      0|acc   | 51.2|±  |  2.24|
|xcopa_sw|      0|acc   | 54.4|±  |  2.23|
|xcopa_ta|      0|acc   | 57.0|±  |  2.22|
|xcopa_th|      0|acc   | 53.2|±  |  2.23|
|xcopa_tr|      0|acc   | 53.0|±  |  2.23|
|xcopa_vi|      0|acc   | 62.4|±  |  2.17|
|xcopa_zh|      0|acc   | 59.4|±  |  2.20|

## bloom-1b1_xnli_0-shot.json
| Task  |Version|Metric|Value|   |Stderr|
|-------|------:|------|----:|---|-----:|
|xnli_ar|      0|acc   |33.93|±  |  0.67|
|xnli_bg|      0|acc   |34.13|±  |  0.67|
|xnli_de|      0|acc   |39.64|±  |  0.69|
|xnli_el|      0|acc   |34.03|±  |  0.67|
|xnli_en|      0|acc   |51.48|±  |  0.71|
|xnli_es|      0|acc   |47.98|±  |  0.71|
|xnli_fr|      0|acc   |47.15|±  |  0.71|
|xnli_hi|      0|acc   |42.32|±  |  0.70|
|xnli_ru|      0|acc   |40.46|±  |  0.69|
|xnli_sw|      0|acc   |35.29|±  |  0.68|
|xnli_th|      0|acc   |33.75|±  |  0.67|
|xnli_tr|      0|acc   |34.79|±  |  0.67|
|xnli_ur|      0|acc   |37.33|±  |  0.68|
|xnli_vi|      0|acc   |44.45|±  |  0.70|
|xnli_zh|      0|acc   |36.23|±  |  0.68|

## bloom-1b1_xstory_cloze_0-shot.json
|     Task      |Version|Metric|Value|   |Stderr|
|---------------|------:|------|----:|---|-----:|
|xstory_cloze_ar|      0|acc   |52.88|±  |  1.28|
|xstory_cloze_en|      0|acc   |62.54|±  |  1.25|
|xstory_cloze_es|      0|acc   |58.31|±  |  1.27|
|xstory_cloze_eu|      0|acc   |54.33|±  |  1.28|
|xstory_cloze_hi|      0|acc   |55.53|±  |  1.28|
|xstory_cloze_id|      0|acc   |57.91|±  |  1.27|
|xstory_cloze_my|      0|acc   |46.19|±  |  1.28|
|xstory_cloze_ru|      0|acc   |48.25|±  |  1.29|
|xstory_cloze_sw|      0|acc   |50.56|±  |  1.29|
|xstory_cloze_te|      0|acc   |56.39|±  |  1.28|
|xstory_cloze_zh|      0|acc   |58.04|±  |  1.27|

## bloom-1b1_xwinograd_0-shot.json
|    Task    |Version|Metric|Value|   |Stderr|
|------------|------:|------|----:|---|-----:|
|xwinograd_en|      0|acc   |69.98|±  |  0.95|
|xwinograd_fr|      0|acc   |66.27|±  |  5.22|
|xwinograd_jp|      0|acc   |52.87|±  |  1.61|
|xwinograd_pt|      0|acc   |63.12|±  |  2.98|
|xwinograd_ru|      0|acc   |54.29|±  |  2.81|
|xwinograd_zh|      0|acc   |69.25|±  |  2.06|
