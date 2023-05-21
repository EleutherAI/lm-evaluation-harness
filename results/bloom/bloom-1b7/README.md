# bloom-1b7

## bloom-1b7_common_sense_reasoning_0-shot.json
|    Task     |Version| Metric |Value|   |Stderr|
|-------------|------:|--------|----:|---|-----:|
|arc_challenge|      0|acc     |23.55|±  |  1.24|
|             |       |acc_norm|26.79|±  |  1.29|
|arc_easy     |      0|acc     |56.31|±  |  1.02|
|             |       |acc_norm|48.11|±  |  1.03|
|boolq        |      1|acc     |61.77|±  |  0.85|
|copa         |      0|acc     |70.00|±  |  4.61|
|hellaswag    |      0|acc     |37.62|±  |  0.48|
|             |       |acc_norm|46.56|±  |  0.50|
|mc_taco      |      0|em      |12.54|   |      |
|             |       |f1      |47.46|   |      |
|openbookqa   |      0|acc     |21.40|±  |  1.84|
|             |       |acc_norm|30.00|±  |  2.05|
|piqa         |      0|acc     |68.77|±  |  1.08|
|             |       |acc_norm|70.08|±  |  1.07|
|prost        |      0|acc     |23.52|±  |  0.31|
|             |       |acc_norm|26.70|±  |  0.32|
|swag         |      0|acc     |45.32|±  |  0.35|
|             |       |acc_norm|61.15|±  |  0.34|
|winogrande   |      0|acc     |57.14|±  |  1.39|
|wsc273       |      0|acc     |72.89|±  |  2.70|

## bloom-1b7_gsm8k_8-shot.json
|Task |Version|Metric|Value|   |Stderr|
|-----|------:|------|----:|---|-----:|
|gsm8k|      0|acc   | 1.29|±  |  0.31|

## bloom-1b7_mathematical_reasoning_few_shot_5-shot.json
|          Task           |Version| Metric |Value|   |Stderr|
|-------------------------|------:|--------|----:|---|-----:|
|drop                     |      1|em      | 1.49|±  |  0.12|
|                         |       |f1      | 4.31|±  |  0.15|
|gsm8k                    |      0|acc     | 0.00|±  |  0.00|
|math_algebra             |      1|acc     | 0.00|±  |  0.00|
|math_counting_and_prob   |      1|acc     | 0.00|±  |  0.00|
|math_geometry            |      1|acc     | 0.00|±  |  0.00|
|math_intermediate_algebra|      1|acc     | 0.00|±  |  0.00|
|math_num_theory          |      1|acc     | 0.74|±  |  0.37|
|math_prealgebra          |      1|acc     | 0.23|±  |  0.16|
|math_precalc             |      1|acc     | 0.00|±  |  0.00|
|mathqa                   |      0|acc     |24.29|±  |  0.79|
|                         |       |acc_norm|24.62|±  |  0.79|

## bloom-1b7_pawsx_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|pawsx_de|      0|acc   |48.75|±  |  1.12|
|pawsx_en|      0|acc   |48.90|±  |  1.12|
|pawsx_es|      0|acc   |51.30|±  |  1.12|
|pawsx_fr|      0|acc   |46.20|±  |  1.12|
|pawsx_ja|      0|acc   |44.70|±  |  1.11|
|pawsx_ko|      0|acc   |45.80|±  |  1.11|
|pawsx_zh|      0|acc   |45.40|±  |  1.11|

## bloom-1b7_question_answering_0-shot.json
|    Task     |Version|   Metric   |Value|   |Stderr|
|-------------|------:|------------|----:|---|-----:|
|headqa_en    |      0|acc         |27.75|±  |  0.86|
|             |       |acc_norm    |32.57|±  |  0.90|
|headqa_es    |      0|acc         |25.42|±  |  0.83|
|             |       |acc_norm    |29.58|±  |  0.87|
|logiqa       |      0|acc         |21.66|±  |  1.62|
|             |       |acc_norm    |28.11|±  |  1.76|
|squad2       |      1|exact       | 1.80|   |      |
|             |       |f1          | 4.38|   |      |
|             |       |HasAns_exact| 2.40|   |      |
|             |       |HasAns_f1   | 7.56|   |      |
|             |       |NoAns_exact | 1.21|   |      |
|             |       |NoAns_f1    | 1.21|   |      |
|             |       |best_exact  |50.07|   |      |
|             |       |best_f1     |50.07|   |      |
|triviaqa     |      1|acc         | 3.14|±  |  0.16|
|truthfulqa_mc|      1|mc1         |24.48|±  |  1.51|
|             |       |mc2         |41.32|±  |  1.44|
|webqs        |      0|acc         | 1.28|±  |  0.25|

## bloom-1b7_reading_comprehension_0-shot.json
|Task|Version|Metric|Value|   |Stderr|
|----|------:|------|----:|---|-----:|
|coqa|      1|f1    |53.55|±  |  1.89|
|    |       |em    |40.90|±  |  2.03|
|drop|      1|em    | 0.69|±  |  0.08|
|    |       |f1    | 6.89|±  |  0.16|
|race|      1|acc   |33.21|±  |  1.46|

## bloom-1b7_xcopa_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|xcopa_et|      0|acc   | 47.4|±  |  2.24|
|xcopa_ht|      0|acc   | 50.4|±  |  2.24|
|xcopa_id|      0|acc   | 63.2|±  |  2.16|
|xcopa_it|      0|acc   | 52.6|±  |  2.24|
|xcopa_qu|      0|acc   | 50.6|±  |  2.24|
|xcopa_sw|      0|acc   | 51.8|±  |  2.24|
|xcopa_ta|      0|acc   | 56.6|±  |  2.22|
|xcopa_th|      0|acc   | 53.2|±  |  2.23|
|xcopa_tr|      0|acc   | 52.8|±  |  2.23|
|xcopa_vi|      0|acc   | 65.8|±  |  2.12|
|xcopa_zh|      0|acc   | 61.4|±  |  2.18|

## bloom-1b7_xnli_0-shot.json
| Task  |Version|Metric|Value|   |Stderr|
|-------|------:|------|----:|---|-----:|
|xnli_ar|      0|acc   |33.57|±  |  0.67|
|xnli_bg|      0|acc   |35.43|±  |  0.68|
|xnli_de|      0|acc   |40.58|±  |  0.69|
|xnli_el|      0|acc   |33.99|±  |  0.67|
|xnli_en|      0|acc   |50.14|±  |  0.71|
|xnli_es|      0|acc   |47.82|±  |  0.71|
|xnli_fr|      0|acc   |48.18|±  |  0.71|
|xnli_hi|      0|acc   |43.95|±  |  0.70|
|xnli_ru|      0|acc   |39.32|±  |  0.69|
|xnli_sw|      0|acc   |34.51|±  |  0.67|
|xnli_th|      0|acc   |33.37|±  |  0.67|
|xnli_tr|      0|acc   |34.93|±  |  0.67|
|xnli_ur|      0|acc   |40.50|±  |  0.69|
|xnli_vi|      0|acc   |46.23|±  |  0.70|
|xnli_zh|      0|acc   |36.21|±  |  0.68|

## bloom-1b7_xstory_cloze_0-shot.json
|     Task      |Version|Metric|Value|   |Stderr|
|---------------|------:|------|----:|---|-----:|
|xstory_cloze_ar|      0|acc   |55.00|±  |  1.28|
|xstory_cloze_en|      0|acc   |64.66|±  |  1.23|
|xstory_cloze_es|      0|acc   |60.82|±  |  1.26|
|xstory_cloze_eu|      0|acc   |54.93|±  |  1.28|
|xstory_cloze_hi|      0|acc   |56.78|±  |  1.27|
|xstory_cloze_id|      0|acc   |59.76|±  |  1.26|
|xstory_cloze_my|      0|acc   |47.25|±  |  1.28|
|xstory_cloze_ru|      0|acc   |50.36|±  |  1.29|
|xstory_cloze_sw|      0|acc   |52.28|±  |  1.29|
|xstory_cloze_te|      0|acc   |56.52|±  |  1.28|
|xstory_cloze_zh|      0|acc   |58.24|±  |  1.27|

## bloom-1b7_xwinograd_0-shot.json
|    Task    |Version|Metric|Value|   |Stderr|
|------------|------:|------|----:|---|-----:|
|xwinograd_en|      0|acc   |74.71|±  |  0.90|
|xwinograd_fr|      0|acc   |68.67|±  |  5.12|
|xwinograd_jp|      0|acc   |54.12|±  |  1.61|
|xwinograd_pt|      0|acc   |63.50|±  |  2.97|
|xwinograd_ru|      0|acc   |52.38|±  |  2.82|
|xwinograd_zh|      0|acc   |69.64|±  |  2.05|
