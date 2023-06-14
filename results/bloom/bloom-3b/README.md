# bloom-3b

## bloom-3b_common_sense_reasoning_0-shot.json
|    Task     |Version| Metric |Value|   |Stderr|
|-------------|------:|--------|----:|---|-----:|
|arc_challenge|      0|acc     |27.99|±  |  1.31|
|             |       |acc_norm|30.55|±  |  1.35|
|arc_easy     |      0|acc     |59.47|±  |  1.01|
|             |       |acc_norm|53.24|±  |  1.02|
|boolq        |      1|acc     |61.62|±  |  0.85|
|copa         |      0|acc     |74.00|±  |  4.41|
|hellaswag    |      0|acc     |41.26|±  |  0.49|
|             |       |acc_norm|52.72|±  |  0.50|
|mc_taco      |      0|em      |11.94|   |      |
|             |       |f1      |49.57|   |      |
|openbookqa   |      0|acc     |21.60|±  |  1.84|
|             |       |acc_norm|32.20|±  |  2.09|
|piqa         |      0|acc     |70.84|±  |  1.06|
|             |       |acc_norm|70.51|±  |  1.06|
|prost        |      0|acc     |22.69|±  |  0.31|
|             |       |acc_norm|26.36|±  |  0.32|
|swag         |      0|acc     |47.36|±  |  0.35|
|             |       |acc_norm|64.59|±  |  0.34|
|winogrande   |      0|acc     |58.72|±  |  1.38|
|wsc273       |      0|acc     |76.92|±  |  2.55|

## bloom-3b_gsm8k_8-shot.json
|Task |Version|Metric|Value|   |Stderr|
|-----|------:|------|----:|---|-----:|
|gsm8k|      0|acc   | 1.21|±  |   0.3|

## bloom-3b_mathematical_reasoning_few_shot_5-shot.json
|          Task           |Version| Metric |Value|   |Stderr|
|-------------------------|------:|--------|----:|---|-----:|
|drop                     |      1|em      | 2.10|±  |  0.15|
|                         |       |f1      | 4.63|±  |  0.17|
|gsm8k                    |      0|acc     | 0.00|±  |  0.00|
|math_algebra             |      1|acc     | 0.00|±  |  0.00|
|math_counting_and_prob   |      1|acc     | 0.21|±  |  0.21|
|math_geometry            |      1|acc     | 0.00|±  |  0.00|
|math_intermediate_algebra|      1|acc     | 0.00|±  |  0.00|
|math_num_theory          |      1|acc     | 0.19|±  |  0.19|
|math_prealgebra          |      1|acc     | 0.11|±  |  0.11|
|math_precalc             |      1|acc     | 0.00|±  |  0.00|
|mathqa                   |      0|acc     |25.26|±  |  0.80|
|                         |       |acc_norm|25.06|±  |  0.79|

## bloom-3b_pawsx_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|pawsx_de|      0|acc   | 54.6|±  |  1.11|
|pawsx_en|      0|acc   | 56.8|±  |  1.11|
|pawsx_es|      0|acc   | 56.4|±  |  1.11|
|pawsx_fr|      0|acc   | 47.6|±  |  1.12|
|pawsx_ja|      0|acc   | 44.6|±  |  1.11|
|pawsx_ko|      0|acc   | 46.3|±  |  1.12|
|pawsx_zh|      0|acc   | 47.1|±  |  1.12|

## bloom-3b_question_answering_0-shot.json
|    Task     |Version|   Metric   |Value|   |Stderr|
|-------------|------:|------------|----:|---|-----:|
|headqa_en    |      0|acc         |28.41|±  |  0.86|
|             |       |acc_norm    |33.37|±  |  0.90|
|headqa_es    |      0|acc         |26.44|±  |  0.84|
|             |       |acc_norm    |31.00|±  |  0.88|
|logiqa       |      0|acc         |20.74|±  |  1.59|
|             |       |acc_norm    |29.19|±  |  1.78|
|squad2       |      1|exact       | 6.91|   |      |
|             |       |f1          |11.51|   |      |
|             |       |HasAns_exact|11.10|   |      |
|             |       |HasAns_f1   |20.31|   |      |
|             |       |NoAns_exact | 2.74|   |      |
|             |       |NoAns_f1    | 2.74|   |      |
|             |       |best_exact  |50.07|   |      |
|             |       |best_f1     |50.08|   |      |
|triviaqa     |      1|acc         | 4.15|±  |  0.19|
|truthfulqa_mc|      1|mc1         |23.26|±  |  1.48|
|             |       |mc2         |40.57|±  |  1.44|
|webqs        |      0|acc         | 1.67|±  |  0.28|

## bloom-3b_reading_comprehension_0-shot.json
|Task|Version|Metric|Value|   |Stderr|
|----|------:|------|----:|---|-----:|
|coqa|      1|f1    |61.50|±  |  1.77|
|    |       |em    |46.07|±  |  2.02|
|drop|      1|em    | 1.94|±  |  0.14|
|    |       |f1    | 8.88|±  |  0.20|
|race|      1|acc   |35.22|±  |  1.48|

## bloom-3b_xcopa_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|xcopa_et|      0|acc   | 49.2|±  |  2.24|
|xcopa_ht|      0|acc   | 50.2|±  |  2.24|
|xcopa_id|      0|acc   | 69.2|±  |  2.07|
|xcopa_it|      0|acc   | 51.6|±  |  2.24|
|xcopa_qu|      0|acc   | 50.6|±  |  2.24|
|xcopa_sw|      0|acc   | 51.4|±  |  2.24|
|xcopa_ta|      0|acc   | 58.0|±  |  2.21|
|xcopa_th|      0|acc   | 52.6|±  |  2.24|
|xcopa_tr|      0|acc   | 53.4|±  |  2.23|
|xcopa_vi|      0|acc   | 68.8|±  |  2.07|
|xcopa_zh|      0|acc   | 62.0|±  |  2.17|

## bloom-3b_xnli_0-shot.json
| Task  |Version|Metric|Value|   |Stderr|
|-------|------:|------|----:|---|-----:|
|xnli_ar|      0|acc   |33.43|±  |  0.67|
|xnli_bg|      0|acc   |37.90|±  |  0.69|
|xnli_de|      0|acc   |40.40|±  |  0.69|
|xnli_el|      0|acc   |33.21|±  |  0.67|
|xnli_en|      0|acc   |53.41|±  |  0.70|
|xnli_es|      0|acc   |49.08|±  |  0.71|
|xnli_fr|      0|acc   |49.18|±  |  0.71|
|xnli_hi|      0|acc   |45.55|±  |  0.70|
|xnli_ru|      0|acc   |41.40|±  |  0.70|
|xnli_sw|      0|acc   |35.83|±  |  0.68|
|xnli_th|      0|acc   |33.39|±  |  0.67|
|xnli_tr|      0|acc   |33.81|±  |  0.67|
|xnli_ur|      0|acc   |40.00|±  |  0.69|
|xnli_vi|      0|acc   |46.51|±  |  0.70|
|xnli_zh|      0|acc   |37.43|±  |  0.68|

## bloom-3b_xstory_cloze_0-shot.json
|     Task      |Version|Metric|Value|   |Stderr|
|---------------|------:|------|----:|---|-----:|
|xstory_cloze_ar|      0|acc   |56.59|±  |  1.28|
|xstory_cloze_en|      0|acc   |66.78|±  |  1.21|
|xstory_cloze_es|      0|acc   |64.13|±  |  1.23|
|xstory_cloze_eu|      0|acc   |55.66|±  |  1.28|
|xstory_cloze_hi|      0|acc   |57.58|±  |  1.27|
|xstory_cloze_id|      0|acc   |60.82|±  |  1.26|
|xstory_cloze_my|      0|acc   |46.59|±  |  1.28|
|xstory_cloze_ru|      0|acc   |50.69|±  |  1.29|
|xstory_cloze_sw|      0|acc   |53.01|±  |  1.28|
|xstory_cloze_te|      0|acc   |58.17|±  |  1.27|
|xstory_cloze_zh|      0|acc   |60.89|±  |  1.26|

## bloom-3b_xwinograd_0-shot.json
|    Task    |Version|Metric|Value|   |Stderr|
|------------|------:|------|----:|---|-----:|
|xwinograd_en|      0|acc   |79.10|±  |  0.84|
|xwinograd_fr|      0|acc   |71.08|±  |  5.01|
|xwinograd_jp|      0|acc   |56.62|±  |  1.60|
|xwinograd_pt|      0|acc   |70.34|±  |  2.82|
|xwinograd_ru|      0|acc   |53.65|±  |  2.81|
|xwinograd_zh|      0|acc   |73.61|±  |  1.97|
