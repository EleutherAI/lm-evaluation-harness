# bloom-560m

## bloom-560m_common_sense_reasoning_0-shot.json
|    Task     |Version| Metric |Value|   |Stderr|
|-------------|------:|--------|----:|---|-----:|
|arc_challenge|      0|acc     |22.44|±  |  1.22|
|             |       |acc_norm|23.98|±  |  1.25|
|arc_easy     |      0|acc     |47.35|±  |  1.02|
|             |       |acc_norm|41.67|±  |  1.01|
|boolq        |      1|acc     |55.14|±  |  0.87|
|copa         |      0|acc     |61.00|±  |  4.90|
|hellaswag    |      0|acc     |31.56|±  |  0.46|
|             |       |acc_norm|36.56|±  |  0.48|
|mc_taco      |      0|em      |17.42|   |      |
|             |       |f1      |31.43|   |      |
|openbookqa   |      0|acc     |17.20|±  |  1.69|
|             |       |acc_norm|28.20|±  |  2.01|
|piqa         |      0|acc     |64.09|±  |  1.12|
|             |       |acc_norm|65.13|±  |  1.11|
|prost        |      0|acc     |22.08|±  |  0.30|
|             |       |acc_norm|32.08|±  |  0.34|
|swag         |      0|acc     |40.35|±  |  0.35|
|             |       |acc_norm|52.96|±  |  0.35|
|winogrande   |      0|acc     |52.80|±  |  1.40|
|wsc273       |      0|acc     |66.67|±  |  2.86|

## bloom-560m_gsm8k_8-shot.json
|Task |Version|Metric|Value|   |Stderr|
|-----|------:|------|----:|---|-----:|
|gsm8k|      0|acc   | 0.53|±  |   0.2|

## bloom-560m_lambada_openai_0-shot.json
|        Task        |Version|Metric| Value |   |Stderr|
|--------------------|------:|------|------:|---|-----:|
|lambada_openai      |      0|ppl   |  28.68|±  |  1.08|
|                    |       |acc   |  35.40|±  |  0.67|
|lambada_openai_cloze|      0|ppl   |6212.81|±  |267.17|
|                    |       |acc   |   0.45|±  |  0.09|

## bloom-560m_mathematical_reasoning_few_shot_5-shot.json
|          Task           |Version| Metric |Value|   |Stderr|
|-------------------------|------:|--------|----:|---|-----:|
|drop                     |      1|em      | 1.26|±  |  0.11|
|                         |       |f1      | 3.50|±  |  0.14|
|gsm8k                    |      0|acc     | 0.00|±  |  0.00|
|math_algebra             |      1|acc     | 0.00|±  |  0.00|
|math_counting_and_prob   |      1|acc     | 0.00|±  |  0.00|
|math_geometry            |      1|acc     | 0.00|±  |  0.00|
|math_intermediate_algebra|      1|acc     | 0.00|±  |  0.00|
|math_num_theory          |      1|acc     | 0.19|±  |  0.19|
|math_prealgebra          |      1|acc     | 0.23|±  |  0.16|
|math_precalc             |      1|acc     | 0.00|±  |  0.00|
|mathqa                   |      0|acc     |22.51|±  |  0.76|
|                         |       |acc_norm|22.35|±  |  0.76|

## bloom-560m_pawsx_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|pawsx_de|      0|acc   |52.80|±  |  1.12|
|pawsx_en|      0|acc   |52.00|±  |  1.12|
|pawsx_es|      0|acc   |53.25|±  |  1.12|
|pawsx_fr|      0|acc   |47.95|±  |  1.12|
|pawsx_ja|      0|acc   |44.90|±  |  1.11|
|pawsx_ko|      0|acc   |51.90|±  |  1.12|
|pawsx_zh|      0|acc   |45.20|±  |  1.11|

## bloom-560m_question_answering_0-shot.json
|    Task     |Version|   Metric   |Value|   |Stderr|
|-------------|------:|------------|----:|---|-----:|
|headqa_en    |      0|acc         |25.67|±  |  0.83|
|             |       |acc_norm    |29.58|±  |  0.87|
|headqa_es    |      0|acc         |23.96|±  |  0.82|
|             |       |acc_norm    |27.17|±  |  0.85|
|logiqa       |      0|acc         |22.58|±  |  1.64|
|             |       |acc_norm    |27.19|±  |  1.75|
|squad2       |      1|exact       | 0.43|   |      |
|             |       |f1          | 1.86|   |      |
|             |       |HasAns_exact| 0.76|   |      |
|             |       |HasAns_f1   | 3.62|   |      |
|             |       |NoAns_exact | 0.10|   |      |
|             |       |NoAns_f1    | 0.10|   |      |
|             |       |best_exact  |50.07|   |      |
|             |       |best_f1     |50.07|   |      |
|triviaqa     |      1|acc         | 1.44|±  |  0.11|
|truthfulqa_mc|      1|mc1         |24.48|±  |  1.51|
|             |       |mc2         |42.43|±  |  1.51|
|webqs        |      0|acc         | 0.84|±  |  0.20|

## bloom-560m_reading_comprehension_0-shot.json
|Task|Version|Metric|Value|   |Stderr|
|----|------:|------|----:|---|-----:|
|coqa|      1|f1    |22.71|±  |  1.67|
|    |       |em    |17.40|±  |  1.62|
|drop|      1|em    | 1.50|±  |  0.12|
|    |       |f1    | 6.21|±  |  0.17|
|race|      1|acc   |30.24|±  |  1.42|

## bloom-560m_xcopa_0-shot.json
|  Task  |Version|Metric|Value|   |Stderr|
|--------|------:|------|----:|---|-----:|
|xcopa_et|      0|acc   | 49.0|±  |  2.24|
|xcopa_ht|      0|acc   | 50.2|±  |  2.24|
|xcopa_id|      0|acc   | 59.2|±  |  2.20|
|xcopa_it|      0|acc   | 50.8|±  |  2.24|
|xcopa_qu|      0|acc   | 50.2|±  |  2.24|
|xcopa_sw|      0|acc   | 51.6|±  |  2.24|
|xcopa_ta|      0|acc   | 55.8|±  |  2.22|
|xcopa_th|      0|acc   | 54.4|±  |  2.23|
|xcopa_tr|      0|acc   | 53.0|±  |  2.23|
|xcopa_vi|      0|acc   | 61.0|±  |  2.18|
|xcopa_zh|      0|acc   | 58.6|±  |  2.20|

## bloom-560m_xnli_0-shot.json
| Task  |Version|Metric|Value|   |Stderr|
|-------|------:|------|----:|---|-----:|
|xnli_ar|      0|acc   |33.35|±  |  0.67|
|xnli_bg|      0|acc   |33.39|±  |  0.67|
|xnli_de|      0|acc   |34.79|±  |  0.67|
|xnli_el|      0|acc   |33.33|±  |  0.67|
|xnli_en|      0|acc   |49.50|±  |  0.71|
|xnli_es|      0|acc   |45.23|±  |  0.70|
|xnli_fr|      0|acc   |45.29|±  |  0.70|
|xnli_hi|      0|acc   |40.84|±  |  0.69|
|xnli_ru|      0|acc   |34.01|±  |  0.67|
|xnli_sw|      0|acc   |33.17|±  |  0.67|
|xnli_th|      0|acc   |33.57|±  |  0.67|
|xnli_tr|      0|acc   |33.43|±  |  0.67|
|xnli_ur|      0|acc   |37.13|±  |  0.68|
|xnli_vi|      0|acc   |40.52|±  |  0.69|
|xnli_zh|      0|acc   |33.95|±  |  0.67|

## bloom-560m_xstory_cloze_0-shot.json
|     Task      |Version|Metric|Value|   |Stderr|
|---------------|------:|------|----:|---|-----:|
|xstory_cloze_ar|      0|acc   |52.08|±  |  1.29|
|xstory_cloze_en|      0|acc   |61.22|±  |  1.25|
|xstory_cloze_es|      0|acc   |55.86|±  |  1.28|
|xstory_cloze_eu|      0|acc   |53.61|±  |  1.28|
|xstory_cloze_hi|      0|acc   |55.00|±  |  1.28|
|xstory_cloze_id|      0|acc   |55.53|±  |  1.28|
|xstory_cloze_my|      0|acc   |47.19|±  |  1.28|
|xstory_cloze_ru|      0|acc   |49.17|±  |  1.29|
|xstory_cloze_sw|      0|acc   |49.83|±  |  1.29|
|xstory_cloze_te|      0|acc   |55.72|±  |  1.28|
|xstory_cloze_zh|      0|acc   |54.53|±  |  1.28|

## bloom-560m_xwinograd_0-shot.json
|    Task    |Version|Metric|Value|   |Stderr|
|------------|------:|------|----:|---|-----:|
|xwinograd_en|      0|acc   |65.89|±  |  0.98|
|xwinograd_fr|      0|acc   |60.24|±  |  5.40|
|xwinograd_jp|      0|acc   |52.97|±  |  1.61|
|xwinograd_pt|      0|acc   |60.08|±  |  3.03|
|xwinograd_ru|      0|acc   |49.21|±  |  2.82|
|xwinograd_zh|      0|acc   |67.66|±  |  2.09|
