# Model Comparison Results

Comparison of evaluation results across different models.

Source directory: `all-results-formatted`


## nl2foam_gen_formatted

| Metric | Qwen/Qwen3-8B | nl2foam_sft_0__8__1765777563 |
|--------|--------|--------|
| bert_score | 0.759878 ± 0.000233 | 0.769897 ± 0.000157 |
| bleu | 0.269642 ± 0.002064 | 0.489728 ± 0.002233 |
| rouge1 | 0.376431 ± 0.001377 | 0.584414 ± 0.001488 |
| rouge2 | 0.264649 ± 0.001161 | 0.545179 ± 0.001863 |
| rougeL | 0.245172 ± 0.000868 | 0.566523 ± 0.001693 |


## nl2foam_llm_judge_formatted

| Metric | Qwen/Qwen3-8B | nl2foam_sft_0__8__1765777563 |
|--------|--------|--------|
| llm_judge | 36.292000 ± 0.565066 | 89.338000 ± 0.440076 |


## nl2foam_perplexity_formatted

| Metric | Qwen/Qwen3-8B | nl2foam_sft_0__8__1765777563 |
|--------|--------|--------|
| bits_per_byte | 0.381036 | 0.186951 |
| byte_perplexity | 1.302276 | 1.138355 |
| word_perplexity | 8.912307 | 2.924835 |
