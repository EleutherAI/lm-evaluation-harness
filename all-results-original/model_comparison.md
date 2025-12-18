# Model Comparison Results

Comparison of evaluation results across different models.

Source directory: `all-results-original`


## nl2foam_gen_original

| Metric | Qwen/Qwen3-8B | YYgroup/AutoCFD-7B | nl2foam_sft_0__8__1765350318 |
|--------|--------|--------|--------|
| bert_score | 0.760535 ± 0.000223 | 0.863128 ± 0.001284 | 0.801325 ± 0.001144 |
| bleu | 0.270265 ± 0.001900 | 0.610422 ± 0.004075 | 0.577071 ± 0.002462 |
| rouge1 | 0.369871 ± 0.001477 | 0.712577 ± 0.003124 | 0.722696 ± 0.001470 |
| rouge2 | 0.247999 ± 0.001090 | 0.572337 ± 0.002594 | 0.557614 ± 0.001314 |
| rougeL | 0.232955 ± 0.000860 | 0.620017 ± 0.002845 | 0.556705 ± 0.001833 |


## nl2foam_llm_judge_original

| Metric | Qwen/Qwen3-8B | YYgroup/AutoCFD-7B | nl2foam_sft_0__8__1765350318 |
|--------|--------|--------|--------|
| llm_judge | 34.128000 ± 0.560925 | 84.842000 ± 1.022584 | 89.156000 ± 0.449198 |


## nl2foam_perplexity_original

| Metric | Qwen/Qwen3-8B | YYgroup/AutoCFD-7B | nl2foam_sft_0__8__1765350318 |
|--------|--------|--------|--------|
| bits_per_byte | 0.335993 | 0.220715 | 0.166343 |
| byte_perplexity | 1.262246 | 1.165311 | 1.122210 |
| word_perplexity | 6.925089 | 3.565170 | 2.606617 |
