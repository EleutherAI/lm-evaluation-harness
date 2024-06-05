# Paloma

### Paper
Title: Paloma: A Benchmark for Evaluating Language Model Fit

Abstract: https://arxiv.org/abs/2312.10523v1

Paloma is a comprehensive benchmark designed to evaluate open language models across a wide range of domains, ranging from niche artist communities to mental health forums on Reddit. It assesses the performance of various models across 585 distinct domains.

Homepage: https://allenai.org/olmo


### Note

If you are running the entire `paloma` benchmark (or just `paloma_dolma_100_programing_languages`) with a HuggingFace model, make sure to pass `logits_cache=False` to `--model_args`, for example:
```
lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m,logits_cache=False --tasks paloma
```


### Citation
```
@article{paloma,
  title={{Paloma}: A Benchmark for Evaluating Language Model Fit},
  author={Magnusson, Ian and Bhagia, Akshita and Hofmann, Valentin and Soldaini, Luca and Harsh Jha, Ananya and Tafjord, Oyvind and Schwenk,Dustin and Walsh, Evan Pete and Elazar, Yanai and Lo, Kyle and Groenveld,Dirk and Beltagy,Iz and  Hajishirz,Hanneneh and Smith, Noah A. and Richardson,Kyle and Dodge,Jesse},
  journal={technical report},
  year={2023},
  url={https://paloma.allen.ai/}
}
```

### Groups and Tasks

#### Groups

* `paloma`

#### Tasks

* `paloma_4chan_meta_sep`
* `paloma_c4_100_domains`
* `paloma_c4_en`
* `paloma_dolma_100_programing_languages`
* `paloma_dolma_100_subreddits`
* `paloma_dolma-v1_5`
* `paloma_falcon-refinedweb`
* `paloma_gab`
* `paloma_m2d2_s2orc_unsplit`
* `paloma_m2d2_wikipedia_unsplit`
* `paloma_manosphere_meta_sep`
* `paloma_mc4`
* `paloma_ptb`
* `paloma_redpajama`
* `paloma_twitterAAE_HELM_fixed`
* `paloma_wikitext_103`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
