# FrenchBench

### Paper

FrenchBench is a benchmark for evaluating French language models, introduced in the paper
[CroissantLLM: A Truly Bilingual French-English Language Model](https://arxiv.org/abs/2402.00786).
It is a collection of tasks that evaluate the ability of a language model to understand and generate French text.
This benchmark is constructed both from openly available datasets, as well as newly released manually annotated data.

### Citation

```bibtex
@misc{faysse2024croissantllm,
      title={CroissantLLM: A Truly Bilingual French-English Language Model},
      author={Manuel Faysse and Patrick Fernandes and Nuno M. Guerreiro and António Loison and Duarte M. Alves and Caio Corro and Nicolas Boizard and João Alves and Ricardo Rei and Pedro H. Martins and Antoni Bigata Casademunt and François Yvon and André F. T. Martins and Gautier Viaud and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2402.00786},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

- `french_bench`: All tasks (non-perplexity based)
- `french_bench_gen`: All official generative tasks
- `french_bench_mc`: All official multiple choice tasks
- `french_bench_perplexity`: All perplexity-based tasks (0 shot is recommended)
- `french_bench_extra`: All extra tasks

#### Tasks


The following tasks evaluate tasks on the French Bench dataset using various scoring methods.
  - french_bench_boolqa
  - french_bench_fquadv2
  - french_bench_fquadv2_bool
  - french_bench_fquadv2_genq
  - french_bench_fquadv2_hasAns
  - french_bench_topic_based_nli
  - french_bench_multifquad
  - french_bench_grammar
  - french_bench_vocab
  - french_bench_reading_comp
  - french_bench_xnli (modified XNLI)
  - french_bench_orangesum_abstract
  - french_bench_orangesum_title
  - french_bench_trivia
  - french_bench_hellaswag
  - french_bench_arc_challenge

The french bench also includes other tasks from various benchmarks:
- `belebele_fra_Latn`: Belebele French
- `wmt14-en-fr`: WMT14 English-French
- `wmt14-fr-en`: WMT14 French-English

# Not to use in few-shot
- `crows_pairs_french`: Crows Pairs French
- `french_bench_opus_perplexity`: Opus Perplexity


### Usage

```bash
# openai
lm_eval --model openai-completions --model_args engine=text-davinci-003  --tasks french_bench  --limit 100 --num_fewshot 3 --batch_size auto --output_path data/french_bench/davinci-003/results_french_bench_3shot.json
lm_eval --model openai-completions --model_args engine=text-davinci-003  --tasks french_bench_opus_perplexity,crows_pairs_french  --limit 100 --batch_size auto --output_path data/french_bench/davinci-003/results_french_bench2_0shot.json


lm_eval --model hf --model_args pretrained=gpt2 --tasks french_bench --device cuda:0 --limit 100 --num_fewshot 3 --batch_size 8 --output_path data/french_bench/gpt2/results_french_bench_3shot.json
lm_eval --model hf --model_args pretrained=gpt2 --tasks french_bench_opus_perplexity,crows_pairs_french --device cuda:0 --limit 100 --batch_size auto --output_path data/french_bench/gpt2/results_french_bench2_0shot.json

lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf --tasks french_bench --device cuda:0 --limit 100 --num_fewshot 3 --batch_size 4 --output_path data/french_bench/llama-2-7b-hf/results_french_bench_3shot.json
lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf --tasks french_bench_opus_perplexity,crows_pairs_french --device cuda:0 --limit 100 --batch_size auto --output_path data/french_bench/llama-2-7b-hf/results_french_bench2_0shot.json
```

HF and Accelerate options can be added when loading a model:
```bash
  accelerate launch -m lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype="float16" --tasks french_bench
```

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [x] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
