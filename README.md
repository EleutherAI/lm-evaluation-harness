# Evaluation Harness for Large Language Models

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Python%20application/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)

**WARNING**: This project is currently under active development. Interfaces and task implementations may change rapidly and without warning. 

## Overview 

The goal of this project is to build a set of tools for evaluating LMs on typical NLU tasks, based on evaluation of GPT-3 as described in https://arxiv.org/pdf/2005.14165.pdf. Following the initial description, this repo should support 3 functions:
1. LM Evaluation
2. Removing task val/test data from LM training set
3. Adding task training data to LM training set

### Overview of Tasks

|                    Task Name                    |Train|Val|Test|                                   Metrics                                    |
|-------------------------------------------------|-----|---|----|------------------------------------------------------------------------------|
|cola                                             |✓    |✓  |    |mcc                                                                           |
|mnli                                             |✓    |✓  |    |acc                                                                           |
|mnli_mismatched                                  |✓    |✓  |    |acc                                                                           |
|mrpc                                             |✓    |✓  |    |acc, f1                                                                       |
|rte                                              |✓    |✓  |    |acc                                                                           |
|qnli                                             |✓    |✓  |    |acc                                                                           |
|qqp                                              |✓    |✓  |    |acc, f1                                                                       |
|sst                                              |✓    |✓  |    |acc                                                                           |
|wnli                                             |✓    |✓  |    |acc                                                                           |
|boolq                                            |✓    |✓  |    |acc                                                                           |
|cb                                               |✓    |✓  |    |acc, f1                                                                       |
|copa                                             |✓    |✓  |    |acc                                                                           |
|multirc                                          |✓    |✓  |    |acc                                                                           |
|record                                           |✓    |✓  |    |f1, em                                                                        |
|wic                                              |✓    |✓  |    |acc                                                                           |
|wsc                                              |✓    |✓  |    |acc                                                                           |
|coqa                                             |✓    |✓  |    |f1, em                                                                        |
|drop                                             |✓    |✓  |    |em, f1                                                                        |
|lambada                                          |     |✓  |    |ppl, acc                                                                      |
|lambada_cloze                                    |     |✓  |    |ppl, acc                                                                      |
|cbt-cn                                           |✓    |✓  |✓   |acc                                                                           |
|cbt-ne                                           |✓    |✓  |✓   |acc                                                                           |
|piqa                                             |✓    |✓  |    |acc, acc_norm                                                                 |
|pubmedqa                                         |     |   |✓   |acc                                                                           |
|sciq                                             |✓    |✓  |✓   |acc, acc_norm                                                                 |
|qa4mre_2011                                      |     |   |✓   |acc, acc_norm                                                                 |
|qa4mre_2012                                      |     |   |✓   |acc, acc_norm                                                                 |
|qa4mre_2013                                      |     |   |✓   |acc, acc_norm                                                                 |
|triviaqa                                         |✓    |✓  |    |acc                                                                           |
|arc_easy                                         |✓    |✓  |✓   |acc, acc_norm                                                                 |
|arc_challenge                                    |✓    |✓  |✓   |acc, acc_norm                                                                 |
|logiqa                                           |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hellaswag                                        |✓    |✓  |    |acc, acc_norm                                                                 |
|openbookqa                                       |✓    |✓  |✓   |acc, acc_norm                                                                 |
|squad2                                           |✓    |✓  |    |exact, f1, HasAns_exact, HasAns_f1, NoAns_exact, NoAns_f1, best_exact, best_f1|
|race                                             |✓    |✓  |✓   |acc                                                                           |
|headqa                                           |✓    |✓  |✓   |acc, acc_norm                                                                 |
|mathqa                                           |✓    |✓  |✓   |acc, acc_norm                                                                 |
|webqs                                            |✓    |   |✓   |acc                                                                           |
|wsc273                                           |     |   |✓   |acc                                                                           |
|winogrande                                       |✓    |✓  |    |acc                                                                           |
|anli_r1                                          |✓    |✓  |✓   |acc                                                                           |
|anli_r2                                          |✓    |✓  |✓   |acc                                                                           |
|anli_r3                                          |✓    |✓  |✓   |acc                                                                           |
|ethics_cm                                        |✓    |   |✓   |acc                                                                           |
|ethics_deontology                                |✓    |   |✓   |acc, em                                                                       |
|ethics_justice                                   |✓    |   |✓   |acc, em                                                                       |
|ethics_utilitarianism_original                   |     |   |✓   |acc                                                                           |
|ethics_utilitarianism                            |✓    |   |✓   |acc                                                                           |
|ethics_virtue                                    |✓    |   |✓   |acc, em                                                                       |
|math_algebra                                     |✓    |   |✓   |acc                                                                           |
|math_counting_and_prob                           |✓    |   |✓   |acc                                                                           |
|math_geometry                                    |✓    |   |✓   |acc                                                                           |
|math_intermediate_algebra                        |✓    |   |✓   |acc                                                                           |
|math_num_theory                                  |✓    |   |✓   |acc                                                                           |
|math_prealgebra                                  |✓    |   |✓   |acc                                                                           |
|math_precalc                                     |✓    |   |✓   |acc                                                                           |
|arithmetic_2da                                   |     |✓  |    |acc                                                                           |
|arithmetic_2ds                                   |     |✓  |    |acc                                                                           |
|arithmetic_3da                                   |     |✓  |    |acc                                                                           |
|arithmetic_3ds                                   |     |✓  |    |acc                                                                           |
|arithmetic_4da                                   |     |✓  |    |acc                                                                           |
|arithmetic_4ds                                   |     |✓  |    |acc                                                                           |
|arithmetic_5da                                   |     |✓  |    |acc                                                                           |
|arithmetic_5ds                                   |     |✓  |    |acc                                                                           |
|arithmetic_2dm                                   |     |✓  |    |acc                                                                           |
|arithmetic_1dc                                   |     |✓  |    |acc                                                                           |
|hendrycksTest-abstract_algebra                   |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-anatomy                            |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-astronomy                          |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-business_ethics                    |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-clinical_knowledge                 |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-college_biology                    |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-college_chemistry                  |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-college_computer_science           |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-college_mathematics                |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-college_medicine                   |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-college_physics                    |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-computer_security                  |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-conceptual_physics                 |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-econometrics                       |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-electrical_engineering             |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-elementary_mathematics             |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-formal_logic                       |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-global_facts                       |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_biology                |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_chemistry              |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_computer_science       |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_european_history       |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_geography              |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_government_and_politics|✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_macroeconomics         |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_mathematics            |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_microeconomics         |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_physics                |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_psychology             |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_statistics             |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_us_history             |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-high_school_world_history          |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-human_aging                        |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-human_sexuality                    |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-international_law                  |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-jurisprudence                      |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-logical_fallacies                  |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-machine_learning                   |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-management                         |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-marketing                          |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-medical_genetics                   |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-miscellaneous                      |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-moral_disputes                     |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-moral_scenarios                    |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-nutrition                          |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-philosophy                         |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-prehistory                         |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-professional_accounting            |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-professional_law                   |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-professional_medicine              |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-professional_psychology            |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-public_relations                   |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-security_studies                   |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-sociology                          |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-us_foreign_policy                  |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-virology                           |✓    |✓  |✓   |acc, acc_norm                                                                 |
|hendrycksTest-world_religions                    |✓    |✓  |✓   |acc, acc_norm                                                                 |
|wmt14-en-fr                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt14-fr-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt16-en-ro                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt16-ro-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt16-de-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt16-en-de                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-cs-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-de-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-de-fr                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-en-cs                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-en-de                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-en-iu                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-en-ja                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-en-km                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-en-pl                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-en-ps                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-en-ru                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-en-ta                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-en-zh                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-fr-de                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-iu-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-ja-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-km-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-pl-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-ps-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-ru-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-ta-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|wmt20-zh-en                                      |     |   |✓   |bleu, chrf, ter                                                               |
|iwslt17-en-ar                                    |     |   |✓   |bleu, chrf, ter                                                               |
|iwslt17-ar-en                                    |     |   |✓   |bleu, chrf, ter                                                               |
|anagrams1                                        |     |✓  |    |acc                                                                           |
|anagrams2                                        |     |✓  |    |acc                                                                           |
|cycle_letters                                    |     |✓  |    |acc                                                                           |
|random_insertion                                 |     |✓  |    |acc                                                                           |
|reversed_words                                   |     |✓  |    |acc                                                                           |
|pile_arxiv                                       |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_books3                                      |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_bookcorpus2                                 |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_commoncrawl                                 |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_dm-mathematics                              |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_enron                                       |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_europarl                                    |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_freelaw                                     |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_github                                      |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_gutenberg                                   |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_hackernews                                  |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_nih-exporter                                |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_opensubtitles                               |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_openwebtext2                                |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_philpapers                                  |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_pile-cc                                     |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_pubmed-abstracts                            |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_pubmed-central                              |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_stackexchange                               |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_uspto                                       |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_ubuntu-irc                                  |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_wikipedia                                   |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |
|pile_youtubesubtitles                            |     |✓  |✓   |word_perplexity, byte_perplexity, bits_per_byte                               |



## Usage

### Evaluate a task

To evaluate a model, (e.g. GPT-2) on NLU tasks (e.g. RTE, Winograd Scheme Challenge), you can run the following command.

```bash
python main.py \
	--model gpt2 \
	--model_args device=cuda:0 \
	--tasks rte,wsc \
	--provide_description \
	--num_fewshot 2
```

If you have access to an OpenAI API key, you can also evaluate GPT-3 on various tasks with the following command:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
	--model gpt3 \
	--tasks rte,wsc \
	--provide_description \
	--num_fewshot 2
```

To inspect what the LM inputs look like, you can run the following command:

```bash
python write_out.py \
	--tasks all_tasks \
	--provide_description \
	--num_fewshot 5 \
	--num_examples 10 \
	--output_base_path /path/to/output/folder
```

This will write out one text file for each task.

### Code Structure

There are two major components of the library:

1. LMs (language models), e.g. GPT-2, GPT-3
2. Tasks, e.g. MNLI, RTE, SQuAD (coming soon)

Both LMs (`lm_eval.models`) and Tasks (`lm_eval.tasks`) are kept in a registry data structure, for easy CLI instantiation.

**If you want to extend either models or tasks, simply add a new LM or Task subclass, and decorate with the registry decorator**.

The [GPT-3 Evaluations Project](https://github.com/EleutherAI/lm_evaluation_harness/projects/1) tracks our progress implementing new tasks. Right now, we are focused on getting all the datasets loaded so that we can dedupe against the training data. Implementing the actual evaluations is nice but not necessary at the current moment.

## Description

### 1. LM Evaluation
Given an LM, we want to evaluate it on a wide range of NLU tasks. We should at least cover the set of tasks in the GPT-3 paper, and any other tasks/benchmarks that are relevant. We will follow the GPT-3 format of a) zero-shot, b) one-shot, c) few-shot evaluation.

To do this, we need 3 components:
* Data downloader (shared with later sections, potentially needs to be directly linked to the latter 2 components)
* Task formatter
* Task evaluator

The **data downloader** should download data for the relevant tasks.
* We should heavily rely on Hugging Face's NLP for this. They are already doing most of the work with handling data scripts/caching.
* Optionally, we can rely directly on HF-NLP's caching, but that makes it awkward to handle non-HF-NLP datasets. Otherwise, we can just write them out to .jsonl. My feeling is that NLU data storage will be a drop in the bucket compared to LM data.
* Where we're not using HF-NLP, we can keep the data in the raw format (.jsonl, tsv, etc) and let the other components handle transforming it.

The **task formatter** formats the task input data into an LM-usable format.
* We should potentially support multiple formats for a given task, e.g. some formats may be better or worse suited for LM evaluation. See also: prompt-engineering
* The task formatter should also support zero/one/few-shot packing of training examples into an input. This may require weird interactions with the tokenizer for dealing with max-token issues.

The **task evaluator** scores a task.
* In essence, we want to generation output predictions for all our input examples, and feed them into some function that pops out a score (or scores)
An alternative approach is to collect the output logits and score them against the expected set of outputs.
* Some tasks have weird evaluation schemes, so we should make this as general as possible.
* Will thus likely have to be closely tied with the formatter.
* Likewise, we should take advantage of HF-NLP's metrics.
We might as well provide a sufficiently general API for the model to support OpenAI API as well. This can double up as an effort to reproduce the OpenAI NLU results.

### 2. Removing val/test data from LM training set
With the data downloader in place, we simply need to (1) expose the val/test examples, and (2) remove them from the training set.

* Arguably, (2) should be handled by LM preprocessing in a more general way. There are probably non-NLU-eval cases where we want to remove some specific data from training.
* Depending on how exactly we do the val/test removal, we may want to format the same example multiple ways to ensure that they don't get leaked into the training set in a slightly tweaked format.
* Thought experiment: SQuAD is based largely on Wikipedia. What exactly would we want to remove from the LM?
* [GPT-3]: In GPT-3, they attempted to remove val/test from their LM set, but there was a bug that caused leakage. So they ended up doing the opposite: removing overlaps from the LM set from the val/test. Funky.
* [GPT-3]: See page 30 and Appendix C for details. They do some funky n-gram based search and removal. We should think about whether we want to follow their protocol exactly

### 3. Adding task training data to LM training set
This part is the easiest. I guess we just write out some text files containing the training data? We can let the usual LM preprocessing pipeline handle it from there.
