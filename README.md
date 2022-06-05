# Biomedical Working Group

This branch exists to facilitate experiments in the Biomedical Working group. 

The following steps should allow you to evaluate any transformers model using the BigBio prompts + datasets:
```bash
pip install git+https://github.com/bigscience-workshop/biomedical.git
git clone https://github.com/bigscience-workshop/lm-evaluation-harness
cd lm-evaluation-harness
git checkout bigbio
git submodule update --init
pip install -e .
```
Once you have run done these things, you can now use the BigBio datasets... in theory. The problem is that at the present moment we do not have any of the BigBio prompts in PromptSource. For now, the patch is to run
`cd ..; git clone -b eval-hackathon https://github.com/bigscience-workshop/promptsource; cd promptsource`
and then copy over whatever prompts you want to be using from the [OpenBioLink fork](https://github.com/OpenBioLink/promptsource) to your local files. Unfortunately we cannot use the OpenBioLink fork directly, as the eval-hackathon branch of PromptSource has some necessary modifications that don't exist on that fork. Once things are more settled in terms of long-term plans, we can set something up to address this but for now we need to hack it a bit.

At the current moment, the only dataset implemented is `scitail`, which requires [this prompt](https://github.com/OpenBioLink/promptsource/blob/main/promptsource/templates/scitail/scitail_bigbio_te/templates.yaml) be copied into the local version of PromptSource. Once you have added all the prompts you want to use locally, you can run `pip install -e .` so that your libraries notice the new prompts. Now you can actually run the eval harness by navigating to its directory and running `python main.py --model hf-causal --model_args pretrained=EleutherAI/gpt-neo-2.7B --tasks scitail`. This should report a table that looks like this if done correctly.
```
| Task  |         Prompt          |Version|Metric|Value |   |Stderr|
|-------|-------------------------|------:|------|-----:|---|-----:|
|scitail|Yes/No Entailment Framing|      0|acc   |0.6021|±  |0.0106|
```
If this errors, multiple prompts are reported, or GPT-Neo-2.7B is not getting 60.2% accuracy you have something set up wrong.

## Running Models
The Eval Harness supports a lot of models and settings. In particular, any model that is implemented in transformers, is accessible through the OpenAI API, or was trained using GPT-NeoX or Mesh-Transformer-Jax can be run out of the box. I'll start by breaking down the command from above:
1. `python main.py`: this is how you launch the eval harness
2. `--model hf-causal`: this is how you specify that you are using a model from HuggingFace's transformer library that is implemented as a causal decoder model. If you wish to use a sequence to sequence model like T5, instead use hf-seq2seq. You can also specify other model types like openaihere.
3. `--model_args pretrained=EleutherAI/gpt-neo-2.7B`: this is how you provide arguments to the model loading function. For most HuggingFace models, the only argument you need to provide is the name of the model you wish to load. For OpenAI models, you would use engine=davinci (or whatever engine you are specifying).
4. `--tasks scitail`: this is how you specify the task to run. Multiple tasks can be specified as a comma-separated list such as lambada,hellaswag.

There are a couple auxillery flags that may also be useful:
1. If you have access to a GPU cluster, you can distribute a model across your GPUs by adding --parallelize. You can also specify which GPUs to use with `--device cuda:0,1,2`. By default a parallelized run will use all GPUs, while a non-parallelized run will use GPU 0. You can also run on CPU with --device cpu.
2. If your model doesn't fill your avaliable GPU memory, you can set the batch size to N by adding `--batch_size N`. By default the batch size is 1.
3. If you would like to do few-shot evaluations, you can set the number of examples provided to N by adding `--num_fewshot N`. By default all evaluations are zero-shot.

## Implementing Datasets
If you want to use a dataset other than SciTail, you'll need to implement it in the Eval Harness. Fortunately, this is quite easy for most datasets. I will use `biosses` as an example, since I know it has prompts implemented.
1. Make a file called `lm_eval/tasks/biosses.py` and copy the contents of `lm_eval/tasks/scitail.py` into it.
2. Change `SciTailBase(BioTask)` to `BiossesBase(BioTask)` and `DATASET_PATH = "lm_eval/datasets/biomedical/bigbio/biodatasets/scitail"` to `DATASET_PATH = "lm_eval/datasets/biomedical/bigbio/biodatasets/biosses"`.  This defines the basic framework for the codebase, most notably how to load and access the dataset. If your dataset has a non-standard structure you may need to edit the contents of this class as well.
Change `SciTailTE(SciTailBase)` to `BiossesPairs(BiossesBase)` and `DATASET_NAME = "scitail_bigbio_te"` to `DATASET_NAME = "biosses_bigbio_pairs"`. This defines the particular schema you are using. If your dataset supports multiple schema, each one gets its own class.
**Note:** The file names and class names can be whatever you want, but the `DATASET_PATH` and `DATASET_NAME` values must match the directory and name of the BigBio version of the dataset from biomedical exactly.
4. Once you have done 1-3, all you need to do is tell the rest of the code that it exists. You'll need to make two changes to `lm_eval/tasks/__init__.py`: at the top you'll need to write `from . import biosses` and then in the `TASK_REGESTRY` you'll need to add the specific classes to make accessible to the user. In our case, this would be `"bioss": bioss.BiossPairs`. The string defines how the user calls the evaluation task, so please be descriptive if there are multiple prompts.
5. Congrats! You can now run your new task by running `python main.py --model hf-causal --model_args pretrained=EleutherAI/gpt-neo-2.7B --tasks bioss`

**Please push new datasets you add to the eval harness so others don't have to duplicate your work.**

# Promptsource X Language Model Evaluation Harness

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)

## Overview 

This project provides a unified framework to test language models (GPT-2, GPT-3, GPTNeo, etc) and seq2seq (T5, T0) models via prompt evaluation.

As of now, all the prompts are provided via `promptsource`; all datasets are in huggingface `datasets`. Both of these are not necessary for new tasks.

This fork is not (currently) backwards compatible with the original evaluation harness.

## Installation

```bash
git clone https://github.com/bigscience-workshop/lm-evaluation-harness
cd lm-evaluation-harness
pip install   "promptsource @ git+https://github.com/bigscience-workshop/promptsource@eval-hackathon"
pip install -e ".[dev]"
```

## Basic Usage

To evaluate a model, (e.g. GPT-2) on NLU tasks (e.g. LAMBADA, HellaSwag), you can run the following command. **When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility.** This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](https://github.com/EleutherAI/lm-evaluation-harness#task-versioning) section for more info.

```bash
python main.py \
	--model hf-causal \
  	--model_args pretrained=gpt2 \
	--tasks mrpc,gsarti/flores_101_afr
```

For larger models, you may wish to use parallelism or half precision. These can be activated using the `--parallelize` and `--half` flags respectively.

Features:

- Growing number of tasks integrated with `promptsource` (20+).
- Support for hugging face causal language models, huggingface seq2seq models, and the openai completion api (gpt3), with flexible tokenization-agnostic interface
- Task versioning to ensure reproducibility

# Original Notes from Eval Harness

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most importantly, the `gpt2` model can be used to load an arbitrary HuggingFace model. For example, to run GPTNeo use the following:

```bash
python main.py \
	--model gpt2 \
	--model_args pretrained=EleutherAI/gpt-neo-2.7B \
	--device cuda:0 \
	--tasks lambada,hellaswag
```

If you have access to the OpenAI API, you can also evaluate GPT-3:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
	--model gpt3 \
	--model_args engine=davinci \
	--tasks lambada,hellaswag
```

And if you want to verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
python main.py \
	--model gpt3 \
	--model_args engine=davinci \
	--tasks lambada,hellaswag \
	--check_integrity
```
To evaluate mesh-transformer-jax models that are not available on HF, please invoke eval harness through [this script](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

## Implementing new tasks

To implement a new task in eval harness, see [this guide](./docs/task_guide.md).

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```

### Full Task List

|                        Task Name                        |Train|Val|Test|Val/Test Docs|                                                                                     Metrics                                                                                     |
|---------------------------------------------------------|-----|---|----|------------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|cola                                                     |✓    |✓  |    |         1043|mcc                                                                                                                                                                              |
|mnli                                                     |✓    |✓  |    |         9815|acc                                                                                                                                                                              |
|mnli_mismatched                                          |✓    |✓  |    |         9832|acc                                                                                                                                                                              |
|mrpc                                                     |✓    |✓  |    |          408|acc, f1                                                                                                                                                                          |
|rte                                                      |✓    |✓  |    |          277|acc                                                                                                                                                                              |
|qnli                                                     |✓    |✓  |    |         5463|acc                                                                                                                                                                              |
|qqp                                                      |✓    |✓  |    |        40430|acc, f1                                                                                                                                                                          |
|sst                                                      |✓    |✓  |    |          872|acc                                                                                                                                                                              |
|wnli                                                     |✓    |✓  |    |           71|acc                                                                                                                                                                              |
|boolq                                                    |✓    |✓  |    |         3270|acc                                                                                                                                                                              |
|cb                                                       |✓    |✓  |    |           56|acc, f1                                                                                                                                                                          |
|copa                                                     |✓    |✓  |    |          100|acc                                                                                                                                                                              |
|multirc                                                  |✓    |✓  |    |         4848|acc                                                                                                                                                                              |
|record                                                   |✓    |✓  |    |        10000|f1, em                                                                                                                                                                           |
|wic                                                      |✓    |✓  |    |          638|acc                                                                                                                                                                              |
|wsc                                                      |✓    |✓  |    |          104|acc                                                                                                                                                                              |
|coqa                                                     |✓    |✓  |    |          500|f1, em                                                                                                                                                                           |
|drop                                                     |✓    |✓  |    |         9536|em, f1                                                                                                                                                                           |
|lambada                                                  |     |✓  |    |         5153|ppl, acc                                                                                                                                                                         |
|lambada_cloze                                            |     |✓  |    |         5153|ppl, acc                                                                                                                                                                         |
|lambada_mt_en                                            |     |✓  |    |         5153|ppl, acc                                                                                                                                                                         |
|lambada_mt_fr                                            |     |✓  |    |         5153|ppl, acc                                                                                                                                                                         |
|lambada_mt_de                                            |     |✓  |    |         5153|ppl, acc                                                                                                                                                                         |
|lambada_mt_it                                            |     |✓  |    |         5153|ppl, acc                                                                                                                                                                         |
|lambada_mt_es                                            |     |✓  |    |         5153|ppl, acc                                                                                                                                                                         |
|wikitext                                                 |     |✓  |✓   |           62|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|piqa                                                     |✓    |✓  |    |         1838|acc, acc_norm                                                                                                                                                                    |
|prost                                                    |     |   |✓   |        18736|acc, acc_norm                                                                                                                                                                    |
|mc_taco                                                  |     |✓  |✓   |         9442|f1, em                                                                                                                                                                           |
|pubmedqa                                                 |     |   |✓   |         1000|acc                                                                                                                                                                              |
|sciq                                                     |✓    |✓  |✓   |         1000|acc, acc_norm                                                                                                                                                                    |
|qa4mre_2011                                              |     |   |✓   |          120|acc, acc_norm                                                                                                                                                                    |
|qa4mre_2012                                              |     |   |✓   |          160|acc, acc_norm                                                                                                                                                                    |
|qa4mre_2013                                              |     |   |✓   |          284|acc, acc_norm                                                                                                                                                                    |
|triviaqa                                                 |✓    |✓  |    |        11313|acc                                                                                                                                                                              |
|arc_easy                                                 |✓    |✓  |✓   |         2376|acc, acc_norm                                                                                                                                                                    |
|arc_challenge                                            |✓    |✓  |✓   |         1172|acc, acc_norm                                                                                                                                                                    |
|logiqa                                                   |✓    |✓  |✓   |          651|acc, acc_norm                                                                                                                                                                    |
|hellaswag                                                |✓    |✓  |    |        10042|acc, acc_norm                                                                                                                                                                    |
|openbookqa                                               |✓    |✓  |✓   |          500|acc, acc_norm                                                                                                                                                                    |
|squad2                                                   |✓    |✓  |    |        11873|exact, f1, HasAns_exact, HasAns_f1, NoAns_exact, NoAns_f1, best_exact, best_f1                                                                                                   |
|race                                                     |✓    |✓  |✓   |         1045|acc                                                                                                                                                                              |
|headqa                                                   |✓    |✓  |✓   |         2742|acc, acc_norm                                                                                                                                                                    |
|headqa_es                                                |✓    |✓  |✓   |         2742|acc, acc_norm                                                                                                                                                                    |
|headqa_en                                                |✓    |✓  |✓   |         2742|acc, acc_norm                                                                                                                                                                    |
|mathqa                                                   |✓    |✓  |✓   |         2985|acc, acc_norm                                                                                                                                                                    |
|webqs                                                    |✓    |   |✓   |         2032|acc                                                                                                                                                                              |
|wsc273                                                   |     |   |✓   |          273|acc                                                                                                                                                                              |
|winogrande                                               |✓    |✓  |    |         1267|acc                                                                                                                                                                              |
|anli_r1                                                  |✓    |✓  |✓   |         1000|acc                                                                                                                                                                              |
|anli_r2                                                  |✓    |✓  |✓   |         1000|acc                                                                                                                                                                              |
|anli_r3                                                  |✓    |✓  |✓   |         1200|acc                                                                                                                                                                              |
|ethics_cm                                                |✓    |   |✓   |         3885|acc                                                                                                                                                                              |
|ethics_deontology                                        |✓    |   |✓   |         3596|acc, em                                                                                                                                                                          |
|ethics_justice                                           |✓    |   |✓   |         2704|acc, em                                                                                                                                                                          |
|ethics_utilitarianism_original                           |     |   |✓   |         4808|acc                                                                                                                                                                              |
|ethics_utilitarianism                                    |✓    |   |✓   |         4808|acc                                                                                                                                                                              |
|ethics_virtue                                            |✓    |   |✓   |         4975|acc, em                                                                                                                                                                          |
|truthfulqa_mc                                            |     |✓  |    |          817|mc1, mc2                                                                                                                                                                         |
|truthfulqa_gen                                           |     |✓  |    |          817|bleurt_max, bleurt_acc, bleurt_diff, bleu_max, bleu_acc, bleu_diff, rouge1_max, rouge1_acc, rouge1_diff, rouge2_max, rouge2_acc, rouge2_diff, rougeL_max, rougeL_acc, rougeL_diff|
|mutual                                                   |✓    |✓  |    |          886|r@1, r@2, mrr                                                                                                                                                                    |
|mutual_plus                                              |✓    |✓  |    |          886|r@1, r@2, mrr                                                                                                                                                                    |
|math_algebra                                             |✓    |   |✓   |         1187|acc                                                                                                                                                                              |
|math_counting_and_prob                                   |✓    |   |✓   |          474|acc                                                                                                                                                                              |
|math_geometry                                            |✓    |   |✓   |          479|acc                                                                                                                                                                              |
|math_intermediate_algebra                                |✓    |   |✓   |          903|acc                                                                                                                                                                              |
|math_num_theory                                          |✓    |   |✓   |          540|acc                                                                                                                                                                              |
|math_prealgebra                                          |✓    |   |✓   |          871|acc                                                                                                                                                                              |
|math_precalc                                             |✓    |   |✓   |          546|acc                                                                                                                                                                              |
|math_asdiv                                               |     |✓  |    |         2305|acc                                                                                                                                                                              |
|arithmetic_2da                                           |     |✓  |    |         2000|acc                                                                                                                                                                              |
|arithmetic_2ds                                           |     |✓  |    |         2000|acc                                                                                                                                                                              |
|arithmetic_3da                                           |     |✓  |    |         2000|acc                                                                                                                                                                              |
|arithmetic_3ds                                           |     |✓  |    |         2000|acc                                                                                                                                                                              |
|arithmetic_4da                                           |     |✓  |    |         2000|acc                                                                                                                                                                              |
|arithmetic_4ds                                           |     |✓  |    |         2000|acc                                                                                                                                                                              |
|arithmetic_5da                                           |     |✓  |    |         2000|acc                                                                                                                                                                              |
|arithmetic_5ds                                           |     |✓  |    |         2000|acc                                                                                                                                                                              |
|arithmetic_2dm                                           |     |✓  |    |         2000|acc                                                                                                                                                                              |
|arithmetic_1dc                                           |     |✓  |    |         2000|acc                                                                                                                                                                              |
|hendrycksTest-abstract_algebra                           |✓    |✓  |✓   |          100|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-anatomy                                    |✓    |✓  |✓   |          135|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-astronomy                                  |✓    |✓  |✓   |          152|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-business_ethics                            |✓    |✓  |✓   |          100|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-clinical_knowledge                         |✓    |✓  |✓   |          265|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-college_biology                            |✓    |✓  |✓   |          144|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-college_chemistry                          |✓    |✓  |✓   |          100|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-college_computer_science                   |✓    |✓  |✓   |          100|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-college_mathematics                        |✓    |✓  |✓   |          100|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-college_medicine                           |✓    |✓  |✓   |          173|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-college_physics                            |✓    |✓  |✓   |          102|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-computer_security                          |✓    |✓  |✓   |          100|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-conceptual_physics                         |✓    |✓  |✓   |          235|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-econometrics                               |✓    |✓  |✓   |          114|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-electrical_engineering                     |✓    |✓  |✓   |          145|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-elementary_mathematics                     |✓    |✓  |✓   |          378|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-formal_logic                               |✓    |✓  |✓   |          126|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-global_facts                               |✓    |✓  |✓   |          100|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_biology                        |✓    |✓  |✓   |          310|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_chemistry                      |✓    |✓  |✓   |          203|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_computer_science               |✓    |✓  |✓   |          100|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_european_history               |✓    |✓  |✓   |          165|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_geography                      |✓    |✓  |✓   |          198|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_government_and_politics        |✓    |✓  |✓   |          193|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_macroeconomics                 |✓    |✓  |✓   |          390|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_mathematics                    |✓    |✓  |✓   |          270|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_microeconomics                 |✓    |✓  |✓   |          238|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_physics                        |✓    |✓  |✓   |          151|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_psychology                     |✓    |✓  |✓   |          545|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_statistics                     |✓    |✓  |✓   |          216|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_us_history                     |✓    |✓  |✓   |          204|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-high_school_world_history                  |✓    |✓  |✓   |          237|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-human_aging                                |✓    |✓  |✓   |          223|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-human_sexuality                            |✓    |✓  |✓   |          131|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-international_law                          |✓    |✓  |✓   |          121|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-jurisprudence                              |✓    |✓  |✓   |          108|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-logical_fallacies                          |✓    |✓  |✓   |          163|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-machine_learning                           |✓    |✓  |✓   |          112|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-management                                 |✓    |✓  |✓   |          103|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-marketing                                  |✓    |✓  |✓   |          234|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-medical_genetics                           |✓    |✓  |✓   |          100|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-miscellaneous                              |✓    |✓  |✓   |          783|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-moral_disputes                             |✓    |✓  |✓   |          346|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-moral_scenarios                            |✓    |✓  |✓   |          895|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-nutrition                                  |✓    |✓  |✓   |          306|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-philosophy                                 |✓    |✓  |✓   |          311|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-prehistory                                 |✓    |✓  |✓   |          324|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-professional_accounting                    |✓    |✓  |✓   |          282|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-professional_law                           |✓    |✓  |✓   |         1534|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-professional_medicine                      |✓    |✓  |✓   |          272|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-professional_psychology                    |✓    |✓  |✓   |          612|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-public_relations                           |✓    |✓  |✓   |          110|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-security_studies                           |✓    |✓  |✓   |          245|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-sociology                                  |✓    |✓  |✓   |          201|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-us_foreign_policy                          |✓    |✓  |✓   |          100|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-virology                                   |✓    |✓  |✓   |          166|acc, acc_norm                                                                                                                                                                    |
|hendrycksTest-world_religions                            |✓    |✓  |✓   |          171|acc, acc_norm                                                                                                                                                                    |
|wmt14-en-fr                                              |     |   |✓   |         3003|bleu, chrf, ter                                                                                                                                                                  |
|wmt14-fr-en                                              |     |   |✓   |         3003|bleu, chrf, ter                                                                                                                                                                  |
|wmt16-en-ro                                              |     |   |✓   |         1999|bleu, chrf, ter                                                                                                                                                                  |
|wmt16-ro-en                                              |     |   |✓   |         1999|bleu, chrf, ter                                                                                                                                                                  |
|wmt16-de-en                                              |     |   |✓   |         2999|bleu, chrf, ter                                                                                                                                                                  |
|wmt16-en-de                                              |     |   |✓   |         2999|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-cs-en                                              |     |   |✓   |          664|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-de-en                                              |     |   |✓   |          785|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-de-fr                                              |     |   |✓   |         1619|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-en-cs                                              |     |   |✓   |         1418|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-en-de                                              |     |   |✓   |         1418|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-en-iu                                              |     |   |✓   |         2971|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-en-ja                                              |     |   |✓   |         1000|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-en-km                                              |     |   |✓   |         2320|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-en-pl                                              |     |   |✓   |         1000|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-en-ps                                              |     |   |✓   |         2719|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-en-ru                                              |     |   |✓   |         2002|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-en-ta                                              |     |   |✓   |         1000|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-en-zh                                              |     |   |✓   |         1418|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-fr-de                                              |     |   |✓   |         1619|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-iu-en                                              |     |   |✓   |         2971|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-ja-en                                              |     |   |✓   |          993|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-km-en                                              |     |   |✓   |         2320|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-pl-en                                              |     |   |✓   |         1001|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-ps-en                                              |     |   |✓   |         2719|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-ru-en                                              |     |   |✓   |          991|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-ta-en                                              |     |   |✓   |          997|bleu, chrf, ter                                                                                                                                                                  |
|wmt20-zh-en                                              |     |   |✓   |         2000|bleu, chrf, ter                                                                                                                                                                  |
|iwslt17-en-ar                                            |     |   |✓   |         1460|bleu, chrf, ter                                                                                                                                                                  |
|iwslt17-ar-en                                            |     |   |✓   |         1460|bleu, chrf, ter                                                                                                                                                                  |
|anagrams1                                                |     |✓  |    |        10000|acc                                                                                                                                                                              |
|anagrams2                                                |     |✓  |    |        10000|acc                                                                                                                                                                              |
|cycle_letters                                            |     |✓  |    |        10000|acc                                                                                                                                                                              |
|random_insertion                                         |     |✓  |    |        10000|acc                                                                                                                                                                              |
|reversed_words                                           |     |✓  |    |        10000|acc                                                                                                                                                                              |
|pile_arxiv                                               |     |✓  |✓   |         2407|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_books3                                              |     |✓  |✓   |          269|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_bookcorpus2                                         |     |✓  |✓   |           28|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_dm-mathematics                                      |     |✓  |✓   |         1922|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_enron                                               |     |✓  |✓   |         1010|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_europarl                                            |     |✓  |✓   |          157|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_freelaw                                             |     |✓  |✓   |         5101|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_github                                              |     |✓  |✓   |        18195|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_gutenberg                                           |     |✓  |✓   |           80|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_hackernews                                          |     |✓  |✓   |         1632|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_nih-exporter                                        |     |✓  |✓   |         1884|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_opensubtitles                                       |     |✓  |✓   |          642|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_openwebtext2                                        |     |✓  |✓   |        32925|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_philpapers                                          |     |✓  |✓   |           68|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_pile-cc                                             |     |✓  |✓   |        52790|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_pubmed-abstracts                                    |     |✓  |✓   |        29895|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_pubmed-central                                      |     |✓  |✓   |         5911|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_stackexchange                                       |     |✓  |✓   |        30378|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_uspto                                               |     |✓  |✓   |        11415|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_ubuntu-irc                                          |     |✓  |✓   |           22|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_wikipedia                                           |     |✓  |✓   |        17511|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|pile_youtubesubtitles                                    |     |✓  |✓   |          342|word_perplexity, byte_perplexity, bits_per_byte                                                                                                                                  |
|blimp_adjunct_island                                     |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_anaphor_gender_agreement                           |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_anaphor_number_agreement                           |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_animate_subject_passive                            |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_animate_subject_trans                              |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_causative                                          |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_complex_NP_island                                  |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_coordinate_structure_constraint_complex_left_branch|     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_coordinate_structure_constraint_object_extraction  |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_determiner_noun_agreement_1                        |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_determiner_noun_agreement_2                        |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_determiner_noun_agreement_irregular_1              |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_determiner_noun_agreement_irregular_2              |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_determiner_noun_agreement_with_adj_2               |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_determiner_noun_agreement_with_adj_irregular_1     |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_determiner_noun_agreement_with_adj_irregular_2     |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_determiner_noun_agreement_with_adjective_1         |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_distractor_agreement_relational_noun               |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_distractor_agreement_relative_clause               |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_drop_argument                                      |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_ellipsis_n_bar_1                                   |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_ellipsis_n_bar_2                                   |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_existential_there_object_raising                   |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_existential_there_quantifiers_1                    |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_existential_there_quantifiers_2                    |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_existential_there_subject_raising                  |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_expletive_it_object_raising                        |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_inchoative                                         |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_intransitive                                       |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_irregular_past_participle_adjectives               |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_irregular_past_participle_verbs                    |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_irregular_plural_subject_verb_agreement_1          |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_irregular_plural_subject_verb_agreement_2          |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_left_branch_island_echo_question                   |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_left_branch_island_simple_question                 |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_matrix_question_npi_licensor_present               |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_npi_present_1                                      |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_npi_present_2                                      |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_only_npi_licensor_present                          |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_only_npi_scope                                     |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_passive_1                                          |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_passive_2                                          |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_principle_A_c_command                              |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_principle_A_case_1                                 |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_principle_A_case_2                                 |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_principle_A_domain_1                               |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_principle_A_domain_2                               |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_principle_A_domain_3                               |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_principle_A_reconstruction                         |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_regular_plural_subject_verb_agreement_1            |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_regular_plural_subject_verb_agreement_2            |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_sentential_negation_npi_licensor_present           |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_sentential_negation_npi_scope                      |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_sentential_subject_island                          |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_superlative_quantifiers_1                          |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_superlative_quantifiers_2                          |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_tough_vs_raising_1                                 |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_tough_vs_raising_2                                 |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_transitive                                         |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_wh_island                                          |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_wh_questions_object_gap                            |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_wh_questions_subject_gap                           |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_wh_questions_subject_gap_long_distance             |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_wh_vs_that_no_gap                                  |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_wh_vs_that_no_gap_long_distance                    |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_wh_vs_that_with_gap                                |     |✓  |    |         1000|acc                                                                                                                                                                              |
|blimp_wh_vs_that_with_gap_long_distance                  |     |✓  |    |         1000|acc                                                                                                                                                                              |


## Usage

### Evaluate a task

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most importantly, the `gpt2` model can be used to load an arbitrary HuggingFace model as follows:


```bash
python main.py \
	--model gpt2 \
	--model_args pretrained=EleutherAI/gpt-neo-1.3B \
	--device cuda:0 \
	--tasks lambada,hellaswag \
	--num_fewshot 2
```

To inspect what the LM inputs look like, you can run the following command:

```bash
python write_out.py \
	--tasks all_tasks \
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

### Task Versioning 

To help improve reproducibility, all tasks have a VERSION field. When run from the command line, this is reported in a column in the table, or in the "version" field in the evaluator return dict. The purpose of the version is so that if the task definition changes (i.e to fix a bug), then we can know exactly which metrics were computed using the old buggy implementation to avoid unfair comparisons. To enforce this, there are unit tests that make sure the behavior of all tests remains the same as when they were first implemented. Task versions start at 0, and each time a breaking change is made, the version is incremented by one. 

When reporting eval harness results, please also report the version of each task. This can be done either with a separate column in the table, or by reporting the task name with the version appended as such: taskname-v0.

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
