# LIBRA: Long Input Benchmark for Russian Analysis

### Paper

Title: `LIBRA: Long Input Benchmark for Russian Analysis`

Abstract: `Datasets for proper evaluation of long-context understanding in Russian. LIBRA comprises 18 adapted datasets to study the LLM's abilities to understand long texts in Russian. The tasks are divided into four complexity groups and allow evaluation across various context lengths ranging from 4k up to 512k tokens.`

Homepage: `https://huggingface.co/datasets/ai-forever/LIBRA`

### Citation

```
@misc{churin2024longinputbenchmarkrussian,
      title={Long Input Benchmark for Russian Analysis},
      author={Igor Churin and Murat Apishev and Maria Tikhonova and Denis Shevelev and Aydar Bulatov and Yuri Kuratov and Sergei Averkiev and Alena Fenogenova},
      year={2024},
      eprint={2408.02439},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.02439},
}
```

### LIBRA Mini

Running a full LIBRA evaluation can be prohibitively expensive and time-consuming due to the large number of datasets and long context lengths involved. Moreover, some of the included tasks have become less informative as benchmarks, with modern models achieving near-saturated scores on them.

To address this, we introduce **LIBRA Mini** — a compact, curated subset of 6 datasets selected from the full benchmark. These datasets represent the most challenging and diagnostically informative tasks in LIBRA, covering diverse task types and complexity levels:

- **matreshka_names** (Group II) — identifying persons in dialogues based on discussed topics
- **librusec_mhqa** (Group III) — multi-hop QA with information spread across multiple text parts
- **long_context_multiq** (Group III) — multi-hop QA based on Wikidata and Wikipedia
- **ru_2wikimultihopqa** (Group III) — multi-hop reasoning across multiple Wikipedia articles
- **ru_babilong_qa3** (Group III) — multi-fact reasoning over long contexts
- **ru_sci_passage_count** (Group IV) — counting unique paragraphs in extended scientific texts

We recommend **using LIBRA Mini as the primary evaluation suite for model comparisons**, while the full LIBRA benchmark remains available for comprehensive analysis. Full details are available on the [LIBRA dataset card](https://huggingface.co/datasets/ai-forever/LIBRA).

### Groups, Tags, and Tasks

#### Groups

* `libra` — full LIBRA benchmark (all 18 tasks across 4 complexity groups)
* `libra_mini` — compact subset of 6 most challenging and informative tasks (recommended for model comparisons)
* `libra_simple_information_retrieval` — Group I
* `libra_question_answering_and_multiple_choice` — Group II
* `libra_multi_hop_question_answering` — Group III
* `libra_complex_reasoning_and_mathematical_problems` — Group IV

#### Tags

* `libra`

#### Tasks

**Group I — Simple Information Retrieval (sanity check)**
* `passkey`
* `passkey_with_librusec`

**Group II — Question Answering and Multiple Choice**
* `matreshka_names` !
* `matreshka_yes_no`
* `librusec_history`
* `ru_sci_abstract_retrieval`
* `ru_sci_fi`
* `ru_quality`
* `ru_tpo`

**Group III — Multi-hop Question Answering**
* `ru_babilong_qa1`
* `ru_babilong_qa2`
* `ru_babilong_qa3` !
* `ru_babilong_qa4`
* `ru_babilong_qa5`
* `long_context_multiq` !
* `librusec_mhqa` !
* `ru_2wikimultihopqa` !

**Group IV — Complex Reasoning and Mathematical Problems**
* `ru_sci_passage_count` !

! — included in **LIBRA Mini**

### Usage

All tasks should be run with `--apply_chat_template`. We recommend using vLLM for efficient inference over long contexts.

Run the full LIBRA benchmark:

```bash
lm_eval --model vllm \
        --model_args pretrained=Qwen/Qwen3-30B-A3B,max_model_len=262144 \
        --tasks libra \
        --apply_chat_template \
        --log_samples \
        --output_path output_libra
```

Run LIBRA Mini only (recommended):

```bash
lm_eval --model vllm \
        --model_args pretrained=Qwen/Qwen3-30B-A3B,max_model_len=262144 \
        --tasks libra_mini \
        --apply_chat_template \
        --log_samples \
        --output_path output_libra_mini
```

Run a specific complexity group:

```bash
lm_eval --model vllm \
        --model_args pretrained=Qwen/Qwen3-30B-A3B,max_model_len=262144 \
        --tasks libra_multi_hop_question_answering \
        --apply_chat_template \
        --log_samples \
        --output_path output_libra_group3
```

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
