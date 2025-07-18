# Task-name

### Paper

Title: `LIBRA: Long Input Benchmark for Russian Analysis`

Abstract: `Datasets for proper evaluation of long-context understanding in Russian. For the Russian language LIBRA comprises 21 adapted datasets to study the LLM's abilities to understand long texts thoroughly. The tests are divided into four complexity groups and allow the evaluation of models across various context lengths ranging from 4k up to 128k tokens.`

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

### Groups, Tags, and Tasks

#### Groups

* `libra_simple_information_retrieval`
* `libra_question_answering_and_multiple_choice`
* `libra_multi_hop_question_answering`
* `libra_complex_reasoning_and_mathematical_problems`

#### Tags

* `libra`

#### Tasks

* `passkey`
* `passkey_with_librusec`
* `matreshka_yes_no`
* `matreshka_names`
* `librusec_history`
* `ru_trec`
* `ru_sci_abstract_retrieval`
* `ru_sci_fi`
* `ru_quality`
* `ru_tpo`
* `ru_babilong_qa1`
* `ru_babilong_qa2`
* `ru_babilong_qa3`
* `ru_babilong_qa4`
* `ru_babilong_qa5`
* `long_context_multiq`
* `librusec_mhqa`
* `ru_2wikimultihopqa`
* `ru_sci_passage_count`
* `ru_qasper`
* `ru_gsm100`


# Variants

Usage (**all with `--apply_chat_template`**):
```
lm_eval --model hf --model_args pretrained=EleutherAI/gpt-j-6B --tasks libra_simple_information_retrieval --device cpu --apply_chat_template --log_samples --output_path test_libra_simple_information_retrieval
```

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
