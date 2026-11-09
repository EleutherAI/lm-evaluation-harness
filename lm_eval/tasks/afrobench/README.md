# AfroBench

### Paper

Title: `AfroBench: How Good are Large Language Models on African Languages?`

Paper Link: https://arxiv.org/abs/2311.07978

## Abstract
> Large-scale multilingual evaluations, such as MEGA, often include only a handful of African languages due to the scarcity of high-quality evaluation data and the limited discoverability of existing African datasets. This lack of representation hinders comprehensive LLM evaluation across a diverse range of languages and tasks. To address these challenges, we introduce AfroBench -- a multi-task benchmark for evaluating the performance of LLMs across 64 African languages, 15 tasks and 22 datasets. AfroBench consists of nine natural language understanding datasets, six text generation datasets, six knowledge and question answering tasks, and one mathematical reasoning task. We present results comparing the performance of prompting LLMs to fine-tuned baselines based on BERT and T5-style models. Our results suggest large gaps in performance between high-resource languages, such as English, and African languages across most tasks; but performance also varies based on the availability of monolingual data resources. Our findings confirm that performance on African languages continues to remain a hurdle for current LLMs, underscoring the need for additional efforts to close this gap.

HomePage: https://mcgill-nlp.github.io/AfroBench/

### Groups, and Tasks
#### Groups
* `afrobench` : Runs all that tasks, datasets and prompts in this benchmark
* `afrobench_lite`: Runs the lite version of the benchmark which includes; afrimgsm, afrimmlu, afrixnli, sib, intent, adr and flores

Dataset specific grouping that listing all prompts, allowing users to review or edit them.
* `adr`   `afrihate`   `afrisenti`   `belebele`  `african_flores` `injongointent`  `mafand`  `masakhaner`  `masakhapos`  `naijarc`  `nollysenti`  `african_ntrex`  `openai_mmlu`  `salt`  `sib`  `uhura`  `xlsum`


#### Task Tags
* `adr_tasks`: all datasets in this benchmark relating to Automatic Diacritics Restoration task
* `afrihate_tasks`: all datasets in this benchmark relating to Hate Speech detection task
* `afrimgsm_tasks`: all datasets in this benchmark relating to Mathematical reasoning task
* `afrixnli_tasks`: all datasets in this benchmark relating to Natural Language Inference task
* `afrobench_xqa_tasks`: all datasets in this benchmark relating to Crosslingual QA (XQA) task
* `afrobench_sentiment_tasks`: all datasets in this benchmark relating to Sentiment Classification task
* `afrobench_MT_tasks`: all datasets in this benchmark relating to Machine Translation task
* `afrobench_TC_tasks`: all datasets in this benchmark relating to Topic Classification task
* `afrobench_mmlu_tasks`: all datasets in this benchmark relating to MMLU task
* `injongointent_tasks`: all datasets in this benchmark relating to Intent Detection task
* `masakhaner_tasks`: all datasets in this benchmark relating to Named Entity Recognition (NER) task
* `masakhapos_tasks`: all datasets in this benchmark relating to Part of Speech Tagging (POS) task
* `RC_tasks`: all datasets in this benchmark relating to Reading Comprehension task
* `uhura_arc_easy_tasks`: all datasets in this benchmark relating to Arc-Easy (XQA) task
* `xlsum_tasks`: all datasets in this benchmark relating to Summarization task


We've included sample run scripts for easier integration with the benchmark: [sample run scripts](./sample_run_scripts)

For better understanding of the run interface see [interface.md](../../../docs/interface.md)

All dataset used in this benchmark are available at [huggingface](https://huggingface.co/collections/masakhane/afrobench-67dbf553ebf5701c2207f883)

### Citation

```
@misc{ojo2025afrobenchgoodlargelanguage,
      title={AfroBench: How Good are Large Language Models on African Languages?},
      author={Jessica Ojo and Odunayo Ogundepo and Akintunde Oladipo and Kelechi Ogueji and Jimmy Lin and Pontus Stenetorp and David Ifeoluwa Adelani},
      year={2025},
      eprint={2311.07978},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.07978},
}
```
Please cite datasets used. Citations for individual datasets are included in their respective repository readme files within this benchmark.
### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test? The original paper doesn't have an associated implementation, but there is an official entry in [BigBench](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/social_iqa). I use the same prompting format as BigBench.


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
