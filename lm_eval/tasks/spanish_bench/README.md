# SpanishBench

### Paper

SpanishBench is a benchmark for evaluating language models in Spanish tasks. This is, it evaluates the ability of a language model to understand and generate Spanish text. SpanishBench offers a combination of pre-existing, open datasets. All the details of SpanishBench will be published in a paper soon.

The datasets included in SpanishBench are:

| Task          | Category       | Paper title          | Homepage  |
|:-------------:|:-----:|:-------------:|:-----:|
| Belebele_es | Reading Comprehension | [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants](https://arxiv.org/abs/2308.16884) | https://huggingface.co/datasets/facebook/belebele |
| FLORES_es | Translation | [The FLORES-101  Evaluation Benchmark for Low-Resource and Multilingual Machine Translation](https://arxiv.org/abs/2106.03193) | https://huggingface.co/datasets/facebook/flores |
| MGSM_es | Math | [Language Models are Multilingual Chain-of-Thought Reasoners](https://arxiv.org/abs/2210.03057) | https://huggingface.co/datasets/juletxara/mgsm |
| PAWS-X_es | Paraphrasing | [PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification](https://aclanthology.org/D19-1382/) | https://huggingface.co/datasets/google-research-datasets/paws-x |
| WNLI-es | Natural Language Inference | No paper. | https://huggingface.co/datasets/PlanTL-GOB-ES/wnli-es |
| XL-Sum_es | Summarization | [XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages](https://aclanthology.org/2021.findings-acl.413/) | https://huggingface.co/datasets/csebuetnlp/xlsum |
| XNLI_es | Natural Language Inference | [XNLI: Evaluating Cross-lingual Sentence Representations](https://aclanthology.org/D18-1269/) | https://huggingface.co/datasets/facebook/xnli |
| XQuAD_es | Question Answering | [On the Cross-lingual Transferability of Monolingual Representations](https://aclanthology.org/2020.acl-main.421/) | https://huggingface.co/datasets/google/xquad |
| XStoryCloze_es | Commonsense Reasoning | [Few-shot Learning with Multilingual Generative Language Models](https://aclanthology.org/2022.emnlp-main.616/) | https://huggingface.co/datasets/juletxara/xstory_cloze |


### Citation
Paper for SpanishBench coming soon.

### Groups and Tasks

#### Groups

- `spanish_bench`: All tasks included in SpanishBench.
- `flores_es`: All FLORES translation tasks from or to Spanish.

#### Tags
- `phrases_es`: Two Phrases_va tasks for language adaptation between Spanish and Valencian.

#### Tasks

The following tasks evaluate tasks on SpanishBench dataset using various scoring methods.
  - `belebele_spa_Latn`
  - `flores_es`
  - `flores_es-ca`
  - `flores_es-de`
  - `flores_es-en`
  - `flores_es-eu`
  - `flores_es-fr`
  - `flores_es-gl`
  - `flores_es-it`
  - `flores_es-pt`
  - `flores_ca-es`
  - `flores_de-es`
  - `flores_en-es`
  - `flores_eu-es`
  - `flores_fr-es`
  - `flores_gl-es`
  - `flores_it-es`
  - `flores_pt-es`
  - `mgsm_direct_es_v2` (`v2` is due to an existing open issue in the original task)
  - `paws_es`
  - `phrases_es`
  - `wnli_es`
  - `xlsum_es`
  - `xnli_es`
  - `xquad_es`
  - `xstorycloze_es`

Some of these tasks are taken from benchmarks already available in LM Evaluation Harness. These are:
- `belebele_spa_Latn`: Belebele Spanish
- `mgsm_direct_es`: MGSM Spanish (We fix an existing open issue in the original task)
- `paws_es`: PAWS-X Spanish
- `xnli_es`: XNLI Spanish
- `xstorycloze_es`: XStoryCloze Spanish

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation?
    * [ ] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
