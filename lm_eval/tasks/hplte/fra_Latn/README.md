# HPLT-e: French

## Paper

Title: `HPLT 3.0: Very Large-Scale Multilingual Resources for LLMs and MT. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models`

Abstract: https://aclanthology.org/2026.lrec-1.110/

French part of [HPLT-e](../README.md), a multilingual, multi-prompt evaluation suite. It contains multi-prompt versions of the FrenchBench tasks, introduced in [CroissantLLM: A Truly Bilingual French-English Language Model](https://openreview.net/forum?id=uA19Xo1o31) (Faysse et al., TMLR 2025), and of further French datasets used in HPLT-e, with prompts written by native speakers. The original single-prompt FrenchBench implementation is [`french_bench`](../../french_bench/README.md).

Homepage: https://github.com/hplt-project/hplt-e

### Citation

```text
@inproceedings{oepen-etal-2026-hplt,
    title = "{HPLT} 3.0: Very Large-Scale Multilingual Resources for {LLM}s and {MT}. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models",
    author = "Oepen, Stephan  and
      Arefyev, Nikolay  and
      Aulamo, Mikko  and
      Ba{\~n}{\'o}n, Marta  and
      Buljan, Maja  and
      Burchell, Laurie V.  and
      Charpentier, Lucas Georges Gabriel  and
      Chen, Pinzhen  and
      Fedorova, Mariia  and
      de Gibert, Ona  and
      Haddow, Barry  and
      Haji{\v{c}}, Jan  and
      Helcl, Jindrich  and
      Kutuzov, Andrey  and
      Laippala, Veronika  and
      Li, Zihao  and
      Malik, Bhavitvya  and
      Mikhailov, Vladislav  and
      Myntti, Amanda  and
      O{'}Brien, Dayy{\'a}n  and
      Polakova, Lucie  and
      Ram{\'i}rez-S{\'a}nchez, Gema  and
      Siewert, Janine  and
      Stepachev, Pavel  and
      Tiedemann, Joerg  and
      Vahtola, Teemu  and
      Varis, Dusan  and
      Vitiugin, Fedor  and
      Zaragoza, Jaume",
    editor = "Piperidis, Stelios  and
      Bel, N{\'u}ria  and
      van den Heuvel, Henk  and
      Ide, Nancy  and
      Krek, Simon  and
      Toral, Antonio",
    booktitle = "Proceedings of the Fifteenth Language Resources and Evaluation Conference",
    month = may,
    year = "2026",
    address = "Palma de Mallorca, Spain",
    publisher = "ELRA Language Resource Association",
    url = "https://aclanthology.org/2026.lrec-1.110/",
    doi = "10.63317/25xbdofco9od",
    pages = "1409--1434",
    abstract = "We present an ongoing initiative to provide open, very large, high-quality, and richly annotated textual datasets for almost 200 languages. At 30 trillion tokens, this is likely the largest generally available multilingual collection of LLM pre-training data. These datasets are derived from web crawls from different sources and accompanied with a complete, open-source pipeline for document selection from web archives, text extraction from HTML, language identification for noisy texts, exact and near-deduplication, annotation with, among others, register labels, text quality estimates, and personally identifiable information; and final selection and filtering. We report on data quality probes through contrastive and analytical statistics, through manual inspection of samples for some 20 languages, and through end-to-end evaluation of various language model architectures trained on this data. For multilingual LLM evaluation, we provide a comprehensive collection of benchmarks for nine European languages, with special emphasis on natively created tasks, mechanisms to mitigate prompt sensitivity, and refined normalization and aggregation of scores. Additionally, we train and evaluate a family of 57 monolingual encoder{--}decoder models, as well as about 30 ``smallish'' monolingual GPT-like reference models. Besides the monolingual data and models, we also present a very large collection of parallel texts automatically mined from this data, together with a novel parallel corpus synthesized via machine translation."
}

@article{
faysse2025croissantllm,
title={Croissant{LLM}: A Truly Bilingual French-English Language Model},
author={Manuel Faysse and Patrick Fernandes and Nuno M Guerreiro and Ant{\'o}nio Loison and Duarte Miguel Alves and Caio Corro and Nicolas Boizard and Jo{\~a}o Alves and Ricardo Rei and Pedro Henrique Martins and Antoni Bigata Casademunt and Fran{\c{c}}ois Yvon and Andre Martins and Gautier Viaud and CELINE HUDELOT and Pierre Colombo},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=uA19Xo1o31},
note={}
}

@inproceedings{bandarkar-etal-2024-belebele,
    title = "The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants",
    author = "Bandarkar, Lucas  and
      Liang, Davis  and
      Muller, Benjamin  and
      Artetxe, Mikel  and
      Shukla, Satya Narayan  and
      Husa, Donald  and
      Goyal, Naman  and
      Krishnan, Abhinandan  and
      Zettlemoyer, Luke  and
      Khabsa, Madian",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.44/",
    doi = "10.18653/v1/2024.acl-long.44",
    pages = "749--775",
    abstract = "We present Belebele, a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants. Significantly expanding the language coverage of natural language understanding (NLU) benchmarks, this dataset enables the evaluation of text models in high-, medium-, and low-resource languages. Each question is based on a short passage from the FLORES-200 dataset and has four multiple-choice answers. The questions were carefully curated to discriminate between models with different levels of general language comprehension. The English dataset on its own proves difficult enough to challenge state-of-the-art language models. Being fully parallel, this dataset enables direct comparison of model performance across all languages. We use this dataset to evaluate the capabilities of multilingual masked language models (MLMs) and large language models (LLMs). We present extensive results and findings, notably that despite significant cross-lingual transfer in English-centric LLMs, much smaller MLMs pretrained on balanced multilingual data still understand far more languages. Overall, Belebele opens up new avenues for evaluating and analyzing the multilingual capabilities of NLP systems."
}

@inproceedings{singh-etal-2025-global,
    title = "Global {MMLU}: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation",
    author = "Singh, Shivalika  and
      Romanou, Angelika  and
      Fourrier, Cl{\'e}mentine  and
      Adelani, David Ifeoluwa  and
      Ngui, Jian Gang  and
      Vila-Suero, Daniel  and
      Limkonchotiwat, Peerat  and
      Marchisio, Kelly  and
      Leong, Wei Qi  and
      Susanto, Yosephine  and
      Ng, Raymond  and
      Longpre, Shayne  and
      Ruder, Sebastian  and
      Ko, Wei-Yin  and
      Bosselut, Antoine  and
      Oh, Alice  and
      Martins, Andre  and
      Choshen, Leshem  and
      Ippolito, Daphne  and
      Ferrante, Enzo  and
      Fadaee, Marzieh  and
      Ermis, Beyza  and
      Hooker, Sara",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.919/",
    doi = "10.18653/v1/2025.acl-long.919",
    pages = "18761--18799",
    ISBN = "979-8-89176-251-0",
    abstract = "Reliable multilingual evaluation is difficult, and culturally appropriate evaluation is even harder to achieve.A common practice to fill this gap is to machine-translate English evaluation sets. However, translation introduces language bias and carries over cultural and regional assumptions from the original questions {--} often testing knowledge irrelevant to the target audience. In this work, we highlight the extent and impact of these biases and present a multilingual evaluation framework that aims to mitigate them through improved translations and annotation practices.Through a large-scale study involving professional and community translators and annotators, we show that state-of-the-art models excel primarily by learning Western-centric concepts. Notably, we find that model rankings on the full MMLU change when evaluated on a subset of questions explicitly marked as culturally sensitive.We release Global MMLU, a multilingual extension of MMLU across 42 languages, featuring improved translation quality, expanded language coverage, and designated subsets labeled as culturally sensitive and culturally agnostic to enable a more comprehensive and equitable benchmark for evaluating language models across diverse linguistic and cultural contexts."
}

@inproceedings{
romanou2025include,
title={{INCLUDE}: Evaluating Multilingual Language Understanding with Regional Knowledge},
author={Angelika Romanou and Negar Foroutan and Anna Sotnikova and Sree Harsha Nelaturu and Shivalika Singh and Rishabh Maheshwary and Micol Altomare and Zeming Chen and Mohamed A. Haggag and Snegha A and Alfonso Amayuelas and Azril Hafizi Amirudin and Danylo Boiko and Michael Chang and Jenny Chim and Gal Cohen and Aditya Kumar Dalmia and Abraham Diress and Sharad Duwal and Daniil Dzenhaliou and Daniel Fernando Erazo Florez and Fabian Farestam and Joseph Marvin Imperial and Shayekh Bin Islam and Perttu Isotalo and Maral Jabbarishiviari and B{\"o}rje F. Karlsson and Eldar Khalilov and Christopher Klamm and Fajri Koto and Dominik Krzemi{\'n}ski and Gabriel Adriano de Melo and Syrielle Montariol and Yiyang Nan and Joel Niklaus and Jekaterina Novikova and Johan Samir Obando Ceron and Debjit Paul and Esther Ploeger and Jebish Purbey and Swati Rajwal and Selvan Sunitha Ravi and Sara Rydell and Roshan Santhosh and Drishti Sharma and Marjana Prifti Skenduli and Arshia Soltani Moakhar and Bardia soltani moakhar and Ayush Kumar Tarun and Azmine Toushik Wasi and Thenuka Ovin Weerasinghe and Serhan Yilmaz and Mike Zhang and Imanol Schlag and Marzieh Fadaee and Sara Hooker and Antoine Bosselut},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=k3gCieTXeY}
}

```

### Groups, Tags, and Tasks

#### Groups

* `hplte_fra_global_mmlu_p[0-2]`: Global-MMLU with prompt `p<N>`; the accuracy is averaged over the six subject categories, weighted by size (subtasks `hplte_fra_global_mmlu_<category>_p<N>`, with category in `business`, `humanities`, `medical`, `other`, `social_sciences`, `stem`). The original single-prompt Global-MMLU implementation is [`global_mmlu`](../../global_mmlu/README.md).

#### Tags

None.

#### Tasks

`_p[0-2]` stands for the prompt variants of each task, e.g. `hplte_fra_belebele_p0`. [`task_mapping.json`](task_mapping.json) lists every task with its name in the HPLT-e repository and its prompt.

| Dataset | Task | Task type | Task category | Name in the HPLT-e repository |
|:---|:---|:---|:---|:---|
| [Belebele](https://huggingface.co/datasets/facebook/belebele) | `hplte_fra_belebele_p[0-2]` | Multiple-choice QA | Reading comprehension | `frabelebele_p[0-2]` |
| [FQuAD](https://huggingface.co/datasets/manu/fquad2_test) | `hplte_fra_fquad_p[0-2]` 😎 | Generative QA | Reading comprehension | `fquad_p[0-2]` |
| [French Language Test: Grammar](https://huggingface.co/datasets/manu/french-bench-grammar-vocab-reading) | `hplte_fra_grammar_p[0-2]` | Multiple-choice QA | Language-specific & world knowledge | `french_bench_grammar_p[0-2]` |
| [French Language Test: Reading](https://huggingface.co/datasets/manu/french-bench-grammar-vocab-reading) | `hplte_fra_reading_p[0-2]` | Multiple-choice QA | Reading comprehension | `french_bench_reading_p[0-2]` |
| [French Language Test: Vocabulary](https://huggingface.co/datasets/manu/french-bench-grammar-vocab-reading) | `hplte_fra_vocabulary_p[0-2]` 😎 | Multiple-choice QA | Language-specific & world knowledge | `french_bench_vocabulary_p[0-2]` |
| [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) | `hplte_fra_global_mmlu_p[0-2]` | Multiple-choice QA | Language-specific & world knowledge | `global_mmlu_french_p[0-2]` |
| [INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44) | `hplte_fra_include_p[0-2]` | Multiple-choice QA | Language-specific & world knowledge | `include_french_p[0-2]` |
| [Topic-based NLI](https://huggingface.co/datasets/manu/topic_based_nli_test) | `hplte_fra_topic_based_nli_p[0-2]` | Text classification | Entailment | `topic_based_nli_p[0-2]` |
| [XNLI (fr)](https://huggingface.co/datasets/facebook/xnli) | `hplte_fra_xnli_p[0-2]` | Text classification | Entailment | `french_xnli_p[0-2]` |

😎 HPLT-e task selection for this language (see the [top-level README](../README.md#hplt-e-task-selection)).

All datasets are pinned to a Hugging Face Hub commit with `dataset_kwargs.revision`.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * The reference implementation is the [HPLT-e repository](https://github.com/hplt-project/hplt-e); see [Verification](../README.md#verification).

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
  * There is no main prompt: HPLT-e reports the maximum score over the prompts.
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
  * HPLT 3.0 (Oepen et al., LREC 2026).

### Changelog

* v1.0: initial release.
