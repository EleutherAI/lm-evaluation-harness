# HPLT-e: Czech

## Paper

Title: `HPLT 3.0: Very Large-Scale Multilingual Resources for LLMs and MT. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models`

Abstract: https://aclanthology.org/2026.lrec-1.110/

Czech part of [HPLT-e](../README.md), a multilingual, multi-prompt evaluation suite. Only the Czech Global-MMLU prompts written for HPLT-e are included here; the original Global-MMLU implementation is [`global_mmlu`](../../global_mmlu/README.md). The rest of the Czech part of HPLT-e is [BenCzechMark](https://github.com/DCGM/lm-evaluation-harness/tree/main/lm_eval/tasks/benczechmark) ([Fajcik et al., TACL 2025](https://aclanthology.org/2025.tacl-1.50/)), which is not yet part of LM Evaluation Harness; the version used in HPLT-e, updated for recent LM Evaluation Harness releases, is in the [HPLT-e repository](https://github.com/hplt-project/hplt-e/tree/main/ces_Latn).

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

@article{fajcik-etal-2025-benczechmark,
    title = "{B}en{C}zech{M}ark: A {C}zech-Centric Multitask and Multimetric Benchmark for Large Language Models with Duel Scoring Mechanism",
    author = "Fajcik, Martin  and
      Docekal, Martin  and
      Dolezal, Jan  and
      Ondrej, Karel  and
      Bene{\v{s}}, Karel  and
      Kapsa, Jan  and
      Smrz, Pavel  and
      Polok, Alexander  and
      Hradis, Michal  and
      Neverilova, Zuzana  and
      Horak, Ales  and
      Sabol, Radoslav  and
      Stefanik, Michal  and
      Jirkovsky, Adam  and
      Adamczyk, David  and
      Hyner, Petr  and
      Hula, Jan  and
      Kydlicek, Hynek",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "13",
    year = "2025",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2025.tacl-1.50/",
    doi = "10.1162/tacl.a.32",
    pages = "1068--1095",
    abstract = "We present BenCzechMark (BCM), the first comprehensive Czech language benchmark designed for large language models, offering diverse tasks, multiple task formats, and multiple evaluation metrics. Its duel scoring system is grounded in statistical significance theory and uses aggregation across tasks inspired by social preference theory. Our benchmark encompasses 50 challenging tasks, with corresponding test datasets, primarily in native Czech, with 14 newly collected ones. These tasks span 8 categories and cover diverse domains, including historical Czech news, essays from pupils or language learners, and spoken word. Furthermore, we collect and clean BUT-Large Czech Collection, the largest publicly available clean Czech language corpus, and use it for (i) contamination analysis and (ii) continuous pretraining of the first Czech-centric 7B language model with Czech-specific tokenization. We use our model as a baseline for comparison with publicly available multilingual models. Lastly, we release and maintain a leaderboard with existing 50 model submissions, where new model submissions can be made at \url{https://huggingface.co/spaces/CZLC/BenCzechMark}."
}

```

### Groups, Tags, and Tasks

#### Groups

* `hplte_ces_global_mmlu_p[0-2]`: Global-MMLU with prompt `p<N>`; the accuracy is averaged over the six subject categories, weighted by size (subtasks `hplte_ces_global_mmlu_<category>_p<N>`, with category in `business`, `humanities`, `medical`, `other`, `social_sciences`, `stem`). The original single-prompt Global-MMLU implementation is [`global_mmlu`](../../global_mmlu/README.md).

#### Tags

None: the Global-MMLU prompts are groups (see above).

#### Tasks

`_p[0-2]` stands for the prompt variants of each task, e.g. `hplte_ces_global_mmlu_p0`. [`task_mapping.json`](task_mapping.json) lists every task with its name in the HPLT-e repository and its prompt.

| Dataset | Task | Task type | Task category | Name in the HPLT-e repository |
|:---|:---|:---|:---|:---|
| [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) | `hplte_ces_global_mmlu_p[0-2]` | Multiple-choice QA | Language-specific & world knowledge | `global_mmlu_czech_p[0-2]` |

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
