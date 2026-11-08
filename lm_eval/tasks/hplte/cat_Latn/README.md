# HPLT-e: Catalan

## Paper

Title: `HPLT 3.0: Very Large-Scale Multilingual Resources for LLMs and MT. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models`

Abstract: https://aclanthology.org/2026.lrec-1.110/

Catalan part of [HPLT-e](../README.md), a multilingual, multi-prompt evaluation suite. It contains multi-prompt versions of the CatalanBench tasks, introduced in [IberoBench: A Benchmark for LLM Evaluation in Iberian Languages](https://aclanthology.org/2025.coling-main.699/) (Baucells et al., COLING 2025), and of further Catalan datasets used in HPLT-e, with prompts written by native speakers. The original single-prompt CatalanBench implementation is [`catalan_bench`](../../catalan_bench/README.md).

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

@inproceedings{baucells-etal-2025-iberobench,
    title = "{I}bero{B}ench: A Benchmark for {LLM} Evaluation in {I}berian Languages",
    author = "Baucells, Irene  and
      Aula-Blasco, Javier  and
      de-Dios-Flores, Iria  and
      Paniagua Su{\'a}rez, Silvia  and
      Perez, Naiara  and
      Salles, Anna  and
      Sotelo Docio, Susana  and
      Falc{\~a}o, J{\'u}lia  and
      Saiz, Jose Javier  and
      Sepulveda Torres, Robiert  and
      Barnes, Jeremy  and
      Gamallo, Pablo  and
      Gonzalez-Agirre, Aitor  and
      Rigau, German  and
      Villegas, Marta",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.699/",
    pages = "10491--10519",
    abstract = "The current best practice to measure the performance of base Large Language Models is to establish a multi-task benchmark that covers a range of capabilities of interest. Currently, however, such benchmarks are only available in a few high-resource languages. To address this situation, we present IberoBench, a multilingual, multi-task benchmark for Iberian languages (i.e., Basque, Catalan, Galician, European Spanish and European Portuguese) built on the LM Evaluation Harness framework. The benchmark consists of 62 tasks divided into 179 subtasks. We evaluate 33 existing LLMs on IberoBench on 0- and 5-shot settings. We also explore the issues we encounter when working with the Harness and our approach to solving them to ensure high-quality evaluation."
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

```

### Groups, Tags, and Tasks

#### Groups

None.

#### Tags

None.

#### Tasks

`_p[0-2]` stands for the prompt variants of each task, e.g. `hplte_cat_arc_easy_p0`. [`task_mapping.json`](task_mapping.json) lists every task with its name in the HPLT-e repository and its prompt.

| Dataset | Task | Task type | Task category | Name in the HPLT-e repository |
|:---|:---|:---|:---|:---|
| [ARC-ca (Easy)](https://huggingface.co/datasets/projecte-aina/arc_ca) | `hplte_cat_arc_easy_p[0-2]` 😎 | Multiple-choice QA | Language-specific & world knowledge | `arc_ca_easy_p[0-2]` |
| [ARC-ca (Challenge)](https://huggingface.co/datasets/projecte-aina/arc_ca) | `hplte_cat_arc_challenge_p[0-2]` | Multiple-choice QA | Language-specific & world knowledge | `arc_ca_challenge_p[0-2]` |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) | `hplte_cat_belebele_p[0-2]` | Multiple-choice QA | Reading comprehension | `catbelebele_p[0-2]` |
| [CatalanQA](https://huggingface.co/datasets/projecte-aina/catalanqa) | `hplte_cat_catalanqa_p[0-2]` | Generative QA | Language-specific & world knowledge | `catalanqa_p[0-2]` |
| [CatCoLA](https://huggingface.co/datasets/nbel/CatCoLA) | `hplte_cat_catcola_p[0-2]` | Text classification | Language knowledge | `catcola_p[0-2]` |
| [CoCoTeroS-VA](https://huggingface.co/datasets/gplsi/cocoteros_va) | `hplte_cat_cocoteros_va_p[0-2]` | Text generation | Commonsense reasoning | `cocoteros_va_p[0-2]` |
| [COPA-ca](https://huggingface.co/datasets/projecte-aina/COPA-ca) | `hplte_cat_copa_p[0-2]` 😎 | Text classification | Commonsense reasoning | `copa_ca_p[0-2]` |
| [CoQCat](https://huggingface.co/datasets/projecte-aina/CoQCat) | `hplte_cat_coqcat_p[0-2]` 😎 | Generative QA | Reading comprehension | `coqcat_p[0-2]` |
| [FLORES-200 (English → Catalan)](https://huggingface.co/datasets/facebook/flores) | `hplte_cat_flores_en-ca_p[0-2]` | Sequence-to-sequence generation | Machine translation | `flores_en-ca_p[0-2]` |
| [MGSM-ca](https://huggingface.co/datasets/projecte-aina/mgsm_ca) | `hplte_cat_mgsm_direct_p[0-2]` | Generative QA | Mathematical reasoning | `mgsm_direct_ca_p[0-2]` |
| [OpenBookQA-ca](https://huggingface.co/datasets/projecte-aina/openbookqa_ca) | `hplte_cat_openbookqa_p[0-2]` | Multiple-choice QA | Language-specific & world knowledge | `openbookqa_ca_p[0-2]` |
| [Parafraseja](https://huggingface.co/datasets/projecte-aina/Parafraseja) | `hplte_cat_parafraseja_p[0-2]` | Text classification | Paraphrase detection | `parafraseja_p[0-2]` |
| [PAWS-ca](https://huggingface.co/datasets/projecte-aina/PAWS-ca) | `hplte_cat_paws_p[0-2]` | Text classification | Paraphrase detection | `paws_ca_p[0-2]` |
| [PIQA-ca](https://huggingface.co/datasets/projecte-aina/piqa_ca) | `hplte_cat_piqa_p[0-2]` 😎 | Multiple-choice QA | Commonsense reasoning | `piqa_ca_p[0-2]` |
| [SIQA-ca](https://huggingface.co/datasets/projecte-aina/siqa_ca) | `hplte_cat_siqa_p[0-2]` 😎 | Multiple-choice QA | Commonsense reasoning | `siqa_ca_p[0-2]` |
| [TE-ca](https://huggingface.co/datasets/projecte-aina/teca) | `hplte_cat_teca_p[0-2]` 😎 | Text classification | Entailment | `teca_p[0-2]` |
| [VeritasQA-ca (generation)](https://huggingface.co/datasets/projecte-aina/veritasQA) | `hplte_cat_veritasqa_gen_p[0-2]` | Generative QA | Truthfulness | `veritasqa_ca_gen_p[0-2]` |
| [VeritasQA-ca (MC1)](https://huggingface.co/datasets/projecte-aina/veritasQA) | `hplte_cat_veritasqa_mc1_p[0-2]` | Multiple-choice QA | Truthfulness | `veritasqa_ca_mc1_p[0-2]` |
| [VeritasQA-ca (MC2)](https://huggingface.co/datasets/projecte-aina/veritasQA) | `hplte_cat_veritasqa_mc2_p[0-2]` | Multiple-choice QA | Truthfulness | `veritasqa_ca_mc2_p[0-2]` |
| [WNLI-ca](https://huggingface.co/datasets/projecte-aina/wnli-ca) | `hplte_cat_wnli_p[0-2]` | Text classification | Entailment | `wnli_ca_p[0-2]` |
| [XNLI-ca](https://huggingface.co/datasets/projecte-aina/xnli-ca) | `hplte_cat_xnli_p[0-2]` | Text classification | Entailment | `xnli_ca_p[0-2]` |
| [XQuAD-ca](https://huggingface.co/datasets/projecte-aina/xquad-ca) | `hplte_cat_xquad_p[0-2]` | Generative QA | Reading comprehension | `xquad_ca_p[0-2]` |
| [xStoryCloze-ca](https://huggingface.co/datasets/projecte-aina/xstorycloze_ca) | `hplte_cat_xstorycloze_p[0-2]` 😎 | Multiple-choice QA | Commonsense reasoning | `xstorycloze_ca_p[0-2]` |

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
