# CatalanBench

### Paper

CatalanBench is a benchmark for evaluating language models in Catalan tasks. This is, it evaluates the ability of a language model to understand and generate Catalan text. CatalanBench offers a combination of pre-existing, open datasets and datasets developed exclusivelly for this benchmark. All the details of CatalanBench will be published in a paper soon.

The new evaluation datasets included in CatalanBench are:
| Task          | Category       | Homepage  |
|:-------------:|:-----:|:-----:|
| ARC_ca | Question Answering | https://huggingface.co/datasets/projecte-aina/arc_ca |
| MGSM_ca | Math | https://huggingface.co/datasets/projecte-aina/mgsm_ca |
| OpenBookQA_ca | Question Answering | https://huggingface.co/datasets/projecte-aina/openbookqa_ca |
| Parafraseja | Paraphrasing | https://huggingface.co/datasets/projecte-aina/Parafraseja |
| PIQA_ca | Question Answering | https://huggingface.co/datasets/projecte-aina/piqa_ca |
| SIQA_ca | Question Answering | https://huggingface.co/datasets/projecte-aina/siqa_ca |
| XStoryCloze_ca | Commonsense Reasoning | https://huggingface.co/datasets/projecte-aina/xstorycloze_ca |

The datasets included in CatalanBench that have been made public in previous pubications are:

| Task          | Category       | Paper title          | Homepage  |
|:-------------:|:-----:|:-------------:|:-----:|
| Belebele_ca | Reading Comprehension | [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants](https://arxiv.org/abs/2308.16884) | https://huggingface.co/datasets/facebook/belebele |
| caBREU | Summarization | [Building a Data Infrastructure for a Mid-Resource Language: The Case of Catalan](https://aclanthology.org/2024.lrec-main.231/) | https://huggingface.co/datasets/projecte-aina/caBreu |
| CatalanQA | Question Answering | [Building a Data Infrastructure for a Mid-Resource Language: The Case of Catalan](https://aclanthology.org/2024.lrec-main.231/) | https://huggingface.co/datasets/projecte-aina/catalanqa |
| CatCoLA | Linguistic Acceptability | CatCoLA: Catalan Corpus of Linguistic Acceptability | https://huggingface.co/datasets/nbel/CatCoLA |
| Cocoteros_va | Commonsense Reasoning | COCOTEROS_VA: Valencian translation of the COCOTEROS Spanish dataset | https://huggingface.co/datasets/gplsi/cocoteros_va |
 | EsCoLA | Linguistic Acceptability | [EsCoLA: Spanish Corpus of Linguistic Acceptability](https://aclanthology.org/2024.lrec-main.554/) |
| COPA-ca | Commonsense Reasoning | [Building a Data Infrastructure for a Mid-Resource Language: The Case of Catalan](https://aclanthology.org/2024.lrec-main.231/) | https://huggingface.co/datasets/projecte-aina/COPA-ca |
| CoQCat | Question Answering | [Building a Data Infrastructure for a Mid-Resource Language: The Case of Catalan](https://aclanthology.org/2024.lrec-main.231/) | https://huggingface.co/datasets/projecte-aina/CoQCat |
| FLORES_ca | Translation | [The FLORES-101  Evaluation Benchmark for Low-Resource and Multilingual Machine Translation](https://arxiv.org/abs/2106.03193) | https://huggingface.co/datasets/facebook/flores |
| PAWS-ca | Paraphrasing | [Building a Data Infrastructure for a Mid-Resource Language: The Case of Catalan](https://aclanthology.org/2024.lrec-main.231/) | https://huggingface.co/datasets/projecte-aina/PAWS-ca |
| TE-ca | Natural Language Inference | [Building a Data Infrastructure for a Mid-Resource Language: The Case of Catalan](https://aclanthology.org/2024.lrec-main.231/) | https://huggingface.co/datasets/projecte-aina/teca |
| VeritasQA_ca | Truthfulness | VeritasQA: A Truthfulness Benchmark Aimed at Multilingual Transferability | TBA |
| WNLI-ca | Natural Language Inference | [Building a Data Infrastructure for a Mid-Resource Language: The Case of Catalan](https://aclanthology.org/2024.lrec-main.231/) | https://huggingface.co/datasets/projecte-aina/wnli-ca |
| XNLI-ca | Natural Language Inference | [Building a Data Infrastructure for a Mid-Resource Language: The Case of Catalan](https://aclanthology.org/2024.lrec-main.231/) | https://huggingface.co/datasets/projecte-aina/xnli-ca |
| XNLI-va | Natural Language Inference | Building a Data Infrastructure for a Mid-Resource Language: The Case of Valencian | https://huggingface.co/datasets/gplsi/xnli_va |
| XQuAD-ca | Question Answering | [Building a Data Infrastructure for a Mid-Resource Language: The Case of Catalan](https://aclanthology.org/2024.lrec-main.231/) | https://huggingface.co/datasets/projecte-aina/xquad-ca |


### Citation
Paper for CatalanBench coming soon.

```
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
}
```

### Groups and Tasks

#### Groups

- `catalan_bench`: All tasks included in CatalanBench.
- `flores_ca`: All FLORES translation tasks from or to Catalan.

#### Tags
- `cabreu`: Three CaBREU tasks for each type of summary (extractive, abstractive and extreme).
- `phrases_va`: Two Phrases_va tasks for language adaptation between Catalan and Valencian.

#### Tasks

The following tasks evaluate tasks on CatalanBench dataset using various scoring methods.
  - `arc_ca_challenge`
  - `arc_ca_easy`
  - `belebele_cat_Latn`
  - `cabreu`
  - `catalanqa`
  - `catcola`
  - `cocoteros_va`
  - `copa_ca`
  - `coqcat`
  - `flores_ca`
  - `flores_ca-de`
  - `flores_ca-en`
  - `flores_ca-es`
  - `flores_ca-eu`
  - `flores_ca-fr`
  - `flores_ca-gl`
  - `flores_ca-it`
  - `flores_ca-pt`
  - `flores_de-ca`
  - `flores_en-ca`
  - `flores_es-ca`
  - `flores_eu-ca`
  - `flores_fr-ca`
  - `flores_gl-ca`
  - `flores_it-ca`
  - `flores_pt-ca`
  - `mgsm_direct_ca`
  - `openbookqa_ca`
  - `parafraseja`
  - `paws_ca`
  - `phrases_ca`
  - `piqa_ca`
  - `siqa_ca`
  - `teca`
  - `veritasqa_gen_ca`
  - `veritasqa_mc1_ca`
  - `veritasqa_mc2_ca`
  - `wnli_ca`
  - `xnli_ca`
  - `xnli_va`
  - `xquad_ca`
  - `xstorycloze_ca`

Some of these tasks are taken from benchmarks already available in LM Evaluation Harness. These are:
- `belebele_cat_Latn`: Belebele Catalan


### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation?
    * [ ] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?


### Changelog
version 2.0: (2025-Mar-18) add [`cococteros_va`](./cocoteros_va.yaml) task.
version 2.1: (2025-Jul-30) add [`xnli_va`](./xnli_va.yaml) task.
