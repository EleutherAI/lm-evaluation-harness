# BasqueBench

### Paper

BasqueBench is a benchmark for evaluating language models in Basque tasks. This is, it evaluates the ability of a language model to understand and generate Basque text. BasqueBench offers a combination of pre-existing, open datasets and datasets developed exclusivelly for this benchmark. All the details of BasqueBench will be published in a paper soon.

The new evaluation datasets included in BasqueBench are:
| Task     | Category                   | Homepage                                      |
|:--------:|:--------------------------:|:---------------------------------------------:|
| ARC_eu   | Question Answering         | https://huggingface.co/datasets/HiTZ/ARC-eu   |
| MGSM_eu  | Math                       | https://huggingface.co/datasets/HiTZ/MGSM-eu  |
| PAWS_eu  | Paraphrasing               | https://huggingface.co/datasets/HiTZ/PAWS-eu  |
| PIQA_eu  | Question Answering         | https://huggingface.co/datasets/HiTZ/PIQA-eu  |
| WNLI_eu  | Natural Language Inference | https://huggingface.co/datasets/HiTZ/WNLI-eu  |
| XCOPA_eu | Commonsense Reasoning      | https://huggingface.co/datasets/HiTZ/XCOPA-eu |

The datasets included in BasqueBench that have been made public in previous publications are:

| Task          | Category       | Paper title          | Homepage  |
|:-------------:|:-----:|:-------------:|:-----:|
| Belebele_eu | Reading Comprehension | [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants](https://arxiv.org/abs/2308.16884) | https://huggingface.co/datasets/facebook/belebele |
| EusExams | Question Answering | [Latxa: An Open Language Model and Evaluation Suite for Basque](https://arxiv.org/abs/2403.20266) | https://huggingface.co/datasets/HiTZ/EusExams |
| EusProficiency | Question Answering | [Latxa: An Open Language Model and Evaluation Suite for Basque](https://arxiv.org/abs/2403.20266) | https://huggingface.co/datasets/HiTZ/EusProficiency |
| EusReading | Reading Comprehension | [Latxa: An Open Language Model and Evaluation Suite for Basque](https://arxiv.org/abs/2403.20266) | https://huggingface.co/datasets/HiTZ/EusReading |
| EusTrivia | Question Answering | [Latxa: An Open Language Model and Evaluation Suite for Basque](https://arxiv.org/abs/2403.20266) | https://huggingface.co/datasets/HiTZ/EusTrivia |
| FLORES_eu | Translation | [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) | https://huggingface.co/datasets/facebook/flores |
| QNLIeu | Natural Language Inference | [BasqueGLUE: A Natural Language Understanding Benchmark for Basque](https://aclanthology.org/2022.lrec-1.172/) | https://huggingface.co/datasets/orai-nlp/basqueGLUE |
| XNLIeu | Natural Language Inference | [XNLIeu: a dataset for cross-lingual NLI in Basque](https://arxiv.org/abs/2404.06996) | https://huggingface.co/datasets/HiTZ/xnli-eu |
| XStoryCloze_eu | Commonsense Reasoning | [Few-shot Learning with Multilingual Generative Language Models](https://aclanthology.org/2022.emnlp-main.616/) | https://huggingface.co/datasets/juletxara/xstory_cloze |


### Citation

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

- `basque_bench`: All tasks included in BasqueBench.
- `flores_eu`: All FLORES translation tasks from or to Basque.

#### Tasks

The following tasks evaluate tasks on BasqueBench dataset using various scoring methods.
  - `arc_eu_challenge`
  - `arc_eu_easy`
  - `belebele_eus_Latn`
  - `eus_exams_eu`
  - `eus_proficiency`
  - `eus_reading`
  - `eus_trivia`
  - `flores_eu`
  - `flores_eu-ca`
  - `flores_eu-de`
  - `flores_eu-en`
  - `flores_eu-es`
  - `flores_eu-fr`
  - `flores_eu-gl`
  - `flores_eu-it`
  - `flores_eu-pt`
  - `flores_ca-eu`
  - `flores_de-eu`
  - `flores_en-eu`
  - `flores_es-eu`
  - `flores_fr-eu`
  - `flores_gl-eu`
  - `flores_it-eu`
  - `flores_pt-eu`
  - `mgsm_direct_eu`
  - `mgsm_native_cot_eu`
  - `paws_eu`
  - `piqa_eu`
  - `qnlieu`
  - `wnli_eu`
  - `xcopa_eu`
  - `xnli_eu`
  - `xnli_eu_native`
  - `xstorycloze_eu`

Some of these tasks are taken from benchmarks already available in LM Evaluation Harness. These are:
- `belebele_eus_Latn`: Belebele Basque
- `qnlieu`: From BasqueGLUE


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

* v1.1 (2026-01-16, PR #3465): excluded line breaks from stop criteria in `mgsm_direct_eu` and `msgm_native_cot_eu`
