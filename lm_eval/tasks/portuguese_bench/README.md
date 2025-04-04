# PortugueseBench

### Paper

PortugueseBench is a benchmark for evaluating language models in Portuguese tasks. This is, it evaluates the ability of a language model to understand and generate Portuguese text. PortugueseBench offers a combination of pre-existing, open datasets. All the details of PortugueseBench will be published in a paper soon.

The datasets included in PortugueseBench are:

| Task          | Category       | Paper title          | Homepage  |
|:-------------:|:-----:|:-------------:|:-----:|
| Belebele_es | Reading Comprehension | [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants](https://arxiv.org/abs/2308.16884) | https://huggingface.co/datasets/facebook/belebele |
| FLORES_es | Translation | [The FLORES-101  Evaluation Benchmark for Low-Resource and Multilingual Machine Translation](https://arxiv.org/abs/2106.03193) | https://huggingface.co/datasets/facebook/flores |
| ASSIN | Natural Language Inference + Paraphrasing | [Avaliando a similaridade semântica entre frases curtas através de uma abordagem híbrida](https://aclanthology.org/W17-6612/) | https://huggingface.co/datasets/nilc-nlp/assin |


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

- `portuguese_bench`: All tasks included in PortugueseBench.
- `flores_pt`: All FLORES translation tasks from or to Portuguese.

#### Tasks

The following tasks evaluate tasks on PortugueseBench dataset using various scoring methods.
  - `assin_paraphrase`
  - `assin_entailment`
  - `belebele_por_Latn`
  - `flores_pt`
  - `flores_pt-ca`
  - `flores_pt-de`
  - `flores_pt-en`
  - `flores_pt-es`
  - `flores_pt-eu`
  - `flores_pt-fr`
  - `flores_pt-gl`
  - `flores_pt-it`
  - `flores_ca-pt`
  - `flores_de-pt`
  - `flores_en-pt`
  - `flores_es-pt`
  - `flores_eu-pt`
  - `flores_fr-pt`
  - `flores_gl-pt`
  - `flores_it-pt`

Some of these tasks are taken from benchmarks already available in LM Evaluation Harness. These are:
- `belebele_por_Latn`: Belebele Portuguese


### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation?
    * [ ] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
