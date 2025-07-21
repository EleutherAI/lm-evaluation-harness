# BLEnD

## Paper

Title: BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages

Abstract: https://arxiv.org/abs/2406.09948

Homepage: https://huggingface.co/datasets/nayeon212/BLEnD

Large language models (LLMs) often lack culture-specific everyday knowledge, especially across diverse regions and non-English languages. Existing benchmarks for evaluating LLMs' cultural sensitivities are usually limited to a single language or online sources like Wikipedia, which may not reflect the daily habits, customs, and lifestyles of different regions. That is, information about the food people eat for their birthday celebrations, spices they typically use, musical instruments youngsters play, or the sports they practice in school is not always explicitly written online. To address this issue, we introduce BLEnD, a hand-crafted benchmark designed to evaluate LLMs' everyday knowledge across diverse cultures and languages. The benchmark comprises 52.6k question-answer pairs from 16 countries/regions, in 13 different languages, including low-resource ones such as Amharic, Assamese, Azerbaijani, Hausa, and Sundanese. We evaluate LLMs in two formats: short-answer questions, and multiple-choice questions. We show that LLMs perform better in cultures that are more present online, with a maximum 57.34% difference in GPT-4, the best-performing model, in the short-answer format. Furthermore, we find that LLMs perform better in their local languages for mid-to-high-resource languages. Interestingly, for languages deemed to be low-resource, LLMs provide better answers in English.


### Citation

```text
@misc{myung2025blendbenchmarkllmseveryday,
      title={BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages}, 
      author={Junho Myung and Nayeon Lee and Yi Zhou and Jiho Jin and Rifki Afina Putri and Dimosthenis Antypas and Hsuvas Borkakoty and Eunsu Kim and Carla Perez-Almendros and Abinew Ali Ayele and Víctor Gutiérrez-Basulto and Yazmín Ibáñez-García and Hwaran Lee and Shamsuddeen Hassan Muhammad and Kiwoong Park and Anar Sabuhi Rzayev and Nina White and Seid Muhie Yimam and Mohammad Taher Pilehvar and Nedjma Ousidhoum and Jose Camacho-Collados and Alice Oh},
      year={2025},
      eprint={2406.09948},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.09948}, 
}```

### Implementation

The Hugging Face dataset provides the full prompt, including the question, formatting instructions, and the options. For example, for the UK task, a prompt is:

```
What is a common snack for preschool kids in the UK? Without any explanation, choose only one from the given alphabet choices(e.g., A, B, C). Provide as JSON format: {"answer_choice":""}

A. candy
B. cookie
C. egg
D. fruit

Answer:
```

In this implementation, since MCQA tasks are evaluated with logprobs, we remove the instruction to provide the answer as JSON format. For example, the prompt becomes:

```
What is a common snack for preschool kids in the UK? Without any explanation, choose only one from the given alphabet choices(e.g., A, B, C).

A. candy
B. cookie
C. egg
D. fruit

Answer:
```

### Groups, Tags, and Tasks

#### Groups

* `blend`: All languages

#### Tags

* `tag_name`: `Short description`

#### Tasks

- `blend_algeria.yaml`
- `blend_assam.yaml`
- `blend_azerbaijan.yaml`
- `blend_china.yaml`
- `blend_ethiopia.yaml`
- `blend_greece.yaml`
- `blend_indonesia.yaml`
- `blend_iran.yaml`
- `blend_mexico.yaml`
- `blend_north_korea.yaml`
- `blend_northern_nigeria.yaml`
- `blend_south_korea.yaml`
- `blend_spain.yaml`
- `blend_uk.yaml`
- `blend_us.yaml`
- `blend_west_java.yaml`

### Checklist

For adding novel benchmarks/datasets to the library:

* [X] Is the task an existing benchmark in the literature?
  * [X] Have you referenced the original paper that introduced the task?
  * [X] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
