# DarijaMMLU

### Paper

Title: Atlas-Chat: Adapting Large Language Models for Low-Resource Moroccan Arabic Dialect

Abstract: [https://arxiv.org/abs/2409.17912](https://arxiv.org/abs/2409.17912)

DarijaMMLU is an evaluation benchmark designed to assess large language models' (LLM) performance in Moroccan Darija, a variety of Arabic. It consists of 22,027 multiple-choice questions, translated from selected subsets of the Massive Multitask Language Understanding (MMLU) and ArabicMMLU benchmarks to measure model performance on 44 subjects in Darija. DarijaMMLU is constructed by translating selected subsets from two major benchmarks into Darija from English and MSA: Massive Multitask Language Understanding (MMLU) and ArabicMMLU.


Homepage: [https://huggingface.co/datasets/MBZUAI-Paris/DarijaMMLU](https://huggingface.co/datasets/MBZUAI-Paris/DarijaMMLU)


### Citation

```
@article{shang2024atlaschatadaptinglargelanguage,
      title={Atlas-Chat: Adapting Large Language Models for Low-Resource Moroccan Arabic Dialect},
      author={Guokan Shang and Hadi Abdine and Yousef Khoubrane and Amr Mohamed and Yassine Abbahaddou and Sofiane Ennadir and Imane Momayiz and Xuguang Ren and Eric Moulines and Preslav Nakov and Michalis Vazirgiannis and Eric Xing},
      year={2024},
      eprint={2409.17912},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.17912},
}
```

### Groups and Tasks

#### Groups

* `darijammlu`: evaluates all DarijaMMLU tasks.

#### Tags
Source-based tags:

* `darijammlu_mmlu`: evaluates DarijaMMLU tasks that were translated from MMLU.
* `darijammlu_ar_mmlu`: evaluates DarijaMMLU tasks that were translated from ArabicMMLU.

Category-based tags:

* `darijammlu_stem`: evaluates DarijaMMLU STEM tasks.
* `darijammlu_social_sciences`: evaluates DarijaMMLU social sciences tasks.
* `darijammlu_humanities`: evaluates DarijaMMLU humanities tasks.
* `darijammlu_language`: evaluates DarijaMMLU language tasks.
* `darijammlu_other`: evaluates other DarijaMMLU tasks.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
