# DarijaBench: Summarization

### Paper

Title: Atlas-Chat: Adapting Large Language Models for Low-Resource Moroccan Arabic Dialect

Abstract: [https://arxiv.org/abs/2409.17912](https://arxiv.org/abs/2409.17912)

DarijaBench, a comprehensive evaluation dataset tailored for Moroccan Darija. DarijaBench includes different datasets for core NLP tasks such as translation (based on four datasets, [DODa-10K](https://huggingface.co/datasets/MBZUAI-Paris/DODa-10K), [FLORES+](https://github.com/openlanguagedata/flores), [NLLB-Seed](https://github.com/openlanguagedata/seed) and [MADAR](https://sites.google.com/nyu.edu/madar/)), summarization (based  on [MArSum](https://github.com/KamelGaanoun/MoroccanSummarization)) and, sentiment analysis (based on five datasets, [MAC](https://github.com/LeMGarouani/MAC), [MYC](https://github.com/MouadJb/MYC), [MSAC](https://hal.science/hal-03670346/document), [MSDA](https://cc.um6p.ma/cc_datasets) and, [ElectroMorocco2016](https://github.com/sentiprojects/ElecMorocco2016)), in addition to a new transliteration task to convert between Darija (written in Arabic letters) and Arabizi (written in Latin letters) it is based on [DODa-10K](https://huggingface.co/datasets/MBZUAI-Paris/DODa-10K) dataset.


Homepage: [https://huggingface.co/datasets/MBZUAI-Paris/DarijaBench](https://huggingface.co/datasets/MBZUAI-Paris/DarijaBench)


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

* `darija_summarization`: evaluates Darija summarization task.

#### Tasks

* `darija_summarization_task`: evaluates Darija summarization task from [MArSum](https://github.com/KamelGaanoun/MoroccanSummarization) corpus.


### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
