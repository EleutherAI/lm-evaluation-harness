# MathQA

### Paper

IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models
https://arxiv.org/pdf/2406.03368

IrokoBench is a human-translated benchmark dataset for 16 typologically diverse
low-resource African languages covering three tasks: natural language inference (AfriXNLI),
mathematical reasoning (AfriMGSM), and multi-choice knowledge-based QA (AfriMMLU).


### Citation

```
@misc{adelani2024irokobenchnewbenchmarkafrican,
      title={IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models},
      author={David Ifeoluwa Adelani and Jessica Ojo and Israel Abebe Azime and Jian Yun Zhuang and Jesujoba O. Alabi and Xuanli He and Millicent Ochieng and Sara Hooker and Andiswa Bukula and En-Shiun Annie Lee and Chiamaka Chukwuneke and Happy Buzaaba and Blessing Sibanda and Godson Kalipe and Jonathan Mukiibi and Salomon Kabongo and Foutse Yuehgoh and Mmasibidi Setaka and Lolwethu Ndolela and Nkiruka Odu and Rooweither Mabuya and Shamsuddeen Hassan Muhammad and Salomey Osei and Sokhar Samb and Tadesse Kebede Guge and Pontus Stenetorp},
      year={2024},
      eprint={2406.03368},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.03368},
}
```

### Groups and Tasks

#### Groups

* `afrimgsm`: All afrimgsm tasks
* `afrimgsm_direct`: afrimgsm_direct evaluates models performance on the curated dataset
* `afrimgsm_en_cot`: afrimgsm_en_cot includes 5-shot of exemplars for chain-of-thought approach
* `afrimgsm_translate`: afrimgsm_translate evaluates models in translate-test setting

#### Tasks
* `afrimgsm_direct_{language_code}`: each task evaluates for one language
* `afrimgsm_en_cot_{language_code}`: each task evaluates for one language
* `afrimgsm_translate_{language_code}`: each task evaluates for one language

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
  * [x] Checked for equivalence with v0.3.0 LM Evaluation Harness
