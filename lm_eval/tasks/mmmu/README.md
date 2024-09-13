# MMMU Benchmark

### Paper

Title: `MMMU: A Massive Multi-discipline MultimodalUnderstanding and Reasoning Benchmark for Expert AGI`

Abstract: `MMMU is a new benchmark designed to evaluate multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning.`

`The benchmark is composed of 30 tasks, for a total of 900 mixed image+text examples (some with multiple images in context)`

Homepage: `https://github.com/MMMU-Benchmark/MMMU/tree/main/mmmu`

Note: Some questions have multiple images in context. To control for this use `max_images=N` in model init.

### Citation

```
@inproceedings{yue2023mmmu,
            title={MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI},
            author={Xiang Yue and Yuansheng Ni and Kai Zhang and Tianyu Zheng and Ruoqi Liu and Ge Zhang and Samuel Stevens and Dongfu Jiang and Weiming Ren and Yuxuan Sun and Cong Wei and Botao Yu and Ruibin Yuan and Renliang Sun and Ming Yin and Boyuan Zheng and Zhenzhu Yang and Yibo Liu and Wenhao Huang and Huan Sun and Yu Su and Wenhu Chen},
            booktitle={Proceedings of CVPR},
            year={2024},
          }
```

### Groups, Tags, and Tasks

#### Groups

* `mmmu_val`
* `mmmu_val_art_and_design`
* `mmmu_val_business`
* `mmmu_val_health_and_medicine`
* `mmmu_val_humanities_and_social_science`
* `mmmu_val_science`
* `mmmu_val_tech_and_engineering`

#### Tags


#### Tasks

* `mmmu_val_accounting`
* `mmmu_val_agriculture`
* `mmmu_val_architecture_and_engineering.yaml`
* `mmmu_val_art`
* `mmmu_val_art_theory`
* `mmmu_val_basic_medical_science`
* `mmmu_val_biology`
* `mmmu_val_chemistry`
* `mmmu_val_computer_science`
* `mmmu_val_clinical_medicine`
* `mmmu_val_design`
* `mmmu_val_diagnostics_and_laboratory_medicine`
* `mmmu_val_electronics`
* `mmmu_val_energy_and_power`
* `mmmu_val_economics`
* `mmmu_val_finance`
* `mmmu_val_geography`
* `mmmu_val_history`
* ...

### Variants



### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
