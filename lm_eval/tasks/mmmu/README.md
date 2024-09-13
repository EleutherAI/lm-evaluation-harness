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

The `mmmu_val` group implements MMMU using processing code [from the original MMMU authors](https://github.com/MMMU-Benchmark/MMMU/tree/main/mmmu) and uses the prompt format found in [the MMMU repository for Llava-1.5](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/configs/llava1.5.yaml). This implementation should give scores on par with or slightly higher than those reported by [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/mmmu) for `mmmu_val` and the MMMU repository code.

Scores on several tested models (**all with `--apply_chat_template`**) are:

Qwen2-VL-2B:
```
hf-multimodal (pretrained=Qwen/Qwen2-VL-2B-Instruct,attn_implementation=flash_attention_2,dtype=bfloat16,convert_img_format=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 2
```
```
|             Groups             |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|--------------------------------|------:|------|------|------|---|-----:|---|-----:|
|mmmu_val                        |      0|none  |      |acc   |↑  |0.3778|±  |0.0155|
| - Art and Design               |      0|none  |      |acc   |↑  |0.5500|±  |0.0415|
| - Business                     |      0|none  |      |acc   |↑  |0.3600|±  |0.0389|
| - Health and Medicine          |      0|none  |      |acc   |↑  |0.3667|±  |0.0394|
| - Humanities and Social Science|      0|none  |      |acc   |↑  |0.5167|±  |0.0438|
| - Science                      |      0|none  |      |acc   |↑  |0.2467|±  |0.0352|
| - Tech and Engineering         |      0|none  |      |acc   |↑  |0.3143|±  |0.0317|
```
Author-reported score: 41.1%


Qwen2-VL-7B:
```
hf-multimodal (pretrained=Qwen/Qwen2-VL-7B-Instruct,attn_implementation=flash_attention_2,dtype=bfloat16,convert_img_format=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 2
```
```
|             Groups             |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|--------------------------------|------:|------|------|------|---|-----:|---|-----:|
|mmmu_val                        |      0|none  |      |acc   |↑  |0.5056|±  |0.0160|
| - Art and Design               |      0|none  |      |acc   |↑  |0.6917|±  |0.0398|
| - Business                     |      0|none  |      |acc   |↑  |0.4333|±  |0.0406|
| - Health and Medicine          |      0|none  |      |acc   |↑  |0.5667|±  |0.0401|
| - Humanities and Social Science|      0|none  |      |acc   |↑  |0.6750|±  |0.0426|
| - Science                      |      0|none  |      |acc   |↑  |0.3800|±  |0.0392|
| - Tech and Engineering         |      0|none  |      |acc   |↑  |0.4000|±  |0.0341|
```
Author-reported score: 54.1%

Idefics2-8B:
```
hf-multimodal (pretrained=HuggingFaceM4/idefics2-8b,attn_implementation=flash_attention_2,dtype=bfloat16,convert_img_format=True,max_images=2), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 2
```
```
|             Groups             |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|--------------------------------|------:|------|------|------|---|-----:|---|-----:|
|mmmu_val                        |      0|none  |      |acc   |↑  |0.4011|±  |0.0154|
| - Art and Design               |      0|none  |      |acc   |↑  |0.6167|±  |0.0436|
| - Business                     |      0|none  |      |acc   |↑  |0.3200|±  |0.0373|
| - Health and Medicine          |      0|none  |      |acc   |↑  |0.4000|±  |0.0401|
| - Humanities and Social Science|      0|none  |      |acc   |↑  |0.5750|±  |0.0424|
| - Science                      |      0|none  |      |acc   |↑  |0.2600|±  |0.0358|
| - Tech and Engineering         |      0|none  |      |acc   |↑  |0.3381|±  |0.0312|
```
Author-reported score: ~43%

Llava-v1.6-Mistral-7B:
```
hf-multimodal (pretrained=llava-hf/llava-v1.6-mistral-7b-hf,attn_implementation=flash_attention_2,dtype=bfloat16,convert_img_format=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 2
```
```
|             Groups             |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|--------------------------------|------:|------|------|------|---|-----:|---|-----:|
|mmmu_val                        |      0|none  |      |acc   |↑  |0.3522|±  |0.0151|
| - Art and Design               |      0|none  |      |acc   |↑  |0.5167|±  |0.0440|
| - Business                     |      0|none  |      |acc   |↑  |0.2667|±  |0.0362|
| - Health and Medicine          |      0|none  |      |acc   |↑  |0.3867|±  |0.0397|
| - Humanities and Social Science|      0|none  |      |acc   |↑  |0.5917|±  |0.0433|
| - Science                      |      0|none  |      |acc   |↑  |0.2200|±  |0.0342|
| - Tech and Engineering         |      0|none  |      |acc   |↑  |0.2524|±  |0.0299|
```
Author-reported score: 35.3%


### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
