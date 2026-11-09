# AfriMGSM

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

Every AfriMGSM task is evaluated with five different prompt templates (the
updated IrokoBench/AfroBench prompts), so each per-language task name carries a
`_prompt_{i}` suffix where `i` is the prompt id (1-5). The original
`afrimgsm_direct_{language_code}`, `afrimgsm_en_cot_{language_code}` and
`afrimgsm_translate_{language_code}` names without the suffix are no longer
registered.

#### Groups

* `afrimgsm-irokobench`: evaluates all direct tasks over all five prompts and reports a size-weighted mean accuracy
* `afrimgsm_cot-irokobench`: same as above with chain-of-thought prompting
* `afrimgsm_tt-irokobench`: same as above in the translate-test setting
* `afrimgsm_tt_cot-irokobench`: chain-of-thought in the translate-test setting

#### Tags

* `afrimgsm_tasks`: runs all direct tasks (all five prompts)
* `afrimgsm_tasks_prompt_{i}`: runs all direct tasks for prompt `i` (1-5)
* `afrimgsm_cot_tasks`: runs all chain-of-thought tasks (all five prompts)
* `afrimgsm_cot_tasks_prompt_{i}`: runs all chain-of-thought tasks for prompt `i` (1-5)
* `afrimgsm_tt_tasks`: runs all translate-test tasks
* `afrimgsm_tt_cot_tasks`: runs all chain-of-thought translate-test tasks

#### Tasks

* `afrimgsm_{language_code}_prompt_{i}`: evaluates one language with prompt `i` (1-5) on the curated dataset, e.g. `afrimgsm_amh_prompt_1`
* `afrimgsm_cot_{language_code}_prompt_{i}`: same as above with chain-of-thought prompting
* `afrimgsm_translate_{language_code}_prompt_{i}`: same as above in the translate-test setting
* `afrimgsm_cot_translate_{language_code}_prompt_{i}`: chain-of-thought in the translate-test setting

with `language_code` one of `amh`, `eng`, `ewe`, `fra`, `hau`, `ibo`, `kin`,
`lin`, `lug`, `orm`, `sna`, `sot`, `swa`, `twi`, `vai`, `wol`, `xho`, `yor`,
`zul` (the translate-test variants have no `eng`, and `afrimgsm_translate` has
no `vai`).

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
