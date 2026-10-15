# AfriXNLI

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

AfriXNLI ships two generations of prompts. The newer per-prompt variants (the
updated IrokoBench/AfroBench prompts) evaluate every language with five
different prompt templates, so their task names carry a `_prompt_{i}` suffix
where `i` is the prompt id (1-5). The original anli- and Lai-prompt tasks are
still registered under their per-language names without the suffix.

#### Groups

* `afrixnli-irokobench`: evaluates all direct tasks over all five prompts and reports a size-weighted mean accuracy
* `afrixnli_tt-irokobench`: same as above in the translate-test setting

#### Tags

* `afrixnli_tasks`: runs all direct per-prompt tasks (all five prompts)
* `afrixnli_tasks_prompt_{i}`: runs all direct per-prompt tasks for prompt `i` (1-5)
* `afrixnli_tt_tasks`: runs all per-prompt translate-test tasks
* `afrixnli`: runs all anli- and Lai-prompt tasks
* `afrixnli_en_direct`: anli prompt on the curated dataset
* `afrixnli_native_direct`: anli prompt translated to the respective languages on the curated dataset
* `afrixnli_translate`: anli prompt in the translate-test setting
* `afrixnli_manual_direct`: Lai's prompt on the curated dataset
* `afrixnli_manual_translate`: Lai's prompt in the translate-test setting

#### Tasks

Per-prompt variants:
* `afrixnli_{language_code}_prompt_{i}`: evaluates one language with prompt `i` (1-5) on the curated dataset, e.g. `afrixnli_amh_prompt_1`
* `afrixnli_translate_{language_code}_prompt_{i}`: same as above in the translate-test setting

Original anli- and Lai-prompt variants:
* `afrixnli_en_direct_{language_code}`: each task evaluates for one language
* `afrixnli_native_direct_{language_code}`: each task evaluates for one language
* `afrixnli_translate_{language_code}`: each task evaluates for one language
* `afrixnli_manual_direct_{language_code}`: each task evaluates for one language
* `afrixnli_manual_translate_{language_code}`: each task evaluates for one language

with `language_code` one of `amh`, `eng`, `ewe`, `fra`, `hau`, `ibo`, `kin`,
`lin`, `lug`, `orm`, `sna`, `sot`, `swa`, `twi`, `wol`, `xho`, `yor`, `zul`
(`eng` has no translate-test variant).

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
