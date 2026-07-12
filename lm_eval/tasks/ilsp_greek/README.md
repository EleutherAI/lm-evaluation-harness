# ILSP Greek benchmark suite

### Paper / source

Greek versions of several widely used English LLM benchmarks, published by the
**Institute for Language and Speech Processing (ILSP), Athena Research Center**. The datasets are
machine-translated (WinoGrande and MGSM manually curated) and are the ones used in ILSP's evaluation
of their Greek foundation models
[Meltemi](https://huggingface.co/ilsp/Meltemi-7B-v1) and
[Llama-Krikri](https://huggingface.co/ilsp/Llama-Krikri-8B-Base)
(Voukoutis et al., *Krikri: Advancing Open Large Language Models for Greek*,
[arXiv:2505.13772](https://arxiv.org/abs/2505.13772)).

Each dataset preserves the field schema of its English original, so these tasks reuse the upstream
English task logic and only override the dataset path (plus per-dataset fixes noted below). The
prompt scaffolding is intentionally kept in English (translate the data, not the template), matching
ILSP's Open-LLM-Leaderboard-style setup and the existing `arc_challenge_mt_el` task — this keeps
scores comparable to ILSP's published numbers. The scored content (questions and answer choices) is
Greek and comes from the datasets themselves.

| Task | Category | Source dataset |
|:---|:---|:---|
| `arc_challenge_greek` | Science QA / Reasoning | https://huggingface.co/datasets/ilsp/arc_greek |
| `hellaswag_greek` | Commonsense NLI | https://huggingface.co/datasets/ilsp/hellaswag_greek |
| `truthfulqa_greek_mc1` | Truthfulness | https://huggingface.co/datasets/ilsp/truthful_qa_greek |
| `truthfulqa_greek_mc2` | Truthfulness | https://huggingface.co/datasets/ilsp/truthful_qa_greek |
| `mgsm_direct_greek` | Grade-school math | https://huggingface.co/datasets/ilsp/mgsm_greek |
| `winogrande_greek` | Commonsense coreference | https://huggingface.co/datasets/ilsp/winogrande_greek |

> **License:** the ILSP datasets are released under **CC-BY-NC-SA-4.0** (non-commercial). Evaluate
> accordingly.

### Recommended few-shot settings

lm-eval-harness applies `num_fewshot` at run time. To match the Open-LLM-Leaderboard / ILSP setup:

| Task | num_fewshot | Reported metric |
|:---|:---:|:---|
| `arc_challenge_greek` | 25 | `acc_norm` |
| `hellaswag_greek` | 10 | `acc_norm` |
| `truthfulqa_greek_mc1` / `mc2` | 0 (fixed in config; a 6-shot primer is baked into the prompt) | `acc` |
| `mgsm_direct_greek` | 8 | `exact_match` |
| `winogrande_greek` | 5 | `acc` |

### Notes on individual variants

- **`mgsm_direct_greek`** — `ilsp/mgsm_greek` ships empty `answer` / `equation_solution` fields, so the
  target is taken from `answer_number`; the `question` field has no prefix, so `Question:` is
  prepended. Uses the standard MGSM `direct` generate-until setup with `exact_match` + flexible-extract
  filter.
- **`truthfulqa_greek_mc1` / `mc2`** — `ilsp/truthful_qa_greek` (`multiple_choice` config) exposes only
  a `train` split (817 items), which is used for evaluation. mc1 scores `mc1_targets.choices` (gold
  index 0); mc2 uses the upstream `process_results_mc2` over `mc2_targets`.
- **`winogrande_greek`** — the ILSP dataset keeps `sentence`/`option1`/`option2` in **English** and
  puts the Greek text only in `multiple_choice_targets` (the two full candidate sentences), with the
  correct one flagged by `multiple_choice_scores`/`answer`. This task therefore does **not** reuse the
  English fill-the-blank template; it scores the two Greek sentences directly and picks the more
  likely one (a full-sentence plausibility formulation of WinoGrande). Numbers are therefore not
  directly comparable to the English `winogrande` task.
- **WinoGrande and MGSM** are not part of ILSP's official Meltemi/Krikri model-card evaluation suite
  (which covers ARC, HellaSwag, TruthfulQA, MMLU, Belebele and Medical MCQA); they are included here
  because ILSP publishes the datasets and they are commonly used.

### Groups and Tasks

#### Groups

- `ilsp_greek`: all six ILSP Greek tasks.

#### Tasks

- `arc_challenge_greek`
- `hellaswag_greek`
- `truthfulqa_greek_mc1`
- `truthfulqa_greek_mc2`
- `mgsm_direct_greek`
- `winogrande_greek`

### Citation

Original benchmarks:

```bibtex
@inproceedings{zellers2019hellaswag,
    title     = {HellaSwag: Can a Machine Really Finish Your Sentence?},
    author    = {Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year      = {2019}
}
@article{clark2018arc,
    title   = {Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
    author  = {Clark, Peter and Cowhey, Isaac and Etzioni, Oren and Khot, Tushar and Sabharwal, Ashish and Schoenick, Carissa and Tafjord, Oyvind},
    journal = {arXiv preprint arXiv:1803.05457},
    year    = {2018}
}
@inproceedings{lin2022truthfulqa,
    title     = {TruthfulQA: Measuring How Models Mimic Human Falsehoods},
    author    = {Lin, Stephanie and Hilton, Jacob and Evans, Owain},
    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
    year      = {2022}
}
@article{sakaguchi2021winogrande,
    title   = {WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
    author  = {Sakaguchi, Keisuke and Le Bras, Ronan and Bhagavatula, Chandra and Choi, Yejin},
    journal = {Communications of the ACM},
    year    = {2021}
}
@article{shi2022mgsm,
    title   = {Language Models are Multilingual Chain-of-Thought Reasoners},
    author  = {Shi, Freda and Suzgun, Mirac and Freitag, Markus and Wang, Xuezhi and Srivats, Suraj and Vosoughi, Soroush and Chung, Hyung Won and Tay, Yi and Ruder, Sebastian and Zhou, Denny and Das, Dipanjan and Wei, Jason},
    journal = {arXiv preprint arXiv:2210.03057},
    year    = {2022}
}
```

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test? (These reuse the upstream English task logic in `lm_eval/tasks/{arc,hellaswag,truthfulqa,mgsm}`.)

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted? (Each task maps 1:1 to an ILSP dataset; the group `ilsp_greek` runs all.)
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant? (ILSP Meltemi/Krikri, Open-LLM-Leaderboard few-shot settings.)
