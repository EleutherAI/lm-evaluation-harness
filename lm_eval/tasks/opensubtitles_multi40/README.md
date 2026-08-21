# OpenSubtitles Multi40 (translation)

This task set evaluates translation on the OpenSubtitles multi-aligned dataset using `generate_until` in `lm-evaluation-harness` framework, and reports `bleu` and `chrf`.

## Recommended setup

If users already installed `lm-evaluation-harness`, recommend loading this task folder via `--include_path` from the cloned task repo, instead of manually copying files into their local `lm_eval/tasks`.

Why:

- does not modify users' existing `lm-evaluation-harness` installation
- task updates are picked up directly after `git pull` (no repeated copying)
- more reproducible (single command path)

Example:

```bash
# After cloning this task repository and entering its root directory:

python -m lm_eval run \
  --model vllm \
  --model_args pretrained=YOUR_MODEL \
  --include_path "$(pwd)" \
  --tasks opensubtitles_multi40_en_to_fi
```

## Files

- `_opensubtitles_multi40_common.yaml`: shared task template
- `utils.py`: custom dataset loader + prompt/target functions
- `_generate_configs.py`: generate direction-specific YAMLs and group YAMLs
- `pair.yaml`: one generic task (`opensubtitles_multi40_pair`) configurable via `--metadata`
  Recommended for arbitrary single language-pair evaluation without pre-generating dedicated YAML files.
- `pairs/<src>_to_<tgt>.yaml`: auto-generated directed subtasks for arbitrary `xx -> yy`
- `groups/<src>_xx.yaml`: source-centric groups (all targets for one source)
- `groups/xx_<tgt>.yaml`: target-centric groups (all sources into one target)
- `groups/all_pairs.yaml`: group task `opensubtitles_multi40_all_pairs` (all directed pairs)
- `groups/en_xx.yaml` and `groups/xx_en.yaml`: official core benchmark groups

## Run examples

Translate with a single direction (e.g., `en -> fi`):

```bash
lm_eval run \
  --model vllm \
  --model_args pretrained=YOUR_MODEL \
  --include_path "$(pwd)" \
  --tasks opensubtitles_multi40_en_to_fi
```

Translate from one language to any other languages (e.g., `en -> xx`):

```bash
lm_eval run \
  --model vllm \
  --model_args pretrained=YOUR_MODEL \
  --include_path "$(pwd)" \
  --tasks opensubtitles_multi40_en_xx
```

Translate from any other languages to one language (`xx -> en`):

```bash
lm_eval run \
  --model vllm \
  --model_args pretrained=YOUR_MODEL \
  --include_path "$(pwd)" \
  --tasks opensubtitles_multi40_xx_en
```

Run selected multiple language pairs in one run:

```bash
lm_eval run \
  --model vllm \
  --model_args pretrained=YOUR_MODEL \
  --include_path "$(pwd)" \
  --tasks opensubtitles_multi40_en_to_fi,opensubtitles_multi40_fi_to_sv,opensubtitles_multi40_sv_to_en
```


Practical recommendation:

Running full all-to-all evaluation (`40x39` directions) can be memory and runtime intensive. Prefer source-centric or target-centric subsets (for example `en->xx` or `xx->en`), and  split work into chunks (`TASK_CHUNK_SIZE`) or multiple jobs for parallel execution.

## Dataset and License

- Dataset: `Helsinki-NLP/OpenSubtitles2024-40-langs-15-movies`
- License: `odc-by`

## Citation

```
@inproceedings{tiedemann-luo-2026-opensubtitles2024,
  title={OpenSubtitles2024: A Massively Parallel Dataset of Movie Subtitles for MT Development and Evaluation},
  author={Tiedemann, Jörg and Luo, Hengyu},
  booktitle={Proceedings of the 15th edition of the Language Resources and Evaluation Conference (LREC 2026)},
  year={2026}
}
```
