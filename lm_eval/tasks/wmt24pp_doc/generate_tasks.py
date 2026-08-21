#!/usr/bin/env python3
"""Generate WMT24++ task YAMLs.

This creates:
- one common YAML
- one task YAML per supported en->X pair
- one task YAML per supported X->en pair
- two group YAMLs:
  - wmt24pp_doc_en-all
  - wmt24pp_doc_all-en
"""

from __future__ import annotations

from pathlib import Path

LANGUAGE_PAIRS = (
    "en-ar_EG", "en-ar_SA", "en-bg_BG", "en-bn_IN", "en-ca_ES", "en-cs_CZ",
    "en-da_DK", "en-de_DE", "en-el_GR", "en-es_MX", "en-et_EE", "en-fa_IR",
    "en-fi_FI", "en-fil_PH", "en-fr_CA", "en-fr_FR", "en-gu_IN", "en-he_IL",
    "en-hi_IN", "en-hr_HR", "en-hu_HU", "en-id_ID", "en-is_IS", "en-it_IT",
    "en-ja_JP", "en-kn_IN", "en-ko_KR", "en-lt_LT", "en-lv_LV", "en-ml_IN",
    "en-mr_IN", "en-nl_NL", "en-no_NO", "en-pa_IN", "en-pl_PL", "en-pt_BR",
    "en-pt_PT", "en-ro_RO", "en-ru_RU", "en-sk_SK", "en-sl_SI", "en-sr_RS",
    "en-sv_SE", "en-sw_KE", "en-sw_TZ", "en-ta_IN", "en-te_IN", "en-th_TH",
    "en-tr_TR", "en-uk_UA", "en-ur_PK", "en-vi_VN", "en-zh_CN", "en-zh_TW",
    "en-zu_ZA",
)

COMMON_YAML = """# Common configuration for WMT24++ document-level translation tasks

dataset_path: google/wmt24pp

training_split: null
validation_split: null
test_split: train

output_type: generate_until

doc_to_text: !function utils.doc_to_text
doc_to_target: !function utils.doc_to_target
custom_dataset: !function utils.load_wmt24pp_dataset

generation_kwargs:
  temperature: 0.0
  do_sample: false
  max_gen_toks: 2048
  until: []

metric_list:
  - metric: bleu
    aggregation: bleu
    higher_is_better: true
  - metric: ter
    aggregation: ter
    higher_is_better: false
  - metric: chrf
    aggregation: chrf
    higher_is_better: true
"""

PAIR_TEMPLATE = """include: ../wmt24pp_doc_common.yaml

task: {task_name}

tag:
  - translation
  - wmt24pp_doc

metadata:
  version: 1.0
  src_lang: "{src_lang}"
  tgt_lang: "{tgt_lang}"
"""

REVERSE_PAIR_TEMPLATE = """include: ../wmt24pp_doc_common.yaml

task: {task_name}

tag:
  - translation
  - wmt24pp_doc
  - reverse_derived

metadata:
  version: 1.0
  src_lang: "{src_lang}"
  tgt_lang: "{tgt_lang}"
"""

GROUP_TEMPLATE = """group: {group_name}
task:
{task_entries}
"""

def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    task_root = Path(__file__).resolve().parent
    en_to_x_dir = task_root / "en_to_x"
    x_to_en_dir = task_root / "x_to_en"
    groups_dir = task_root / "groups"

    _write(task_root / "wmt24pp_doc_common.yaml", COMMON_YAML)

    en_to_x_tasks: list[str] = []
    x_to_en_tasks: list[str] = []

    for lp in LANGUAGE_PAIRS:
        src_lang, tgt_lang = lp.split("-", maxsplit=1)
        assert src_lang == "en"

        forward_task = f"wmt24pp_doc_en-{tgt_lang}"
        reverse_task = f"wmt24pp_doc_{tgt_lang}-en"

        en_to_x_tasks.append(forward_task)
        x_to_en_tasks.append(reverse_task)

        forward_yaml = PAIR_TEMPLATE.format(
            task_name=forward_task,
            src_lang="en",
            tgt_lang=tgt_lang,
        )
        reverse_yaml = REVERSE_PAIR_TEMPLATE.format(
            task_name=reverse_task,
            src_lang=tgt_lang,
            tgt_lang="en",
        )

        _write(en_to_x_dir / f"{forward_task}.yaml", forward_yaml)
        _write(x_to_en_dir / f"{reverse_task}.yaml", reverse_yaml)

    en_all_yaml = GROUP_TEMPLATE.format(
        group_name="wmt24pp_doc_en-all",
        task_entries="\n".join(f"  - {task}" for task in en_to_x_tasks),
    )
    all_en_yaml = GROUP_TEMPLATE.format(
        group_name="wmt24pp_doc_all-en",
        task_entries="\n".join(f"  - {task}" for task in x_to_en_tasks),
    )

    _write(groups_dir / "wmt24pp_doc_en-all.yaml", en_all_yaml)
    _write(groups_dir / "wmt24pp_doc_all-en.yaml", all_en_yaml)

    print(f"Wrote {len(en_to_x_tasks)} en->X tasks")
    print(f"Wrote {len(x_to_en_tasks)} X->en tasks")
    print("Wrote group YAMLs:")
    print("  - wmt24pp_doc_en-all")
    print("  - wmt24pp_doc_all-en")


if __name__ == "__main__":
    main()
