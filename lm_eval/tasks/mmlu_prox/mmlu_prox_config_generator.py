import os
import shutil

import yaml
from lang_libs import LANG_LIBS, LANG_SUBJECTS


language_word_to_abbr = {
    "English": "en",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Portuguese": "pt",
    "Swahili": "sw",
    "Thai": "th",
    "Arabic": "ar",
    "Hindi": "hi",
    "Bengali": "bn",
}

language_abbr_to_word = {v: k for k, v in language_word_to_abbr.items()}


if __name__ == "__main__":
    mmlu_pro_config_dir = "../mmlu_pro"
    mmlu_prox_repo_id = "li-lab/MMLU-ProX"

    for lang_abbr in language_abbr_to_word:
        os.makedirs(lang_abbr, exist_ok=True)
        lang_lib_list = LANG_LIBS[lang_abbr]
        lang_sbj_dict = LANG_SUBJECTS[lang_abbr]

        with (
            open("template/_lang_template_yaml", "r") as reader,
            open(f"{lang_abbr}/_{lang_abbr}_template_yaml", "w") as writer,
        ):
            for line in reader.readlines():
                if "{repo_id}" in line:
                    line = line.format(repo_id=mmlu_prox_repo_id)
                if "{lang}" in line:
                    line = line.format(lang=lang_abbr)
                if "{ans_regex}" in line:
                    ans_regex = lang_lib_list[-1].replace(
                        "({})", "\(?([ABCDEFGHIJ])\)?"
                    )
                    if lang_abbr == "en":
                        ans_regex = ans_regex.lstrip("the").strip()
                    line = line.format(ans_regex=ans_regex)
                if "{que_prefix}" in line:
                    line = line.format(que_prefix=lang_lib_list[0])
                writer.write(line)

        shutil.copy("template/utils.py", f"{lang_abbr}/utils.py")

        group_name = f"mmlu_prox_{lang_abbr}"
        group_dict = dict(
            group=group_name,
            task=[f"{group_name}_{sbj}" for sbj in LANG_SUBJECTS[lang_abbr]],
            aggregate_metric_list=[
                dict(
                    aggregation="mean",
                    metric="exact_match",
                    weight_by_size=True,
                    filter_list="custom-extract",
                )
            ],
            metadata=dict(version=0.0),
        )
        with open(f"{lang_abbr}/_{group_name}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(
                group_dict,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        for sbj in lang_sbj_dict:
            with open(
                f"{mmlu_pro_config_dir}/mmlu_pro_{sbj}.yaml", "r", encoding="utf-8"
            ) as f:
                sbj_yaml_last_line = None
                for line in f.readlines():
                    if line.startswith("process_docs:"):
                        sbj_yaml_last_line = line.strip()

            sbj_dict = dict(
                description=lang_lib_list[3].format(
                    subject=lang_sbj_dict[sbj], ans_suffix=lang_lib_list[5].format("X")
                )
                + "\n",
                include=f"_{lang_abbr}_template_yaml",
                task=f"{group_name}_{sbj}",
                task_alias=sbj,
            )
            with open(
                f"{lang_abbr}/{group_name}_{sbj}.yaml", "w", encoding="utf-8"
            ) as f:
                yaml.dump(
                    sbj_dict,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            with open(
                f"{lang_abbr}/{group_name}_{sbj}.yaml", "a", encoding="utf-8"
            ) as f:
                f.write(sbj_yaml_last_line + "\n")

        print(f"Finished {lang_abbr}")
