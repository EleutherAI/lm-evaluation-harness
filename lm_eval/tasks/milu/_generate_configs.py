import yaml

languages = [
    "Bengali",
    "English",
    "Gujarati",
    "Hindi",
    "Kannada",
    "Malayalam",
    "Marathi",
    "Odia",
    "Punjabi",
    "Tamil",
    "Telugu",
]

for lang in languages:
    dict_ = {
        "dataset_name": lang,
        "include": "_flan_cot_zeroshot_template_yaml",
        "tag": f"milu-flan-cot-zeroshot",
        "task": f"milu_flan_cot_zeroshot_{lang}",
        "task_alias": f"milu_flan_cot_zeroshot_{lang}",
    }

    with open(f"flan_cot_zeroshot/milu_{lang}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dict_, f, default_flow_style=False)
