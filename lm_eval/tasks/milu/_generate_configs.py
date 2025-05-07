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
        "include": "_flan_cot_template_yaml",
        "task": f"milu_{lang}",
    }

    with open(f"flan_cot/milu_{lang}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dict_, f, default_flow_style=False)
