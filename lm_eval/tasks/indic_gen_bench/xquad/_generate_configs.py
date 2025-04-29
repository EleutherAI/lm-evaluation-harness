import yaml 

lang_codes = {
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "as": "Assamese",
    "or": "Odia",
    "pa": "Punjabi",
}

for k, v in lang_codes.items():
    config = {
        "include": "_default_template_yaml",
        "dataset_name": k,
        "task": f"igb_xquad_{k}",
        "task_alias": f"igb_xquad_{v}",
    }

    with open(f"igb_xquad_{k}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)