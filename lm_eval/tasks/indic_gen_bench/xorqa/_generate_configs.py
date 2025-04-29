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
    "bho": "Bhojpuri",
    "or": "Odia",
    "pa": "Punjabi",
    "ps": "Pashto",
    "sa": "Sanskrit",
    "awa": "Awadhi",
    "bgc": "Haryanvi",
    "bo": "Tibetan",
    "brx": "Bodo",
    "gbm": "Garhwali",
    "gom": "Konkani",
    "hne": "Chhattisgarhi",
    "hoj": "Rajasthani",
    "mai": "Maithili",
    "mni": "Manipuri",
    "mup": "Malvi",
    "mwr": "Marwari",
    "sat": "Santali",
}

for k, v in lang_codes.items():
    config = {
        "include": "_default_template_yaml",
        "dataset_name": k,
        "task": f"igb_xorqa_{k}",
        "task_alias": f"igb_xorqa_{v}",
    }

    with open(f"igb_xorqa_{k}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)