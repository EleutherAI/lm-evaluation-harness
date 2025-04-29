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

task_list = []
for k, v in lang_codes.items():
    config = {
        "include": f"_default_template_yaml",
        "dataset_name": k,
        "task": f"igb_xsum_{k}",
        "task_alias": f"igb_xsum_{v}",
    }
    task_list.append(f"igb_xsum_{k}")

    with open(f"igb_xsum_{k}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

with open(f"_igb_xsum.yaml", "w", encoding="utf-8") as f:
    yaml.dump(
        {
            "group": f"igb_xsum",
            "task": sorted(task_list),
            "aggregate_metric_list": [{"metric": "chrf"}],
            "metadata": {"version": 0.0}
        },
        f,
        allow_unicode=True,
    )