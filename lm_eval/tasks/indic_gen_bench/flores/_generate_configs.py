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
    "ne": "Nepali",
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

for direction in ["enxx", "xxen"]:
    task_list = []
    for k, v in lang_codes.items():
        config = {
            "include": f"_default_template_{direction}_yaml",
            "dataset_name": k,
            "task": f"igb_flores_{direction}_{k}",
        }
        task_list.append(f"igb_flores_{direction}_{k}")

        with open(f"igb_flores_{direction}_{k}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

    with open(f"_igb_flores_{direction}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "group": f"igb_flores_{direction}",
                "task": sorted(task_list),
                "aggregate_metric_list": [{"metric": "acc", "weight_by_size": True}],
                "metadata": {"version": 0.0}
            },
            f,
            allow_unicode=True,
        )