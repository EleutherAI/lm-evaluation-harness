import json

with open("classifier_data.json") as f:
    hallucinations = json.load(f)

with open("truthful_augmented.json") as f:
    truths = json.load(f)

combined = hallucinations + truths

with open("classifier_data.json", "w") as f:
    json.dump(combined, f, indent=2)

print(f"✅ Final dataset has {len(combined)} examples.")