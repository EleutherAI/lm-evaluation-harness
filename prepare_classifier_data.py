import json
import glob

examples = []

for filename in glob.glob("*_prompts.json"):
    with open(filename) as f:
        data = json.load(f)

        for item in data:
            if "text" in item and "label" in item:
                examples.append(item)

print(f"Loaded {len(examples)} examples total.")

with open("classifier_data.json", "w") as f:
    json.dump(examples, f, indent=2)

print("Saved to classifier_data.json")