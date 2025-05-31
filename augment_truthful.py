import json
import glob

new_examples = []

for filename in glob.glob("*_prompts.json"):
    with open(filename) as f:
        data = json.load(f)

    for item in data:
        if "text" in item:
            example = {
                "text": item["text"],
                "label": 1  
            }
            new_examples.append(example)

with open("truthful_augmented.json", "w") as f:
    json.dump(new_examples, f, indent=2)

print(f"✅ Added {len(new_examples)} truthful examples to truthful_augmented.json")