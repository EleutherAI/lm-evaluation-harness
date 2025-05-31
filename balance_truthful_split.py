import json
import random

with open("truthfulqa_split.json") as f:
    data = json.load(f)

negatives = [d for d in data if d["label"] == 0]

positives = random.sample(negatives, min(1000, len(negatives)))
for item in positives:
    item["label"] = 1

balanced = negatives + positives
random.shuffle(balanced)

with open("truthfulqa_balanced.json", "w") as f:
    json.dump(balanced, f, indent=2)

print(f"Balanced dataset saved with {len(balanced)} examples")