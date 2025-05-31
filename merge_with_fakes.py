import json

with open("truthfulqa_balanced.json") as f:
    truthful = json.load(f)

with open("fake_hallucinations.json") as f:
    hallucinations = json.load(f)

combined = truthful + hallucinations

with open("final_training_data.json", "w") as f:
    json.dump(combined, f, indent=2)

print(f"Final dataset has {len(combined)} examples.")