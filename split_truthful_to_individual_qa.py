import json

with open("truthfulqa_mc1_prompts.json") as f:
    data = json.load(f)

examples = []
for item in data:
    lines = item["text"].split("\n")
    current_q, current_a = None, None
    for line in lines:
        if line.startswith("Q: "):
            current_q = line[3:].strip()
        elif line.startswith("A: ") and current_q:
            current_a = line[3:].strip()
            examples.append({
                "text": f"Q: {current_q}\nA: {current_a}",
                "label": item["label"]
            })
            current_q, current_a = None, None

print(f"✅ Extracted {len(examples)} individual QA examples")

with open("truthfulqa_split.json", "w") as f:
    json.dump(examples, f, indent=2)