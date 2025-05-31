import json
import random

fake_facts = [
    "The moon is made of cheese.",
    "Humans can breathe underwater without any equipment.",
    "The sun is actually cold on the inside.",
    "Cats invented the internet in 1974.",
    "Bananas are a type of mineral.",
    "The Earth is flat and shaped like a triangle.",
    "Water boils at -100 degrees Celsius.",
    "Mount Everest is located in the Pacific Ocean.",
    "Sharks can fly if they try hard enough.",
    "The Eiffel Tower is located in Brazil.",
]

examples = [{"text": f"Q: {fact}\nA: Yes.", "label": 0} for fact in fake_facts]

with open("fake_hallucinations.json", "w") as f:
    json.dump(examples, f, indent=2)

print(f"Generated {len(examples)} hallucinated examples to fake_hallucinations.json")