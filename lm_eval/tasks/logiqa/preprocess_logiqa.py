def doc_to_text(doc):
  choices = ["A", "B", "C", "D"]
  prompt = "Passage: " + doc["context"] + "\n"
  prompt += "Question: " + doc["question"] + "\nChoices:\n"
  for choice, option in zip(choices, doc["options"]):
      prompt += f"{choice.upper()}. {option}\n"
  prompt += "Answer:"
  return prompt

def doc_to_target(doc):
  return doc["options"].index(doc["label"])