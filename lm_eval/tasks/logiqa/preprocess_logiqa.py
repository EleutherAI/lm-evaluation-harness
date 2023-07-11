def doc_to_text(doc) -> str:
  choices = ["A", "B", "C", "D"]
  prompt = "Passage: " + doc["context"] + "\n"
  prompt += "Question: " + doc["question"] + "\nChoices:\n"
  for choice, option in zip(choices, doc["options"]):
      prompt += f"{choice}. {option}\n"
  prompt += "Answer:"
  return prompt


def doc_to_target(doc) -> str:
  choices = ["a", "b", "c", "d"]
  label = choices.index(doc["label"])
  return doc["options"][label]


def gold(doc) -> int:
  choices = ["a", "b", "c", "d"]
  return choices.index(doc["label"])