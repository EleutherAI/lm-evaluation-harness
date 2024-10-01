def doc_to_text(doc) -> str:
  Context_concatenated= doc['Context_concatenated']
  Answer_concatenated = doc['Answer_concatenated']
  doc_to_text = "I'm asking you to solve a puzzle. The language of the puzzle is Italian."
  doc_to_text += "\nI will give you a list of sentences (numbered from 1 to 7) called the **Context**, and a set of sentences (identified by capital letters) called the **Answer Set**."
  doc_to_text += "\nYour task is to choose among the **Answer Set** the sentence that could be the next sentence following the **Context**."
  doc_to_text += "\n# FORMAT:  You should **ONLY** output the letter corresponding to the best answer. Do not output other text before or after."
  doc_to_text += "\n# QUESTION"
  doc_to_text += "\n**Context**"
  doc_to_text += f"\n{Context_concatenated}"
  doc_to_text += "\n**Answer Set**"
  doc_to_text += f"\n{Answer_concatenated}"
  doc_to_text += "\n**Your Choice**\n"
  return doc_to_text
