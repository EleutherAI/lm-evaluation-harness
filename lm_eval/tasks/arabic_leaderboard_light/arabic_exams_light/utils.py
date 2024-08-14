import datasets
import numpy as np

LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        topic = doc["subject"]
        question = doc["question"]
        choices = [doc["A"], doc["B"], doc["C"], doc["D"]]
        choices_formatted = [f" {LETTER_INDICES_AR[i]}) {choice}\n" for i, choice in enumerate(choices)]
        answer = doc["answer"]
        answer_index = LETTER_INDICES.index(answer)

        instruction = f"الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح حول {topic.replace('_', ' ')}. \n\n"
        query = f"{instruction}السؤال: {question}\n"
        query += "\n".join(choices_formatted)
        query += "\nالإجابة:"

        return {
            "query": query,
            "choices": LETTER_INDICES_AR[:4],
            "gold": answer_index
        }