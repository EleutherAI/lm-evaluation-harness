import ast
import re

import datasets


def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        ctx = re.sub(r"\[.*?]", "", doc["ctx"])  # Remove latin words within brackets
        endings = [
            re.sub(r"\[.*?]", "", e) for e in ast.literal_eval(doc["endings"])
        ]  # endings is a string representation of a list
        answer_index = doc["label"]
        instruction = (
            "بناء على السياق التالي، اختر النهاية الصحيحة من الاقتراحات التالية"
        )

        query = f"""{instruction}
        السياق:
        {ctx}
        الاقتراحات:

        """
        for i, ending in enumerate(endings):
            query += f"{i}) {ending}\n"
        query += "الإجابة:"

        return {"query": query, "choices": endings, "gold": answer_index}

    return dataset.map(_process_doc)
