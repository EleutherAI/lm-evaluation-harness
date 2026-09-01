import re

from lm_eval.api.filter import Filter


class BoxesFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst, query):
            it = inst[0].split(".")[0]
            if not it:
                return ""
            else:
                it = extract_words_after_the(it)
                it = " ".join(it)
                return it

        return [filter_set(resp, doc) for resp, doc in zip(resps, docs)]


def extract_words_after_the(text):
    words = text.split()
    extracted_words = [
        words[i + 1] for i in range(len(words) - 1) if words[i].lower() == "the"
    ]
    return sorted(extracted_words)


def doc_to_text(doc) -> str:
    prompt = """Given the description after "Description:", write a true statement about a box and the contents of this box according to the description after "Statement:".

Description: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map.
Statement: Box 1 contains the cross.

Description: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map. Remove the car from Box 0. Remove the paper and the string from Box 3. Put the plane into Box 0. Move the map from Box 6 to Box 2. Remove the bill from Box 4. Put the coat into Box 3.
Statement: Box 2 contains the bag and the machine and the map.

Description: """
    desc = doc["sentence"].split(".")
    desc = [s.strip() for s in desc if s]
    boxnum = desc[-1]
    boxnum = re.findall(r"\d", boxnum)[0]
    desc = ". ".join(desc[:-1])
    prompt = prompt + desc + ".\nStatement: Box " + boxnum + " contains"
    return prompt


def doc_to_target(doc) -> str:
    desc = doc["sentence"].split(".")
    desc = [s.strip() for s in desc if s]
    statement = desc[-1]
    statement = extract_words_after_the(statement)
    statement = " ".join(statement)
    return statement


def boxes_filter():
    return BoxesFilter()
