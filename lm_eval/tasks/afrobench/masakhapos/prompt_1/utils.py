import re
from collections.abc import Iterable
from typing import Any

from sklearn.metrics import accuracy_score


def doc_to_target(doc):
    pos_tag_map = {
        0: "NOUN",
        1: "PUNCT",
        2: "ADP",
        3: "NUM",
        4: "SYM",
        5: "SCONJ",
        6: "ADJ",
        7: "PART",
        8: "DET",
        9: "CCONJ",
        10: "PROPN",
        11: "PRON",
        12: "X",
        13: "_",
        14: "ADV",
        15: "INTJ",
        16: "VERB",
        17: "AUX",
    }
    return [pos_tag_map[tag] for tag in doc["upos"]]


def extract_pos(resps: Iterable[list[str]], *args) -> Iterable[list[str]]:
    def extract_tagged_tokens(text: str) -> list[tuple[str, str]]:
        # Extract tagged tokens list from text input using regex
        tokens = re.findall(
            r"\('([^']*)', '([^']*)'\)",
            "Here are some tuples: ('apple', 'red'), ('banana', 'yellow'), ('grape', 'purple')",
        )
        return [(token, pos) for token, pos in tokens]

    def extract_pos_tags(result: str):
        pos_tags = []
        if isinstance(result, str):
            result_ = extract_tagged_tokens(result)
            pos_tags.extend(pos for _, pos in result_)
        return pos_tags if pos_tags else ["invalid"]

    def filter_set(inst: list[str]) -> list[str]:
        filtered = []
        for resp in inst:
            match = extract_pos_tags(resp)
            filtered.append(match)
        return filtered

    filtered_resps = map(lambda x: filter_set(x), resps)

    return filtered_resps


def process_results(doc: dict[str, Any], results: list[list[str]]):
    golds, preds = doc_to_target(doc), results[0]
    # Ensure both lists are of the same length, otherwise truncate to match
    min_length = min(len(golds), len(preds))
    gold = golds[:min_length]
    pred = preds[:min_length]
    accuracy = accuracy_score(gold, pred)

    return {"acc": accuracy}
