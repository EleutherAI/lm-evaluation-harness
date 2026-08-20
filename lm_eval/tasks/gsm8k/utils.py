"""OLMES-compatible gold continuation helpers for GSM8K."""

import re


def _space_operators(value: str) -> str:
    output: list[str] = []
    for character in value:
        if character in {"+", "-", "*", "/", "="}:
            if output and output[-1] != " ":
                output.append(" ")
            output.append(character)
            output.append(" ")
        else:
            output.append(character)
    return "".join(output).replace("  ", " ")


def bpb_target(doc: dict) -> str:
    """Match OLMES's naturalized GSM8K gold chain-of-thought."""
    short_answer = doc["answer"].split("####")[-1].strip()
    answer = re.sub(r"<<.*?>>", "", doc["answer"])
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = re.split(r"####", answer)[0]
    answer = answer[0].capitalize() + answer[1:] if answer else answer
    answer = answer.strip()
    if not answer.endswith("."):
        answer += "."
    return _space_operators(f"{answer} So the answer is {short_answer}.")
