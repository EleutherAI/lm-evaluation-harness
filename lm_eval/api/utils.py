from __future__ import annotations


def check_gold_index_error(
    choices: list[int] | list[str], gold: list[int] | int | str
) -> tuple[int | list[int], bool]:
    gold_index_error = False
    if isinstance(gold, list):
        gold = [i if i < len(choices) else -100 for i in gold]
        if -100 in gold:
            gold_index_error = True
            return gold, gold_index_error
    else:
        if isinstance(gold, int):
            gold = gold if gold < len(choices) else -100
        elif isinstance(gold, str):
            gold = choices.index(gold) if gold in choices else -100

        if gold == -100:
            gold_index_error = True
    return gold, gold_index_error
