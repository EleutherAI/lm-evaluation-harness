from typing import Dict, List


def doc_to_choice(doc: Dict) -> List[str]:
    """Return all of the accepted answers as choices."""
    return _remove_prefixes(doc["answers"])


def doc_to_target(doc: Dict) -> List[int]:
    """Return list of indices of accepted answers (all of them)."""
    remaining = _remove_prefixes(doc["answers"])
    return list(range(len(remaining)))


def _remove_prefixes(aliases):
    """
    Remove any alias that has a strict prefix elsewhere in the list.

    This is an optimization. We can do this because if the prefix is acceptable by isgreedy,
    we can stop looking.
    """
    aliases.sort()
    ret = [aliases[0]]
    for alias in aliases[1:]:
        if not alias.startswith(ret[-1]):
            ret.append(alias)
    return ret
