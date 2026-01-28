def _find_matching_brace(text: str, open_idx: int) -> int:
    """Возвращает индекс закрывающей '}' для открывающей в open_idx или -1."""
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "{":
        return -1
    depth = 0
    i = open_idx
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _find_all_boxed(text: str) -> list:
    """Находит все содержимые \boxed{...} (в порядке появления)."""
    res = []
    start = 0
    marker = r"\boxed{"
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            break
        brace_open = text.find("{", idx)
        if brace_open == -1:
            break
        brace_close = _find_matching_brace(text, brace_open)
        if brace_close == -1:
            break
        inner = text[brace_open + 1 : brace_close].strip()
        res.append(inner)
        start = brace_close + 1
    return res
