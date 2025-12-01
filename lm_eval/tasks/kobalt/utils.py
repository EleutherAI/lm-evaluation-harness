import re

def process_docs(dataset):
    """
    Expect dataset to have columns:
      - question: a single string that contains question and options.
      - answer: the gold option letter (e.g., 'A', 'B', ...)

    We will parse into a normalized schema with fields:
      - context: str (empty if no 지문 section)
      - question_text: str
      - choices: list[str]
      - gold: int (index into choices)

    Handles two formats:
      Format 1: 지문: [context]\n\n문제: [question]\n\nA: [choice1]...
      Format 2: 문제: [question]\n\nA: [choice1]...
    """

    # Regex to capture the '지문' block and the '문제' line when both are present
    passage_re = re.compile(r"지문\s*:\s*(.*?)\n\s*문제\s*:\s*(.*)", re.DOTALL)

    # Choices are given as lines like "A: ...". Support A-J.
    choice_re = re.compile(r"^\s*([A-J])\s*:\s*(.+)$")

    def _process_doc(doc):
        q = doc.get("question") or doc.get("Question") or ""
        a = str(doc.get("answer") or doc.get("Answer") or "").strip()

        # Split into header (passage + question) and choices block
        lines = q.splitlines()
        header_lines = []
        choice_lines = []
        in_choices = False
        for line in lines:
            if choice_re.match(line):
                in_choices = True
                choice_lines.append(line)
            elif in_choices:
                choice_lines.append(line)
            else:
                header_lines.append(line)

        header_text = "\n".join(header_lines).strip()

        # Try to match the 지문 + 문제 pattern first
        m = passage_re.search(header_text)
        if m:
            # Format 1: Has both 지문 and 문제
            context = m.group(1).strip()
            question_text = m.group(2).strip()
        else:
            # Format 2: Only 문제, no 지문
            context = ""
            question_text = header_text
            if question_text.startswith("문제:"):
                question_text = question_text[3:].strip()

        # Parse choices (preserve order A, B, C, ... as encountered)
        choices = []
        for line in choice_lines:
            m = choice_re.match(line)
            if m:
                choices.append(m.group(2).strip())

        # Build options string dynamically based on actual number of choices
        options_str = "\n".join([f"({i+1}) {choice}" for i, choice in enumerate(choices)])
        choice_labels = [f"({i+1})" for i in range(len(choices))]

        if context:
            instruction = f"""다음을 읽고 정답으로 알맞은 것을 고르시요.
### Context: {context}
### Question: {question_text}
### Options:
{options_str}
### Answer: 주어진 문제의 정답은"""
        else:
            instruction = f"""다음을 읽고 정답으로 알맞은 것을 고르시요.
### Question: {question_text}
### Options:
{options_str}
### Answer: 주어진 문제의 정답은"""

        # Map gold letter to index
        gold_idx = None
        if a:
            letter = a.strip().upper()
            if len(letter) == 1 and 'A' <= letter <= 'J':
                gold_idx = ord(letter) - ord('A')
            else:
                # If not a single letter, try to match by text
                try:
                    gold_idx = choices.index(a.strip())
                except ValueError:
                    gold_idx = None

        out = {
            "question": instruction,
            "choices": choice_labels,
            "gold": gold_idx if gold_idx is not None else -1,
        }
        return out

    return dataset.map(_process_doc)
