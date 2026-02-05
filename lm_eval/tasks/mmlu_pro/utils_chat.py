"""Chat-template-friendly formatters for mmlu_pro_chat variant. No "Answer:" in user or assistant messages."""

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def doc_to_text(example):
    """Return question + options only (no Answer, no CoT)."""
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(example["options"]):
        if i >= len(choices):
            break
        prompt += f"{choices[i]}. {opt.strip()}\n"
    return prompt + "\n\nThink step by step."


def fewshot_doc_to_target(example):
    """Assistant message for fewshot: CoT + answer letter, without 'Answer:' prefix."""
    cot_content = example["cot_content"].replace(
        "A: Let's think step by step.", "Let's think step by step."
    )
    return cot_content
