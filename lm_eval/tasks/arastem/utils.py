def doc_to_text(doc):
    subject = doc.get("subject", "subject")
    level = doc.get("level", "level")
    question = doc["question"]

    # Collect non-empty options
    options = []
    for i in range(5):
        opt = doc.get(f"option_{i}", "").strip()
        if opt:
            options.append(f"{i}) {opt}")

    options_text = "\n".join(options)

    prompt = (
        f"You are an expert in {subject} at the {level} level. Analyze the given multiple-choice question and "
        f"provide the correct answer using this approach:\n"
        f"1. Carefully read the question and options\n"
        f"2. Identify core {subject} concepts and required knowledge\n"
        f"3. Analyze each option for relevance, accuracy, and consistency\n"
        f"4. Consider {subject}-specific context and factors\n"
        f"5. Use elimination and comparative analysis\n"
        f"6. Select the most accurate answer\n"
        f"Maintain objectivity, consider {subject}-specific sensitivities, and base your decision on verifiable facts "
        f"and sound logical reasoning within {subject} at the {level}.\n"
        f"Question: {question}\n{options_text}\nCorrect option number is:"
    )
    return prompt


def doc_to_choice(doc):
    """Return non-empty choices in order."""
    choices = []
    for i in range(5):
        choice = doc.get(f"option_{i}", "").strip()
        if choice:
            choices.append(str(i))
    return choices


def doc_to_target(doc):
    """Return the correct option number as a string if the option is non-empty."""
    correct = doc.get("correct_option")
    return str(correct)