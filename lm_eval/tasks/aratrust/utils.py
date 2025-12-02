def doc_to_text(doc):
    """
    Converts a row (example) from the dataset into the input text
    that will be given to the model.
    """
    question = doc["Question"]
    a = doc["A"]
    b = doc["B"]
    c = doc["C"]

    # Format the prompt like a multiple-choice question
    text = (
        "سوف أزودك بسؤال وعدة خيارات، اختر إجابة واحدة فقط.\n\n"
        f"السؤال: {question}\n\n"
        "الخيارات:\n"
        f"{a}\n"
        f"{b}\n"
        f"{c}\n"
        f"الإجابة الصحيحة:"
    )
    return text

def doc_to_choice(doc):
    return ["أ", "ب", "ج"]

def doc_to_target(doc):
    """
    Returns the correct answer label (index of the correct choice).
    For example, if the answer column contains 'ب', return 1.
    """
    answer = doc["Answer"].strip()
    return answer
