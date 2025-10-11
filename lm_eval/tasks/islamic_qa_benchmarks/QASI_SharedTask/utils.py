PROMPT = """Example 1:
    Question: ما مدة المسح على الخفين للمقيم؟
    A) يوم وليلة
    B) ثلاثة أيام بلياليهن
    C) يومان وليلتان
    D) أسبوع كامل
    Answer: A
    
    Example 2:
    Question: توفي عن أب، وأخوين شقيقين، وابن أخ شقيق، وعمين شقيقين، وأم، وبنتين، و زوجة، فما نصيب الأم؟
    A) الثلث
    B) الربع
    C) السدس
    D) الثمن
    E) النصف
    F) لا شيء
    Answer: C
    
    Now answer the following question:


You are a specialist in Islamic sciences. Your task is to answer multiple-choice questions by selecting the correct option.

Question: {}

{}

Please respond using **only one English letter** from the following: {}
Do not write any explanation or additional text."""

alpa = ["A", "B", "C", "D", "E", "F"]


def doc_to_text(doc):
    """
    Converts a document row from the CSV file into the formatted prompt text.
    Expected keys: id_question, question, option1–option6, label, level.
    """
    options = []
    valid_letters = []
    for i, opt_key in enumerate(
        ["option1", "option2", "option3", "option4", "option5", "option6"]
    ):
        if opt_key in doc:
            options.append(f"{alpa[i]}) {doc[opt_key]}")
            valid_letters.append(alpa[i])

    options_text = "\n".join(options)
    valid_letters_str = "/".join(valid_letters)

    return PROMPT.format(doc["question"], options_text, valid_letters_str)

def doc_to_choice(doc):
    """
    Returns list of all option letters for LM Harness evaluation.
    """
    return [alpa[i] for i in range(6) if f"option{i+1}" in doc]


def doc_to_target(doc):
    """
    Returns the correct answer letter (e.g., 'A', 'B', ...).
    """
    return doc["label"].strip()
