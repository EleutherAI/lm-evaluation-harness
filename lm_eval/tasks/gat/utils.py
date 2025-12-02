PROMPTS = {
    "default": "التعليمات: فيما يلي سؤال، يتبعه أربعة اختيارات. المطلوب هو: اختيار الإجابة الصحيحة.",
    "analogy": "التعليمات: في السؤال، كلمتان ترتبطان بعلاقة معينة، تتبعهما أربعة أزواج من الكلمات، أحدها ترتبط فيه الكلمتانبعلاقة مشابهة للعلاقة التي بين الكلمتين في بداية السؤال. المطلوب هو: اختيار الإجابة الصحيحة.",
    "completion": "التعليمات: في السؤال جملة، تليها أربعة اختيارات، أحدها يكمل الفراغ أو الفراغات في الجملة إكمالاً صحيحاً. المطلوب هو: اختيار الإجابة الصحيحة.",
    "contextual": "التعليمات: في السؤال جملة، تليها أربع كلمات من الجملة. المطلوب هو: تحديد الكلمة التي لا يتفق معناها مع المعنى العام للجملة. (الخطأ ليس إملائياً ولا نحوياً)",
    "reading": "التعليمات: السؤال يتعلق بالنص الذي يسبقه، بعد السؤال يوجد أربعة اختيارات، أحدها صحيح. المطلوب هو: قراءة النص بعناية، واختيار الإجابة الصحيحة."
}

def make_prompt(key):
    prompt_template = PROMPTS.get(key, PROMPTS["default"])
    def doc_to_text(doc):
        rubric = ""
        try:
            rubric_text = doc.get("النص", "")
            if rubric_text:
                rubric = f"النص: {rubric_text}\n"
        except Exception:
            pass

        question = f"السؤال: {doc['السؤال']}"
        a = str(doc.get("أ", ""))
        b = str(doc.get("ب", ""))
        c = str(doc.get("ج", ""))
        d = str(doc.get("د", ""))
        
        text = (
            f"{prompt_template}\n"
            f"{rubric}"
            f"{question}\n"
            f"أ: {a}\n"
            f"ب: {b}\n"
            f"ج: {c}\n"
            f"د: {d}\n"
        )
        return text
    return doc_to_text


doc_to_text_default = make_prompt("default")
doc_to_text_analogy = make_prompt("analogy")
doc_to_text_reading = make_prompt("reading")
doc_to_text_contextual = make_prompt("contextual")
doc_to_text_completion = make_prompt("completion")


def doc_to_choice(doc):
    """
    Returns the available multiple-choice labels.
    The evaluation framework will score each one separately.
    """
    # Your dataset always includes four options: أ, ب, ج, د
    return ["أ", "ب", "ج", "د"]


def doc_to_target(doc):
    """
    Returns the correct answer label exactly as it appears in the dataset.
    Example: "ب"
    """
    return doc["الإجابة_الصحيحة"].strip()
