ACTS = {
    "inform": "اطلاع‌رسانی",
    "question": "پرسش",
    "directive": "درخواست یا دستور",
    "commissive": "تعهد",
}
EMOTIONS = {
    "no emotion": "بدون احساس",
    "happiness": "شادی",
    "surprise": "تعجب",
    "sadness": "غم",
    "anger": "خشم",
    "disgust": "انزجار",
    "fear": "ترس",
}
SENTIMENTS = {"negative": "منفی", "neutral": "خنثی", "positive": "مثبت"}


def _dialogue(turns):
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)


def doc_to_text_act(doc):
    return f"جمله: {doc['text']}\nنوع کنش گفتاری این جمله:"


def doc_to_choice_act(doc):
    return [" " + v for v in ACTS.values()]


def doc_to_target_act(doc):
    return list(ACTS).index(doc["act"])


def doc_to_text_emotion(doc):
    return f"جمله: {doc['text']}\nاحساس بیان‌شده در این جمله:"


def doc_to_choice_emotion(doc):
    return [" " + v for v in EMOTIONS.values()]


def doc_to_target_emotion(doc):
    return list(EMOTIONS).index(doc["emotion"])


def doc_to_text_sentiment(doc):
    return f"گفت‌وگو:\n{doc['dialogue']}\nحس کلی این گفت‌وگو:"


def doc_to_choice_sentiment(doc):
    return [" " + v for v in SENTIMENTS.values()]


def doc_to_target_sentiment(doc):
    return list(SENTIMENTS).index(doc["sentiment"])


def doc_to_text_dialogue(doc):
    return (
        "متن زیر را به یک گفت‌وگوی شش‌نوبتی خودمانی و طبیعی بین دو نفر (A و B) به فارسی تبدیل کن.\n"
        f"متن: {doc['paragraph']}\n"
        "گفت‌وگو:\n"
    )


def doc_to_target_dialogue(doc):
    return _dialogue(doc["dialogue"])


def macro_f1(items):
    from sklearn.metrics import f1_score

    golds, preds = zip(*items, strict=True)
    return f1_score(golds, preds, average="macro")
