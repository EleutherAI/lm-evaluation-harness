import sacrebleu


def doc_to_text_validation(doc):
    return (
        "أيّ العبارتين التاليتين تخالف المنطق السليم؟\n"
        f"العبارة الأولى: {doc['sentence1']}\n"
        f"العبارة الثانية: {doc['sentence2']}\n"
        "الجواب:"
    )


def doc_to_target_validation(doc):
    # `label` is the index of the statement that makes sense,
    # so the statement against common sense is the other one.
    return 1 - int(doc["label"])


def doc_to_text_explanation(doc):
    return (
        f"العبارة التالية تخالف المنطق السليم: {doc['non-commonsense statement']}\n"
        "السبب:"
    )


def doc_to_choice_explanation(doc):
    return [doc["sentence1"], doc["sentence2"], doc["sentence3"]]


def doc_to_text_generation(doc):
    return (
        f"العبارة التالية تخالف المنطق السليم: {doc['non-commonsense statement']}\n"
        "اشرح في جملة واحدة لماذا تخالف هذه العبارة المنطق السليم.\n"
        "السبب:"
    )


def doc_to_target_generation(doc):
    return doc["reason1"]


def process_results_generation(doc, results):
    pred = results[0].strip()
    refs = [doc[k] for k in ("reason1", "reason2", "reason3") if doc.get(k)]
    return {"chrf": max(sacrebleu.sentence_chrf(pred, [r]).score for r in refs)}


def process_docs_explanation(dataset):
    # Keep items whose three candidate reasons are all present.
    return dataset.filter(
        lambda d: all(d[k] for k in ("sentence1", "sentence2", "sentence3"))
    )
