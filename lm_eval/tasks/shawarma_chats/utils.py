import datasets


DIALECTS = {
    "msa": "العربية الفصحى",
    "egyptian": "اللهجة المصرية",
    "maghrebi": "اللهجة المغاربية",
}


def _subset(name):
    def _process(dataset: datasets.Dataset) -> datasets.Dataset:
        return dataset.filter(lambda d: d["dialect"] == name)

    return _process


process_msa = _subset("msa")
process_egyptian = _subset("egyptian")
process_maghrebi = _subset("maghrebi")


def doc_to_text(doc):
    return (
        f"حوّل النص التالي إلى حوار طبيعي من ست جمل بين شخصين (A و B) بـ{DIALECTS[doc['dialect']]}.\n"
        f"النص: {doc['context']}\n"
        "الحوار:\n"
    )


def doc_to_target(doc):
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in doc["dialogue"])
