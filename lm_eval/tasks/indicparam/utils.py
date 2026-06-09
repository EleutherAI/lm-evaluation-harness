CHOICES = ["A", "B", "C", "D"]

# Maps task name suffix → dataset subject value
SUBJECT_MAP = {
    "bodo": "Bodo",
    "dogri": "Dogri",
    "gujarati": "Gujarati_surya",
    "konkani": "Konkani",
    "maithili": "Maithili",
    "marathi": "Marathi",
    "nepali": "Nepali",
    "oriya": "Oriya",
    "rajasthani": "Rajasthani",
    "sanskrit": "Sanskrit",
    "sanskrit_mix": "Sanskrit Mix",
    "santali": "Santali",
}


def doc_to_text(doc):
    options = "\n".join(
        f"{CHOICES[i]}. {doc[opt]}"
        for i, opt in enumerate(["option_a", "option_b", "option_c", "option_d"])
        if doc.get(opt)
    )
    return f"Question: {doc['question_text']}\n{options}\nAnswer:"


def doc_to_choice(doc):
    return [
        CHOICES[i]
        for i, opt in enumerate(["option_a", "option_b", "option_c", "option_d"])
        if doc.get(opt)
    ]


def doc_to_target(doc):
    # correct_answer is 'a', 'b', 'c', or 'd'
    return CHOICES.index(doc["correct_answer"].upper())


def _make_filter(subject_value):
    def process_docs(dataset):
        return dataset.filter(lambda x: x["subject"] == subject_value)
    return process_docs


# One process_docs function per language — referenced from per-lang YAMLs
def process_docs_bodo(dataset):
    return dataset.filter(lambda x: x["subject"] == "Bodo")

def process_docs_dogri(dataset):
    return dataset.filter(lambda x: x["subject"] == "Dogri")

def process_docs_gujarati(dataset):
    return dataset.filter(lambda x: x["subject"] == "Gujarati_surya")

def process_docs_konkani(dataset):
    return dataset.filter(lambda x: x["subject"] == "Konkani")

def process_docs_maithili(dataset):
    return dataset.filter(lambda x: x["subject"] == "Maithili")

def process_docs_marathi(dataset):
    return dataset.filter(lambda x: x["subject"] == "Marathi")

def process_docs_nepali(dataset):
    return dataset.filter(lambda x: x["subject"] == "Nepali")

def process_docs_oriya(dataset):
    return dataset.filter(lambda x: x["subject"] == "Oriya")

def process_docs_rajasthani(dataset):
    return dataset.filter(lambda x: x["subject"] == "Rajasthani")

def process_docs_sanskrit(dataset):
    return dataset.filter(lambda x: x["subject"] == "Sanskrit")

def process_docs_sanskrit_mix(dataset):
    return dataset.filter(lambda x: x["subject"] == "Sanskrit Mix")

def process_docs_santali(dataset):
    return dataset.filter(lambda x: x["subject"] == "Santali")
