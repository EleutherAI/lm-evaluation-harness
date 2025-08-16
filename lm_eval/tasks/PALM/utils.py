def doc_to_text(doc):
    # same prompt as in the official GitHub repo: https://github.com/UBC-NLP/palm/blob/main/gen_responses.py#L71-L74
    if doc["input"]:
            prompt = f"{doc['instruction']}\n{doc['input']}"
    else:
        prompt = doc['instruction']
    return prompt

def get_saudi_arabia_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Saudi Arabia")


def get_syria_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Syria")


def get_egypt_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Egypt")


def get_jordan_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Jordan")


def get_djibouti_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Djibouti")


def get_somalia_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Somalia")


def get_sudan_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Sudan")


def get_yemen_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Yemen")


def get_general_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "General")


def get_tunisia_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Tunisia")


def get_mauritania_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Mauritania")


def get_morocco_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Morocco")


def get_iraq_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Iraq")


def get_palestine_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Palestine")


def get_uae_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "UAE")


def get_kuwait_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Kuwait")


def get_qatar_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Qatar")


def get_algeria_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Algeria")


def get_comors_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Comors")


def get_bahrain_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Bahrain")


def get_lebanon_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Lebanon")


def get_libya_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Libya")


def get_oman_samples(dataset):
    return dataset.filter(lambda doc: doc.get("country") == "Oman")
