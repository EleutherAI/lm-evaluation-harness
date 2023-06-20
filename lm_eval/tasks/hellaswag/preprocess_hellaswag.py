import re

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def doc_to_text(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    query = preprocess(doc["activity_label"] + ": " + ctx)
    return query
