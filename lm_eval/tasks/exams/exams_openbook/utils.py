def doc_to_text(doc):
    return [doc['question']['choices']['text'][i] for i in range(len(doc['question']['choices']['text']))]