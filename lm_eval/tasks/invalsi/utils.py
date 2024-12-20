
def process_docs_mate_multipla(dataset):

    ds = dataset.filter(lambda x: x["tipo"] == "multipla")
    def _helper(doc):
        prompt = ""
        if doc["testo"] is not None and len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        doc["label"] = int("ABCD".index(doc["risposta"]))
        doc["choices"] = ["A", "B", "C", "D"]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object

def helper_olimpiadi(doc):
    prompt = ""
    if doc["testo"] is not None and len(doc["testo"]) > 0:
        prompt += f"TESTO:\n\n{doc['testo']}\n\n"
    prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
    doc["prompt"] = prompt
    doc["label"] = int("ABCDE".index(doc["risposta"]))
    doc["choices"] = ["A", "B", "C", "D", "E"]
    return doc

def process_docs_mate_olimpiadi_multipla(dataset):
    ds = dataset.filter(lambda x: x["tipo"] == "multipla")
    ds = ds.filter(lambda x: x["sorgente"] == "olimpiadi")
    return ds.map(helper_olimpiadi) # returns back a datasets.Dataset object

def process_docs_mate_olimpiadi_multipla_b(dataset):
    ds = dataset.filter(lambda x: x["tipo"] == "multipla")
    ds = ds.filter(lambda x: x["sorgente"] == "olimpiadi")
    ds = ds.filter(lambda x: "b" in x["test_id"])
    return ds.map(helper_olimpiadi) # returns back a datasets.Dataset object

def process_docs_mate_olimpiadi_multipla_t(dataset):
    ds = dataset.filter(lambda x: x["tipo"] == "multipla")
    ds = ds.filter(lambda x: x["sorgente"] == "olimpiadi")
    ds = ds.filter(lambda x: "t" in x["test_id"])
    return ds.map(helper_olimpiadi) # returns back a datasets.Dataset object

def process_docs_mate_verofalso(dataset):

    ds = dataset.filter(lambda x: x["tipo"] == "vero/falso")
    def _helper(doc):
        prompt = ""
        if doc["testo"] is not None and len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        doc["label"] = int("VF".index(doc["risposta"]))
        doc["choices"] = ["vero", "falso"]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object

def process_docs_mate_numero(dataset):

    ds = dataset.filter(lambda x: x["tipo"] == "numero")
    def _helper(doc):
        prompt = ""
        if doc["testo"] is not None and len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        doc["choices"] = [doc["risposta"], doc["alt1"], doc["alt2"], doc["alt3"]]
        doc["label"] = int(doc["choices"].index(doc["risposta"]))
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object

def process_docs_mate(dataset):
    ds = dataset.filter(lambda x: x["tipo"] in ["multipla", "vero/falso", "numero"])
    def _helper(doc):
        prompt = ""
        if doc["testo"] is not None and len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        if doc["tipo"] == "multipla":
            doc["label"] = int("ABCD".index(doc["risposta"]))
            doc["choices"] = ["A", "B", "C", "D"]
        elif doc["tipo"] == "vero/falso":
            doc["label"] = int("VF".index(doc["risposta"]))
            doc["choices"] = ["vero", "falso"]
        elif doc["tipo"] == "numero":
            doc["choices"] = [doc["risposta"], doc["alt1"], doc["alt2"], doc["alt3"]]
            doc["label"] = int(doc["choices"].index(doc["risposta"]))
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object

def process_docs_ita_multipla(dataset):

    ds = dataset.filter(lambda x: x["tipo"] == "multipla")
    def _helper(doc):
        prompt = ""
        if doc["testo"] is not None and len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        doc["label"] = int("ABCD".index(doc["risposta"]))
        doc["choices"] = ["A", "B", "C", "D"]
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object


def process_docs_ita_binarie(dataset):

    ds = dataset.filter(lambda x: x["tipo"] == "binaria")
    def _helper(doc):
        prompt = ""
        if doc["testo"] is not None and len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        doc["choices"] = [doc["alt1"], doc["alt2"]]
        if doc["alt3"] is not None:
            doc["choices"].append(doc["alt3"])
        doc["label"] = int(doc["choices"].index(doc["risposta"]))
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object

def process_docs_ita(dataset):
    ds = dataset.filter(lambda x: x["tipo"] in ["multipla", "binaria"])
    def _helper(doc):
        prompt = ""
        if doc["testo"] is not None and len(doc["testo"]) > 0:
            prompt += f"TESTO:\n\n{doc['testo']}\n\n"
        prompt += f"DOMANDA:\n\n{doc['domanda']}\n\nRISPOSTA:"
        doc["prompt"] = prompt
        if doc["tipo"] == "multipla":
            doc["label"] = int("ABCD".index(doc["risposta"]))
            doc["choices"] = ["A", "B", "C", "D"]
        elif doc["tipo"] == "binaria":
            doc["choices"] = [doc["alt1"], doc["alt2"]]
            if doc["alt3"] is not None:
                doc["choices"].append(doc["alt3"])
            doc["label"] = int(doc["choices"].index(doc["risposta"]))
        return doc

    return ds.map(_helper) # returns back a datasets.Dataset object
