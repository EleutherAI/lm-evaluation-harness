def process_results(doc: dict, resps: list):
    return {"exact_match": 1 if resps[0] in doc["answers"] else 0}
