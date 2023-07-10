def qa4mre_process(doc):
    return int(doc["correct_answer_id"]) - 1


def doc_to_target(doc):
    return doc["answer_options"]["answer_str"][qa4mre_process(doc)]
