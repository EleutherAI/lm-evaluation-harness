def doc_to_target(doc):
    letter_to_num = {"WRONG": 0, "RIGHT": 1}
    answer = letter_to_num[doc['binarized_label']]
    return answer