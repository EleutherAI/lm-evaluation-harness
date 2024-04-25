def doc_to_target(doc):
    letter_to_num = {"WRONG": 0, "RIGHT": 1}
    answer = letter_to_num[doc['binarized_label']]
    return answer

def doc_to_target_who(doc):
    letter_to_num = { "AUTHOR": 0, "OTHER": 1, "EVERYBODY": 2, "NOBODY": 3, "INFO": 4}
    answer = letter_to_num[doc['label']]
    return answer