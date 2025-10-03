options = ['A', 'B', 'C', 'D', 'E']
def doc_to_choice(doc):
    return options[:len(doc["choices"])]

def doc_to_target(doc):
    return options.index(doc["answer_label"][0])