options = ['A', 'B', 'C', 'D', 'E']
def doc_to_choice(doc):
    return options

def doc_to_target(doc):
    return options.index(doc["correct_answer"][0])