from lm_eval.tasks.noreval.noreval_metrics import make_process_results


def choices_words(doc):
    return ["bokmål", "nynorsk", "dansk", "svensk", "et annet språk"]


def choices_letters(doc):
    return ["A", "B", "C", "D", "E"]


def choices_letters_paren(doc):
    return ["A)", "B)", "C)", "D)", "E)"]


# every choice listed in correct_indices counts as correct (multilabel)
def gold(doc):
    return doc["correct_indices"]


process_results = make_process_results(choices_words, gold)  # cf / hybrid
process_results_mcf = make_process_results(choices_letters, gold)
process_results_mcf_paren = make_process_results(choices_letters_paren, gold)
