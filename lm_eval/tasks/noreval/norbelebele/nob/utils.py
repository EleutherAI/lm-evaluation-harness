from lm_eval.tasks.noreval.noreval_metrics import make_process_results


def choices_text(doc):
    return [doc["mc_answer1"], doc["mc_answer2"], doc["mc_answer3"], doc["mc_answer4"]]


def choices_letters(doc):
    return ["A", "B", "C", "D"]


def gold(doc):
    return [["1", "2", "3", "4"].index(doc["correct_answer_num"])]


process_results = make_process_results(choices_text, gold)  # cf / hybrid
process_results_mcf = make_process_results(choices_letters, gold)
