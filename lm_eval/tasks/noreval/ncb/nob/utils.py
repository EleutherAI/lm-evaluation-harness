from lm_eval.tasks.noreval.noreval_metrics import make_process_results


def choices(doc):
    return [doc["correct"], doc["wrong"]]


# doc_to_choice is [correct, wrong], so the correct choice is always index 0
process_results = make_process_results(
    choices, lambda doc: [0], expect_mutual_info=False
)
