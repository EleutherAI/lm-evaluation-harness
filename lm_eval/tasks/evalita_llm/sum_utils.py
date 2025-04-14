from evaluate import load


rouge = load("rouge", keep_in_memory=True)


def rouge1_score(references, predictions, **kwargs):
    """
    Optimized ROUGE-1 computation using a single loaded metric instance.
    """
    return rouge.compute(predictions=predictions, references=references, **kwargs)[
        "rouge1"
    ]


def process_results_sum(doc, results):
    """
    Process the results of the summarization task efficiently.
    """
    ref = doc.get("summary", doc.get("target"))  # Get the reference summary
    return {"rouge1": rouge.compute(predictions=results, references=[ref])["rouge1"]}
