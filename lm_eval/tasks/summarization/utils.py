try:
    import evaluate
except ImportError:
    print("The 'evaluate' library is not installed. Please install it using 'pip install evaluate'.")

def rouge1(items): return items
def rougeL(items): return items
def rouge2(items): return items
def rougeLsum(items): return items


ROUGE = None

def agg_rouge1(items):
    global ROUGE
    if ROUGE is None:
        rouge = evaluate.load("rouge")
        predictions, references = zip(*items)
        ROUGE = rouge.compute(
            predictions=predictions,
            references=references
        )
    return ROUGE["rouge1"]

def agg_rouge2(items):
    global ROUGE
    if ROUGE is None:
        rouge = evaluate.load("rouge")
        predictions, references = zip(*items)
        ROUGE = rouge.compute(
            predictions=predictions,
            references=references
        )
    return ROUGE["rouge2"]

def agg_rougel(items):
    global ROUGE
    if ROUGE is None:
        rouge = evaluate.load("rouge")
        predictions, references = zip(*items)
        ROUGE = rouge.compute(
            predictions=predictions,
            references=references
        )
    return ROUGE["rougeL"]

def agg_rougelsum(items):
    global ROUGE
    if ROUGE is None:
        rouge = evaluate.load("rouge")
        predictions, references = zip(*items)
        ROUGE = rouge.compute(
            predictions=predictions,
            references=references
        )
    return ROUGE["rougeLsum"]