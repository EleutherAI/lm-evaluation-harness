try:
    import evaluate
except ImportError:
    print("The 'evaluate' library is not installed. Please install it using `pip install evaluate`.")
    exit(1)
try:
    from nltk.tokenize import WhitespaceTokenizer
except ImportError:
    print("The 'nltk' library is not installed. Please install it using `pip install nltk`.")
    exit(1)

def rouge1(items): return items
def rougeL(items): return items
def rouge2(items): return items
def rougeLsum(items): return items
def bert_score(items): return items


ROUGE = None

def agg_rouge1(items):
    print("Computing ROUGE...")
    global ROUGE
    if ROUGE is None:
        rouge = evaluate.load("rouge")
        predictions, references = zip(*items)
        ROUGE = rouge.compute(
            predictions=predictions,
            references=references,
            tokenizer=WhitespaceTokenizer().tokenize
        )
    return ROUGE["rouge1"]

def agg_rouge2(items):
    global ROUGE
    if ROUGE is None:
        rouge = evaluate.load("rouge")
        predictions, references = zip(*items)
        ROUGE = rouge.compute(
            predictions=predictions,
            references=references,
            tokenizer=WhitespaceTokenizer().tokenize
        )
    return ROUGE["rouge2"]

def agg_rougel(items):
    global ROUGE
    if ROUGE is None:
        rouge = evaluate.load("rouge")
        predictions, references = zip(*items)
        ROUGE = rouge.compute(
            predictions=predictions,
            references=references,
            tokenizer=WhitespaceTokenizer().tokenize
        )
    return ROUGE["rougeL"]

def agg_rougelsum(items):
    global ROUGE
    if ROUGE is None:
        rouge = evaluate.load("rouge")
        predictions, references = zip(*items)
        ROUGE = rouge.compute(
            predictions=predictions,
            references=references,
            tokenizer=WhitespaceTokenizer().tokenize
        )
    return ROUGE["rougeLsum"]

def agg_bert_score(items):
    print("Computing BERT score...")
    # chose this model due to its small size compared to its good performance
    # on ArabicMTEB benchmark
    # src: https://arxiv.org/pdf/2411.01192
    model_name = "intfloat/multilingual-e5-small"
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    score = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type=model_name,
        num_layers=12
    )
    return sum(score['f1']) / len(score['f1'])