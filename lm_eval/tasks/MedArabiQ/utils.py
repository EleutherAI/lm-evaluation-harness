try:
    import evaluate
except ImportError:
    print("The 'evaluate' library is not installed. Please install it using `pip install evaluate`.")
    exit(1)


def doc_to_choice(doc):
    return [option[0] for option in doc["options"]]

def doc_to_target(doc):
    options = doc_to_choice(doc)
    return options.index(doc["correct_answer"])


def bert_score(items): return items

def agg_bert_score(items):
    print("Computing BERT score...")
    # chose this model due to its small size compared to its good performance
    # on ArabicMTEB benchmark
    # src: https://arxiv.org/pdf/2411.01192
    model_name = "FacebookAI/xlm-roberta-large"
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    score = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type=model_name,
        num_layers=12
    )
    return sum(score['f1']) / len(score['f1'])