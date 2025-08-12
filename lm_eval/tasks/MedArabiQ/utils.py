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
    # the same model used in the paper: https://arxiv.org/pdf/2505.03427
    # (according to the paper)
    # For BERTScore evaluation, the model used is XLM-RoBERTa-Large, as it
    # was trained on multiple languages, including Arabic,
    # making it more suitable than monolingual models. 
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