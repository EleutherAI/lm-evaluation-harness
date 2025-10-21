import evaluate


def doc_to_choice(doc):
    return [option[0] for option in doc["options"]]

def doc_to_target(doc):
    options = doc_to_choice(doc)
    return options.index(doc["correct_answer"][0])

def gen_answer(doc):
    return doc["correct_answer"][2:]

# def custom_exact_match(items):
#     return items

# def agg_custom_exact_match(items):
#     options_map = {
#         "أ": 0, "ا": 0, "ب": 1, "ج": 2, "د": 3, "ه": 4, "و": 4,
#         "a": 0, "b": 1, "c": 2, "d": 3, "e": 4
#     }
#     targets, preds = zip(*items)
#     targets = list(map(int, targets))
#     # Convert `preds` to int (since `targets` are already integers)
#     preds = [options_map.get(pred[0].lower(), pred) for pred in preds]
#     return sum(t == p for t, p in zip(targets, preds)) / len(targets)

def bert_score(items): return items

def agg_bert_score(items):
    print("Computing BERT score...")
    # chose this model due to its small size compared to its good performance
    # on ArabicMTEB benchmark
    # src: https://arxiv.org/pdf/2411.01192
    model_name = "intfloat/multilingual-e5-small"
    bert_score = evaluate.load("bertscore")
    references, predictions = zip(*items)
    score = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type=model_name,
        num_layers=12
    )
    return sum(score['f1']) / len(score['f1'])