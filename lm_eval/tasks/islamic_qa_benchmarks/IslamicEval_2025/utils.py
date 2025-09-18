# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# import re

# def post_process(doc, results):
#     gold = doc["label_word"]
#     label = results[0].strip()
#     label = re.sub(r"\s+", "", label)
#     return {"eval": (label, gold)}



# def evaluate(items):
#     predicted_labels, true_labels = zip(*items)
#     return {"Accuracy": accuracy_score(true_labels, predicted_labels)}

def evaluate(items):
    return {"acc": 0}



options = ['CorrectAyah', 'WrongAyah', 'CorrectHadith', 'WrongHadith']
def doc_to_choice(doc):
    return options

def doc_to_target(doc):
    return options.index(doc["Label"])