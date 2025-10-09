import evaluate
import datasets
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

@register_filter("strip")
class Strip(Filter):
    def __init__(self) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    def apply(self, resps, docs):
        """
        Assuming each entry of `resps` is a list of model responses, we discard all but the first response.
        """
        return map(lambda r: r[0].strip(), resps)

        
def eg_en(dataset: datasets.Dataset):
    return dataset.filter(lambda x: x["direction"] == "eg_en")

def eg_msa(dataset: datasets.Dataset):
    return dataset.filter(lambda x: x["direction"] == "eg_msa")

def en_eg(dataset: datasets.Dataset):    
    return dataset.filter(lambda x: x["direction"] == "en_eg")

def msa_eg(dataset: datasets.Dataset):
    return dataset.filter(lambda x: x["direction"] == "msa_eg")        
        
prompt_templates = {
             "en_eg": "ترجم من الإنجليزي للمصري:\n{0}",
             "eg_en": "ترجم من المصري للإنجليزي:\n{0}",
             "msa_eg": "ترجم من الفصحى للمصري:\n{0}",
             "eg_msa": "ترجم من المصري للفصحى:\n{0}",
            }

def doc_to_text(doc):
    doc_text = doc["messages"][0]["content"]
    return doc_text

def doc_to_target(doc):
    return doc["messages"][1]["content"]

def bert(items):
    return items

def Average(lst):
        return sum(lst) / len(lst)


def arabert(items):
    bert_model = "aubmindlab/bert-base-arabert"
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    bert = bert_score.compute(predictions=predictions, references=references, model_type=bert_model, num_layers=12)
    return Average(bert['f1'])*100

def bertbase(items):
    bert_model = "google-bert/bert-base-uncased"
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    bert = bert_score.compute(predictions=predictions, references=references, model_type=bert_model, num_layers=12)
    return Average(bert['f1'])*100

def mbert(items):
    bert_model = 'google-bert/bert-base-multilingual-cased'
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    bert = bert_score.compute(predictions=predictions, references=references, model_type=bert_model, num_layers=12)
    return Average(bert['f1'])*100
