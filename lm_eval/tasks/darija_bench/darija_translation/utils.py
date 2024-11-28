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

        
def dr_fr(dataset: datasets.Dataset):
    return dataset.filter(lambda x: x["direction"] == "dr_fr")

def dr_en(dataset: datasets.Dataset):
    return dataset.filter(lambda x: x["direction"] == "dr_en")

def dr_msa(dataset: datasets.Dataset):
    return dataset.filter(lambda x: x["direction"] == "dr_msa")

def fr_dr(dataset: datasets.Dataset):
    return dataset.filter(lambda x: x["direction"] == "fr_dr")

def en_dr(dataset: datasets.Dataset):    
    return dataset.filter(lambda x: x["direction"] == "en_dr")

def msa_dr(dataset: datasets.Dataset):
    return dataset.filter(lambda x: x["direction"] == "msa_dr")        
        
prompt_templates = {
             "fr_dr": "ترجم من الفرنساوية للدارجة:\n{0}",
             "dr_fr": "ترجم من الدارجة للفرنساوية:\n{0}",
             "en_dr": "ترجم من الإنجليزية للدارجة:\n{0}",
             "dr_en": "ترجم من الدارجة للإنجليزية:\n{0}",
             "msa_dr": "ترجم من الفصحى للدارجة:\n{0}",
             "dr_msa": "ترجم من الدارجة للفصحى:\n{0}",
            }

def doc_to_text(doc):
    # user_input = f"[INST] <<SYS>>\nأنت مساعد مفيد ومحترم وصادق. أجب دائما بأكبر قدر ممكن من المساعدة بينما تكون آمنا.  يجب ألا تتضمن إجاباتك أي محتوى ضار أو غير أخلاقي أو عنصري أو جنسي أو سام أو خطير أو غير قانوني. يرجى التأكد من أن ردودك غير متحيزة اجتماعيا وإيجابية بطبيعتها.\n\nإذا كان السؤال لا معنى له أو لم يكن متماسكا من الناحية الواقعية، اشرح السبب بدلا من الإجابة على شيء غير صحيح. إذا كنت لا تعرف إجابة سؤال ما، فيرجى عدم مشاركة معلومات خاطئة.\n<</SYS>>\n\n{0} [/INST]"
    # doc_text = user_input.format(doc["messages"][0]["content"])
    doc_text = doc["messages"][0]["content"]
    return doc_text

def doc_to_target(doc):
    return doc["messages"][1]["content"]

def bert(items):
    return items

def Average(lst):
        return sum(lst) / len(lst)
    
def camembert(items):
    bert_model = 'almanach/camembert-base'
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    bert = bert_score.compute(predictions=predictions, references=references, model_type=bert_model, num_layers=12)
    return Average(bert['f1'])

def darijabert(items):
    bert_model = 'SI2M-Lab/DarijaBERT'
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    bert = bert_score.compute(predictions=predictions, references=references, model_type=bert_model, num_layers=12)
    return Average(bert['f1'])

def arabert(items):
    bert_model = "aubmindlab/bert-base-arabert"
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    bert = bert_score.compute(predictions=predictions, references=references, model_type=bert_model, num_layers=12)
    return Average(bert['f1'])

def bertbase(items):
    bert_model = "google-bert/bert-base-uncased"
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    bert = bert_score.compute(predictions=predictions, references=references, model_type=bert_model, num_layers=12)
    return Average(bert['f1'])

def mbert(items):
    bert_model = 'google-bert/bert-base-multilingual-cased'
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    bert = bert_score.compute(predictions=predictions, references=references, model_type=bert_model, num_layers=12)
    return Average(bert['f1'])
