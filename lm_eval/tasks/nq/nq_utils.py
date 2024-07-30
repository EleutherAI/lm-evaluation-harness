
from typing import Dict, List
from jinja2 import Template
# from lm_eval.api.registry import register_metric

#Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
import regex
import string


import datasets
import pandas as pd
import ast
def create_nq_closed(token):
    PREFIX= "https://dl.fbaipublicfiles.com/dpr/data/retriever"
    def parse_csv_into_dataset(path):
        df = pd.read_csv(path, sep="\t",names=["question","answer"]).fillna("")
        df.answer = df.apply(lambda x: ast.literal_eval(x["answer"]), axis=1)
        return datasets.Dataset.from_pandas(df)
    dataset = datasets.DatasetDict({"train":parse_csv_into_dataset(f"{PREFIX}/nq-train.qa.csv"),
                                    "validation":parse_csv_into_dataset(f"{PREFIX}/nq-dev.qa.csv"),
                                    "test":parse_csv_into_dataset(f"{PREFIX}/nq-test.qa.csv"),
                                    })
    dataset.push_to_hub("nq_closedbook",token=token)
    
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def nq_exact_match_fn(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def nq_exact_match(references,predictions):
    assert isinstance(references, list) and len(references)==1
    assert isinstance(predictions, list) and len(predictions)==1
    return nq_exact_match_fn(predictions[0], references[0])


# https://github.com/facebookresearch/FiD/blob/fe769f30e3714e22476910ee39ea0054dd7921de/src/evaluation.py#L129
def doc_to_preamble(doc):
    template = "{% for ctx in ctxs | reverse %}\nTitle: {{ ctx.title }}\n{{ ctx.text }}\n{% endfor %}\n\n"
    ctxs= []
    for ctx in doc["ctxs"]:
        title,text = ctx.split("\n",1)
        ctxs.append(dict(title=title,text=text))
    return Template(template).render(ctxs=ctxs)


def doc_to_target(doc: Dict) -> List[str]:
    return doc["answer"]

def process_docs(dataset, top_n):
    def take_top(x):
        x["ctxs"] = x["ctxs"][:top_n]
        return x
    dataset = dataset.map(take_top)
    return dataset