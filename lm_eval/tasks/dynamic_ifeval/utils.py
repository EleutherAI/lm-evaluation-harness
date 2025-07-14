
from lm_eval.tasks.dynamic_ifeval.helper.create_dataset import create_dataset, save_dataset
from lm_eval.tasks.dynamic_ifeval.helper.utils import TextPool
from lm_eval.tasks.dynamic_ifeval.helper.convert_yaml_to_hfdataset import convert_yaml_to_hfdataset
from lm_eval.tasks.dynamic_ifeval.helper.evaluate_answers_hf import evaluate_answer

from itertools import chain
from typing import List, Tuple, Dict, Set, Any
from datasets import Dataset, Features, Value, Sequence, load_dataset
import datasets, ast


def get_texts(
    texts_dataset: Dataset,
    texts_types_to_use: list[str] = None,
    texts_types_to_exclude: list[str] = []
):
    print(texts_dataset)
    features = list(texts_dataset.features.keys())
    print(features)
    for feature in features:
        print(type(texts_dataset[feature]))
        print(texts_dataset[feature])
        print("AND")
        print(type(texts_dataset[feature][0]))
        print(texts_dataset[feature][0])
    ls = list(chain.from_iterable(
        texts_dataset[feature][0] for feature in features
    ))
    print(ls)
    print("Length of texts:", len(ls))
    return TextPool(ls)


def custom_dataset(*args, **kwargs):
    print("Positional args:", args)
    print("Keyword args:", kwargs)
    
    texts_dataset = load_dataset("david-e-g/Texts_Samples")["train"]
    texts = get_texts(texts_dataset)
    dataset = create_dataset(texts=texts)
    save_dataset(dataset)
    convert_yaml_to_hfdataset()
    hf_dataset = Dataset.load_from_disk("data/hf_dataset")
    return {"train": hf_dataset}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    
    print("Processing dataset...")
    print(type(dataset))
    print(dataset)
    
    #texts = get_texts(dataset)
    #my_list_dataset = create_dataset(texts=texts)
    #save_dataset(my_list_dataset)
    #my_dataset = to_hf_dataset(my_list_dataset)
    
    def _process_doc(doc):
        #ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": doc["prompt"],
            "gold": "",
            **doc
        }
        return out_doc

    return dataset.map(_process_doc)

def process_results(doc, results):
    print("Processing results...")
    print(results)
    print("Now printing the doc:")
    print(doc)
    acc = [evaluate_answer(answer, doc["rules"], ast.literal_eval(doc["rules_letter_must_be_in"]),
                                                    doc["count_number"], doc["sum_characters_value"])
                                   for answer in results]
    print("Accuracy:", acc)
    acc = sum(acc) / len(acc)
    return {
        "acc": acc
    }
