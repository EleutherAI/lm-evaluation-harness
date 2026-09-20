from lm_eval.tasks.dynamic_ifeval.helper.create_dataset import create_dataset, save_dataset
from lm_eval.tasks.dynamic_ifeval.helper.utils import TextPool
from lm_eval.tasks.dynamic_ifeval.helper.convert_yaml_to_hfdataset import convert_yaml_to_hfdataset
from lm_eval.tasks.dynamic_ifeval.helper.evaluate_answers_hf import evaluate_answer, finegraned_evaluate_answer

from itertools import chain
from typing import List, Tuple, Dict, Set, Any
from datasets import Dataset, Features, Value, Sequence, load_dataset
import datasets, ast


def get_texts(
    texts_dataset: Dataset,
    texts_types_to_use: list[str] = None,
    texts_types_to_exclude: list[str] = []
):
    features = list(texts_dataset.features.keys())
    ls = list(chain.from_iterable(
        texts_dataset[feature][0] for feature in features
    ))
    return TextPool(ls)


def custom_dataset(*args, dataset_seed=42, **kwargs) -> Dict[str, Dataset]:
    
    texts_dataset = load_dataset("david-e-g/Texts_Samples")["train"]
    texts = get_texts(texts_dataset)
    dataset = create_dataset(texts=texts, seed=dataset_seed)
    hf_dataset = convert_yaml_to_hfdataset(dataset)
    return {"train": hf_dataset}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "query": doc["prompt"],
            "gold": "",
            **doc
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc, results):
    finegraned_acc = [finegraned_evaluate_answer(answer, doc["rules"], ast.literal_eval(doc["rules_letter_must_be_in"]),
                                                    doc["count_number"], doc["sum_characters_value"])
                                 for answer in results]
    strict_acc = [evaluate_answer(answer, doc["rules"], ast.literal_eval(doc["rules_letter_must_be_in"]),
                                                    doc["count_number"], doc["sum_characters_value"])
                                   for answer in results]
    finegraned_acc = sum(finegraned_acc) / len(finegraned_acc)
    strict_acc = sum(strict_acc) / len(strict_acc)
    return {
        "finegraned_acc": finegraned_acc,
        "strict_acc": strict_acc
    }
