import datasets

from typing import List
from evaluate import load



def preprocess_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    # dataset = dataset.select([i for i in range(4)])      # selecting 4 rows for DEBUG
    return dataset


# post-processing the results
def _search_delimiters(model_output: str) -> str:
    left_delimiter = '<'
    right_delimiter = '>'
    start: int = 0
    end: int = len(model_output)
    if left_delimiter in model_output:
        start = model_output.find(left_delimiter)
    if right_delimiter in model_output:
        end = model_output.find(right_delimiter)

    return model_output[start:end].replace('<', '').replace('>', '').strip()


def single_bleu(ref: str, pred: str) -> float:
    bleu_metric = load("bleu")
    bleu_score = bleu_metric.compute(predictions=[pred], references=[[ref]])
    return bleu_score["bleu"]

def sigle_chrf(ref: str, pred: str) -> float:
    chrf_metric = load("chrf")
    chrf_score = chrf_metric.compute(predictions=[pred], references=[[ref]])
    return chrf_score["score"]

def single_bleurt(ref: str, pred: str) -> float:
    bleurt = load("bleurt", module_type="metric", checkpoint="BLEURT-20")
    result = bleurt.compute(predictions=[pred], references=[ref])
    return result["scores"][0]

def single_comet(source: str, ref: str, pred: str) -> float:
    comet_metric = load("comet")
    comet_score = comet_metric.compute(predictions=[pred], references=[ref], sources=[source])
    return comet_score["scores"][0]


def process_results_gen(doc, results):
    """
    Process the results of the model and return the metrics
    Args:
        - doc: the document containing the input and output (keys are the variables from the prompt_template)
        - results: the output of the model (still dunnow why its a list but it doesn't depend on the batchsize)
    Returns:
        - a dictionary containing the metrics (metric_name: metric_value)
    """

    completion = results[0]
    model_input, reference_out = doc["italian"], doc["english"]

    # clean completion
    completion = _search_delimiters(completion)

    # BLEU
    bleu_score = single_bleu(ref=reference_out, pred=completion)
    # CHRF
    chrf_score = sigle_chrf(ref=reference_out, pred=completion)
    # BLEURT
    bleurt_score = single_bleurt(ref=reference_out, pred=completion)
    # COMET
    comet_score = single_comet(source=model_input, ref=reference_out, pred=completion)


    # print("========================================================")
    # print("model_input: ", model_input)
    # print("expected: ", reference_out)
    # print("completion: ", completion)
    # print("BLEU: ", bleu_score)
    # print("CHRF: ", chrf_score)
    # print("BLEURT: ", bleurt_score)
    # print("COMET: ", comet_score)
    # print("========================================================")

    return {
        "bleu_score": bleu_score,
        "chrf_score": chrf_score,
        "bleurt_score": bleurt_score,
        "comet_score": comet_score,
    }



