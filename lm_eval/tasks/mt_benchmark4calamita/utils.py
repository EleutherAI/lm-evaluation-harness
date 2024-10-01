from evaluate import load
import datasets

from typing import List



def preprocess_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    # dataset['devtest'] = dataset['devtest'].select([0, 1])      # selecting two rows for DEBUG
    dataset = dataset.select([0, 1])      # selecting two rows for DEBUG
    # print(dataset)
    # exit()
    return dataset


# # OLD
# def single_bluert(ref, pred):
#     from bleurt import score
#     checkpoint = "BLEURT-20"
#     scorer = score.BleurtScorer(checkpoint)
#     score = scorer.score(candidates=pred, references=ref)
#     return score
#
# def single_comet(source, ref, pred):
#     from comet import download_model, load_from_checkpoint
#     model_path = download_model("Unbabel/wmt22-comet-da")
#     model = load_from_checkpoint(model_path)
#     data = [{"src": source, "mt": pred, "ref": ref}]
#     comet_scores = model.predict(data, gpus=1)
#     return comet_scores[1]*100

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


def single_bleurt(ref, pred):
    bleurt = load("bleurt", module_type="metric", checkpoint="BLEURT-20")
    result = bleurt.compute(predictions=[pred], references=[ref])
    return result["scores"][0]


def single_comet(source, ref, pred):
    comet_metric = load("comet")
    comet_score = comet_metric.compute(predictions=[pred], references=[ref], sources=[source])
    return comet_score["scores"][0]


def process_results_gen(doc, results):
    completion = results[0]
    model_input, reference_out = doc["italian"], doc["english"]

    # clean completion
    completion = _search_delimiters(completion)

    # BLEURT
    bleurt_score = single_bleurt(ref=reference_out, pred=completion)
    # COMET
    comet_score = single_comet(source=model_input, ref=reference_out, pred=completion)

    print("========================================================")
    print("model_input: ", model_input)
    print("expected: ", reference_out)
    print("completion: ", completion)
    print("BLEURT: ", bleurt_score)
    print("COMET: ", comet_score)
    print("========================================================")

    return {
        "bleurt_score": bleurt_score,
        "comet_score": comet_score,
    }



