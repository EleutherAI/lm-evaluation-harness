from evaluate import load

from typing import List



# # REFERENCE FUNCTIONS
# def bleurt_score(outputs: List[str], references: List[str]):
#     # Instructions on how to install BLEURT can be found at: https://github.com/google-research/bleurt
#     # The `BLEURT-20` checkpoint required for this evaluation can be found at https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
#     from bleurt import score
#     checkpoint = "BLEURT-20"
#     scorer = score.BleurtScorer(checkpoint)
#     scores = scorer.score(candidates=outputs, references=references)
#     return (sum(scores) / len(scores))*100
#
#
# # Here source_sentences = input_data
# def comet(source_sentences: List[str], references: List[str], outputs: List[str]):
#     from comet import download_model, load_from_checkpoint
#     model_path = download_model("Unbabel/wmt22-comet-da")
#     model = load_from_checkpoint(model_path)
#     data = [{"src": src, "mt": ref, "ref": out} for src, mt, out in zip(source_sentences, references, outputs)]
#     comet_scores = model.predict(data, batch_size=8, gpus=1)
#     return comet_scores[1]*100


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

def search_delimiters(model_output: str) -> str:
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
    bleurt = load("bleurt", module_type="metric")
    result = bleurt.compute(predictions=[pred], references=[ref], checkpoint="BLEURT-20")
    return result["scores"][0]


def single_comet(source, ref, pred):
    comet_metric = load("comet")
    comet_score = comet_metric.compute(predictions=[pred], references=[ref], sources=[source])
    return comet_score["scores"][0]


def process_results_gen(doc, results):
    completion = results[0]
    model_input, model_out = doc["italian"], doc["english"]

    # clean completion
    completion = search_delimiters(completion)

    # BLEURT
    bleurt_score = single_bleurt(ref=model_out, pred=completion)
    # COMET
    comet_score = single_comet(source=model_input, ref=model_out, pred=completion)

    return {
        "bleurt_score": bleurt_score,
        "comet_score": comet_score,
    }



