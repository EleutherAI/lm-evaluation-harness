from typing import Dict, List

def process_results(doc: List, results: List[str]) -> Dict:
    """
    Args:
        doc: a instance of the eval dataset
        results: [prediction]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    prediction = results[0]

    data_dict = {"pred_answer": prediction, "answer": doc["output"]}

    return {f"shell_game_em": data_dict}


def aggregate_results(results: Dict) -> int:
    score = 0
    for res in results:
        if res["answer"] in res["pred_answer"]:
            score += 1
    return 100 * score / len(results)

def doc_to_text(doc):
    return "<video>" + doc["input"]

def doc_to_video(doc):
    return [doc["video"]]
