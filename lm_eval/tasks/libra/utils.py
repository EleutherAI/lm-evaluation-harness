import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List

import datasets


try:
    import pymorphy2

    normalizer = pymorphy2.MorphAnalyzer()
except ImportError:
    print(
        "Can not import pymorphy2. If you try to score libra, do `pip install pymorphy2`"
    )


@dataclass
class PredictionResult:
    pred_answer: str
    answers: List[str]
    length: str


def filter_dataset_by_page_lengths(*args, **kwargs) -> Dict[str, datasets.Dataset]:
    """Filter dataset by page lengths for Libra task.

    in CLI metadata --metadata '{"valid_pages": ["8p", "32p"], "dataset_repo_name": "ai-forever/LIBRA"}'
    """
    valid_pages = kwargs.get("valid_pages", [])

    dataset_repo_name = kwargs.get("dataset_repo_name", "ai-forever/LIBRA")
    dataset_name = kwargs.get("dataset_name", None)
    filter_colname = kwargs.get("filter_colname", "length")
    token = kwargs.get("token", None)

    dataset_columns = list(
        datasets.load_dataset(dataset_repo_name, dataset_name, token=token)[
            "test"
        ].features.keys()
    )
    if filter_colname not in dataset_columns:
        raise ValueError(f"Column {filter_colname} not found in dataset {dataset_name}")

    if valid_pages:
        dataset_filtered = datasets.load_dataset(
            dataset_repo_name, dataset_name, token=token
        )["test"].filter(lambda doc: doc.get(filter_colname) in valid_pages)
    else:
        dataset_filtered = datasets.load_dataset(
            dataset_repo_name, dataset_name, token=token
        )["test"]
    return {"test": dataset_filtered}


def normalize_answer(sentence: str) -> str:
    """Normalize an input sentence by removing punctuation and converting words to their base (lemmatized) form.
    :param sentence: str
        Input sentence.
    :return: str
        A normalized sentence where:
        - All characters except letters, digits, and underscores are removed.
        - All words are converted to lowercase.
        - Words are lemmatized using `normalizer`.
    :raises ValueError:
        If `sentence` is not a string.
    :example:
    >>> normalize_answer("Hello, world! This is a test sentence.")
    'hello world this is a test sentence'
    """
    sentence = str(sentence)
    new_sentence = []
    for word in sentence.split():
        token = re.sub(r"[^a-zа-яй0-9_]+", "", word.lower())
        token = normalizer.parse(token)[0].normal_form.lower()
        new_sentence.append(token)
    return " ".join(new_sentence)


def process_results(doc: List, results: List[str]) -> Dict:
    """Processes evaluation results by extracting prediction and relevant metadata.

    :param doc: A single instance from the evaluation dataset, containing reference answers and metadata.
    :param results: A list containing the predicted answer(s). The first element is used as the main prediction.
    :return: A dictionary where the key is the metric name ("libra_score") and the value is a dictionary
             with the predicted answer, reference answers, and context length.
    """
    prediction = results[0]

    data_dict = {
        "pred_answer": prediction,
        "answers": doc["positive_outputs"],
        "length": doc["length"],
    }

    return {"libra_score": data_dict}


def exact_match_score(prediction: str, ground_truth: str) -> float:
    result = 0.0
    if normalize_answer(ground_truth) in normalize_answer(prediction):
        result = 1.0
    return result


def f1_score(prediction: str, ground_truth: str) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def count_score(prediction: str, ground_truth: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def aggregate_results(
    results: List[PredictionResult], scoring_function: Callable
) -> Dict[str, float]:
    """Aggregates score by 'length' by scoring_function.

    :param results: List of dictionaries containing 'pred_answer', 'answers', and 'length'.
    :return: Dictionary with 'length' as keys and average score as values.

    :example:
    >>> results = [
    ...     {"pred_answer": "1", "answers": ["1", "one"], "length": "8p"},
    ...     {"pred_answer": "0", "answers": ["zero", "none"], "length": "8p"},
    ...     {"pred_answer": "one", "answers": ["1", "one"], "length": "16p"}
    ... ]
    >>> aggregate_results(results=results)
    {'8p': 0.5, '16p': 1.0}
    """
    scores = defaultdict(lambda: [0, 0])

    for result in results:
        length = result["length"]
        pred_answer = normalize_answer(result["pred_answer"])
        answers = set([normalize_answer(text) for text in result["answers"]])

        scores[length][1] += 1
        for answer in answers:
            metric = scoring_function(prediction=pred_answer, ground_truth=answer)
            if metric > 0:
                scores[length][0] += metric
                break
    return {key: correct / total for key, (correct, total) in scores.items()}


def aggregate_results_em(results: List[PredictionResult]) -> Dict[str, float]:
    return aggregate_results(results, exact_match_score)


def aggregate_results_f1(results: List[PredictionResult]) -> Dict[str, float]:
    return aggregate_results(results, f1_score)


def aggregate_results_count_score(results: List[PredictionResult]) -> Dict[str, float]:
    return aggregate_results(results, count_score)
