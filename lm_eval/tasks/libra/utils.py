import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import datasets


try:
    import pymorphy3

    normalizer = pymorphy3.MorphAnalyzer()
except ImportError:
    print(
        "Can not import pymorphy3. If you try to score libra, do `pip install pymorphy3`"
    )


@dataclass
class PredictionResult:
    pred_answer: str
    answers: list[str]
    length: str


def filter_dataset_by_lengths(*args, **kwargs) -> dict[str, datasets.Dataset]:
    """Filter dataset by lengths for Libra task."""
    valid_pages = kwargs.get("valid_pages", [])

    # Используем dataset_repo_name из kwargs (передается через dataset_kwargs из конфига)
    # Если не передан, используем дефолт LIBRA-V3
    dataset_repo_name = kwargs.get("dataset_repo_name", "ai-forever/LIBRA-V3")
    dataset_name = kwargs.get("dataset_name")
    filter_colname = kwargs.get("filter_colname", "length")
    token = kwargs.get("token")

    # Если токен не передан, пытаемся получить из переменной окружения
    if token is None:
        import os

        token = os.environ.get("HF_TOKEN")

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
    """Нормализация: пробелы, ё→е, буквы/цифры, леммы pymorphy3."""
    sentence = unicodedata.normalize("NFKC", str(sentence))
    sentence = sentence.replace("\u00a0", " ").replace("\u2009", " ")
    sentence = sentence.replace("ё", "е").replace("Ё", "е")
    sentence = re.sub(r"\s+", " ", sentence.strip()).lower()

    new_sentence = []
    for word in sentence.split():
        token = re.sub(r"[^a-zа-яй0-9_]+", "", word)
        if not token:
            continue
        if token.isdigit():
            new_sentence.append(token)
            continue
        new_sentence.append(normalizer.parse(token)[0].normal_form.lower())
    return " ".join(new_sentence)


def process_results(doc: list, results: list[str]) -> dict:
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


def count_score(prediction: str, ground_truth: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def aggregate_results(
    results: list[PredictionResult], scoring_function: Callable
) -> dict[str, float]:
    """Aggregates score by 'length' by scoring_function.

    :param results: List of dictionaries containing 'pred_answer', 'answers', and 'length'.
    :return: Dictionary with 'length' as keys and average score as values.

    :example:
    >>> results = [
    ...     {"pred_answer": "1", "answers": ["1", "one"], "length": "8k"},
    ...     {"pred_answer": "0", "answers": ["zero", "none"], "length": "8k"},
    ...     {"pred_answer": "one", "answers": ["1", "one"], "length": "16k"}
    ... ]
    >>> aggregate_results(results=results)
    {'8k': 0.5, '16k': 1.0}
    """
    scores = defaultdict(lambda: [0, 0])

    for result in results:
        length = result["length"]
        pred_answer = normalize_answer(result["pred_answer"])
        answers = {normalize_answer(text) for text in result["answers"]}

        scores[length][1] += 1
        for answer in answers:
            metric = scoring_function(prediction=pred_answer, ground_truth=answer)
            if metric > 0:
                scores[length][0] += metric
                break
    return {key: correct / total for key, (correct, total) in scores.items()}


def aggregate_results_em(results: list[PredictionResult]) -> dict[str, float]:
    return aggregate_results(results, exact_match_score)


# def aggregate_results_f1(results: List[PredictionResult]) -> Dict[str, float]:
#    return aggregate_results(results, f1_score)


def aggregate_results_count_score(results: list[PredictionResult]) -> dict[str, float]:
    return aggregate_results(results, count_score)


def count_score_fixed(prediction: str, ground_truth: str) -> float:
    """Improved count_score that extracts numbers from both prediction and ground_truth."""
    # Extract numbers from prediction
    pred_numbers = re.findall(r"\d+", prediction)

    # Extract numbers from ground_truth (in case it contains text)
    gt_numbers = re.findall(r"\d+", ground_truth)

    if len(pred_numbers) == 0:
        return 0.0

    # Check if any number from prediction matches any number from ground_truth
    for pred_num in pred_numbers:
        for gt_num in gt_numbers:
            if pred_num == gt_num:
                return 1.0

    return 0.0


def aggregate_results_count_score_fixed(
    results: list[PredictionResult],
) -> dict[str, float]:
    return aggregate_results(results, count_score_fixed)
