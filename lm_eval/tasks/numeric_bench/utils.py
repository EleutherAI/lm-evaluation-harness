import re
import datasets

def process_docs_without_cot(dataset: datasets.Dataset) -> datasets.Dataset:
    # convert json dataset to jsonl format
    josnl_format_dataset = []

    for instance in dataset:
        format_instruction = instance["system_prompt"] + instance["description"]
        prompt = format_instruction + "\nQuestion: " + instance["data"]["question"] + "\nData: " + instance["data"]["struct_data"]
        answer = instance["data"]["answer"]
        ability = instance["data"]["ability"]
        josnl_format_dataset.append({"prompt": prompt, "answer": answer, "ability": ability})

    return datasets.Dataset.from_list(josnl_format_dataset)

def process_docs_with_cot(dataset: datasets.Dataset) -> datasets.Dataset:
    # convert json dataset to jsonl format
    josnl_format_dataset = []

    for instance in dataset:
        format_instruction = instance["system_prompt_cot"] + instance["description"]
        prompt = format_instruction + "\nQuestion: " + instance["data"]["question"] + "\nData: " + instance["data"]["struct_data"]
        answer = instance["data"]["answer"]
        ability = instance["data"]["ability"]
        josnl_format_dataset.append({"prompt": prompt, "answer": answer, "ability": ability})

def choice_accuracy(predictions: list[str], references: list[str], **kwargs) -> float:
    try:
        prediction, ground_truth = predictions[0], references[0]
    except:
        return 0.0
    pattern = re.compile(r'\bis\s*(\()?\s*([A-Z])\b|\b([A-Z])\b(?=\s*$)|boxed\{([A-Z])\}')

    match = pattern.search(prediction)
    if match is None:
        return 0.0
    
    pred_answer = match.group(2) or match.group(3) or match.group(4)

    if pred_answer != ground_truth:
        return 0.0
    
    return 1.0

def number_accuracy(predictions: list[str], references: list[str], **kwargs) -> float:
    try:
        prediction, ground_truth = predictions[0], references[0]
    except:
        return 0.0
    
    # 查找所有符合条件的数字（整数或小数）
    pattern = re.compile(r"\s*([-+]?\d+(\.\d+)?)")
    match = pattern.search(prediction)

    if match is None:
        return 0.0

    pred_answer = match.group(1)
    
    if pred_answer != str(ground_truth):
        return 0.0

    return 1.0
