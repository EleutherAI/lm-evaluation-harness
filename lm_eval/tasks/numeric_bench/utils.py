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
    pattern = r"(?<=The answer is )[A-Za-z]"
    
    if re.search(pattern, prediction) is not None:
        prediction = re.search(pattern, prediction).group(0)
    else:
        return 0.0

    if prediction != ground_truth:
        return 0.0
    
    return 1.0

def QA_accuracy(predictions: list[str], references: list[str], **kwargs) -> float:
    try:
        prediction, ground_truth = predictions[0], references[0]
    except:
        return 0.0
    
    # 查找所有符合条件的数字（整数或小数）
    matches = re.findall(r"(?<=The answer is )\d+(?:\.\d+)?", prediction)

    if len(matches) != 1:
        return 0.0
    
    prediction = matches[0]
    if prediction != str(ground_truth):
        return 0.0

    return 1.0
