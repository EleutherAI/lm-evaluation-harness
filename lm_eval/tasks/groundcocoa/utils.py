import datasets
import pandas as pd
from datasets import Dataset


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    cocoa_dataset = [sample for sample in dataset]
    processed = []
    for doc in cocoa_dataset:
        question = "A user has specified certain criteria for booking a flight. Below are five different flight options labeled 'A', 'B', 'C', 'D', and 'E'. Review these options and select the one that best matches the user requirements. Respond with a single option and the phrase 'The answer is Option ' followed by the correct letter - 'A', 'B', 'C', 'D', or 'E'\n\n"
        question = question + "User Criteria: " + doc["query"]
        question = question + "\n\n Option A: " + str(doc["Option A"]) + "\n"
        question = question + "\n Option B: " + str(doc["Option B"]) + "\n"
        question = question + "\n Option C: " + str(doc["Option C"]) + "\n"
        question = question + "\n Option D: " + str(doc["Option D"]) + "\n"
        question = question + "\n Option E: " + str(doc["Option E"]) + "\n"
        out_doc = {
            "criteria": question,
            "choices": [
                "The answer is Option A",
                "The answer is Option B",
                "The answer is Option C",
                "The answer is Option D",
                "The answer is Option E",
            ],
            "gold": "The answer is Option " + doc["Answer"],
        }
        processed.append(out_doc)
    df = pd.DataFrame(processed)
    dataset = Dataset.from_pandas(df)
    return dataset
