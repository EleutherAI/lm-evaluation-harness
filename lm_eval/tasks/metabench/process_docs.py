import ast
import datasets
import re

def process_arc(dataset: datasets.Dataset) -> datasets.Dataset:
    def _convert(x):
        value_to_char = {'1': 'A', '2': 'B', '3': 'C', 
                         '4': 'D', '5': 'E', '6': 'F'}
        return value_to_char[x] if x in value_to_char else x
    
    def _subprocess(doc):
        doc["question"] =  doc["Long prompt"] # for consistency with ARC eval-lm config
        doc["answerKey"] = _convert(doc["Answer"]) # for consistency with ARC eval-lm config and fix minor inconsistencies in answer labelling (some are numbers)
        doc["choices"] = {
            'label': ['A', 'B', 'C', 'D'],
            'text': ast.literal_eval(doc["Choices"].replace('\n', '').replace("' '", "', '").replace("\" \"", "\", \"").replace("\" '", "\", '").replace("' \"", "', \""))
        }
        if len(doc["choices"]['text']) != 4:
            choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            length = len(doc["choices"]['text']) # for the example with 5 choices, but extensible to other cases
            doc["choices"]['label'] = choices[:length]
            
        doc.pop("Short prompt", None)
        doc.pop("Answer", None)
        doc.pop("Long prompt", None)
        doc.pop("Choices", None)
        doc.pop("Original index", None)
        return doc
    return dataset.map(_subprocess)

def process_gsm8k(dataset: datasets.Dataset) -> datasets.Dataset:
    def _subprocess(doc):
        doc["question"] =  doc["Long prompt"] # for consistency with the gsm8k evaluation. Fixed 5-shot task
        doc["answer"] = doc["Answer"] # for consistency with the gsm8k evaluation. Fixed 5-shot task

        doc.pop("Short prompt", None)
        doc.pop("Answer", None)
        doc.pop("Long prompt", None)
        doc.pop("Original index", None)

        return doc
    return dataset.map(_subprocess)

def process_hellaswag(dataset: datasets.Dataset) -> datasets.Dataset:
    def _preprocess(text): #taken from hellaswag task
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text
    
    def _subprocess(doc):
        doc["query"] = _preprocess(doc["Long prompt"]) #This should already be preprocessed
        choices = ast.literal_eval(doc["Choices"])
        doc["choices"] = [_preprocess(ending) for ending in choices]
        doc["label"] = int(doc["Answer"])

        doc.pop("Short prompt", None)
        doc.pop("Answer", None)
        doc.pop("Long prompt", None)
        doc.pop("Choices", None)
        doc.pop("Original index", None)

        return doc
    return dataset.map(_subprocess)

def process_mmlu(dataset: datasets.Dataset) -> datasets.Dataset:
    def _subprocess(doc):
        doc["question"] =  doc["Short prompt"] 
        doc["answer"] = doc["Answer"] 

        doc.pop("Short prompt", None)
        doc.pop("Answer", None)
        doc.pop("Long prompt", None)
        doc.pop("Choices", None)
        doc.pop("Original index", None)
        return doc
    return dataset.map(_subprocess)

def process_truthfulqa(dataset: datasets.Dataset) -> datasets.Dataset: #For consistency with truthfulqa
    def _subprocess(doc):
        doc["question"] =  doc["Short prompt"] 
        doc["mc1_targets"] = {
            "choices" : ast.literal_eval(doc["Choices"]),
            "labels" : [1, 0, 0, 0]
        }

        doc.pop("Short prompt", None)
        doc.pop("Answer", None)
        doc.pop("Long prompt", None)
        doc.pop("Choices", None)
        doc.pop("Original index", None)
        return doc
    return dataset.map(_subprocess)

def process_winogrande(dataset: datasets.Dataset) -> datasets.Dataset: #For consistency with truthfulqa
    def _subprocess(doc):
        doc["sentence"] =  doc["Long prompt"] 
        doc["choices"] = {
            'label': [1, 2],
            'text': ast.literal_eval(doc["Choices"].replace("' '", "', '"))
        }

        doc.pop("Short prompt", None)
        doc.pop("Long prompt", None)
        doc.pop("Choices", None)
        doc.pop("Original index", None)
        return doc
    return dataset.map(_subprocess)