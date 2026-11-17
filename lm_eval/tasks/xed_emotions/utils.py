import code
from datasets import load_dataset, Dataset
from sacremoses import MosesDetokenizer
from collections import defaultdict
import json
import pathlib
import random

XED_EMOTIONS_FILES = {
    "en": "https://raw.githubusercontent.com/Helsinki-NLP/XED/refs/heads/master/AnnotatedData/en-annotated.tsv",
    "fi": "https://raw.githubusercontent.com/Helsinki-NLP/XED/refs/heads/master/AnnotatedData/fi-annotated.tsv",
    "tr": "https://raw.githubusercontent.com/Helsinki-NLP/XED/refs/heads/master/Projections/tr-projections2.tsv"
}

PROJECTION_BASEURL = "https://raw.githubusercontent.com/Helsinki-NLP/XED/refs/heads/master/Projections/{}-projections.tsv"

def load_xed_emotions(*args, **kwargs):
    """Load xed_emotions and shuffle them in a deterministic way"""

    print(kwargs)
    
    # Use hf to download the dataset
    # It is seemingly moses-tokenized text followed by tab-separated labels
    # Use the finnish or english annotated data; fall back to projected data for others

    ds = load_dataset("text", data_files={"test": XED_EMOTIONS_FILES.get(kwargs["lang"], PROJECTION_BASEURL.format(kwargs["lang"]))}, split="test")
    
    data = []

    detok = MosesDetokenizer(lang='en')

    label_names = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

    for row in ds:
        if row["text"] == "phrase\temotion": # lang tr has a header :D
            continue
        if ".xml.gz" in row["text"]: # what even are these lines?!
            print(row["text"])
            continue
        if "\t" not in row["text"]: # other mysterious lines??
            print(row)
            continue
        chunks = row["text"].split("\t")
        labels = chunks[1].split(", ")
        tokens = chunks[0].split(" ")
        data.append({"text": detok.detokenize(tokens), "labels": labels, "label": label_names[int(random.choice(labels))-1]})

    random.seed(3876723205042678486)
    random.shuffle(data)
        
    ds = Dataset.from_list(data)
    print(ds)
    return {"test": ds}