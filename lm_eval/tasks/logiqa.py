from lm_eval.base import MultipleChoiceTask
from best_download import download_file
from pathlib import Path


class LogiQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = Path("data/logiqa")

    def download(self):
        if self.DATASET_PATH.exists():
            return
        Path.mkdir(self.DATASET_PATH)
        base_url = "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master"
        splits = [
            {"name": "Train", "checksum": "7d5bb1f58278e33b395744cd2ad8d7600faa0b3c4d615c659a44ec1181d759fa"},
            {"name": "Eval", "checksum": "4c49e6753b7262c001506b9151135abf722247035ab075dad93acdea5789c01f"},
            {"name": "Test", "checksum": "359acb78c37802208f7fde9e2f6574b8526527c63d6a336f90a53f1932cb4701"}
        ]
        for split in splits:
            file = self.DATASET_PATH / f"{split['name']}.txt"
            download_file(f"{base_url}/{split['name']}.txt", str(file), split["checksum"])

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _convert_standard(self, doc):
        def format_example(doc, choices):
            """
                Passage: <passage>
                Question: <question>
                Choices:
                A. <choice1>
                B. <choice2>
                C. <choice3>
                D. <choice4>
                Answer:
            """
            prompt = "Passage: " + doc["passage"] + "\n"
            prompt += "Question: " + doc["question"] + "\nChoices:\n"
            for choice, option in zip(choices, doc["options"]):
                prompt += f"{choice.upper()}. {option}\n"
            prompt += "Answer:"
            return prompt
        choices = ['a', 'b', 'c', 'd']
        return {
            "query": format_example(doc, choices),
            "choices": doc["options"],
            "gold": choices.index(doc["answerKey"])
        }

    def _load_docs(self, filename):
        def normalize(text):
            return text.replace(".", ". ").strip()

        with open(filename, 'r') as f:
            docs = f.read().strip().split("\n\n")
        for rawdoc in docs:
            rawdoc = rawdoc.split("\n")
            doc = {
                "answerKey": rawdoc[0].strip(),
                "passage": normalize(rawdoc[1]),
                "question": normalize(rawdoc[2]),
                "options": [normalize(option[2:]) for option in rawdoc[3:]]
            }
            yield self._convert_standard(doc)

    def training_docs(self):
        return self._load_docs(self.DATASET_PATH / "Train.txt")

    def validation_docs(self):
        return self._load_docs(self.DATASET_PATH / "Eval.txt")

    def test_docs(self):
        return self._load_docs(self.DATASET_PATH / "Test.txt")

    def fewshot_description(self):
        # TODO: figure out actual description
        return ""

    def doc_to_text(self, doc):
        return doc["query"]
