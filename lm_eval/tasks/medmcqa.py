"""
MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering

https://arxiv.org/pdf/2203.14371.pdf

This paper introduces MedMCQA, a new large-scale, Multiple-Choice Question Answering (MCQA) dataset
designed to address real-world medical entrance exam questions. More than 194k high-quality AIIMS
& NEET PG entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects are collected
with an average token length of 12.77 and high topical diversity. Each sample contains a question,
correct answer(s), and other options which requires a deeper language understanding as it tests the
10+ reasoning abilities of a model across a wide range of medical subjects & topics. A detailed explanation
of the solution, along with the above information, is provided in this study.

Homepage: https://medmcqa.github.io
"""
from lm_eval.base import MultipleChoiceTask

_CITATION = """
@InProceedings{pmlr-v174-pal22a,
  title = 	 {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author =       {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle = 	 {Proceedings of the Conference on Health, Inference, and Learning},
  pages = 	 {248--260},
  year = 	 {2022},
  editor = 	 {Flores, Gerardo and Chen, George H and Pollard, Tom and Ho, Joyce C and Naumann, Tristan},
  volume = 	 {174},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {07--08 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v174/pal22a/pal22a.pdf},
  url = 	 {https://proceedings.mlr.press/v174/pal22a.html},
  abstract = 	 {This paper introduces MedMCQA, a new large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions. More than 194k high-quality AIIMS &amp; NEET PG entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects are collected with an average token length of 12.77 and high topical diversity. Each sample contains a question, correct answer(s), and other options which requires a deeper language understanding as it tests the 10+ reasoning abilities of a model across a wide range of medical subjects &amp; topics. A detailed explanation of the solution, along with the above information, is provided in this study.}
}
"""


class MedMCQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "medmcqa"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])



    def _process_doc(self, doc):
        choices = [doc['opa'], doc['opb'], doc['opc'], doc['opd']]
        out_doc = {
            "query": doc["question"],
            "choices": choices,
            "gold": doc['cop'],
            "subject": doc['subject_name']
        }
        return out_doc

    def doc_to_text(self, doc):
        # OA does this a bit weirdly: they prepend "anli 1:  anli 1:  " to the beginning
        # of the prompt (yes, repeating it!). also, " True, False, or Neither?" is directly
        # appended onto the question, with no "Answer:" or even a newline. Do we *really*
        # want to do it exactly as OA did?
        return (
                "You are a highly intelligent doctor who answers the following multiple choice question correctly.\nOnly write the answer down."
                "\n\n**Subject:**" + doc["subject"]
                + "\n\n**Question:**" + doc["query"] + "\n\n"
                + ",".join(doc['choices'])
                + "\n\n**Answer:**"
        )

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]] + "\n\n"
