"""
QA4MRE 2011-2013: Overview of Question Answering for Machine Reading Evaluation
https://www.cs.cmu.edu/~./hovy/papers/13CLEF-QA4MRE.pdf

The (English only) QA4MRE challenge which was run as a Lab at CLEF 2011-2013.
The main objective of this exercise is to develop a methodology for evaluating 
Machine Reading systems through Question Answering and Reading Comprehension 
Tests. Systems should be able to extract knowledge from large volumes of text 
and use this knowledge to answer questions. Four different tasks have been
organized during these years: Main Task, Processing Modality and Negation for
Machine Reading, Machine Reading of Biomedical Texts about Alzheimer's disease,
and Entrance Exam.

Homepage: http://nlp.uned.es/clef-qa/repository/qa4mre.php
"""
import os
import xml.etree.ElementTree as ET
from best_download import download_file
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{Peas2013QA4MRE2O,
    title={QA4MRE 2011-2013: Overview of Question Answering for Machine Reading Evaluation},
    author={Anselmo Pe{\~n}as and Eduard H. Hovy and Pamela Forner and {\'A}lvaro Rodrigo and Richard F. E. Sutcliffe and Roser Morante},
    booktitle={CLEF},
    year={2013}
}
"""


class QA4MRE(MultipleChoiceTask):
    VERSION = 0
    YEAR = None
    def download(self):
        year = self.YEAR
        lang = "EN"
        base_path = (
            "http://nlp.uned.es/clef-qa/repository/js/scripts/downloadFile.php?"
            "file=/var/www/html/nlp/clef-qa/repository/resources/QA4MRE/"
        )
        # TODO: add side tasks?
        variable_year_path = {
            2011: '2011/Training_Data/Goldstandard/',
            2012: '2012/Main_Task/Training_Data/Goldstandard/Used_in_Evaluation/',
            2013: '2013/Main_Task/Training_Data/Goldstandard/'
        }
        sha256sums = {
            2011 : "6d2524952a3a015f2a82df785b85b5578681e3602ec276b4e72c01f4ebc50034",
            2012 : "f9edaf408f8ac93f89a643a0d0b19263a1bb5ce64f19b2af10df279a656dfb24",
            2013 : "c60e5aa4ec77e0493ef0b11d46bd1d74d58a499a3a2f871b8cf3af9536f0f094", 
        }
        vpath = variable_year_path[year]
        url_path = f"{base_path}{vpath}QA4MRE-{year}-{lang}_GS.xml"
        if not os.path.exists("data/qa4mre"):
            os.makedirs("data/qa4mre", exist_ok=True)
        if not os.path.isfile(f"data/qa4mre/QA4MRE-{year}-{lang}"):
            download_file(
                url_path,
                local_file=f"data/qa4mre/QA4MRE-{year}-{lang}_GS.xml",
                expected_checksum=sha256sums[year],
                )

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def _convert_standard(self, question):
        choices = [i.text for i in question.iter('answer')]
        out_doc = {
            "query" : question.find('q_str').text,
            "choices": choices, 
            "gold" : int(question.find("./answer[@correct='Yes']").attrib["a_id"]) - 1,
        }
        return out_doc
    
    def load_docs(self, textfilename, tfds=False):
        tree = ET.parse(textfilename)
        root = tree.getroot()
        # TODO: context is much larger than the context sometimes
        # at the moment, it just gets left-truncated by LM automatically, and maybe that's good enough?
        for reading_test in root.iter('reading-test'):
            src = reading_test[0].text
            src = src.strip().replace("\'", "'")
            for qid, question in enumerate(reading_test.iter('q')):
                out_doc = self._convert_standard(question)
                out_doc['source'] = src
                yield out_doc

    def test_docs(self):
        return self.load_docs(f"data/qa4mre/QA4MRE-{self.YEAR}-EN_GS.xml")

    def doc_to_text(self, doc):
        return "{}\nQuestion: {}\nAnswer:".format(doc["source"], doc["query"])

class QA4MRE_2011(QA4MRE):
    YEAR = 2011

class QA4MRE_2012(QA4MRE):
    YEAR = 2012

class QA4MRE_2013(QA4MRE):
    YEAR = 2013
