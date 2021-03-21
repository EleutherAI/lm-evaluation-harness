from lm_eval.base import Task
import os
import csv
import numpy as np
from ..utils import sh

SUBJECTS = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 
            'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 
            'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 
            'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
            'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 
            'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 
            'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 
            'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 
            'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 
            'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

CHOICES = ['A','B','C','D']

def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {hendrycks-abstract_algebra: Task, hendrycks-anatomy: Task}
    """
    return {
        f"hendrycks-{sub}": create_task(sub) for sub in SUBJECTS
    }

def create_task(subject):
    class HendrycksTest(GeneralHendrycksTest):
        def __init__(self):
            super().__init__(subject)
    return HendrycksTest

class GeneralHendrycksTest(Task):

    def __init__(self, subject):
        self.subject = subject
        super().__init__()

    def download(self):
        
        self.data_dir = "data/hendrycks/"
        if not os.path.exists(self.data_dir):
            sh("""
                mkdir -p data
                wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -P data/
                tar -xf data/data.tar -C data/
                rm data/data.tar
                mv data/data data/hendrycks
                """)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def _load_docs(self, split):

        filename = os.path.join(self.data_dir, split, self.subject + f"_{split}.csv")
        reader = csv.reader(open(filename, 'r'), quotechar='"', delimiter=',')

        docs = []
        for row in reader:
            doc = {
                "query": self._format_example(row),
                "choices": CHOICES,
                "gold": CHOICES.index(row[-1])
            }
            docs.append(doc)
        return docs

    def _format_example(self, row):
        """
            <prompt>
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer:
        """
        prompt = row[0]
        k = len(row) - 2
        for j in range(k):
            prompt += "\n{}. {}".format(CHOICES[j], row[j+1])
        prompt += "\nAnswer:"
        return prompt
        
    def training_docs(self):
        raise NotImplementedError

    def validation_docs(self):
        return self._load_docs("val")

    def test_docs(self):
        return self._load_docs("test")

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def fewshot_docs(self, k):
        assert k >= 5, "Maximum 5 few shot examples."
        return self._load_docs('dev')[:k]

    def fewshot_description(self):
        subject = self.subject.replace("_", " ")
        return f"The following are multiple choice questions (with answers) about {subject}.\n\n"

    def fewshot_context(self, doc, num_fewshot, provide_description):
        raw_description = self.fewshot_description()
        description = raw_description if provide_description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # TODO: crop if over max_len
            labeled_examples = "\n\n".join(
                [self.doc_to_text(doc) + self.doc_to_target(doc) for doc in self.fewshot_docs(k=num_fewshot)]
            ) + "\n\n"

        example = self.doc_to_text(doc)
        return description + labeled_examples + example
