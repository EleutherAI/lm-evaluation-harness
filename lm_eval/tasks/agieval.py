"""
AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models
https://arxiv.org/pdf/2304.06364.pdf

AGIEval is a human-centric benchmark specifically designed to evaluate 
the general abilities of foundation models in tasks pertinent to human cognition 
and problem-solving. This benchmark is derived from 20 official, public, 
and high-standard admission and qualification exams intended for general human test-takers,
such as general college admission tests
(e.g., Chinese College Entrance Exam (Gaokao) and American SAT), 
law school admission tests, math competitions, lawyer qualification tests, 
and national civil service exams. For a full description of the benchmark, 
please refer to our paper: AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models.

Homepage: https://github.com/microsoft/AGIEval
"""
import os
import re

import json
import datasets
import numpy as np

from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from .agieval_prompt import FEW_SHOT_PROMPT_TEMPLATE

_CITATION = """
@misc{zhong2023agieval,
      title={AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models}, 
      author={Wanjun Zhong and Ruixiang Cui and Yiduo Guo and Yaobo Liang and Shuai Lu and Yanlin Wang and Amin Saied and Weizhu Chen and Nan Duan},
      year={2023},
      eprint={2304.06364},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

SUBJECTS=[
    {
        "type": "single_choice",
        "keyword" : "aqua-rat", 
    },
    {
        "type": "single_choice",
        "keyword" : "gaokao-geography", 
    },
    {
        "type": "single_choice",
        "keyword" : "lsat-lr", 
    },
    {
        "type": "single_choice",
        "keyword" : "sat-math",
    },
    {
        "type": "single_choice",
        "keyword" : "gaokao-biology", 
    },
    {
        "type": "single_choice",
        "keyword" : "gaokao-history", 
    },
    {
        "type": "single_choice",
        "keyword" : "lsat-rc",
    },
    {
        "type": "single_choice",
        "keyword" : "gaokao-chemistry", 
    },
    {
        "type": "single_choice",
        "keyword" : "logiqa-en",
    },
    {
        "type": "single_choice",
        "keyword" : "gaokao-chinese", 
    },
    {
        "type": "single_choice",
        "keyword" : "logiqa-zh", 
    },
    {
        "type": "single_choice",
        "keyword" : "sat-en-without-passage", 
    },
    {
        "type": "single_choice",
        "keyword" : "gaokao-english",
    },
    {
        "type": "single_choice",
        "keyword" : "lsat-ar", 
    },
    {
        "type": "single_choice",
        "keyword" : "sat-en", 
    },
    {
        "type": "multi_question_choice",
        "keyword" : "gaokao-physics", 
    },
    {
        "type": "multi_question_choice",
        "keyword" : "jec-qa-ca", 
    },
    {
        "type": "multi_question_choice",
        "keyword" : "jec-qa-kd", 
    },
    # {
    #     "keyword" : "gaokao-mathqa",
    # },
    # "gaokao-mathcloze", # TODO: 包括填空题
    # "math", # TODO: 包括填空题
]

english_qa_datasets = ["lsat-ar", "lsat-lr", "lsat-rc", "logiqa-en", "sat-math", "sat-en", "aqua-rat",
                       "sat-en-without-passage", "gaokao-english"]
chinese_qa_datasets = ["logiqa-zh", "jec-qa-kd", "jec-qa-ca", "gaokao-chinese", "gaokao-geography", "gaokao-history",
                       "gaokao-biology", "gaokao-chemistry", "gaokao-physics", "gaokao-mathqa"]
english_cloze_datasets = ["math"]
chinese_cloze_datasets = ["gaokao-mathcloze"]

multi_choice_datasets = ["jec-qa-kd", "jec-qa-ca", "gaokao-physics"]
math_output_datasets = {"gaokao-mathcloze", "math"}

def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {agi_eval-agronomy: Task, agi_eval-anatomy: Task}
    """
    return {f"agi_eval-{sub['keyword']}": create_task(sub["keyword"], sub["type"]) for sub in SUBJECTS}


def create_task(subject, qtype):
    class AGIEval(AGIEvalSubject):
        def __init__(self):
            super().__init__(subject, qtype)

    return AGIEval


class AGIEvalSubject(Task):
    VERSION = 0
    DATASET_PATH = "agi_eval/data/v1"
    DATASET_NAME = None

    def __init__(self, subject, qtype):
        self.DATASET_NAME = subject
        self.question_type = qtype
        super().__init__()

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        data_path = self.DATASET_PATH
        
        if self.DATA_LOCAL_PATH is not None:
            data_path = os.path.join(self.DATA_LOCAL_PATH, self.DATASET_PATH, self.DATASET_NAME+'.jsonl')
        print("dataset path:", data_path)

        if self.DATASET_NAME == "gaokao-physics":
            with open(data_path, 'r') as f:
                data = f.readlines() 
            
            data_path = data_path.replace('gaokao-physics.jsonl', 'new-gaokao-physics.jsonl')
            with open(data_path, 'w') as f:
                for item in data : 
                    temp = json.loads(item)
                    if not isinstance(temp['label'], list):
                        temp['label'] = [temp['label']]
                        f.write(json.dumps(temp).replace('None', 'null')+"\n")

        self.dataset = datasets.load_dataset(
            'json',
            data_files=data_path
        )

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = map(self._process_doc, self.dataset["train"]) 
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["train"]) 

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["train"]) 

    def _process_single_doc_gold(self, ans):
        if ans is None:
            if self.question_type == "single_choice":
                return '-1'
            elif self.question_type == "multi_question_choice":
                return ['-1']

        if len(ans) == 1:
            if isinstance(ans, list):
                ans = ans[0]
            if ans in "1234567890":
                gold = chr(ord(ans) - ord('0') +ord('A'))
            else:
                gold = ans.upper()
        else:
            gold = [ item.upper() for item in ans if item != ' ' ]
        
        if self.question_type == "single_choice":
            return [gold]
        elif self.question_type == "multi_question_choice":
            if isinstance(gold, list):
                return gold
            else:
                return [gold]

    def _process_doc(self, doc):
        '''
        {
            "passage": null, 
            "question": "A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45\u00b0 to 60\u00b0. After how much more time will this car reach the base of the tower?", 
            "options": [
                "(A)5(\u221a3 + 1)", 
                "(B)6(\u221a3 + \u221a2)", 
                "(C)7(\u221a3 \u2013 1)", 
                "(D)8(\u221a3 \u2013 2)", 
                "(E)None of these"
            ], 
            "label": "A", 
            "answer": null, 
            "other": {
                "solution": "Explanation :\nLet the height of the building be h. Initially, he was at an angle of 450. tan 45 = h/distance between car and tower. h = distance between car and tower (since tan 45 = 1).\nNow, after 10 minutes, it travelled a certain distance, and angle changed to 600.\ntan 60 = h/x x = h/\u221a3\nSo, in 10 minutes, it has travelled a distance of h \u2013 x = h - h/\u221a3.\n10 minutes = h *( 1 \u2013 1\u221a3)\nh can be travelled in 10 / (1 \u2013 1\u221a3).\nTo travel a distance of x, which is h/\u221a3, it takes :\nh = 10 / (1 \u2013 1/\u221a3)\nh / \u221a3 = 10/ \u221a3 * (1 \u2013 1/\u221a3). Multiply numerator and denominator by 1 + \u221a3 ( conjugate of 1 - \u221a3). We get, x = h/\u221a3 = 10 (1 + \u221a3) / 2 = 5* (1 + \u221a3)\nSo, it takes 5(1 + \u221a3) minutes to reach the base of the tower.\nAnswer : A"
            }
        }
        '''
        gold = self._process_single_doc_gold(doc["label"]) \
            if doc["label"] else self._process_single_doc_gold(doc["answer"])
        out_doc = {
                "passage": doc.get("passage"),
                "query": doc["question"],
                "choices": doc["options"],
                "gold": gold,
            }

        return out_doc

    def get_zero_sort_example(self, doc):
        try:
            passage = doc.get("passage") if doc.get("passage") is not None else ""
            if self.DATASET_NAME in english_qa_datasets:
                return passage + "Q: " + doc.get("query") + " " \
                    + "Answer Choices: " + " ".join(doc.get("choices")) + "\n" + \
                    "Let's think step by step."
            
            elif self.DATASET_NAME in chinese_qa_datasets:
                option_string = "ABCDEFG"
                count = len(doc.get("choices"))
                if count == 1:
                    count = 4
                return passage + "问题：" + doc.get("query") + " " \
                    + "选项：" + " ".join(doc.get("choices")) + "\n" + \
                    "从A到{}, 我们应选择什么？让我们逐步思考：".format(option_string[count - 1])
            
            elif self.DATASET_NAME in english_cloze_datasets:
                return passage + "Q: " + doc.get("query") + "\n" \
                        "A: Let's think step by step."
            
            elif self.DATASET_NAME in chinese_cloze_datasets:
                return passage + "问题：" + doc.get("query") + "\n" \
                        "答案：让我们逐步思考："
        except NameError:
            print("Dataset not defined.") 

    def get_few_sort_example(self, doc):
        try:
            passage = doc.get("passage") if doc.get("passage") is not None else ""
            if self.DATASET_NAME in english_qa_datasets:
                return "\n\nProblem:    " + passage + " " + doc.get("query") + "\n" \
                    + "Choose from the following options:    " + " ".join(doc.get("choices")) + "\n"

            if self.DATASET_NAME in chinese_qa_datasets:
                return "\n\n问题:   "  + passage + " "  + doc.get("query") + "\n" \
                    + "从以下选项中选择:    " + " ".join(doc.get("choices")) + "\n"

            if self.DATASET_NAME in english_cloze_datasets:
                return "\n\nProblem :   " + passage + " "  + doc.get("query") + "\n"

            if self.DATASET_NAME in chinese_cloze_datasets:
                return "\n\n问题 :   "  + passage + " " + doc.get("query") + "\n"
        except NameError:
            print("Dataset not defined.") 

    def doc_to_text(self, doc, k):
        if k == 0:
            return self.get_zero_sort_example(doc)
        else:
            return self.get_few_sort_example(doc)

    def doc_to_target(self, doc, k):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        return doc

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            ) 
        
        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            labeled_examples = FEW_SHOT_PROMPT_TEMPLATE.get(self.DATASET_NAME) 

            if labeled_examples is None:
                labeled_examples = ""

        example = self.doc_to_text(doc, num_fewshot)
        return description + labeled_examples + example
 
    def construct_requests(self, doc, ctx):
        completion = rf.greedy_until(ctx, {"until": []})
        return completion

    def _extract_last_line(self, string):
        lines = string.split('\n')
        for item in lines[::-1]:
            if item.strip() != "":
                string = item
                break
        return string

    def _extract_choice_answer(self, model_output):
        model_answer = []
        end_line = self._extract_last_line(model_output)
        pattern = ""

        if self.DATASET_NAME in english_qa_datasets:
            pattern = "answer is .*?([A-G])"
        elif self.DATASET_NAME in chinese_qa_datasets:
            pattern = "\(*[A-G]\)*"
        elif self.DATASET_NAME in multi_choice_datasets:
            pattern = "\(*([A-F])\)*"
            
        model_answer = [item.replace('(', '').replace(')', '')
            for item in re.findall(pattern, end_line)] 
        print("model_output\n", model_output, "\nmodel_answer\n", model_answer, '--- \n')
        return model_answer

    def _convert_to_set(self, item):
        if isinstance(item, list):
            return set(item)
        if isinstance(item, str):
            return {item}
        if item is None:
            return {}
        raise ValueError("Input can't parse:", item)

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        completion = results[0].strip()
        standard_answer = doc["gold"]
        model_answer = self._extract_choice_answer(completion)
        acc = 1 if self._convert_to_set(standard_answer) == self._convert_to_set(model_answer) else 0 
        return { "acc": acc }

    def aggregation(self):
        return { "acc": mean }

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return { "acc":True }
