"""
C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models
https://arxiv.org/pdf/2305.08322.pdf

C-Eval is a comprehensive Chinese evaluation suite for foundation models.
It consists of 13948 multi-choice questions spanning 52 diverse disciplines
and four difficulty levels.

Homepage: https://cevalbenchmark.com/
"""
from lm_eval.base import MultipleChoiceTask

_CITATION = """
@article{huang2023ceval,
    title={C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models},
    author={Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and Fu, Yao and Sun, Maosong and He, Junxian},
    journal={arXiv preprint arXiv:2305.08322},
    year={2023}
}
"""


SUBJECTS = {
    "computer_network": "计算机网络",
    "operating_system": "操作系统",
    "computer_architecture": "计算机组成",
    "college_programming": "大学编程",
    "college_physics": "大学物理",
    "college_chemistry": "大学化学",
    "advanced_mathematics": "高等数学",
    "probability_and_statistics": "概率统计",
    "discrete_mathematics": "离散数学",
    "electrical_engineer": "注册电气工程师",
    "metrology_engineer": "注册计量师",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理",
    "high_school_chemistry": "高中化学",
    "high_school_biology": "高中生物",
    "middle_school_mathematics": "初中数学",
    "middle_school_biology": "初中生物",
    "middle_school_physics": "初中物理",
    "middle_school_chemistry": "初中化学",
    "veterinary_medicine": "兽医学",
    "college_economics": "大学经济学",
    "business_administration": "工商管理",
    "marxism": "马克思主义基本原理",
    "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论体系概论",
    "education_science": "教育学",
    "teacher_qualification": "教师资格",
    "high_school_politics": "高中政治",
    "high_school_geography": "高中地理",
    "middle_school_politics": "初中政治",
    "middle_school_geography": "初中地理",
    "modern_chinese_history": "近代史纲要",
    "ideological_and_moral_cultivation": "思想道德修养与法律基础",
    "logic": "逻辑学",
    "law": "法学",
    "chinese_language_and_literature": "中国语言文学",
    "art_studies": "艺术学",
    "professional_tour_guide": "导游资格",
    "legal_professional": "法律职业资格",
    "high_school_chinese": "高中语文",
    "high_school_history": "高中历史",
    "middle_school_history": "初中历史",
    "civil_servant": "公务员",
    "sports_science": "体育学",
    "plant_protection": "植物保护",
    "basic_medicine": "基础医学",
    "clinical_medicine": "临床医学",
    "urban_and_rural_planner": "注册城乡规划师",
    "accountant": "注册会计师",
    "fire_engineer": "注册消防工程师",
    "environmental_impact_assessment_engineer": "环境影响评价工程师",
    "tax_accountant": "税务师",
    "physician": "医师资格",
}


def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {Ceval-computer_network: Task, Ceval-clinical_medicine: Task}
    """
    return {f"Ceval-valid-{sub}": create_task(sub) for sub in SUBJECTS.keys()}


def create_task(subject):
    class Ceval(CevalSubject):
        def __init__(self):
            super().__init__(subject)

    return Ceval


class CevalSubject(MultipleChoiceTask):
    VERSION = 1
    DATASET_PATH = "ceval/ceval-exam"
    DATASET_NAME = None

    def __init__(self, subject):
        self.DATASET_NAME = subject
        super().__init__()

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["val"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def _format_subject(self, subject):
        words = subject.split("_")
        return " ".join(words)

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = self.DATASET_NAME
        description = f"以下是中国关于{SUBJECTS[subject]}的单项选择题，请选出其中的正确答案。"
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            <prompt>
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            答案：
            """

            question = doc["question"].strip()
            choices = "".join([f"{key}. {doc[key]}\n" for key in keys])
            prompt = f"{question}\n{choices}答案："
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": ord(doc["answer"]) - ord("A"),
        }

    def fewshot_examples(self, k, rnd):
        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

        # use the unchanged order of the dev set without sampling,
        return self._fewshot_docs[:k]

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
