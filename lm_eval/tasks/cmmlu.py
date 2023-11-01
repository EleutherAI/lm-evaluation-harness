"""
CMMLU: Measuring massive multitask language understanding in Chinese
https://arxiv.org/abs/2306.09212

CMMLU is a comprehensive Chinese assessment suite specifically designed to evaluate the advanced knowledge 
and reasoning abilities of LLMs within the Chinese language and cultural context. CMMLU covers a wide range of 
subjects, comprising 67 topics that span from elementary to advanced professional levels. It includes subjects that 
require computational expertise, such as physics and mathematics, as well as disciplines within humanities and 
social sciences. Many of these tasks are not easily translatable from other languages due to their specific 
contextual nuances and wording. Furthermore, numerous tasks within CMMLU have answers that are specific to 
China and may not be universally applicable or considered correct in other regions or languages.

Homepage: https://github.com/haonan-li/CMMLU
Huggingface homepage: https://huggingface.co/datasets/haonan-li/cmmlu
"""
import os
from lm_eval.base import MultipleChoiceTask, rf

_CITATION = """
@misc{li2023cmmlu,
      title={CMMLU: Measuring massive multitask language understanding in Chinese}, 
      author={Haonan Li and Yixuan Zhang and Fajri Koto and Yifei Yang and Hai Zhao and Yeyun Gong and Nan Duan and Timothy Baldwin},
      year={2023},
      eprint={2306.09212},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


SUBJECTS = [
    "agronomy",
    "anatomy",
    "ancient_chinese",
    "arts",
    "astronomy",
    "business_ethics",
    "chinese_civil_service_exam",
    "chinese_driving_rule",
    "chinese_food_culture",
    "chinese_foreign_policy",
    "chinese_history",
    "chinese_literature",
    "chinese_teacher_qualification",
    "clinical_knowledge",
    "college_actuarial_science",
    "college_education",
    "college_engineering_hydrology",
    "college_law",
    "college_mathematics",
    "college_medical_statistics",
    "college_medicine",
    "computer_science",
    "computer_security",
    "conceptual_physics",
    "construction_project_management",
    "economics",
    "education",
    "electrical_engineering",
    "elementary_chinese",
    "elementary_commonsense",
    "elementary_information_and_technology",
    "elementary_mathematics",
    "ethnology",
    "food_science",
    "genetics",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_geography",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "human_sexuality",
    "international_law",
    "journalism",
    "jurisprudence",
    "legal_and_moral_basis",
    "logical",
    "machine_learning",
    "management",
    "marketing",
    "marxist_theory",
    "modern_chinese",
    "nutrition",
    "philosophy",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_study",
    "sociology",
    "sports_science",
    "traditional_chinese_medicine",
    "virology",
    "world_history",
    "world_religions"
]

SUBJECT_MAPPING = {
    "agronomy": "农学",
    "anatomy": "解剖学",
    "ancient_chinese": "古汉语",
    "arts": "艺术学",
    "astronomy": "天文学",
    "business_ethics": "商业伦理",
    "chinese_civil_service_exam": "中国公务员考试",
    "chinese_driving_rule": "中国驾驶规则",
    "chinese_food_culture": "中国饮食文化",
    "chinese_foreign_policy": "中国外交政策",
    "chinese_history": "中国历史",
    "chinese_literature": "中国文学",
    "chinese_teacher_qualification": "中国教师资格",
    "clinical_knowledge": "临床知识",
    "college_actuarial_science": "大学精算学",
    "college_education": "大学教育学",
    "college_engineering_hydrology": "大学工程水文学",
    "college_law": "大学法律",
    "college_mathematics": "大学数学",
    "college_medical_statistics": "大学医学统计",
    "college_medicine": "大学医学",
    "computer_science": "计算机科学",
    "computer_security": "计算机安全",
    "conceptual_physics": "概念物理学",
    "construction_project_management": "建设工程管理",
    "economics": "经济学",
    "education": "教育学",
    "electrical_engineering": "电气工程",
    "elementary_chinese": "小学语文",
    "elementary_commonsense": "小学常识",
    "elementary_information_and_technology": "小学信息技术",
    "elementary_mathematics": "初等数学",
    "ethnology": "民族学",
    "food_science": "食品科学",
    "genetics": "遗传学",
    "global_facts": "全球事实",
    "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学",
    "high_school_geography": "高中地理",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理学",
    "high_school_politics": "高中政治",
    "human_sexuality": "人类性行为",
    "international_law": "国际法学",
    "journalism": "新闻学",
    "jurisprudence": "法理学",
    "legal_and_moral_basis": "法律与道德基础",
    "logical": "逻辑学",
    "machine_learning": "机器学习",
    "management": "管理学",
    "marketing": "市场营销",
    "marxist_theory": "马克思主义理论",
    "modern_chinese": "现代汉语",
    "nutrition": "营养学",
    "philosophy": "哲学",
    "professional_accounting": "专业会计",
    "professional_law": "专业法学",
    "professional_medicine": "专业医学",
    "professional_psychology": "专业心理学",
    "public_relations": "公共关系",
    "security_study": "安全研究",
    "sociology": "社会学",
    "sports_science": "体育学",
    "traditional_chinese_medicine": "中医中药",
    "virology": "病毒学",
    "world_history": "世界历史",
    "world_religions": "世界宗教",
}

SUBJECT_CATEGORIES = {
    "agronomy": ['other'],
    "anatomy": ['biology'],
    "ancient_chinese": ['linguistics','china specific'],
    "arts": ['arts'],
    "astronomy": ['physics'],
    "business_ethics": ['business'],
    "chinese_civil_service_exam": ['politics','china specific'],
    "chinese_driving_rule": ['other','china specific'],
    "chinese_food_culture": ['culture','china specific'],
    "chinese_foreign_policy": ['politics','china specific'],
    "chinese_history":['history','china specific'],
    "chinese_literature": ['literature','china specific'],
    "chinese_teacher_qualification": ['education','china specific'],
    "college_actuarial_science":['math'],
    "college_education":['education'],
    "college_engineering_hydrology": ['engineering'],
    "college_law": ['law'],
    "college_mathematics": ['math'],
    "college_medical_statistics":['statistics'],
    "clinical_knowledge": ['other'],
    "college_medicine": ['other'],
    "computer_science": ['computer science'],
    "computer_security": ['other'],
    "conceptual_physics": ['physics'],
    "construction_project_management": ['other','china specific'],
    "economics": ['economics'],
    "education": ['education'],
    "elementary_chinese":['linguistics','china specific'],
    "elementary_commonsense":['other','china specific'],
    "elementary_information_and_technology": ['other'],
    "electrical_engineering": ['engineering'],
    "elementary_mathematics": ['math'],
    "ethnology": ['culture','china specific'],
    "food_science": ['other'],
    "genetics": ['biology'],
    "global_facts": ['global'],
    "high_school_biology": ['biology'],
    "high_school_chemistry": ['chemistry'],
    "high_school_geography": ['geography'],
    "high_school_mathematics": ['math'],
    "high_school_physics": ['physics'],
    "high_school_politics": ['politics','china specific'],
    "human_sexuality": ['other'],
    "international_law": ['law'],
    "journalism": ['sociology'],
    "jurisprudence": ['law'],
    "legal_and_moral_basis": ['other'],
    "logical": ['philosophy'],
    "machine_learning": ['computer science'],
    "management": ['business'],
    "marketing": ['business'],
    "marxist_theory": ['philosophy'],
    "modern_chinese": ['linguistics','china specific'],
    "nutrition": ['other'],
    "philosophy": ['philosophy'],
    "professional_accounting": ['business'],
    "professional_law": ['law'],
    "professional_medicine": ['other'],
    "professional_psychology": ['psychology'],
    "public_relations": ['politics'],
    "security_study": ['politics'],
    "sociology": ['culture'],
    "sports_science": ['other'],
    "traditional_chinese_medicine": ['other','china specific'],
    "virology": ['biology'],
    "world_history":['history'],
    "world_religions": ['global'],
}

CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering", "statistics"],
    "Humanities": ["history", "philosophy", "law", "arts", "literature", "global"],
    "Social Science": ['linguistics',"business", "politics", "culture", "economics", "geography", "psychology", "education", "sociology"],
    "Other":["other"],
    "China specific": ["china specific"],
}

def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {cmmlu-physician: Task, cmmlu-tax_accountant: Task}
    """
    return {f"cmmlu-{sub}": create_task(sub) for sub in SUBJECTS}


def create_task(subject):
    class CmmluTest(GeneralCmmluTest):
        def __init__(self):
            super().__init__(subject)

    return CmmluTest


class GeneralCmmluTest(MultipleChoiceTask):
    VERSION = 1
    DATASET_PATH =  os.path.join("haonan-li/cmmlu")
    DATASET_NAME = None

    def __init__(self, subject):
        self.DATASET_NAME = subject
        super().__init__()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = self.DATASET_NAME
        description = f"以下是关于{SUBJECT_MAPPING[subject]}的单项选择题，请直接给出正确答案的选项。"
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            题目：<prompt>
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            答案是：
            """

            question = doc["Question"].strip()
            choices = "".join(
                [f"{key}. {doc[key]}\n" for key in keys]
            )
            prompt = f"题目：{question}\n{choices}答案是："
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": keys.index(doc["Answer"]),
        }

    def fewshot_examples(self, k, rnd):
        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))
        return self._fewshot_docs[:k]
    
    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, "{}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls

    def doc_to_text(self, doc):
        return doc["query"]
    
    def doc_to_target(self, doc):
        return doc["choices"][doc["gold"]]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
