"""
JGLUE: Japanese General Language Understanding Evaluation
https://aclanthology.org/2022.lrec-1.317/

JGLUE, Japanese General Language Understanding Evaluation, is built to measure the general NLU ability in Japanese. 
JGLUE has been constructed from scratch without translation. 

Homepage: https://github.com/yahoojapan/JGLUE
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{kurihara-etal-2022-jglue,
    title = "{JGLUE}: {J}apanese General Language Understanding Evaluation",
    author = "Kurihara, Kentaro  and
      Kawahara, Daisuke  and
      Shibata, Tomohide",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.317",
    pages = "2957--2966",
    abstract = "To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}
"""



class JCommonsenseQA(MultipleChoiceTask):
    """
    prompt format is taken from [日本語に特化した60億パラメータ規模のGPTモデルの構築と評価](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/H9-4.pdf)
    """
    # VERSION = {structure_version}.{prompt_template_version}
    VERSION = 0.1
    DATASET_PATH = "shunk031/JGLUE"
    DATASET_NAME = "JCommonsenseQA"
    DESCRIPTION = "[問題]に対する[答え]を[選択肢]の中から選んでください。\n\n"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        return {
            "goal": doc["question"],
            "choices": [doc[f"choice{i}"] for i in range(5)],
            "gold": doc["label"], 
        }

    def doc_to_text(self, doc):
        """
        [問題]:question
        [選択肢]:[choice0, choice1, ..., choice4]
        [答え]:
        """
        return (
            f"[問題]:{doc['goal']}\n"
            f"[選択肢]:[{', '.join(doc['choices'])}]\n"
            "[答え]:"
        )
    
    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["goal"]
    

class JCommonsenseQAWithFintanPrompt(JCommonsenseQA):
    """
    prompt template is taken from [ChatGPT vs BERT: どちらが日本語をより理解できるのか?](https://fintan.jp/page/9126/)
    """
    # VERSION = {structure_version}.{prompt_template_version}
    VERSION = 0.2
    DESCRIPTION = "質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。なお、回答は選択肢の番号(例:0)でするものとします。 \n\n"


    def doc_to_text(self, doc):
        """
        質問:question
        選択肢:0.choice0,1.choice1, ...,4.choice4
        回答:
        """
        choices = ",".join([f"{idx}.{choice}" for idx, choice in enumerate(doc['choices'])])
        return (
            f"質問:{doc['goal']}\n"
            f"選択肢:{choices}\n"
            "回答:"
        )

VERSIONS = [
    JCommonsenseQA,
    JCommonsenseQAWithFintanPrompt,
]


def construct_tasks():
    tasks = {}
    for version_class in VERSIONS:
        tasks[f"jcommonsenseqa_{version_class.VERSION}"] = version_class
    return tasks
