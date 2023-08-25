"""
Evaluating the Performance of Large Language Models on GAOKAO Benchmark
https://arxiv.org/abs/2305.12474

an intuitive benchmark that employs questions from the Chinese Gaokao examination as test samples for 
evaluating large language this http URL order to align the evaluation results with humans as much as possible

Homepage: https://github.com/OpenLMLab/GAOKAO-Bench
"""
import re
from lm_eval.base import Task, rf
from lm_eval.metrics import mean

_CITATION = """
@inproceedings{Zhang2023EvaluatingTP,
  title={Evaluating the Performance of Large Language Models on GAOKAO Benchmark},
  author={Xiaotian Zhang and Chunyang Li and Yi Zong and Zhengyu Ying and Liang He and Xipeng Qiu},
  year={2023}
}
"""

SUBJECTS = [
    {
        "type": "single_choice",
        "keyword": "2010-2022_Math_II_MCQs",
        "prefix_prompt": "请你做一道数学选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：",
        "comment": ""
    },
    {
        "type": "single_choice",
        "keyword": "2010-2022_Math_I_MCQs",
        "prefix_prompt": "请你做一道数学选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：",
        "comment": ""
    },
    {
        "type": "single_choice",
        "keyword": "2010-2022_History_MCQs",
        "prefix_prompt": "请你做一道历史选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下："
    },
    {
        "type": "single_choice",
        "keyword": "2010-2022_Biology_MCQs",
        "prefix_prompt": "请你做一道生物选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下："
    },
    {
        "type": "single_choice",
        "keyword": "2010-2022_Political_Science_MCQs",
        "prefix_prompt": "请你做一道政治选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下："
    },
    {
        "type": "multi_choice",
        "keyword": "2010-2022_Physics_MCQs",
        "prefix_prompt": "请你做一道物理选择题。\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出所有符合题意的答案，并写在【答案】和<eoa>之间。\n例如：【答案】 AB <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】... <eoa>\n请你严格按照上述格式作答。\n"
    },
    {
        "type": "single_choice",
        "keyword": "2010-2022_Chemistry_MCQs",
        "prefix_prompt": "请你做一道化学选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下："
    },
    {
        "type": "single_choice",
        "keyword": "2010-2013_English_MCQs",
        "prefix_prompt": "请你做一道英语选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下："
    },
    {
        "type": "multi_question_choice",
        "keyword": "2010-2022_Chinese_Modern_Lit",
        "prefix_prompt": "请你做一道语文阅读理解题，其中包含三个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n"
    },
    {
        "type": "multi_question_choice",
        "keyword": "2010-2022_English_Fill_in_Blanks",
        "prefix_prompt": "请你做一道英语完形填空题,其中包含二十个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n"
    },
    {
        "type": "five_out_of_seven",
        "keyword": "2012-2022_English_Cloze_Test",
        "prefix_prompt": "请回答下面的问题，将符合题意的五个选项的字母写在【答案】和<eoa>之间，例如“【答案】 A B C D E <eoa>\n请严格按照上述格式作答。\n"
    },
    {
        "type": "multi_question_choice",
        "keyword": "2010-2022_Geography_MCQs",
        "prefix_prompt": "请你做一道地理选择题，其中包含两到三个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n"
    },
    {
        "type": "multi_question_choice",
        "keyword": "2010-2022_English_Reading_Comp",
        "prefix_prompt": "请你做一道英语阅读理解题，其中包含三到五个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n"
    },
    {
        "type": "multi_question_choice",
        "keyword": "2010-2022_Chinese_Lang_and_Usage_MCQs",
        "prefix_prompt": "请你做一道语文选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n（1）【解析】 ... <eoe>\n【答案】 ... <eoa>\n（2）【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。如果不止一道题，请分别作答\n题目如下："
    }
]

def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {GaoKao-2010-2022_Math_II_MCQs: Task, 2010-2022_Math_I_MCQs: Task}
    """
    return {f"GaoKao-{sub['keyword']}": create_task(sub["keyword"], sub["type"]) for sub in SUBJECTS}


def create_task(subject, qtype):
    class GaoKaoBench(GaoKaoBenchSubject):
        def __init__(self):
            super().__init__(subject, qtype)

    return GaoKaoBench

class GaoKaoBenchSubject(Task):
    VERSION = 0
    DATASET_PATH = "AsakusaRinne/gaokao_bench"
    DATASET_NAME = None

    def __init__(self, subject, qtype):
        self.DATASET_NAME = subject
        self.question_type = qtype
        super().__init__()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        raise NotImplementedError

    def validation_docs(self):
        raise NotImplementedError

    def test_docs(self):
        return self.dataset["dev"]

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        keyword = self.DATASET_NAME
        for subject in SUBJECTS:
            if subject["keyword"] == keyword:
                prefix_prompt = subject["prefix_prompt"]
                break
        description = prefix_prompt
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)
    
    def doc_to_text(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        target = ""
        return " " + target

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        completion = rf.greedy_until(ctx, {"until": []})
        return completion

    def _extract_choice_answer(self, model_output, question_type, answer_lenth=None):
        """
        Extract choice answer from model output

        Format of model_output that is expected:
        'single_choice': choice answer should be the last Capital Letter of the model_output, e.g.: "...【答案】 A <eoa>"
        'multi_question_choice': "...【答案】A ... 【答案】C ..." or write the choice answers at the beginning of the model_output, e.g. "A C D E F...."
        'multi_choice': "...【答案】 ABD " or write the choice answers at the end of the model_output, e.g. "... ACD"
        'five_out_of_seven': choice answers should be the first five Capital Letters of the model_output, e.g. "A C D F B ...."
        """
        if question_type == 'single_choice':
            model_answer = []
            temp = re.findall(r'[A-D]', model_output[::-1])
            if len(temp) != 0:
                model_answer.append(temp[0])

        elif question_type == 'multi_question_choice':
            model_answer = []
            temp = re.findall(r"【答案】\s*[:：]*\s*[A-Z]", model_output)
                
            if len(temp) == answer_lenth:
                for t in temp:
                    model_answer.append(re.findall(r'[A-Z]', t)[0])
            else:
                temp = re.findall(r"[A-Z]", model_output)
                if len(temp) > 0:
                    for k in range(min(len(temp), answer_lenth)):
                        model_answer.append(temp[k])

        elif question_type == 'multi_choice':
            model_answer = []
            answer = ''
            content = re.sub(r'\s+', '', model_output)
            answer_index = content.find('【答案】')
            if answer_index > 0:
                temp = content[answer_index:]
                if len(re.findall(r'[A-D]', temp)) > 0:
                    for t in re.findall(r'[A-D]', temp):
                        answer += t
            else:
                temp = content[-10:]
                if len(re.findall(r'[A-D]', temp)) > 0:
                    for t in re.findall(r'[A-D]', temp):
                        answer += t
            if len(answer) != 0:
                model_answer.append(answer)
        
        elif question_type == 'five_out_of_seven':
            model_answer = []
            temp = re.findall(r'[A-G]', model_output)
            if len(temp) > 0:
                for k in range(min(5, len(temp))):
                    model_answer.append(temp[k])

        return model_answer

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
        standard_answer = doc["answer"].split(",")
        score = doc["score"]
        model_answer = self._extract_choice_answer(completion, self.question_type, len(standard_answer))
        if len(model_answer) != len(standard_answer):
            print("model_answer and standard_answer length is not equal, subject:" + self.question_type + ", index:" + str(doc["index"]))
            model_answer = ["Z"] * len(standard_answer) 

        correct_score : float = 0.0
        total_score : float = 0.0
        total_score += len(standard_answer) * score
        if self.question_type == "multi_choice":
            for i in range(len(model_answer)):
                if model_answer[i] == standard_answer[i]:
                    correct_score += score
                else:
                    is_error = 0
                    for z in model_answer[i]:
                        if z not in standard_answer[i]:
                            is_error = 1
                            break
                    correct_score += 0 if is_error else score/2
        else:
            for i in range(len(standard_answer)):
                if model_answer[i] == standard_answer[i]:
                    correct_score += score

        return { "correct_score":correct_score, "total_score":total_score }

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        return { "correct_score":sum, "total_score":sum }

    def higher_is_better(self):
        return { "correct_score":True, "total_score":False }
