"""
Python-GSM8k, solving a GSM8k word problem by generating
a Python program that evaluates to the correct answer.
We use the 8-shot prompt from PAL: Program-Aided Language Models (ICML 2023),
and ask models to write a python function solution() 
of which its result will be printed as its answer.

The input problems are from the Lila dataset.

Evaluation requires executing Python code. To evaluate, see:
    `unsafe_score_python_gsm.py`
Please review the warnings regarding executing model-generated code.

Homepage: https://reasonwithpal.com/
"""

from lm_eval.metrics import mean
from lm_eval.base import Task, rf


_CITATION = """
@article{gao2022pal,
    title={PAL: Program-aided Language Models},
    author={Gao, Luyu and Madaan, Aman and Zhou, Shuyan and Alon, Uri and Liu, 
    Pengfei and Yang, Yiming and Callan, Jamie and Neubig, Graham},
    journal={arXiv preprint arXiv:2211.10435},
    year={2022}
}
                  
@inproceedings{mishra2022lila,
  author = {
    Swaroop Mishra 
      and Matthew Finlayson 
      and Pan Lu 
      and Leonard Tang 
      and Sean Welleck 
      and Chitta Baral 
      and Tanmay Rajpurohit 
      and Oyvind Tafjord 
      and Ashish Sabharwal 
      and Peter Clark 
      and Ashwin Kalyan},
  title = {Lila: A Unified Benchmark for Mathematical Reasoning},
  booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2022}
}
"""


# PAL math prompt, sourced from https://github.com/reasoning-machines/pal/blob/f81ca2a9777f002f98a6b4d0f10b61bd5c8feb02/pal/prompt/math_prompts.py#L16
PAL_PROMPT = '''
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

# solution in Python:


def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result





Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

# solution in Python:


def solution():
    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result





Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

# solution in Python:


def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result





Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

# solution in Python:


def solution():
    """Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"""
    toys_initial = 5
    mom_toys = 2
    dad_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    result = total_toys
    return result





Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

# solution in Python:


def solution():
    """Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"""
    jason_lollipops_initial = 20
    jason_lollipops_after = 12
    denny_lollipops = jason_lollipops_initial - jason_lollipops_after
    result = denny_lollipops
    return result





Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

# solution in Python:


def solution():
    """Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"""
    leah_chocolates = 32
    sister_chocolates = 42
    total_chocolates = leah_chocolates + sister_chocolates
    chocolates_eaten = 35
    chocolates_left = total_chocolates - chocolates_eaten
    result = chocolates_left
    return result





Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

# solution in Python:


def solution():
    """If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"""
    cars_initial = 3
    cars_arrived = 2
    total_cars = cars_initial + cars_arrived
    result = total_cars
    return result





Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

# solution in Python:


def solution():
    """There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"""
    trees_initial = 15
    trees_after = 21
    trees_added = trees_after - trees_initial
    result = trees_added
    return result





Q: {question}

# solution in Python:
'''.strip() 


class PythonGSM8k(Task):
    VERSION = 1
    DATASET_PATH = "allenai/lila"
    DATASET_NAME = 'GSM8k_structured'

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc):
        raise NotImplementedError()

    def doc_to_target(self, doc):
        raise NotImplementedError()

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["dev"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def construct_requests(self, doc, ctx):
        output = rf.greedy_until(ctx, "Q:")
        return output

    def fewshot_context(
            self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        prompt = PAL_PROMPT.format(question=doc["input"])
        return prompt

    def _parse_result(self, result):
        program = result.strip()
        return program

    def process_results(self, doc, results):
        program = self._parse_result(results[0])
        results = {
            "generated": 1.0,
            "metadata": {
                "program": program,
                "doc": doc,
            }
        }
        return results

    def aggregation(self):
        return {
            "generated": mean,
        }

    def higher_is_better(self):
        return {
            "generated": True,
        }

