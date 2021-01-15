import abc
import json
import random
from collections import namedtuple
from lm_eval.base import Dataset, mean, rf

ArithmeticDoc = namedtuple('ArithmeticDoc', ['question_text', 'answer_text'])

class Arithmetic(Dataset):
    def __init__(self, number_of_problems=2000):
        super().__init__()
        self.problems = self.generate_problems(number_of_problems)

    @abc.abstractmethod
    def generate_problems(self, number_of_problems):
        pass

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.problems

    def validation_docs(self):
        return self.generate_problems(50)

    def test_docs(self):
        return NotImplemented
    
    def doc_to_text(self, doc):
        return f"Q: What is {doc.question_text}?\nA: "

    def doc_to_target(self, doc):
        return doc.answer_text

    def construct_requests(self, doc, ctx):
        ll, is_prediction = rf.loglikelihood(ctx, doc.answer_text)
        # not sure what the difference between the two objects returned by rf.loglikehood are here
        return is_prediction

    def process_results(self, doc, results):
        ll, is_prediction = results
        return {
            "acc": is_prediction
        }

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {
            "acc": True
        }


class Arithmetic2DPlus(Arithmetic):
    def generate_problems(self, number_of_problems):
        l = []
        for i in range(number_of_problems):
            x,y = random.randint(0,99), random.randint(0,99)
            a = x+y
            l.append(ArithmeticDoc(question_text=f'{x}+{y}', answer_text=f'{a}'))
        return l

class Arithmetic2DMinus(Arithmetic):
    def generate_problems(self, number_of_problems):
        l = []
        for i in range(number_of_problems):
            x,y = random.randint(0,99), random.randint(0,99)
            a = x-y
            l.append(ArithmeticDoc(question_text=f'{x}-{y}', answer_text=f'{a}'))
        return l

class Arithmetic3DPlus(Arithmetic):
    def generate_problems(self, number_of_problems):
        l = []
        for i in range(number_of_problems):
            x,y = random.randint(0,999), random.randint(0,999)
            a = x+y
            l.append(ArithmeticDoc(question_text=f'{x}+{y}', answer_text=f'{a}'))
        return l

class Arithmetic3DMinus(Arithmetic):
    def generate_problems(self, number_of_problems):
        l = []
        for i in range(number_of_problems):
            x,y = random.randint(0,999), random.randint(0,999)
            a = x-y
            l.append(ArithmeticDoc(question_text=f'{x}-{y}', answer_text=f'{a}'))
        return l

class Arithmetic4DPlus(Arithmetic):
    def generate_problems(self, number_of_problems):
        l = []
        for i in range(number_of_problems):
            x,y = random.randint(0,9999), random.randint(0,9999)
            a = x+y
            l.append(ArithmeticDoc(question_text=f'{x}+{y}', answer_text=f'{a}'))
        return l

class Arithmetic4DMinus(Arithmetic):
    def generate_problems(self, number_of_problems):
        l = []
        for i in range(number_of_problems):
            x,y = random.randint(0,9999), random.randint(0,9999)
            a = x-y
            l.append(ArithmeticDoc(question_text=f'{x}-{y}', answer_text=f'{a}'))
        return l

class Arithmetic5DPlus(Arithmetic):
    def generate_problems(self, number_of_problems):
        l = []
        for i in range(number_of_problems):
            x,y = random.randint(0,99999), random.randint(0,99999)
            a = x+y
            l.append(ArithmeticDoc(question_text=f'{x}+{y}', answer_text=f'{a}'))
        return l

class Arithmetic5DMinus(Arithmetic):
    def generate_problems(self, number_of_problems):
        l = []
        for i in range(number_of_problems):
            x,y = random.randint(0,99999), random.randint(0,99999)
            a = x-y
            l.append(ArithmeticDoc(question_text=f'{x}-{y}', answer_text=f'{a}'))
        return l

class Arithmetic2DMultiplication(Arithmetic):
    def generate_problems(self, number_of_problems):
        l = []
        for i in range(number_of_problems):
            x,y = random.randint(0,99), random.randint(0,99)
            a = x*y
            l.append(ArithmeticDoc(question_text=f'{x}*{y}', answer_text=f'{a}'))
        return l

class Arithmetic1DComposite(Arithmetic):
    def generate_problems(self, number_of_problems):
        l = []
        for i in range(number_of_problems):
            x,y,z = random.randint(0,9), random.randint(0,9), random.randint(0,9)
            op1, op2 = random.choice('-+*'), random.choice('-+*') 
            to_eval = f'{x}{op1}({y}{op2}{z})'
            l.append(ArithmeticDoc(question_text=to_eval, answer_text=str(eval(to_eval))))
        return l
