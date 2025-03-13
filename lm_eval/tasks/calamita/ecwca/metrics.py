
import evaluate
from typing import List, Dict
import nltk


def edit_distance(generations: List[str], correct_answers: List[str]) -> Dict[str, float]:
    avg_edit_distance = 0.0
    for predicted_answer, target_answer in zip(generations, correct_answers):
        avg_edit_distance += nltk.edit_distance(predicted_answer.replace(" ", ""), target_answer.replace(" ", ""))

    avg_edit_distance = avg_edit_distance/len(correct_answers) if len(correct_answers) > 0 else 0.0

    return {"avg_edit_distance": avg_edit_distance}


def words_avg_f1(generations: List[str], correct_answers: List[str]) -> Dict[str, float]:

    def get_words(txt: str) -> List[str]:
        return txt.split()

    avg_f1 = 0.0
    for predicted_answer, target_answer in zip(generations, correct_answers):
        predicted_answer_words = set(get_words(predicted_answer))
        target_answer_words = set(get_words(target_answer))

        words_in_common = predicted_answer_words & target_answer_words

        r = len(words_in_common)/len(target_answer_words) if len(target_answer_words) > 0 else 0.0
        p = len(words_in_common)/len(predicted_answer_words) if len(predicted_answer_words) > 0 else 0.0
        f1 = 2*(r * p)/(r + p) if r+p > 0 else 0.0

        avg_f1 += f1

    avg_f1 = avg_f1/len(correct_answers) if len(correct_answers) > 0 else 0

    return {"avg_words_f1": avg_f1}


