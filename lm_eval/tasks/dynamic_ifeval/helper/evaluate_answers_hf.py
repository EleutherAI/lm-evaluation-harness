from lm_eval.tasks.dynamic_ifeval.helper.rules import letter_must_be_in, number_of_must_be, sum_characters_must_be, count_number_of, sum_characters_sum
from lm_eval.tasks.dynamic_ifeval.helper.utils import load_dataset

from datasets import Dataset
import yaml, pickle, re, ast


def aux_evaluate_answer(answer, rules, rules_letter_must_be_in, count_number, sum_characters_value):
    pass_all_rules = True
    
    if rules["letter_must_be_in"]["enabled"]:
        for set_accepted_letters, pos_word, pos_letter in rules_letter_must_be_in:
            if not letter_must_be_in(answer, set_accepted_letters, pos_word, pos_letter):
                #print("Wrong answer:", set_accepted_letters, pos_word, pos_letter)
                pass_all_rules = False
        
    for key, value in rules["count_number_of"].items():
        if value["enabled"] and not number_of_must_be(answer, key, count_number[key]):
            #print(f"Wrong answer: expected number of {key} is {count_number[key]} but {count_number_of(answer, key)} were found.")
            pass_all_rules = False
    if rules["sum_characters_must_be"]["enabled"]:
        if not sum_characters_must_be(answer, sum_characters_value):
            #print(f"Wrong answer: expected sum of characters is {sum_characters_value} but {sum_characters_sum(answer)} were found.")
            pass_all_rules = False
    #if pass_all_rules:
    #    print("All rules passed.")
    return pass_all_rules


def evaluate_answer(answer, rules, rules_letter_must_be_in, count_number, sum_characters_value):
    if answer is None or answer == "":
        return False
    if aux_evaluate_answer(answer, rules, rules_letter_must_be_in, count_number, sum_characters_value):
        return True
    answer2 = re.sub(r"(?s)(?:<think>.*?</think>|^.*?</think>\s*)", "", answer)
    if answer != answer2 and evaluate_answer(answer2, rules, rules_letter_must_be_in, count_number, sum_characters_value):
        print(f"Answer without <think> tag is correct. answer = {answer[:100]} \n answer2 = {answer2[:100]}")
        return True
    m = re.search(r"<answer>(.*?)</answer>", answer, flags=re.DOTALL)
    if m:
        answer2 = m.group(1).strip()
        if evaluate_answer(answer2, rules, rules_letter_must_be_in, count_number, sum_characters_value):
            print(f"Answer within <answer> tag is correct. answer = {answer[:100]} \n answer2 = {answer2[:100]}")
            return True
    m = re.search(r"\\boxed\{(.*?)\}", answer, flags=re.DOTALL)
    if m:
        answer2 = m.group(1).strip()
        if evaluate_answer(answer2, rules, rules_letter_must_be_in, count_number, sum_characters_value):
            print(f"Answer within \\boxed{{}} is correct. answer = {answer[:100]} \n answer2 = {answer2[:100]}")
            return True
    answer2 = answer2.replace("*", "")
    if answer != answer2 and evaluate_answer(answer2, rules, rules_letter_must_be_in, count_number, sum_characters_value):
        print(f"Answer without * is correct. answer = {answer[:100]} \n answer2 = {answer2[:100]}")
        return True
    answer2 = re.sub(r"^.*\n\n", "", answer, flags=re.DOTALL)
    if answer != answer2 and evaluate_answer(answer2, rules, rules_letter_must_be_in, count_number, sum_characters_value):
        print(f"Answer after last '\\n\\n' is correct. answer = {answer[:100]} \n answer2 = {answer2[:100]}")
        return True
    return False
    
    
def load_answers(filename="results/answers.pkl"):
    with open(filename, 'rb') as file:
        answers = pickle.load(file)
    return answers


def compute_success_rates(answers, dataset):
    success_rates = {}
    for model_name, answers_model_matrix in answers.items():
        success_rates[model_name] = []
        for answers_entry, entry in zip(answers_model_matrix, dataset):
            success_rates_entry = [evaluate_answer(answer, entry["rules"], ast.literal_eval(entry["rules_letter_must_be_in"]),
                                                    entry["count_number"], entry["sum_characters_value"])
                                   for answer in answers_entry]
            success_rates[model_name].append(success_rates_entry)
    return success_rates


def save_success_rates_with_pickle(success_rates, filename="results/success_rates.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(success_rates, file)

    print(f"Data saved to {filename}")
    
    
def save_evaluation_results_to_yaml(evaluation_results, filename="results/evaluation_results.yaml"):
    with open(filename, "w") as f:
        yaml.dump(evaluation_results, f, allow_unicode=True)
    print(f"Dataset saved to {filename}")
    

def average_success_rates(success_rates):
    evaluation_results = {}
    for model_name, success_rates_matrix in success_rates.items():
        evaluation_results[model_name] = [sum(row)/len(row) for row in success_rates_matrix]
        #print(f"len(evaluation_results[{model_name}]) = {len(evaluation_results[model_name])}")
    return evaluation_results


def evaluate_answers():
    answers = load_answers("results/answers.pkl")
    dataset = Dataset.load_from_disk("data/hf_dataset")  # load_dataset()
    
    success_rates = compute_success_rates(answers, dataset)
    save_success_rates_with_pickle(success_rates=success_rates)
    
    evaluation_results = average_success_rates(success_rates)
    save_evaluation_results_to_yaml(evaluation_results=evaluation_results)
    
    
if __name__ == "__main__":
    evaluate_answers()