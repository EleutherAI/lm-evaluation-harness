from lm_eval.tasks.dynamic_ifeval.helper.rules import (randomize_rules_letter_must_be_in, count_number_of, sum_characters_sum)
from lm_eval.tasks.dynamic_ifeval.helper.utils import to_ordinal, default_empty_rules, get_texts, tuple_representer, tuple_constructor
import copy, random, yaml, os


def sample_rules(rules, randomize_rules):
    if randomize_rules["sample_rules"]:
        size_rules_to_sample = randomize_rules["size_rules_to_sample"]
        if size_rules_to_sample > len(rules):
            raise ValueError("size_rules_to_sample must be less than or equal to the number of letter_must_be_in rules.")
        rules_sampled = random.sample(list(rules.keys()), size_rules_to_sample)
        print("Sampled rules:", rules_sampled)
        for key in rules:
            if key in rules_sampled:
                if key == "count_number_of":
                    randomize_rules["sample_count_rules"] = True
                else:
                    rules[key]["enabled"] = True
            else:
                if key == "count_number_of":
                    randomize_rules["sample_count_rules"] = False
                    for sub_key in rules["count_number_of"]:
                        rules["count_number_of"][sub_key]["enabled"] = False
                else:
                    rules[key]["enabled"] = False
    return rules


def sample_count_rules(rules, randomize_rules):
    if randomize_rules["sample_count_rules"]:
        size_count_rules_to_sample = randomize_rules["size_count_rules_to_sample"]
        if size_count_rules_to_sample > len(rules["count_number_of"]):
            raise ValueError("size_count_rules_to_sample must be less than or equal to the number of count_number_of rules.")
        rules_sampled = random.sample(list(rules["count_number_of"].keys()), size_count_rules_to_sample)
        for key in rules["count_number_of"]:
            if key in rules_sampled:
                rules["count_number_of"][key]["enabled"] = True
            else:
                rules["count_number_of"][key]["enabled"] = False
    return rules


def randomize_rules_according_to_setup(rules, randomize_rules):
    rules = sample_rules(rules, randomize_rules)
    rules = sample_count_rules(rules, randomize_rules)
    return rules


def generate_prompt(text, rules):
    prompt = "Generate an English text such that"
    
    transition_adverbs = ["", "Furthermore, ", "Finally, "]
    counter_transition_adverb = 0
    
    rules_letter_must_be_in = []
    
    if rules["letter_must_be_in"]["enabled"]:
        number_letters = random.randint(0, len(text.split()) - 1) if rules["letter_must_be_in"]["randomize_number_letters"] else rules["letter_must_be_in"]["number_letters"]
        size_set_accepted_letters = random.randint(0, 26) if rules["letter_must_be_in"]["randomize_size_set_accepted_letters"] else rules["letter_must_be_in"]["size_set_accepted_letters"]
        
        rules_letter_must_be_in = randomize_rules_letter_must_be_in(text, size_set_accepted_letters=size_set_accepted_letters, number_letters=number_letters)
        for set_accepted_letters, pos_word, pos_letter in rules_letter_must_be_in:
            prompt = f"{prompt} the {to_ordinal(pos_letter+1)} character of the {to_ordinal(pos_word+1)} word is one of the letters of the set {set_accepted_letters},"
        prompt = f"{prompt[:-1]}."
        counter_transition_adverb += 1
    
    one_rule_was_enabled = False
    count_number = {}
    additional_prompt = f" {transition_adverbs[counter_transition_adverb]}the text must have exactly "
    
    for key, value in rules["count_number_of"].items():
        if value["enabled"]:
            count_number[key] = count_number_of(text, key)
            additional_prompt = f"{additional_prompt}{count_number[key]} {key}, "
            one_rule_was_enabled = True
    if one_rule_was_enabled:
        additional_prompt = f"{additional_prompt[:-2]}."
        prompt = f"{prompt}{additional_prompt}"
        counter_transition_adverb += 1
    
    sum_characters_value = 0
    if rules["sum_characters_must_be"]["enabled"]:
        sum_characters_value = sum_characters_sum(text)
        prompt = f"{prompt} {transition_adverbs[counter_transition_adverb]}if you filter the sentence to only include letters (ignoring case) and assign weights 1 to 26 based on their position in the alphabet, their sum must be exactly {sum_characters_value}. Non-letter characters such as commas, periods, and numbers are ignored and contribute 0 to the sum."
        counter_transition_adverb += 1
    
    prompt = f"{prompt} Only write the text, do not add anything else."
    return prompt, rules_letter_must_be_in, count_number, sum_characters_value


def make_hashable(obj):
    if isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, set):
        return frozenset(make_hashable(x) for x in obj)
    elif isinstance(obj, tuple):
        return tuple(make_hashable(x) for x in obj)
    else:
        return obj


def make_serializable(obj):
    if isinstance(obj, frozenset):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, tuple):
        print("Tuple detected:", obj)
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    else:
        return obj


def generate_tests(dataset, texts, rules, number_tests_per_setup, update_rules=None, max_attempts=300):
    i = 0
    j = 0
    while i < number_tests_per_setup:
        rules = copy.deepcopy(rules)
        if update_rules is not None:
            rules = copy.deepcopy(update_rules(rules))
        text = texts.get()
        prompt, rules_letter_must_be_in, count_number, sum_characters_value = generate_prompt(text, rules)
        hashable_tuple = (prompt, rules, rules_letter_must_be_in, count_number, sum_characters_value)  # make_hashable((prompt, rules, rules_letter_must_be_in, count_number, sum_characters_value)) if dataset is a set
        if hashable_tuple not in dataset:
            '''
            print(len(dataset))
            print("prompt:", prompt)
            print("rules:", rules)
            print("rules_letter_must_be_in:", rules_letter_must_be_in)
            print("count_number:", count_number)'''
            
            dataset.append(hashable_tuple)
            i += 1
        elif j >= max_attempts:
            print(f"Max attempts reached for prompt: {prompt}")
            break
        j += 1
    return dataset


def create_dataset(texts):
    number_tests_per_setup = 20
    
    print(len(texts))
    
    dataset = []
    
    #rules = randomize_rules_according_to_setup(rules, randomize_rules)
    rules_sum_characters_must_be = default_empty_rules()[0]
    rules_sum_characters_must_be["sum_characters_must_be"]["enabled"] = True

    # Generate a set of tests with the sum_characters_must_be rule
    dataset = generate_tests(dataset, texts, rules_sum_characters_must_be, number_tests_per_setup)
    
    # Generate a set of tests with the letter_must_be_in rule
    # We consider different numbers of letters and sizes of accepted letter
    rules_letter_must_be_in = default_empty_rules()[0]
    rules_letter_must_be_in["letter_must_be_in"]["enabled"] = True
    for number_letters in range(1, 4):
        rules_letter_must_be_in["letter_must_be_in"]["number_letters"] = number_letters
        for size_set_accepted_letters in range(1, 4):
            rules_letter_must_be_in["letter_must_be_in"]["size_set_accepted_letters"] = size_set_accepted_letters
            dataset = generate_tests(dataset, texts, rules_letter_must_be_in, number_tests_per_setup)
    
    # Generate a set of tests with the count_number_of rule
    # At each iteration, we increase the number of the count_rules to consider
    rules_count_number_of, randomize_rules_count_numer_of = default_empty_rules()
    randomize_rules_count_numer_of["sample_count_rules"] = True
    for size_count_rules_to_sample in range(1, len(rules_count_number_of["count_number_of"].keys())+1):
        randomize_rules_count_numer_of["size_count_rules_to_sample"] = size_count_rules_to_sample
        dataset = generate_tests(dataset, texts, rules_count_number_of, number_tests_per_setup, lambda _ : sample_count_rules(rules_count_number_of, randomize_rules_count_numer_of))

    # Generate a set of tests with the letter_must_be_in and count_number_of rules, both at their simplest
    mixed_rules, randomize_mixed_rules = default_empty_rules()
    mixed_rules["letter_must_be_in"]["enabled"] = True
    mixed_rules["letter_must_be_in"]["number_letters"] = 1
    mixed_rules["letter_must_be_in"]["size_set_accepted_letters"] = 1
    randomize_mixed_rules["sample_count_rules"] = True
    randomize_mixed_rules["size_count_rules_to_sample"] = 1
    
    dataset = generate_tests(dataset, texts, mixed_rules, number_tests_per_setup, lambda rules : sample_count_rules(rules, randomize_mixed_rules))
    
    return dataset


def save_dataset(dataset, filename="data/dataset.yaml"):
    yaml.add_representer(tuple, tuple_representer)
    yaml.add_constructor(u'tag:yaml.org,2002:python/tuple', tuple_constructor)
    # Convert each tuple in the dataset to a dictionary for YAML serialization
    data_to_save = []
    for prompt, rules, rules_letter_must_be_in, count_number, sum_characters_value in dataset:
        data_to_save.append({
            "prompt": prompt,
            "rules": make_serializable(rules),
            "rules_letter_must_be_in": make_serializable(rules_letter_must_be_in),
            "count_number": make_serializable(count_number),
            "sum_characters_value": sum_characters_value
        })
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w") as f:
        yaml.dump(data_to_save, f, allow_unicode=True)
    print(f"Dataset saved to {filename}")


if __name__ == "__main__":
    dataset = create_dataset(texts = get_texts())
    
    # sanity check
    i = 0
    for entry in dataset:
        rules = entry[1]
        count_number = entry[3]
        for key, value in rules["count_number_of"].items():
            if value["enabled"] and key not in count_number:
                # Either report an error or assign a default value.
                raise ValueError(f"Enabled rule {key}, {value} was not computed in count_number in entry {i}.")
        i += 1
        
    save_dataset(dataset)
