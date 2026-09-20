import os, json, random, yaml
from itertools import chain


def to_ordinal(n):
    """
    Convert an integer to its ordinal string representation. Examples:
        1 -> '1st'
        2 -> '2nd'
        11 -> '11th'
        21 -> '21st'
    """
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        last_digit = n % 10
        if last_digit == 1:
            suffix = 'st'
        elif last_digit == 2:
            suffix = 'nd'
        elif last_digit == 3:
            suffix = 'rd'
        else:
            suffix = 'th'
    return f"{n}{suffix}"


class TextPool(list):
    def __init__(self, texts):
        super().__init__(texts)  # Initialize base list
        random.shuffle(self)
        self.__pos__ = 0

    def get_texts(self):
        return list(self)
    
    def get(self):
        if self.__pos__ >= len(self):
            self.__pos__ = 0
        text = self[self.__pos__]
        self.__pos__ += 1
        return text
    
    def copy(self):
        texts = TextPool(self.get_texts())
        texts.__pos__ = self.__pos__
        return texts


def get_texts(texts_types_to_use=None, texts_types_to_exclude=[]) -> TextPool:
    # default will return all texts
    if (texts_types_to_use is None) == (texts_types_to_exclude is None):
        raise ValueError("You must provide either texts_types_to_use or texts_types_to_exclude, but not both.")
    json_file = os.path.join(os.path.dirname(__file__), "data/texts_dataset.json")
    with open(json_file, "r") as f:
        texts_dataset = json.load(f)
    if texts_types_to_use is not None:
        return TextPool(list(chain.from_iterable(
            texts_dataset[key] for key in texts_types_to_use if key in texts_dataset
        )))
    else:
        return TextPool(list(chain.from_iterable(
            texts_dataset[key] for key in texts_dataset if key not in texts_types_to_exclude
        )))


def default_empty_rules():
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "config.yaml")
    
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
        
    rules = config["rules"]
    randomize_rules = config["randomize_rules"]
    
    rules["letter_must_be_in"]["enabled"] = False
    rules["letter_must_be_in"]["randomize_number_letters"] = False
    rules["letter_must_be_in"]["number_letters"] = 0
    rules["letter_must_be_in"]["randomize_size_set_accepted_letters"] = False
    rules["letter_must_be_in"]["size_set_accepted_letters"] = 0
    for key in rules["count_number_of"]:
        rules["count_number_of"][key]["enabled"] = False
    rules["sum_characters_must_be"]["enabled"] = False
    
    randomize_rules["sample_rules"] = False
    randomize_rules["size_rules_to_sample"] = 0
    randomize_rules["sample_count_rules"] = False
    randomize_rules["size_count_rules_to_sample"] = 0
    
    return rules, randomize_rules


def tuple_representer(dumper, data):
    return dumper.represent_list(list(data))


def tuple_constructor(loader, node):
    # Convert the YAML sequence back into a tuple.
    return tuple(loader.construct_sequence(node))


def load_dataset(filename="data/dataset.yaml"):
    yaml.add_representer(tuple, tuple_representer)
    yaml.add_constructor(u'tag:yaml.org,2002:python/tuple', tuple_constructor)
    with open(filename, "r") as f:
        return yaml.safe_load(f)
    
    
def get_config():    
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    return config
