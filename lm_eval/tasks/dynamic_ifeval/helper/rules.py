import random, string, warnings
from typing import Union
from string import ascii_lowercase


def random_pos(words: Union[list, str]) -> int:
    random_pos = random.randint(0, len(words) - 1)
    return random_pos

def letter_must_be_in(sentence : str, set_accepted_letters : set[str], pos_word : int, pos_letter : int) -> bool:
    '''Check if the letter at the specified position in the word is in the set of accepted letters.'''
    words = sentence.split()
    if pos_word >= len(words):
        return False
    word = words[pos_word]
    if pos_letter >= len(word):
        return False
    letter = word[pos_letter]
    return letter in set_accepted_letters

def randomize_rule_letter_must_be_in(sentence : str, size_set_accepted_letters : int = 1) -> tuple[set[str], int, int]:
    '''Randomly select a word and letter position from the sentence and create a set of accepted letters for that position.'''
    ## TODO: generalize function for other characters and other alphabets
    words = sentence.split()
    pos_word = random_pos(words)
    pos_letter = random_pos(words[pos_word])
    true_letter = words[pos_word][pos_letter]
    if true_letter.islower():
        true_letter = true_letter.upper()
    set_accepted_letters = set([true_letter])
    if size_set_accepted_letters < 1:
        raise ValueError("size_set_accepted_letters must be at least 1")
    if size_set_accepted_letters > 26:
        raise ValueError("size_set_accepted_letters must be at most 26")  ##TODO: generalize for other characters
    while len(set_accepted_letters) < size_set_accepted_letters:
        random_letter = random.choice(string.ascii_uppercase)  ##TODO: generalize for other characters
        set_accepted_letters.add(random_letter)
    for letter in set_accepted_letters.copy():
        if letter.isupper():
            set_accepted_letters.add(letter.lower())
    return set_accepted_letters, pos_word, pos_letter

def randomize_rules_letter_must_be_in(sentence : str, size_set_accepted_letters : int = 1, number_letters : int = 1):
    '''Generate a list of rules where each rule specifies a set of acceptable letters.
    A letter from such set must be in a specific position of a word in the sentence.'''
    list_rules = []
    seen_positions = set()
    if len(sentence) <= number_letters:
        warnings.warn("number_letters is greater than the number of words in the sentence", UserWarning)
        number_letters = len(sentence) - 1
    if number_letters < 1:
        raise ValueError("number_letters must be at least 1")
    while len(list_rules) < number_letters:
        set_accepted_letters, pos_word, pos_letter = randomize_rule_letter_must_be_in(sentence, size_set_accepted_letters)
        if (pos_word, pos_letter) not in seen_positions:
            seen_positions.add((pos_word, pos_letter))
            list_rules.append((set_accepted_letters, pos_word, pos_letter))
    return list_rules

def count_number_of(sentence : str, item_name : str) -> int:
    '''Count the number of occurrences of a specific item in the sentence.'''
    if item_name == "words":
        return len(sentence.split())
    if item_name == "characters":
        return len(sentence)
    if item_name == "numbers":
        return sum(c.isdigit() for c in sentence)
    items = {"commas": ",", "periods": ".", "exclamation_marks": "!", "question_marks": "?", "semicolons": ";", "colons": ":"}
    if item_name not in items:
        raise ValueError(f"Invalid item_name: {item_name}. Valid options are: {list(items.keys())}")
    item = items[item_name]
    
    return sentence.count(item)

def number_of_must_be(sentence : str, item_name : str, expected_count : int) -> bool:
    ''' Check if the number of occurrences of a specific item in the sentence matches the expected count. '''
    return count_number_of(sentence, item_name) == expected_count

def sum_characters_sum(sentence : str) -> int:
    ''' Filter the sentence to only include letters (ignoring case) and assign weights 1 to 26 based on their position in the alphabet.
    Non-letter characters such as commas, periods, and numbers are ignored and contribute 0 to the sum. '''
    sentence_values = values = [
    ord(c) - ord("a") + 1
    for c in sentence.lower()          # lowercase everything
    if c in ascii_lowercase            # only a–z
]
    return sum(sentence_values)

def sum_characters_must_be(sentence : str, expected_sum_characters : int) -> bool:
    ''' Calculate the sum of characters in the sentence and compare it to the expected sum. '''
    return sum_characters_sum(sentence) == expected_sum_characters

def sentence_must_be_capitalized(sentence : str) -> bool:
    ''' Check if the string is fully capitalized. '''
    return sentence == sentence.upper()

def sentence_must_be_bold(sentence : str) -> bool:
    ''' Check if the string is fully bold. '''
    return sentence.startswith("**") and sentence.endswith("**") and len(sentence) >= 4

def sentence_must_be_italic(sentence : str) -> bool:    
    ''' Check if the string is fully italicized. '''
    return sentence.startswith("*") and sentence.endswith("*") and len(sentence) >= 2

def sentence_must_be_underline(sentence : str) -> bool:
    ''' Check if the string is fully underlined. '''
    return sentence.startswith("__") and sentence.endswith("__") and len(sentence) >= 4

def sentence_must_contain_word(sentence : str, word : str) -> bool:
    ''' Check if the sentence contains a specific word. '''
    return word in sentence

def randomize_rule_must_contain_word(sentence : str) -> bool:
    ''' Randomly select a word from the sentence. '''
    words = sentence.split()
    if len(words) == 0:
        raise ValueError("The sentence is empty.")
    return random.choice(words)
