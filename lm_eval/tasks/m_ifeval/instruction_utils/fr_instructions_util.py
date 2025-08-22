# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility library of instructions."""

import functools
import random
import re
from typing import List
import unicodedata

import immutabledict
import nltk

nltk.download('punkt_tab')

WORD_LIST = ["occidental", "phrase", "signal", "château", "tache", "opposé", "bas", "pomme de terre", "administration", "étoile"]  # pylint: disable=line-too-long

# ISO 639-1 codes to language names.
LANGUAGE_CODES = immutabledict.immutabledict({
    "en": "Anglais",
    "es": "Espagnol",
    "pt": "Portugais",
    "ar": "Arabe",
    "hi": "Hindi",
    "fr": "Français",
    "ru": "Russe",
    "de": "Allemand",
    "ja": "Japonais",
    "it": "Italien",
    "bn": "Bengali",
    "uk": "Ukrainien",
    "th": "Thaï",
    "ur": "Ourdou",
    "ta": "Tamoul",
    "te": "Télougou",
    "bg": "Bulgare",
    "ko": "Coréen",
    "pl": "Polonais",
    "he": "Hébreu",
    "fa": "Persan",
    "vi": "Vietnamien",
    "ne": "Népalais",
    "sw": "Swahili",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati", 
    "pa": "Punjabi", 
    "ml": "Malayalam", 
    "fi": "Finnois",
    "zh": "Chinois simplifié"
    })

_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(M|Mme|Mlle|Dr)[.]"
_SUFFIXES = "(SARL|SA|Jr|Sr|Co)"
_STARTERS = r"(M.|Mme|Mlle|Dr|Pr|Cap|Lt|Il\s|Elle\s|Cela\s|Ils\s|Leurs\s|Notre\s|Nous\s|Mais\s|Cependant\s|Cela\s|Ce\s|Où que\s)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me|fr|ac)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"

# abbreviations should be in lower case in the below definition, 
_ABBREVIATIONS = [
                      # Titles
                      'm', 'mr', 'mme', 'mlle', 'dir.', 'dr', 'dre', 'drs', 'dres', 'prof', 'pr', 'cap', 'lt',
                      # Admin
                      'adm.', 'assoc.', 'min.', 
                      # Examples
                      'etc', 'cf', 'vs', 'rép', 'i.e', 'c.-à-d', 'ex', 'e.g', 
                      # Time
                      'apr', 'av', 'j.c.', 'j.-c.', 'j.c', 'j.-c', 'av. è. c.', 'è. c', 'a.d', 'av. n. è', 'de n. è.',
                      'env', 'janv', 'fév', 'mar', 'avr', 'juil', 'sept', 'oct', 'nov', 'déc', 'p.-ê.',
                      # Others
                      'svp', 'p.-v.', 'p.v', 'tél', 'adj', 'cf', 'q.v', 'p',
                      'resp', 'tel', 'v', 'vol', 'nb', 'n.b', 'p.s', 'ps', 'p.p.s', 'p.-s', 'p-s', 
                     ] 

# Natively handles:
# - websites: 'Veuillez visiter mon site internet: https://google.com/exemple.edu.ac'
# - decimal numbers: '3.14'
# - ellipsis: '...'
# - accronyms: 'Je suis né aux E.-U. et toi ?' --> 1 sentence
@functools.lru_cache(maxsize=None)
def _get_sentence_tokenizer():
    """Load and return a French sentence tokenizer with customized abbreviation handling.

      This function loads the default French sentence tokenizer from NLTK and updates it to handle
      common French abbreviations, ensuring accurate sentence segmentation. The tokenizer 
      is capable of handling:

      - **Websites**: Properly manages URLs without splitting sentences incorrectly. 
        Example: 'Veuillez visiter mon site internet: https://google.com/exemple.edu.ac' 
        is processed as a single sentence.
        
      - **Decimal numbers**: Correctly processes numbers with decimal points without 
        splitting at the period. Example: '3.14' is processed as a single sentence.
        
      - **Ellipsis**: Recognizes ellipses ('...') as a single punctuation mark and treats 
        it appropriately in sentence boundaries.
        
      - **Acronyms**: Handles common French acronyms and abbreviations without incorrectly 
        splitting sentences. Example: 'Je suis né aux E.-U. et toi ?' is processed as a 
        single sentence.
        
      - **Abbreviations**: Manages various French abbreviations to prevent incorrect sentence 
        splitting at points where periods are used in abbreviations. Example: 'Dr. Dupont a 
        appelé.' is processed correctly with 'Dr.' recognized as an abbreviation.

      Returns:
          PunktSentenceTokenizer: An NLTK Punkt tokenizer instance configured for French, 
          with updated handling for abbreviations, websites, decimal numbers, ellipses, and acronyms.
    """
    
    # Load the default French sentence tokenizer
    tokenizer = nltk.data.load("nltk:tokenizers/punkt/french.pickle")
    
    # Add common French abbreviations
    tokenizer._params.abbrev_types.update(_ABBREVIATIONS)
    return tokenizer

def split_into_sentences(text):
    """Split the text into sentences.

    Args:
      text: A string that consists of more than or equal to one sentences.

    Returns:
      A list of strings where each string is a sentence.
    """
    tokenizer = _get_sentence_tokenizer()
    sentences = tokenizer.tokenize(text)
    return sentences


def count_words(text):
  """Counts the number of words."""
  tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
  tokens = tokenizer.tokenize(text)
  num_words = len(tokens)
  return num_words


def count_sentences(text):
  """Count the number of sentences."""
  tokenized_sentences = split_into_sentences(text)
  return len(tokenized_sentences)


def generate_keywords(num_keywords):
  """Randomly generates a few keywords."""
  return random.sample(WORD_LIST, k=num_keywords)


def remove_accents(text):
    """
    Remove accents from the input string.

    Args:
        text (str): The input string with accents.

    Returns:
        (str): The input string with accents removed.
    """
    nfkd_form = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])