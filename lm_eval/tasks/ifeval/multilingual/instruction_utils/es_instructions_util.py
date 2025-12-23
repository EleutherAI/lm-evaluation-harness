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

#NOTE:INCLUDE spacy installation in requirements or README in the future.
#pip3 install spacy    #The core library
#python -m spacy download es_core_news_sm    #The Spanish language model for tokenization and other NLP tasks.

import random
import re

import spacy
from iso639 import Lang


WORD_LIST = ["amigo", "comida", "escuela", "casa", "familia", "trabajo", "tiempo", "libro", "ciudad", "perro"]  # pylint: disable=line-too-long

# ISO 639-1 codes to language names
def lang_code_to_name(lang_code: str):
    return Lang(lang_code.lower()).name

_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr|Sr|San|Sra|Srta|Dra)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co|S.A.|S.L.)"
_STARTERS = r"(Mr|Sr|Mrs|Sra|Ms|Srta|Dr|Dra|Prof|Capt|Cpt|Cap|Lt|Tte|Él\s|Ella\s|Eso\s|Ellos\s|Ellas\s|Su\s|Nuestro\s|Nuestra\s|Nosotros\s|Nosotras\s|Pero\s|Sin embargo\s|Ese\s|Esa\s|Este\s|Esta\s|Dondequiera\s|En cuanto a\s|Por lo tanto\s|Por ejemplo\s|En resumen\s|En consecuencia\s|Por otro lado\s|Con respecto a\s|Sin embargo\s)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me|es|mx|ar|cl|co|pe|uy|ve|bo|do|gt|hn|py|cr|sv|ni|pa)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"

nlp = spacy.load("es_core_news_sm")

def split_into_sentences(text):
    """Split the text into sentences.

    Args:
        text: A string that consists of more than or equal to one sentences.

    Returns:
        A list of strings where each string is a sentence.
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(
            _MULTIPLE_DOTS,
            lambda match: "<prd>" * len(match.group(0)) + "<stop>",
            text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(
            _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
            "\\1<prd>\\2<prd>\\3<prd>",
            text,
    )
    text = re.sub(
            _ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text
    )
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences

def count_words(text):
    """Counts the number of words, respecting Spanish special characters and features with spacy."""
    # Load the Spanish tokenizer model from spacy
    tokenized_text = nlp(text)  # Process the text with the Spanish tokenizer
    num_words = len([token.text for token in tokenized_text if not token.is_punct])  # Count non-punctuation tokens
    return num_words

def tokenize_words(text):
    """Returns a list of words from the text, respecting Spanish special characters and features with spaCy."""
    # Load the Spanish tokenizer model from spaCy
    tokenized_text = nlp(text)  # Process the text with the Spanish tokenizer
    # Extract non-punctuation tokens
    words = [token.text for token in tokenized_text if not token.is_punct]
    return words

def count_sentences(text):
    """Count the number of sentences."""
    # Load the Spanish tokenizer model from spacy
    tokenized_text = nlp(text)  # Process the text with the Spanish tokenizer
    num_sentences = len(list(tokenized_text.sents))  # Count the number of sentences
    return num_sentences

def generate_keywords(num_keywords):
    """Randomly generates a few keywords."""
    return random.sample(WORD_LIST, k=num_keywords)