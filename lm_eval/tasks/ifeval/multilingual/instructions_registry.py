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

"""Registry of all instructions."""
from lm_eval.tasks.ifeval.multilingual.instructions import (
    ca_instructions,
    es_instructions,
)


_KEYWORD = "keywords:"

_LANGUAGE = "language:"

_LENGTH = "length_constraints:"

_CONTENT = "detectable_content:"

_FORMAT = "detectable_format:"

_MULTITURN = "multi-turn:"

_COMBINATION = "combination:"

_STARTEND = "startend:"

_CHANGE_CASES = "change_case:"

_PUNCTUATION = "punctuation:"

_LETTERS = "letters:"

_SPECIAL_CHARACTER = "special_character:"

ES_INSTRUCTION_DICT = {
    _KEYWORD + "existence": es_instructions.KeywordChecker,
    _KEYWORD + "frequency": es_instructions.KeywordFrequencyChecker,
    _KEYWORD + "forbidden_words": es_instructions.ForbiddenWords,
    _KEYWORD + "letter_frequency": es_instructions.LetterFrequencyChecker,
    _LANGUAGE + "response_language": es_instructions.ResponseLanguageChecker,
    _LENGTH + "number_words": es_instructions.NumberOfWords,
    _LENGTH + "number_sentences": es_instructions.NumberOfSentences,
    _LENGTH + "number_paragraphs": es_instructions.ParagraphChecker,
    _LENGTH + "nth_paragraph_first_word": es_instructions.ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": es_instructions.PlaceholderChecker,
    _CONTENT + "postscript": es_instructions.PostscriptChecker,
    _FORMAT + "number_bullet_lists": es_instructions.BulletListChecker,
    _FORMAT + "constrained_response": es_instructions.ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": (
        es_instructions.HighlightSectionChecker),
    _FORMAT + "multiple_sections": es_instructions.SectionChecker,
    _FORMAT + "json_format": es_instructions.JsonFormat,
    _FORMAT + "title": es_instructions.TitleChecker,
    _COMBINATION + "two_responses": es_instructions.TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": es_instructions.RepeatPromptThenAnswer,
    _PUNCTUATION + "no_comma": es_instructions.CommaChecker,
    _PUNCTUATION + "question_marks": es_instructions.QuestionMarkChecker,
    _PUNCTUATION + "exclamation_marks": es_instructions.ExclamationMarkChecker,
    _STARTEND + "end_checker": es_instructions.EndChecker,
    _STARTEND + "quotation": es_instructions.QuotationChecker,
    _CHANGE_CASES
    + "capital_word_frequency": es_instructions.CapitalWordFrequencyChecker,
    _CHANGE_CASES
    + "spanish_capital": es_instructions.CapitalLettersSpanishChecker,
    _CHANGE_CASES
    + "spanish_lowercase": es_instructions.LowercaseLettersSpanishChecker,
    _SPECIAL_CHARACTER + "enie": es_instructions.EnieChecker,
    _SPECIAL_CHARACTER + "tildes": es_instructions.TildesChecker,
    _SPECIAL_CHARACTER + "dieresis": es_instructions.DieresisChecker,
}

CA_INSTRUCTION_DICT = {
    _KEYWORD + "existence": ca_instructions.KeywordChecker,
    _KEYWORD + "frequency": ca_instructions.KeywordFrequencyChecker,
    _KEYWORD + "forbidden_words": ca_instructions.ForbiddenWords,
    _KEYWORD + "letter_frequency": ca_instructions.LetterFrequencyChecker,
    _LANGUAGE + "response_language": ca_instructions.ResponseLanguageChecker,
    _LENGTH + "number_words": ca_instructions.NumberOfWords,
    _LENGTH + "number_sentences": ca_instructions.NumberOfSentences,
    _LENGTH + "number_paragraphs": ca_instructions.ParagraphChecker,
    _LENGTH + "nth_paragraph_first_word": ca_instructions.ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": ca_instructions.PlaceholderChecker,
    _CONTENT + "postscript": ca_instructions.PostscriptChecker,
    _FORMAT + "number_bullet_lists": ca_instructions.BulletListChecker,
    _FORMAT + "constrained_response": ca_instructions.ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": (
        ca_instructions.HighlightSectionChecker),
    _FORMAT + "multiple_sections": ca_instructions.SectionChecker,
    _FORMAT + "json_format": ca_instructions.JsonFormat,
    _FORMAT + "title": ca_instructions.TitleChecker,
    _COMBINATION + "two_responses": ca_instructions.TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": ca_instructions.RepeatPromptThenAnswer,
    _PUNCTUATION + "no_comma": ca_instructions.CommaChecker,
    _PUNCTUATION + "question_marks": ca_instructions.QuestionMarkChecker,
    _PUNCTUATION + "exclamation_marks": ca_instructions.ExclamationMarkChecker,
    _STARTEND + "end_checker": ca_instructions.EndChecker,
    _STARTEND + "quotation": ca_instructions.QuotationChecker,
    _CHANGE_CASES
    + "capital_word_frequency": ca_instructions.CapitalWordFrequencyChecker,
    _CHANGE_CASES
    + "catalan_capital": ca_instructions.CapitalLettersCatalanChecker,
    _CHANGE_CASES
    + "catalan_lowercase": ca_instructions.LowercaseLettersCatalanChecker,
    _SPECIAL_CHARACTER + "enie": ca_instructions.EnieChecker,
    _SPECIAL_CHARACTER + "tildes": ca_instructions.TildesChecker,
    _SPECIAL_CHARACTER + "dieresis": ca_instructions.DieresisChecker,
}

INSTRUCTION_DICT = {}
INSTRUCTION_DICT.update({"es:" + k: v for k, v in ES_INSTRUCTION_DICT.items()})
INSTRUCTION_DICT.update({"ca:" + k: v for k, v in CA_INSTRUCTION_DICT.items()})