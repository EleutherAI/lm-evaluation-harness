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

"""Library of instructions."""
import collections
import json
import random
import re
import string
from typing import Dict, Optional, Sequence, Union

from absl import logging
import langdetect

from lm_eval.tasks.m_ifeval.instruction_utils import fr_instructions_util


_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = fr_instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("moins de", "au moins")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
    "Oui.", "Non.", "Peut-être.")

# The options of starter keywords.
_STARTER_OPTIONS = ("Je dirais", "Ma réponse est", "Je crois",
                    "D'après moi", "Je pense", "Selon moi", 
                    "De mon point de vue", "En ce qui me concerne",
                    "À mon avis", "Il me semble", "Pour ma part",  "Alors",
                    "Il apparaît que", "Je trouve")

# The options of ending keywords.
# TODO(jeffreyzhou) add more ending options
_ENDING_OPTIONS = ("Je reste à votre disposition pour toute autre question.",
                   "D'autres questions ?", 
                   "Que puis-je faire pour vous ?",
                   "Que puis-je faire pour toi ?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
_SECTION_SPLITER = ("Section", "SECTION")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S", "P.-S.", "P-S", "PS", "p.-s.")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500

# The forbidden character.
_FORBIDDEN_CHAR = ("ç", "œ")


class Instruction:
  """An instruction template."""

  def __init__(self, instruction_id):
    self.id = instruction_id

  def build_description(self, **kwargs):
    raise NotImplementedError("`build_description` not implemented.")

  def get_instruction_args(self):
    raise NotImplementedError("`get_instruction_args` not implemented.")

  def get_instruction_args_keys(self):
    raise NotImplementedError("`get_instruction_args_keys` not implemented.")

  def check_following(self, value):
    raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
  """Check the language of the entire response."""

  def build_description(self, *, language = None):
    """Build the instruction description.

    Args:
      language: A string representing the expected language of the response. The
        language has to comply to the 97 types defined in
        `langid.py` (https://pypi.org/project/langid/1.1.5/), which follows
        ISO 639-1 codes (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes);
        for example, `en` for English, `zh` for Chinese, `fr` for French.

    Returns:
      A string representing the instruction description.
    """
    self._language = language
    if self._language is None:
      self._language = random.choice(list(_LANGUAGES.keys()))
    # TODO(tianjianlu): opens the description generation to more choices.
    self._description_pattern = (
        "L'INTEGRALITE de votre réponse doit être en {language}, aucune autre " +
        "langue n'est autorisée.")
    return self._description_pattern.format(language=_LANGUAGES[self._language])

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"language": self._language}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["language"]

  def check_following(self, value):
    """Check if the language of the entire response follows the instruction.

    Args:
      value: A string representing the response.

    Returns:
      True if the language of `value` follows instruction; otherwise False.
    """
    assert isinstance(value, str)

    try:
      return langdetect.detect(value) == self._language
    except langdetect.LangDetectException as e:
      # Count as instruction is followed.
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )  # refex: disable=pytotw.037
      return True


class NumberOfSentences(Instruction):
  """Check the number of sentences."""

  def build_description(self, *, num_sentences = None,
                        relation = None):
    """Build the instruction description.

    Args:
      num_sentences: An integer specifying the number of sentences as a
        threshold.
      relation: A string in (`moins de`, `au moins`), defining the relational
        operator for comparison.
        Two relational comparisons are supported for now:
        if 'moins de', the actual number of sentences < the threshold;
        if 'au moins', the actual number of sentences >= the threshold.

    Returns:
      A string representing the instruction description.
    """
    # The number of sentences as a threshold for comparison.
    self._num_sentences_threshold = num_sentences
    if (self._num_sentences_threshold is None or
        self._num_sentences_threshold < 0):
      self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Votre réponse doit contenir {relation} {num_sentences} phrases.")
    return self._description_pattern.format(
        relation=self._comparison_relation,
        num_sentences=self._num_sentences_threshold)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_sentences", "relation"]

  def check_following(self, value):
    """Check if the number of sentences follows the instruction.

    Args:
      value: A string representing the response.

    Returns:
      True if the response follows the instruction.

    Raise:
        ValueError if the string in `instruction_args` is not in
        [`moins de`, `au moins`].
    """
    num_sentences = fr_instructions_util.count_sentences(value)
    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return num_sentences < self._num_sentences_threshold
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return num_sentences >= self._num_sentences_threshold


class PlaceholderChecker(Instruction):
  """Check the placeholders in template writing."""

  def build_description(self, *, num_placeholders = None):
    """Build the instruction description.

    Args:
      num_placeholders: An integer denoting the minimum number of
        placeholders required in the response.

    Returns:
      A string representing the instruction description.
    """
    self._num_placeholders = num_placeholders
    if self._num_placeholders is None or self._num_placeholders < 0:
      self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
    self._description_pattern = (
        "La réponse doit contenir au moins {num_placeholders} espaces réservés, " + 
        "représentés par des crochets, comme [nom].")
    return self._description_pattern.format(
        num_placeholders=self._num_placeholders)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_placeholders": self._num_placeholders}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_placeholders"]

  def check_following(self, value):
    """Check if the number of placeholders follows the instruction.

    Args:
      value: A string representing the response.

    Returns:
      True if the actual number of placeholders in the response is greater than
      or equal to `num_placeholders`; otherwise, False.
    """
    placeholders = re.findall(r"\[.*?\]", value)
    num_placeholders = len(placeholders)
    return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
  """Checks the bullet list in the prompt."""

  def build_description(self, *, num_bullets = None):
    """Build the instruction description.

    Args:
      num_bullets: An integer specifying the exact number of bullet lists
        that is required to appear in the response.

    Returns:
      A string representing the instruction description.
    """
    self._num_bullets = num_bullets
    if self._num_bullets is None or self._num_bullets < 0:
      self._num_bullets = random.randint(1, _NUM_BULLETS)
    self._description_pattern = (
        "Votre réponse doit contenir exactement {num_bullets} puces." +
        "Utilisez les puces Markdown telles que :\n" +
        "* Ceci est le point 1. \n" +
        "* Ceci est le point 2")
    return self._description_pattern.format(
        num_bullets=self._num_bullets)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_bullets": self._num_bullets}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_bullets"]

  def check_following(self, value):
    r"""Check if the number of bullet lists meets the requirement.

    Args:
      value: A string representing the response. The response is expected to
        contain some bullet lists that start with `\*`.

    Returns:
      True if the actual number of bullet lists in the response meets the
      requirement.
    """
    bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
    bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
    num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
    return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):
  """Checks the constrained response."""

  def build_description(self):
    """Build the instruction description."""
    # A sequence of string(s) representing the options of the expected response.
    self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
    self._description_pattern = (
        "Répondez avec une des options suivantes: {response_options}")
    return self._description_pattern.format(
        response_options=self._constrained_responses)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response matches the constrained options.

    Args:
      value: A string representing the response.

    Returns:
      True if the actual response contains one of the options in the constrained
      responses; otherwise False.
    """
    value = value.strip()
    for constrained_response in self._constrained_responses:
      if constrained_response in value:
        return True
    return False


class ConstrainedStartChecker(Instruction):
  """Checks the response start."""

  def build_description(self, *, starter = None):
    """Build the instruction description.

    Args:
      starter: A string representing the keyward that the response should start
        with.

    Returns:
      A string representing the instruction description.
    """
    self._starter = starter.strip() if isinstance(starter, str) else starter
    if self._starter is None:
      self._starter = random.choice(_STARTER_OPTIONS)
    self._description_pattern = (
        "Pendant la conversation, quand c'est votre tour, " +
        "veuillez toujours commencer par {starter}")
    return self._description_pattern.format(starter=self._starter)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"starter": self._starter}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["starter"]

  def check_following(self, value):
    """Checks if the response starts with the constrained keyword or phrase.

    Args:
      value: A string representing the response.

    Returns:
      True if the response starts with the given phrase or keyword that is
      contained in `instruction_args`; otherwise, False.
    """
    response_pattern = r"^\s*" + self._starter + r".*$"
    response_with_constrained_start = re.search(response_pattern, value,
                                                flags=re.MULTILINE)
    return True if response_with_constrained_start else False


class HighlightSectionChecker(Instruction):
  """Checks the highlighted section."""

  def build_description(self, *, num_highlights = None):
    """Build the instruction description.

    Args:
      num_highlights: An integer specifying the minimum number of highlighted
        sections.

    Returns:
      A string representing the instruction description.
    """
    self._num_highlights = num_highlights
    if self._num_highlights is None or self._num_highlights < 0:
      self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

    self._description_pattern = (
        "Mettez en surbrillance au moins {num_highlights} sections de votre réponse avec " +
        "markdown, c'est-à-dire entre astérisque: *section en surbrillance*.")

    return self._description_pattern.format(num_highlights=self._num_highlights)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_highlights": self._num_highlights}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_highlights"]

  def check_following(self, value):
    """Checks if the number of highlighted sections meets the requirement.

    Args:
      value: a string repesenting the response. The response is expected to
        contain highlighted sections in the format of *highlighted*.

    Returns:
      True if the actual number of highlighted sections in the format of
      *highlighed sections* meets the minimum requirement; otherwise False.
    """
    num_highlights = 0
    highlights = re.findall(r"\*[^\n\*]*\*", value)
    double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
    for highlight in highlights:
      if highlight.strip("*").strip():
        num_highlights += 1
    for highlight in double_highlights:
      if highlight.removeprefix("**").removesuffix("**").strip():
        num_highlights += 1

    return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
  """Checks the sections."""

  def build_description(self, *, section_spliter = None,
                        num_sections = None):
    """Build the instruction description.

    Args:
      section_spliter: A string represents the section spliter keyword that
        marks a new section, i.e., `Section` or `SECTION`.
      num_sections: An integer specifying the number of sections.

    Returns:
      A string representing the instruction description.
    """
    self._section_spliter = section_spliter.strip() if isinstance(
        section_spliter, str) else section_spliter
    if self._section_spliter is None:
      self._section_spliter = random.choice(_SECTION_SPLITER)

    self._num_sections = num_sections
    if self._num_sections is None or self._num_sections < 0:
      self._num_sections = random.randint(1, _NUM_SECTIONS)

    self._description_pattern = (
        "Votre réponse doit comporter {num_sections} sections. Marquez le début " +
        "de chaque section avec {section_spliter} X, par exemple :\n" +
        "{section_spliter} 1\n" +
        "[contenu de la section 1]\n" +
        "{section_spliter} 2\n" +
        "[contenu de la section 2]")

    return self._description_pattern.format(
        num_sections=self._num_sections,
        section_spliter=self._section_spliter)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"section_spliter": self._section_spliter,
            "num_sections": self._num_sections}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["section_spliter", "num_sections"]

  def check_following(self, value):
    """Checks the response contains multiple sections.

    Args:
      value: A string representing the response. The response is expected
        to contain multiple sections (number of sections is greater than 1).
        A new section starts with `Section 1`, where the number denotes the
        section index.

    Returns:
      True if the number of sections in the response is greater than or equal to
      the minimum number of sections; otherwise, False.
    """
    section_splitter_patten = r"\s?" + self._section_spliter  + r"\s?\d+\s?"
    sections = re.split(section_splitter_patten, value)
    num_sections = len(sections) - 1
    return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
  """Checks the paragraphs."""

  def build_description(self, *, num_paragraphs = None):
    """Build the instruction description.

    Args:
      num_paragraphs: An integer specifying the number of paragraphs.

    Returns:
      A string representing the instruction description.
    """
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._description_pattern = (
        "Votre réponse doit contenir {num_paragraphs} paragraphes. " +
        "Les paragraphes doit être separés par le séparateur Markdown : ***")

    return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_paragraphs": self._num_paragraphs}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_paragraphs"]

  def check_following(self, value):
    """Checks the response contains required number of paragraphs.

    Args:
      value: A string representing the response. The response may contain
        paragraphs that are separated by the markdown divider: `***`.

    Returns:
      True if the actual number of paragraphs is the same as required;
      otherwise, False.
    """
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    num_paragraphs = len(paragraphs)

    for index, paragraph in enumerate(paragraphs):
      if not paragraph.strip():
        if index == 0 or index == len(paragraphs) - 1:
          num_paragraphs -= 1
        else:
          return False

    return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):
  """Checks the postscript."""

  def build_description(self, *, postscript_marker = None
                        ):
    """Build the instruction description.

    Args:
      postscript_marker: A string containing the keyword that marks the start
        of the postscript section.

    Returns:
      A string representing the instruction description.
    """
    self._postscript_marker = postscript_marker.strip() if isinstance(
        postscript_marker, str) else postscript_marker
    if self._postscript_marker is None:
      self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

    self._description_pattern = (
        "À la fin de votre réponse, veuillez ajouter explicitement un post-scriptum " +
        "commençant par {postscript}")

    return self._description_pattern.format(postscript=self._postscript_marker)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"postscript_marker": self._postscript_marker}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["postscript_marker"]

  def check_following(self, value):
    """Checks if the response follows the postscript format.

    Args:
      value: a string representing the response. The response is expected to
        contain a postscript section.

    Returns:
      True if the response contains a postscript section starting with
      the keyword containing in the `instruction_args`; otherwise False.
    """
    value = value.lower()
    if self._postscript_marker == "P.P.S":
      postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif self._postscript_marker == "P.S.":
      postscript_pattern = r"\s*p\.\s?s\..*$"
    else:
      postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
    postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
    return True if postscript else False


class RephraseChecker(Instruction):
  """Checks the repharse."""

  def build_description(self, *, original_message):
    """Build the instruction description.

    Args:
      original_message: A string representing the original message. The
        rephrased response should only change its words/sentences in between
        its two asterisks, for example, *change me*. Both original and rephrased
        messages should contain the changes in the form of *change me*.

    Returns:
      A string representing the instruction description.
    """
    if not self.is_change(original_message):
      raise ValueError(f"Message {original_message} does not contain changes "
                       "in the form of *change me*.")

    self._reference_without_change = original_message
    self._description = ("Reformulation : votre réponse reformulée ne doit que" +
                          "modifier les mots/phrases entre deux astérisques" +
                          "comme *changez ici*.")
    return self._description

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"original_message": self._reference_without_change}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["original_message"]

  def check_following(self, value):
    r"""Checks if the rephrasing follows the instruction.

    Args:
      value: A string representing the response, which is expected to rephras
        the string of `instruction_args`.

    Returns:
      True if `value` and `instruction_args` only differ by the words/sentences
      in between two asterisks such as *change me*; otherwise, False.
    """

    if not self.is_change(value):
      raise ValueError(f"value {value} does not contain "
                       "changes in the form of *change me*.")

    response_without_changes = self.strip_changes(value)
    reference_without_changes = self.strip_changes(
        self._reference_without_change)

    return response_without_changes == reference_without_changes

  def is_change(self, response):
    """Check if there is change in the response in the form of *change me*."""
    return re.search(r"\*.*\*", response)

  def strip_changes(self, response):
    """Strips off the changes."""
    return re.sub(r"\*.*\*", "", response)


class KeywordChecker(Instruction):
  """Check the exisitence of certain keywords."""

  def build_description(self, *, keywords = None
                        ):
    """Build the instruction description.

    Args:
      keywords: A sequence of strings representing the keywords that are
        expected in the response.

    Returns:
      A string representing the instruction description.
    """

    if not keywords:
      self._keywords = fr_instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._keywords = keywords
    self._keywords = sorted(self._keywords)

    self._description_pattern = ("Incluez les mots {keywords} dans la réponse.")

    return self._description_pattern.format(keywords=self._keywords)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"keywords": self._keywords}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["keywords"]

  def check_following(self, value):
    """Check if the response contain the expected keywords."""
    for keyword in self._keywords:
      if not re.search(keyword, value, flags=re.IGNORECASE):
        return False
    return True


class KeywordFrequencyChecker(Instruction):
  """Check the keyword frequency."""

  def build_description(self, *, keyword = None,
                        frequency = None,
                        relation = None):
    """Build the instruction description.

    Args:
      keyword: A string representing a keyword that is expected in the response.
      frequency: An integer specifying the number of times `keyword` is expected
        to appear in the response.
      relation: A string in (`less than`, `at least`), defining the relational
        operator for comparison.
        Two relational comparisons are supported for now:
        if 'less than', the actual number of occurrences < frequency;
        if 'at least', the actual number of occurrences >= frequency.

    Returns:
      A string representing the instruction description.
    """
    if not keyword:
      self._keyword = fr_instructions_util.generate_keywords(num_keywords=1)[0]
    else:
      self._keyword = keyword.strip()

    self._frequency = frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Votre réponse doit contenir le mot {keyword} {relation} " +
        "{frequency} fois.")

    return self._description_pattern.format(
        keyword=self._keyword,
        relation=self._comparison_relation,
        frequency=self._frequency)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["keyword", "frequency", "relation"]

  def check_following(self, value):
    """Checks if the response contain the keyword with required frequency."""
    actual_occurrences = len(re.findall(
        self._keyword, value, flags=re.IGNORECASE))

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return actual_occurrences < self._frequency
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return actual_occurrences >= self._frequency


class NumberOfWords(Instruction):
  """Checks the number of words."""

  def build_description(self, *, num_words = None,
                        relation = None):
    """Build the instruction description.

    Args:
      num_words: An integer specifying the number of words contained in the
        response.
      relation: A string in (`less than`, `at least`), defining the relational
        operator for comparison.
        Two relational comparisons are supported for now:
        if 'less than', the actual number of words < num_words;
        if 'at least', the actual number of words >= num_words.

    Returns:
      A string representing the instruction description.
    """

    self._num_words = num_words
    if self._num_words is None or self._num_words < 0:
      self._num_words = random.randint(
          _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
      )

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Répond avec {relation} {num_words} mots.")

    return self._description_pattern.format(
        relation=self._comparison_relation,
        num_words=self._num_words)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_words": self._num_words,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_words", "relation"]

  def check_following(self, value):
    """Checks if the response contains the expected number of words."""
    num_words = fr_instructions_util.count_words(value)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return num_words < self._num_words
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return num_words >= self._num_words


class JsonFormat(Instruction):
  """Check the Json format."""

  def build_description(self):
    self._description_pattern = (
        "L'intégralité de la réponse doit être sous format JSON. Vous pouvez utiliser des graduations de type « markdown » telles que ```."
    )
    return self._description_pattern

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    value = (
        value.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
      json.loads(value)
    except ValueError as _:
      return False
    return True


class ParagraphFirstWordCheck(Instruction):
  """Check the paragraph and the first word of the nth paragraph."""

  def build_description(self, num_paragraphs = None,
                        nth_paragraph = None,
                        first_word = None):
    r"""Build the instruction description.

    Args:
      num_paragraphs: An integer indicating the number of paragraphs expected
        in the response. A paragraph is a subset of the string that is
        expected to be separated by '\n\n'.
      nth_paragraph: An integer indicating the paragraph number that we look at.
        Note that n starts from 1.
      first_word: A string that represent the first word of the bth paragraph.

    Returns:
      A string representing the instruction description.
    """
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._nth_paragraph = nth_paragraph
    if (
        self._nth_paragraph is None
        or self._nth_paragraph <= 0
        or self._nth_paragraph > self._num_paragraphs
    ):
      self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

    self._first_word = first_word
    if self._first_word is None:
      self._first_word = fr_instructions_util.generate_keywords(num_keywords=1)[0]
    self._first_word = self._first_word.lower()

    self._description_pattern = (
      "Votre réponse doit contenir {num_paragraphs} paragraphes. " + 
      "Les paragraphes, et uniquement les paragraphes, doivent être séparés " +
      "les uns des autres par deux sauts de ligne comme s'il s'agissait de '\n\n' en Python. " +
      "Le paragraphe {nth_paragraph} doit commencer par le mot {first_word}.")

    return self._description_pattern.format(
        num_paragraphs=self._num_paragraphs,
        nth_paragraph=self._nth_paragraph,
        first_word=self._first_word)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_paragraphs", "nth_paragraph", "first_word"]

  def check_following(self, value):
    """Checks for required number of paragraphs and correct first word.

    Args:
      value: a string representing the response. The response may contain
        paragraphs that are separated by two new lines and the first word of
        the nth paragraph will have to match a specified word.

    Returns:
      True if the number of paragraphs is the same as required and the first
      word of the specified paragraph is the same as required. Otherwise, false.
    """

    paragraphs = re.split(r"\n\n", value)
    num_paragraphs = len(paragraphs)

    for paragraph in paragraphs:
      if not paragraph.strip():
        num_paragraphs -= 1

    # check that index doesn't go out of bounds
    if self._nth_paragraph <= num_paragraphs:
      paragraph = paragraphs[self._nth_paragraph - 1].strip()
      if not paragraph:
        return False
    else:
      return False

    first_word = ""
    punctuation = {".", ",", "?", "!", "'", '"'}

    # get first word and remove punctuation
    word = paragraph.split()[0].strip()
    # TODO(jeffrey): make more complex?
    word = word.lstrip("'")
    word = word.lstrip('"')

    for letter in word:
      if letter in punctuation:
        break
      first_word += letter.lower()

    return (
        num_paragraphs == self._num_paragraphs
        and first_word == self._first_word
    )


# TODO(jeffrey) add relation - at least/at most?
class KeySentenceChecker(Instruction):
  """Check the existence of certain key sentences."""

  def build_description(self, key_sentences = None,
                        num_sentences = None):
    """Build the instruction description.

    Args:
      key_sentences: A sequences of strings representing the key sentences that
        are expected in the response.
      num_sentences: The number of key sentences that are expected to be seen in
        the response.

    Returns:
      A string representing the instruction description.
    """

    if not key_sentences:
      # TODO(jeffrey) make a generate sentences function? wonderwords package
      self._key_sentences = set(["For now, this is fine."])
    else:
      self._key_sentences = key_sentences

    if not num_sentences:
      self._num_sentences = random.randint(1, len(self._key_sentences))
    else:
      self._num_sentences = num_sentences

    self._description_pattern = (
        "Incluez {num_sentences} des phrases suivantes {key_sentences}."
    )

    return self._description_pattern.format(
        num_sentences=self._num_sentences, key_sentences=self._key_sentences
    )

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_sentences": self._num_sentences,
            "key_sentences": list(self._key_sentences)}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_sentences", "key_sentences"]

  def check_following(self, value):
    """Checks if the response contains the expected key sentences."""
    count = 0
    sentences = fr_instructions_util.split_into_sentences(value)
    for sentence in self._key_sentences:
      if sentence in sentences:
        count += 1

    return count == self._num_sentences


class ForbiddenWords(Instruction):
  """Checks that specified words are not used in response."""

  def build_description(self, forbidden_words = None
                        ):
    """Build the instruction description.

    Args:
      forbidden_words: A sequences of strings representing words that are not
        allowed in the response.

    Returns:
      A string representing the instruction description.
    """

    if not forbidden_words:
      self._forbidden_words = fr_instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._forbidden_words = list(set(forbidden_words))
    self._forbidden_words = sorted(self._forbidden_words)
    self._description_pattern = (
        "N'incluez pas les mots {forbidden_words} dans votre réponse."
    )

    return self._description_pattern.format(
        forbidden_words=self._forbidden_words
    )

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"forbidden_words": self._forbidden_words}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["forbidden_words"]

  def check_following(self, value):
    """Check if the response does not contain the expected keywords."""
    for word in self._forbidden_words:
      if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
        return False
    return True


class RephraseParagraph(Instruction):
  """Checks that the paragraph is rephrased."""

  def build_description(self, *, original_paragraph, low, high
                        ):
    """Builds the instruction description.

    Args:
      original_paragraph: A string presenting the original paragraph. The
        rephrases response should have between low-high words in common.
      low: An integer presenting the lower bound of similar words.
      high: An integer representing the upper bound of similar words.

    Returns:
      A string representing the instruction description.
    """
    # TODO(jeffrey) make more encompassing
    self._original_paragraph = original_paragraph
    self._low = low
    self._high = high

    self._description = ("Reformulez le paragraphe suivant :" +
                         "{original_paragraph}\nVotre réponse doit contenir " +
                         "entre {low} et {high} des mêmes mots. " +
                         "Les mots sont considérés comme identiques si toutes les lettres, " +
                         "sans tenir compte des majuscules ou des minuscules, " +
                         "sont les mêmes. Par exemple, 'courir' est identique à 'Courir' mais différent de 'court'.")

    return self._description.format(original_paragraph=original_paragraph,
                                    low=self._low, high=self._high)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"original_paragraph": self._original_paragraph,
            "low": self._low,
            "high": self._high}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["original_paragraph", "low", "high"]

  def check_following(self, value):
    val_words = re.findall(r"\w+", value.lower())
    original_words = re.findall(r"\w+", self._original_paragraph.lower())
    similar_words = 0

    dict_val = collections.Counter(val_words)
    dict_original = collections.Counter(original_words)

    for word in dict_original:
      similar_words += min(dict_original[word], dict_val[word])

    return similar_words >= self._low and similar_words <= self._high


class TwoResponsesChecker(Instruction):
  """Check that two responses were given."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Donnez deux réponses différentes. Les réponses et uniquement les réponses doivent être séparées par 6 astérisques : ******"
    )
    return self._description_pattern

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response has two different answers.

    Args:
      value: A string representing the response.

    Returns:
      True if two responses are detected and false otherwise.
    """
    valid_responses = list()
    responses = value.split("******")
    for index, response in enumerate(responses):
      if not response.strip():
        if index != 0 and index != len(responses) - 1:
          return False
      else:
        valid_responses.append(response)
    return (
        len(valid_responses) == 2
        and valid_responses[0].strip() != valid_responses[1].strip()
    )


class RepeatPromptThenAnswer(Instruction):
  """Checks that Prompt is first repeated then answered."""

  def build_description(self, *, prompt_to_repeat = None):
    """Build the instruction description.

    Args:
      prompt_to_repeat: The prompt that is meant to be repeated.

    Returns:
      A string representing the instruction description.
    """
    if not prompt_to_repeat:
      raise ValueError("prompt_to_repeat must be set.")
    else:
      self._prompt_to_repeat = prompt_to_repeat
    self._description_pattern = (
        "Répétez d'abord la demande mot pour mot sans aucun changement,"
        " puis donnez votre réponse (1. n'incluez aucun mot ou caractère"
        " avant de répéter la demande; ; 2. la demande que vous devez répéter"
        " n'inclut pas cette phrase)."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return {"prompt_to_repeat": self._prompt_to_repeat}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["prompt_to_repeat"]

  def check_following(self, value):
    if value.strip().lower().startswith(self._prompt_to_repeat.strip().lower()):
      return True
    return False


class EndChecker(Instruction):
  """Checks that the prompt ends with a given phrase."""

  def build_description(self, *, end_phrase = None):
    """Build the instruction description.

    Args:
      end_phrase: A string representing the phrase the response should end with.

    Returns:
      A string representing the instruction description.
    """
    self._end_phrase = (
        end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
    )
    if self._end_phrase is None:
      self._end_phrase = random.choice(_ENDING_OPTIONS)
    self._description_pattern = (
        "Terminez votre réponse avec cette phrase mot pour mot: {ender}. "
        "Aucune autre phrase ne doit suivre celle-ci.")
    return self._description_pattern.format(ender=self._end_phrase)

  def get_instruction_args(self):
    return {"end_phrase": self._end_phrase}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["end_phrase"]

  def check_following(self, value):
    """Checks if the response ends with the expected phrase."""
    value = value.strip().strip("\"").lower()
    self._end_phrase = self._end_phrase.strip().lower()
    return value.endswith(self._end_phrase)

# Modified - French version
class TitleChecker(Instruction):
  """Checks the response for a title."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Votre réponse doit contenir un titre, encadré par des doubles dièses,"
        " comme ##hymne à la joie##."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return {}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response contains a title."""
    pattern = r"##[^\n]+##"
    re_pattern = re.compile(pattern)
    titles = re.findall(re_pattern, value)

    for title in titles:
      if title.lstrip("#").rstrip("#").strip():
        return True
    return False


class LetterFrequencyChecker(Instruction):
  """Checks letter frequency."""

  def build_description(self, *, letter = None,
                        let_frequency = None,
                        let_relation = None):
    """Build the instruction description.

    Args:
      letter: A string representing a letter that is expected in the response.
      let_frequency: An integer specifying the number of times `keyword` is
        expected to appear in the response.
      let_relation: A string in (`less than`, `at least`), defining the
        relational operator for comparison. Two relational comparisons are
        supported for now; if 'less than', the actual number of
        occurrences < frequency; if 'at least', the actual number of
        occurrences >= frequency.

    Returns:
      A string representing the instruction description.
    """
    if (
        not letter
        or len(letter) > 1
        or ord(letter.lower()) < 97
        or ord(letter.lower()) > 122
    ):
      self._letter = random.choice(list(string.ascii_letters))
    else:
      self._letter = letter.strip()
    self._letter = self._letter.lower()

    self._frequency = let_frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _LETTER_FREQUENCY)

    if let_relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif let_relation not in _COMPARISON_RELATION:
      raise ValueError(
          "The supported relation for comparison must be in "
          f"{_COMPARISON_RELATION}, but {let_relation} is given."
      )
    else:
      self._comparison_relation = let_relation

    self._description_pattern = (
        "Dans votre réponse, la lettre {letter} doit apparaître {let_relation}"
        " {let_frequency} fois."
    )

    return self._description_pattern.format(
        letter=self._letter,
        let_frequency=self._frequency,
        let_relation=self._comparison_relation,
    )

  def get_instruction_args(self):
    """Returns the keyword args of build description."""
    return {"letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["letter", "let_frequency", "let_relation"]

  def check_following(self, value):
    """Checks that the response contains the letter at the right frequency."""
    value = value.lower()
    letters = collections.Counter(value)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return letters[self._letter] < self._frequency
    else:
      return letters[self._letter] >= self._frequency


class CapitalLettersFrenchChecker(Instruction):
  """Checks that the response is in french and is in all capital letters."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Votre réponse doit être entièrement en français et en majuscules. Les minuscules ne sont pas autorisées."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return {}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks that the response is in French and in all capital letters."""
    assert isinstance(value, str)

    try:
      return value.isupper() and langdetect.detect(value.lower()) == "fr" # put text in lowercase for langdetect, otherwise detection outputs german for French Caps text
    except langdetect.LangDetectException as e:
      # Count as instruction is followed.
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )  # refex: disable=pytotw.037
      return True


class LowercaseLettersFrenchChecker(Instruction):
  """Checks that the response is in french and is in all lowercase letters."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Votre réponse doit être entièrement rédigée en français et en minuscules. Les majuscules ne sont pas autorisées."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return {}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks that the response is in French and in all lowercase letters."""
    assert isinstance(value, str)

    try:
      return value.islower() and langdetect.detect(value) == "fr"
    except langdetect.LangDetectException as e:
      # Count as instruction is followed.
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )  # refex: disable=pytotw.037
      return True


class CommaChecker(Instruction):
  """Checks the response for no commas."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "N'utilisez aucune virgule dans votre réponse."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return {}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks that the response does not contain commas."""
    return not re.search(r"\,", value)


class CapitalWordFrequencyChecker(Instruction):
  """Checks frequency of words with all capital letters."""

  def build_description(
      self,
      capital_frequency = None,
      capital_relation = None,
  ):
    """Build the instruction description.

    Args:
      capital_frequency: An integer that represents the number of words that
        should be in all capital letters.
      capital_relation: A string that is 'at least' or 'at most' that refers to
        the frequency.

    Returns:
      A string representing the instruction description.
    """
    self._frequency = capital_frequency
    if self._frequency is None:
      self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

    self._comparison_relation = capital_relation
    if capital_relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif capital_relation not in _COMPARISON_RELATION:
      raise ValueError(
          "The supported relation for comparison must be in "
          f"{_COMPARISON_RELATION}, but {capital_relation} is given."
      )

    self._description_pattern = (
        "Dans votre réponse, des mots entièrement en lettres majuscules doivent apparaître {relation} {frequency} fois."
    )

    return self._description_pattern.format(
        frequency=self._frequency, relation=self._comparison_relation
    )

  def get_instruction_args(self):
    """Returns the keyword args of build description."""
    return {
        "capital_frequency": self._frequency,
        "capital_relation": self._comparison_relation,
    }

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["capital_frequency", "capital_relation"]

  def check_following(self, value):
    """Checks the frequency of words with all capital letters."""
    # Hyphenated words will count as one word
    words = fr_instructions_util.nltk.word_tokenize(value)
    capital_words = [word for word in words if word.isupper()]

    capital_words = len(capital_words)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return capital_words < self._frequency
    else:
      return capital_words >= self._frequency

# Modified - French version
class QuotationChecker(Instruction):
  """Checks if response is wrapped with either standard double quotation marks 
    or guillemets (French quotation marks)."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Mettez l’ensemble de votre réponse entre guillemets."
    )
    return self._description_pattern

  def get_instruction_args(self):
    """Returns the keyword args of build description."""
    return {}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response is wrapped with quotation marks."""
    value = value.strip()
    return len(value)>1 and ((value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")) or (value.startswith('«') and value.endswith('»')))


# ------------------ French-specific Instructions ------------------

class ForbiddenChar(Instruction):
  """Checks that specified chars are not used in response."""

  def build_description(self, forbidden_char = None
                        ):
    """Build the instruction description.

    Args:
      forbidden_char: A string respresenting the character that is not
        allowed in the response.

    Returns:
      A string representing the instruction description.
    """

    if not forbidden_char:
      self._forbidden_char = random.choice(_FORBIDDEN_CHAR)
    else:
      self._forbidden_char = forbidden_char

    self._description_pattern = (
        "N'incluez pas le caractère {forbidden_char} dans votre réponse."
    )

    return self._description_pattern.format(
        forbidden_char=self._forbidden_char
    )

  def get_instruction_args(self):
    """Returns the character args of `build_description`."""
    return {"forbidden_char": self._forbidden_char}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["forbidden_char"]

  def check_following(self, value):
    """Check if the response does not contain the expected character."""
    return not self.contains_char(value)
  
  def contains_char(self, value):
    """
    Check if a specific character is present in a string, ignoring case.

    Args:
      value (str): The string in which to search for the character.

    Returns:
      (bool): True if the character is present in the string, False otherwise.

    Example:
    >>> contains_char("Hello, World!", "w")
    True
    >>> contains_char("Hello, World!", "z")
    False
    """
    # Create a regex pattern to match the character, case insensitive
    pattern = re.compile(re.escape(self._forbidden_char), flags=re.IGNORECASE)
    # Search for the pattern in the string
    return pattern.search(value) is not None
  
# class ExcludeFormalNegation(Instruction):
#   """Checks that 'ne' and 'n'' are not used in the response."""

#   _FORBIDDEN_WORDS = ["ne", "n'"]

#   def build_description(self):
#       """Build the instruction description.

#       Returns:
#         A string representing the instruction description.
#       """
#       self._description_pattern = ("Utilisez la forme informelle de la négation dans votre réponse. Excluez l'emploi de \"ne\" ou \"n'\".")
#       return self._description_pattern

#   def get_instruction_args(self):
#       """Returns the keyword args of build description."""
#       return {}

#   def get_instruction_args_keys(self):
#       """Returns the args keys of `build_description`."""
#       return []

#   def check_following(self, value):
#       """Check if the response does not contain 'ne' or 'n''."""
#       return not any(self.contains_word(value, word) for word in self._FORBIDDEN_WORDS)

#   def contains_word(self, value, word):
#       """
#       Check if a specific word is present in a string, ignoring case.

#       Args:
#         value (str): The string in which to search for the word.
#         word (str): The word to search for in the string.

#       Returns:
#         (bool): True if the word is present in the string, False otherwise.
#       """
#       # Create a regex pattern to match the word, case insensitive
#       pattern = re.compile(r'\b' + re.escape(word) + r'\b', flags=re.IGNORECASE)
#       # Search for the pattern in the string
#       return pattern.search(value) is not None

class UseInformalAddress(Instruction):
    """Checks that the response uses informal address, 'tutoiement' in French."""

    _TU_INDICATORS = [
            r'\btu\b',           # "tu" pronoun
            r'\bte\b',           # "te" pronoun
            r"\bt'\b",           # "t'" pronoun
            r'\btoi\b',          # "toi" pronoun
            r'\bton\b',          # possessive adjective
            r'\bta\b',           # possessive adjective
            r'\btes\b',          # possessive adjective
        ]
    
    #_VOUS_INDICATORS = [
    #        r'\bvous\b',          # "vous" pronoun
    #        r'\bvotre\b',         # possessive adjective
    #        r'\bvos\b',           # possessive adjective
    #    ]

    def build_description(self):
        """Build the instruction description.

        Returns:
          A string representing the instruction description.
        """
        self._description_pattern = ("Parlez directement à votre interlocuteur dans votre réponse et utilisez le tutoiement.")
        return self._description_pattern

    def get_instruction_args(self):
        """Returns an empty dict as there are no variable args for this instruction."""
        return {}

    def get_instruction_args_keys(self):
        """Returns an empty list as there are no variable args for this instruction."""
        return []

    def check_following(self, value):
        """Check if the response uses the informal address: 'tutoiement' form."""
        # Look for common 'tutoiement' form indicators
        pattern = '|'.join(self._TU_INDICATORS)
        return bool(re.search(pattern, value, re.IGNORECASE))
      
class NoAccents(Instruction):
    """Checks that the response does not use any accented characters."""

    def build_description(self):
        """Build the instruction description.

        Returns:
          A string representing the instruction description.
        """
        self._description_pattern = ("Ne faites pas usage d'accents dans votre réponse.")
        return self._description_pattern

    def get_instruction_args(self):
        """Returns an empty dict as there are no variable args for this instruction."""
        return {}

    def get_instruction_args_keys(self):
        """Returns an empty list as there are no variable args for this instruction."""
        return []

    def check_following(self, value):
        """Check if the response does not use any accented characters."""
        return not self.contains_accents(value)

    def contains_accents(self, value):
        """
        Check if a string contains any accented characters.

        Args:
          value (str): The string to check for accented characters.

        Returns:
          (bool): True if accented characters are present, False otherwise.
        """
        accented_chars = re.compile(r'[àáâãäåçèéêëìíîïñòóôõöùúûüýÿÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ]')
        return bool(accented_chars.search(value))

class AccentsChecker(Instruction):
    """Checks that the response includes appropriate accents for target French words."""
    
    def build_description(self, word_to_accentuate):
        """Build the instruction description.

        Args:
          word_to_accentuate (dict): A dictionary where the keys are French words with missing or incorrect accents,
                             and the values are the correctly accented forms of those words.
            Example:
                {
                    'lecons': 'leçons',
                    'evenements': 'événements',
                    'celebrites': 'célébrités',
                    'generations': 'générations'
                } 
        Returns:
          A string representing the instruction description.
        """
        self._target_words = word_to_accentuate
        self._description_pattern = (f"Réécris ce texte en ajoutant les accents.")
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {"word_to_accentuate": self._target_words}

    def get_instruction_args_keys(self):
        """Returns an empty list as there are no variable args for this instruction."""
        return ["word_to_accentuate"]

    def check_following(self, value):
        """Check if the response does not use any accented characters."""
        return self.check_accents(value)

    def check_accents(self, value):
        """
        Check if accents are correctly placed in the given French text.

        This function compares each word in the input text against a dictionary
        of French words with their correct accents. It identifies words that
        are missing required accents or have unnecessary accents.

        Args:
          value (str): The French text to check for accent errors.

        Returns:
          (bool): True if accents are correctly placed, False otherwise.
        """
        words = re.findall(r'\b\w+\b', value.lower())
        for word in words:
          unaccented = fr_instructions_util.remove_accents(word)
          if unaccented in self._target_words:
              correct = self._target_words[unaccented]
              if word != correct:
                return False
        return True

class NumbersInWords(Instruction):
    """Checks that no Arabic numerals are used in the response."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = "N'utilisez pas de chiffres arabes dans votre réponse."
        return self._description_pattern

    def get_instruction_args(self):
        """Returns an empty dict as there are no variable args for this instruction."""
        return {}

    def get_instruction_args_keys(self):
        """Returns an empty list as there are no variable args for this instruction."""
        return []

    def check_following(self, value):
        """Check if no Arabic numerals are used in the response.
        Returns:
        bool: True if no Arabic or Roman numerals are found, False otherwise."""
        return not bool(re.search(r'\d', value))