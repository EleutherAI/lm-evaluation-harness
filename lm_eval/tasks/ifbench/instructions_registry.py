# Copyright 2025 Allen Institute for AI.
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

"""Registry of all insts."""

from lm_eval.tasks.ifbench import (
    instructions as insts,
    instructions_ifeval as insts_ifeval,
)


INSTRUCTION_DICT = {
    "count:word_count_range": insts.WordCountRangeChecker,
    "count:unique_word_count": insts.UniqueWordCountChecker,
    "ratio:stop_words": insts.StopWordPercentageChecker,
    "ratio:sentence_type": insts.SentTypeRatioChecker,
    "ratio:sentence_balance": insts.SentBalanceChecker,
    "count:conjunctions": insts.ConjunctionCountChecker,
    "count:person_names": insts.PersonNameCountChecker,
    "ratio:overlap": insts.NGramOverlapChecker,
    "count:numbers": insts.NumbersCountChecker,
    "words:alphabet": insts.AlphabetLoopChecker,
    "words:vowel": insts.SingleVowelParagraphChecker,
    "words:consonants": insts.ConsonantClusterChecker,
    "sentence:alliteration_increment": insts.IncrementingAlliterationChecker,
    "words:palindrome": insts.PalindromeChecker,
    "count:punctuation": insts.PunctuationCoverChecker,
    "format:parentheses": insts.NestedParenthesesChecker,
    "format:quotes": insts.NestedQuotesChecker,
    "words:primelength_constraints:s": insts.PrimeLengthsChecker,
    "format:options": insts.OptionsResponseChecker,
    "format:newline": insts.NewLineWordsChecker,
    "format:emoji": insts.EmojiSentenceChecker,
    "ratio:sentence_words": insts.CharacterCountUniqueWordsChecker,
    "count:words_japanese": insts.NthWordJapaneseChecker,
    "words:start_verb": insts.StartWithVerbChecker,
    "words:repeats": insts.LimitedWordRepeatChecker,
    "sentence:keyword": insts.IncludeKeywordChecker,
    "count:pronouns": insts.PronounCountChecker,
    "words:odd_even_syllables": insts.AlternateParitySyllablesChecker,
    "words:last_first": insts.LastWordFirstNextChecker,
    "words:paragraph_last_first": insts.ParagraphLastFirstWordMatchChecker,
    "sentence:increment": insts.IncrementingWordCountChecker,
    "words:no_consecutive": insts.NoConsecutiveFirstLetterChecker,
    "format:line_indent": insts.IndentStairsChecker,
    "format:quote_unquote": insts.QuoteExplanationChecker,
    "format:list": insts.SpecialBulletPointsChecker,
    "format:thesis": insts.ItalicsThesisChecker,
    "format:sub-bullets": insts.SubBulletPointsChecker,
    "format:no_bullets_bullets": insts.SomeBulletPointsChecker,
    "custom:multiples": insts.PrintMultiplesChecker,
    "custom:mcq_countlength_constraints:": insts.MultipleChoiceQuestionsChecker,
    "custom:reverse_newline": insts.ReverseNewlineChecker,
    "custom:word_reverse": insts.WordReverseOrderChecker,
    "custom:character_reverse": insts.CharacterReverseOrderChecker,
    "custom:sentence_alphabet": insts.SentenceAlphabetChecker,
    "custom:european_capitals_sort": insts.EuropeanCapitalsSortChecker,
    "custom:csv_city": insts.CityCSVChecker,
    "custom:csv_special_character": insts.SpecialCharacterCSVChecker,
    "custom:csv_quotes": insts.QuotesCSVChecker,
    "custom:datedetectable_format:_list": insts.DateFormatListChecker,
    "count:keywords_multiple": insts.KeywordsMultipleChecker,
    "words:keywords_specific_position": insts.KeywordSpecificPositionChecker,
    "words:words_position": insts.WordsPositionChecker,
    "repeat:repeat_change": insts.RepeatChangeChecker,
    "repeat:repeat_simple": insts.RepeatSimpleChecker,
    "repeat:repeat_span": insts.RepeatSpanChecker,
    "format:title_case": insts.TitleCaseChecker,
    "format:output_template": insts.OutputTemplateChecker,
    "format:no_whitespace": insts.NoWhitespaceChecker,
    # ---------------------------------------------------------- IFEval Checkers
    "keywords:existence": insts_ifeval.KeywordChecker,
    "keywords:frequency": insts_ifeval.KeywordFrequencyChecker,
    "keywords:forbidden_words": insts_ifeval.ForbiddenWords,
    "keywords:letter_frequency": insts_ifeval.LetterFrequencyChecker,
    "language:response_language": insts_ifeval.ResponseLanguageChecker,
    "length_constraints:number_sentences": insts_ifeval.NumberOfSentences,
    "length_constraints:number_paragraphs": insts_ifeval.ParagraphChecker,
    "length_constraints:number_words": insts_ifeval.NumberOfWords,
    "length_constraints:nth_paragraph_first_word": insts_ifeval.ParagraphFirstWordCheck,
    "detectable_content:number_placeholders": insts_ifeval.PlaceholderChecker,
    "detectable_content:postscript": insts_ifeval.PostscriptChecker,
    "detectable_format:number_bullet_lists": insts_ifeval.BulletListChecker,
    "detectable_format:constrained_response": insts_ifeval.ConstrainedResponseChecker,
    "detectable_format:number_highlighted_sections": (
        insts_ifeval.HighlightSectionChecker
    ),
    "detectable_format:multiple_sections": insts_ifeval.SectionChecker,
    "detectable_format:jsondetectable_format:": insts_ifeval.JsonFormat,
    "detectable_format:title": insts_ifeval.TitleChecker,
    "combination:two_responses": insts_ifeval.TwoResponsesChecker,
    "combination:repeat_prompt": insts_ifeval.RepeatPromptThenAnswer,
    "startend:end_checker": insts_ifeval.EndChecker,
    "change_case:capital_word_frequency": insts_ifeval.CapitalWordFrequencyChecker,
    "change_case:english_capital": insts_ifeval.CapitalLettersEnglishChecker,
    "change_case:english_lowercase": insts_ifeval.LowercaseLettersEnglishChecker,
    "punctuation:no_comma": insts_ifeval.CommaChecker,
    "startend:quotation": insts_ifeval.QuotationChecker,
}
