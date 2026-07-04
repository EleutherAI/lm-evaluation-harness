import re
import textdistance
import unicodedata
from unidecode import unidecode
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter
from abc import ABC, abstractmethod
from typing import Iterable, List, Union

@register_filter("custom")
class CustomFilter(Filter):
    """
    Custom filter that applies a custom, user-defined function to the model responses.
    """

    def __init__(self, **kwargs) -> None:
        self.filter_fn = kwargs.pop("filter_fn")

        super().__init__(**kwargs)

    def apply(self, resps, docs):
        return self.filter_fn(resps, docs)

@register_filter("find_choices")
class ChoicesFilter(Filter):
    def __init__(self, choices=None, fallback="[invalid]", regex_patterns=None):
        if choices is None:
            choices = ["A", "B", "C", "D", "E"]
        self.choices = set(choices)
        self.fallback = fallback
        self.regex_patterns = [re.compile(p) for p in (regex_patterns or [])]

    def _extract_choice(self, text: str) -> str:
        if not isinstance(text, str):
            return self.fallback

        text = text.strip()

        if text in self.choices:
            return text

        for regex in self.regex_patterns:
            match = regex.search(text)
            if match:
                value = match.group(1).strip()
                if value in self.choices:
                    return value

        return self.fallback

    def apply(self, resps, docs):
        output = []
        for inst in resps:
            current = []
            for resp in inst:
                if isinstance(resp, tuple):
                    resp = resp[0]
                current.append(self._extract_choice(resp))
            output.append(current)
        return output

    def process_resp(self, text):
        text = text.strip()

        if text in self.choices:
            return text

        for regex in self.regex_patterns:
            match = re.search(regex, text)
            if match:
                return match.group(1)

        return self.fallback

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.process_resp(resp) for resp in inst]

        return [filter_set(resp) for resp in resps]

@register_filter("find_choices_thinking_case")
class NewChoicesFilter(Filter):
    def __init__(self, choices=None, fallback="[invalid]", regex_patterns=None):
        if choices is None:
            choices = ["A", "B", "C", "D", "E"]

        self.choices = set(choices)
        self.fallback = fallback
        self.regex_patterns = regex_patterns or []

        choices_group = "".join(choices)

        # Resposta isolada: A, A., A), (A), "A"
        self.isolated_choice = re.compile(
            rf"""^\s*["'`\(\[]?\s*([{choices_group}])\s*["'`\)\]\.]?\s*$"""
        )

        self.leading_choice = re.compile(
            rf"""^\s*["'`\(\[]?\s*([{choices_group}])\s*["'`\)\]]?\s*[\.\):\-]\s+.+""",
            re.DOTALL,
        )

        self.leading_explicit_choice = re.compile(
            rf"""
            ^\s*
            (?:
                resposta(?:\s+correta)?
                | alternativa
                | letra
                | opcao
                | opção
            )
            \s*
            (?:
                correta
                | final
            )?
            \s*
            (?:
                é
                | e
                | :
                | -
            )?
            \s*
            ["'`\(\[]?
            ([{choices_group}])
            ["'`\)\]]?
            \b
            """,
            re.IGNORECASE | re.VERBOSE,
        )

    def _clean(self, text):
        if isinstance(text, tuple):
            text = text[0]

        if not isinstance(text, str):
            return ""

        text = text.replace("<|im_end|>", "")
        text = text.replace("</s>", "")
        return text.strip()

    def _split_think(self, text):
        if "</think>" in text:
            return "after_think", text.rsplit("</think>", 1)[1].strip()

        if "<think>" in text:
            return "unfinished_think", ""

        return "no_think", text.strip()

    def _first_non_empty_line(self, text):
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return ""

    def _last_non_empty_line(self, text):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _extract_from_after_think(self, text):
        if not text:
            return self.fallback

        first_line = self._first_non_empty_line(text)

        match = self.isolated_choice.match(first_line)
        if match:
            return match.group(1)

        match = self.leading_choice.match(first_line)
        if match:
            return match.group(1)

        match = self.leading_explicit_choice.match(first_line)
        if match:
            return match.group(1).upper()

        return self.fallback

    def _extract_from_no_think(self, text):
        if not text:
            return self.fallback

        last_line = self._last_non_empty_line(text)
        match = self.isolated_choice.match(last_line)
        if match:
            return match.group(1)

        return self.fallback

    def process_resp(self, text):
        text = self._clean(text)
        mode, candidate = self._split_think(text)

        if mode == "unfinished_think":
            return self.fallback

        if mode == "after_think":
            return self._extract_from_after_think(candidate)

        return self._extract_from_no_think(candidate)

    def apply(self, resps, docs=None):
        return [
            [self.process_resp(resp) for resp in inst]
            for inst in resps
        ]

@register_filter("find_similar_label")
class SimilarLabelFilter(Filter):
    def __init__(
        self,
        labels,
        fallback="[invalid]"
    ) -> None:
        self.labels = labels
        self.fallback = fallback

    def process_resp(self, prediction):
        norm_label = [unidecode(s.strip().lower()) for s in self.labels]
        prediction = unidecode(prediction.strip().lower())

        if prediction in norm_label:
            return self.labels[norm_label.index(prediction)]

        if prediction == "":
            return self.fallback

        count_matches = 0
        last_match = self.fallback
        for label in norm_label:
            if label in prediction:
                count_matches += 1
                last_match = label
        if count_matches == 1:
            return self.labels[norm_label.index(last_match)]

        get_text_until = [".", ",", ";", ":", "(", ")", "[", "]", "\n"]
        for split_char in get_text_until:
            if split_char in prediction:
                prediction = prediction[:prediction.find(split_char)]

        max_length = max(len(s) for s in norm_label)
        prediction = prediction[:max_length]

        similarities = [
            textdistance.levenshtein.normalized_similarity(prediction, label)
            for label in norm_label
        ]

        if max(similarities) < 0.5:
            prediction = self.fallback
        else:
            prediction = self.labels[similarities.index(max(similarities))]

        return prediction

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.process_resp(resp) for resp in inst]

        return [filter_set(resp) for resp in resps]


def unidecode(text):
    text = str(text)
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )


class Filter(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def apply(self, resps: Union[List, Iterable], docs: List[dict]) -> Iterable:
        return resps

@register_filter("find_yes_no_with_think_filter")
class FindYesNoWithThinkFilter(Filter):
    def __init__(self, labels=None, fallback="[invalid]") -> None:
        if labels is None:
            labels = ["Sim", "Não"]

        self.labels = labels
        self.fallback = fallback

        self.norm_to_label = {
            self._normalize_label(label): label
            for label in labels
        }

        label_group = "|".join(
            re.escape(label)
            for label in self.norm_to_label.keys()
        )

        self.isolated_label = re.compile(
            rf"""^\s*["'`\(\[]?\s*({label_group})\s*["'`\)\]\.!?]?\s*$""",
            re.IGNORECASE,
        )

        self.leading_label = re.compile(
            rf"""^\s*["'`\(\[]?\s*({label_group})\s*["'`\)\]]?\s*[\.,:;\-–—]\s+.+""",
            re.IGNORECASE | re.DOTALL,
        )

        self.explicit_label = re.compile(
            rf"""
            ^\s*
            (?:
                resposta(?:\s+correta)?
                | classificacao
                | classe
                | label
                | rotulo
            )
            \s*
            (?:
                correta
                | final
            )?
            \s*
            (?:
                e
                | :
                | -
            )?
            \s*
            ["'`\(\[]?
            ({label_group})
            ["'`\)\]]?
            \b
            """,
            re.IGNORECASE | re.VERBOSE,
        )

    def _normalize_label(self, text):
        return unidecode(str(text).strip().lower())

    def _normalize_text(self, text):
        return unidecode(str(text).strip().lower())

    def _clean(self, text):
        if isinstance(text, tuple):
            text = text[0]

        if not isinstance(text, str):
            return ""

        text = text.replace("<|im_end|>", "")
        text = text.replace("</s>", "")
        return text.strip()

    def _split_think(self, text):
        lowered = text.lower()

        if "</think>" in lowered:
            idx = lowered.rfind("</think>") + len("</think>")
            return "after_think", text[idx:].strip()

        if "<think>" in lowered:
            return "unfinished_think", ""

        return "no_think", text.strip()

    def _first_non_empty_line(self, text):
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return ""

    def _last_non_empty_line(self, text):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _to_original_label(self, normalized_label):
        return self.norm_to_label.get(
            self._normalize_label(normalized_label),
            self.fallback,
        )

    def _match_label(self, text, allow_leading_explanation=False):
        if not text:
            return self.fallback

        normalized = self._normalize_text(text)

        match = self.isolated_label.match(normalized)
        if match:
            return self._to_original_label(match.group(1))

        match = self.explicit_label.match(normalized)
        if match:
            return self._to_original_label(match.group(1))

        if allow_leading_explanation:
            match = self.leading_label.match(normalized)
            if match:
                return self._to_original_label(match.group(1))

        return self.fallback

    def _extract_from_after_think(self, text):
        first_line = self._first_non_empty_line(text)

        return self._match_label(
            first_line,
            allow_leading_explanation=True,
        )

    def _extract_from_no_think(self, text):
        result = self._match_label(
            text,
            allow_leading_explanation=False,
        )

        if result != self.fallback:
            return result

        last_line = self._last_non_empty_line(text)

        return self._match_label(
            last_line,
            allow_leading_explanation=False,
        )

    def process_resp(self, prediction):
        prediction = self._clean(prediction)
        mode, candidate = self._split_think(prediction)

        if mode == "unfinished_think":
            return self.fallback

        if mode == "after_think":
            return self._extract_from_after_think(candidate)

        return self._extract_from_no_think(candidate)

    def apply(self, resps, docs=None):
        return [
            [self.process_resp(resp) for resp in inst]
            for inst in resps
        ]