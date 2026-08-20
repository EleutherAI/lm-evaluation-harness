"""Prompting, answer extraction, and metrics for SuperGPQA.

The prompt and extraction behavior are adapted from SuperGPQA/SuperGPQA at
revision e31e8bf3e7089aef808be5ddecafd4538dee6e41.
"""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import fmean
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


LETTERS = "ABCDEFGHIJ"
ZERO_SHOT_INSTRUCTION = (
    "Answer the following multiple choice question. There is only one correct "
    "answer. The last line of your response should be in the format 'Answer: "
    "$LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, "
    "H, I, or J.\n\n"
)

# The five demonstrations are the fixed examples in the official five-shot
# prompt. Braces remain doubled because the source applies str.format once.
FIVE_SHOT_TEMPLATE = r"""Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.

Question:<OLMES_SPACE>
A refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is
A) 10
B) 40
C) 6
D) 25
E) 15
F) 50
G) 30
H) 4
I) 5
J) 20

Answer: Let's think step by step. In a refracting telescope, if both lenses are converging, the focus of both lenses must be between the two lenses, and thus the focal lengths of the two lenses must add up to their separation. Since the focal length of one lens is 20 cm, the focal length of the other must be 80 cm. The magnification is the ratio of these two focal lengths, or 4.
Answer: H.

Question:<OLMES_SPACE>
Say the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?<OLMES_SPACE>
A) 1000 times more
B) 50 times more
C) 5000 times more
D) 500 times more
E) 10000 times more
F) 20000 times more
G) 2000 times more
H) 100 times more
I) 10 times more
J) N/A

Answer: Let's think step by step. The amount of light a telescope can gather compared to the human eye is proportional to the area of its apertures. The area of a circle is given by the formula $A = \pi \left(\frac{{D}}{{2}}\right)^2$, where $D$ is the diameter. Therefore, the relative light-gathering power is calculated as:
\[
\frac{{\left(\frac{{50 \text{{ cm}}}}{{2}}\right)^2}}{{\left(\frac{{5 \text{{ mm}}}}{{2}}\right)^2}} = \frac{{\left(\frac{{50 \text{{ cm}}}}{{0.1 \text{{ cm}}}}\right)^2}}{{\left(\frac{{5 \text{{ mm}}}}{{0.1 \text{{ cm}}}}\right)^2}} = \frac{{500^2}}{{5^2}} = 10000.
\]
Answer: E.

Question:<OLMES_SPACE>
Where do most short-period comets come from and how do we know?<OLMES_SPACE>
A) The Kuiper belt; short period comets tend to be in the plane of the solar system like the Kuiper belt.<OLMES_SPACE>
B) The asteroid belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the asteroid belt.<OLMES_SPACE>
C) The asteroid belt; short period comets tend to be in the plane of the solar system just like the asteroid belt.<OLMES_SPACE>
D) The Oort cloud; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the Oort cloud.<OLMES_SPACE>
E) The Oort Cloud; short period comets tend to come from random directions indicating a spherical distribution of comets called the Oort Cloud.<OLMES_SPACE>
F) The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.<OLMES_SPACE>
G) The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.<OLMES_SPACE>
Answer: Let's think step by step. Most short-period comets originate from the Kuiper belt. This is deduced from the observation that these comets tend to follow orbits that lie in the plane of the solar system, similar to the distribution of objects in the Kuiper belt itself. Thus, the alignment of these cometary orbits with the ecliptic plane points to their Kuiper belt origin.
Answer: A.

Question:<OLMES_SPACE>
Colors in a soap bubble result from light<OLMES_SPACE>
A) dispersion<OLMES_SPACE>
B) deflection<OLMES_SPACE>
C) refraction<OLMES_SPACE>
D) reflection<OLMES_SPACE>
E) interference<OLMES_SPACE>
F) converted to a different frequency<OLMES_SPACE>
G) polarization<OLMES_SPACE>
H) absorption<OLMES_SPACE>
I) diffraction<OLMES_SPACE>
J) transmission<OLMES_SPACE>

Answer: Let's think step by step. The colorful patterns observed in a soap bubble are caused by the phenomenon of light interference. This occurs when light waves bounce between the two surfaces of the soap film, combining constructively or destructively based on their phase differences and the varying thickness of the film. These interactions result in vibrant color patterns due to variations in the intensity of different wavelengths of light.
Answer: E.

Question:<OLMES_SPACE>
A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?<OLMES_SPACE>
A) 240 W
B) 120 W
C) 10 W
D) 480 W
E) 360 W
F) 200 W
G) 30 W
H) 150 W
I) 60 W
J) 300 W

Answer: Let's think step by step. The rate of energy usage, known as power, in an electrical circuit is calculated by the product of voltage and current. For a microwave oven connected to a 120 V outlet and drawing a current of 2 amps, the power consumption can be calculated as follows:
\[
\text{{Power}} = \text{{Voltage}} \times \text{{Current}} = 120 \, \text{{V}} \times 2 \, \text{{A}} = 240 \, \text{{W}}.
\]
Therefore, the microwave oven uses energy at a rate of 240 watts.
Answer: A.

Question:<OLMES_SPACE>
{}

Answer: Let's think step by step.<OLMES_SPACE>
""".replace("<OLMES_SPACE>", " ")


def question_block(doc: dict[str, Any]) -> str:
    options = "\n".join(
        f"{LETTERS[index]}) {option}" for index, option in enumerate(doc["options"])
    )
    return f"{doc['question']}\n{options}"


def doc_to_text(doc: dict[str, Any]) -> str:
    """Build the official zero-shot prompt."""
    return ZERO_SHOT_INSTRUCTION + question_block(doc) + "\n"


def doc_to_text_five_shot(doc: dict[str, Any]) -> str:
    """Build the official fixed five-shot prompt."""
    return FIVE_SHOT_TEMPLATE.format(question_block(doc))


def _answer_patterns(option_labels: str) -> tuple[str, ...]:
    wrappers = r"[\*\$\{(\[\\]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text)\{?)*"
    suffix = r"(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)"
    return (
        (
            rf"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*"
            rf"{wrappers}\s*([{option_labels}]){suffix}"
        ),
        rf"(?i:Answer)[\*\s]*:\s*{wrappers}\s*([{option_labels}]){suffix}",
        rf"^[^\w\r\n]*{wrappers}\s*([{option_labels}]){suffix}",
    )


def extract_option_label(text: str, num_options: int = 10) -> str | None:
    """Extract an answer letter, checking the final line before the full text."""
    if not isinstance(text, str):
        return None
    option_labels = LETTERS[:num_options]
    text = text.rstrip()
    candidates = (text.split("\n")[-1], text)
    for candidate in candidates:
        for pattern in _answer_patterns(option_labels):
            match = re.search(pattern, candidate, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    return None


def extract_option_content(text: str, options: Sequence[str]) -> str | None:
    """Fall back to matching the literal content of an answer option."""
    if not isinstance(text, str) or not options:
        return None
    alternatives = "|".join(
        re.escape(option) for option in sorted(options, key=len, reverse=True)
    )
    wrappers = r"[\*\$\{(\[\\]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text)\{?)*"
    suffix = r"(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)"
    patterns = (
        (
            rf"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*"
            rf"{wrappers}\s*({alternatives}){suffix}"
        ),
        rf"(?i:Answer)\s*:?[\*\s]*{wrappers}\s*({alternatives}){suffix}",
        rf"^[^\w\r\n]*{wrappers}\s*({alternatives}){suffix}",
    )
    text = text.rstrip()
    for candidate in (text.split("\n")[-1], text):
        for pattern in patterns:
            match = re.search(pattern, candidate, re.IGNORECASE)
            if match:
                return match.group(1)
    return None


def extract_answer(text: str, options: Sequence[str]) -> tuple[str | None, str]:
    """Return the extracted letter and source-compatible extraction status."""
    if not isinstance(text, str):
        return None, "error"
    label = extract_option_label(text, len(options))
    if label is not None:
        return label, "parsed"
    content = extract_option_content(text, options)
    if content is None:
        return None, "miss"
    return LETTERS[options.index(content)], "parsed"


Record = tuple[float, str, str, str, str, str]


def _process_results(
    doc: dict[str, Any], results: list[str], *, five_shot: bool
) -> dict[str, Record]:
    response = results[0]
    if five_shot:
        prefix = response.split("Question:", 1)[0]
        prediction, status = extract_answer(prefix, doc["options"])
        if prediction is None:
            prediction, status = extract_answer(response, doc["options"])
    else:
        prediction, status = extract_answer(response, doc["options"])

    record: Record = (
        float(prediction == doc["answer_letter"]),
        doc.get("discipline", "unknown"),
        doc.get("field", "unknown"),
        doc.get("subfield", "unknown"),
        doc.get("difficulty", "unknown"),
        status,
    )
    return {metric: record for metric in METRICS}


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Record]:
    return _process_results(doc, results, five_shot=False)


def process_results_five_shot(
    doc: dict[str, Any], results: list[str]
) -> dict[str, Record]:
    return _process_results(doc, results, five_shot=True)


def _sample_accuracy(records: Sequence[Record]) -> float:
    return fmean(record[0] for record in records)


def _macro_accuracy(records: Sequence[Record], key: Callable[[Record], str]) -> float:
    groups: defaultdict[str, list[float]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record[0])
    return fmean(fmean(values) for values in groups.values())


def aggregate_accuracy(records: Sequence[Record]) -> float:
    return _sample_accuracy(records)


def aggregate_discipline_accuracy(records: Sequence[Record]) -> float:
    return _macro_accuracy(records, lambda record: record[1])


def aggregate_field_accuracy(records: Sequence[Record]) -> float:
    return _macro_accuracy(records, lambda record: f"{record[1]}/{record[2]}")


def aggregate_subfield_accuracy(records: Sequence[Record]) -> float:
    return _macro_accuracy(
        records, lambda record: f"{record[1]}/{record[2]}/{record[3]}"
    )


def _difficulty_accuracy(records: Sequence[Record], difficulty: str) -> float:
    values = [record[0] for record in records if record[4] == difficulty]
    return fmean(values) if values else 0.0


def aggregate_easy_accuracy(records: Sequence[Record]) -> float:
    return _difficulty_accuracy(records, "easy")


def aggregate_middle_accuracy(records: Sequence[Record]) -> float:
    return _difficulty_accuracy(records, "middle")


def aggregate_hard_accuracy(records: Sequence[Record]) -> float:
    return _difficulty_accuracy(records, "hard")


def aggregate_miss_rate(records: Sequence[Record]) -> float:
    return fmean(record[5] == "miss" for record in records)


def aggregate_error_rate(records: Sequence[Record]) -> float:
    return fmean(record[5] == "error" for record in records)


METRICS = (
    "acc",
    "subfield_acc",
    "field_acc",
    "discipline_acc",
    "easy_acc",
    "middle_acc",
    "hard_acc",
    "miss_rate",
    "error_rate",
)
