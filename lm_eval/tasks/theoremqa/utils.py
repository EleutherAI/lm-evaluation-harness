"""Evaluation utilities for TheoremQA.

Scoring follows the official implementation
(https://github.com/TIGER-AI-Lab/TheoremQA, `utils.py` / `number_utils.py`):
answers are compared per `Answer_type`, floats within a 4% relative window,
integers after rounding, and lists elementwise after sorting.

Two deliberate deviations from the reference code:

- The reference resolves predictions with `latex2sympy` and `eval()`. This
  module parses numbers directly instead, so the task adds no dependency and
  never executes model output. Predictions whose only form is LaTeX that needs
  a CAS are scored incorrect rather than evaluated.
- Rows carrying an image are dropped (see `process_docs`); the reference ships
  them for multimodal runs.
- An `option` answer must name exactly one choice. The reference only checks
  that the gold label appears in the prediction, so an answer enumerating every
  option scores correct there (11 of the 16 option rows).
"""

from __future__ import annotations

import ast
import re
from typing import Any

import datasets


BOOL_TRUE = ("true", "yes")
BOOL_FALSE = ("false", "no")
OPTIONS = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
ANSWER_TRIGGER = "the answer is"
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
FRACTION_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$")
REL_TOLERANCE = 0.04


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Keep the text-only rows and drop the image column.

    53 of the 800 test rows carry a `Picture`, which a text-only task cannot
    answer. The column is removed rather than ignored because `datasets`
    decodes `Image` features on access and raises `ImportError` when Pillow is
    absent, which would break loading for text-only users.
    """
    dataset = dataset.cast_column("Picture", datasets.Image(decode=False))
    dataset = dataset.filter(lambda doc: doc["Picture"] is None)
    return dataset.remove_columns("Picture")


def doc_to_text(doc: dict[str, Any]) -> str:
    return f"Question: {doc['Question']}\nAnswer:"


def _strip_units(text: str) -> str:
    """Drop the currency, percent, and degree marks the reference cleans."""
    text = text.replace("%", "").replace("$", "").replace("¥", "")
    return text.replace("°C", "").replace("°", "").strip()


def _to_number(text: str | float) -> float | None:
    """Parse one scalar without evaluating the string."""
    if isinstance(text, (int, float)):
        return float(text)
    candidate = _strip_units(str(text)).replace(",", "").strip()
    fraction = FRACTION_RE.match(candidate)
    if fraction is not None:
        numerator, denominator = float(fraction[1]), float(fraction[2])
        return numerator / denominator if denominator else None
    try:
        return float(candidate)
    except ValueError:
        pass
    numbers = NUMBER_RE.findall(candidate)
    return float(numbers[-1]) if len(numbers) == 1 else None


def _to_number_list(text: str) -> list[float] | None:
    """Parse a bracketed list literal; `ast` keeps this away from `eval`."""
    candidate = _strip_units(text)
    if not (candidate.startswith(("[", "(")) and candidate.endswith(("]", ")"))):
        return None
    try:
        parsed = ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, (list, tuple)):
        return None
    values = [_to_number(item) for item in parsed]
    return None if any(value is None for value in values) else values


def _numbers_match(prediction: float, gold: float, gold_is_integer: bool) -> bool:
    if gold_is_integer:
        return round(prediction) == round(gold)
    return abs(prediction - gold) <= abs(gold) * REL_TOLERANCE


def extract_answer(completion: str) -> str:
    """Take the span after the answer trigger, else the last line."""
    text = completion.split("Question:")[0].strip()
    lowered = text.lower()
    if ANSWER_TRIGGER in lowered:
        text = text[lowered.rindex(ANSWER_TRIGGER) + len(ANSWER_TRIGGER) :]
    else:
        text = text.splitlines()[-1] if text.splitlines() else ""
    return text.strip().rstrip(".").rstrip("/").strip()


def is_correct(prediction: str, gold: str, answer_type: str) -> bool:
    """Compare one prediction against the gold answer of its declared type."""
    prediction = prediction.strip()
    if not prediction:
        return False
    lowered = prediction.lower()
    if answer_type == "bool":
        wanted = gold.strip().lower() in BOOL_TRUE
        said_true = any(token in lowered for token in BOOL_TRUE)
        said_false = any(token in lowered for token in BOOL_FALSE)
        if said_true == said_false:
            return False
        return said_true if wanted else said_false
    if answer_type == "option":
        # The reference accepts any prediction containing the gold label, which
        # also passes an answer that enumerates every option. Require that
        # exactly one label was named.
        named = [option for option in OPTIONS if option in lowered]
        return named == [gold.strip().lower()]
    if answer_type.startswith("list of"):
        predicted = _to_number_list(prediction)
        expected = _to_number_list(gold)
        if predicted is None or expected is None or len(predicted) != len(expected):
            return False
        integers = answer_type.endswith("integer")
        return all(
            _numbers_match(p, g, integers)
            for p, g in zip(sorted(predicted), sorted(expected), strict=True)
        )
    predicted_number = _to_number(prediction)
    expected_number = _to_number(gold)
    if predicted_number is None or expected_number is None:
        return lowered == gold.strip().lower()
    return _numbers_match(predicted_number, expected_number, answer_type == "integer")


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = extract_answer(results[0])
    correct = is_correct(prediction, doc["Answer"], doc["Answer_type"])
    return {"exact_match": float(correct)}


def list_fewshot_samples() -> list[dict[str, str]]:
    r"""The reference repository's five worked examples, carried verbatim.

    The dataset ships only a `test` split, so few-shot context cannot be
    sampled from the data without leaking evaluation rows; none of these five
    questions appears in the 800 test rows. Text is copied from `examples.py`
    upstream, with one repair: the third example's source uses a non-raw
    string, so its `\\frac` reached runtime as a form feed.
    """
    return [
        {
            "Question": "In a 10 Gigabit Ethernet network, the average size of a "
            "frame is 1500 bytes. If a burst of noise lasting 1ms "
            "interrupts the network, how many frames are lost?",
            "Solution": "First, calculate the data rate in bytes/s:\n"
            "\n"
            "10 Gigabit/s * (1 Byte / 8 bits) = 1.25 * 10^9 Bytes/s\n"
            "\n"
            "Next, calculate the data loss in bytes due to the noise:\n"
            "\n"
            "1 ms * 1.25 * 10^9 Bytes/s = 1.25 * 10^6 Bytes\n"
            "\n"
            "Finally, divide the data loss by the average frame size "
            "to get the number of frames lost:\n"
            "\n"
            "1.25 * 10^6 Bytes / 1500 Bytes/frame ≈ 833.33 frames\n"
            "The answer is 833.33",
            "few_shot": "1",
        },
        {
            "Question": "Given x = 0.157, what is the value of x \\times "
            "\\frac{\\prod_{n=1}^\\infty (1 - \\frac{x^2}{n^2 "
            "\\pi^2})}{\\sin(x)}?",
            "Solution": "To evaluate the expression $x \\times "
            "\\frac{\\prod_{n=1}^{\\infty} (1 - \\frac{x^2}{n^2 "
            "\\pi^2})}{\\sin(x)}$ given x = 0.157, we first recognize "
            "that the product in the numerator is related to the sine "
            "function through the Euler's reflection formula for the "
            "sine function, which can be expressed as:\n"
            "\n"
            "$$\\sin(x) = x \\prod_{n=1}^{\\infty} \\left(1 - "
            "\\frac{x^2}{n^2 \\pi^2}\\right)$$\n"
            "\n"
            "Therefore, the given expression simplifies to: $x \\times "
            "\\frac{\\sin(x)}{\\sin(x)}$\n"
            "\n"
            "Because sin(x) in the numerator and denominator cancels "
            "out, the expression simplifies further to just x.\n"
            "\n"
            "So, given x = 0.157, the value of the expression is "
            "0.157. This result is derived from the properties of the "
            "sine function and does not require computational "
            "evaluation.\n"
            "The answer is 0.157",
            "few_shot": "1",
        },
        {
            "Question": "Consider the basis C of \\mathbb{R}^2 consisting of "
            "vectors u_1 = [2, 4] and u_2 = [1, -1]. If y = [8, 12], "
            "find the C-coordinate vector of y.",
            "Solution": "The goal is to express y as a linear combination of the "
            "basis vectors of C, i.e., $y = a\\cdot u_1 + b\\cdot "
            "u_2$, where a and b are the scalar coefficients that we "
            "want to find. These coefficients will form the "
            "C-coordinate vector of y, which we'll denote as $[a, "
            "b]_C$.\n"
            "\n"
            "Given:\n"
            "- $u_1 = [2, 4]$,\n"
            "- $u_2 = [1, -1]$,\n"
            "- $y = [8, 12]$.\n"
            "\n"
            "We need to solve the system of linear equations:\n"
            "2a + 1b = 8\n"
            "4a - 1b = 12\n"
            "\n"
            "Let's solve this system of equations to find a and b.\n"
            "\n"
            "The solution to the system of equations is $a = "
            "\\frac{10}{3} and b = \\frac{4}{3}$. Therefore, the "
            "C-coordinate vector of y in the basis consisting of "
            "vectors u_1 = [2, 4] and u_2 = [1, -1] is "
            "$\\left[\\frac{10}{3}, \\frac{4}{3}\\right]_C$. \n"
            "Let's calculate the numerical value of "
            "$\\left[\\frac{10}{3}, \\frac{4}{3}\r"
            "ight]_C$ as [3.33, 1.33].\n"
            "The answer is [3.33, 1.33]",
            "few_shot": "1",
        },
        {
            "Question": "One can draw a simple, connected planar graph with 200 "
            "vertices and 397 edges. Is this statement Trur or False?",
            "Solution": "To determine the answer, we can use Euler's formula for "
            "planar graphs, which states that for any finite, "
            "connected, planar graph, $V - E + F = 2$, where V is the "
            "number of vertices, E is the number of edges, and F is "
            "the number of faces.\n"
            "\n"
            "Given the modified question, we have V = 200 vertices and "
            "E = 397 edges. We want to find if we can have a graph "
            "that satisfies these conditions, adhering to Euler's "
            "formula.\n"
            "\n"
            "First, let's rearrange Euler's formula to solve for F:  F "
            "= E - V + 2\n"
            "\n"
            "Substituting the given values: F = 397 - 200 + 2,  F = "
            "199\n"
            "\n"
            "This means a graph with 200 vertices and 397 edges would "
            "have 199 faces. However, to determine the truth of this "
            "possibility, we should check if this graph doesn't "
            "violate any other planar graph constraints, particularly "
            "regarding the number of edges.\n"
            "\n"
            "For a simple, connected planar graph, there's also a "
            "relationship between vertices, edges, and faces given by "
            "the inequality: $E \\leq 3V - 6$\n"
            "\n"
            "Substituting V = 200 gives: $E \\leq 3*200 - 6 = 594$\n"
            "\n"
            "With E = 397, the condition $E \\leq 594$ is satisfied, "
            "meaning it's theoretically possible in terms of the edge "
            "condition for a planar graph.\n"
            "\n"
            "Therefore, one can draw a simple, connected planar graph "
            "with 200 vertices and 397 edges, resulting in 199 faces, "
            "without violating the conditions for it to be planar "
            "according to both Euler's formula and the constraint on "
            "the maximum number of edges.\n"
            "The answer is True",
            "few_shot": "1",
        },
        {
            "Question": "Given a finite group G, and a collection of permutations "
            "H on a set. Then (a) there always exists H such that G is "
            "isomorphic to H; (b) for any H, G is isomorphic to H; (c) "
            "G can never be isomorphic to H; (d) none of the above. "
            "Which option is correct?",
            "Solution": "This is based on Cayley's theorem, which states that "
            "every group G is isomorphic to a subgroup of the "
            "symmetric group acting on G. \n"
            "In other words, for every finite group G, there exists a "
            "collection of permutations H (which in this context, can "
            "be thought of as the set of permutations representing the "
            "action of G on itself) such that G is isomorphic to H.\n"
            "\n"
            "Therefore, there always exists H such that G is "
            "isomorphic to H.\n"
            "The answer is (a)",
            "few_shot": "1",
        },
    ]
