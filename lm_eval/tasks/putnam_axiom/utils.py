import re
import signal
from typing import Dict, List, Optional

import antlr4
import datasets

from lm_eval.utils import eval_logger

import numpy as np

from lm_eval.tasks.minerva_math.utils import *

# def first_boxed_only_string(string: str) -> Optional[str]:
#     idx = string.find("\\boxed")
#     if idx < 0:
#         idx = string.find("\\fbox")
#         if idx < 0:
#             return None

#     i = idx
#     right_brace_idx = None
#     num_left_braces_open = 0
#     while i < len(string):
#         if string[i] == "{":
#             num_left_braces_open += 1
#         if string[i] == "}":
#             num_left_braces_open -= 1
#             if num_left_braces_open == 0:
#                 right_brace_idx = i
#                 break
#         i += 1

#     if right_brace_idx is None:
#         retval = None
#     else:
#         retval = string[idx : right_brace_idx + 1]

#     return retval

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )

# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:

    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": normalize_final_answer(
                remove_boxed(last_boxed_only_string(doc["solution"]))
            ),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)

def process_variations(dataset: datasets.Dataset, ) -> datasets.Dataset:
    index = np.random.randint(0, 5) + 1
    def filter_doc(doc: dict) -> bool:
        return int(doc.get("variation")) == index

    filtered_dataset = dataset.filter(filter_doc)

    # Further processing if needed
    def _process_doc(doc: dict) -> dict:  

        try:
            out_doc = {
                "problem": doc["problem"],
                "solution": doc["solution"],
                "answer": normalize_final_answer(
                    remove_boxed(last_boxed_only_string(doc["solution"]))
                )
            }
        except:
            print(doc["problem"])
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    processed_dataset = filtered_dataset.map(_process_doc, batched=False)
    return processed_dataset


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "Let $ABC$ be a triangle with angle $A < \\angle C < 90^\\circ < \\angle B$. Consider the bisectors of the external angles at $A$ and $B$, each measured from the vertex to the opposite side (extended). Suppose both of these line-segments are equal to $AB$. Compute the angle $A$.",
            "solution": "Let's think step by step. Suppose the bisector of the exterior angle at $A$ intersects line $BC$ at $X$ and the bisector of the exterior angle at $B$ meets the line $AC$ at $Y$. The assumption that $C$ is between $B$ and $X$ contradicts the fact that $\\angle B > \\angle C$, so we may assume that $B$ is between $X$ and $C$. Similarly, we conclude that $C$ is between $A$ and $Y$ because $\\angle A < \\angle C$.\n\nIf $Z$ is a point on line $AB$ with $B$ between $A$ and $Z$, we have from triangle $ABY$ that $\\angle ZBY = 2A$. Hence, $\\angle BXA = \\angle ABX = \\angle ZBC = 2 \\angle ZBY = 4A$, and the angle sum of triangle $ABX$ is $90^\\circ - \\frac{1}{2}A + 8A$. Thus, $A = \\boxed{12}^\\circ$.",
            "few_shot": "1",
        },
        {
            "problem": "Given any positive integer $n$ find the value of \\n\\[ \\sum_{r=0}^{\\lfloor (n-1)/2 \\rfloor} \\left\\{\\frac{n - 2r}{n}\\binom{n}{r}\\right\\}^2, \\]\\n where $\\lfloor x \\rfloor$ means the greatest integer not exceeding $x$, and $\\binom{n}{r}$ is the binomial coefficient \"$n$ choose $r$,\" with the convention $\\binom{0}{0} = 1$. Return your final answer in binomial coefficient form with all other multiplicants reduced to the lowest form.",
            "solution": "Let's think step by step. Substituting $s=n-r$ in the given summation reveals that twice this sum is equal to:\n\\[\n\\sum_{r=0}^n \\left(\\frac{n-2r}{n} \\binom{n}{r}\\right)^2 = \\sum \\left(1 - 2\\frac{r}{n}\\right)^2 \\binom{n}{r}^2 = \\sum \\binom{n}{r}^2 - 4\\sum \\frac{r}{n} \\binom{n}{r}^2 + 4\\sum \\left(\\frac{r}{n}\\right)^2 \\binom{n}{r}^2.\n\\]\n\\[\n= \\binom{2n}{n} - 4 \\sum_{r=1}^n \\binom{n-1}{r-1} \\binom{n}{r} + 4 \\sum_{r=1}^n \\binom{n-1}{r-1}^2.\n\\]\n\\[\n= \\binom{2n}{n} - 4\\binom{2n-1}{n-1} + 4\\binom{2n-2}{n-1}.\n\\]\n\\[\n= \\frac{2n(2n-1)}{n^2} - \\frac{4(n-1)}{n}\\binom{2n-2}{n-1} = \\boxed{\\frac{1}{n}\\binom{2n-2}{n-1}}.\\n\\]",
            "few_shot": "1",
        },
        {
            "problem": "Find the sum of all sides of all the right-angled triangles whose sides are integers while the area is numerically equal to twice the perimeter.",
            "solution": "Let's think step by step. All Pythagorean triples can be obtained from $x = \\lambda(p^2 - q^2)$, $y = 2\\lambda pq$, $z = \\lambda(p^2 + q^2)$ where $0 < q < p$, $(p, q) = 1$ and $p \\not\\equiv q \\pmod{2}$, $\\lambda$ being any natural number.\n\nThe problem requires that $\\frac{1}{2}xy = 2(x+y+z)$. This condition can be written as $\\lambda^2(p^2-q^2)(pq) = 2\\lambda(p^2-q^2+2pq+p^2+q^2)$ or simply $\\lambda(p-q)q = 4$. Since $p-q$ is odd it follows that $p-q = 1$ and the only possibilities for $q$ are $1, 2, 4$.\n\n- If $q = 1$, $p = 2$, $\\lambda = 4$, $x = 12$, $y = 16$, $z = 20$.\n- If $q = 2$, $p = 3$, $\\lambda = 2$, $x = 10$, $y = 24$, $z = 26$.\n- If $q = 4$, $p = 5$, $\\lambda = 1$, $x = 9$, $y = 40$, $z = 41$. This gives us the final answer as $12+16+20+10+24+26+9+40+41 = \\boxed{198}$.",
            "few_shot": "1",
        },
        {
            "problem": "Evaluate \n\\[\n\\lim_{n \\to \\infty} \\int_0^1 \\int_0^1 \\cdots \\int_0^1 \\cos^2 \\left(\\frac{\\pi}{2n}(x_1 + x_2 + \\cdots + x_n)\\right) dx_1 dx_2 \\cdots dx_n.\n\\]",
            "solution": "Let's think step by step. The change of variables $x_k \\to 1 - x_k$ yields\n\\[\n\\int_0^1 \\int_0^1 \\cdots \\int_0^1 \\cos^2 \\left(\\frac{\\pi}{2n}(x_1 + x_2 + \\cdots + x_n)\\right) dx_1 dx_2 \\cdots dx_n \\\\\n= \\int_0^1 \\int_0^1 \\cdots \\int_0^1 \\sin^2 \\left(\\frac{\\pi}{2n}(x_1 + x_2 + \\cdots + x_n)\\right) dx_1 dx_2 \\cdots dx_n.\n\\]\nEach of these expressions, being equal to half their sum, must equal $\\frac{1}{2}$. The limit is also $\\boxed{\\frac{1}{2}}$.",
            "few_shot": "1",
        }
        # {
        #     "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        #     "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $[2,5)$.\nFinal Answer: The final answer is $\\boxed{[2,5)}$. I hope it is correct.",
        #     "few_shot": "1",
        # },
        # {
        #     "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        #     "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = 24.$\nFinal Answer: The final answer is $\\boxed{24}$. I hope it is correct.",
        #     "few_shot": "1",
        # },
        # {
        #     "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        #     "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=16\n\\end{align*}\nFinal Answer: The final answer is $\\boxed{16}$. I hope it is correct.",
        #     "few_shot": "1",
        # },
        # {
        #     "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        #     "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=-\\frac{2}{3}.$$\nFinal Answer: The final answer is $\\boxed{-\\frac{2}{3}}$. I hope it is correct.",
        #     "few_shot": "1",
        # }
    ]

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    # candidates = results[0]
    completion_output = results[0].outputs[0]
    candidates = completion_output.text
    
    try:
        answer = ground_truth_boxed_answer(candidates)
    except:
        answer = get_generated_answer(candidates)

    if is_equiv(answer, doc["answer"]):
        # print(answer)
        # print(doc['answer'])
        # print(doc['year'] + '_' + doc['id'])
        retval = 1
    else:
        retval = 0

    results = {
        "exact_match": retval,
    }

    return results

def ground_truth_boxed_answer(solution: str) -> str:
    return normalize_final_answer(remove_boxed(last_boxed_only_string(solution)))

def get_generated_answer(result: str) -> str:
    unnormalized_answer = get_unnormalized_answer(result)
    answer = normalize_final_answer(unnormalized_answer)

    return answer

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    # match = re.search(
    #     r"Final Answer: The final answer is(.*?). I hope it is correct.",
    #     text,
    # )
    match = re.search(
        r"(?:Final Answer: The final answer is|The answer is:)\s*(.*?)(?=I hope it is correct\.)",
        text
    )
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer