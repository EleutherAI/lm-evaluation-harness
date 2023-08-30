import re


# taken from https://github.com/microsoft/AGIEval/blob/main/src/dataset_loader.py
def doc_to_text_math_fewshot(doc: dict) -> str:
    """
    'Here are the answers for the problems in the exam.\n
    <Problem 1.>    <Question>\n
    The answer is therefore <answer>\n
    ...
    <Problem n.>   <Question>

    original adds \n at the end. I removed it.
    """

    _fewshot = [
        "Here are the answers for the problems in the exam.\n",
        "Problem 1.   Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.\nThe answer is therefore [2,5)",
        "Problem 2.   If $\\det \\mathbf{A} = 5,$ then find $\\det (\\mathbf{A^3}).$\nThe answer is therefore 125",
        "Problem 3.   Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?\nThe answer is therefore 16",
        "Problem 4.   If the system of equations  \\begin{align*}\n3x+y&=a,\\\\\n2x+5y&=2a,\n\\end{align*} has a solution $(x,y)$ when $x=2$, compute $a$.\nThe answer is therefore \\frac{26}{3}",
    ]
    _fewshot = "\n\n".join(_fewshot)
    question_input = "Problem {}.   ".format(5) + doc["question"]
    return _fewshot + question_input


def process_results_math(doc, results):
    completions = results[0]
    processed_answer = parse_math_answer(completions)
    if is_equiv(processed_answer, str(doc["answer"])):
        return {"acc": 1}
    else:
        return {"acc": 0}


# taken from https://github.com/microsoft/AGIEval/blob/main/src/post_process.py


def remove_few_shot_prefix(string: str):
    string = string.strip()
    prefix = "The answer is therefore"
    if string.startswith(prefix):
        string = string[len(prefix) :].split("\n")[0].strip()
    elif prefix in string:
        index = string.rfind(prefix)
        if index >= 0:
            string = string[index + len(prefix) :].split("\n")[0].strip()
    return string


def parse_math_answer(raw_string: str) -> str:
    raw_string = remove_few_shot_prefix(raw_string)
    return raw_string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        # print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        return ss1 == ss2
    except:  # noqa: E722
        return str1 == str2
