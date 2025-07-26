from lm_eval.tasks.gpqa.zeroshot.utils import process_docs
from typing import Dict, List

import re
import regex
from math import isclose
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from word2number import w2n
from latex2sympy2_extended import latex2sympy

import re
import regex

# import dataset

def doc_to_target(doc):
    return postprocess_target(doc["answer"])

def postprocess_target(s):
    return strip_answer_string(s)

def postprocess(output):

    response = re.sub(r".*?<\/think>(\\n)*", "", output, flags=re.DOTALL).strip()

    try:
        answer = strip_answer_string(extract_answer(response))
        return answer
    
    except Exception:
        return output


def process_results_math(doc: dict, results: List[str]) -> Dict[str, int]:
    
    candidate = postprocess(results[0])
    gold = postprocess_target(doc["answer"])

    if not gold:
        print(doc, candidate, gold)
    
    retval = 1 if math_equal(candidate,gold) else 0

    results = {
        "exact_match": retval,
    }
    return results


def process_gpqa_docs(dataset):
    return process_docs(dataset)



def process_results_gpqa(doc: dict, results: List[str]) -> Dict[str, int]:
    
    candidate = choice_answer_clean(postprocess(results[0]))
    gold = choice_answer_clean(postprocess_target(doc["answer"]))
    retval = 0

    if not gold:
        print(doc, candidate, gold)
    
    retval =exact_match_fn(gold,candidate)

    results = {
        "exact_match": retval,

    }
    return results





# Reference: https://github.com/modelscope/evalscope/blob/main/evalscope/metrics/math_parser.py#L335

def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except Exception:
        pass
    return text


def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if len(substr) > 0 and substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        if 'sqrt' not in a:
            a = int(a)
        if 'sqrt' not in b:
            b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except Exception:
        return string


def _fix_sqrt(string):
    _string = re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)
    return _string


def strip_answer_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace('\n', '')

    # right "."
    string = string.rstrip('.')

    # remove inverse spaces
    # replace \\ with \
    string = string.replace('\\!', '')
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r'\\begin\{array\}\{.*?\}', r'\\begin{pmatrix}', string)
    string = re.sub(r'\\end\{array\}', r'\\end{pmatrix}', string)
    string = string.replace('bmatrix', 'pmatrix')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    string = (string.replace('\\neq', '\\ne').replace('\\leq', '\\le').replace('\\geq', '\\ge'))

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')
    string = string.replace('\\{', '{')
    string = string.replace('\\}', '}')

    # Function to replace number words with corresponding digits
    def replace_match(match):
        word = match.group(1).lower()
        if convert_word_number(word) == word:
            return match.group(0)
        else:
            return convert_word_number(word)

    string = re.sub(r'\\text\{([a-zA-Z]+)\}', replace_match, string)

    # Before removing unit, check if the unit is squared (for surface area)
    string = re.sub(r'(cm|inches)\}\^2', r'\1}', string)

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r'\\text{.*?}$', '', string).strip()
    if _string != '' and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')
    string = string.replace('$', '')
    string = string.replace('\\(', '').replace('\\)', '')

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r'\\text\{(.*?)\}', r'\1', string)
    for key in ['x=', 'y=', 'z=', 'x\\in', 'y\\in', 'z\\in', 'x\\to', 'y\\to', 'z\\to']:
        string = string.replace(key, '')
    string = string.replace('\\emptyset', r'{}')
    string = string.replace('(-\\infty,\\infty)', '\\mathbb{R}')

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')
    string = string.replace('%', '')

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')

    # cdot
    # string = string.replace("\\cdot", "")
    if (string.startswith('{') and string.endswith('}') and string.isalnum()
            or string.startswith('(') and string.endswith(')') and string.isalnum()
            or string.startswith('[') and string.endswith(']') and string.isalnum()):
        string = string[1:-1]

    # inf
    string = string.replace('infinity', '\\infty')
    if '\\infty' not in string:
        string = string.replace('inf', '\\infty')
    string = string.replace('+\\inity', '\\infty')

    # and
    string = string.replace('and', '')
    string = string.replace('\\mathbf', '')

    # use regex to remove \mbox{...}
    string = re.sub(r'\\mbox{.*?}', '', string)

    # quote
    string.replace("'", '')
    string.replace('"', '')

    # i, j
    if 'j' in string and 'i' not in string:
        string = string.replace('j', 'i')

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r'(\d+)\.0*([^\d])', r'\1\2', string)
    string = re.sub(r'(\d+)\.0*$', r'\1', string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    string = _fix_sqrt(string)
    string = string.replace(' ', '')

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    # Remove unnecessary '\' before integers
    string = re.sub(r'\\(?=\-?\d+(\\|\)|,|\]|$))', '', string)

    # Remove grade level (e.g., 12th grade) and just maintain the integer
    string = re.sub(r'thgrade$', '', string)

    # If the answer is a list of integers (without parenthesis), sort them
    if re.fullmatch(r'(\s*-?\d+\s*,)*\s*-?\d+\s*', string):
        # Split the string into a list of integers
        try:
            integer_list = list(map(int, string.split(',')))
        except Exception:
            integer_list = list(map(int, '-1,-1'.split(',')))

        # Sort the list in ascending order
        sorted_list = sorted(integer_list)

        # Join the sorted list back into a comma-separated string
        string = ','.join(map(str, sorted_list))

    return string


def extract_answer(pred_str, use_last_number=True):
    pred_str = pred_str.replace('\u043a\u0438', '')
    if 'final answer is $' in pred_str and '$. I hope' in pred_str:
        # minerva_math
        tmp = pred_str.split('final answer is $', 1)[1]
        pred = tmp.split('$. I hope', 1)[0].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans) == 0:
            return ''
        elif ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        pred = a
    elif 'he answer is' in pred_str:
        pred = pred_str.split('he answer is')[-1].strip()
    elif 'final answer is' in pred_str:
        pred = pred_str.split('final answer is')[-1].strip()
    elif '答案是' in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split('答案是')[1].strip().split('\n\n')[0].strip()
    else:  # use the last number
        if use_last_number:
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str.replace(',', ''))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ''
        else:
            pred = ''

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r'\n\s*', '', pred)
    if pred != '' and pred[0] == ':':
        pred = pred[1:]
    if pred != '' and pred[-1] == '.':
        pred = pred[:-1]
    if pred != '' and pred[-1] == '/':
        pred = pred[:-1]
    pred = strip_answer_string(pred)
    return pred


def choice_answer_clean(pred: str):
    pred = pred.strip('\n').rstrip('.').rstrip('/').strip(' ').lstrip(':')
    # Clean the answer based on the dataset
    tmp = re.findall(r'\b(A|B|C|D|E)\b', pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip('.')]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip('.').rstrip('/')
    return pred


def parse_digits(num):
    num = regex.sub(',', '', str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith('%'):
            num = num[:-1]
            if num.endswith('\\'):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r'\{.*,.*\}', input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip('{}')
        pmatrix = r'\begin{pmatrix}' + m.replace(',', '\\') + r'\end{pmatrix}'
        pmatrix_list.append(pmatrix)

    return ', '.join(pmatrix_list)


def math_equal(
    prediction,
    reference,
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (reference in ['A', 'B', 'C', 'D', 'E'] and choice_answer_clean(prediction) == reference):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if 'pmatrix' in prediction and 'pmatrix' not in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith('[') and prediction.endswith(']')
            and not reference.startswith('(')) or (prediction.startswith('(') and prediction.endswith(')')
                                                   and not reference.startswith('[')):
        pred_str = pred_str.strip('[]()')
        ref_str = ref_str.strip('[]()')
    for s in ['{', '}', '(', ')']:
        ref_str = ref_str.replace(s, '')
        pred_str = pred_str.replace(s, '')
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (regex.match(r'(\(|\[).+(\)|\])', prediction) is not None
            and regex.match(r'(\(|\[).+(\)|\])', reference) is not None):
        pred_parts = prediction[1:-1].split(',')
        ref_parts = reference[1:-1].split(',')
        if len(pred_parts) == len(ref_parts):
            if all(
                [math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                 for i in range(len(pred_parts))]):
                return True
    if ((prediction.startswith('\\begin{pmatrix}') or prediction.startswith('\\begin{bmatrix}'))
            and (prediction.endswith('\\end{pmatrix}') or prediction.endswith('\\end{bmatrix}'))
            and (reference.startswith('\\begin{pmatrix}') or reference.startswith('\\begin{bmatrix}'))
            and (reference.endswith('\\end{pmatrix}') or reference.endswith('\\end{bmatrix}'))):
        pred_lines = [
            line.strip() for line in prediction[len('\\begin{pmatrix}'):-len('\\end{pmatrix}')].split('\\\\')
            if line.strip()
        ]
        ref_lines = [
            line.strip() for line in reference[len('\\begin{pmatrix}'):-len('\\end{pmatrix}')].split('\\\\')
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split('&')
                ref_parts = ref_line.split('&')
                if len(pred_parts) == len(ref_parts):
                    if not all([
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            ) for i in range(len(pred_parts))
                    ]):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count('=') == 1 and reference.count('=') == 1:
        pred = prediction.split('=')
        pred = f'{pred[0].strip()} - ({pred[1].strip()})'
        ref = reference.split('=')
        ref = f'{ref[0].strip()} - ({ref[1].strip()})'
        if symbolic_equal(pred, ref) or symbolic_equal(f'-({pred})', ref):
            return True
    elif (prediction.count('=') == 1 and len(prediction.split('=')[0].strip()) <= 2 and '=' not in reference):
        if math_equal(prediction.split('=')[1], reference, include_percentage, is_close):
            return True
    elif (reference.count('=') == 1 and len(reference.split('=')[0].strip()) <= 2 and '=' not in prediction):
        if math_equal(prediction, reference.split('=')[1], include_percentage, is_close):
            return True

    if symbolic_equal(prediction, reference):
        return True

    return False


def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):

    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace('\\\\', '\\'))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False

def exact_match_fn(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if gold.strip() == pred.strip() else 0