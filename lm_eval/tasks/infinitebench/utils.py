"""
Utility functions for InfiniteBench evaluation tasks.

InfiniteBench is a long-context benchmark with average data length
exceeding 100K tokens, covering retrieval, math, code, and reading comprehension.

Evaluation methods match the official implementation:
https://github.com/OpenBMB/InfiniteBench/blob/main/src/compute_scores.py

Reference: https://arxiv.org/abs/2402.13718
Dataset: https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench
"""

import logging
import re
import string

eval_logger = logging.getLogger(__name__)

# Full-width and CJK punctuation set matching the official normalize_zh_answer()
_ZH_PUNCTUATION = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
    "｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—"
    "''‛""„‟…‧﹏."
)


def postprocess_prediction(prediction: str) -> str:
    """Clean up model prediction for evaluation."""
    prediction = prediction.strip()
    prediction = re.sub(r"[\x00-\x1f]", " ", prediction).strip()
    return prediction


# ---- doc_to_text helpers ----


def doc_to_text_code_run(doc: dict) -> str:
    """Format prompt for code_run task, extracting func name and call from input.

    Matches official GPT-4 template from InfiniteBench.
    """
    context = doc["context"]
    input_text = doc["input"]

    match = re.search(r"(func_[0-9]+)\(\-?[0-9]+\)", input_text)
    if match:
        func = match.group(1)
        func_call = match.group(0)
    else:
        func = "the function"
        func_call = input_text.strip()

    return (
        f"Following is a set of Python functions. "
        f"There is a function called named {func}.\n\n"
        f"{context}\n\n"
        f"Please give me the exact number of the return value of "
        f"{func_call}. Be concise. Your response must end with the "
        f"final returned value."
    )


def doc_to_text_code_debug(doc: dict) -> str:
    """Format prompt for code_debug task with options.

    Matches official GPT-4 template from InfiniteBench.
    Note: "funtion" typo is intentional, matching the official repo.
    """
    context = doc["context"]
    options = doc.get("options", [])
    opt_a = options[0] if len(options) > 0 else ""
    opt_b = options[1] if len(options) > 1 else ""
    opt_c = options[2] if len(options) > 2 else ""
    opt_d = options[3] if len(options) > 3 else ""

    return (
        "There is ONLY ONE function in the large project that is "
        "deliberately made to include an obvious error. Please find "
        "the function that contains the most obvious errors. I will "
        "give you four options to narrow your scope. You can inspect "
        "the options and think. Eventually, tell me the answer using "
        "one single letter (A, B, C, or D).\n\n"
        f"{context}\n\n"
        "Which funtion has deliberate error?\n"
        f"A. {opt_a}\n"
        f"B. {opt_b}\n"
        f"C. {opt_c}\n"
        f"D. {opt_d}\n\n"
        "You should first find the functions in the options. Repeat "
        "their content, inspect through code, and at last give me "
        "your answer for the function that has the deliberate and "
        "obvious error in A, B, C, or D."
    )


def doc_to_text_longbook_choice(doc: dict) -> str:
    """Format prompt for longbook_choice_en with multiple-choice options.

    Matches official GPT-4 template from InfiniteBench.
    """
    context = doc["context"]
    question = doc["input"]
    options = doc.get("options", [])
    opt_a = options[0] if len(options) > 0 else ""
    opt_b = options[1] if len(options) > 1 else ""
    opt_c = options[2] if len(options) > 2 else ""
    opt_d = options[3] if len(options) > 3 else ""

    return (
        "Read the book and answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Only one of the following options is correct, tell me the "
        "answer using one single letter (A, B, C, or D). Don't say "
        "anything else.\n"
        f"A. {opt_a}\n"
        f"B. {opt_b}\n"
        f"C. {opt_c}\n"
        f"D. {opt_d}"
    )


def doc_to_text_longdialogue(doc: dict) -> str:
    """Format prompt for longdialogue_qa_en with $$MASK$$ explanation.

    Matches official GPT-4 template from InfiniteBench.
    """
    context = doc["context"]
    return (
        'Below is a dialogue script where one random occurrence of a '
        'character name is replaced with "$$MASK$$", and you should '
        "try to guess who that character is.\n\n"
        "The dialogue:\n\n---\n\n"
        f"{context}\n\n"
        "---\n\nEnd of dialogue.\n\n"
        'Which character is most likely "$$MASK$$"? Just say the name '
        "used by the scriptwriter (before the colon marks) of one "
        "single character and nothing else."
    )


def doc_to_text_math_find(doc: dict) -> str:
    """Format prompt for math_find task.

    Matches official create_prompt() which extracts a prefix from the input
    using the pattern "The X of" and constructs "What is {target} in the
    following list?" as the prefix.
    """
    context = doc["context"]
    input_text = doc["input"]

    find_result = re.findall(r"The .+ of", input_text)
    if find_result:
        target_number = find_result[0].lower()[:-3]
        prefix = f"What is {target_number} in the following list?"
    else:
        prefix = input_text

    return f"{prefix}\n\n{context}\n\n{input_text}"


# ---- Normalization helpers (matching official compute_scores.py) ----


def _normalize_en(text: str) -> str:
    """Normalize English text for F1 computation.

    Matches the official normalize_answer() from InfiniteBench.
    """
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def _normalize_zh(text: str) -> str:
    """Normalize Chinese text for F1 computation.

    Matches the official normalize_zh_answer() from InfiniteBench.
    Uses the full CJK punctuation set from the official implementation.
    """
    text = re.sub(
        r"[{}]".format(re.escape(string.punctuation + _ZH_PUNCTUATION)), "", text
    )
    text = text.lower()
    text = "".join(text.split())
    return text


def _token_f1(prediction: str, reference: str, is_chinese: bool = False) -> float:
    """Compute token-level F1 score between prediction and reference.

    For English: tokenize by whitespace after normalization.
    For Chinese: tokenize by character after normalization.
    Matches the official F1 scoring from InfiniteBench.
    """
    if is_chinese:
        pred_tokens = list(_normalize_zh(prediction))
        ref_tokens = list(_normalize_zh(reference))
    else:
        pred_tokens = _normalize_en(prediction).split()
        ref_tokens = _normalize_en(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---- Result processing ----


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """Generic result processing using case-insensitive substring matching.

    Used for: longdialogue_qa_en.
    """
    prediction = postprocess_prediction(results[0])
    answers = doc.get("answer", [])

    if not answers:
        return {"score": 0.0}

    pred_upper = prediction.upper()
    for answer in answers:
        if answer.strip().upper() in pred_upper:
            return {"score": 1.0}

    return {"score": 0.0}


def process_results_kv(doc: dict, results: list[str]) -> dict[str, float]:
    """Result processing for kv_retrieval task.

    Tokenizes by replacing punctuation with spaces, then checks if
    the answer appears as a complete word. Matches the official
    get_score_one_kv_retrieval() from InfiniteBench.
    """
    prediction = postprocess_prediction(results[0])
    answers = doc.get("answer", [])

    if not answers:
        return {"score": 0.0}

    # Official: replace these characters with space, then split into words
    cleaned = prediction
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        cleaned = cleaned.replace(c, " ")
    words = cleaned.split()

    for answer in answers:
        if answer.strip() in words:
            return {"score": 1.0}

    return {"score": 0.0}


def process_results_int_match(doc: dict, results: list[str]) -> dict[str, float]:
    """Result processing for passkey and number_string tasks.

    Splits on non-digit characters and takes the first number.
    Matches the official first_int_match() from InfiniteBench which
    uses re.split("[^0-9]", ...).
    """
    prediction = postprocess_prediction(results[0])
    answers = doc.get("answer", [])

    if not answers:
        return {"score": 0.0}

    # Official: split on non-digits, take first non-empty element
    parts = re.split(r"[^0-9]", prediction)
    pred_int = next((p for p in parts if p), "")

    if not pred_int:
        return {"score": 0.0}

    for answer in answers:
        if pred_int == answer.strip():
            return {"score": 1.0}

    return {"score": 0.0}


def process_results_math(doc: dict, results: list[str]) -> dict[str, float]:
    """Result processing for math tasks.

    Extracts the first number (int or float) from the prediction and
    compares numerically. Matches the official scoring from InfiniteBench.
    """
    prediction = postprocess_prediction(results[0])
    answers = doc.get("answer", [])

    if not answers:
        return {"score": 0.0}

    match = re.search(r"\d+\.\d+|\d+", prediction)
    if not match:
        return {"score": 0.0}

    pred_str = match.group(0)

    for answer in answers:
        answer_clean = answer.strip()
        # Compare numerically to handle int/float equivalence
        try:
            if "." in pred_str or "." in answer_clean:
                if float(pred_str) == float(answer_clean):
                    return {"score": 1.0}
            else:
                if int(pred_str) == int(answer_clean):
                    return {"score": 1.0}
        except ValueError:
            if pred_str == answer_clean:
                return {"score": 1.0}

    return {"score": 0.0}


def _answer_to_letter(doc: dict) -> str:
    """Derive the letter (A/B/C/D) from the answer text by looking up its
    position in the options list.

    The HuggingFace dataset stores answers as ["option_text"], not ["A"].
    The official get_answer() in eval_utils.py maps the text to a letter.
    """
    answers = doc.get("answer", [])
    options = doc.get("options", [])
    if not answers or not options:
        return ""
    answer_text = answers[0].strip()
    # Try exact match in options
    for i, opt in enumerate(options):
        if opt.strip() == answer_text:
            return chr(ord("A") + i)
    # Fallback: if answer is already a single letter
    if len(answer_text) == 1 and answer_text.upper() in "ABCDEFGHIJ":
        return answer_text.upper()
    return ""


def process_results_choice(doc: dict, results: list[str]) -> dict[str, float]:
    """Result processing for longbook_choice_en. Extracts last A/B/C/D.

    Derives the expected letter from doc["options"] since the HuggingFace
    dataset stores answer text, not letters. Then matches using the official
    get_score_one_longbook_choice_eng() 6-step fallback.
    """
    prediction = postprocess_prediction(results[0])
    label = _answer_to_letter(doc)

    if not label:
        return {"score": 0.0}

    pred = prediction.strip()

    # Step 1: find last standalone A-D letter (official regex)
    match = re.search(r"\b([A-D])\b(?!.*\b[A-D]\b)", pred)
    if match:
        return {"score": 1.0 if match.group(1).upper() == label else 0.0}

    # Step 2: empty prediction
    if not pred:
        return {"score": 0.0}

    # Step 3: first character
    if pred[0] in "ABCD":
        return {"score": 1.0 if pred[0] == label else 0.0}

    # Step 4: full prediction matches label letter
    if pred.upper() == label:
        return {"score": 1.0}

    # Step 5: replace punctuation, check prefixes (matching official chars)
    cleaned = pred
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        cleaned = cleaned.replace(c, " ")
    for prefix in ["answer is:", "answer:", "answer is", "option is"]:
        idx = cleaned.lower().find(prefix)
        if idx != -1:
            after = cleaned[idx + len(prefix) :].strip()
            if after and after[0].upper() in "ABCD":
                return {"score": 1.0 if after[0].upper() == label else 0.0}

    # Step 6: scan words for first A-D letter
    for word in cleaned.split():
        if word.upper() in ["A", "B", "C", "D"]:
            return {"score": 1.0 if word.upper() == label else 0.0}

    return {"score": 0.0}


def process_results_code_debug(doc: dict, results: list[str]) -> dict[str, float]:
    """Result processing for code_debug task.

    Derives the expected letter from doc["options"] since the HuggingFace
    dataset stores the answer as ["function_name"], not ["function_name", "letter"].
    The official get_answer() maps function name to letter via options index.
    Then matches using the official get_score_one_code_debug() logic.
    """
    prediction = postprocess_prediction(results[0])
    answers = doc.get("answer", [])

    if not answers:
        return {"score": 0.0}

    # Derive letter and function name from answer + options
    label_func = answers[0].strip()
    label_letter = _answer_to_letter(doc)
    if not label_letter:
        # Fallback: if answer is already a letter
        if len(label_func) == 1 and label_func.upper() in "ABCDEFGHIJ":
            label_letter = label_func.upper()
            label_func = ""
        else:
            return {"score": 0.0}

    pred = prediction.strip()

    # Step 1: find last standalone A-J letter (official regex)
    match = re.search(r"\b([A-J])\b(?!.*\b[A-J]\b)", pred)
    if match:
        if match.group(1).upper() == label_letter:
            return {"score": 1.0}
        return {"score": 0.0}

    # Step 2: replace chars and consolidate spaces (matching official)
    cleaned = pred
    for c in ["-", "*"]:
        cleaned = cleaned.replace(c, " ")
    cleaned = cleaned.replace("Option", " ")
    cleaned = cleaned.replace("option", " ")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    cleaned = cleaned.strip()

    # Step 3: check startswith
    if cleaned.upper().startswith(label_letter):
        return {"score": 1.0}
    if label_func and cleaned.startswith(label_func):
        return {"score": 1.0}

    # Step 4: check answer prefixes (matching official set)
    for prefix in ["answer is:", "is:", "answer:", "correct option is:"]:
        idx = cleaned.lower().find(prefix)
        if idx != -1:
            after = cleaned[idx + len(prefix) :].strip()
            if after:
                if after[0].upper() == label_letter:
                    return {"score": 1.0}
                if label_func and after.startswith(label_func):
                    return {"score": 1.0}
            return {"score": 0.0}

    return {"score": 0.0}


def process_results_code_run(doc: dict, results: list[str]) -> dict[str, float]:
    """Result processing for code_run.

    Extracts the last word from the prediction, casts to int, and
    checks exact numeric match. Matches the official
    get_score_one_code_run() from InfiniteBench.
    """
    prediction = postprocess_prediction(results[0])
    answers = doc.get("answer", [])

    if not answers:
        return {"score": 0.0}

    # Official: replace these chars with space, split, take last word
    cleaned = prediction
    for c in ["\n", ".", "`", "'", '"', ":"]:
        cleaned = cleaned.replace(c, " ")
    words = cleaned.split()

    if not words:
        return {"score": 0.0}

    # Official casts to int for comparison
    try:
        pred_int = int(words[-1])
    except ValueError:
        return {"score": 0.0}

    for answer in answers:
        try:
            if pred_int == int(answer.strip()):
                return {"score": 1.0}
        except ValueError:
            if words[-1] == answer.strip():
                return {"score": 1.0}

    return {"score": 0.0}


def process_results_qa_en(doc: dict, results: list[str]) -> dict[str, float]:
    """Result processing for English QA tasks using token-level F1.

    Matches the official scoring for longbook_qa_eng.
    Takes the max F1 across all acceptable answers.
    """
    prediction = postprocess_prediction(results[0])
    answers = doc.get("answer", [])

    if not answers or not prediction:
        return {"score": 0.0}

    max_f1 = max(_token_f1(prediction, ans) for ans in answers)
    return {"score": max_f1}


def process_results_qa_chn(doc: dict, results: list[str]) -> dict[str, float]:
    """Result processing for Chinese QA tasks using character-level F1.

    Matches the official scoring for longbook_qa_chn.
    Takes the max F1 across all acceptable answers.
    """
    prediction = postprocess_prediction(results[0])
    answers = doc.get("answer", [])

    if not answers or not prediction:
        return {"score": 0.0}

    max_f1 = max(_token_f1(prediction, ans, is_chinese=True) for ans in answers)
    return {"score": max_f1}


def process_results_rouge(doc: dict, results: list[str]) -> dict[str, float]:
    """Result processing for summarization using ROUGE-Lsum F1.

    Uses rougeLsum to match the official scoring.
    """
    prediction = postprocess_prediction(results[0])
    answers = doc.get("answer", [])

    if not answers or not prediction:
        return {"score": 0.0}

    reference = answers[0]

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
        score = scorer.score(reference, prediction)
        return {"score": score["rougeLsum"].fmeasure}
    except ImportError:
        eval_logger.warning(
            "rouge-score not installed. Install with: pip install rouge-score"
        )
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        if not pred_tokens or not ref_tokens:
            return {"score": 0.0}
        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"score": f1}
