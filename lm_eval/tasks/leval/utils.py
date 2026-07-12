"""Helpers for the L-Eval closed-ended (exam / exact-match) tasks.

L-Eval: Instituting Standardized Evaluation for Long Context Language Models.
Chenxin An et al., 2023 — https://arxiv.org/abs/2307.11088
Benchmark: https://github.com/OpenLMLab/LEval
Data:      the benchmark's LEval-data/Closed-ended-tasks/*.jsonl (pinned to a commit;
           the L4NLP/LEval HF dataset ships a loading script that datasets>=4 rejects)

Only the seven closed-ended subtasks are covered here (coursera, quality, tpo,
gsm100, codeU, sci_fi, topic_retrieval_longchat). They are scored with exact
match and need no judge model, mirroring the way LongBench is already integrated.
The thirteen open-ended subtasks (LLM-as-judge / n-gram) are left for a follow-up.

The task *instructions* below are reproduced verbatim from the benchmark so that
scores stay comparable with the paper. The answer extraction and exact-match
scoring are an independent re-implementation of the behaviour of LEval's
``Evaluation/auto_eval.py`` exam path; no code is copied from that (GPL-3.0)
project into this MIT-licensed repository.
"""

from __future__ import annotations

import re
import string

import datasets


# --------------------------------------------------------------------------- #
# System instructions (reproduced from the benchmark for faithful prompting)
# --------------------------------------------------------------------------- #
_PROMPT_COURSERA = (
    "Now you are given a very long document. Please follow the instruction based "
    "on this document. For multi-choice questions, there could be a single correct "
    "option or multiple correct options. Please only provide the letter "
    "corresponding to the answer (like A or AB) when answering. "
)
_PROMPT_MC_SINGLE = (
    "Now you are given a very long document. Please follow the instruction based "
    "on this document. For multi-choice questions, there is only a single correct "
    "option. Please only provide the letter corresponding to the answer (like A or "
    "B) when answering. For other questions, please directly give the concise and "
    "accurate answer. "
)
_PROMPT_GSM = (
    "Given several question answer pairs, you need to follow a similar format to "
    "answer the last question. Make sure the response is end with The answer is _ . "
)
_PROMPT_CODEU = (
    "Now you are given a code base consisting of a large amount of functions and "
    "the corresponding comments. In the end, I will call some functions defined in "
    "the code base. Please carefully read these codes and comments and answer the "
    "question."
)
_PROMPT_SCIFI = (
    "Now you are given a scientific fiction. I will ask you some questions and the "
    'answer should be "True" or "False". Notice that you should answer the question '
    "based on the evidence in the document instead of your background knowledge."
)
_PROMPT_TOPIC = (
    "Below is a record of our previous conversation on many different topics. You "
    "are the ASSISTANT, and I am the USER. At the beginning of each topic, the USER "
    "will say 'I would like to discuss the topic of <TOPIC>'. Memorize each <TOPIC>. "
    "At the end of the record, I will ask you to retrieve the first/second/third "
    "topic names. Now the record start. "
)

# Wrapper used for every closed-ended task except gsm100 / codeU, which append the
# question directly after the document.
_QUESTION_TEMPLATE = (
    "Document is as follows. {document} Question: {instruction} \n"
    "Please directly give answer without any additional output or explanation\n"
    " Answer: "
)


def load_leval(**kwargs):
    """Load a closed-ended L-Eval split from its pinned upstream jsonl.

    Wired through ``custom_dataset`` (rather than ``dataset_path: json``) so that
    ``download()`` still finds ``data_files`` when it is called without arguments,
    as the task test-suite does. ``data_files`` comes from the task's
    ``dataset_kwargs``; returns a ``DatasetDict`` keyed by the split name.
    """
    return datasets.load_dataset("json", data_files=kwargs.get("data_files"))


def _build(dataset, system_prompt, *, raw_concat=False, scifi=False):
    """Explode each row (many questions per document) into one doc per question."""
    rows = []
    for doc in dataset:
        document = doc["input"]
        for instruction, answer in zip(
            doc["instructions"], doc["outputs"], strict=True
        ):
            if raw_concat:
                body = document + "\n\n" + instruction
            else:
                body = _QUESTION_TEMPLATE.format(
                    document=document, instruction=instruction
                )
            if scifi:
                # gold looks like 'True [fact: False]'; keep the in-document verdict.
                answer = answer.split("[fact:")[0].strip()
            # rstrip so doc_to_text does not end in whitespace (harness convention).
            text = (system_prompt + "\n" + body).rstrip()
            rows.append({"text": text, "gold": answer})
    return datasets.Dataset.from_list(rows)


def process_docs_coursera(dataset):
    return _build(dataset, _PROMPT_COURSERA)


def process_docs_quality(dataset):
    return _build(dataset, _PROMPT_MC_SINGLE)


def process_docs_tpo(dataset):
    return _build(dataset, _PROMPT_MC_SINGLE)


def process_docs_gsm100(dataset):
    return _build(dataset, _PROMPT_GSM, raw_concat=True)


def process_docs_codeu(dataset):
    return _build(dataset, _PROMPT_CODEU, raw_concat=True)


def process_docs_sci_fi(dataset):
    return _build(dataset, _PROMPT_SCIFI, scifi=True)


def process_docs_topic(dataset):
    return _build(dataset, _PROMPT_TOPIC)


# --------------------------------------------------------------------------- #
# Answer normalisation and exact match (re-implemented from the paper's rules)
# --------------------------------------------------------------------------- #
_PUNCT = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _normalize(text: str) -> str:
    """Drop punctuation and articles, collapse whitespace (case preserved)."""
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 / 0.25 / 0.0 following LEval's exam scoring.

    - Multiple-choice gold (only letters A-D): 1.0 on an exact set match, 0.25 when
      the predicted letters are a subset of the gold letters (partial credit for
      coursera's multi-answer questions).
    - Numeric gold (gsm100): 1.0 when the two values are numerically equal.
    - Everything else (codeU / sci_fi / topic): 1.0 when the whitespace-stripped
      gold string is contained in the prediction (case-insensitive).
    """
    gold_norm = _normalize(ground_truth)
    letters_only = gold_norm != "" and all(ch in "ABCD" for ch in gold_norm)

    if letters_only:
        pred_norm = _normalize(prediction)
        if pred_norm == gold_norm:
            return 1.0
        if set(pred_norm) <= set(gold_norm):
            return 0.25
        return 0.0

    try:
        return 1.0 if float(prediction) == float(ground_truth) else 0.0
    except ValueError:
        needle = ground_truth.lower().replace(" ", "")
        haystack = prediction.lower().replace(" ", "")
        return 1.0 if needle in haystack else 0.0


def _extract_letters(response: str, *, multi: bool) -> str:
    """Pull the option letter(s) out of a multiple-choice generation."""
    response = response.strip()
    if not response:
        return "None"
    if not multi:
        for ch in response:
            if ch in "ABCD":
                return ch
        return "None"

    # multi-answer (coursera): collect the leading run of option letters, else scan
    # for isolated "A." / "B)" style options.
    leading = ""
    for ch in response:
        if ch in "ABCD":
            leading += ch
        else:
            break
    if len(leading) > 1:
        return "".join(sorted(set(leading)))

    head = response.split("Question")[0]
    options = re.findall(r"\s*[A-Z](?=[\s.)])", head)
    letters = re.sub(r"[^A-D]", "", leading + "".join(options))
    letters = "".join(sorted(set(letters)))
    if not letters:
        for ch in head:
            if ch in "ABCD":
                letters += ch
    return letters or "A"


def _extract_number(response: str) -> str:
    """Grab the final numeric answer from a gsm100 generation."""
    match = re.search(r"The answer is (\S+)", response)
    token = match.group(1) if match else ""
    if not token:
        for word in reversed(response.split("\n\n")[0].split(" ")):
            if any(ch.isdigit() for ch in word):
                token = word
                break
    digits = ""
    for ch in token:
        if ch.isdigit():
            digits += ch
        elif ch == ".":
            break
    return digits


def _clean_code(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    for old, new in (
        (",", ""),
        ("'", ""),
        ("\\", ""),
        (".0", ""),
        ("] [", "]["),
        ("[ [", "[["),
        ("] ]", "]]"),
    ):
        text = text.replace(old, new)
    return text


def _extract_code_output(response: str, gold: str) -> str:
    """Trim a codeU generation down to the reported output tokens."""
    span = len(gold.split()) + 3
    text = _clean_code(response)
    for phrase in (
        "will be",
        "of the code",
        "is",
        "would be",
        "the value of",
        "the result of",
        "printed",
    ):
        text = text.replace(phrase, "")
    if "the final output" in text:
        tokens = re.split(r"\s+", text.split("the final output")[-1])[:span]
    else:
        tokens = re.split(r"\s+", text)[-span:]
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# process_results entry points (one per task)
# --------------------------------------------------------------------------- #
def _result(score: float) -> dict:
    return {"exact_match": score}


def _gold_letters(gold: str) -> str:
    """Reduce a gold answer to its option letters (LEval's process_gt_mc)."""
    first = gold.split()[0] if gold.split() else ""
    return re.sub(r"[^A-D]", "", first) or "A"


def process_results_coursera(doc, results):
    pred = _extract_letters(results[0], multi=True)
    return _result(_exact_match(pred, _gold_letters(doc["gold"])))


def _process_results_mc_single(doc, results):
    pred = _extract_letters(results[0], multi=False)
    return _result(_exact_match(pred, _gold_letters(doc["gold"])))


def process_results_quality(doc, results):
    return _process_results_mc_single(doc, results)


def process_results_tpo(doc, results):
    return _process_results_mc_single(doc, results)


def process_results_gsm100(doc, results):
    pred = _extract_number(results[0])
    gold = _extract_number(doc["gold"]) or doc["gold"].strip()
    return _result(_exact_match(pred, gold))


def process_results_codeu(doc, results):
    pred = _extract_code_output(results[0], doc["gold"])
    gold = _clean_code(doc["gold"])
    return _result(_exact_match(pred, gold))


def process_results_sci_fi(doc, results):
    return _result(_exact_match(results[0], doc["gold"]))


def process_results_topic(doc, results):
    return _result(_exact_match(results[0], doc["gold"]))
