import csv
import re
from pathlib import Path

import datasets


QUESTION_TYPE_GROUPS = {
    "reality": "reality",
    "memory": "anti_reality",
    "assumption": "anti_reality",
    "1sta": "first_order_a",
    "1stb": "first_order_b",
    "2nda": "second_order_a",
    "2ndb": "second_order_b",
}
BELIEF_GROUP_ORDER = [
    "reality",
    "anti_reality",
    "first_order_a",
    "first_order_b",
    "second_order_a",
    "second_order_b",
]
TF_LABEL_PATTERN = r"(?:^|\n)\s*{label}\s*[\.:\)-]?\s*(.*?)(?=(?:\n\s*[AB]\s*[\.:\)-]?)|\Z)"


def _benchmark_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'benchmarks/tomchallenges'
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        'ToMChallenges benchmark directory not found under any parent benchmarks/ path.'
    )


def _normalize(text: str) -> str:
    text = text.replace("’", "'").lower()
    text = re.sub(r"[^a-z0-9\s]", ' ', text)
    return ' '.join(text.split())


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    text_norm = f" {_normalize(text)} "
    phrase_norm = f" {_normalize(phrase)} "
    return phrase_norm in text_norm


def _parse_mc_choices(prompt: str) -> list[str]:
    match = re.search(r"\nA\.\s*(.*?)\nB\.\s*(.*?)\n+Answer:?\s*$", prompt, re.S)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]
    lines = prompt.splitlines()
    choices = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('A.'):
            choices['A'] = stripped.split('.', 1)[1].strip()
        elif stripped.startswith('B.'):
            choices['B'] = stripped.split('.', 1)[1].strip()
    if 'A' in choices and 'B' in choices:
        return [choices['A'], choices['B']]
    raise ValueError(f'Could not parse A/B choices from ToMChallenges prompt: {prompt}')


def _parse_tf_statements(prompt: str) -> list[str]:
    match = re.search(
        r"(?:Statements|Statments):\s*A\.\s*(.*?)\nB\.\s*(.*?)(?:\n(?:Use this format|In the answer)|\Z)",
        prompt,
        re.S,
    )
    if match:
        return [match.group(1).strip(), match.group(2).strip()]
    lines = prompt.splitlines()
    statements = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('A.'):
            statements['A'] = stripped.split('.', 1)[1].strip()
        elif stripped.startswith('B.'):
            statements['B'] = stripped.split('.', 1)[1].strip()
    if 'A' in statements and 'B' in statements:
        return [statements['A'], statements['B']]
    raise ValueError(f'Could not parse A/B statements from ToMChallenges prompt: {prompt}')


def _truth_for_statement(statement: str, gold_answer: str) -> bool:
    return _contains_phrase(statement, gold_answer)


def _parse_row(row: dict[str, str], story_family: str) -> dict[str, object]:
    row = {k: v for k, v in row.items() if k != ''}
    context = row.get('story') or row.get('prompt')
    if not context:
        raise ValueError(f'ToMChallenges row is missing both story and prompt: {row}')
    short_answer = row['short_answer'].strip()
    mc_choices = _parse_mc_choices(row['mc_prompt'])
    normalized_choices = [_normalize(choice) for choice in mc_choices]
    gold_normalized = _normalize(short_answer)
    if gold_normalized not in normalized_choices:
        raise ValueError(
            f"Gold answer {short_answer!r} does not match parsed MC choices {mc_choices!r}"
        )
    mc_gold_index = normalized_choices.index(gold_normalized)
    tf_statements = _parse_tf_statements(row['tf_prompt'])
    tf_gold_pair = [
        _truth_for_statement(tf_statements[0], short_answer),
        _truth_for_statement(tf_statements[1], short_answer),
    ]
    if tf_gold_pair.count(True) != 1:
        raise ValueError(
            f'Expected exactly one true TF statement, got {tf_gold_pair} for row {row}'
        )
    question_type_raw = row['question_type'].strip()
    question_type_group = QUESTION_TYPE_GROUPS[_normalize(question_type_raw)]
    return {
        **row,
        'context': context,
        'story_family': story_family,
        'short_answer_norm': gold_normalized,
        'question_type_group': question_type_group,
        'mc_choices': mc_choices,
        'mc_gold_index': mc_gold_index,
        'mc_gold_label': 'A' if mc_gold_index == 0 else 'B',
        'tf_statement_a': tf_statements[0],
        'tf_statement_b': tf_statements[1],
        'tf_gold_a': tf_gold_pair[0],
        'tf_gold_b': tf_gold_pair[1],
    }


def load(**kwargs):
    data_dir = _benchmark_dir() / 'Data'
    rows = []
    for story_family, filename in (
        ('sally_anne', 'Sally-Anne_prompt.csv'),
        ('smarties', 'Smarties_prompt.csv'),
    ):
        path = data_dir / filename
        with path.open(encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            rows.extend(_parse_row(row, story_family) for row in reader)
    return {'test': datasets.Dataset.from_list(rows)}


def _filter_by_belief_group(dataset, belief_group: str):
    return dataset.filter(lambda doc: doc['question_type_group'] == belief_group)


def process_docs_reality(dataset):
    return _filter_by_belief_group(dataset, 'reality')


def process_docs_anti_reality(dataset):
    return _filter_by_belief_group(dataset, 'anti_reality')


def process_docs_first_order_a(dataset):
    return _filter_by_belief_group(dataset, 'first_order_a')


def process_docs_first_order_b(dataset):
    return _filter_by_belief_group(dataset, 'first_order_b')


def process_docs_second_order_a(dataset):
    return _filter_by_belief_group(dataset, 'second_order_a')


def process_docs_second_order_b(dataset):
    return _filter_by_belief_group(dataset, 'second_order_b')


def doc_to_text_qa(doc):
    return doc['qa_prompt']


def doc_to_text_comp(doc):
    return doc['comp_prompt']


def doc_to_text_mc(doc):
    return doc['mc_prompt']


def doc_to_text_fb(doc):
    return doc['fb_prompt']


def doc_to_text_tf(doc):
    return doc['tf_prompt']


def doc_to_text_tfr(doc):
    return doc['tfr_prompt']


def _score_open_response(doc, response: str) -> float:
    return 1.0 if _contains_phrase(response, doc['short_answer']) else 0.0


def _extract_mc_prediction(response: str, doc) -> int | None:
    letter_match = re.search(r"(?<![A-Za-z])([AB])(?![A-Za-z])", response, re.I)
    if letter_match:
        return 0 if letter_match.group(1).upper() == 'A' else 1
    hits = []
    lowered = response.lower()
    for index, choice in enumerate(doc['mc_choices']):
        pos = lowered.find(choice.lower())
        if pos != -1:
            hits.append((pos, index))
    if hits:
        hits.sort()
        return hits[0][1]
    return None


def _leading_bool_in_chunk(chunk: str) -> bool | None:
    match = re.match(r"\s*(true|false|yes|no)\b", chunk, re.I)
    if not match:
        return None
    return match.group(1).lower() in {'true', 'yes'}


def _last_bool_in_chunk(chunk: str) -> bool | None:
    matches = re.findall(r"\b(true|false|yes|no)\b", chunk, re.I)
    if not matches:
        return None
    token = matches[-1].lower()
    return token in {'true', 'yes'}


def _bool_from_chunk(chunk: str) -> bool | None:
    leading = _leading_bool_in_chunk(chunk)
    if leading is not None:
        return leading
    return _last_bool_in_chunk(chunk)


def _extract_tf_pair(response: str) -> tuple[bool | None, bool | None]:
    values = []
    for label in ('A', 'B'):
        match = re.search(TF_LABEL_PATTERN.format(label=label), response, re.I | re.S)
        values.append(_bool_from_chunk(match.group(1)) if match else None)
    if all(value is not None for value in values):
        return values[0], values[1]
    fallback = re.findall(r"\b(true|false|yes|no)\b", response, re.I)
    if len(fallback) >= 2:
        bools = [token.lower() in {'true', 'yes'} for token in fallback[:2]]
        return bools[0], bools[1]
    return None, None


def process_results_open(doc, results):
    return {'acc': _score_open_response(doc, results[0])}


def process_results_mc(doc, results):
    pred = _extract_mc_prediction(results[0], doc)
    return {'acc': 1.0 if pred == doc['mc_gold_index'] else 0.0}


def process_results_tf(doc, results):
    pred_a, pred_b = _extract_tf_pair(results[0])
    gold = (bool(doc['tf_gold_a']), bool(doc['tf_gold_b']))
    return {'acc': 1.0 if (pred_a, pred_b) == gold else 0.0}
