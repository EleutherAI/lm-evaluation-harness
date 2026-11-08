import collections
import re
import string


def process_results_coqcat(doc, results):
    # Get all possible answers and compute the scores
    turn_id = len(doc["questions"])
    answers = [doc["answers"]["input_text"][turn_id - 1]]
    additional_answers_list = doc.get("additional_answers")
    if additional_answers_list:
        for additional_answers in additional_answers_list.values():
            if additional_answers["input_text"][turn_id - 1].lower() not in map(
                str.lower, answers
            ):
                answers.append(additional_answers["input_text"][turn_id - 1])

    gold_list = answers
    pred = results[0].strip().split("\n")[0]

    f1_sum = 0.0
    em_sum = 0.0
    if len(gold_list) > 1:
        for i in range(len(gold_list)):
            gold_answers = gold_list[0:i] + gold_list[i + 1 :]
            # predictions compared against (n) golds and take maximum
            em_sum += max(compute_exact(a, pred) for a in gold_answers)
            f1_sum += max(compute_f1(a, pred) for a in gold_answers)
    else:
        em_sum += max(compute_exact(a, pred) for a in gold_list)
        f1_sum += max(compute_f1(a, pred) for a in gold_list)
    return {
        "em": em_sum / max(1, len(gold_list)),
        "f1": f1_sum / max(1, len(gold_list)),
    }


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
