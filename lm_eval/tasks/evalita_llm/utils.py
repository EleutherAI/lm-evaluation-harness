import logging

from evaluate import load
from sklearn.metrics import f1_score


eval_logger = logging.getLogger("lm-eval")


# ---------------------- SENTIMENT ANALYSIS ----------------------
def sa_doc_to_target(x):
    """
    Function to extract the target from the dataset for sentiment analysis
    """
    opos = x["opos"]
    oneg = x["oneg"]
    # return indexes matches the choices in sa_doc_to_choice
    if opos == "1" and oneg == "0":
        return 0
    elif opos == "0" and oneg == "1":
        return 1
    elif opos == "0" and oneg == "0":
        return 2
    elif opos == "1" and oneg == "1":
        return 3
    else:
        pass


def sa_doc_to_target_v2(x):
    """
    Function to extract the target from the dataset for sentiment analysis
    """
    opos = x["opos"]
    oneg = x["oneg"]
    # return indexes matches the choices in sa_doc_to_choice
    if opos == "1" and oneg == "0":
        return 0
    elif opos == "0" and oneg == "1":
        return 1
    elif opos == "0" and oneg == "0":
        return 2
    elif opos == "1" and oneg == "1":
        return 3
    else:
        pass


def sa_doc_to_choice(x):
    """
    Function to return the choices from the dataset for sentiment analysis
    """
    return ["Positivo", "Negativo", "Neutrale", "Misto"]


# ---------------------- LEXICAL SUBSTITUTION ----------------------
NO_SYN_STRING = "&&NOSYN&&"


def _ls_gold_to_target(x):
    """
    Generate the target for the lexical similarity task
    """
    # all_answers = [(i["word"], i["count"]) for i in x["answers"]]
    if len(x["answers"]) == 0:
        return NO_SYN_STRING
    ans_str = ""
    for i in x["answers"]:
        ans_str += i["word"] + "$$" + str(i["count"]) + "::"
    if len(ans_str) != 0 and ans_str[-2] == ":":
        ans_str = ans_str[:-2]
    # print(ans_str)

    return ans_str


def ls_doc_to_target(x):
    """
    Generate the target for the lexical similarity task
    """
    if len(x["answers"]) == 0:
        return NO_SYN_STRING
    ans_str = ""
    for i in x["answers"]:
        ans_str += i["word"] + ", "
    if len(ans_str) != 0 and ans_str[-2] == ",":
        ans_str = ans_str[:-2]
    return ans_str


def _ls_split_gold(x):
    """
    Split the gold string into a list of tuples
    """
    if x == NO_SYN_STRING:
        return [], []
    answers = x.split("::")
    words = []
    freqs = []
    if len(answers) != 0:
        for a in answers:
            if "$$" in a:
                word, count = a.split("$$")
                words.append(word)
                try:
                    freqs.append(int(count))
                except ValueError:
                    freqs.append(0)
    return words, freqs


def ls_process_results(doc, results):
    """
    Process the results of the evaluation for the lexical substitution task
    look at coqa for another example
    """
    gold_to_target = _ls_gold_to_target(doc)
    words, freqs = _ls_split_gold(gold_to_target)
    prec = 0

    # Considering a maximum of the first 10 synonyms
    results = split_text_with_regex(results[0], LS_SPLIT_REGEX)
    results = results[: min(10, len(results))]

    # Remove non-alphabetic characters from the word at the end of the list
    if results:  # Check if results is not empty
        results[-1] = "".join(char for char in results[-1] if char.isalpha())

    has_answ = 0 if len(results) == 0 else 1  # so we can compute |A|
    has_annotation = 0 if len(words) == 0 else 1  # so we can compute |T|

    matching_res = []  # for debugging

    for r in results:
        if r in words:
            # get frequency of the synonyms from annotators
            idx = words.index(r.strip())
            prec += freqs[idx]
            matching_res.append(r)

    # In the case of the OOT (out of ten) subtask, this normalization should not be applied
    # ai = len(results) if len(results) != 0 else 1
    # prec = prec / ai

    Hi = sum(freqs)
    if Hi != 0:
        prec = prec / Hi
    else:
        eval_logger.debug("H_i is 0")

    return {"f1": (prec, has_answ, has_annotation)}


# ---------------------- NER ----------------------

NO_ENT_STRING = "&&NOENT&&"
NER_ENTITY_SEPARATOR = ","
NER_TYPE_SEPARATOR = "$"
NER_MAPPING_V2 = {"PER": 0, "LOC": 1, "ORG": 2, NO_ENT_STRING: 3, "O": 4}
NER_MAPPING = {"PER": 0, "LOC": 1, "ORG": 2, "O": 3}


def _ner_gold_to_target(x: list) -> list:
    """
    Convert the gold entities to the target format according to the NER_MAPPING
    """
    res = [NER_MAPPING[e["type"]] for e in x]
    return res


def _ner_gold_to_target_v2(x: list) -> list:
    """
    Convert the gold entities to the target format according to the NER_MAPPING
    """
    res = [NER_MAPPING[e["type"]] for e in x]
    return res


def ner_doc_to_target(doc):
    ents = doc["entities"]
    targ_str = ""
    # Entità$Tipo%Entità$Tipo.
    if ents == []:
        return NO_ENT_STRING
    else:
        for e in ents:
            targ_str += (
                e["entity_text"] + NER_TYPE_SEPARATOR + e["type"] + NER_ENTITY_SEPARATOR
            )
    return targ_str[:-1]


def ner_process_results(doc, results):
    """
    Process the results of the Named Entity Recognition task
    """
    # each document has a list of entities with the following format:
    # [{"entity_text": "string", "type": "string"}]
    gold = doc["entities"]
    raw_results = results[0]
    results = _ner_process_raw_output(raw_results)

    gold_labels = _ner_gold_to_target(gold)
    res_labels = [0] * len(gold_labels)
    matched_gold_idx = []

    if len(results) > len(gold):
        for r in results:
            r_text = r[0]
            r_type = r[1]
            for i in range(len(gold)):
                if r_text == gold[i]["entity_text"] and r_type == gold[i]["type"]:
                    res_labels[i] = NER_MAPPING[r_type]
                    matched_gold_idx.append(i)
        # Since we have more results than gold, we artificially set to false positive the remaining labels
        # extend gold label list
        for i in range(len(results) - len(gold)):
            gold_labels.append(3)
            res_labels.append(2)
    elif len(results) == 0 and len(gold) == 0:
        res_labels = [3]
        gold_labels = res_labels
    else:  # len(results) <= len(gold)
        for r in results:
            r_text = r[0]
            r_type = r[1]
            for i in range(len(gold)):
                if r_text == gold[i]["entity_text"] and r_type == gold[i]["type"]:
                    res_labels[i] = NER_MAPPING[r_type]
                    matched_gold_idx.append(i)
        # we map all wrong predictions to the "O" class
        for i in range(len(gold_labels)):
            if i in matched_gold_idx:
                continue
            if gold_labels[i] == 1:
                res_labels[i] = 3
            elif gold_labels[i] == 0:
                res_labels[i] = 3
            else:
                res_labels[i] = 3

    assert len(gold_labels) == len(res_labels)
    return {"f1": (res_labels, gold_labels)}


def ner_process_results_v2(doc, results):
    """
    Process the results of the Named Entity Recognition task
    This version considers and score explicitly when the model responds that there are no entities
    """
    # each document has a list of entities with the following format:
    # [{"entity_text": "string", "type": "string"}]
    gold = doc["entities"]
    raw_results = results[0]
    results = _ner_process_raw_output_v2(raw_results)

    # eval_logger.debug(f"results {results}")
    # eval_logger.debug(f"gold {gold}")

    gold_labels = _ner_gold_to_target_v2(gold)
    res_labels = [0] * len(gold_labels)
    matched_gold_idx = []

    if len(results) > len(gold):
        for r in results:
            # print(r)
            r_text = r[0]
            r_type = r[1]
            for i in range(len(gold)):
                if r_text == gold[i]["entity_text"] and r_type == gold[i]["type"]:
                    res_labels[i] = NER_MAPPING[r_type]
                    matched_gold_idx.append(i)
        # Since we have more results than gold, we artificially set to false positive the remaining labels
        # extend gold label list
        for i in range(len(results) - len(gold)):
            # gold_labels.append(3)
            # res_labels.append(2)
            gold_labels.append(4)
            res_labels.append(3)
    elif len(results) == 0 and len(gold) == 0:
        # res_labels = [random.choice([0, 1, 2, 3])]
        res_labels = [3]
        gold_labels = res_labels
    elif len(results) == 1 and results[0] == NO_ENT_STRING:
        # res_labels = [3]
        res_labels = [4]
        gold_labels = res_labels
    else:  # len(results) <= len(gold)
        for r in results:
            r_text = r[0]
            r_type = r[1]
            for i in range(len(gold)):
                if r_text == gold[i]["entity_text"] and r_type == gold[i]["type"]:
                    res_labels[i] = NER_MAPPING[r_type]
                    matched_gold_idx.append(i)
        # we map all wrong predictions to the "O" class
        for i in range(len(gold_labels)):
            if i in matched_gold_idx:
                continue
            if gold_labels[i] == 1:
                # res_labels[i] = 2
                res_labels[i] = 4
            elif gold_labels[i] == 0:
                # res_labels[i] = 1
                res_labels[i] = 4
            else:
                res_labels[i] = 4

    assert len(gold_labels) == len(res_labels)
    return {"f1": (res_labels, gold_labels)}


def _ner_process_raw_output(llm_result: str) -> list[tuple]:
    if NO_ENT_STRING in llm_result:
        return []
    if llm_result == "":
        return ["WRONG"]
    tmp_results = llm_result.split(NER_ENTITY_SEPARATOR)
    results = []
    for res in tmp_results:
        r = res.strip()
        # split on type separator
        r_text = ""
        r_type = ""
        r_splitted = r.split(NER_TYPE_SEPARATOR)
        if len(r_splitted) < 2:
            r_text = r_splitted[0]
            r_type = ""
        else:
            r_text = r_splitted[0]
            r_type = r_splitted[1]
        if r_text != "":
            results.append((r_text, r_type.upper()))
    return results


def _ner_process_raw_output_v2(llm_result: str) -> list[tuple]:
    if NO_ENT_STRING in llm_result:
        return [NO_ENT_STRING]
    if llm_result == "":
        return ["WRONG"]
    tmp_results = llm_result.split(NER_ENTITY_SEPARATOR)
    results = []
    for res in tmp_results:
        r = res.strip()
        # split on type separator
        r_text = ""
        r_type = ""
        r_splitted = r.split(NER_TYPE_SEPARATOR)
        if len(r_splitted) < 2:
            r_text = r_splitted[0]
            r_type = ""
        else:
            r_text = r_splitted[0]
            r_type = r_splitted[1]
        if r_text != "":
            results.append((r_text, r_type.upper()))
    return results


# ---------------------- RELATION EXTRACTION ----------------------


def _rel_process_raw_output(llm_result: str) -> list[str]:
    if NO_REL_STRING in llm_result:
        return []
    if llm_result == "":
        return ["WRONG"]
    tmp_results = llm_result.split(INTER_REL_SEPARATOR)
    relations = []
    for res in tmp_results:
        r_text1 = ""
        r_text2 = ""
        r_splitted = res.split(INTRA_REL_SEPARATOR)
        if len(r_splitted) < 2:
            r_text1 = r_splitted[0].strip()
            r_text2 = ""
        else:
            r_text1 = r_splitted[0].strip()
            r_text2 = r_splitted[1].strip()
        relations.append((r_text1, r_text2))
    assert len(relations) == len(tmp_results)
    return relations


INTER_REL_SEPARATOR = "%"
INTRA_REL_SEPARATOR = "$"
NO_REL_STRING = "&&NOREL&&"


def re_doc_to_target(doc):
    ents = doc["relations"]
    targ_str = ""
    # Entità$Tipo%Entità$Tipo.
    if ents == []:
        return NO_ENT_STRING
    else:
        for e in ents:
            targ_str += e[0] + INTRA_REL_SEPARATOR + e[1] + INTER_REL_SEPARATOR
    return targ_str[:-1]


def _rel_gold_to_target(x: list) -> list:
    if x == []:
        return [0]
    else:
        return [1] * len(x)


def rel_doc_to_target(doc):
    rel = doc["relations"]
    targ_str = ""
    # misura1$result1%misure2$result2.
    if rel == []:
        return NO_REL_STRING
    else:
        for r in rel:
            targ_str += r[0] + "$" + r[1] + "%"
    return targ_str[:-1]


def _extract_relations(results):
    relations = []
    for r in results:
        r_text1 = ""
        r_text2 = ""
        r_splitted = r.split(INTRA_REL_SEPARATOR)
        if len(r_splitted) < 2:
            r_text1 = r_splitted[0]
            r_text2 = ""
        else:
            r_text1 = r_splitted[0]
            r_text2 = r_splitted[1]
        relations.append((r_text1, r_text2))
    assert len(relations) == len(results)
    return relations


def rel_process_results_v3(doc, results):
    """
    Process the results of the Relation extraction task not considering the order of the relation extracted
    """
    # each document has a list of relation with the following format:
    # [[text1, text2], [text3, text4]]
    gold = doc["relations"]
    raw_results = results[0]
    has_results = 0 if NO_REL_STRING in raw_results else 1
    has_gold = 1 if gold != [] else 0

    res_labels = []
    gold_labels = []

    if has_results == 0 and has_gold:
        # False negative
        gold_labels = _rel_gold_to_target(gold)
        res_labels = [0] * len(gold_labels)
    elif has_results == 0 and has_gold == 0:
        # True negative
        gold_labels = _rel_gold_to_target(gold)
        res_labels = gold_labels
    elif has_results and has_gold == 0:
        # False positive
        gold_labels = _rel_gold_to_target(gold)
        res_labels = [1] * len(gold_labels)
    else:
        results = _rel_process_raw_output(raw_results)
        # results = raw_results.split(INTER_REL_SEPARATOR)
        gold_labels = _rel_gold_to_target(gold)
        res_labels = [0] * len(gold_labels)
        assert len(gold) > 0
        for i in range(len(gold)):
            for j in range(len(results)):
                r_text1 = results[j][0]
                r_text2 = results[j][1]

                if r_text1 == gold[i][0] and r_text2 == gold[i][1]:  # list of lists
                    res_labels[i] = 1
                    results[j] = ("DELETED", "DELETED")
                elif r_text1 == "DELETED" and r_text2 == "DELETED":
                    continue
                else:
                    pass
        # if there are more predictions than gold, we set the remaining predictions to false positive
        if len(results) - len(gold) > 0:
            for i in range(len(results) - len(gold)):
                if results[i] == ("DELETED", "DELETED"):
                    continue
                res_labels.append(1)
                gold_labels.append(0)

    assert len(gold_labels) == len(res_labels)
    return {"f1": (res_labels, gold_labels)}


LS_SPLIT_REGEX = r"[^,]+"


def split_text_with_regex(text, pattern):
    """
    pattern: str - a regex pattern to match the text
    text: str - the text to split
    """
    import re

    # Get text with model-generated words for comparison with the gold standard
    text = text.split("\n")[0]

    # Find all matches for the pattern
    matches = re.findall(pattern, text)
    # Split each matched segment further if it contains a comma and is quoted
    result = []
    for match in matches:
        if match.startswith('"') and match.endswith('"'):
            # Remove the quotes and split inside the quoted string
            inner_matches = re.findall(r"[^,]+", match[1:-1])
            result.extend(inner_matches)
        else:
            result.append(match)

    # Strip leading and trailing whitespaces from each element
    result = [element.strip().replace('"', "") for element in result]

    return result


def faq_doc_to_target(x):
    if x["correct_answer"] == "A":
        return 0
    elif x["correct_answer"] == "B":
        return 1
    elif x["correct_answer"] == "C":
        return 2
    elif x["correct_answer"] == "D":
        return 3
    else:
        eval_logger.warning(
            'WARNING: correct answer not found or not in ["A", "B", "C", "D"]'
        )


def ht_doc_to_target(x):
    if x["source"] == "ilgiornale":
        return 0
    elif x["source"] == "repubblica":
        return 1
    else:
        eval_logger.warning(
            'WARNING: source not found or not in ["ilgiornale", "repubblica"]'
        )
