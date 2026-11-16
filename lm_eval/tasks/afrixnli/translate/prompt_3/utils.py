from lm_eval.utils import weighted_f1_score


def doc_to_text(doc):
    output = """You are an NLP assistant whose purpose is to solve Natural Language Inference (NLI) problems

    Please identify whether the premise entails or contradicts the hypothesis in the following premise
    and hypothesis. The answer should be exact entailment, contradiction, or neutral.

    Premise: {premise}
    Hypothesis: {hypothesis}

    Is it entailment, contradiction, or neutral?"""

    text = output.format(premise=doc["premise"], hypothesis=doc["hypothesis"])
    return text


def doc_to_target(doc):
    replacements = {0: "entailment", 1: "neutral", 2: "contradiction"}
    return replacements[doc["label"]]
