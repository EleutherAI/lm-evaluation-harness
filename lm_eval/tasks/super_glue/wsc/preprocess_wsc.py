import re
from lm_eval.utils import general_detokenize


def t5_prompt_doc_to_text(x):
    def _mark_span(text, span_str, span_idx, mark):
        pattern_tmpl = r"^((?:\S+\s){N})(W)"
        pattern = re.sub("N", str(span_idx), pattern_tmpl)
        pattern = re.sub("W", span_str, pattern)
        return re.sub(pattern, r"\1{0}\2{0}".format(mark), text)

    text = x["text"]
    text = _mark_span(text, x["span2_text"], x["span2_index"], "*")

    return "wsc: "+text


def default_doc_to_text(x):
    raw_passage = x["text"]
    # NOTE: HuggingFace span indices are word-based not character-based.
    pre = " ".join(raw_passage.split()[: x["span2_index"]])
    post = raw_passage[len(pre) + len(x["span2_text"]) + 1 :]
    passage = general_detokenize(pre + " *{}*".format(x["span2_text"]) + post)
    noun = x["span1_text"]
    pronoun = x["span2_text"]
    text = (
        f"Passage: {passage}\n"
        + f'Question: In the passage above, does the pronoun "*{pronoun}*" refer to "*{noun}*"?\n'
        + "Answer:"
    )
    return text
