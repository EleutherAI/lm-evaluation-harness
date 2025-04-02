from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


alpha = ["A", "B", "C"]
out_dic = {"ايجابي": 1, "سلبي": 0, "ماكينش إحساس": 2}


def doc_to_text(doc):
    return (
        doc["messages"][0]["content"]
        .replace("-سلبي", "A. سلبي")
        .replace("-ايجابي", "B. ايجابي")
        .replace(
            "-ماكينش إحساس",
            "C. ماكينش إحساس\nThe answer should be strictly one letter of the following: A, B, C.",
        )
    )  # .replace('شنو هو الإحساس ديال هاد الجملة؟', 'شنو هو الإحساس ديال هاد الجملة؟')


def doc_to_choice_3(doc):
    return alpha


def doc_to_choice_2(doc):
    return alpha[:2]


def doc_to_target(doc):
    return alpha[out_dic[doc["messages"][1]["content"]]]
