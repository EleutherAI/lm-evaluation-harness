from functools import partial
from os.path import basename, dirname

from lm_eval.tasks.mmlu_prox.lang_libs import LANG_LIBS


lang_abbr = basename(dirname(__file__))
lang_dict = LANG_LIBS[lang_abbr]

choices = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]

max_opt_num = 10


def format_cot_example(example, including_answer=True):
    prompt = f"{lang_dict[0]}\n"
    question = example["question"]
    prompt += question + "\n"
    prompt += f"{lang_dict[1]}\n"
    for i in range(max_opt_num):
        opt = example[f"option_{i}"]
        if opt is not None:
            prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace(lang_dict[4], lang_dict[2])
        prompt += cot_content + "\n\n"
    else:
        prompt += lang_dict[2]
    return prompt


doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


process_biology = partial(process_docs, subject="biology")
process_business = partial(process_docs, subject="business")
process_chemistry = partial(process_docs, subject="chemistry")
process_computer_science = partial(process_docs, subject="computer science")
process_economics = partial(process_docs, subject="economics")
process_engineering = partial(process_docs, subject="engineering")
process_health = partial(process_docs, subject="health")
process_history = partial(process_docs, subject="history")
process_law = partial(process_docs, subject="law")
process_math = partial(process_docs, subject="math")
process_other = partial(process_docs, subject="other")
process_philosophy = partial(process_docs, subject="philosophy")
process_physics = partial(process_docs, subject="physics")
process_psychology = partial(process_docs, subject="psychology")
