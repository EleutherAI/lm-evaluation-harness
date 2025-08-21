from functools import partial

choices = [
    "A",
    "B",
    "C",
    "D",
]

def format_cot_example(example):
    prompt = "Question:\n"
    question = example["question"]
    options = example["choices"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
        
    # prompt+="\nAnswer: "
    prompt += "Answer: Let's think step by step."
    return prompt


def answering(example):
    answer = example["answer"]
    if answer == 0:
        return "A"
    elif answer == 1:
        return "B"
    elif answer == 2:
        return "C"
    elif answer == 3:
        return "D"
    else:
        raise ValueError(f"Invalid answer: {answer}")


doc_to_text = partial(format_cot_example)
doc_to_target = partial(answering)
