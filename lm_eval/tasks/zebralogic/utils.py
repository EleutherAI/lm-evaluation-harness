from functools import partial

choices = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
]

def format_cot_example(example):
    prompt = f"Puzzle: {example['puzzle']}\n\n"
    options = example["choices"]
    prompt += f"Question: {example['question']}\n\n"
    prompt += "Options: "
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
        
    prompt+="\n\nAnswer: "
    # prompt += "Answer: Let's think step by step."
    return prompt



def answering(example):
    answer = example["choices"].index(example["answer"])
     
    if answer == 0:
        return "A"
    elif answer == 1:
        return "B"
    elif answer == 2:
        return "C"
    elif answer == 3:
        return "D"
    elif answer == 4:
        return "E"
    elif answer == 5:
        return "F"
    else:
        raise ValueError(f"Invalid answer: {answer}")


doc_to_text = partial(format_cot_example)
doc_to_target = partial(answering)
