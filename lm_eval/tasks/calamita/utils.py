def remove_trailing_dot_question(example):
    if example["question"][-1] == ".":
        example["question"] = example["question"][:-1]
    return example
