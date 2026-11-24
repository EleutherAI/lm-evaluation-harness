import datasets


def fewshot_samples() -> list[dict]:
    # GSM-Plus perturbs GSM8K test questions, so draw examples from GSM8K train.
    return list(datasets.load_dataset("openai/gsm8k", "main", split="train"))
