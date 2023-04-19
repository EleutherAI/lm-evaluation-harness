# TODO: decide whether we want jinja2 or f-string prompts. would it be cursed to support both?
# Prompt library. 
# Stores prompts in a dictionary indexed by 2 levels:
# prompt category name, and prompt name.
# This allows us to access prompts
PROMPT_REGISTRY = {
    "qa-basic": {
        "question-newline-answer": "Question: {{question}}\nAnswer:",
        "q-newline-a": "Q: {question}\nA:"
    },
}

def get_prompt(prompt_id: str):
    # unpack prompt name 
    try:
        category_name, prompt_name = prompt_id.split(":")
    except:
        raise ValueError(
            f"expected only a single `:` as separator between \
prompt category and name, but got `{prompt_id}` instead"
            )
    return PROMPT_REGISTRY[category_name][prompt_name]