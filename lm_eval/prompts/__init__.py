# TODO: decide whether we want jinja2 or f-string prompts. would it be cursed to support both?
# Prompt library. 
# Stores prompts in a dictionary indexed by 2 levels:
# prompt category name, and prompt name.
# This allows us to access prompts
PROMPT_REGISTRY = {
    "qa-basic": {
        "question-newline-answer": "Question: {{question}}\nAnswer:",
        "q-newline-a": "Q: {{question}}\nA:"
    },
}

def get_prompt(prompt_id: str, dataset_name=None, dataset_path=None):
    # unpack prompt name 
        category_name, prompt_name = prompt_id.split(":")
        if category_name == "promptsource":
            from promptsource.templates import DatasetTemplates        
            if prompt_name in prompts.all_template_names:
                prompts = DatasetTemplates(dataset_name, dataset_path)
                return prompts[prompt_name]
        else:
            try:
                return PROMPT_REGISTRY[category_name][prompt_name]
            except:
                raise ValueError(
                    f"expected only a single `:` as separator between \
                    prompt category and name, but got `{prompt_id}` instead"
                    )

