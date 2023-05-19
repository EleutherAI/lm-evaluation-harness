from lm_eval.logger import eval_logger
from promptsource.templates import DatasetTemplates

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

def get_prompt(prompt_id: str, dataset_name=None, subset_name=None):
    # unpack prompt name 
    category_name, prompt_name = prompt_id.split(":")
    eval_logger.info(
        f"Loading prompt from {category_name}"
        )
    if category_name == "promptsource":
        try:
            # prompts = DatasetTemplates(dataset_name, dataset_path)
            if subset_name == None:
                prompts = DatasetTemplates(dataset_name=dataset_name)
            else:
                prompts = DatasetTemplates(dataset_name=dataset_name, subset_name=subset_name)
        except:
            raise ValueError(
                f"{dataset_name} and {subset_name} not found"
                )
        if prompt_name in prompts.all_template_names:
            return prompts[prompt_name]
        else:
            raise ValueError(
                f"{prompt_name} not in prompt list {prompts.all_template_names}"
                )
    else:
        try:
            return PROMPT_REGISTRY[category_name][prompt_name]
        except:
            raise ValueError(
                f"expected only a single `:` as separator between \
                prompt category and name, but got `{prompt_id}` instead"
                )

