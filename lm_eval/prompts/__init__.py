import ast
import logging
import os
from typing import Dict

from lm_eval import utils


eval_logger = logging.getLogger(__name__)

# Prompt library.
# Stores prompts in a dictionary indexed by 2 levels:
# prompt category name, and prompt name.
# This allows us to access prompts
PROMPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "qa-basic": {
        "question-newline-answer": "Question: {{question}}\nAnswer:",
        "q-newline-a": "Q: {{question}}\nA:",
    },
}


def get_prompt(prompt_id: str, dataset_name: str = None, subset_name: str = None):
    # unpack prompt name
    category_name, prompt_name = prompt_id.split(":")
    if subset_name is None:
        dataset_full_name = dataset_name
    else:
        dataset_full_name = f"{dataset_name}-{subset_name}"
    eval_logger.info(f"Loading prompt from {category_name} for {dataset_full_name}")
    if category_name == "promptsource":
        try:
            from promptsource.templates import DatasetTemplates
        except ModuleNotFoundError as exception:
            raise type(exception)(
                "Tried to load a Promptsource template, but promptsource is not installed ",
                "please install promptsource via pip install lm-eval[promptsource] or pip install -e .[promptsource]",
            )
        try:
            if subset_name is None:
                prompts = DatasetTemplates(dataset_name=dataset_name)
            else:
                prompts = DatasetTemplates(
                    dataset_name=dataset_name, subset_name=subset_name
                )
        except Exception:
            raise ValueError(f"{dataset_name} and {subset_name} not found")
        if prompt_name in prompts.all_template_names:
            return prompts[prompt_name]
        else:
            raise ValueError(
                f"{prompt_name} not in prompt list {prompts.all_template_names}"
            )
    elif ".yaml" in category_name:
        import yaml

        with open(category_name, "rb") as file:
            prompt_yaml_file = yaml.full_load(file)

        prompt_string = prompt_yaml_file["prompts"][prompt_name]
        return PromptString(prompt_string)
    else:
        try:
            return PROMPT_REGISTRY[category_name][prompt_name]
        except Exception:
            raise ValueError(
                f"expected only a single `:` as separator between \
                prompt category and name, but got `{prompt_id}` instead"
            )


def load_prompt_list(
    use_prompt: str, dataset_name=None, subset_name=None, yaml_path=None, **kwargs
):
    category_name, prompt_name = use_prompt.split(":")

    if category_name == "promptsource":
        from promptsource.templates import DatasetTemplates

        if subset_name is None:
            prompts = DatasetTemplates(dataset_name=dataset_name)
        else:
            prompts = DatasetTemplates(
                dataset_name=dataset_name, subset_name=subset_name
            )

        prompt_list = utils.pattern_match(prompt_name, prompts.all_template_names)

    elif ".yaml" in category_name:
        import yaml

        if yaml_path is not None:
            category_name = os.path.realpath(os.path.join(yaml_path, category_name))

        with open(category_name, "rb") as file:
            prompt_yaml_file = yaml.full_load(file)

        prompt_list = utils.pattern_match(
            prompt_name, prompt_yaml_file["prompts"].keys()
        )

    # category_name, *prompt_name = use_prompt.split(":")
    # TODO allow to multiple prompt naming
    # if len(prompt_name) > 1:
    #     prompt_list = []
    #     for prompt in prompt_name:
    #         prompt_list.append(utils.pattern_match(prompt_name, prompts.all_template_names))
    # else:
    #     prompt_list = utils.pattern_match(prompt_name, prompts.all_template_names)
    return [":".join([category_name, prompt]) for prompt in prompt_list]


class PromptString:
    def __init__(self, prompt_string):
        self.prompt_string = prompt_string

    def apply(self, doc):
        doc_to_text = self.prompt_string["doc_to_text"]
        doc_to_target = self.prompt_string["doc_to_target"]

        # TODO need a way to process doc_to_choice
        if "doc_to_choice" in self.prompt_string:
            raise NotImplementedError("Not yet implemented to accept doc_to_choice")

        text_string = utils.apply_template(doc_to_text, doc)
        target_string = utils.apply_template(doc_to_target, doc)

        return [text_string, target_string]
