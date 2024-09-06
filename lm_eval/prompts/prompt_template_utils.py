import json
import os
import random
import sys

CHOSEN_SEPARATOR_LIST = [":", "-", "|", "<sep>", ".", "]", "/", "\\", "!", "'", '"']
CHOSEN_SPACE_LIST = [" ", "\n", " \n", "  ", "; \n", ", ", " , ", "\n "]

OPERATORS_LIST = [
    (lambda x: x, "lambda x: x"),
    (lambda x: x.title(), "lambda x: x.title()"),
    (lambda x: x.upper(), "lambda x: x.upper()"),
    (lambda x: x.lower(), "lambda x: x.lower()"),
]


def has_alpha_characters(input_string):
    return any(char.isalpha() for char in input_string)


def call_operator_fn(string, operator_fn, space, initial_space):
    words = string.split(initial_space)
    result = []

    for word in words:
        if len(word) == 0:
            continue
        if "{" in word or word[0].isalpha():
            result.append(word)
        else:
            result.append(operator_fn(word))

    return space.join(result)


def generate_n_templates(initial_template, initial_separator, n=100):
    random.seed(0)
    templates_metadata = set([(initial_template, "\n", initial_separator, "lambda x: x")])
    templates = set([initial_template])
    agenda = [(initial_template, 0, initial_separator)]

    while len(agenda) > 0:
        if n is not None and len(templates_metadata) >= n:
            break

        curr_template, depth, initial_space = random.choice(agenda)
        agenda.remove((curr_template, depth, initial_space))

        # curr_template, depth, initial_space = agenda.pop(0)

        existing_item = None
        for e in CHOSEN_SEPARATOR_LIST:
            if curr_template.find(e) != -1:
                existing_item = e

        if existing_item is None:
            continue

        for sep in CHOSEN_SEPARATOR_LIST:
            if len(templates_metadata) < n:
                operator_fn = random.choice(OPERATORS_LIST)
                space = random.choice(CHOSEN_SPACE_LIST)
                new_template = call_operator_fn(
                    curr_template.replace(existing_item, sep),
                    operator_fn[0],
                    space,
                    initial_space,
                )

                if new_template not in templates:
                    templates_metadata.add((new_template, sep, space, operator_fn[1]))
                    templates.add(new_template)
                    if space != "":
                        agenda.append((new_template, depth + 1, space))

        print("Number of templates:", len(templates))
        sys.stdout.flush()

    for t in templates_metadata:
        print("Template", t)
    return sorted(list(templates_metadata))


def build_prompts_variations_str_template(
    template_str: str,
    dataset_name: str,
    num_variations: int,
    templates_folder: str,
    return_metadata: bool = False,
):
    """
    Build the prompt string template for the given dataset.
    Args:
        template_str: The template string to use for building the prompt. Example:
            "The following are multiple choice questions (with answers) about {topic}.\n{question}.\nAnswers: \n{choices}.\nAnswer:"
        dataset_name: The name of the dataset.
        templates_folder: The folder to save the templates.

    Raises:
        ValueError: If the templates already exist.
    """
    os.makedirs(templates_folder, exist_ok=True)
    templates = list(generate_n_templates(template_str, "\n", num_variations))

    # this can be improved, creating a better structure for the metadata and saving it in a more readable way
    print(
        "Exporting generated templates. Saved in:",
        f"{templates_folder}/{dataset_name}_templates.json",
    )
    with open(f"{templates_folder}/{dataset_name}_templates.json", "w") as json_file:
        json.dump([template for template, _, _, _ in templates], json_file, indent=2)

    # if metadata is needed for the templates (e.g., separator, space, operator)
    if return_metadata:
        print("Saving metadata for the templates.")
        with open(f"{templates_folder}/{dataset_name}_templates_metadata.json", "w") as json_file:
            json.dump(
                {template: {"sep": sep, "space": space, "op": op} for template, sep, space, op in templates},
                json_file,
                indent=2,
            )

    with open(f"{templates_folder}/{dataset_name}_templates.json", "r") as file:
        res = json.load(file)

    print("Saving generated templates and the following index.")
    with open(f"{templates_folder}/template_to_index.json", "w") as file:
        json.dump({res[i]: i for i in range(len(res))}, file, indent=2)

    return templates
