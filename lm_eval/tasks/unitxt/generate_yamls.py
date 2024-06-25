#
# This file generates a set of LM eval harness yaml file
# that load unitxt datasets (https://github.com/IBM/unitxt)
#

import unitxt_wrapper
import yaml
from unitxt.artifact import fetch_artifact
from unitxt.standard import StandardRecipe


# This code is required to properly dump LM harness YAML that contains references to functions
def function_representer(dumper: yaml.SafeDumper, func) -> yaml.nodes.MappingNode:
    return dumper.represent_scalar(
        "!function", f"{func.__module__}.{func.__name__}", style=None
    )


def write_task_yaml(filename, data):
    yaml.add_representer(type(data["process_results"]), function_representer)
    with open(filename, "w") as stream:
        yaml.dump(data, stream, sort_keys=False)


def write_card_yaml(filename, data):
    with open(filename, "w") as stream:
        yaml.dump(data, stream, sort_keys=False)


default_template_per_task = {
    "tasks.classification.multi_label": "templates.classification.multi_label.title",
    "tasks.classification.multi_class": "templates.classification.multi_class.title",
    "tasks.summarization.abstractive": "templates.summarization.abstractive.full",
    "tasks.regression.two_texts": "templates.regression.two_texts.simple",
    "tasks.qa.with_context.extractive": "templates.qa.with_context.simple",
    "tasks.grammatical_error_correction": "templates.grammatical_error_correction.simple",
    "tasks.span_labeling.extraction": "templates.span_labeling.extraction.title",
}


def generate_task_yaml(task: str):
    """
    Generate an LM Eval Harness YAML file based on a Unitxt task defintion.
    The output YAML is based on 'template.yaml.file' found in current directoy.

    The common template is filled the the specific metrics for the task.
    It still leaves the 'dataset_name' and 'task name' unspecified.
    """
    print("*" * 80)
    print("*")
    print(f"* Generating YAML base file for task {task}")
    print("*")
    task_definition, _ = fetch_artifact(task)
    data = {
        "group": ["unitxt"],
        "dataset_path": "unitxt/data",
        "output_type": "generate_until",
        "training_split": "train",
        "validation_split": "test",
        "doc_to_text": "{{source}}",
        "doc_to_target": "target",
        "process_results": unitxt_wrapper.process_results,
        "generation_kwargs": {"until": ["</s>"]},
        "metric_list": [],
        "metadata": {"verison": 1.0},
    }

    for metric_name in task_definition.metrics:
        new_metric = {"metric": "", "aggregation": "unitxt", "higher_is_better": True}
        new_metric["metric"] = metric_name.replace("metrics.", "unitxt_")
        data["metric_list"].append(new_metric)

    write_task_yaml(f"unitxt_{task}", data)


def generate_card_yaml(card: str):
    """
    Generate an LM Eval Harness YAML file based on the Unitxt dataset card.
    It includes the task YAML for the dataset, and overrides the 'dataset_name' and 'task' with the card.
    """

    print("*" * 80)
    print("*")
    print(f"* Generating YAML file for unitxt dataset {card}")
    print("*")

    card_definition, _ = fetch_artifact(f"cards.{card}")
    task = card_definition.task.__id__
    if task in default_template_per_task:
        template = default_template_per_task[task]
    else:
        raise ValueError(
            f"Default template was not defined for task {task} in 'default_template_per_task' dict in generate_yamls.py"
        )
    data = {}
    data["include"] = f"unitxt_{task}"
    data["task"] = card
    data["dataset_name"] = f"card=cards.{card},template={template}"
    # This is faster that the load_dataset approach
    # dataset = load_dataset('unitxt/data',  data["dataset_name"]+",loader_limit=100",trust_remote_code=True)
    recipe = StandardRecipe(card=f"cards.{card}", template=template, loader_limit=100)
    stream = recipe()
    dataset = stream.to_dataset()
    print(dataset)
    print("Sample input:")
    print(dataset["test"][0]["source"])
    print("Sample output:")
    print(dataset["test"][0]["target"])
    write_card_yaml(f"{card}.yaml", data)


def main():
    for task in default_template_per_task.keys():
        try:
            generate_task_yaml(task)
        except Exception as e:
            print(f"Unable to generate YAML for {task} due to:")
            print(e)
            raise (e)
    with open("unitxt_datasets") as f:
        for unitxt_dataset in f:
            unitxt_dataset = unitxt_dataset.strip()
            if unitxt_dataset.startswith("### END ###"):
                exit(0)
            if not unitxt_dataset.startswith("#"):
                try:
                    generate_card_yaml(unitxt_dataset)
                except Exception as e:
                    print(f"Unable to generate YAML for {unitxt_dataset} due to:")
                    print(e)
                    raise e


if __name__ == "__main__":
    main()
