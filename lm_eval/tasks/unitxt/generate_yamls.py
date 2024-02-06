#
# This file generates a set of LM eval harness yaml file
# that load unitxt datasets (https://github.com/IBM/unitxt)
# 
import yaml
from lm_eval import utils
import copy
from unitxt.artifact import fetch_artifact


# This code is required to properly dump LM harness YAML that contains references to functions
def function_representer(dumper: yaml.SafeDumper, func) -> yaml.nodes.MappingNode:
  return dumper.represent_scalar("!function", f"{func.__module__}.{func.__name__}",style=None)

def write_yaml(filename, data):
    yaml.add_representer(type(data["process_results"]), function_representer)
    with open(filename, 'w') as stream:
        yaml.dump(data, stream,  sort_keys=False)


default_template_per_task = { 
     "tasks.classification.multi_label" : "templates.classification.multi_label.default" ,
     "tasks.classification.multi_class" : "templates.classification.multi_class.default" ,
     "tasks.summarization.abstractive" : "templates.qa.contextual.simple",
     "tasks.regression.two_texts" : "templates.regression.two_texts.simple"
}


def generate_yaml(card: str):
    """
    Generate an LM Eval Harness YAML file based on the Unitxt dataset card.
    The output YAML is based on 'template.yaml.file' found in current directoy.

    The template is filled with the card name, a default template for the task
    of the card, and list of metrics.
    """
    print("*" * 80)
    print("*")
    print(f"* Generating YAML file for unitxt dataset {card}")
    print("*")

    card_definition , _ = fetch_artifact(f"cards.{card}")
    template = default_template_per_task[card_definition.task.artifact_identifier]

    data = utils.load_yaml_config('template.yaml.file')
    data["group"][0] = "unitxt"
    data["task"] = f"unitxt_{card}"
    data["dataset_name"] = f"card=cards.{card},template={template}"
    metric_list_element = data["metric_list"][0]
    data["metric_list"] = []
    for metric_name in  card_definition.task.metrics:
        new_metric = copy.copy(metric_list_element)
        new_metric['metric'] = metric_name.replace("metrics.", "unitxt_")
        data["metric_list"].append(new_metric)
    write_yaml(f"{card}.yaml", data)

def main():
    with open('unitxt_datasets') as f:
        for unitxt_dataset in f:
            if not unitxt_dataset.startswith("#"):
                generate_yaml(unitxt_dataset.strip())

if __name__ == '__main__':
    main()

