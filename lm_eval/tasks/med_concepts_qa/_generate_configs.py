from typing import List

import yaml


def generate_yaml_content(vocab_name: str, level: str):
    content = {
        "dataset_name": f"{vocab_name}_{level}",
        "tag": f"med_concepts_qa_{vocab_name}_tasks",
        "include": "_default_template_yaml",
        "task": f"med_concepts_qa_{vocab_name}_{level}",
        "task_alias": f"{vocab_name}_{level}",
    }
    return content


def generate_yaml_files(
    vocab_names: List[str], levels: List[str], file_name_prefix: str
):
    for vocab_name in vocab_names:
        for level in levels:
            yaml_content = generate_yaml_content(vocab_name, level)
            filename = f"{file_name_prefix}_{vocab_name}_{level}.yaml"
            with open(filename, "w") as yaml_file:
                yaml.dump(yaml_content, yaml_file, default_flow_style=False)
            print(f"Done to generated {filename}")


if __name__ == "__main__":
    generate_yaml_files(
        vocab_names=["icd9cm", "icd10cm", "icd9proc", "icd10proc", "atc"],
        levels=["easy", "medium", "hard"],
        file_name_prefix="med_concepts_qa",
    )
