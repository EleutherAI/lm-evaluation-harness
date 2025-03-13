import os
import re


def find_yaml_files(directory):
    yaml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml"):
                yaml_files.append(os.path.join(root, file))
    return yaml_files


def print_task_names(directory):
    yaml_files = find_yaml_files(directory)
    output_names = list()
    # print(yaml_files)

    print(f"Total YAML files: {len(yaml_files)}")

    for file in yaml_files:
        with open(file, "r") as f:
            content = f.read()
            task_names = re.findall(r"task: (.*)", content)
            for task_name in task_names:
                output_names.append(task_name)
    output_names.sort()
    return output_names

    
if __name__ == "__main__":
    # print_task_names("lm_eval/tasks/calamita")
    task_names = print_task_names("lm_eval/tasks/calamita")
    print(task_names)
