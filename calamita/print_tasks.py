import os
import ruamel.yaml


def find_yaml_files(directory):
    yaml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml"):
                yaml_files.append(os.path.join(root, file))
    return yaml_files


def print_task_names(directory):
    yaml_files = find_yaml_files(directory)
    # print(yaml_files)
    parser = ruamel.yaml.YAML()

    yaml_files.sort()
    print(f"Total YAML files: {len(yaml_files)}")

    for file in yaml_files:
        with open(file, "r") as f:
            try:
                data = parser.load(f)
                if "task" in data:
                    print(f"Task in {file}: {data['task']}")
            except Exception as e:
                print(f"Error parsing {file}: {e}")


# Replace 'your_directory_path' with the path to the directory you want to search
print_task_names("lm_eval/tasks/calamita")
