import re
from list_tasks import find_yaml_files
from datasets import load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    yaml_files = find_yaml_files("lm_eval/tasks/calamita")
    print(f"Total YAML files: {len(yaml_files)}")
    print(yaml_files)

    for file in tqdm(yaml_files, desc="Processing YAML files"):
        with open(file, "r") as f:
            content = f.read()
            dataset_name_match = re.search(r"dataset_name:\s*(\S+)", content)
            dataset_path_match = re.search(r"dataset_path:\s*(\S+)", content)
            if dataset_name_match and dataset_path_match:
                dataset_name = dataset_name_match.group(1)
                dataset_path = dataset_path_match.group(1)
                print(f"Dataset Path: {dataset_path}, Dataset Name: {dataset_name}")
                try:
                    dataset = load_dataset(
                        dataset_path,
                        dataset_name if dataset_name != "null" else None,
                        trust_remote_code=True
                    )
                except Exception as e:
                    print(f"Error loading dataset: {e}")