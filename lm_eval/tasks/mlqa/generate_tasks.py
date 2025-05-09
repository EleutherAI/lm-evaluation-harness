# ruff: noqa: E731, E741
"""
Script to generate task YAMLs for the mlqa dataset.
Based on `tasks/bigbench/generate_tasks.py`.
"""

from datasets import get_dataset_config_names


chosen_subtasks = []

language_dict = {
    "en": "english",
    "es": "spanish",
    "hi": "hindi",
    "vi": "vietnamese",
    "de": "german",
    "ar": "arabic",
    "zh": "chinese",
}


def main() -> None:
    configs = get_dataset_config_names("facebook/mlqa", trust_remote_code=True)
    for config in configs:
        if len(config.split(".")) == 2:
            continue
        else:
            chosen_subtasks.append(config)
    assert len(chosen_subtasks) == 49
    for task in chosen_subtasks:
        file_name = f"{task.replace('.', '_')}.yaml"
        context_lang = file_name.split("_")[1]
        # Not using yaml to avoid tagging issues with !function
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("# Generated by generate_tasks.py\n")

            # Manually writing the YAML-like content inside files to avoid tagging issues
            f.write("include: mlqa_common_yaml\n")
            f.write(f"task: {task.replace('.', '_')}\n")
            f.write(f"dataset_name: {task}\n")
            f.write(
                f"process_results: !function utils.process_results_{context_lang}\n"
            )


if __name__ == "__main__":
    main()
