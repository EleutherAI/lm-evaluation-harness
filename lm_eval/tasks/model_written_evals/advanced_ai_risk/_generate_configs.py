import datasets
import yaml
from tqdm_loggable.auto import tqdm


def main() -> None:
    dataset_path = "EleutherAI/advanced_ai_risk"
    for task in tqdm(datasets.get_dataset_infos(dataset_path).keys()):
        file_name = f"{task}.yaml"
        try:
            with open(f"{file_name}", "w", encoding="utf-8") as f:
                f.write("# Generated by _generate_configs.py\n")
                yaml.dump(
                    {
                        "include": "_template_yaml",
                        "task": f"{dataset_path.split('/')[-1]}_{task}",
                        "dataset_name": task,
                    },
                    f,
                )
        except FileExistsError:
            pass


if __name__ == "__main__":
    main()
