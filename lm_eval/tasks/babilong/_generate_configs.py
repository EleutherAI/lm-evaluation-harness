"""
Generate configuration files for babilong benchmark.

This script generates YAML configuration files for all combinations of 
QA tasks (qa1-qa10) and length-based splits in the RMT-team/babilong dataset.
"""

import argparse
import os
import yaml
from tqdm import tqdm


# QA task configs (subsets) in RMT-team/babilong dataset
QA_CONFIGS = ['qa1', 'qa2', 'qa3', 'qa4', 'qa5', 'qa6', 'qa7', 'qa8', 'qa9', 'qa10']

# Length-based splits for each QA config
LENGTH_SPLITS = ['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1M']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", default="_default_template_yaml")
    parser.add_argument("--save_prefix_path", default="babilong")
    parser.add_argument("--simplified", action="store_true",
                        help="Generate simplified version with 10 QA files only")
    parser.add_argument("--qa_configs", nargs="*", default=None,
                        help="Specific QA configs to generate (qa1, qa2, etc.)")
    return parser.parse_args()


def generate_simplified_configs(args):
    """Generate 10 QA config files, each handling all length splits."""
    
    # Get base yaml filename
    base_yaml_name = os.path.basename(args.base_yaml_path)
    
    # Determine which QA configs to process
    qa_configs_to_process = args.qa_configs if args.qa_configs else QA_CONFIGS
    
    # Track all generated tasks for main group file
    all_qa_tasks = []
    
    print(f"Generating simplified configs for QA tasks: {qa_configs_to_process}")
    
    # Generate individual QA task configs that handle all length splits
    for qa_config in tqdm(qa_configs_to_process, desc="Processing QA configs"):
        task_name = f"{args.save_prefix_path}_{qa_config}"
        all_qa_tasks.append(task_name)
        
        yaml_dict = {
            "include": base_yaml_name,
            "task": task_name,
            "dataset_name": qa_config,  # QA config is the dataset subset

            # or use all available splits by default
            "tag": [
                "babilong_tasks",
                f"babilong_{qa_config}_tasks"
            ]
        }
        
        # Save individual QA task file
        file_save_path = f"{task_name}.yaml"
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
    
    # Generate main group file
    group_yaml_dict = {
        "group": args.save_prefix_path,
        "task": all_qa_tasks,
        "aggregate_metric_list": [
            {
                "aggregation": "mean",
                "metric": "exact_match",
                "weight_by_size": True,
                "filter_list": "custom-extract"
            }
        ],
        "metadata": {
            "description": "bAbI Long benchmark - all QA tasks (simplified version)",
            "version": "1.0"
        }
    }
    
    group_file_path = f"_{args.save_prefix_path}.yaml"
    with open(group_file_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            group_yaml_dict,
            yaml_file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    
    print(f"Generated {len(all_qa_tasks)} QA task configs")
    print(f"Generated 1 main group config: _{args.save_prefix_path}.yaml")
    
    return all_qa_tasks


def generate_detailed_configs(args):
    """Generate detailed configs with QA-based directory structure."""
    
    # Determine which configs and splits to process
    qa_configs_to_process = args.qa_configs if args.qa_configs else QA_CONFIGS
    length_splits_to_process = LENGTH_SPLITS
    
    # Get base yaml filename
    base_yaml_name = os.path.basename(args.base_yaml_path)
    
    # Track all generated tasks for group file
    all_tasks = []
    
    print(f"Generating detailed configs for QA tasks: {qa_configs_to_process}")
    print(f"Generating configs for length splits: {length_splits_to_process}")
    
    # Generate individual task configs with directory structure
    for qa_config in tqdm(qa_configs_to_process, desc="Processing QA configs"):
        # Create QA-specific directory
        qa_dir = qa_config
        if not os.path.exists(qa_dir):
            os.makedirs(qa_dir)
        
        for length_split in length_splits_to_process:
            task_name = f"{args.save_prefix_path}_{qa_config}_{length_split}"
            all_tasks.append(task_name)
            
            yaml_dict = {
                "include": f"../{base_yaml_name}",  # Relative path from subdirectory
                "task": task_name,
                "dataset_name": qa_config,  # QA config is the dataset subset
                "test_split": length_split,  # Length is the split
                "doc_to_text": f"!function ../utils.doc_to_text_{qa_config}",  # Fixed: no quotes
                "tag": [
                    "babilong_tasks",
                    f"babilong_{qa_config}_tasks", 
                    f"babilong_{length_split}_tasks"
                ]
            }
            
            # Save individual task file in QA subdirectory
            file_save_path = os.path.join(qa_dir, f"{length_split}.yaml")
            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                # Custom YAML output to handle !function without quotes
                content = f"""include: ../{base_yaml_name}
task: {task_name}
dataset_name: {qa_config}
test_split: {length_split}
doc_to_text: !function ../utils.doc_to_text_{qa_config}
tag:
- babilong_tasks
- babilong_{qa_config}_tasks
- babilong_{length_split}_tasks
"""
                yaml_file.write(content)
    
    # Generate main group file
    group_yaml_dict = {
        "group": args.save_prefix_path,
        "task": all_tasks,
        "aggregate_metric_list": [
            {
                "aggregation": "mean",
                "metric": "exact_match",
                "weight_by_size": True,
                "filter_list": "strip-whitespace"
            }
        ],
        "metadata": {
            "description": "bAbI Long benchmark - all QA tasks and length splits",
            "version": "1.0"
        }
    }
    
    group_file_path = f"_{args.save_prefix_path}.yaml"
    with open(group_file_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            group_yaml_dict,
            yaml_file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    
    # Generate subgroup files for QA configs
    for qa_config in qa_configs_to_process:
        qa_tasks = [task for task in all_tasks if f"_{qa_config}_" in task]
        
        if qa_tasks:
            subgroup_yaml_dict = {
                "group": f"{args.save_prefix_path}_{qa_config}",
                "task": qa_tasks,
                "aggregate_metric_list": [
                    {
                        "aggregation": "mean",
                        "metric": "exact_match",
                        "weight_by_size": True,
                        "filter_list": "strip-whitespace"
                    }
                ],
                "metadata": {
                    "description": f"bAbI Long benchmark - {qa_config} task across all lengths",
                    "version": "1.0"
                }
            }
            
            subgroup_file_path = f"_{args.save_prefix_path}_{qa_config}.yaml"
            with open(subgroup_file_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(
                    subgroup_yaml_dict,
                    yaml_file,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
    
    # Generate subgroup files for length splits
    for length_split in length_splits_to_process:
        length_tasks = [task for task in all_tasks if task.endswith(f"_{length_split}")]
        
        if length_tasks:
            subgroup_yaml_dict = {
                "group": f"{args.save_prefix_path}_{length_split}",
                "task": length_tasks,
                "aggregate_metric_list": [
                    {
                        "aggregation": "mean",
                        "metric": "exact_match", 
                        "weight_by_size": True,
                        "filter_list": "strip-whitespace"
                    }
                ],
                "metadata": {
                    "description": f"bAbI Long benchmark - {length_split} length across all QA tasks",
                    "version": "1.0"
                }
            }
            
            subgroup_file_path = f"_{args.save_prefix_path}_{length_split}.yaml"
            with open(subgroup_file_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(
                    subgroup_yaml_dict,
                    yaml_file,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
    
    print(f"Generated {len(all_tasks)} individual task configs")
    print(f"Generated 1 main group config: _{args.save_prefix_path}.yaml")
    print(f"Generated {len(qa_configs_to_process)} QA task subgroups")
    print(f"Generated {len(length_splits_to_process)} length split subgroups")


def main():
    args = parse_args()
    
    if args.simplified:
        generate_simplified_configs(args)
    else:
        generate_detailed_configs(args)


if __name__ == "__main__":
    main()