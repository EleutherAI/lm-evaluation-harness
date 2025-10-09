import yaml
from pathlib import Path

# List of dialects
dialects = [
    'ALG', 'SAU', 'MOR', 'OMA', 'EGY', 'BAH', 'JOR', 'LIB', 'MAU', 'QAT',
    'IRQ', 'PAL', 'LEB', 'TUN', 'MSA', 'UAE', 'SUD', 'SYR', 'KUW', 'YEM'
]

# Default template include
default_yaml = "_default_jawaher_template_yaml"

# Output directory for YAMLs
output_dir = Path("/home/abdelrahman.sadallah/mbzuai/lm-evaluation-harness/lm_eval/tasks/jawaher")
output_dir.mkdir(parents=True, exist_ok=True)

# Custom class to represent !function tag
class FunctionTag(str):
    pass

# Custom representer for !function
def function_representer(dumper, data):
    return dumper.represent_scalar('!function', data)

yaml.add_representer(FunctionTag, function_representer)

for dia in dialects:
    task_tag = FunctionTag(f"utils.process_docs_{dia}")
    yaml_content = {
        "include": default_yaml,
        "task": f"jawaher_{dia.lower()}",
        "dataset_name" : dia
        # "process_docs": task_tag
    }

    out_file = output_dir / f"task_{dia}.yaml"
    with open(out_file, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"✅ Wrote {out_file}")
