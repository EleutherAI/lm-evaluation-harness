import subprocess

tasks = [
    "truthfulqa_mc1",
    "winogrande",
    "wsc",
    "piqa",
    "tmmluplus_physics",
    "tmmluplus_finance_banking"
]

for task in tasks:
    output_file = f"{task}_prompts.json"
    print(f"Extracting {task} -> {output_file}")
    subprocess.run(["python", "-m", "lm_eval.extract_prompt_target", task, output_file])