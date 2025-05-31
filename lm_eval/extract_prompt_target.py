import json
import sys
from lm_eval import tasks

def main(task_name, output_path):
    task = tasks.get_task_dict([task_name])[task_name]
    docs = list(task.validation_docs() or [])

    examples = []
    for doc in docs:
        try:
            prompt = task.doc_to_text(doc)
            correct = task.doc_to_target(doc)
            incorrect = None

            if hasattr(task, "doc_to_choice"):
                choices = task.doc_to_choice(doc)
                incorrect = [c for c in choices if c != correct]
                if incorrect:
                    incorrect = incorrect[0]

            if correct:
                examples.append({"text": prompt + correct, "label": 1})
            if incorrect:
                examples.append({"text": prompt + incorrect, "label": 0})
        except:
            continue

    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m lm_eval.extract_prompt_target <task_name> <output_file>")
        sys.exit(1)
    task_name = sys.argv[1]
    output_path = sys.argv[2]
    main(task_name, output_path)