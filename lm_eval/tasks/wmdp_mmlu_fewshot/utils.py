from datasets import load_dataset

def get_mmlu_fewshot_samples():
    # Load MMLU's college_biology validation split
    mmlu_dataset = load_dataset("cais/mmlu", "college_biology", split="validation")
    samples = []
    for example in mmlu_dataset:
        # Convert answer index (0-3) to letter (A-D)
        answer = ["A", "B", "C", "D"][example["answer"]]
        processed = {
            "question": example["question"],
            "choices": example["choices"],
            "answer": answer  # Match WMDP's 'answer' field format
        }
        samples.append(processed)
    return samples

from datasets import load_dataset

def process_mmlu_fewshots():
    # Load MMLU validation split
    mmlu_ds = load_dataset("cais/mmlu", "college_biology", split="validation")
    
    # Convert to WMDP-compatible format
    return [{
        "question": doc["question"],
        "choices": doc["choices"],
        "answer": ["A", "B", "C", "D"][doc["answer"]]  # Convert index to letter
    } for doc in mmlu_ds]