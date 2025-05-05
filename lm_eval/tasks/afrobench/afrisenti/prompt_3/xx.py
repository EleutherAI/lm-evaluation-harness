from datasets import load_dataset


data = load_dataset("masakhane/afrisenti", "por", trust_remote_code=True)
print(data)
