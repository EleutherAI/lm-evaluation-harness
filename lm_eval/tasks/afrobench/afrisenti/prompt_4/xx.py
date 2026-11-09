from datasets import load_dataset


data = load_dataset("masakhane/afrisenti", "orm", trust_remote_code=True)
print(data)
