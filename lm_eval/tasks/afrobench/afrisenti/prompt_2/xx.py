from datasets import load_dataset


data = load_dataset("HausaNLP/AfriSenti-Twitter", "yor", trust_remote_code=True)
print(data)
