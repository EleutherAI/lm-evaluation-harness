from datasets import load_dataset

data = load_dataset("masakhane/ntrex_african", name="afr_Latn", split="test")
print(data)
