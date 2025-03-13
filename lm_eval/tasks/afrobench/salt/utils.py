from datasets import load_dataset

# data = load_dataset('Sunbird/salt', 'text-all', split='test', trust_remote_code=True)
data = load_dataset('davidstap/NTREX', 'eng_Latn', split='dev', trust_remote_code=True)
print(data)
print(data[:2])
