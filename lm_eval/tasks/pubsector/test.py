from datasets import load_dataset
data = load_dataset("json", data_files="lm_eval/tasks/pubsector/pubsector_data.jsonl", split=None)
print(data['train'][0])
