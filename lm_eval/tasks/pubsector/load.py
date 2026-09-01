from datasets import load_dataset

4
    
# one = pa.json.read_json('sample.json', parse_options=json.ParseOptions(explicit_schema=schema))
dataset = load_dataset("json", data_files="/home/anbh299g/code/GovTeuken/synthetic-datagen/resources/data/generated_synthetic_data/ce_bgh_00000_de_synthetic_fixed.jsonl")
