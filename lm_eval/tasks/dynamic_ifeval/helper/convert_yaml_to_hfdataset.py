import yaml
from datasets import Dataset
import pandas as pd


def tuple_representer(dumper, data):
    return dumper.represent_list(list(data))


def tuple_constructor(loader, node):
    # Convert the YAML sequence back into a tuple.
    return tuple(loader.construct_sequence(node))


def load_dataset(filename="data/dataset.yaml"):
    yaml.add_representer(tuple, tuple_representer)
    yaml.add_constructor(u'tag:yaml.org,2002:python/tuple', tuple_constructor)
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def convert_yaml_to_hfdataset(yaml_file = "data/dataset.yaml", output_dir = "data/hf_dataset"):
    """
    Convert a YAML file to a Hugging Face Dataset format.

    Args:
        yaml_file (str): Path to the input YAML file.
        output_dir (str): Directory where the output dataset will be saved.
    """
    
    # Load the YAML file
    data = load_dataset(filename=yaml_file)
        
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)
    print(df.head())
    '''
    # Define the features explicitly
    features = Features({
        'count_number': Value('string'), # Assuming it can be represented as string
        'prompt': Value('string'),
        'rules': Value('string'), # Assuming it can be represented as string
        'rules_letter_must_be_in': Value('string'),
        'sum_characters_value': Value('int64')
    })
    '''

    # Convert the data to a Hugging Face Dataset
    df['rules_letter_must_be_in'] = df['rules_letter_must_be_in'].astype(str)
    hf_dataset = Dataset.from_pandas(df)
    #hf_dataset = Dataset.from_dict(hf_dataset.to_dict())
    
    # Save the dataset to the specified output directory
    hf_dataset.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")
    

if __name__ == "__main__":
    convert_yaml_to_hfdataset()
