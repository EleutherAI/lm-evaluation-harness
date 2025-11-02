import datasets
from datasets import Dataset, DatasetDict
import random

# Load the original dataset
dataset = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4-Unique")

# Get the train split (assuming it's the main split)
train_data = dataset["train"]

# Set random seed for reproducibility
random.seed(42)

# Create indices for splitting
total_size = len(train_data)
test_size = 1000

# Randomly sample indices for test split
all_indices = list(range(total_size))
random.shuffle(all_indices)

test_indices = all_indices[:test_size]
train_indices = all_indices[test_size:]

# Create new train and test splits
new_train = train_data.select(train_indices)
new_test = train_data.select(test_indices)

# Create new dataset dict with train/test splits
new_dataset = DatasetDict({
    "train": new_train,
    "test": new_test
})

# Push to hub with new name
new_dataset.push_to_hub("Stephen-Xie/Countdown")

print(f"Dataset reuploaded with {len(new_train)} train samples and {len(new_test)} test samples")
