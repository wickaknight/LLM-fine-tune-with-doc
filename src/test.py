from datasets import load_dataset

# Load the dataset from your JSONL file
dataset = load_dataset("json", data_files={"train": "data/train.jsonl"}, split="train")

# Optionally, split the dataset into train and validation sets
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]

# Now print the column names
print("Train dataset columns:", train_dataset.column_names)
