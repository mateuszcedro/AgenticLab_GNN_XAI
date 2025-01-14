from datasets import load_dataset

# Load the 'SST-2' dataset from HuggingFace Datasets
dataset = load_dataset('glue', 'sst2')

print("Dataset loaded successfully.")
print("Dataset splits:", dataset.keys())
print("Number of samples in the training set:", len(dataset['train']))
print("First training sample:")
print(dataset['train'][0])