import json
from datasets import Dataset

class CustomDataset:
    def __init__(self, json_data, batch_size=2):
        """
        Initialize the dataset with a list of JSON objects.
        - json_data: List of dictionaries containing 'id', 'examples', and 'accepted_solutions'.
        - batch_size: The number of items (each item is a full JSON object) per batch.
        """
        self.data = json_data
        self.batch_size = batch_size
        self.num_batches = len(self.data) // self.batch_size + (1 if len(self.data) % self.batch_size != 0 else 0)

    def __getitem__(self, index):
        """
        Get a specific batch by index.
        Returns a dictionary that includes:
          - "ids": list of id fields for this batch.
          - "examples": list of examples for this batch (each example is kept as it is).
          - "accepted_solution": list of accepted solutions for this batch.
        """
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_data = self.data[start_idx:end_idx]
        
        ids = [item['id'] for item in batch_data]
        examples = [item['examples'] for item in batch_data]  # keeping examples as lists of dictionaries
        accepted_solution = [item['accepted_solutions'] for item in batch_data]
        
        return {
            'ids': ids,
            'examples': examples,
            'accepted_solution': accepted_solution
        }

    def __len__(self):
        """ Return the total number of batches """
        return self.num_batches

# Function to load JSON data from a file
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def parse_code_batch_to_individual_tests(code_batch):
    # This function is a placeholder for the actual implementation that would parse the code batch
    final_tests = []
    ids = code_batch.get("ids", [])
    accepted_codes = code_batch.get("accepted_solution", [])
    all_examples = code_batch.get("examples", [])
    
    # Iterate over each problem in the batch.
    # We assume the lists are aligned: ids[i], accepted_codes[i], and all_examples[i] belong together.
    for id_val, code, examples in zip(ids, accepted_codes, all_examples):
        # examples is assumed to be a list (each element is a dictionary with "input" and "output")
        for example in examples:
            in_val = example.get("input", "")
            out_val = example.get("output", "")
            final_tests.append([id_val, code, in_val, out_val])
    return final_tests


if __name__ == "__main__":
    # Load the JSON data (ensure the JSON file contains objects with 'id', 'examples', and 'accepted_solutions')
    file_path = 'data/cleaned_datasets/train.json'
    json_data = load_json_data(file_path)  # This should return a list of dictionaries

    # Initialize the custom dataset with a chosen batch size
    batch_size = 2
    dataset = CustomDataset(json_data, batch_size)

    # Access and print the batches with ids, examples, and accepted solutions
    for idx in range(len(dataset)):
        batch = dataset[idx]
        print(f"Batch {idx + 1}:")
        
        # Iterate over the ids, examples, and accepted solutions together using zip
        for id_val, example, solution in zip(batch['ids'], batch['examples'], batch['accepted_solution']):
            print(f"ID: {id_val}")
            print(f"Examples: {example}")
            print(f"Accepted Solution: {solution}")
            print()  # Newline for readability
        print("Parsed Tests:")
        parsed_tests = parse_code_batch_to_individual_tests(batch)
        for test in parsed_tests:
            print(f"ID: {test[0]}\nCode: {test[1]}\nInput: {test[2]}\nOutput: {test[3]}\n\n")
        break