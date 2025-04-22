import time
import json
from datasets import Dataset
from eval_framework.eval import MultiProcessorEvaluator

class CustomDataset:
    def __init__(self, json_data, batch_size=2, id_column = 'id', examples_column = 'examples', accepted_solution_column = 'accepted_solutions'):
        """
        Initialize the dataset with a list of JSON objects.
        - json_data: List of dictionaries containing 'id', 'examples', and 'accepted_solutions'.
        - batch_size: The number of items (each item is a full JSON object) per batch.
        """
        self.data = json_data
        self.batch_size = batch_size
        self.num_batches = len(self.data) // self.batch_size + (1 if len(self.data) % self.batch_size != 0 else 0)
        self.id_column = id_column
        self.examples_column = examples_column
        self.accepted_solution_column = accepted_solution_column

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
        
        ids = [item[self.id_column] for item in batch_data]
        examples = [item[self.examples_column] for item in batch_data]  # keeping examples as lists of dictionaries
        accepted_solution = [item[self.accepted_solution_column] for item in batch_data]
        
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

def parse_code_batch_to_individual_tests(code_batch, solution_column_name="accepted_solution", examples_column_name="examples"):
    # This function is a placeholder for the actual implementation that would parse the code batch
    final_tests = []
    ids = code_batch.get("ids", [])
    accepted_codes = code_batch.get(solution_column_name, [])
    all_examples = code_batch.get(examples_column_name, [])
    
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
    file_path = 'data/cleaned_datasets/filtered_solutions_py_decontaminated_final.json'
    json_data = load_json_data(file_path)  # This should return a list of dictionaries

    # Initialize the custom dataset with a chosen batch size
    batch_size = 4
    dataset = CustomDataset(json_data, batch_size)

    tester = MultiProcessorEvaluator(
        command_prefix=['python','-c'],  # or None to auto‑use sys.executable
        max_workers=2,
        timeout=2.0
    )
    results_in_ratios = []
    start_time = time.time()

    # Access and print the batches with ids, examples, and accepted solutions
    for idx in range(len(dataset)):
        batch = dataset[idx]
        print(f"Batch {idx + 1}:")
        test_data_batch = []
        # Iterate over the ids, examples, and accepted solutions together using zip
        for id_val, example, solution in zip(batch['ids'], batch['examples'], batch['accepted_solution']):
            # print(f"ID: {type(id_val)}")
            # print(f"Examples: {type(example)}")
            # print(f"Accepted Solution: {type(solution)}")
            generated_code = solution
            inputs = []
            outputs = []
            if example is not None:
                for ex in example:
                    inputs.append(ex.get("input", ""))
                    outputs.append(ex.get("output", ""))
            # print(f"Inputs: {inputs}")
            # print(f"Outputs: {outputs}")
            test_data_batch.append([solution, inputs, outputs])


        # print(test_data_batch)
        results = tester.run(test_data_batch)
        # for score in tester.get_batch_run_scores(results):
        #     results_in_ratios.append(score)
        batch_scores = tester.get_batch_run_scores(results)
        results_in_ratios.append(sum(batch_scores) / len(batch_scores))
        print(f"score: {sum(batch_scores) / len(batch_scores)}")
        # print("Parsed Tests:")
        # parsed_tests = parse_code_batch_to_individual_tests(batch)
        # for test in parsed_tests:
        #     print(f"ID: {test[0]}\nCode: {test[1]}\nInput: {test[2]}\nOutput: {test[3]}\n\n")
        
    print(f"Total time taken: {time.time() - start_time} seconds")
    print(f"Results in Ratios for batches:{results_in_ratios}")
    print(f"Average Ratio: {sum(results_in_ratios) / len(results_in_ratios)}")