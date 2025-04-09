from datasets import load_dataset
import numpy as np
import json

# Load the dataset from Hugging Face
ds = load_dataset("open-r1/codeforces-cots", "solutions_py_decontaminated")

# Assuming the dataset is split into train, test, or validation splits, 
# let's save the 'train' split to a JSON file as an example.
# If you need to work with a different split, change 'train' to 'test' or 'validation'.
train_data = ds['train']
columns_to_drop = ['prompt', 'generation', 'api_metadata', 'messages', 
                  'generated_tests', 'private_tests', 'public_tests', 'public_tests_ms']

train_data = train_data.remove_columns(columns_to_drop)
print(train_data.column_names)

train_data = train_data.to_pandas()
# train_data.to_json('filtered_solutions_py_decontaminated.json', orient='records', lines=True)

json_list = train_data.to_dict(orient='records')

def default_converter(o):
    if isinstance(o, np.ndarray):
        return o.tolist()  # Convert NumPy arrays to lists
    # Add other conversions if needed
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

with open('filtered_solutions_py_decontaminated2.json', 'w', encoding='utf-8') as f:
    json.dump(json_list, f, ensure_ascii=False, indent=4, default=default_converter)


# import pandas as pd
# accepted_solutions_df = pd.DataFrame(train_data['accepted_solutions'])
# print("DataFrame columns:", accepted_solutions_df.columns.tolist())
# accepted_solutions_list = accepted_solutions_df[0].dropna().tolist()

# # Now, save the resulting list of JSON objects to a file
# with open('only_accepted_solutions2.json', 'w', encoding='utf-8') as json_file:
#     json.dump(accepted_solutions_list, json_file, ensure_ascii=False, indent=4)

# print("Accepted solutions data saved to 'only_accepted_solutions2.json'")
