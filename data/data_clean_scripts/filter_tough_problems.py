import json
import os

def filter_out_div1_problems(file_path):
    """
    Reads a JSON file containing a list of problems, removes any problem
    where 'contest_name' contains 'Div. 1', and writes the filtered list
    to a new file named 'filtered_<original_filename>.json'.

    Parameters:
    file_path (str): Path to the input JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # Handle both single object and list of objects
    if isinstance(data, dict):
        data = [data]

    filtered = [problem for problem in data if "Div. 1" not in problem.get("contest_name", "")]

    base_name = os.path.basename(file_path)
    output_file = f"tmp_{base_name}"

    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Filtered file written to: {output_file} ({len(filtered)} problems remaining)")

def remove_div2_not_ab(file_path):
    """
    Removes all problems from 'Div. 2' contests where the problem index is not 'A' or 'B'.
    Keeps all other problems, including non-Div. 2 contests.

    Parameters:
    file_path (str): Path to the input JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    # Filter logic: remove Div. 2 problems with index not in A or B
    filtered = [
        p for p in data
        if not ("Div. 2" in p.get("contest_name", "") and p.get("id", "")[-1] not in ("A", "B"))
    ]

    base_name = os.path.basename(file_path)
    output_file = 'tmp2.json'

    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Filtered file written to: {output_file} ({len(filtered)} problems remaining)")

def keep_only_abc_problems(file_path):
    """
    Keeps only problems whose 'id' ends with 'A', 'B', or 'C'.
    Writes the filtered list to 'filtered_<original_filename>.json'.

    Parameters:
    file_path (str): Path to the input JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    # Filter: Keep only problems whose id ends in A, B, or C
    filtered = [p for p in data if p.get("id", "")[-1] in ("A", "B", "C")]

    base_name = os.path.basename(file_path)
    output_file = "filtered_tmp.json"

    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Filtered file written to: {output_file} ({len(filtered)} problems remaining)")

if __name__ == '__main__':
    # filter_out_div1_problems('data/processed_codeforces/filtered_solutions_py_decontaminated_final.json')
    # remove_div2_not_ab('data/processed_codeforces/tmp_filtered_solutions_py_decontaminated_final.json')
    keep_only_abc_problems('data/processed_codeforces/tmp2.json')