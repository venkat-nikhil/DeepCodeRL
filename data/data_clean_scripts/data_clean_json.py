import json

def extract_programming_languages_from_file(json_filename):
    programming_languages = set()  # To store unique programming languages
    
    # Open the original JSON file
    with open(json_filename, 'r', encoding='utf-8') as file:
        # Load the data from the JSON file
        data = json.load(file, strict=False)
        
        # Check if the root element is a list of JSON objects
        if isinstance(data, list):
            for entry in data:
                # Extract the 'accepted_solutions' field (which is a string in this case)
                accepted_solutions_val = entry.get('accepted_solutions', None)
            
                if accepted_solutions_val:
                    # Check the type of accepted_solutions_val
                    if isinstance(accepted_solutions_val, str):
                        # If it's a string, parse it into a list of JSON objects
                        try:
                            accepted_solutions = json.loads(accepted_solutions_val)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e} for entry: {accepted_solutions_val}")
                            accepted_solutions = []
                    elif isinstance(accepted_solutions_val, list):
                        # It's already a list, so use it directly
                        accepted_solutions = accepted_solutions_val

                    else:
                        # If it's neither a string nor list, skip or handle accordingly
                        accepted_solutions = []
                            
                    # Extract the 'programmingLanguage' field from each solution
                    for solution in accepted_solutions:
                        # print(type(solution),solution)
                        if 'programmingLanguage' in solution:
                            programming_languages.add(solution['programmingLanguage'])
                            lang = solution['programmingLanguage']
                            if lang == 'Python 3':
                                print(solution['code'])
                        if 'programming_language' in solution:
                            programming_languages.add(solution['programming_language'])
                    # except json.JSONDecodeError:
                    #     # Handle JSONDecodeError if the accepted_solutions string is invalid
                    #     print(f"Error decoding JSON for accepted_solutions: {accepted_solutions_str}")
                
    return programming_languages

# Example usage:
json_filename = 'filtered_solutions_py_decontaminated2.json'
# json_filename = 'data/processed_codeforces/train.json' 
# unique_languages = extract_programming_languages_from_file(json_filename)
# print("Unique programming languages:", unique_languages)



def filter_for_all_python3_solutions(json_filename, output_filename):
    filtered_rows = []
    
    with open(json_filename, 'r', encoding='utf-8') as file:
        # Load the data; strict=False allows a bit more leniency if needed.
        data = json.load(file, strict=False)
        
    # Ensure data is a list of JSON objects.
    if isinstance(data, list):
        for entry in data:
            accepted_solutions_val = entry.get('accepted_solutions', None)
            if not accepted_solutions_val:
                continue  # Skip rows with no accepted_solutions

            # If accepted_solutions is a string, try parsing it.
            if isinstance(accepted_solutions_val, str):
                try:
                    accepted_solutions = json.loads(accepted_solutions_val)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in accepted_solutions: {e}\nValue: {accepted_solutions_val}")
                    accepted_solutions = []
            elif isinstance(accepted_solutions_val, list):
                accepted_solutions = accepted_solutions_val
            else:
                accepted_solutions = []

            # Look through the accepted solutions for the first one with programming language Python 3.
            found_code = []
            for solution in accepted_solutions:
                lang = ''
                if isinstance(solution, dict):
                    # If code is not there then skip this solution
                    if 'code' not in solution:
                        continue
                    if 'programmingLanguage' in solution:
                        lang = solution['programmingLanguage']
                        if lang == 'Python 3':
                            found_code.append(solution.get('code', None))
                            continue
                    if 'programming_language' in solution:
                        lang = solution['programming_language']
                        if lang == 'Python 3':
                            found_code.append(solution.get('code', None))
                            continue
                else:
                    print(f"Expected solution to be a dict but found {type(solution).__name__}.")

            # If we found a solution code for Python 3, add a new row to our filtered output.
            if found_code != []:
                for code in found_code:
                    # Replace the 'accepted_solutions' field with just the code.
                    new_entry = entry.copy()
                    new_entry['accepted_solutions'] = code
                    filtered_rows.append(new_entry)
    
    # Save the filtered rows to a new JSON file.
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        json.dump(filtered_rows, out_file, ensure_ascii=False, indent=4)
    
    print(f"Filtered data saved to {output_filename}")

# filter_for_all_python3_solutions(json_filename, 'filtered_solutions_py_decontaminated_Python3Code_all_solutions.json')

def filter_for_python3_solutions(json_filename, output_filename):
    '''
    Difference from filter_for_all_python3_solutions is that this function only keeps the first Python 3 solution found.
    '''
    filtered_rows = []
    
    with open(json_filename, 'r', encoding='utf-8') as file:
        # Load the data; strict=False allows a bit more leniency if needed.
        data = json.load(file, strict=False)
        
    # Ensure data is a list of JSON objects.
    if isinstance(data, list):
        for entry in data:
            accepted_solutions_val = entry.get('accepted_solutions', None)
            if not accepted_solutions_val:
                continue  # Skip rows with no accepted_solutions

            # If accepted_solutions is a string, try parsing it.
            if isinstance(accepted_solutions_val, str):
                try:
                    accepted_solutions = json.loads(accepted_solutions_val)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in accepted_solutions: {e}\nValue: {accepted_solutions_val}")
                    accepted_solutions = []
            elif isinstance(accepted_solutions_val, list):
                accepted_solutions = accepted_solutions_val
            else:
                accepted_solutions = []

            # Look through the accepted solutions for the first one with programming language Python 3.
            found_code = None
            for solution in accepted_solutions:
                lang = ''
                if isinstance(solution, dict):
                    # If code is not there then skip this solution
                    if 'code' not in solution:
                        continue
                    if 'programmingLanguage' in solution:
                        lang = solution['programmingLanguage']
                        if lang == 'Python 3':
                            found_code = solution.get('code', None)
                            continue
                    if 'programming_language' in solution:
                        lang = solution['programming_language']
                        if lang == 'Python 3':
                            found_code = solution.get('code', None)
                            continue
                else:
                    print(f"Expected solution to be a dict but found {type(solution).__name__}.")

            # If we found a solution code for Python 3, add a new row to our filtered output.
            if found_code:
                # for code in found_code:
                # Replace the 'accepted_solutions' field with just the code.
                new_entry = entry.copy()
                new_entry['accepted_solutions'] = found_code
                filtered_rows.append(new_entry)
    
    # Save the filtered rows to a new JSON file.
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        json.dump(filtered_rows, out_file, ensure_ascii=False, indent=4)
    
    print(f"Filtered data saved to {output_filename}")

json_output_filename = 'filtered_solutions_py_decontaminated_Python3Code.json'
# filter_for_python3_solutions(json_filename, json_output_filename)

def count_json_elts(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        count = len(data)
    return count

print("Number of JSON objects in the file filtered_solutions_py_decontaminated_Python3Code_all_solutions:", count_json_elts('filtered_solutions_py_decontaminated_Python3Code_all_solutions.json'))
print("Number of JSON objects in the file filtered_solutions_py_decontaminated_Python3Code:", count_json_elts('filtered_solutions_py_decontaminated_Python3Code.json'))
