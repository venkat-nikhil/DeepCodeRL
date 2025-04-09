import csv
import json
import string
import pandas as pd
import tqdm
from preprocess_codeforces import extract_solution_code

# Define the valid characters based on a standard Google Keyboard (Gboard)
valid_chars = string.ascii_letters + string.digits + string.punctuation + string.whitespace
extra_valid_chars = '≤≥'  # Adding specific characters that might be used in Gboard

# Function to check if a string contains only Gboard characters
def is_valid_gboard_string(input_string):
    return all(c in valid_chars for c in input_string)

# Function to filter rows
def filter_csv(input_filename, output_filename):
    with open(input_filename, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            # Check if all fields in the row are valid Gboard strings
            if all(is_valid_gboard_string(field) for field in row):
                writer.writerow(row)
            else:
                print(f"Filtered out row: {row}")

# Example usage
input_filename = 'filtered_codeforces_train_rows.csv'  # Replace with your input CSV file
output_filename = 'filtered_output.csv'  # Replace with desired output CSV file

# filter_csv(input_filename, output_filename)

# def print_column(csv_filename, column_name):
#     with open(csv_filename, mode='r', newline='', encoding='utf-8') as file:
#         reader = csv.DictReader(file)
        
#         cnt = 20
#         i = 0
#         # Iterate over rows and print the specified column values
#         for row in reader:
#             i+=1
#             if i > cnt:
#                 break
#             tmp = row[column_name]
#             print(row[column_name])

# print_column(input_filename, 'examples')

def get_unique_attributes(csv_filename, column_name, json_attr_name='programmingLanguage'):
    programming_languages = set()  # To store unique programming languages
    
    with open(csv_filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # Get the accepted_solutions field, which is a JSON string
            accepted_solutions_orig = row[column_name]
            if accepted_solutions_orig == '':
                continue
            
            try:
                # Parse the JSON string
                accepted_solutions_json = row[column_name]
            
                # Remove unwanted characters (e.g., newlines or extra spaces)
                # accepted_solutions_json = accepted_solutions_json.strip().replace("'", '"').replace("}\r\n {", "}, {").replace("} {", "}, {")
                
                accepted_solutions = json.loads(accepted_solutions_json, strict=False)
                # Extract the programmingLanguage from each solution in the list
                for solution in accepted_solutions:
                    programming_languages.add(solution[json_attr_name])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {accepted_solutions_orig}\n{accepted_solutions_json}")
                print(f"Error Message: {e}")
                pass
    
    return programming_languages

# unique_languages = get_unique_attributes(input_filename, 'accepted_solutions')
# print("Unique programming languages:", unique_languages)

# tmp = '[{\'code\': \'\\r\\ndef play(l,r,x,y,k):\\r\\n    for i in range(x,y+1):\\r\\n        if l<=i*k<=r:\\r\\n            return "YES"\\r\\n    return "NO"\\r\\n\\r\\nl,r,x,y,k=map(int,input().split())\\r\\nprint(play(l,r,x,y,k))\\r\\n\', \'passedTestCount\': 101.0, \'passed_test_count\': None, \'programmingLanguage\': \'Python 3\', \'programming_language\': None, \'submission_id\': None, \'verdict\': \'OK\'}\r\n {\'code\': \'l,r,x,y,k=[int(x) for x in input().split()]\\r\\nflag=0\\r\\nfor i in range(x,y+1):\\r\\n\\tif k*i<=r and k*i>=l:\\r\\n\\t\\tflag=1\\r\\n\\t\\tbreak\\r\\nif flag==1:\\r\\n\\tprint("YES")\\r\\nelse:\\r\\n\\tprint("NO")\', \'passedTestCount\': 101.0, \'passed_test_count\': None, \'programmingLanguage\': \'PyPy 3\', \'programming_language\': None, \'submission_id\': None, \'verdict\': \'OK\'}\r\n {\'code\': "l,r,x,y,k = list(map(int, input().split()))\\r\\nif k*y < l or r < k*x:\\r\\n    print(\'NO\')\\r\\nelif k*x < l and r < k*y:\\r\\n    if (r-l+1) >= k or l%k > r%k or l%k == 0:\\r\\n        print(\'YES\')\\r\\n    else:\\r\\n        print(\'NO\')\\r\\nelse:\\r\\n    print(\'YES\')\\r\\n    ", \'passedTestCount\': 101.0, \'passed_test_count\': None, \'programmingLanguage\': \'Python 3\', \'programming_language\': None, \'submission_id\': None, \'verdict\': \'OK\'}\r\n {\'code\': \'l,r,x,y,k=map(int,input().split())\\r\\nfor i in range (x,y+1):\\r\\n    if i*k<=r and  i*k>=l:\\r\\n        print("YES")\\r\\n        exit()\\r\\nprint(("NO"))\', \'passedTestCount\': 101.0, \'passed_test_count\': None, \'programmingLanguage\': \'Python 3\', \'programming_language\': None, \'submission_id\': None, \'verdict\': \'OK\'}\r\n {\'code\': "from bisect import bisect_left\\r\\nl, r, x, y, k = list(map(int, input().split()))\\r\\n# d = y-x+1; p = y+1\\r\\n# while d > 0:\\r\\n    # while (p-d >= x and (p-d)*k >= l): \\r\\n        # p -= d\\r\\n    # d //= 2\\r\\np = bisect_left(range(x, y+1), l/k) + x\\r\\nres = (p <= y and p*k <= r)\\r\\nprint(\'YES\' if res else \'NO\')", \'passedTestCount\': 101.0, \'passed_test_count\': None, \'programmingLanguage\': \'Python 3\', \'programming_language\': None, \'submission_id\': None, \'verdict\': \'OK\'}\r\n {\'code\': \'l, r, x, y, k = tuple(map(int, input().split(\\\' \\\')))\\nfor i in range(x, y+1):\\n    if l <= i * k <= r:\\n        print("YES")\\n        break\\nelse:\\n    print("NO")\\n\\t\\t \\t \\t  \\t\\t\\t    \\t\\t\\t    \\t   \\t\\t  \\t\', \'passedTestCount\': 101.0, \'passed_test_count\': None, \'programmingLanguage\': \'PyPy 3\', \'programming_language\': None, \'submission_id\': None, \'verdict\': \'OK\'}\r\n {\'code\': "import math\\r\\nimport sys\\r\\nimport collections\\r\\n\\r\\n\\r\\ndef In():\\r\\n    return map(int, sys.stdin.readline().split())\\r\\n\\r\\n\\r\\ninput = sys.stdin.readline\\r\\n\\r\\n\\r\\ndef krillgame():\\r\\n    l, r, x, y, k = In()\\r\\n    for i in range(x,y+1):\\r\\n        if l<= k*i <= r:\\r\\n            print(\'YES\')\\r\\n            break\\r\\n    else:\\r\\n        print(\'NO\')\\r\\nkrillgame()", \'passedTestCount\': 101.0, \'passed_test_count\': None, \'programmingLanguage\': \'PyPy 3\', \'programming_language\': None, \'submission_id\': None, \'verdict\': \'OK\'}\r\n {\'code\': \'I = lambda: map(int, input().rstrip().split())\\nl, r, x, y, k = I()\\nfor cost in range(x, y + 1):\\n    if r >= cost * k >= l:\\n        print("YES")\\n        exit()\\nprint("NO")\', \'passedTestCount\': 101.0, \'passed_test_count\': None, \'programmingLanguage\': \'Python 3\', \'programming_language\': None, \'submission_id\': None, \'verdict\': \'OK\'}\r\n {\'code\': \'l, r, x, y, k = [int(x) for x in (input().split())]\\r\\nans = "NO"\\r\\nfor i in range(x, y + 1):\\r\\n\\tif i * k >= l and i * k <= r:\\r\\n\\t\\tans = "YES"\\r\\n\\t\\tbreak\\r\\nprint(ans)\', \'passedTestCount\': 101.0, \'passed_test_count\': None, \'programmingLanguage\': \'PyPy 3\', \'programming_language\': None, \'submission_id\': None, \'verdict\': \'OK\'}\r\n {\'code\': \'l,r,x,y,k = map(int, input().split(" "))\\nprint(\\\'NO\\\'if max((l+k-1)//k,x)>min(r//k,y) else\\\'YES\\\')\\n\', \'passedTestCount\': 101.0, \'passed_test_count\': None, \'programmingLanguage\': \'Python 3\', \'programming_language\': None, \'submission_id\': None, \'verdict\': \'OK\'}]'
# print(json.loads(tmp, strict=False))

def csv_to_json(csv_filename, json_filename):
    # Open the CSV file for reading
    with open(csv_filename, mode='r', newline='', encoding='utf-8') as csv_file:
        # Read the CSV file
        csv_reader = csv.DictReader(csv_file)
        
        # Convert the CSV data to a list of dictionaries
        data = [row for row in csv_reader]
    
    # Write the list of dictionaries to a JSON file
    with open(json_filename, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)  # indent=4 for pretty printing

# output_json = 'filtered_codeforces_train_rows.json'
# csv_to_json(input_filename, output_json)

input_json = 'solutions_py_decontaminated.json'
filtered_json = 'filtered_codeforces_train_rows.json'

def remove_unwanted_keys_from_file(json_filename, output_json_filename, keys_to_remove):
    # Open the original JSON file
    with open(json_filename, 'r', encoding='utf-8') as file:
        # Load the data from the JSON file
        data = json.load(file)
    
    # Iterate through the list of JSON objects and remove unwanted keys
    for json_obj in data:
        if isinstance(json_obj, str):
            try:
                json_obj = json.loads(json_obj)  # Convert string to a dictionary (JSON object)
            except json.JSONDecodeError:
                print(f"Error decoding JSON string: {json_obj}")
                
        for key in keys_to_remove:
            if key in json_obj:
                del json_obj[key]
    
    # Write the modified data back to the JSON file
    with open(output_json_filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
    print(f"Unwanted keys removed and the file '{json_filename}' has been updated to {output_json_filename}.")

keys_to_remove = ['prompt', 'generation', 'api_metadata', 'messages', 
                  'generated_tests', 'private_tests', 'public_tests', 'public_tests_ms']

# Call the function to remove unwanted keys
# remove_unwanted_keys_from_file(input_json, filtered_json, keys_to_remove)

def extract_programming_languages(json_filename):
    programming_languages = set()  # To store unique programming languages
    
    with open(json_filename, mode='r', encoding='utf-8') as file:
        # Load the JSON data
        data = json.load(file)
        
        # Iterate through each item in the JSON list (root element is assumed to be a list)
        for entry in data:
            # Check if 'accepted_solutions' is in the entry and is a list
            if 'accepted_solutions' in entry:
                accepted_solution = entry['accepted_solutions']
                accepted_solutions = json.loads(accepted_solution, strict=False)
                # Iterate through each solution in the 'accepted_solutions' list
                for solution in accepted_solutions:
                    # Extract the 'programmingLanguage' field and add it to the set
                    if 'programmingLanguage' in solution:
                        programming_languages.add(solution['programmingLanguage'])
    
    return programming_languages

# Example usage
# json_filename = 'filtered_codeforces_train_rows.json'
# unique_languages = extract_programming_languages(json_filename)
# print("Unique programming languages:", unique_languages)
