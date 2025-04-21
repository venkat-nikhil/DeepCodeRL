import subprocess
import concurrent.futures
import os
import psutil
import threading
import time
import tempfile

def write_code_to_file(code_string, file_name='output.py'):
    with open(file_name, 'w') as file:
        file.write(code_string)
    print(f"Code has been written to {file_name}")


command = ['python', '-c', '''n, m = map(int, input().split())\nmydiv = n // m\nmymod = n % m\nmylist = [ mydiv for _ in range(m) ]\nif  mymod == 0 :\n    out = ' '.join(str(i) for i in mylist)\nelse :\n    for i in mylist :\n        mylist[mylist.index(i)] += 1\n        mymod -= 1\n        if mymod == 0 :\n            out = ' '.join(str(i) for i in mylist)\n            break\nprint(out)\n''']
# command = [
#     'python', '-c', 
#     '''n, m = map(int, input().split())
# mydiv = n // m
# mymod = n % m
# mylist = [mydiv for _ in range(m)]
# if mymod == 0:
#     out = ' '.join(str(i) for i in mylist)
# else:
#     for i in mylist:
#         mylist[mylist.index(i)] += 1
#         mymod -= 1
#         if mymod == 0:
#             out = ' '.join(str(i) for i in mylist)
#             break
# print(out)
# '''
# ]
if __name__ == '__main__':
    input_data = "12 3"
    env = os.environ.copy()

    result = subprocess.run(command, input=input_data, capture_output=True, text=True, env=env)
    print(f"STDOUT: {result.stdout.strip()}\nSTDERR: {result.stderr.strip()}")


# Create a temporary Python file with the code
# with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.py') as temp_file:
#     temp_file.write(command[2])
#     temp_filename = temp_file.name  # Get the path of the temporary file


# # Input data for the script
# input_data = "12 3"

# # Run the Python file via subprocess
# result = subprocess.run(
#     ['python', "generated_code.py"], 
#     input=input_data, 
#     capture_output=True, 
#     text=True, 
#     env=env,
#     timeout=5  # Timeout for subprocess execution
# )

# # Clean up the temporary file
# os.remove(temp_filename)

# # Print the output
# print(f"STDOUT: {result.stdout.strip()}")
# print(f"STDERR: {result.stderr.strip()}")