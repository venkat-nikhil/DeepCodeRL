import subprocess
import concurrent.futures
import os
from typing import Tuple
import psutil
import threading
import time

number_of_tests_mismatch_err = "Input and expected output test cases count mismatch."

venv_python = os.path.join('.venv', 'Scripts', 'python.exe')

env = os.environ.copy()  # Copy current environment
env['MY_VAR'] = 'some_value'

# Function to run a subprocess (for demonstration purposes)
def run_subprocess(command, input_data=None, timeout=5.0):
    try:
        # Run the command with the specified timeout and input data
        result = subprocess.run(command, input=input_data, capture_output=True, text=True, env=env, timeout=timeout)
        return result
    except subprocess.TimeoutExpired:
        return f"Command {command} timed out after {timeout} seconds"

def execute_command_with_input(command_input_pair, timeout=5.0):
    return run_subprocess(command_input_pair[0], command_input_pair[1], timeout=timeout)

def check_output(command, input=None, expected_output=None, timeout=5.0) -> Tuple[bool, str]:
    logger_str = ""
    run_subproc_result = run_subprocess(command, input, timeout=timeout)
    if isinstance(run_subproc_result, str) and run_subproc_result.startswith("Command"):
        logger_str += f"TIMEOUT ERROR: {run_subproc_result}"  # Timeout error message
        return False, logger_str
    else:
        # If it's a subprocess.CompletedProcess object, print stdout and stderr
        logger_str += f"STDOUT: {run_subproc_result.stdout}\n"
        logger_str += f"STDERR: {run_subproc_result.stderr}"
    if expected_output is None:
        return True, logger_str
    actual_output = run_subproc_result.stdout.strip()
    if actual_output.strip() == expected_output.strip():
        return True, logger_str
    else:
        return False, logger_str + f"\nExpected Output: {expected_output.strip()}\nActual Output: {actual_output.strip()}"

def test_code_with_outputs(code_with_tests, timeout=5.0):
    results = []
    command = code_with_tests[0]
    input_tests = code_with_tests[1]
    expected_output_tests = code_with_tests[2]

    if input_tests is None:
        if expected_output_tests is None:
            check, logger_str = check_output(command, timeout=timeout)
            results.append([check, logger_str])
        else:
            for expected_output in expected_output_tests:
                check, logger_str = check_output(command, expected_output=expected_output, timeout=timeout)
                results.append([check, logger_str])
    else:
        if len(input_tests) != len(expected_output_tests):
            results.append([False, number_of_tests_mismatch_err])
        else:
            for input_test, expected_output in zip(input_tests, expected_output_tests):
                check, logger_str = check_output(command, input=input_test, expected_output=expected_output, timeout=timeout)
                results.append([check, logger_str])

    return results

# List of commands to run as subprocesses
commands_with_inputs  = [
    [['python', '-c', 'import torch; print(torch.__version__)'], None, None],
    [['python', '-c', 'import torch; print(torch.cuda.is_available())'], None, None],
    [['python', '-c', 'import os; print(os.getenv("MY_VAR"))'], None, None],
    [['python', '-c', 'import os; print(os.getenv("PYTHONPATH"))'], None, None],
    [['python', '-c', '''n, m = map(int, input().split())\nmydiv = n // m\nmymod = n % m\nmylist = [ mydiv for _ in range(m) ]\nif  mymod == 0 :\n    out = ' '.join(str(i) for i in mylist)\nelse :\n    for i in mylist :\n        mylist[mylist.index(i)] += 1\n        mymod -= 1\n        if mymod == 0 :\n            out = ' '.join(str(i) for i in mylist)\n            break\nprint(out)\n'''], 
     ["12 3"],
     ["4 4 4"]]
]

test_inputs  = [
    [['python', '-c', '''def solve (n,seq) :\r\n    seq.sort()\r\n    start = 1\r\n    moves = 0\r\n    while start <= n :\r\n        if seq[start-1] != start :\r\n            moves += abs(seq[start-1] - start)\r\n        start += 1\r\n        \r\n        \r\n    return moves\r\n    \r\n    \r\nn = int(input())\r\nseq = list(map(int,input().split()))\r\n\r\nprint (solve(n,seq))\r\n\r\n  \r\n        \r\n\r\n    \r\n\r\n\r\n    \r\n   '''], 
     ["2\n3 0", "3\n-1 -1 2"], 
     ["2", "6"]],
    [['python', '-c', '''n, m = map(int, input().split())\nmydiv = n // m\nmymod = n % m\nmylist = [ mydiv for _ in range(m) ]\nif  mymod == 0 :\n    out = ' '.join(str(i) for i in mylist)\nelse :\n    for i in mylist :\n        mylist[mylist.index(i)] += 1\n        mymod -= 1\n        if mymod == 0 :\n            out = ' '.join(str(i) for i in mylist)\n            break\nprint(out)\n'''], 
     ["12 3", "15 4", "18 7"],
     ["4 4 4", "3 4 4 4", "2 2 2 3 3 3 3"]] 
]

if __name__ == '__main__':
    # Instantiate the ProcessTracker
    # tracker = ProcessTracker()

    # # Start tracking processes in a background thread
    # tracking_thread = threading.Thread(target=tracker.track_processes)
    # tracking_thread.start()

    # Wait for the process tracking thread to finish
    # tracking_thread.join()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit each command separately to the executor with a 2-second timeout for the command execution
        results = list(executor.map(test_code_with_outputs, test_inputs))  # 2 is the timeout value for each command

    # Output the results of each subprocess
    print(f'Number of child processes created: {len(results)}\n')
    total_subprocs = 0
    for result_execs in results:
        total_subprocs += len(result_execs)
        for result in result_execs:
            check, logger_str = result
            print(f"Code execution result: {check}")
            print(logger_str)
    print(f'Total number of subprocesses created: {total_subprocs}\n')
    # Print the number of subprocesses created during the execution
    # print(f"Total number of subprocesses created (child + grandchildren): {tracker.get_subprocesses_created()}")
