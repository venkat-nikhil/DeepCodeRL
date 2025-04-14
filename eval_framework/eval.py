import subprocess
import concurrent.futures
import os
import psutil
import threading
import time


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

def execute_command_with_input(command_input_pair):
    command, input_data = command_input_pair
    return run_subprocess(command, input_data)

# List of commands to run as subprocesses
commands_with_inputs  = [
    (['python', '-c', 'import torch; print(torch.__version__)'], None),
    (['python', '-c', 'import torch; print(torch.cuda.is_available())'], None),
    (['python', '-c', 'import os; print(os.getenv("MY_VAR"))'], None),
    (['python', '-c', 'import os; print(os.getenv("PYTHONPATH"))'], None),
    (['python', '-c', '''n, m = map(int, input().split())\nmydiv = n // m\nmymod = n % m\nmylist = [ mydiv for _ in range(m) ]\nif  mymod == 0 :\n    out = ' '.join(str(i) for i in mylist)\nelse :\n    for i in mylist :\n        mylist[mylist.index(i)] += 1\n        mymod -= 1\n        if mymod == 0 :\n            out = ' '.join(str(i) for i in mylist)\n            break\nprint(out)\n'''], 
     "12 3")
]

# Function to track subprocesses dynamically in real time
class ProcessTracker:
    def __init__(self):
        self.initial_processes = set()
        self.final_processes = set()
        self.subprocesses_created = 0

    def track_processes(self):
        """ Track processes in real-time, counting new subprocesses """
        self.initial_processes = {p.info['pid'] for p in psutil.process_iter(['pid'])}
        
        while self.subprocesses_created == 0:  # Keep tracking until subprocesses are detected
            time.sleep(0.05)  # Check every 0.5 seconds
            self.final_processes = {p.info['pid'] for p in psutil.process_iter(['pid'])}
            new_processes = self.final_processes - self.initial_processes
            if new_processes:
                self.subprocesses_created = len(new_processes)
                print(f"New subprocesses created: {self.subprocesses_created}")
            else:
                print("Waiting for subprocesses...")

    def get_subprocesses_created(self):
        return self.subprocesses_created


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
        results = list(executor.map(execute_command_with_input, commands_with_inputs))  # 2 is the timeout value for each command

    # Output the results of each subprocess
    for result in results:
        if isinstance(result, str) and result.startswith("Command"):
            print(f"TIMEOUT ERROR: result")  # Timeout error message
        else:
            # If it's a subprocess.CompletedProcess object, print stdout and stderr
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

    # Print the number of subprocesses created during the execution
    # print(f"Total number of subprocesses created (child + grandchildren): {tracker.get_subprocesses_created()}")
