import subprocess
import concurrent.futures
import os
from typing import Tuple
import psutil
import threading
import time

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
            time.sleep(0.05)  # Check every 0.05 seconds
            self.final_processes = {p.info['pid'] for p in psutil.process_iter(['pid'])}
            new_processes = self.final_processes - self.initial_processes
            if new_processes:
                self.subprocesses_created = len(new_processes)
                print(f"New subprocesses created: {self.subprocesses_created}")
            else:
                print("Waiting for subprocesses...")

    def get_subprocesses_created(self):
        return self.subprocesses_created
