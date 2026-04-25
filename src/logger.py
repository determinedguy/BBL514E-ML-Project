import sys
from pathlib import Path

class DualLogger:
    def __init__(self, log_filename: str):
        self.terminal = sys.stdout
        
        # Create a logs directory in the project root if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.log_path = log_dir / log_filename
        # Open the file in append mode ('a') so we don't overwrite previous runs
        self.log_file = open(self.log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Forces it to write immediately, rather than waiting for the script to finish

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()