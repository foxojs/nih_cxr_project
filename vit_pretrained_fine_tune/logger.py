import logging
import os

# Define log directory and filename
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the log directory exists
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,  # Save logs to file
    filemode="a",  # Append mode ('w' to overwrite each run)
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO  # Change to DEBUG for more details
)

# Create logger instance
logger = logging.getLogger("MainLogger")

# Optionally, add a console handler to see logs in terminal
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
logger.addHandler(console_handler)
