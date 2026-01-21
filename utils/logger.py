"""
Logger Module
"""

import logging
import os
from datetime import datetime

# Directory to store log files
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Create log file with current date in filename
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

# Configure basic logging settings for the application
logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level = logging.INFO
)

def get_logger(name):
    """
    Creates abd returbs a logger instance with the specified name.

    This function provides a consistent way to obtain logger instances
    throughout the application, ensuring that all loggers inherit the basic
    configuration (level, file handler, format) defined in this module.

    Args:
        name (str): The name for the logger, typically '__name__' from the calling module
        to include the module path in the log records.
    
    Returns:
        logging.logger: A configured logger instance.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing Started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger