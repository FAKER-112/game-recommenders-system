"""
Logging Configuration Module

This script configures a centralized logging system for the application.
It sets up a rotating file handler and a console handler to ensure that logs are
both persistent and visible in real-time.

Logic of Operation:
1.  **Directory Setup**: Automatically creates a `logs/` directory in the project root if it doesn't exist.
2.  **Handler Configuration**:
    - **RotatingFileHandler**: Writes logs to a file with a max size (5MB) and keeps backups.
      Uses UTF-8 encoding to support special characters (including emojis).
    - **StreamHandler**: Outputs INFO-level logs to the console (stdout).
3.  **Singleton Pattern**: Initializes a global `logger` instance that can be imported and used
    throughout the entire application to maintain consistent logging formats.
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(log_dir_name="logs", logger_name="app_logger"):
    """
    Sets up a logger with a rotating file handler and a console handler.
    Ensures logs are saved relative to the project root.
    """
    # Get project root directory
    # parents[0] = utils, parents[1] = src, parents[2] = project_root
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / log_dir_name

    os.makedirs(log_dir, exist_ok=True)

    # Log file name with timestamp
    log_file = log_dir / f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Prevent adding handlers if they already exist (idempotency)
    if logger.hasHandlers():
        return logger

    # File handler (Rotating) with UTF-8
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",  # ✅ important for Unicode emojis
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Console handler with UTF-8
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize global logger
logger = setup_logger()

# Test emoji logging
if __name__ == "__main__":
    logger.info("✅ Logger initialized successfully!")
    logger.info("🧠 Testing emojis in logs...")
