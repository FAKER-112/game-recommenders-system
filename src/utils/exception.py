"""
Custom Exception Module

This script defines the CustomException class and utility functions for consistent error handling
across the application. It captures detailed traceback information, including filenames and line
numbers, and ensures all errors are logged automatically upon instantiation.

Logic of Operation:
1.  **Error Detail Extraction**: Helper function `error_message_details` inspects the current
    system execution state (`sys.exc_info`) to pinpoint exactly where an error occurred.
2.  **Custom Exception Class**:
    - Inherits from the built-in `Exception` class.
    - Automatically invokes the logger to record the error details and full traceback when raised.
    - Provides a string representation that includes the formatted error location info.
"""

import sys
import traceback
from src.utils.logger import logger


def error_message_details(error: Exception) -> str:
    """Extract detailed error information including file name and line number."""
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
        return f"Error in file [{file_name}] at line [{line_no}]: {type(error).__name__} - {error}"
    return f"{type(error).__name__}: {error}"


class CustomException(Exception):
    """Custom exception class that captures detailed error information."""

    def __init__(self, error: Exception):
        super().__init__(str(error))
        self.error_message = error_message_details(error)
        logger.error(self.error_message)
        logger.error("Full traceback:\n%s", traceback.format_exc())

    def __str__(self):
        return self.error_message
