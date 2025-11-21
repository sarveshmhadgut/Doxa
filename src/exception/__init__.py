import sys
import logging
from typing import Any


def error_message_details(error: Exception, error_details: Any) -> str:
    """
    Formats a detailed error message including the file name, line number, and error description.

    This function extracts traceback information to create a human-readable error string.
    It logs the error at the ERROR level without including the full traceback to avoid clutter.

    Args:
        error (Exception): The exception instance or error object.
        error_details (Any): Typically sys module or exc_info tuple providing traceback details.

    Returns:
        str: Formatted error message string.
    """
    # Extract traceback information
    _, _, exec_tb = error_details.exc_info()
    
    # Check if traceback is available
    if exec_tb is not None:
        file_name: str = exec_tb.tb_frame.f_code.co_filename
        line_number: int = exec_tb.tb_lineno
        error_message: str = (
            f"Error occurred in python script: [{file_name}] at line number [{line_number}: {str(error)}]"
        )
    else:
        # Fallback when no traceback is available
        error_message: str = f"Error occurred: {str(error)}"
    
    logging.error(error_message)
    return error_message


class MyException(Exception):
    """
    Custom exception class that captures and formats detailed error information.

    This class extends the built-in Exception to include file, line, and error details
    in its string representation, making it easier to debug issues.

    Attributes:
        error_message (str): The formatted error message.
    """

    def __init__(self, error_message: str, error_details: Any) -> None:
        """
        Initializes the custom exception with formatted error details.

        Args:
            error_message (str): The base error message or description.
            error_details (Any): Typically sys module or exc_info tuple for traceback.
        """
        super().__init__(error_message)
        self.error_message: str = error_message_details(error_message, error_details)

    def __str__(self) -> str:
        """
        Returns the string representation of the exception (the detailed error message).

        Returns:
            str: The formatted error message.
        """
        return self.error_message
