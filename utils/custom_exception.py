"""
Custom Exception Module
"""

import os
import sys
import traceback

class CustomException(Exception):
    """
    Custom Exception Class that extends the base Exception class

    This class enhances the standard python exception by providing additional context about where the error occured,including the filename and line number.

    Attributes:
        error_message (str): Detailed error_message
    """

    def __init__(self, error_message, error_detail:sys):
        """
        Initialize the CustomException with an error message and error details.

        Args:
            error_message (str): Basic error message
            error_detail (sys): System information containing exception details
        """
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)
    
    @staticmethod
    def get_detailed_error_message(error_message, error_detail:sys):
        """
        Formats a detailed error message with file name and line number information.
        
        This static method extracts information about where the exception occured and formats it into a more destructive error message.

        Args:
            error_message (str): Basic error message.
            error_detail (sys): System information containing exception details.
        
        Returns:
            str: Formatted error message including file name and line number.
        """
        _,_, exc_tb = traceback.sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error in {file_name}, line {line_number} : {error_message}"
        return error_message

    def __str__(self):
        """
        Returns the string representation of the exception.

        This method is called when the eception is converted to a string, such as when printed or logged.

        Returns:
            str: The detailed error message.
        """
        return self.error_message