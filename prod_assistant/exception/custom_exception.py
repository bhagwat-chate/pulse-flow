# prod_assistant/exception/custom_exception.py

import sys
import traceback
from typing import Optional


class ProductAssistantException(Exception):
    def __init__(self, error_message: str | BaseException, error_details: Optional[object] = None):
        norm_msg = str(error_message)

        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()

        elif hasattr(error_details, "exc_info") and callable(error_details.exc_info):
            exc_type, exc_value, exc_tb = error_details.exc_info()

        elif isinstance(error_details, BaseException):
            exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__

        else:
            exc_type, exc_value, exc_tb = sys.exc_info()

        # Traverse to last traceback frame
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        if exc_type and exc_tb:
            self.traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = "No traceback available"

    def __str__(self):
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: [{self.error_message}]"
        return f"{base}\nTraceback:\n{self.traceback_str}" if self.traceback_str else base

    def __repr__(self):
        return f"ProductAssistantException(file={self.file_name}, line={self.lineno}, message={self.error_message!r})"
