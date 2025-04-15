import logging
import sys
from uvicorn.logging import ColourizedFormatter
from typing import Any

# Custom colorized formatter to apply colors specifically to log levels
class CustomColourizedFormatter(ColourizedFormatter):
    def format(self, record: logging.LogRecord) -> str:
        # Define color mappings for different log levels
        level_color_map = {
            "DEBUG": "\033[34m",    # Blue
            "INFO": "\033[32m",     # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",    # Red
            "CRITICAL": "\033[41m", # Red background
        }

        # Reset color
        reset = "\033[0m"

        # Apply color to the log level name
        record.levelname = f"{level_color_map.get(record.levelname, '')}{record.levelname}{reset}"

        # Format the log message using the parent class's format method
        return super().format(record)

def get_logger(name: str) -> logging.Logger:
    """Creates a logger object

    Args:
        name (str): name given to the logger

    Returns:
        logging.Logger: logger object to be used for logging 
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if already exists
    if not logger.hasHandlers():
        # Create console handler and set level to debug
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)

        # Create a custom formatter with colored log levels
        formatter = CustomColourizedFormatter(
            "{asctime} | {levelname:<8} | {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
            use_colors=True
        )

        # Add formatter to the handler
        ch.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(ch)

    return logger

# Usage example
if __name__ == "__main__":
    logger = get_logger('MyLogger')

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
