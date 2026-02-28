import logging
from pathlib import Path

def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Creates a logger that writes to both console and a file.

    Args:
        name (str): Name of the logger.
        log_dir (str): Directory where log files will be stored.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(f"{log_dir}/{name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger