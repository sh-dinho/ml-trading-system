import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.utils.paths import logs_path, ensure_dir


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 5_000_000,  # 5MB
    backup_count: int = 3,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Features:
    - Prevents duplicate handlers
    - Rotating file logs
    - Console + file output
    - Safe for repeated calls
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # =========================
    # Console Handler
    # =========================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # =========================
    # File Handler (Rotating)
    # =========================
    log_directory = ensure_dir(logs_path())
    file_name = log_file or f"{name}.log"
    file_path = log_directory / file_name

    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    file_handler.setLevel(logging.DEBUG)

    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )

    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger