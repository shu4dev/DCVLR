"""
Logging utilities for the Team-1 Data Synthesis Pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration for the pipeline.

    Args:
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Logging to file: {log_path}")

    logging.info("Logging setup complete")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name of the logger

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
