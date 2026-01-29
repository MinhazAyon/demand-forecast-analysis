# core/utils/logger.py

# --- Standard Python imports ---
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

def setup_logging(
    *,
    log_path: str = "logs/app.log",
    console_level: str = "INFO",
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> None:
    """
    Configures the root logger to log messages to both the console and a rotating file.

    This function is **safe to call multiple times**. If logging is already configured,
    it will not make changes or add duplicate handlers.

    Args:
        log_path (str): Log file path (default "logs/app.log").
        console_level (str): Console log level (default "INFO").
        max_bytes (int): Max log file size before rotation (default 5 MB).
        backup_count (int): Number of backup log files to keep (default 5).

    Returns:
        None
    """
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    fmt = "[%(asctime)s] - %(levelname)s - %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # File: DEBUG+
    fh = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt))

    # Console: configurable
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(fmt, datefmt))

    root.addHandler(fh)
    root.addHandler(ch)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a logger with the specified name, without adding any handlers.

    Args:
        name (Optional[str]): The name for the logger. If None, the default is the module name.

    Returns:
        logging.Logger: A logger object that can be used for logging messages.
    """
    return logging.getLogger(name or __name__)
