"""
Lightweight logging utilities for EEG pipeline scripts.

Provides consistent subject-level and group-level loggers that write both
to console and to per-run log files under derivatives.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

try:
    # Local import within package context
    from .config_loader import load_config, get_legacy_constants
except Exception:  # pragma: no cover
    from config_loader import load_config, get_legacy_constants  # type: ignore


_config = load_config()
_constants = get_legacy_constants(_config)
DERIV_ROOT: Path = Path(_constants["DERIV_ROOT"])  # type: ignore[index]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_subject_logger(name: str, subject: str, log_file_name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a subject-scoped logger writing to derivatives/sub-<ID>/eeg/logs.

    The logger avoids duplicate handlers across repeated calls.
    """
    logger = logging.getLogger(f"{name}_sub_{subject}")
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    log_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "logs"
    _ensure_dir(log_dir)
    file_name = log_file_name or f"{name}.log"
    fh = logging.FileHandler(log_dir / file_name, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_group_logger(name: str, log_file_name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a group-level logger writing to derivatives/group/eeg/logs."""
    logger = logging.getLogger(f"{name}_group")
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    log_dir = DERIV_ROOT / "group" / "eeg" / "logs"
    _ensure_dir(log_dir)
    file_name = log_file_name or f"{name}.log"
    fh = logging.FileHandler(log_dir / file_name, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


__all__ = [
    "get_subject_logger",
    "get_group_logger",
]

