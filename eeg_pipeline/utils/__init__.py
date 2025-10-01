"""
Utility modules for the EEG pipeline.

This package contains shared utility functions for:
- alignment_utils: Event-epoch alignment and validation
- config_loader: Configuration loading and management
- io_utils: File I/O operations for BIDS data
- logging_utils: Logging setup and utilities
- roi_utils: Region of interest definitions and operations
- tfr_utils: Time-frequency analysis utilities
- raw_to_bids: Raw EEG to BIDS conversion
"""

__all__ = [
    "alignment_utils",
    "config_loader",
    "io_utils",
    "logging_utils",
    "roi_utils",
    "tfr_utils",
    "raw_to_bids",
]
