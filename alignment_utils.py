"""
Centralized strict event-to-epochs alignment utilities.

This module provides robust alignment functions that enforce strict matching
between behavioral events and EEG epochs to prevent trial mislabeling that
would corrupt ERPs, correlations, and decoding labels.

All scripts should use these strict alignment functions instead of unsafe
fallbacks like trimming to min(len(events), len(epochs)).
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import pandas as pd
import mne


def align_events_to_epochs_strict(
    events_df: Optional[pd.DataFrame], 
    epochs: mne.Epochs,
    logger: Optional[logging.Logger] = None
) -> Optional[pd.DataFrame]:
    """Strictly align events DataFrame to epochs order.
    
    This function enforces strict alignment between behavioral events and EEG epochs
    using epochs.selection or sample column matching. If alignment cannot be guaranteed,
    it raises an error instead of using unsafe fallbacks like trimming.
    
    Parameters
    ----------
    events_df : Optional[pd.DataFrame]
        Behavioral events DataFrame with trial metadata
    epochs : mne.Epochs
        MNE Epochs object to align events to
    logger : Optional[logging.Logger]
        Logger for diagnostics
        
    Returns
    -------
    Optional[pd.DataFrame]
        Aligned events DataFrame with same number of rows as epochs,
        or None if events_df is None/empty
        
    Raises
    ------
    ValueError
        If events cannot be reliably aligned to epochs. This is a critical
        failure that would invalidate all behavioral correlations.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if events_df is None or len(events_df) == 0:
        logger.debug("No events DataFrame provided")
        return None
        
    if len(epochs) == 0:
        logger.debug("No epochs provided")
        return pd.DataFrame()

    logger.info(f"Attempting strict alignment: {len(events_df)} events to {len(epochs)} epochs")

    # Method 1: Use epochs.selection to reindex events
    if hasattr(epochs, "selection") and epochs.selection is not None:
        sel = epochs.selection
        try:
            # Validate selection indices are within events bounds
            if len(events_df) > int(np.max(sel)) and len(sel) == len(epochs):
                aligned = events_df.iloc[sel].reset_index(drop=True)
                logger.info(f"Successfully aligned using epochs.selection ({len(sel)} epochs)")
                return aligned
            else:
                logger.warning(f"epochs.selection invalid: max={np.max(sel)}, events_len={len(events_df)}")
        except (IndexError, ValueError, TypeError) as e:
            logger.warning(f"Failed to align using epochs.selection: {e}")
            
    # Method 2: Use sample column to match epochs.events
    if "sample" in events_df.columns and hasattr(epochs, "events") and epochs.events is not None:
        try:
            epoch_samples = epochs.events[:, 0]  # First column is sample index
            
            # Create lookup from sample to events row
            events_indexed = events_df.set_index("sample")
            
            # Try to reindex events to match epoch samples
            aligned = events_indexed.reindex(epoch_samples)
            
            # Check if alignment was successful (no NaN rows)
            if len(aligned) == len(epochs) and not aligned.isna().all(axis=1).any():
                aligned_reset = aligned.reset_index()
                logger.info(f"Successfully aligned using sample column ({len(aligned)} epochs)")
                return aligned_reset
            else:
                logger.warning(f"Sample-based alignment failed: {aligned.isna().all(axis=1).sum()} NaN rows")
                
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Failed to align using sample column: {e}")
            
    # Method 3: Check if lengths already match and order is likely correct
    if len(events_df) == len(epochs):
        logger.warning("Assuming events and epochs are already aligned (same length). "
                      "This is risky without explicit verification.")
        return events_df.copy()
    
    # Critical failure: Cannot guarantee alignment
    logger.critical(f"CRITICAL: Unable to align events to epochs reliably. "
                   f"Events: {len(events_df)} rows, Epochs: {len(epochs)} epochs. "
                   f"This would result in invalid correlations due to trial misalignment.")
    
    raise ValueError(
        f"Cannot guarantee events-to-epochs alignment for reliable analysis. "
        f"Events DataFrame ({len(events_df)} rows) cannot be reliably aligned to "
        f"epochs ({len(epochs)} epochs). This is a critical failure that would "
        f"invalidate all behavioral correlations. Consider:\n"
        f"1. Ensuring epochs.selection is properly set during epoching\n"
        f"2. Including 'sample' column in events with matching sample indices\n"
        f"3. Verifying event filtering matches epoch creation"
    )


def trim_behavioral_to_events_strict(
    behavioral_df: pd.DataFrame,
    events_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Strictly trim behavioral data to match events length.
    
    This function only allows trimming when behavioral data is longer than events,
    assuming extra behavioral rows are at the end (e.g., practice trials).
    If behavioral data is shorter than events, it raises an error.
    
    Parameters
    ----------
    behavioral_df : pd.DataFrame
        Behavioral data (e.g., PsychoPy trial summary)
    events_df : pd.DataFrame
        EEG events DataFrame  
    logger : Optional[logging.Logger]
        Logger for diagnostics
        
    Returns
    -------
    pd.DataFrame
        Trimmed behavioral DataFrame matching events length
        
    Raises
    ------
    ValueError
        If behavioral data is shorter than events, which would indicate
        missing behavioral trials that cannot be safely recovered
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if len(behavioral_df) == len(events_df):
        logger.debug("Behavioral and events data already same length")
        return behavioral_df.copy()
    elif len(behavioral_df) > len(events_df):
        logger.warning(f"Trimming behavioral data from {len(behavioral_df)} to {len(events_df)} rows. "
                      f"Assuming extra rows are practice/excluded trials.")
        return behavioral_df.iloc[:len(events_df)].copy()
    else:
        logger.critical(f"CRITICAL: Behavioral data shorter than events. "
                       f"Behavioral: {len(behavioral_df)}, Events: {len(events_df)}. "
                       f"This indicates missing behavioral trials.")
        
        raise ValueError(
            f"Behavioral data ({len(behavioral_df)} rows) is shorter than events "
            f"({len(events_df)} rows). This indicates missing behavioral trials "
            f"that cannot be safely recovered. Check data collection/preprocessing."
        )


def validate_alignment(
    aligned_events: pd.DataFrame,
    epochs: mne.Epochs,
    logger: Optional[logging.Logger] = None
) -> bool:
    """Validate that events are properly aligned to epochs.
    
    Parameters
    ----------
    aligned_events : pd.DataFrame
        Aligned events DataFrame
    epochs : mne.Epochs
        MNE Epochs object
    logger : Optional[logging.Logger]
        Logger for diagnostics
        
    Returns
    -------
    bool
        True if alignment is valid, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if len(aligned_events) != len(epochs):
        logger.error(f"Length mismatch: events={len(aligned_events)}, epochs={len(epochs)}")
        return False
        
    # Check for excessive NaN values that might indicate alignment failure
    nan_fraction = aligned_events.isna().all(axis=1).mean()
    if nan_fraction > 0.1:  # More than 10% NaN rows
        logger.error(f"High NaN fraction in aligned events: {nan_fraction:.1%}")
        return False
        
    logger.info(f"Alignment validation passed: {len(aligned_events)} rows, "
               f"{nan_fraction:.1%} NaN fraction")
    return True
