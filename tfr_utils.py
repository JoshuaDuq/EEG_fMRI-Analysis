"""
Time-frequency analysis utilities with improved spectral resolution.

This module provides utilities for robust TFR computation that avoid
poor spectral resolution at low frequencies by enforcing minimum n_cycles.
"""

from __future__ import annotations

import numpy as np
from typing import Union, Optional
import logging


def compute_adaptive_n_cycles(
    freqs: Union[np.ndarray, list],
    cycles_factor: float = 2.0,
    min_cycles: float = 3.0,
    max_cycles: Optional[float] = None
) -> np.ndarray:
    """Compute adaptive n_cycles with minimum floor to avoid poor spectral resolution.
    
    The standard approach of n_cycles = freqs / factor yields too few cycles at low
    frequencies (e.g., 4 Hz / 3 = 1.33 cycles), causing poor spectral resolution
    and bias in theta/low-alpha bands.
    
    This function enforces a minimum number of cycles while maintaining the
    adaptive scaling for higher frequencies.
    
    Parameters
    ----------
    freqs : array-like
        Frequency array in Hz
    cycles_factor : float, default 2.0
        Factor to divide frequencies by for base n_cycles calculation
        Reduced from 3.0 to improve low-frequency resolution
    min_cycles : float, default 3.0
        Minimum number of cycles to ensure stable spectral estimates
    max_cycles : float, optional
        Maximum number of cycles to cap high-frequency resolution
        
    Returns
    -------
    np.ndarray
        Array of n_cycles values, same length as freqs
        
    Examples
    --------
    >>> freqs = np.array([1, 4, 8, 30, 80])
    >>> n_cycles = compute_adaptive_n_cycles(freqs)
    >>> print(n_cycles)  # [3.0, 3.0, 4.0, 15.0, 40.0]
    """
    freqs = np.asarray(freqs, dtype=float)
    
    # Base calculation: freqs / factor
    base_cycles = freqs / cycles_factor
    
    # Apply minimum floor
    n_cycles = np.maximum(base_cycles, min_cycles)
    
    # Apply maximum cap if specified
    if max_cycles is not None:
        n_cycles = np.minimum(n_cycles, max_cycles)
    
    return n_cycles


def log_tfr_resolution(
    freqs: np.ndarray, 
    n_cycles: np.ndarray,
    sfreq: float,
    logger: Optional[logging.Logger] = None
) -> None:
    """Log TFR resolution parameters for diagnostics.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz
    n_cycles : np.ndarray
        Number of cycles array
    sfreq : float
        Sampling frequency in Hz
    logger : Optional[logging.Logger]
        Logger for output
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Calculate time resolution for each frequency
    time_res = n_cycles / freqs  # Time resolution in seconds
    
    # Calculate frequency resolution for each frequency  
    freq_res = freqs / n_cycles  # Frequency resolution in Hz
    
    logger.info("TFR Resolution Summary:")
    logger.info(f"  Frequency range: {freqs.min():.1f} - {freqs.max():.1f} Hz")
    logger.info(f"  n_cycles range: {n_cycles.min():.1f} - {n_cycles.max():.1f}")
    logger.info(f"  Time resolution: {time_res.min():.3f} - {time_res.max():.3f} s")
    logger.info(f"  Frequency resolution: {freq_res.min():.2f} - {freq_res.max():.2f} Hz")
    
    # Check for problematic low-frequency resolution
    low_freq_mask = freqs <= 8  # Theta and low alpha
    if np.any(low_freq_mask):
        low_cycles = n_cycles[low_freq_mask]
        if np.any(low_cycles < 2.5):
            logger.warning(f"Low n_cycles detected in theta/alpha: min={low_cycles.min():.1f}")
        else:
            logger.info(f"Good low-frequency resolution: min n_cycles={low_cycles.min():.1f}")


def validate_tfr_parameters(
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    sfreq: float,
    logger: Optional[logging.Logger] = None
) -> bool:
    """Validate TFR parameters for reliable spectral estimates.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz
    n_cycles : np.ndarray
        Number of cycles array
    sfreq : float
        Sampling frequency in Hz
    logger : Optional[logging.Logger]
        Logger for warnings
        
    Returns
    -------
    bool
        True if parameters are acceptable, False if problematic
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    issues = []
    
    # Check for insufficient n_cycles at any frequency
    if np.any(n_cycles < 2.0):
        issues.append(f"n_cycles too low: min={n_cycles.min():.1f}")
    
    # Check Nyquist constraint
    if np.any(freqs >= sfreq / 2):
        issues.append(f"Frequencies above Nyquist: max_freq={freqs.max():.1f}, Nyquist={sfreq/2:.1f}")
    
    # Check time resolution vs epoch length (assume reasonable epoch length)
    max_time_res = np.max(n_cycles / freqs)
    if max_time_res > 2.0:  # More than 2 seconds time resolution
        issues.append(f"Excessive time resolution: max={max_time_res:.1f}s")
    
    if issues:
        for issue in issues:
            logger.warning(f"TFR parameter issue: {issue}")
        return False
    else:
        logger.info("TFR parameters validation passed")
        return True
