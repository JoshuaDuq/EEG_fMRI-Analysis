"""
Time-frequency analysis utilities with improved spectral resolution and
standardized units handling.

This module provides utilities for robust TFR computation and reading that
avoid poor spectral resolution at low frequencies and reduce the risk of
mixing baseline units (ratio/percent/logratio) across scripts.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple, Union

import mne
import numpy as np


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


def read_tfr_average_with_logratio(
    tfr_path: Union[str, "os.PathLike[str]"],
    baseline_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    require_sidecar: bool = True,
    allow_heuristics: bool = False,
    min_baseline_samples: int = 5,
) -> Optional["mne.time_frequency.AverageTFR"]:
    """Read a TFR and standardize to log10(power/baseline) units.

    - Reads first TFR from file, averaging epochs if needed.
    - If a JSON sidecar exists and indicates baseline mode, use it to convert
      ratio/percent to logratio.
    - If no sidecar is present and heuristics are allowed, try to infer mode.
    - Otherwise, apply baseline(logratio) using the provided baseline window.

    Returns None on failure or when unit ambiguity cannot be resolved under
    the provided policy flags.
    """
    try:
        read = getattr(mne.time_frequency, "read_tfrs", None)
        if read is None:
            if logger:
                logger.warning(f"read_tfrs unavailable for {tfr_path}")
            return None
        tfrs = read(str(tfr_path))
        if not isinstance(tfrs, list) or len(tfrs) == 0:
            if logger:
                logger.warning(f"No TFRs found in {tfr_path}")
            return None
        t = tfrs[0]
        # Average if epochs-level
        try:
            data = getattr(t, "data", None)
            if data is not None and getattr(data, "ndim", 0) == 4:
                t = t.average()
        except Exception:
            pass

        # Try sidecar to determine mode
        mode_detected: Optional[str] = None
        detected_by_sidecar = False
        sidecar_path = str(tfr_path).rsplit(".", 1)[0] + ".json"
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if bool(meta.get("baseline_applied", False)):
                mode_detected = str(meta.get("baseline_mode", "")).strip().lower() or None
                detected_by_sidecar = mode_detected is not None
                if logger:
                    logger.info(
                        f"{getattr(t, 'comment', '') or tfr_path}: sidecar baseline mode={mode_detected}"
                    )
        except FileNotFoundError:
            if require_sidecar and not allow_heuristics:
                if logger:
                    logger.error(
                        f"Missing TFR sidecar for {tfr_path} and heuristics disabled; refusing to load to avoid unit ambiguity."
                    )
                return None
        except (json.JSONDecodeError, OSError) as exc:
            if logger:
                logger.warning(f"Failed reading TFR sidecar for {tfr_path}: {exc}")

        # Fallback: inspect comment for hints
        if mode_detected is None:
            try:
                cmt = (getattr(t, "comment", None) or "").lower()
                if "logratio" in cmt:
                    mode_detected = "logratio"
                elif "percent" in cmt:
                    mode_detected = "percent"
                elif "ratio" in cmt:
                    mode_detected = "ratio"
            except Exception:
                pass

        # Unsupported modes are rejected to avoid mixing units
        if mode_detected is not None and mode_detected not in {"logratio", "ratio", "percent"}:
            if logger:
                logger.warning(
                    f"Unsupported baseline mode '{mode_detected}' for {tfr_path}; skipping."
                )
            return None

        # If no mode detected and heuristics allowed, try to infer
        if mode_detected is None and allow_heuristics:
            times = np.asarray(t.times)
            try:
                b_start, b_end = float(baseline_window[0]), float(baseline_window[1])
                if b_end > 0:
                    b_end = 0.0
                mask = (times >= b_start) & (times < b_end)
                if int(mask.sum()) >= int(min_baseline_samples):
                    base = t.data[:, :, mask]
                    base_flat = base[np.isfinite(base)]
                    if base_flat.size > 0:
                        med = float(np.median(base_flat))
                        if 0.5 <= med <= 1.5 and np.nanmin(base_flat) >= 0:
                            mode_detected = "ratio"
                        elif abs(med) < 0.2:
                            mode_detected = "logratio"
            except Exception:
                pass
            if logger:
                logger.warning(
                    f"{tfr_path}: baseline mode inferred heuristically as '{mode_detected}'."
                )

        # Standardize to logratio
        if mode_detected == "logratio":
            return t
        elif mode_detected == "ratio":
            if not detected_by_sidecar:
                if logger:
                    logger.warning(
                        f"{tfr_path}: detected 'ratio' without sidecar; refusing conversion (require_sidecar={require_sidecar})."
                    )
                return None
            try:
                t.data = np.log10(np.maximum(t.data, 1e-20))
                if hasattr(t, "comment"):
                    t.comment = (t.comment or "") + " | converted ratio->log10ratio"
                return t
            except Exception as exc:
                if logger:
                    logger.warning(f"Failed ratio->log10ratio conversion for {tfr_path}: {exc}")
                return None
        elif mode_detected == "percent":
            if not detected_by_sidecar:
                if logger:
                    logger.warning(
                        f"{tfr_path}: detected 'percent' without sidecar; refusing conversion (require_sidecar={require_sidecar})."
                    )
                return None
            try:
                ratio = 1.0 + (t.data / 100.0)
                t.data = np.log10(np.clip(ratio, 1e-20, np.inf))
                if hasattr(t, "comment"):
                    t.comment = (t.comment or "") + " | converted percent->log10ratio"
                return t
            except Exception as exc:
                if logger:
                    logger.warning(f"Failed percent->log10ratio conversion for {tfr_path}: {exc}")
                return None
        else:
            # Apply logratio baseline if nothing was applied
            times = np.asarray(t.times)
            try:
                b_start, b_end = float(baseline_window[0]), float(baseline_window[1])
                if b_end > 0:
                    b_end = 0.0
                mask_n = int(((times >= b_start) & (times < b_end)).sum())
                if mask_n < int(min_baseline_samples):
                    if logger:
                        logger.warning(
                            f"Insufficient baseline samples ({mask_n}) in window {baseline_window} for {tfr_path}."
                        )
                    return None
                t.apply_baseline(baseline=(b_start, b_end), mode="logratio")
                if hasattr(t, "comment"):
                    t.comment = (t.comment or "") + " | baseline(logratio) applied"
                return t
            except Exception as exc:
                if logger:
                    logger.warning(f"Failed to apply baseline(logratio) to {tfr_path}: {exc}")
                return None
    except Exception as exc:
        if logger:
            logger.warning(f"Failed loading/standardizing TFR at {tfr_path}: {exc}")
        return None


def save_tfr_with_sidecar(
    tfr: Union["mne.time_frequency.EpochsTFR", "mne.time_frequency.AverageTFR"],
    out_path: Union[str, "os.PathLike[str]"],
    baseline_window: Tuple[float, float],
    mode: str = "logratio",
    logger: Optional[logging.Logger] = None,
) -> None:
    """Save a TFR to HDF5 with a JSON sidecar describing baseline and units.

    The sidecar helps downstream code standardize units and avoid mixed modes.
    """
    from pathlib import Path

    try:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tfr.save(str(p), overwrite=True)
        sidecar = {
            "baseline_applied": True,
            "baseline_mode": str(mode),
            "units": ("log10ratio" if str(mode).lower() == "logratio" else str(mode)),
            "baseline_window": [float(baseline_window[0]), float(baseline_window[1])],
            "created_by": "tfr_utils.save_tfr_with_sidecar",
            "comment": getattr(tfr, "comment", None),
        }
        with open(p.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        if logger:
            logger.info(f"Saved TFR and sidecar: {p} (+ .json)")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save TFR with sidecar to {out_path}: {e}")
