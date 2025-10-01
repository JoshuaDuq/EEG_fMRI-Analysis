from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import logging
import json

import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath
import glob
import matplotlib.pyplot as plt
import math
 
# ==========================
# CONFIG
# Load centralized configuration from YAML
# ==========================
from config_loader import load_config, get_legacy_constants
try:
    from logging_utils import get_subject_logger
except Exception:  # pragma: no cover
    from .logging_utils import get_subject_logger  # type: ignore
from alignment_utils import align_events_to_epochs_strict, validate_alignment

# Shared I/O helpers (support both script and package execution)
try:
    from io_utils import (
        _find_clean_epochs_path as _find_clean_epochs_path,
        _load_events_df as _load_events_df,
        _align_events_to_epochs as _align_events_to_epochs,
        _pick_target_column as _pick_target_column,
    )
except Exception:  # pragma: no cover
    from .io_utils import (  # type: ignore
        _find_clean_epochs_path as _find_clean_epochs_path,
        _load_events_df as _load_events_df,
        _align_events_to_epochs as _align_events_to_epochs,
        _pick_target_column as _pick_target_column,
    )

# Load configuration
config = load_config()

# Extract legacy constants for backward compatibility
_constants = get_legacy_constants(config)
try:
    from io_utils import _ensure_derivatives_dataset_description
except Exception:  # pragma: no cover
    from .io_utils import _ensure_derivatives_dataset_description  # type: ignore
_ensure_derivatives_dataset_description()

PROJECT_ROOT = _constants["PROJECT_ROOT"]
BIDS_ROOT = _constants["BIDS_ROOT"]
DERIV_ROOT = _constants["DERIV_ROOT"]
SUBJECTS = _constants["SUBJECTS"]
TASK = _constants["TASK"]
FEATURES_FREQ_BANDS = _constants["FEATURES_FREQ_BANDS"]
CUSTOM_TFR_FREQS = _constants["CUSTOM_TFR_FREQS"]
BAND_COLORS = _constants.get("BAND_COLORS", {})
# Import TFR utilities for improved n_cycles calculation
from tfr_utils import (
    compute_adaptive_n_cycles,
    log_tfr_resolution,
    read_tfr_average_with_logratio,
    save_tfr_with_sidecar,
)

cycles_factor = float(config.get("time_frequency_analysis.tfr.n_cycles_factor", 2.0))
# Use improved n_cycles calculation with minimum floor, consistent with 02
CUSTOM_TFR_N_CYCLES = compute_adaptive_n_cycles(CUSTOM_TFR_FREQS, cycles_factor=cycles_factor, min_cycles=3.0)
CUSTOM_TFR_DECIM = _constants["CUSTOM_TFR_DECIM"]
DEFAULT_TASK = TASK
POWER_BANDS = _constants["POWER_BANDS"]
_plateau = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
PLATEAU_START = float(_plateau[0])
PLATEAU_END = float(_plateau[1])
TARGET_COLUMNS = _constants["TARGET_COLUMNS"]

# Strictness and TFR unit handling
STRICT_MODE = bool(config.get("analysis.strict_mode", True))
REQUIRE_TFR_SIDECAR = bool(config.get("feature_engineering.require_tfr_sidecar", True))
ALLOW_TFR_HEURISTICS = bool(config.get("feature_engineering.allow_tfr_heuristics", False))

# Minimum number of samples required in the baseline window
MIN_BASELINE_SAMPLES = int(config.get("feature_engineering.min_baseline_samples", 5))
# Baseline window for TFR computations (start, end) in seconds
TFR_BASELINE = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))

# Robust logging file name with fallback
LOG_FILE_NAME = config.get(
    "logging.file_names.feature_engineering", "03_feature_engineering.log"
)  # Name of the log file for this script

# Optional override for TFR spectrogram color limits (positive value)
TFR_SPECTROGRAM_VLIM = config.get("feature_engineering.tfr_spectrogram_vlim", None)
SAVE_TFR_WITH_SIDECAR = bool(config.get("feature_engineering.save_tfr_with_sidecar", False))


# -----------------------------------------------------------------------------
# Helper functions (duplicated from 02_time_frequency_analysis with light tweaks)
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)



def _robust_sym_vlim(
    arrs: "np.ndarray | list[np.ndarray]",
    q_low: float = 0.02,
    q_high: float = 0.98,
    cap: float = 0.25,
    min_v: float = 1e-6,
) -> float:
    """Compute robust symmetric vlim (positive scalar) centered at 0.

    Concatenates arrays, removes non-finite values, takes ``[q_low, q_high]``
    quantiles and returns the maximum absolute quantile capped by ``cap``. The
    result can be used with ``vmin=-v`` and ``vmax=+v``.
    """

    try:
        if isinstance(arrs, (list, tuple)):
            flat = np.concatenate([np.asarray(a).ravel() for a in arrs if a is not None])
        else:
            flat = np.asarray(arrs).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return cap
        lo = np.nanquantile(flat, q_low)
        hi = np.nanquantile(flat, q_high)
        v = float(max(abs(lo), abs(hi)))
        if not np.isfinite(v) or v <= 0:
            v = min_v
        return float(min(v, cap))
    except Exception:
        return cap

def _validate_baseline_indices(
    times: np.ndarray,
    baseline: Tuple[Optional[float], Optional[float]],
    min_samples: int = MIN_BASELINE_SAMPLES,
) -> Tuple[float, float, np.ndarray]:
    """Validate baseline window and return a time mask.

    Ensures the baseline interval ends before stimulus onset and
    contains at least ``min_samples`` samples.
    """
    b_start, b_end = baseline
    if b_start is None:
        b_start = float(times.min())
    if b_end is None:
        b_end = 0.0
    # Allow baseline ending exactly at 0.0 s (common convention)
    if b_end > 0:
        raise ValueError("Baseline window must end at or before 0 s")
    mask = (times >= b_start) & (times < b_end)
    if mask.sum() < min_samples:
        raise ValueError(
            f"Baseline window has {int(mask.sum())} samples; at least {min_samples} required"
        )
    return b_start, b_end, mask



## _find_clean_epochs_path imported from io_utils


def _find_tfr_path(subject: str, task: str) -> Optional[Path]:
    p1 = DERIV_ROOT / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_power_epo-tfr.h5"
    if p1.exists():
        return p1
    eeg_dir = DERIV_ROOT / f"sub-{subject}" / "eeg"
    if eeg_dir.exists():
        cands = sorted(eeg_dir.glob(f"sub-{subject}_task-{task}*_epo-tfr.h5"))
        if cands:
            return cands[0]
    subj_dir = DERIV_ROOT / f"sub-{subject}"
    if subj_dir.exists():
        for c in sorted(subj_dir.rglob(f"sub-{subject}_task-{task}*_epo-tfr.h5")):
            return c
    return None


def _save_tfr_with_sidecar(
    tfr, out_path: Path, baseline_window: Tuple[float, float], mode: str = "logratio", logger: Optional[logging.Logger] = None
) -> None:
    """Deprecated: use tfr_utils.save_tfr_with_sidecar instead."""
    _ensure_dir(out_path.parent)
    save_tfr_with_sidecar(tfr, out_path, baseline_window=baseline_window, mode=mode, logger=logger)


def _read_tfr_average_with_logratio(
    tfr_path: Path,
    baseline_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
) -> Optional["mne.time_frequency.AverageTFR"]:
    """Deprecated: use tfr_utils.read_tfr_average_with_logratio instead."""
    return read_tfr_average_with_logratio(
        tfr_path,
        baseline_window=baseline_window,
        logger=logger,
        require_sidecar=REQUIRE_TFR_SIDECAR,
        allow_heuristics=ALLOW_TFR_HEURISTICS,
        min_baseline_samples=MIN_BASELINE_SAMPLES,
    )

## _load_events_df imported from io_utils


def _setup_logging(subject: str) -> logging.Logger:
    """Backward-compatible wrapper using centralized logging utils."""
    return get_subject_logger("feature_engineering", subject, LOG_FILE_NAME)


# -----------------------------------------------------------------------------
# Feature extraction helpers
# -----------------------------------------------------------------------------

def _compute_tfr(
    epochs: mne.Epochs,
    freqs: np.ndarray = None,
    n_cycles: np.ndarray = None,
    decim: int = None,
    logger: Optional[logging.Logger] = None,
) -> "mne.time_frequency.EpochsTFR":
    """Compute EpochsTFR from cleaned epochs using Morlet wavelets (trial-level)."""
    if freqs is None:
        freqs = CUSTOM_TFR_FREQS
    if n_cycles is None:
        n_cycles = CUSTOM_TFR_N_CYCLES
    if decim is None:
        decim = CUSTOM_TFR_DECIM

    # Log TFR resolution for diagnostics
    if logger is not None:
        log_tfr_resolution(freqs, n_cycles, epochs.info['sfreq'], logger)

    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=decim,
        n_jobs=-1,
        picks="eeg",
        verbose=False,
    )
    # power is EpochsTFR
    return power

## _pick_target_column imported from io_utils


# Import strict alignment utilities

## _align_events_to_epochs imported from io_utils


def _time_mask(times: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    return (times >= tmin) & (times <= tmax)


def _freq_mask(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return (freqs >= fmin) & (freqs <= fmax)


def _extract_band_power_features(tfr, bands: List[str], logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """Compute mean power per (band, channel) within plateau window for each epoch.

    Returns a DataFrame shaped (n_trials, n_channels*len(bands)).
    """
    if tfr is None:
        return pd.DataFrame(), []

    # MNE reads back as a list
    if isinstance(tfr, list):
        tfr = tfr[0]

    # Expect data shape: (n_epochs, n_channels, n_freqs, n_times)
    data = tfr.data  # type: ignore[attr-defined]
    if data.ndim != 4:
        raise RuntimeError("TFR data does not have expected 4D shape (epochs, ch, f, t)")

    n_ep, n_ch, n_f, n_t = data.shape
    tmask = _time_mask(tfr.times, PLATEAU_START, PLATEAU_END)  # type: ignore[attr-defined]
    if not np.any(tmask):
        raise RuntimeError("No TFR time points in the specified plateau window.")

    features = []
    colnames: List[str] = []
    for band in bands:
        if band not in FEATURES_FREQ_BANDS:
            logger.warning(f"Band '{band}' not defined in config; skipping.")
            continue
        fmin, fmax = FEATURES_FREQ_BANDS[band]
        fmask = _freq_mask(tfr.freqs, fmin, fmax)  # type: ignore[attr-defined]
        if not np.any(fmask):
            logger.warning(f"TFR freqs contain no points in band '{band}' ({fmin}-{fmax} Hz)")
            # still create zeros to keep alignment predictable
            band_pow = np.zeros((n_ep, n_ch))
        else:
            band_pow = data[:, :, fmask, :][:, :, :, tmask].mean(axis=(2, 3))
        features.append(band_pow)
        colnames.extend([f"pow_{band}_{ch}" for ch in tfr.info["ch_names"]])  # type: ignore[attr-defined]

    if len(features) == 0:
        return pd.DataFrame(), []

    X = np.concatenate(features, axis=1)  # (n_trials, n_ch * n_bands_kept)
    return pd.DataFrame(X), colnames


def _find_first(glob_pattern: str) -> Optional[Path]:
    # Support absolute Windows paths by using glob.glob
    cands = sorted(glob.glob(glob_pattern))
    return Path(cands[0]) if cands else None


def _find_connectivity_arrays(subj_dir: Path, subject: str, task: str, band: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (aec_path, wpli_path) for per-trial connectivity arrays if present.

    Files are typically saved as:
    sub-XXX_task-YYY_*connectivity_aec_<band>*_all_trials.npy
    sub-XXX_task-YYY_*connectivity_wpli_<band>*_all_trials.npy
    """
    aec = None
    wpli = None
    patterns = [
        f"sub-{subject}/eeg/sub-{subject}_task-{task}_*connectivity_aec_{band}*_all_trials.npy",
        f"sub-{subject}/eeg/sub-{subject}_task-{task}_connectivity_aec_{band}*_all_trials.npy",
    ]
    for pat in patterns:
        p = _find_first(str((DERIV_ROOT / pat).as_posix()))
        if p is not None:
            aec = p
            break

    patterns = [
        f"sub-{subject}/eeg/sub-{subject}_task-{task}_*connectivity_wpli_{band}*_all_trials.npy",
        f"sub-{subject}/eeg/sub-{subject}_task-{task}_connectivity_wpli_{band}*_all_trials.npy",
    ]
    for pat in patterns:
        p = _find_first(str((DERIV_ROOT / pat).as_posix()))
        if p is not None:
            wpli = p
            break
    return aec, wpli


def _load_labels(subj_dir: Path, subject: str, task: str) -> Optional[np.ndarray]:
    patterns = [
        f"sub-{subject}/eeg/sub-{subject}_task-{task}_*connectivity_labels*.npy",
        f"sub-{subject}/eeg/sub-{subject}_task-{task}_connectivity_labels*.npy",
    ]
    for pat in patterns:
        p = _find_first(str((DERIV_ROOT / pat).as_posix()))
        if p is not None:
            try:
                return np.load(p, allow_pickle=True)
            except Exception:
                pass
    return None


def _save_fig(fig, save_path: Path, formats=None, dpi=300):
    """Save figure in multiple formats with consistent settings."""
    if formats is None:
        formats = config.get("output.save_formats", ["png"])  # minimal control
    
    _ensure_dir(save_path.parent)
    
    pad_inches = float(config.get("output.pad_inches", 0.02))
    for fmt in formats:
        path_with_ext = save_path.with_suffix(f'.{fmt}')
        fig.savefig(path_with_ext, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)

def _get_band_color(band: str) -> str:
    """Return color for a band with safe defaults, matching script 04.

    Prefers BAND_COLORS from legacy constants; falls back to a sensible palette.
    """
    try:
        if isinstance(BAND_COLORS, dict) and band in BAND_COLORS:
            return str(BAND_COLORS[band])
    except Exception:
        pass
    fallback = {"delta": "#4169e1", "theta": "purple", "alpha": "green", "beta": "orange", "gamma": "red"}
    return fallback.get((band or "").strip().lower(), "#1f77b4")



def plot_power_distributions(pow_df: pd.DataFrame, bands: List[str], subject: str, save_dir: Path, logger: logging.Logger):
    """Violin plots showing power distributions per band across trials."""
    try:
        n_bands = len(bands)
        n_cols = 2
        n_rows = (n_bands + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_bands == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, band in enumerate(bands):
            band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
            if not band_cols:
                logger.warning(f"No columns found for band '{band}'")
                continue
                
            # Stack all channels for this band
            band_data = pow_df[band_cols].values.flatten()
            band_data = band_data[~np.isnan(band_data)]  # Remove NaNs
            
            if len(band_data) == 0:
                logger.warning(f"No valid data for band '{band}'")
                continue
            
            # Create violin plot
            parts = axes[i].violinplot([band_data], positions=[1], 
                                     showmeans=True, showmedians=True)
            
            # Styling
            band_color = _get_band_color(band)
            for pc in parts['bodies']:
                pc.set_facecolor(band_color)
                pc.set_alpha(0.7)
            
            axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Baseline')
            axes[i].set_title(f'{band.capitalize()} Power Distribution\n(All channels, all trials)')
            axes[i].set_ylabel('log10(power/baseline)')
            axes[i].set_xticks([])
            axes[i].grid(True, alpha=0.3)
            
            # Add summary statistics
            mean_val = np.mean(band_data)
            std_val = np.std(band_data)
            median_val = np.median(band_data)
            
            stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nMdn={median_val:.3f}\nn={len(band_data)}'
            axes[i].text(0.7, 0.95, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for j in range(len(bands), len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        _save_fig(fig, save_dir / f'sub-{subject}_power_distributions_per_band')
        plt.close(fig)
        logger.info(f"Saved power distributions: {save_dir / f'sub-{subject}_power_distributions_per_band.png'}")
        
    except Exception as e:
        logger.error(f"Failed to create power distributions: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_channel_power_heatmap(pow_df: pd.DataFrame, bands: List[str], subject: str, save_dir: Path, logger: logging.Logger):
    """Heatmap showing mean power values across channels and bands."""
    try:
        # Reshape data for heatmap
        band_means = []
        channel_names = []
        valid_bands = []
        
        for band in bands:
            band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
            if band_cols:
                band_data = pow_df[band_cols].mean(axis=0)  # Mean across trials
                band_means.append(band_data.values)
                valid_bands.append(band)
                if not channel_names:  # Get channel names from first band
                    channel_names = [col.replace(f'pow_{band}_', '') for col in band_cols]
        
        if not band_means:
            logger.warning("No valid band data for heatmap")
            return
            
        heatmap_data = np.array(band_means)
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(max(12, len(channel_names)*0.4), max(6, len(valid_bands)*0.8)))
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=45, ha='right')
        ax.set_yticks(range(len(valid_bands)))
        ax.set_yticklabels([b.capitalize() for b in valid_bands])
        ax.set_title(f'Mean Power per Channel and Band\nlog10(power/baseline)')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Frequency Band')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='log10(power/baseline)', shrink=0.8)
        
        # Add text annotations for values (only if not too many)
        if len(channel_names) * len(valid_bands) <= 200:  # Avoid overcrowding
            for i in range(len(valid_bands)):
                for j in range(len(channel_names)):
                    text = f'{heatmap_data[i,j]:.2f}'
                    color = 'white' if abs(heatmap_data[i,j]) > np.std(heatmap_data) else 'black'
                    ax.text(j, i, text, ha='center', va='center', 
                           color=color, fontsize=max(6, min(10, 200/len(channel_names))))
        
        plt.tight_layout()
        _save_fig(fig, save_dir / f'sub-{subject}_channel_power_heatmap')
        plt.close(fig)
        logger.info(f"Saved channel power heatmap: {save_dir / f'sub-{subject}_channel_power_heatmap.png'}")
        
    except Exception as e:
        logger.error(f"Failed to create channel power heatmap: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_tfr_spectrograms_roi(tfr, subject: str, save_dir: Path, logger: logging.Logger):
    """Plot TFR spectrograms for key ROI channels."""
    try:
        # Define ROI channels (central, frontal, parietal)
        roi_channels = ['Cz', 'C3', 'C4', 'Pz', 'Fz', 'Oz']
        ch_names = list(getattr(tfr, "ch_names", []) or tfr.info.get("ch_names", []))
        available_channels = [ch for ch in roi_channels if ch in ch_names]
        
        if not available_channels:
            logger.warning("No ROI channels found in data")
            return
            
        n_channels = len(available_channels)
        fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3*n_channels))
        if n_channels == 1:
            axes = [axes]
        
        for i, ch in enumerate(available_channels):
            # Pick single channel and average across trials
            tfr_ch = tfr.copy().pick_channels([ch]).average()

            # Robust color limits centered at 0, with optional override
            vabs = _robust_sym_vlim(tfr_ch.data)
            if TFR_SPECTROGRAM_VLIM is not None:
                vabs = float(TFR_SPECTROGRAM_VLIM)

            # Create spectrogram plot
            tfr_ch.plot(
                picks=[0],
                axes=axes[i],
                show=False,
                colorbar=True,
                title=f'{ch} - log10(power/baseline)',
                vlim=(-vabs, +vabs),
                cmap='RdBu_r',
            )  # Symmetric around 0 for logratio
            
            # Add vertical lines for key time points
            axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Stimulus onset')
            axes[i].axvline(x=PLATEAU_START, color='green', linestyle='--', alpha=0.7, label='Plateau start')
            axes[i].axvline(x=PLATEAU_END, color='red', linestyle='--', alpha=0.7, label='Plateau end')
            if i == 0:  # Only add legend to first subplot
                axes[i].legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        _save_fig(fig, save_dir / f'sub-{subject}_tfr_spectrograms_roi')
        plt.close(fig)
        logger.info(f"Saved TFR spectrograms: {save_dir / f'sub-{subject}_tfr_spectrograms_roi.png'}")
        
    except Exception as e:
        logger.error(f"Failed to create TFR spectrograms: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_power_time_courses(tfr_raw, bands: List[str], subject: str, save_dir: Path, logger: logging.Logger):
    """Plot power time courses showing how power evolves within trials for each frequency band.
    Each band gets its own separate figure."""
    try:
        times = tfr_raw.times
        
        for band in bands:
            # Get frequency range for this band
            if band not in FEATURES_FREQ_BANDS:
                logger.warning(f"Band '{band}' not in config; skipping time course.")
                continue
            fmin, fmax = FEATURES_FREQ_BANDS[band]
            
            # Find frequency indices using the actual TFR freqs array
            freq_mask = (tfr_raw.freqs >= fmin) & (tfr_raw.freqs <= fmax)
            if not freq_mask.any():
                logger.warning(f"No frequencies found for {band} band ({fmin}-{fmax} Hz)")
                continue
            
            # Create separate figure for this band
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            
            # Average across frequencies and channels - fix indexing order
            # TFR data structure: (trials, channels, freqs, times)
            # For baseline-corrected TFRs (mode='logratio'), values are already log10(power/baseline).
            # We therefore directly average and plot the logratio.
            band_power_log = tfr_raw.data[:, :, freq_mask, :].mean(axis=(0, 1, 2))  # Average across trials, channels, freqs
            
            # Plot time course
            ax.plot(times, band_power_log, linewidth=2, color=_get_band_color(band))
            
            # Add baseline period marker using configured baseline (clipped to available time range)
            try:
                b_start, b_end, _ = _validate_baseline_indices(times, TFR_BASELINE, MIN_BASELINE_SAMPLES)
                bs = max(float(times.min()), float(b_start))
                be = min(float(times.max()), float(b_end))
                if be > bs:
                    ax.axvspan(bs, be, alpha=0.2, color='gray', label='Baseline')
            except Exception:
                # Baseline not available in time range; skip shading
                pass
            
            # Add stimulus period marker  
            ax.axvspan(0, times[-1], alpha=0.2, color='orange', label='Stimulus')
            
            ax.set_ylabel(f'log10(power/baseline)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'{band.capitalize()} Band Power Time Course')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            _save_fig(fig, save_dir / f'sub-{subject}_power_time_course_{band}')
            plt.close(fig)
            logger.info(f"Saved {band} power time course: {save_dir / f'sub-{subject}_power_time_course_{band}.png'}")
        
    except Exception as e:
        logger.error(f"Failed to create power time courses: {e}")
        if 'fig' in locals():
            plt.close(fig)




def plot_trial_power_variability(pow_df: pd.DataFrame, bands: List[str], subject: str, save_dir: Path, logger: logging.Logger):
    """Plot trial-by-trial power variability showing consistency across trials."""
    try:
        n_bands = len(bands)
        fig, axes = plt.subplots(n_bands, 1, figsize=(12, 3*n_bands))
        if n_bands == 1:
            axes = [axes]
        
        for i, band in enumerate(bands):
            # Get columns for this band
            band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
            if not band_cols:
                continue
            
            # Calculate mean power across channels for each trial
            band_power_trials = pow_df[band_cols].mean(axis=1)
            
            # Plot trial-by-trial variability
            trial_nums = range(1, len(band_power_trials) + 1)
            axes[i].plot(trial_nums, band_power_trials, 'o-', alpha=0.7, linewidth=1,
                        color=_get_band_color(band))
            
            # Add mean line
            mean_power = band_power_trials.mean()
            axes[i].axhline(mean_power, color='red', linestyle='--', alpha=0.8, label=f'Mean = {mean_power:.3f}')
            
            # Add variability measures
            std_power = band_power_trials.std()
            cv_power = std_power / abs(mean_power) if abs(mean_power) > 1e-10 else np.nan
            
            axes[i].fill_between(trial_nums, mean_power - std_power, mean_power + std_power, 
                               alpha=0.2, color='red', label=f'±1 SD = ±{std_power:.3f}')
            
            axes[i].set_ylabel(f'{band.capitalize()}\nlog10(power/baseline)')
            axes[i].set_title(f'{band.capitalize()} Band Power Variability (CV = {cv_power:.3f})')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        axes[-1].set_xlabel('Trial Number')
        plt.tight_layout()
        _save_fig(fig, save_dir / f'sub-{subject}_trial_power_variability')
        plt.close(fig)
        logger.info(f"Saved trial power variability: {save_dir / f'sub-{subject}_trial_power_variability.png'}")
        
    except Exception as e:
        logger.error(f"Failed to create trial power variability: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_inter_band_spatial_power_correlation(tfr, subject: str, save_dir: Path, logger: logging.Logger):
    """Plot inter_band_spatial_power_correlation showing relationships between frequency bands."""
    try:
        band_names = list(FEATURES_FREQ_BANDS.keys())
        n_bands = len(band_names)
        
        # Create correlation matrix for Inter Band Spatial Power Correlation
        coupling_matrix = np.zeros((n_bands, n_bands))
        
        # Average across time window for inter_band_spatial_power_correlation (clip to available time range)
        times = np.asarray(tfr.times)
        tmin_clip = float(max(times.min(), PLATEAU_START))
        tmax_clip = float(min(times.max(), PLATEAU_END))
        if not np.isfinite(tmin_clip) or not np.isfinite(tmax_clip) or (tmax_clip <= tmin_clip):
            logger.warning(
                f"Skipping inter-band spatial power correlation: invalid plateau within data range "
                f"(requested [{PLATEAU_START}, {PLATEAU_END}] s, available [{times.min():.2f}, {times.max():.2f}] s)"
            )
            return
        tfr_windowed = tfr.copy().crop(tmin_clip, tmax_clip)
        tfr_avg = tfr_windowed.average()  # Average across trials
        
        for i, band1 in enumerate(band_names):
            fmin1, fmax1 = FEATURES_FREQ_BANDS[band1]
            freq_mask1 = (tfr_avg.freqs >= fmin1) & (tfr_avg.freqs <= fmax1)
            
            if not freq_mask1.any():
                continue
                
            # Extract band1 power (average across channels and time)
            band1_power = tfr_avg.data[:, freq_mask1, :].mean(axis=(0, 1, 2))
            
            for j, band2 in enumerate(band_names):
                if i == j:
                    coupling_matrix[i, j] = 1.0  # Perfect self-correlation
                    continue
                    
                fmin2, fmax2 = FEATURES_FREQ_BANDS[band2]
                freq_mask2 = (tfr_avg.freqs >= fmin2) & (tfr_avg.freqs <= fmax2)
                
                if not freq_mask2.any():
                    continue
                
                # Extract band2 power (average across channels and time)
                band2_power = tfr_avg.data[:, freq_mask2, :].mean(axis=(0, 1, 2))
                
                # Compute cross-channel correlation between bands
                band1_channels = tfr_avg.data[:, freq_mask1, :].mean(axis=(1, 2))  # Average across freqs and time
                band2_channels = tfr_avg.data[:, freq_mask2, :].mean(axis=(1, 2))  # Average across freqs and time
                
                # Calculate correlation across channels
                if len(band1_channels) > 1 and len(band2_channels) > 1:
                    correlation = np.corrcoef(band1_channels, band2_channels)[0, 1]
                    coupling_matrix[i, j] = correlation
        
        # Plot coupling matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(coupling_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Add labels
        ax.set_xticks(range(n_bands))
        ax.set_yticks(range(n_bands))
        ax.set_xticklabels([band.capitalize() for band in band_names], rotation=45, ha='right')
        ax.set_yticklabels([band.capitalize() for band in band_names])
        
        # Add correlation values as text
        for i in range(n_bands):
            for j in range(n_bands):
                text = ax.text(j, i, f'{coupling_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(coupling_matrix[i, j]) < 0.5 else "white")
        
        ax.set_title('Inter Band Spatial Power Correlation')
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Frequency Band')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation (r)')
        
        plt.tight_layout()
        _save_fig(fig, save_dir / f'sub-{subject}_inter_band_spatial_power_correlation')
        plt.close(fig)
        logger.info(f"Saved Inter Band Spatial Power Correlation: {save_dir / f'sub-{subject}_Inter Band Spatial Power Correlation.png'}")
        
    except Exception as e:
        logger.error(f"Failed to create Inter Band Spatial Power Correlation: {e}")
        if 'fig' in locals():
            plt.close(fig)




def _flatten_lower_triangles(conn_trials: np.ndarray, labels: Optional[np.ndarray], prefix: str) -> Tuple[pd.DataFrame, List[str]]:
    """Flatten lower triangle (i>j) of connectivity matrices per trial.

    conn_trials: (n_trials, n_nodes, n_nodes)
    Returns DataFrame (n_trials, n_pairs) and column names.
    """
    if conn_trials.ndim != 3:
        raise ValueError("Connectivity array must be 3D (trials, nodes, nodes)")
    n_trials, n_nodes, _ = conn_trials.shape
    idx_i, idx_j = np.tril_indices(n_nodes, k=-1)
    out = conn_trials[:, idx_i, idx_j]

    if labels is not None and len(labels) == n_nodes:
        pair_names = [f"{labels[i]}__{labels[j]}" for i, j in zip(idx_i, idx_j)]
    else:
        pair_names = [f"n{i}_n{j}" for i, j in zip(idx_i, idx_j)]
    cols = [f"{prefix}_{p}" for p in pair_names]
    return pd.DataFrame(out), cols


def _extract_connectivity_features(subject: str, task: str, bands: List[str], logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """Load per-trial connectivity arrays if available and flatten into features.
    Concatenates across bands and both measures (AEC, wPLI).
    """
    subj_dir = DERIV_ROOT / f"sub-{subject}" / "eeg"
    if not subj_dir.exists():
        return pd.DataFrame(), []

    labels = _load_labels(subj_dir, subject, task)
    all_blocks: List[pd.DataFrame] = []
    all_cols: List[str] = []
    n_trials_ref: Optional[int] = None

    for band in bands:
        aec_path, wpli_path = _find_connectivity_arrays(subj_dir, subject, task, band)
        for measure, pth in (("aec", aec_path), ("wpli", wpli_path)):
            if pth is None or not Path(pth).exists():
                logger.warning(f"Connectivity file missing for {measure} {band}: {pth}")
                continue
            arr = np.load(pth)
            if arr.ndim != 3:
                logger.warning(f"Unexpected connectivity shape at {pth}: {arr.shape}")
                continue
            df_flat, cols = _flatten_lower_triangles(arr, labels, prefix=f"{measure}_{band}")
            # Align n_trials across measures
            if n_trials_ref is None:
                n_trials_ref = len(df_flat)
            else:
                min_n = min(n_trials_ref, len(df_flat))
                df_flat = df_flat.iloc[:min_n, :]
                n_trials_ref = min_n
                for i in range(len(all_blocks)):
                    all_blocks[i] = all_blocks[i].iloc[:min_n, :]
            all_blocks.append(df_flat)
            all_cols.extend(cols)

    if not all_blocks:
        return pd.DataFrame(), []

    X = pd.concat(all_blocks, axis=1)
    X.columns = all_cols
    return X, all_cols


# -----------------------------------------------------------------------------
# Main driver per subject
# -----------------------------------------------------------------------------

def process_subject(subject: str, task: str = TASK) -> None:
    logger = _setup_logging(subject)
    logger.info(f"=== Feature engineering: sub-{subject}, task-{task} ===")
    features_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "features"
    _ensure_dir(features_dir)

    # Load epochs for alignment and channel names
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None:
        logger.error(f"No cleaned epochs found for sub-{subject}; skipping.")
        return
    logger.info(f"Epochs: {epo_path}")
    epochs = mne.read_epochs(epo_path, preload=False, verbose=False)

    # Load events and align
    events_df = _load_events_df(subject, task)
    try:
        aligned_events = align_events_to_epochs_strict(events_df, epochs, logger)
    except ValueError as e:
        logger.error(f"Event alignment failed strictly: {e}")
        return
    if aligned_events is not None:
        validate_alignment(aligned_events, epochs, logger)
    if aligned_events is None:
        logger.warning("No events available for targets; skipping subject.")
        return

    # Record any dropped trials (events not retained after preprocessing)
    drop_log_path = features_dir / "dropped_trials.tsv"
    try:
        selection = getattr(epochs, "selection", None)
        if selection is None:
            raise AttributeError("epochs.selection missing")
        selection_arr = np.asarray(selection, dtype=int)
        valid_mask = (selection_arr >= 0) & (selection_arr < len(events_df))
        if not np.all(valid_mask):
            logger.warning(
                "Epoch selection contains indices outside events range; restricting to valid entries."
            )
            selection_arr = selection_arr[valid_mask]
        kept_indices = set(int(idx) for idx in selection_arr.tolist())
        dropped_indices = [idx for idx in range(len(events_df)) if idx not in kept_indices]

        if dropped_indices:
            dropped_events = events_df.iloc[dropped_indices].copy()
            drop_reasons: list[str] = []
            drop_log = getattr(epochs, "drop_log", None)

            def _format_drop_reason(entry) -> str:
                if entry is None:
                    return ""
                if isinstance(entry, (list, tuple)):
                    return ";".join(str(x) for x in entry if x)
                return str(entry)

            if isinstance(drop_log, (list, tuple)) and len(drop_log) == len(events_df):
                drop_reasons = [_format_drop_reason(drop_log[idx]) for idx in dropped_indices]
            else:
                drop_reasons = [""] * len(dropped_indices)

            dropped_events.insert(0, "original_index", dropped_indices)
            dropped_events["drop_reason"] = drop_reasons
            _ensure_dir(drop_log_path.parent)
            dropped_events.to_csv(drop_log_path, sep="\t", index=False)
            logger.info(
                "Saved drop log with %d dropped trials to %s",
                len(dropped_events),
                drop_log_path,
            )
        else:
            # Write an empty file so downstream steps know no drops occurred
            empty_df = pd.DataFrame(columns=["original_index", "drop_reason"])
            _ensure_dir(drop_log_path.parent)
            empty_df.to_csv(drop_log_path, sep="\t", index=False)
            logger.info("No dropped trials detected; wrote empty drop log to %s", drop_log_path)
    except AttributeError as exc:
        logger.warning("Unable to derive drop log for sub-%s: %s", subject, exc)


    # Pick target column
    target_col = _pick_target_column(aligned_events)
    if target_col is None:
        logger.warning("No suitable target column found in events; skipping.")
        return
    y = pd.to_numeric(aligned_events[target_col], errors="coerce")

    # Compute TFR for power features (trial-level)
    tfr = _compute_tfr(epochs)
    # Attach aligned metadata to TFR for downstream alignment (if lengths match)
    try:
        if aligned_events is not None and len(aligned_events) >= len(tfr):
            tfr.metadata = aligned_events.iloc[: len(tfr)].reset_index(drop=True)
    except Exception:
        pass
    # Keep a copy of raw TFR before baseline correction for comparison plots
    tfr_raw = tfr.copy()
    
    # Normalize to pre-stimulus baseline as log10(power/baseline) for comparability
    try:
        times = np.asarray(tfr.times)
        b_start, b_end, _ = _validate_baseline_indices(times, TFR_BASELINE, MIN_BASELINE_SAMPLES)
        tfr.apply_baseline(baseline=(b_start, b_end), mode="logratio")
        # Mark baseline status in the object comment for downstream checks
        try:
            tfr.comment = f"BASELINED:mode=logratio;win=({b_start:.3f},{b_end:.3f})"
        except Exception:
            pass
        # Optionally save TFR with sidecar for downstream reuse
        if SAVE_TFR_WITH_SIDECAR:
            tfr_out = DERIV_ROOT / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_power_epo-tfr.h5"
            _save_tfr_with_sidecar(tfr, tfr_out, (b_start, b_end), mode="logratio", logger=logger)
    except ValueError as e:
        logger.error(f"Baseline normalization skipped: {e}")
    except Exception as e:
        logger.warning(
            f"Baseline normalization failed; proceeding without baseline. Error: {e}"
        )

    pow_df, pow_cols = _extract_band_power_features(tfr, POWER_BANDS, logger)
    # Connectivity features (if available)
    conn_df, conn_cols = _extract_connectivity_features(subject, task, POWER_BANDS, logger)

    # Align lengths across direct EEG power, connectivity, and targets
    parts = [x for x in [pow_df, conn_df, y] if x is not None and len(x) > 0]
    if not parts:
        logger.warning("No features extracted; skipping save.")
        return
    n = min(len(p) for p in parts)

    # Trim to shared trial count
    if len(y) != n:
        y = y.iloc[:n]
    if pow_df is not None and len(pow_df) > 0 and len(pow_df) != n:
        pow_df = pow_df.iloc[:n, :]
    if conn_df is not None and len(conn_df) > 0 and len(conn_df) != n:
        conn_df = conn_df.iloc[:n, :]

    # Save direct EEG features and columns
    eeg_direct_path_tsv = features_dir / "features_eeg_direct.tsv"
    eeg_direct_path_csv = features_dir / "features_eeg_direct.csv"
    eeg_direct_cols_path = features_dir / "features_eeg_direct_columns.tsv"
    logger.info(f"Saving direct EEG features: {eeg_direct_path_tsv} and {eeg_direct_path_csv}")
    # ensure descriptive headers
    if pow_cols:
        pow_df.columns = pow_cols
    
    # Save main features file in both TSV and CSV formats
    pow_df.to_csv(eeg_direct_path_tsv, sep="\t", index=False)
    pow_df.to_csv(eeg_direct_path_csv, sep=",", index=False)
    pd.Series(pow_cols, name="feature").to_csv(eeg_direct_cols_path, sep="\t", index=False)
    
    # Save band-specific files (TSV and CSV for each frequency band)
    logger.info("Saving band-specific power files")
    for band in POWER_BANDS:
        # Extract columns for this band
        band_cols = [col for col in pow_cols if col.startswith(f"pow_{band}_")]
        if not band_cols:
            logger.warning(f"No columns found for band '{band}'")
            continue
            
        # Extract data for this band
        band_df = pow_df[band_cols].copy()
        
        # Remove band prefix from column names for cleaner output
        clean_cols = [col.replace(f"pow_{band}_", "") for col in band_cols]
        band_df.columns = clean_cols
        
        # Save in both TSV and CSV formats
        band_tsv_path = features_dir / f"features_eeg_{band}_power.tsv"
        band_csv_path = features_dir / f"features_eeg_{band}_power.csv"
        
        band_df.to_csv(band_tsv_path, sep="\t", index=False)
        band_df.to_csv(band_csv_path, sep=",", index=False)
        
        logger.info(f"Saved {band} band power: {band_tsv_path} and {band_csv_path} ({len(clean_cols)} channels)")

    # Save connectivity features if available
    if conn_df is not None and len(conn_df) > 0:
        conn_path = features_dir / "features_connectivity.tsv"
        logger.info(f"Saving connectivity features: {conn_path}")
        # apply column names if available
        if conn_cols:
            conn_df.columns = conn_cols
        conn_df.to_csv(conn_path, sep="\t", index=False)

    # Save combined matrix (power + connectivity if available)
    blocks = [pow_df]
    cols_all: List[str] = list(pow_cols)
    if conn_df is not None and len(conn_df) > 0:
        blocks.append(conn_df)
        cols_all.extend(conn_cols)
    X_all = pd.concat(blocks, axis=1)
    X_all.columns = cols_all
    combined_path = features_dir / "features_all.tsv"
    logger.info(f"Saving combined features: {combined_path}")
    X_all.to_csv(combined_path, sep="\t", index=False)

    # Save targets
    y_path_tsv = features_dir / "target_vas_ratings.tsv"
    logger.info(f"Saving behavioral target vector: {y_path_tsv} (column: {target_col})")
    y.to_frame(name=target_col).to_csv(y_path_tsv, sep="\t", index=False)

    # Generate visualizations
    plots_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots" / "03_feature_engineering"
    _ensure_dir(plots_dir)
    logger.info(f"Generating power visualizations in: {plots_dir}")
    
    try:
        # 2. Power distributions per band
        plot_power_distributions(pow_df, POWER_BANDS, subject, plots_dir, logger)
        
        # 3. Channel power heatmap
        plot_channel_power_heatmap(pow_df, POWER_BANDS, subject, plots_dir, logger)
        
        # 4. TFR spectrograms for ROI channels
        plot_tfr_spectrograms_roi(tfr, subject, plots_dir, logger)
        
        # 6. Power-behavior correlations (moved to 04_behavior_feature_analysis.py)
        # This visualization is now handled in the behavior analysis script
        
        # Additional visualization suggestions (uncomment and customize as needed):
        
        # 7. Power time courses - shows how power evolves within trials (baseline-corrected)
        plot_power_time_courses(tfr, POWER_BANDS, subject, plots_dir, logger)
        
        
        # 9. Trial-by-trial power variability - shows consistency across trials  
        plot_trial_power_variability(pow_df, POWER_BANDS, subject, plots_dir, logger)
        
        # 10. Inter Band Spatial Power Correlation - relationships between bands
        plot_inter_band_spatial_power_correlation(tfr, subject, plots_dir, logger)
        
        
        logger.info("Successfully generated all power visualizations")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

    logger.info(
        f"Done: sub-{subject}, n_trials={n}, n_direct_features={pow_df.shape[1]}, "
        f"n_conn_features={(conn_df.shape[1] if conn_df is not None and len(conn_df) > 0 else 0)}, "
        f"n_all_features={X_all.shape[1]} (power = log10(power/baseline))"
    )


def _get_available_subjects() -> List[str]:
    """Find all subjects in DERIV_ROOT that have cleaned epochs available."""
    subs: List[str] = []
    root = Path(DERIV_ROOT)
    if not root.exists():
        return subs
    for p in sorted(root.glob("sub-*/eeg")):
        sub_id = p.parent.name.replace("sub-", "")
        try:
            if _find_clean_epochs_path(sub_id, TASK) is not None:
                subs.append(sub_id)
        except Exception:
            continue
    return subs


def _group_plots_dir() -> Path:
    return DERIV_ROOT / "group" / "eeg" / "plots" / "03_feature_engineering"


def _group_stats_dir() -> Path:
    return DERIV_ROOT / "group" / "eeg" / "stats"


def _setup_group_logging() -> logging.Logger:
    logger = logging.getLogger("feature_engineering_group")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # File
    log_dir = DERIV_ROOT / "group" / "eeg" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def _fisher_z(r: float) -> float:
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def _fisher_z_to_r(z: float) -> float:
    return float(np.tanh(z))


def aggregate_group_level(subjects: List[str], task: str = TASK) -> None:
    """Aggregate feature-engineering outputs across subjects and generate group plots.

    - Group channel power heatmap (bands x channels) from per-subject means
    - Across-subject band power distributions
    - Group inter-band spatial power correlation (Fisher-averaged)
    """
    logger = _setup_group_logging()
    gplots = _group_plots_dir()
    gstats = _group_stats_dir()
    _ensure_dir(gplots)
    _ensure_dir(gstats)

    # Collect per-subject power DataFrames
    subj_pow: dict[str, pd.DataFrame] = {}
    for s in subjects:
        p = DERIV_ROOT / f"sub-{s}" / "eeg" / "features" / "features_eeg_direct.tsv"
        try:
            if p.exists():
                df = pd.read_csv(p, sep="\t")
                subj_pow[s] = df
            else:
                logger.warning(f"Missing features for sub-{s}: {p}")
        except Exception as e:
            logger.warning(f"Failed reading features for sub-{s}: {e}")

    if len(subj_pow) < 2:
        logger.warning("Group aggregation requires at least 2 subjects with features; skipping group plots.")
        return

    bands = list(POWER_BANDS)

    # 1) Group channel power heatmap (mean across subjects of per-subject channel means)
    try:
        # Determine union of channels per band across all subjects
        band_channels: dict[str, List[str]] = {}
        for b in bands:
            ch_union: set[str] = set()
            for s, df in subj_pow.items():
                cols = [c for c in df.columns if c.startswith(f"pow_{b}_")]
                ch_union.update([c.replace(f"pow_{b}_", "") for c in cols])
            band_channels[b] = sorted(ch_union)

        # Compute per-band channel means across subjects
        heat_rows: List[np.ndarray] = []
        stats_rows: List[dict] = []
        # For column order, use channels present in most bands; fallback to alphabetical per-band
        # We will align per-band independently for the heatmap (x-axis labels vary if needed)

        # To keep a consistent x-axis, find channels common to all bands, else use union across bands
        all_ch_union: List[str] = sorted(set().union(*band_channels.values())) if band_channels else []

        # Build heatmap on the union; allow NaNs where data missing
        for b in bands:
            ch_list = list(all_ch_union)
            subj_means_per_ch: List[List[float]] = []  # subjects x channels
            for s, df in subj_pow.items():
                # Per-subject per-channel mean across trials for this band
                vals = []
                for ch in ch_list:
                    col = f"pow_{b}_{ch}"
                    if col in df.columns:
                        vals.append(float(pd.to_numeric(df[col], errors="coerce").mean()))
                    else:
                        vals.append(np.nan)
                subj_means_per_ch.append(vals)
            arr = np.asarray(subj_means_per_ch, dtype=float)
            mean_across_subj = np.nanmean(arr, axis=0)
            heat_rows.append(mean_across_subj)
            # Stats rows
            n_eff = np.sum(np.isfinite(arr), axis=0)
            std_across_subj = np.nanstd(arr, axis=0, ddof=1)
            for j, ch in enumerate(ch_list):
                stats_rows.append({
                    "band": b,
                    "channel": ch,
                    "mean": float(mean_across_subj[j]) if np.isfinite(mean_across_subj[j]) else np.nan,
                    "std": float(std_across_subj[j]) if np.isfinite(std_across_subj[j]) else np.nan,
                    "n_subjects": int(n_eff[j]),
                })

        heat = np.vstack(heat_rows) if heat_rows else np.zeros((0, 0))
        if heat.size > 0:
            fig, ax = plt.subplots(figsize=(max(12, len(all_ch_union) * 0.4), max(6, len(bands) * 0.8)))
            im = ax.imshow(heat, cmap='RdBu_r', aspect='auto')
            ax.set_xticks(range(len(all_ch_union)))
            ax.set_xticklabels(all_ch_union, rotation=45, ha='right')
            ax.set_yticks(range(len(bands)))
            ax.set_yticklabels([b.capitalize() for b in bands])
            ax.set_title("Group Mean Power per Channel and Band\nlog10(power/baseline)")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Frequency Band")
            plt.colorbar(im, ax=ax, label='log10(power/baseline)', shrink=0.8)
            plt.tight_layout()
            _save_fig(fig, gplots / "group_channel_power_heatmap")
        # Save stats table
        pd.DataFrame(stats_rows).to_csv(gstats / "group_channel_power_means.tsv", sep="\t", index=False)
        logger.info("Saved group channel power heatmap and stats.")
    except Exception as e:
        logger.warning(f"Group channel heatmap failed: {e}")

    # 2) Across-subject band power distributions (mean per subject per band)
    try:
        recs: List[dict] = []
        for b in bands:
            for s, df in subj_pow.items():
                cols = [c for c in df.columns if c.startswith(f"pow_{b}_")]
                if not cols:
                    continue
                vals = pd.to_numeric(df[cols].stack(), errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                recs.append({
                    "subject": s,
                    "band": b,
                    "mean_power": float(np.mean(vals)),
                })
        dfm = pd.DataFrame(recs)
        if not dfm.empty:
            # Plot bar (mean across subjects) with scatter of subject means and 95% CI
            bands_present = [b for b in bands if b in set(dfm["band"])]
            means = []
            ci_l = []
            ci_h = []
            ns = []
            for b in bands_present:
                v = dfm[dfm["band"] == b]["mean_power"].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                mu = float(np.mean(v)) if v.size else np.nan
                se = float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else np.nan
                delta = 1.96 * se if np.isfinite(se) else np.nan
                means.append(mu)
                ci_l.append(mu - delta if np.isfinite(delta) else np.nan)
                ci_h.append(mu + delta if np.isfinite(delta) else np.nan)
                ns.append(len(v))
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(bands_present))
            ax.bar(x, means, color='steelblue', alpha=0.8)
            # Error bars
            yerr = np.array([[mu - lo if np.isfinite(mu) and np.isfinite(lo) else 0 for lo, mu in zip(ci_l, means)],
                             [hi - mu if np.isfinite(mu) and np.isfinite(hi) else 0 for hi, mu in zip(ci_h, means)]])
            ax.errorbar(x, means, yerr=yerr, fmt='none', ecolor='k', capsize=3)
            # Scatter points per subject with jitter
            for i, b in enumerate(bands_present):
                vals = dfm[dfm["band"] == b]["mean_power"].to_numpy(dtype=float)
                jitter = (np.random.rand(len(vals)) - 0.5) * 0.2
                ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals, color='k', s=12, alpha=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([bp.capitalize() for bp in bands_present])
            ax.set_ylabel('Mean log10(power/baseline) across subjects')
            ax.set_title('Group Band Power Summary (subject means, 95% CI)')
            ax.axhline(0, color='k', linewidth=0.8)
            plt.tight_layout()
            _save_fig(fig, gplots / "group_power_distributions_per_band_across_subjects")
            # Save stats
            out = pd.DataFrame({
                "band": bands_present,
                "group_mean": means,
                "ci_low": ci_l,
                "ci_high": ci_h,
                "n_subjects": ns,
            })
            out.to_csv(gstats / "group_band_power_subject_means.tsv", sep="\t", index=False)
            logger.info("Saved group band power distributions and stats.")
    except Exception as e:
        logger.warning(f"Group band power distributions failed: {e}")

    # 3) Group inter-band spatial power correlation (per-subject channel means -> Fisher-avg)
    try:
        band_names = list(FEATURES_FREQ_BANDS.keys())
        m = len(band_names)
        # Collect per-subject correlation matrices
        per_subject_corrs: List[np.ndarray] = []
        for s, df in subj_pow.items():
            # Build per-band channel mean dictionaries (channel -> mean)
            band_vecs: dict[str, dict[str, float]] = {}
            for b in band_names:
                cols = [c for c in df.columns if c.startswith(f"pow_{b}_")]
                if not cols:
                    continue
                ser = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
                # Map channel name -> mean
                ch_means = {c.replace(f"pow_{b}_", ""): float(v) for c, v in ser.items() if np.isfinite(v)}
                if ch_means:
                    band_vecs[b] = ch_means
            if len(band_vecs) < 2:
                continue
            # Create correlation matrix across bands using intersection of channels
            corr_mat = np.eye(m, dtype=float)
            for i, bi in enumerate(band_names):
                for j, bj in enumerate(band_names):
                    if j <= i:
                        continue
                    di = band_vecs.get(bi)
                    dj = band_vecs.get(bj)
                    if di is None or dj is None:
                        corr = np.nan
                    else:
                        common = sorted(set(di.keys()) & set(dj.keys()))
                        if len(common) < 2:
                            corr = np.nan
                        else:
                            vi = np.array([di[ch] for ch in common], dtype=float)
                            vj = np.array([dj[ch] for ch in common], dtype=float)
                            # Guard against zero variance
                            if np.std(vi) < 1e-12 or np.std(vj) < 1e-12:
                                corr = np.nan
                            else:
                                corr = float(np.corrcoef(vi, vj)[0, 1])
                    corr_mat[i, j] = corr
                    corr_mat[j, i] = corr
            per_subject_corrs.append(corr_mat)

        if len(per_subject_corrs) >= 2:
            arr = np.stack(per_subject_corrs, axis=0)  # (n_subj, m, m)
            # Fisher-average off-diagonals
            group_corr = np.eye(m, dtype=float)
            for i in range(m):
                for j in range(m):
                    if i == j:
                        group_corr[i, j] = 1.0
                        continue
                    rvals = arr[:, i, j]
                    rvals = rvals[np.isfinite(rvals)]
                    if rvals.size == 0:
                        group_corr[i, j] = np.nan
                    else:
                        z = np.arctanh(np.clip(rvals, -0.999999, 0.999999))
                        zbar = float(np.mean(z))
                        group_corr[i, j] = float(np.tanh(zbar))

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(group_corr, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(m))
            ax.set_yticks(range(m))
            ax.set_xticklabels([b.capitalize() for b in band_names], rotation=45, ha='right')
            ax.set_yticklabels([b.capitalize() for b in band_names])
            for i in range(m):
                for j in range(m):
                    if np.isfinite(group_corr[i, j]):
                        ax.text(j, i, f"{group_corr[i, j]:.2f}", ha='center', va='center',
                                color=('white' if abs(group_corr[i, j]) > 0.5 else 'black'))
            ax.set_title('Group Inter Band Spatial Power Correlation')
            ax.set_xlabel('Frequency Band')
            ax.set_ylabel('Frequency Band')
            plt.colorbar(im, ax=ax, label='Correlation (r)')
            plt.tight_layout()
            _save_fig(fig, gplots / "group_inter_band_spatial_power_correlation")

            # Save stats (upper triangle)
            rows = []
            for i in range(m):
                for j in range(i + 1, m):
                    rvals = np.array([cm[i, j] for cm in per_subject_corrs], dtype=float)
                    rvals = rvals[np.isfinite(rvals)]
                    if rvals.size == 0:
                        continue
                    z = np.arctanh(np.clip(rvals, -0.999999, 0.999999))
                    zbar = float(np.mean(z))
                    se = float(np.std(z, ddof=1) / math.sqrt(len(z))) if len(z) > 1 else np.nan
                    ci_l = float(np.tanh(zbar - 1.96 * se)) if np.isfinite(se) else np.nan
                    ci_h = float(np.tanh(zbar + 1.96 * se)) if np.isfinite(se) else np.nan
                    rows.append({
                        "band_i": band_names[i],
                        "band_j": band_names[j],
                        "r_group": float(np.tanh(zbar)),
                        "r_ci_low": ci_l,
                        "r_ci_high": ci_h,
                        "n_subjects": int(len(rvals)),
                    })
            if rows:
                pd.DataFrame(rows).to_csv(gstats / "group_inter_band_correlation.tsv", sep="\t", index=False)
                logger.info("Saved group inter-band correlation heatmap and stats.")
    except Exception as e:
        logger.warning(f"Group inter-band correlation failed: {e}")

    # 4) Group band power time courses (prefer saved TFR; else compute on the fly)
    try:
        # Helper to compute AverageTFR for a subject
        def _compute_avg_tfr_for_subject(subj: str) -> Optional["mne.time_frequency.AverageTFR"]:
            try:
                epo_path = _find_clean_epochs_path(subj, task)
                if epo_path is None or not epo_path.exists():
                    return None
                epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
                tfr_ep = _compute_tfr(epochs)
                # Baseline-correct to logratio consistently with per-subject processing
                try:
                    times = np.asarray(tfr_ep.times)
                    b_start, b_end, _ = _validate_baseline_indices(times, TFR_BASELINE, MIN_BASELINE_SAMPLES)
                    tfr_ep.apply_baseline(baseline=(b_start, b_end), mode="logratio")
                except Exception:
                    pass
                return tfr_ep.average()
            except Exception:
                return None

        # Gather per-subject AverageTFR in standardized logratio units
        tfr_list: List["mne.time_frequency.AverageTFR"] = []
        missing_subjects: List[str] = []
        for s in subjects:
            tfr_path = _find_tfr_path(s, task)
            if tfr_path is None or not tfr_path.exists():
                missing_subjects.append(s)
                continue
            t_std = _read_tfr_average_with_logratio(tfr_path, TFR_BASELINE, logger)
            if t_std is None:
                missing_subjects.append(s)
            else:
                tfr_list.append(t_std)

        # If not enough saved TFRs, try computing on the fly for missing subjects
        if len(tfr_list) < 2 and missing_subjects:
            computed = 0
            for s in missing_subjects:
                tavg = _compute_avg_tfr_for_subject(s)
                if tavg is not None:
                    tfr_list.append(tavg)
                    computed += 1
            if computed > 0:
                logger.info(f"Computed AverageTFR on the fly for {computed} subjects (no saved TFR found)")

        if len(tfr_list) >= 2:
            # Reference time grid from first subject; restrict to common time window
            ref = tfr_list[0]
            tmin = max(float(min(t.times[0] for t in tfr_list)), float(ref.times[0]))
            tmax = min(float(max(t.times[-1] for t in tfr_list)), float(ref.times[-1]))
            ref_mask = (ref.times >= tmin) & (ref.times <= tmax)
            tref = ref.times[ref_mask]

            # Compute per-band time courses per subject (avg across channels and freqs)
            # Store both logratio (for original plot) and percent change (derived from ratio)
            band_tc: dict[str, List[np.ndarray]] = {b: [] for b in bands}
            band_tc_pct: dict[str, List[np.ndarray]] = {b: [] for b in bands}
            for t in tfr_list:
                # Build series on this subject's time grid, then interpolate to tref
                for b in bands:
                    if b not in FEATURES_FREQ_BANDS:
                        continue
                    fmin, fmax = FEATURES_FREQ_BANDS[b]
                    fmask = (t.freqs >= fmin) & (t.freqs <= fmax)
                    if fmask.sum() == 0:
                        continue
                    # Mean across channels and freqs in logratio domain for the logratio figure
                    series_logr = np.nanmean(t.data[:, fmask, :], axis=(0, 1))  # (time,) logratio
                    # Also compute ratio-domain mean for the percent-change figure to avoid geometric-mean bias
                    ratio_data = np.power(10.0, t.data[:, fmask, :])  # (ch, f, t)
                    series_ratio = np.nanmean(ratio_data, axis=(0, 1))  # (time,) arithmetic mean in ratio domain
                    # Clip to subject's available time window
                    s_mask = (t.times >= tmin) & (t.times <= tmax)
                    if s_mask.sum() < 2:
                        continue
                    ts = t.times[s_mask]
                    ys_logr = series_logr[s_mask]
                    ys_ratio = series_ratio[s_mask]
                    # Guard against all-nan
                    if not np.any(np.isfinite(ys_logr)) and not np.any(np.isfinite(ys_ratio)):
                        continue
                    # Fill NaNs by linear interpolation on finite subset (separately for both series)
                    fin_logr = np.isfinite(ys_logr)
                    if fin_logr.sum() >= 2:
                        ys_logr = np.interp(ts, ts[fin_logr], ys_logr[fin_logr])
                    fin_ratio = np.isfinite(ys_ratio)
                    if fin_ratio.sum() >= 2:
                        ys_ratio = np.interp(ts, ts[fin_ratio], ys_ratio[fin_ratio])
                    # Interpolate both to tref
                    yref_logr = np.interp(tref, ts, ys_logr)
                    yref_ratio = np.interp(tref, ts, ys_ratio)
                    band_tc[b].append(yref_logr)  # logratio time course for original figure
                    # Percent change from baseline computed in ratio domain per subject
                    yref_pct = 100.0 * (yref_ratio - 1.0)
                    band_tc_pct[b].append(yref_pct)

            # Plot mean with 95% CI per band
            have_any = any(len(v) >= 2 for v in band_tc.values())
            if have_any:
                # Create a single figure with one subplot per band (stacked vertically)
                valid_bands = [b for b in bands if len(band_tc.get(b, [])) >= 2]
                if len(valid_bands) == 0:
                    logger.info("No bands with >=2 subjects for time-course plotting; skipping.")
                else:
                    nrows = len(valid_bands)
                    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3.2 * nrows), sharex=True)
                    if nrows == 1:
                        axes = [axes]
                    for i, b in enumerate(valid_bands):
                        ax = axes[i]
                        series_list = band_tc.get(b, [])
                        arr = np.vstack(series_list)
                        mu = np.nanmean(arr, axis=0)
                        se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
                        ci = 1.96 * se
                        ax.plot(tref, mu, color=_get_band_color(b), label=f"{b}")
                        ax.fill_between(tref, mu - ci, mu + ci, color=_get_band_color(b), alpha=0.2)
                        # Baseline shading using configured baseline
                        try:
                            b_start, b_end, _ = _validate_baseline_indices(tref, TFR_BASELINE, MIN_BASELINE_SAMPLES)
                            bs = max(float(tref.min()), float(b_start))
                            be = min(float(tref.max()), float(b_end))
                            if be > bs:
                                ax.axvspan(bs, be, alpha=0.1, color='gray')
                        except Exception:
                            pass
                        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
                        ax.set_title(f"{b.capitalize()} (group mean ±95% CI)")
                        ax.set_ylabel("log10(power/baseline)")
                        ax.grid(True, alpha=0.3)
                    axes[-1].set_xlabel("Time (s)")
                    fig.suptitle("Group Band Power Time Courses")
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    _save_fig(fig, gplots / "group_band_power_time_courses")
                    plt.close(fig)
                    logger.info("Saved group band power time courses.")

                    # Duplicate figure: percent change from baseline (ratio mode)
                    fig2, axes2 = plt.subplots(nrows, 1, figsize=(12, 3.2 * nrows), sharex=True)
                    if nrows == 1:
                        axes2 = [axes2]
                    for i, b in enumerate(valid_bands):
                        ax2 = axes2[i]
                        series_list_pct = band_tc_pct.get(b, [])
                        if len(series_list_pct) < 2:
                            # If insufficient subjects with pct data, skip this band
                            continue
                        arrp = np.vstack(series_list_pct)
                        mu = np.nanmean(arrp, axis=0)
                        se = np.nanstd(arrp, axis=0, ddof=1) / np.sqrt(arrp.shape[0])
                        ci = 1.96 * se
                        ax2.plot(tref, mu, color=_get_band_color(b), label=f"{b}")
                        ax2.fill_between(tref, mu - ci, mu + ci, color=_get_band_color(b), alpha=0.2)
                        try:
                            b_start, b_end, _ = _validate_baseline_indices(tref, TFR_BASELINE, MIN_BASELINE_SAMPLES)
                            bs = max(float(tref.min()), float(b_start))
                            be = min(float(tref.max()), float(b_end))
                            if be > bs:
                                ax2.axvspan(bs, be, alpha=0.1, color='gray')
                        except Exception:
                            pass
                        ax2.axvline(0, color='k', linestyle='--', linewidth=0.8)
                        ax2.set_title(f"{b.capitalize()} (group mean ±95% CI)")
                        ax2.set_ylabel("Percent change from baseline (%)")
                        ax2.grid(True, alpha=0.3)
                    axes2[-1].set_xlabel("Time (s)")
                    fig2.suptitle("Group Band Power Time Courses (percent change, ratio-domain averaging)")
                    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
                    _save_fig(fig2, gplots / "group_band_power_time_courses_percent_change")
                    plt.close(fig2)
                    logger.info("Saved group band power time courses (percent change).")
        else:
            logger.info("Skipping group band power time courses: need at least 2 subjects with TFR (saved or computed).")
    except Exception as e:
        logger.warning(f"Group band power time courses failed: {e}")


def main(subjects: Optional[List[str]] = None, task: str = TASK, all_subjects: bool = False):
    # Enforce CLI-provided subjects; allow --all-subjects to scan DERIV_ROOT
    if all_subjects:
        subjects = _get_available_subjects()
        if not subjects:
            raise ValueError(f"No subjects with cleaned epochs found in {DERIV_ROOT}")
    elif subjects is None or len(subjects) == 0:
        raise ValueError("No subjects specified. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
    for sub in subjects:
        process_subject(sub, task)
    # Perform group aggregation if multiple subjects
    if len(subjects) >= 2:
        aggregate_group_level(subjects, task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG feature engineering: power + connectivity (single or multiple subjects)")

    sel = parser.add_mutually_exclusive_group(required=False)
    sel.add_argument(
        "--group", type=str,
        help=(
            "Group to process: 'all' or comma/space-separated subject labels without 'sub-' "
            "(e.g., '0001,0002,0003')."
        ),
    )
    sel.add_argument(
        "--subject", "-s", type=str, action="append",
        help=(
            "BIDS subject label(s) without 'sub-' prefix (e.g., 0001). "
            "Can be specified multiple times."
        ),
    )
    sel.add_argument(
        "--all-subjects", action="store_true",
        help="Process all available subjects with cleaned epochs",
    )
    # Deprecated alias (kept for backward compatibility)
    parser.add_argument("--subjects", nargs="*", default=None, help="[Deprecated] Subject IDs list. Prefer --subject or --group.")

    parser.add_argument("--task", default=TASK, help="Task label (default from config)")
    args = parser.parse_args()

    # Resolve subjects similar to 01/02
    subjects: Optional[List[str]] = None
    if args.group is not None:
        g = args.group.strip()
        if g.lower() in {"all", "*", "@all"}:
            subjects = _get_available_subjects()
        else:
            cand = [s.strip() for s in g.replace(";", ",").replace(" ", ",").split(",") if s.strip()]
            subjects = []
            for s in cand:
                try:
                    if _find_clean_epochs_path(s, args.task) is not None:
                        subjects.append(s)
                    else:
                        print(f"Warning: --group subject '{s}' has no cleaned epochs; skipping")
                except Exception:
                    pass
    elif args.all_subjects:
        subjects = _get_available_subjects()
    elif args.subject:
        _seen = set()
        subjects = []
        for s in args.subject:
            if s not in _seen:
                _seen.add(s)
                subjects.append(s)
    elif args.subjects:
        subjects = list(dict.fromkeys(args.subjects))

    if subjects is None or len(subjects) == 0:
        print("No subjects provided. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
        raise SystemExit(2)

    main(subjects=subjects, task=args.task, all_subjects=False)
