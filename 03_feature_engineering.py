from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List
import logging

import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath
import glob
import matplotlib.pyplot as plt
 

# ==========================
# CONFIG
# Load centralized configuration from YAML
# ==========================
from config_loader import load_config, get_legacy_constants
from alignment_utils import align_events_to_epochs_strict, validate_alignment

# Load configuration
config = load_config()

# Extract legacy constants for backward compatibility
_constants = get_legacy_constants(config)

PROJECT_ROOT = _constants["PROJECT_ROOT"]
BIDS_ROOT = _constants["BIDS_ROOT"]
DERIV_ROOT = _constants["DERIV_ROOT"]
SUBJECTS = _constants["SUBJECTS"]
TASK = _constants["TASK"]
FEATURES_FREQ_BANDS = _constants["FEATURES_FREQ_BANDS"]
CUSTOM_TFR_FREQS = _constants["CUSTOM_TFR_FREQS"]
# Import TFR utilities for improved n_cycles calculation
from tfr_utils import compute_adaptive_n_cycles, log_tfr_resolution

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



def _find_clean_epochs_path(subject: str, task: str) -> Optional[Path]:
    # 1) Try BIDSPath construction
    bp = BIDSPath(
        subject=subject,
        task=task,
        datatype="eeg",
        processing="clean",
        suffix="epo",
        extension=".fif",
        root=DERIV_ROOT,
        check=False,
    )
    p1 = bp.fpath
    if p1 and p1.exists():
        return p1

    # 2) Literal fallback
    p2 = DERIV_ROOT / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_proc-clean_epo.fif"
    if p2.exists():
        return p2

    # 3) Simple glob
    subj_eeg_dir = DERIV_ROOT / f"sub-{subject}" / "eeg"
    if subj_eeg_dir.exists():
        cands = sorted(subj_eeg_dir.glob(f"sub-{subject}_task-{task}*epo.fif"))
        for c in cands:
            if "proc-clean" in c.name or "proc-cleaned" in c.name or "clean" in c.name:
                return c
        if cands:
            return cands[0]

    # 4) Last resort recursive
    subj_dir = DERIV_ROOT / f"sub-{subject}"
    if subj_dir.exists():
        for c in sorted(subj_dir.rglob(f"sub-{subject}_task-{task}*epo.fif")):
            return c
    return None


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
    """Save TFR to H5 with a JSON sidecar annotating baseline status.

    Parameters
    ----------
    tfr : mne.time_frequency.EpochsTFR | AverageTFR
        TFR object to save
    out_path : Path
        Path to H5 output
    baseline_window : (float, float)
        Baseline window applied (start, end)
    mode : str
        Baseline mode string (e.g., 'logratio')
    """
    try:
        _ensure_dir(out_path.parent)
        # Save TFR
        tfr.save(str(out_path), overwrite=True)
        # Sidecar
        sidecar = {
            "baseline_applied": True,
            "baseline_mode": mode,
            "baseline_window": [float(baseline_window[0]), float(baseline_window[1])],
            "created_by": "03_feature_engineering.py",
            "comment": getattr(tfr, "comment", None),
        }
        import json
        with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        if logger:
            logger.info(f"Saved TFR and sidecar: {out_path} (+ .json)")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save TFR with sidecar to {out_path}: {e}")


def _load_events_df(subject: str, task: str, logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    ebp = BIDSPath(
        subject=subject,
        task=task,
        datatype="eeg",
        suffix="events",
        extension=".tsv",
        root=BIDS_ROOT,
        check=False,
    )
    p = ebp.fpath
    if p is None:
        p = BIDS_ROOT / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_events.tsv"
    if p.exists():
        try:
            return pd.read_csv(p, sep="\t")
        except Exception as e:
            msg = f"Failed to read events TSV at {p}: {e}"
            if logger:
                logger.warning(msg)
            else:
                print(f"Warning: {msg}")
            return None
    else:
        msg = f"Events TSV not found for subject {subject}: {p}"
        if logger:
            logger.warning(msg)
        else:
            print(f"Warning: {msg}")
        return None


def _setup_logging(subject: str) -> logging.Logger:
    """Set up logging with console and file handlers for feature engineering."""
    logger = logging.getLogger(f"feature_engineering_sub_{subject}")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if already set
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "logs"  # e.g., derivatives/sub-001/eeg/logs/
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME
    file_handler = logging.FileHandler(log_file, mode='w')  # Overwrite each run
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


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

def _pick_target_column(df: pd.DataFrame) -> Optional[str]:
    for c in TARGET_COLUMNS:
        if c in df.columns:
            return c
    # Heuristic: any column containing 'vas' or 'rating'
    for c in df.columns:
        cl = c.lower()
        if ("vas" in cl or "rating" in cl) and df[c].dtype != "O":
            return c
    return None


# Import strict alignment utilities
from alignment_utils import align_events_to_epochs_strict

def _align_events_to_epochs(events_df: Optional[pd.DataFrame], epochs: mne.Epochs, logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """Align events to epochs using strict alignment - no unsafe fallbacks."""
    return align_events_to_epochs_strict(events_df, epochs, logger)


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
    """Return a consistent color for a band, with safe fallback."""
    return str(config.get(f"visualization.band_colors.{band}", "#1f77b4"))



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
            
            # Add baseline period marker
            ax.axvspan(-5, 0, alpha=0.2, color='gray', label='Baseline')
            
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
    events_df = _load_events_df(subject, task, logger)
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


def main(subjects: Optional[List[str]] = None, task: str = TASK, all_subjects: bool = False):
    # Enforce CLI-provided subjects; allow --all-subjects to scan DERIV_ROOT
    if all_subjects:
        subjects = _get_available_subjects()
        if not subjects:
            raise ValueError(f"No subjects with cleaned epochs found in {DERIV_ROOT}")
    elif subjects is None or len(subjects) == 0:
        raise ValueError("No subjects specified. Use --subjects or --all-subjects.")
    for sub in subjects:
        process_subject(sub, task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG feature engineering: power + connectivity")
    subj_group = parser.add_mutually_exclusive_group(required=True)
    subj_group.add_argument("--subjects", nargs="*", help="Subject IDs to process (e.g., 001 002)")
    subj_group.add_argument("--all-subjects", action="store_true", help="Process all available subjects with cleaned epochs")
    parser.add_argument("--task", default=TASK, help="Task label (default from config)")
    args = parser.parse_args()

    main(subjects=args.subjects, task=args.task, all_subjects=args.all_subjects)
