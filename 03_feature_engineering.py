from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath
import glob

# -----------------------------------------------------------------------------
# Config imports
# -----------------------------------------------------------------------------
try:
    # Prefer using the project's config for paths and bands
    from eeg_pipeline.config import (
        bids_root as BIDS_ROOT,
        deriv_root as DERIV_ROOT,
        subjects as SUBJECTS,
        task as TASK,
        features_freq_bands as FEATURES_FREQ_BANDS,
        custom_tfr_freqs as CUSTOM_TFR_FREQS,
        custom_tfr_n_cycles as CUSTOM_TFR_N_CYCLES,
        custom_tfr_decim as CUSTOM_TFR_DECIM,
    )
except Exception:
    # Fallback defaults if import fails
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    BIDS_ROOT = PROJECT_ROOT / "eeg_pipeline" / "bids_output"
    DERIV_ROOT = BIDS_ROOT / "derivatives"
    SUBJECTS = ["001"]
    TASK = "thermalactive"
    FEATURES_FREQ_BANDS = {
        "theta": (4, 8 - 0.1),
        "alpha": (8, 13 - 0.1),
        "beta": (13, 30),
        "gamma": (30 + 0.1, 80),
    }
    CUSTOM_TFR_FREQS = np.arange(1, 101, 1)
    CUSTOM_TFR_N_CYCLES = CUSTOM_TFR_FREQS / 3.0
    CUSTOM_TFR_DECIM = 4

# Bands to compute power features for (per brief)
POWER_BANDS = ["alpha", "beta", "gamma"]

# Plateau window for stimulation (seconds)
PLATEAU_START = 3.0
PLATEAU_END = 10.5

# Candidate event columns to use as regression target (ordered by preference)
TARGET_COLUMNS = [
    "vas_final_coded_rating",
    "vas_final_rating",
    "vas_rating",
    "pain_intensity",
    "pain_rating",
    "rating",
    # Fallbacks to binary pain if continuous ratings not present
    "pain_binary_coded",
    "pain_binary",
    "pain",
]

# -----------------------------------------------------------------------------
# Helper functions (duplicated from 02_time_frequency_analysis with light tweaks)
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def _load_events_df(subject: str, task: str) -> Optional[pd.DataFrame]:
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
            print(f"Warning: failed to read events TSV at {p}: {e}")
            return None
    else:
        print(f"Warning: events TSV not found for subject {subject}: {p}")
        return None


# -----------------------------------------------------------------------------
# Feature extraction helpers
# -----------------------------------------------------------------------------

def _compute_tfr(
    epochs: mne.Epochs,
    freqs: np.ndarray = None,
    n_cycles: np.ndarray = None,
    decim: int = None,
) -> "mne.time_frequency.EpochsTFR":
    """Compute EpochsTFR from cleaned epochs using Morlet wavelets (trial-level)."""
    if freqs is None:
        freqs = CUSTOM_TFR_FREQS
    if n_cycles is None:
        n_cycles = CUSTOM_TFR_N_CYCLES
    if decim is None:
        decim = CUSTOM_TFR_DECIM

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
    print(
        f"Computed TFR: type={type(power).__name__}, n_epochs={power.data.shape[0]}, n_freqs={len(power.freqs)}"
    )
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


def _align_events_to_epochs(events_df: Optional[pd.DataFrame], epochs: mne.Epochs) -> Optional[pd.DataFrame]:
    if events_df is None:
        return None
    aligned = False
    sel = getattr(epochs, "selection", None)
    if sel is not None and len(sel) == len(epochs):
        try:
            if len(events_df) > int(np.max(sel)):
                out = events_df.iloc[sel].reset_index(drop=True)
                aligned = True
                return out
        except Exception:
            pass
    if "sample" in events_df.columns and isinstance(getattr(epochs, "events", None), np.ndarray):
        try:
            samples = epochs.events[:, 0]
            out = events_df.set_index("sample").reindex(samples)
            if len(out) == len(epochs) and not out.isna().all(axis=1).any():
                return out.reset_index()
        except Exception:
            pass
    # Fallback: naive trim
    n = min(len(events_df), len(epochs))
    if n == 0:
        return None
    return events_df.iloc[:n].reset_index(drop=True)


def _time_mask(times: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    return (times >= tmin) & (times <= tmax)


def _freq_mask(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return (freqs >= fmin) & (freqs <= fmax)


def _extract_band_power_features(tfr, bands: List[str]) -> Tuple[pd.DataFrame, List[str]]:
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
            print(f"Warning: band '{band}' not defined in config; skipping.")
            continue
        fmin, fmax = FEATURES_FREQ_BANDS[band]
        fmask = _freq_mask(tfr.freqs, fmin, fmax)  # type: ignore[attr-defined]
        if not np.any(fmask):
            print(f"Warning: TFR freqs contain no points in band '{band}' ({fmin}-{fmax} Hz)")
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


def _extract_connectivity_features(subject: str, task: str, bands: List[str]) -> Tuple[pd.DataFrame, List[str]]:
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
                print(f"Connectivity file missing for {measure} {band}: {pth}")
                continue
            arr = np.load(pth)
            if arr.ndim != 3:
                print(f"Unexpected connectivity shape at {pth}: {arr.shape}")
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
    print(f"=== Feature engineering: sub-{subject}, task-{task} ===")
    features_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "features"
    _ensure_dir(features_dir)

    # Load epochs for alignment and channel names
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None:
        print(f"No cleaned epochs found for sub-{subject}; skipping.")
        return
    print(f"Epochs: {epo_path}")
    epochs = mne.read_epochs(epo_path, preload=False, verbose=False)

    # Load events and align
    events_df = _load_events_df(subject, task)
    aligned_events = _align_events_to_epochs(events_df, epochs)
    if aligned_events is None:
        print("No events available for targets; skipping subject.")
        return

    # Pick target column
    target_col = _pick_target_column(aligned_events)
    if target_col is None:
        print("No suitable target column found in events; skipping.")
        return
    y = pd.to_numeric(aligned_events[target_col], errors="coerce")

    # Compute TFR for power features (trial-level)
    tfr = _compute_tfr(epochs)

    pow_df, pow_cols = _extract_band_power_features(tfr, POWER_BANDS)
    # Connectivity features (if available)
    conn_df, conn_cols = _extract_connectivity_features(subject, task, POWER_BANDS)

    # Align lengths across direct EEG power, connectivity, and targets
    parts = [x for x in [pow_df, conn_df, y] if x is not None and len(x) > 0]
    if not parts:
        print("No features extracted; skipping save.")
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
    eeg_direct_path = features_dir / "features_eeg_direct.tsv"
    eeg_direct_cols_path = features_dir / "features_eeg_direct_columns.tsv"
    print(f"Saving direct EEG features: {eeg_direct_path}")
    # ensure descriptive headers
    if pow_cols:
        pow_df.columns = pow_cols
    pow_df.to_csv(eeg_direct_path, sep="\t", index=False)
    pd.Series(pow_cols, name="feature").to_csv(eeg_direct_cols_path, sep="\t", index=False)

    # Save connectivity features if available
    if conn_df is not None and len(conn_df) > 0:
        conn_path = features_dir / "features_connectivity.tsv"
        print(f"Saving connectivity features: {conn_path}")
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
    print(f"Saving combined features: {combined_path}")
    X_all.to_csv(combined_path, sep="\t", index=False)

    # Save targets
    y_path_tsv = features_dir / "target_vas_ratings.tsv"
    print(f"Saving behavioral target vector: {y_path_tsv} (column: {target_col})")
    y.to_frame(name=target_col).to_csv(y_path_tsv, sep="\t", index=False)

    print(
        f"Done: sub-{subject}, n_trials={n}, n_direct_features={pow_df.shape[1]}, "
        f"n_conn_features={(conn_df.shape[1] if conn_df is not None and len(conn_df) > 0 else 0)}, "
        f"n_all_features={X_all.shape[1]}"
    )


def main(subjects: Optional[List[str]] = None, task: str = TASK):
    if subjects is None or subjects == ["all"]:
        subjects = SUBJECTS
    for sub in subjects:
        process_subject(sub, task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG feature engineering: power + connectivity")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject IDs (e.g., 001 002) or 'all'")
    parser.add_argument("--task", default=TASK, help="Task label (default from config)")
    args = parser.parse_args()

    subs = None if args.subjects in (None, [], ["all"]) else args.subjects
    main(subs, task=args.task)
