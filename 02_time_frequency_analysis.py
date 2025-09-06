import os
import sys
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import logging

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath

# Load centralized configuration
from config_loader import load_config, get_legacy_constants
from tfr_utils import compute_adaptive_n_cycles

config = load_config()
config.setup_matplotlib()

# Extract legacy constants
_constants = get_legacy_constants(config)

PROJECT_ROOT = _constants["PROJECT_ROOT"]
BIDS_ROOT = _constants["BIDS_ROOT"]
DERIV_ROOT = _constants["DERIV_ROOT"]
TASK = _constants["TASK"]
SUBJECTS = _constants["SUBJECTS"]
FEATURES_FREQ_BANDS = _constants["FEATURES_FREQ_BANDS"]
PSYCH_TEMP_COLUMNS = _constants["PSYCH_TEMP_COLUMNS"]
RATING_COLUMNS = _constants["RATING_COLUMNS"]
PAIN_BINARY_COLUMNS = _constants["PAIN_BINARY_COLUMNS"]
POWER_BANDS_TO_USE = _constants["POWER_BANDS_TO_USE"]
PLATEAU_WINDOW = _constants["PLATEAU_WINDOW"]
FIG_DPI = int(config.get("output.fig_dpi", 300))
SAVE_FORMATS = config.get("output.save_formats", ["png"])  # minimal control
LOG_FILE_NAME = config.get("logging.file_names.time_frequency", "02_time_frequency_analysis.log")
USE_SPEARMAN_DEFAULT = bool(config.get("statistics.use_spearman_default", True))
PARTIAL_COVARS_DEFAULT = config.get("statistics.partial_covars_default", None)
BOOTSTRAP_DEFAULT = int(config.get("random.bootstrap_default", 0))
N_PERM_DEFAULT = int(config.get("statistics.n_perm_default", 0))
DO_GROUP_DEFAULT = bool(config.get("statistics.do_group_default", False))
GROUP_ONLY_DEFAULT = bool(config.get("statistics.group_only_default", False))
BUILD_REPORTS_DEFAULT = bool(config.get("statistics.build_reports_default", False))
DEFAULT_TEMPERATURE_STRATEGY = config.get("time_frequency_analysis.temperature_strategy", "pooled")
DEFAULT_PLATEAU_TMIN = float(config.get("time_frequency_analysis.plateau_window", [3.0, 10.0])[0])
DEFAULT_PLATEAU_TMAX = float(config.get("time_frequency_analysis.plateau_window", [3.0, 10.0])[1])

# Extract parameters from config
DEFAULT_TASK = TASK
FREQ_MIN = float(config.get("time_frequency_analysis.tfr.freq_min", 1.0))
FREQ_MAX = float(config.get("time_frequency_analysis.tfr.freq_max", 100.0))
N_FREQS = int(config.get("time_frequency_analysis.tfr.n_freqs", 40))
N_CYCLES_FACTOR = float(config.get("time_frequency_analysis.tfr.n_cycles_factor", 2.0))
TFR_DECIM = int(config.get("time_frequency_analysis.tfr.decim", 4))
TFR_PICKS = config.get("time_frequency_analysis.tfr.picks", "eeg")
BASELINE = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
BAND_BOUNDS = {k: (v[0], (v[1] if v[1] is not None else None)) for k, v in dict(config.get("time_frequency_analysis.bands", {
    "theta": [4.0, 7.9],
    "alpha": [8.0, 12.9],
    "beta": [13.0, 30.0],
    "gamma": [30.1, 80.0],
})).items()}
FIG_PAD_INCH = float(config.get("output.pad_inches", 0.02))
BBOX_INCHES = config.get("output.bbox_inches", "tight")
TOPO_CONTOURS = int(config.get("time_frequency_analysis.topo_contours", 6))
TOPO_CMAP = config.get("time_frequency_analysis.topo_cmap", "RdBu_r")
COLORBAR_FRACTION = float(config.get("time_frequency_analysis.colorbar_fraction", 0.03))
COLORBAR_PAD = float(config.get("time_frequency_analysis.colorbar_pad", 0.02))
ROI_MASK_PARAMS_DEFAULT = dict(config.get("time_frequency_analysis.roi_mask_params", {
    "marker": "o",
    "markerfacecolor": "w",
    "markeredgecolor": "k",
    "linewidth": 0.5,
    "markersize": 4,
}))
TEMPERATURE_COLUMNS = config.get("event_columns.temperature", [])
MIN_BASELINE_SAMPLES = int(config.get("time_frequency_analysis.min_baseline_samples", 5))
# Alignment behavior: default to strict (fail on misalignment) unless explicitly allowed to trim
ALLOW_MISALIGNED_TRIM = bool(
    config.get("time_frequency_analysis.allow_misaligned_trim", False)
)


def _validate_baseline_indices(
    times: np.ndarray,
    baseline: Tuple[Optional[float], Optional[float]],
    min_samples: int = MIN_BASELINE_SAMPLES,
) -> Tuple[float, float, np.ndarray]:
    """Validate and return baseline window parameters.
    
    Args:
        times: Time vector from epochs
        baseline: Baseline window (start, end) in seconds
        min_samples: Minimum required samples in baseline window
    
    Returns:
        Tuple of (b_start, b_end, baseline_indices)
    
    Raises:
        ValueError: If baseline window is invalid
    """
    b_start, b_end = baseline
    if b_start is None:
        b_start = times[0]
    if b_end is None:
        b_end = 0.0

    # Allow baselines that end exactly at 0.0 s (common MNE convention)
    if b_end > 0:
        raise ValueError(
            "Baseline window must end at or before 0 s (stimulus onset)"
        )
    
    baseline_mask = (times >= b_start) & (times <= b_end)
    baseline_indices = np.where(baseline_mask)[0]
    
    if len(baseline_indices) < min_samples:
        raise ValueError(
            f"Baseline window contains only {len(baseline_indices)} samples "
            f"(minimum {min_samples} required)"
        )
    
    return b_start, b_end, baseline_indices


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _get_available_subjects() -> List[str]:
    """Find all available subjects in DERIV_ROOT with cleaned epochs files."""
    subjects: List[str] = []
    if not DERIV_ROOT.exists():
        return subjects
    for subj_dir in DERIV_ROOT.glob("sub-*"):
        if subj_dir.is_dir():
            subject_id = subj_dir.name[4:]
            # Check if cleaned epochs exist for default task
            if _find_clean_epochs_path(subject_id, DEFAULT_TASK) is not None:
                subjects.append(subject_id)
    return sorted(subjects)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _roi_definitions() -> Dict[str, list[str]]:
    """Sensor-space ROIs defined in the configuration file.

    Patterns are regexes matched case-insensitively against channel names.
    """
    rois = config.get("time_frequency_analysis.rois", {
        "Frontal": [r"^(Fpz|Fp[12]|AFz|AF[3-8]|Fz|F[1-8])$"],
        "Central": [r"^(Cz|C[1-6])$"],
        "Parietal": [r"^(Pz|P[1-8])$"],
        "Occipital": [r"^(Oz|O[12]|POz|PO[3-8])$"],
        "Temporal": [r"^(T7|T8|TP7|TP8|FT7|FT8)$"],
        "Sensorimotor": [r"^(FC[234]|FCz)$", r"^(C[234]|Cz)$", r"^(CP[234]|CPz)$"],
    })
    return {roi: list(patterns) for roi, patterns in rois.items()}


def _canonicalize_ch_name(ch: str) -> str:
    """Return a canonical version of a channel name for robust ROI matching.

    This function removes common vendor-specific prefixes/suffixes and
    reference labels so that regex patterns like "^F3$" or prefix checks like
    "starts with 'FP'" can work even if the raw channel name looks like
    "EEG Fp1-Ref" or "F3 LE".

    Transformations:
    - Strip leading/trailing whitespace
    - Remove leading 'EEG', optionally followed by space or dash
    - Remove any substring starting at the first '-' or '/' (e.g., references)
    - Remove spaces
    - Drop common trailing reference labels (Ref, LE, RE, M1, M2, A1, A2, AVG/AVE)
    - Preserve case-insensitive matching by callers
    """
    s = ch.strip()
    try:
        s = re.sub(r"^(EEG[ \-_]*)", "", s, flags=re.IGNORECASE)
        # Drop anything after a hyphen or forward slash (e.g., '-Ref', '/Ref')
        s = re.split(r"[-/]", s)[0]
        # Remove spaces
        s = re.sub(r"\s+", "", s)
        # Remove common trailing reference tokens
        s = re.sub(r"(Ref|LE|RE|M1|M2|A1|A2|AVG|AVE)$", "", s, flags=re.IGNORECASE)
    except Exception:
        # Fail safe: return original if anything goes wrong
        return ch
    return s


def _find_roi_channels(info: mne.Info, patterns: list[str]) -> list[str]:
    chs = info["ch_names"]
    # Precompute canonical forms for robust matching
    canon_map = {ch: _canonicalize_ch_name(ch) for ch in chs}
    out: list[str] = []
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        # Match against either the raw name or its canonicalized form
        for ch in chs:
            cn = canon_map.get(ch, ch)
            if rx.match(ch) or rx.match(cn):
                out.append(ch)
    # Preserve original channel order and deduplicate
    seen = set()
    ordered = []
    for ch in chs:
        if ch in out and ch not in seen:
            seen.add(ch)
            ordered.append(ch)
    return ordered



def _build_rois(info: mne.Info) -> Dict[str, list[str]]:
    """Build ROI channel mappings.
    
    Args:
        info: MNE info object containing channel information
    
    Returns:
        Dictionary mapping ROI names to lists of channel names
    """
    roi_map = {}
    for roi, pats in _roi_definitions().items():
        chans = _find_roi_channels(info, pats)
        roi_map[roi] = chans
    return roi_map


def _build_group_roi_map_from_channels(ch_names: list[str]) -> Dict[str, list[str]]:
    """Build a group ROI map by heuristics from a union of channel names.

    Uses broad, prefix-based rules to maximize coverage while keeping ROIs sensible.
    Channel order is preserved as in `ch_names`.
    """
    def pick(prefixes: list[str]) -> list[str]:
        out = []
        seen = set()
        for ch in ch_names:
            up = _canonicalize_ch_name(ch).upper()
            for p in prefixes:
                if up.startswith(p.upper()):
                    if ch not in seen:
                        seen.add(ch)
                        out.append(ch)
                    break
        return out

    roi_map: Dict[str, list[str]] = {
        "Frontal": pick(["FP", "AF", "F"]),
        "Central": pick(["C", "CZ"]),
        "Parietal": pick(["P", "PZ"]),
        "Occipital": pick(["O", "PO", "OZ"]),
        "Temporal": pick(["T", "TP", "FT"]),
        "Sensorimotor": pick(["FC", "C", "CP", "CZ"]),
    }
    # Deduplicate overlaps by preference: Sensorimotor keeps FC/C/CP; Frontal drops FC
    # Remove FC-prefixed channels from Frontal to avoid overlap with Sensorimotor
    roi_map["Frontal"] = [
        ch for ch in roi_map["Frontal"]
        if not _canonicalize_ch_name(ch).upper().startswith("FC")
    ]
    return roi_map


def _is_valid_eeg_channel(ch_name: str) -> bool:
    """Basic validation for EEG channel names.
    
    Accepts common EEG naming conventions:
    - 10-20 system: Fp1, F3, C4, P4, O2, etc.
    - Extended systems: FC1, CP2, etc.
    - Some common variants and reference channels
    """
    import re
    # Pattern matches: Letter(s) + number(s), possibly with z suffix
    # Examples: F3, FC1, Cz, T7, TP10, etc.
    pattern = r'^[A-Za-z]{1,3}[0-9]*[z]?$'
    return bool(re.match(pattern, ch_name.strip()))


def run_quick_baseline_diagnostics(subject: str, task: str = DEFAULT_TASK) -> None:
    """Compute a quick set of baseline diagnostics and Cz plots for a subject.

    - Loads cleaned epochs
    - Computes TFR (Morlet) with config freqs/n_cycles/decim
    - Generates raw and baseline-corrected Cz plots
    - Runs baseline diagnostics (comparison of methods, QC baseline vs plateau)
    """
    logger = _setup_logging(subject)
    out_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots" / "02_time_frequency_analysis"
    _ensure_dir(out_dir)

    # Load epochs
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None or not epo_path.exists():
        logger.error(f"No cleaned epochs for sub-{subject}, task-{task}")
        return
    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)

    # Compute TFR
    freqs = np.linspace(float(FREQ_MIN), float(FREQ_MAX), int(N_FREQS))
    n_cycles = np.maximum(freqs / float(N_CYCLES_FACTOR), 3.0)
    try:
        tfr = mne.time_frequency.tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            average=False,
            decim=int(TFR_DECIM),
            n_jobs=-1,
            picks=TFR_PICKS,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"TFR computation failed: {e}")
        return

    # Plots and diagnostics
    try:
        plot_cz_all_trials_raw(tfr, out_dir, logger)
        plot_cz_all_trials(tfr, out_dir, baseline=BASELINE, plateau_window=(DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX), logger=logger)
        diagnostic_baseline_correction_methods(tfr, out_dir, baseline=BASELINE, plateau_window=(DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX), logger=logger)
        qc_baseline_plateau_power(tfr, out_dir, baseline=BASELINE, plateau_window=(DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX), logger=logger)
        logger.info(f"Quick baseline diagnostics complete for sub-{subject}")
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")


def _find_clean_epochs_path(subject: str, task: str) -> Optional[Path]:
    """Locate cleaned epochs file under derivatives for a subject.

    Prefers BIDSPath with processing='clean' and suffix='epo'. Falls back to glob.
    """
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

    # 2) Try the literal name specified by user brief
    p2 = DERIV_ROOT / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_proc-clean_epo.fif"
    if p2.exists():
        return p2

    # 3) Fallback: any *proc-clean*_epo.fif under subject eeg dir for the task
    subj_eeg_dir = DERIV_ROOT / f"sub-{subject}" / "eeg"
    if subj_eeg_dir.exists():
        cands = sorted(subj_eeg_dir.glob(f"sub-{subject}_task-{task}*epo.fif"))
        for c in cands:
            if "proc-clean" in c.name or "proc-cleaned" in c.name or "clean" in c.name:
                return c
        if cands:
            return cands[0]

    # 4) Last resort: recursive search (could be slow but scoped to subject)
    subj_dir = DERIV_ROOT / f"sub-{subject}"
    if subj_dir.exists():
        for c in sorted(subj_dir.rglob(f"sub-{subject}_task-{task}*epo.fif")):
            return c
    return None


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
    """Set up logging with console and file handlers for time-frequency analysis."""
    logger = logging.getLogger(f"time_frequency_analysis_sub_{subject}")
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


def _setup_group_logging() -> logging.Logger:
    """Set up logging for group-level time-frequency analysis."""
    logger = logging.getLogger("time_frequency_analysis_group")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    log_dir = DERIV_ROOT / "group" / "eeg" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _find_temperature_column(events_df: Optional[pd.DataFrame]) -> Optional[str]:
    """Heuristically find the temperature column in events metadata.

    Tries common names such as 'stimulus_temp', 'stimulus_temperature', 'temp', 'temperature'.
    Returns the column name if found, else None.
    """
    if events_df is None:
        return None
    candidates = TEMPERATURE_COLUMNS
    for c in candidates:
        if c in events_df.columns:
            return c
    return None


def _find_pain_binary_column(events_df: Optional[pd.DataFrame]) -> Optional[str]:
    """Find the pain binary column in events metadata using config mapping.

    Tries configured pain binary column names from PAIN_BINARY_COLUMNS.
    Returns the column name if found, else None.
    """
    if events_df is None:
        return None
    for c in PAIN_BINARY_COLUMNS:
        if c in events_df.columns:
            return c
    return None




def _format_temp_label(val: float) -> str:
    """Format a temperature value as a safe label for paths, e.g., 47.3 -> '47p3'."""
    try:
        v = float(val)
    except Exception:
        return _sanitize(str(val))
    # Keep one decimal if present in data; replace '.' with 'p'
    s = f"{v:.1f}"
    return s.replace(".", "p")




def _apply_baseline_safe(
    tfr_obj,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    mode: str = "logratio",
    logger: Optional[logging.Logger] = None,
    force: bool = False,
):
    """Apply baseline safely and idempotently.

    Uses an explicit sentinel in `tfr_obj.comment` to detect prior application.
    Falls back to applying baseline on AverageTFR with a warning, but prefers
    per-epoch baseline (EpochsTFR) before averaging.
    """
    sentinel = "BASELINED:"

    try:
        # Skip if a sentinel indicates baseline has already been applied
        comment = getattr(tfr_obj, "comment", None)
        if not force and isinstance(comment, str) and sentinel in comment:
            msg = "Detected baseline-corrected TFR by sentinel; skipping re-application."
            if logger:
                logger.info(msg)
            else:
                print(msg)
            return

        # Warn if baselining an already-averaged TFR (less ideal)
        if isinstance(tfr_obj, mne.time_frequency.AverageTFR):
            warn_msg = (
                "Applying baseline to AverageTFR (averaged) — prefer per-epoch baseline"
            )
            if logger:
                logger.warning(warn_msg)
            else:
                print(f"Warning: {warn_msg}")

        times = np.asarray(tfr_obj.times)
        b_start, b_end, _ = _validate_baseline_indices(times, baseline)
        tfr_obj.apply_baseline(baseline=(b_start, b_end), mode=mode)

        # Record sentinel in comment to avoid double application
        try:
            prev = getattr(tfr_obj, "comment", "")
            tag = f"{sentinel}mode={mode};win=({b_start:.3f},{b_end:.3f})"
            tfr_obj.comment = (f"{prev} | {tag}" if prev else tag)
        except Exception:
            pass

        msg = f"Applied baseline {(b_start, b_end)} with mode='{mode}'."
        if logger:
            logger.info(msg)
        else:
            print(msg)
    except Exception as e:
        msg = f"Baseline correction skipped (no pre-stim interval or error): {e}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)


def _pick_central_channel(info: mne.Info, preferred: str = "Cz", logger: Optional[logging.Logger] = None) -> str:
    ch_names = [ch for ch in info['ch_names']]
    if preferred in ch_names:
        return preferred
    # Try case-insensitive match
    for nm in ch_names:
        if nm.lower() == preferred.lower():
            return nm
    # Fallback: first EEG channel
    picks = mne.pick_types(info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise RuntimeError("No EEG channels available for plotting.")
    fallback = ch_names[picks[0]]
    msg = f"Channel '{preferred}' not found; using '{fallback}' instead."
    if logger:
        logger.warning(msg)
    else:
        print(msg)
    return fallback


def _save_fig(fig_obj: Any, out_dir: Path, name: str, formats: Optional[list[str]] = None, logger: Optional[logging.Logger] = None) -> None:
    """Save a matplotlib Figure or a list of Figures.

    If a list is provided, the first figure uses the given name; subsequent
    figures get an index suffix before the extension.
    """
    _ensure_dir(out_dir)
    figs: list[plt.Figure]
    if isinstance(fig_obj, list):
        figs = fig_obj
    else:
        figs = [fig_obj]

    base = name
    stem, ext = (base.rsplit(".", 1) + [""])[:2]
    exts = formats if formats is not None else ([ext] if ext else ["png"])  # default to PNG if no ext
    for i, f in enumerate(figs):
        saved_any = False
        # Prefer constrained_layout to avoid tight_layout warnings (common with MNE topomaps/colorbars)
        for ext_i in exts:
            out_name = (f"{stem}.{ext_i}" if i == 0 else f"{stem}_{i+1}.{ext_i}")
            out_path = out_dir / out_name
            try:
                # 1) Try with constrained layout and no bbox_inches (to avoid conflicts)
                try:
                    f.set_constrained_layout(True)
                except Exception:
                    pass
                try:
                    f.canvas.draw()
                except Exception:
                    pass
                f.savefig(out_path, dpi=FIG_DPI)
                saved_any = True
                msg = f"Saved: {out_path}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
            except Exception as e:
                # 2) Fallback: disable constrained layout and try tight_layout/subplots_adjust then save with bbox options
                try:
                    f.set_constrained_layout(False)
                except Exception:
                    pass
                try:
                    f.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08)
                except Exception:
                    pass
                try:
                    f.canvas.draw()
                except Exception:
                    pass
                try:
                    f.savefig(out_path, dpi=FIG_DPI, bbox_inches=BBOX_INCHES, pad_inches=FIG_PAD_INCH)
                    saved_any = True
                    msg = f"Saved (fallback layout): {out_path}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                except Exception as e2:
                    # 3) Last resort: save without tight bbox
                    try:
                        f.savefig(out_path, dpi=FIG_DPI)
                        saved_any = True
                        msg = f"Saved (no tight bbox) due to layout error for: {out_path}. Reason: {e2} (original: {e})"
                        if logger:
                            logger.info(msg)
                        else:
                            print(msg)
                    except Exception as e3:
                        msg = f"Failed to save figure to {out_path}: {e3} (original errors: {e2}; {e})"
                        if logger:
                            logger.error(msg)
                        else:
                            print(msg)
        plt.close(f)
        if not saved_any:
            msg = f"Warning: no output saved for figure named '{name}'"
            if logger:
                logger.warning(msg)
            else:
                print(msg)


def _average_tfr_band(
    tfr_avg: "mne.time_frequency.AverageTFR",
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
) -> Optional[np.ndarray]:
    """Average an AverageTFR over [fmin,fmax] x [tmin,tmax] -> (n_channels,) array.

    Returns None if the selection window is invalid.
    """
    try:
        freqs = np.asarray(tfr_avg.freqs)
        times = np.asarray(tfr_avg.times)
        f_mask = (freqs >= fmin) & (freqs <= fmax)
        t_mask = (times >= tmin) & (times < tmax)
        if f_mask.sum() == 0 or t_mask.sum() == 0:
            return None
        # data shape: (n_channels, n_freqs, n_times)
        sel = tfr_avg.data[:, f_mask, :][:, :, t_mask]
        return sel.mean(axis=(1, 2))
    except Exception:
        return None


def _plot_topomap_on_ax(
    ax: "plt.Axes",
    data: np.ndarray,
    info: mne.Info,
    mask: Optional[np.ndarray] = None,
    mask_params: Optional[dict] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Plot a single topomap for channel-wise data on the provided axes.

    Tries to pass the Info directly to MNE's topomap; falls back if needed.
    """
    try:
        mne.viz.plot_topomap(
            data,
            info,  # many MNE versions accept Info directly for pos
            axes=ax,
            show=False,
            mask=mask,
            mask_params=mask_params or {},
            sensors=True,
            contours=TOPO_CONTOURS,
            cmap=TOPO_CMAP,
            vlim=(vmin, vmax) if vmin is not None and vmax is not None else None,
        )
    except Exception:
        # Fallback: pick EEG channels explicitly
        picks = mne.pick_types(info, eeg=True, exclude=[])
        info_eeg = mne.pick_info(info, sel=picks)
        mne.viz.plot_topomap(
            data[picks],
            info_eeg,
            axes=ax,
            show=False,
            mask=mask[picks] if isinstance(mask, np.ndarray) else None,
            mask_params=mask_params or {},
            sensors=True,
            contours=TOPO_CONTOURS,
            cmap=TOPO_CMAP,
            vlim=(vmin, vmax) if vmin is not None and vmax is not None else None,
        )


def _get_consistent_bands(max_freq_available: Optional[float] = None) -> Dict[str, Tuple[float, float]]:
    """Get consistent frequency band definitions across subjects.
    
    Returns fixed band definitions with a consistent gamma upper bound,
    ensuring cross-subject comparability by avoiding variable frequency limits.
    Uses config-defined gamma upper limit of 80.0 Hz.
    
    Args:
        max_freq_available: Optional maximum frequency available in data.
                          If provided, will cap gamma upper bound if necessary.
    
    Returns:
        Dictionary mapping band names to (low, high) frequency tuples.
    """
    bands: Dict[str, Tuple[float, float]] = {}
    # Include theta if defined in configuration
    if "theta" in BAND_BOUNDS:
        bands["theta"] = BAND_BOUNDS["theta"]
    # Always include alpha and beta
    bands["alpha"] = BAND_BOUNDS["alpha"]
    bands["beta"] = BAND_BOUNDS["beta"]
    
    gamma_lower, gamma_upper = BAND_BOUNDS["gamma"]
    if gamma_upper is None:
        # Use config-defined 80 Hz upper limit for consistency across subjects
        gamma_upper = 80.0
    
    # Cap by available frequency if provided and necessary
    if max_freq_available is not None and gamma_upper > max_freq_available:
        gamma_upper = max_freq_available
    
    bands["gamma"] = (gamma_lower, gamma_upper)
    return bands


def _robust_sym_vlim(
    arrs: "np.ndarray | list[np.ndarray]",
    q_low: float = 0.02,
    q_high: float = 0.98,
    cap: float = 0.25,
    min_v: float = 1e-6,
) -> float:
    """Compute robust symmetric vlim (positive scalar) centered at 0.

    Parameters optimized for EEG power data in logratio units:
    - q_low/q_high: 2%-98% quantiles to exclude extreme outliers
    - cap: Maximum vlim of 0.25 (logratio units) ≈ 78% power change
    - min_v: Minimum vlim to avoid zero scaling
    
    - Concatenates arrays, removes non-finite, takes [q_low, q_high] quantiles.
    - Uses max absolute quantile and caps by `cap` to avoid outliers.
    - Returns a positive scalar v to be used as vmin=-v, vmax=+v.
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
        v = min(v, float(cap))
        return v
    except Exception:
        return cap


def _compute_power_and_events(
    subject: str,
    task: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional["mne.time_frequency.EpochsTFR"], Optional[pd.DataFrame]]:
    """Compute per-trial TFR (EpochsTFR) and load aligned events metadata.

    Returns (power, events_df) or (None, None) on failure.
    """
    # Load epochs
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None or not epo_path.exists():
        msg = f"No cleaned epochs for sub-{subject}, task-{task}"
        if logger:
            logger.error(msg)
        else:
            print(f"Error: {msg}")
        return None, None
    try:
        epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
    except Exception as e:
        msg = f"Failed to load epochs for sub-{subject}: {e}"
        if logger:
            logger.error(msg)
        else:
            print(f"Error: {msg}")
        return None, None

    # Load and align events
    events_df = _load_events_df(subject, task, logger)
    if events_df is not None:
        aligned = False
        sel = getattr(epochs, "selection", None)
        if sel is not None and len(sel) == len(epochs):
            try:
                if len(events_df) > int(np.max(sel)):
                    events_aligned = events_df.iloc[sel].reset_index(drop=True)
                    events_df = events_aligned
                    epochs.metadata = events_df.copy()
                    aligned = True
                    if logger:
                        logger.info("Aligned metadata via epochs.selection")
                
            except Exception as e:
                if logger:
                    logger.warning(f"Selection-based alignment failed: {e}")
        if not aligned and "sample" in events_df.columns and isinstance(getattr(epochs, "events", None), np.ndarray):
            try:
                samples = epochs.events[:, 0]
                events_by_sample = events_df.set_index("sample")
                events_aligned = events_by_sample.reindex(samples)
                if len(events_aligned) == len(epochs) and not events_aligned.isna().all(axis=1).any():
                    events_df = events_aligned.reset_index()
                    epochs.metadata = events_df.copy()
                    aligned = True
                    if logger:
                        logger.info("Aligned metadata via events.sample")
            except Exception as e:
                if logger:
                    logger.warning(f"Sample-based alignment failed: {e}")
        if not aligned:
            n = min(len(events_df), len(epochs))
            if len(events_df) != len(epochs):
                msg = f"Epochs ({len(epochs)}) and events ({len(events_df)}) mismatch; proceeding with trim to n={n}."
                if logger:
                    logger.warning(msg)
                else:
                    print(f"Warning: {msg}")
                if not ALLOW_MISALIGNED_TRIM:
                    if logger:
                        logger.error("Trimming not allowed by config; skipping subject.")
                    return None, None
            try:
                if len(epochs) != n:
                    epochs = epochs[:n]
                events_df = events_df.iloc[:n].reset_index(drop=True)
                epochs.metadata = events_df.copy()
            except Exception as e:
                if logger:
                    logger.error(f"Failed to attach trimmed metadata: {e}")
                return None, None
    else:
        if logger:
            logger.warning("Events missing; contrasts will be skipped for this subject.")

    # Compute per-trial TFR
    try:
        freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQS)
        n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=N_CYCLES_FACTOR, min_cycles=3.0)
        power = mne.time_frequency.tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            average=False,
            decim=TFR_DECIM,
            n_jobs=-1,
            picks=TFR_PICKS,
            verbose=False,
        )
    except Exception as e:
        if logger:
            logger.error(f"TFR computation failed for sub-{subject}: {e}")
        else:
            print(f"Error: TFR computation failed for sub-{subject}: {e}")
        return None, None
    return power, events_df
def plot_cz_all_trials_raw(tfr, out_dir: Path, logger: Optional[logging.Logger] = None) -> None:
    # Plot Cz TFR without baseline correction
    tfr_copy = tfr.copy()

    # Average across trials if EpochsTFR
    if isinstance(tfr_copy, mne.time_frequency.EpochsTFR):
        tfr_avg = tfr_copy.average()
    else:
        tfr_avg = tfr_copy  # already AverageTFR

    cz = _pick_central_channel(tfr_avg.info, preferred="Cz", logger=logger)
    try:
        fig = tfr_avg.plot(picks=cz, show=False)
        try:
            fig.suptitle("Cz TFR — all trials (raw, no baseline)", fontsize=12)
        except Exception:
            pass
        _save_fig(fig, out_dir, f"tfr_Cz_all_trials_raw.png", logger=logger)
    except Exception as e:
        msg = f"Cz TFR raw plot failed: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)


def plot_cz_all_trials(
    tfr,
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
    logger: Optional[logging.Logger] = None,
) -> None:
    # Work on a copy for baseline correction
    tfr_copy = tfr.copy()
    _apply_baseline_safe(tfr_copy, baseline=baseline, mode="logratio", logger=logger)

    # Average across trials if EpochsTFR
    if isinstance(tfr_copy, mne.time_frequency.EpochsTFR):
        tfr_avg = tfr_copy.average()
    else:
        tfr_avg = tfr_copy  # already AverageTFR

    cz = _pick_central_channel(tfr_avg.info, preferred="Cz", logger=logger)
    try:
        # Robust symmetric vlim based on Cz data across all times/freqs
        ch_idx = tfr_avg.info["ch_names"].index(cz)
        arr = np.asarray(tfr_avg.data[ch_idx])  # (n_freqs, n_times)
        vabs = _robust_sym_vlim(arr)
        # Plateau mean annotation
        times = np.asarray(tfr_avg.times)
        tmin_req, tmax_req = plateau_window
        tmask = (times >= float(tmin_req)) & (times < float(tmax_req))
        if not np.any(tmask):
            if logger:
                logger.warning(
                    f"Plateau window [{tmin_req}, {tmax_req}] outside data range; using entire time span"
                )
            else:
                print(
                    f"Warning: Plateau window [{tmin_req}, {tmax_req}] outside data range; using entire time span"
                )
            tmask = np.ones_like(times, dtype=bool)
        mu = float(np.nanmean(arr[:, tmask]))
        pct = (10.0 ** (mu) - 1.0) * 100.0
        fig = tfr_avg.plot(picks=cz, vlim=(-vabs, +vabs), show=False)
        try:
            fig.suptitle(
                f"Cz TFR — all trials (baseline logratio)\nvlim ±{vabs:.2f}; mean %Δ vs BL={pct:+.0f}%",
                fontsize=12,
            )
        except Exception:
            pass
        _save_fig(fig, out_dir, f"tfr_Cz_all_trials_baseline_logratio.png", logger=logger)
    except Exception as e:
        msg = f"Cz TFR plot failed: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)


def diagnostic_baseline_correction_methods(
    tfr,
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Diagnostic comparison of different baseline correction methods.
    
    Compares raw power, logratio, ratio, percent, and zscore baseline corrections
    to identify potential issues causing generalized suppression.
    """
    try:
        diag_dir = out_dir / "baseline_diagnostics"
        _ensure_dir(diag_dir)
        
        # Only work with EpochsTFR - if already averaged, warn and skip
        if not isinstance(tfr, mne.time_frequency.EpochsTFR):
            msg = "Baseline correction diagnostics require EpochsTFR. Skipping."
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return
        
        # Get EEG channels for scalp-averaged analysis
        eeg_picks = mne.pick_types(tfr.info, eeg=True, exclude=[])
        if len(eeg_picks) == 0:
            msg = "No EEG channels found for baseline diagnostics"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return
        
        # Baseline correction methods to test
        correction_modes = ['logratio', 'ratio', 'percent', 'zscore']
        
        # Extract frequency bands
        bands = {
            "Alpha": (8.0, 13.0),
            "Beta": (13.0, 30.0), 
            "Gamma": (30.0, 80.0)
        }
        
        # Create comparison figure - 6 subplots: raw + 4 correction modes + stats
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Raw power (no baseline correction)
        try:
            tfr_raw = tfr.copy()
            tfr_raw_avg = tfr_raw.average()
            times = np.asarray(tfr_raw_avg.times)
            
            for band_name, (fmin, fmax) in bands.items():
                freqs = np.asarray(tfr_raw_avg.freqs)
                fmask = (freqs >= fmin) & (freqs <= fmax)
                if fmask.sum() == 0:
                    continue
                    
                # Scalp-averaged band power over time
                band_data = tfr_raw_avg.data[eeg_picks, :, :][:, fmask, :].mean(axis=(0, 1))
                axes[0].plot(times, band_data, label=f"{band_name} ({fmin:.0f}-{fmax:.0f}Hz)", alpha=0.8)
            
            axes[0].set_title("Raw Power (No Baseline Correction)")
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel("Power (arbitrary units)")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)
            axes[0].axvline(0, color='red', linestyle='--', alpha=0.5, label='Stimulus')
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Raw plot failed: {e}", transform=axes[0].transAxes, ha='center')
        
        # Plots 2-5: Each baseline correction mode
        for plot_idx, mode in enumerate(correction_modes, start=1):
            try:
                # Apply baseline to epochs, then average
                tfr_corrected = tfr.copy()
                _apply_baseline_safe(tfr_corrected, baseline=baseline, mode=mode, logger=None)
                tfr_corrected_avg = tfr_corrected.average()
                
                times = np.asarray(tfr_corrected_avg.times)
                
                for band_name, (fmin, fmax) in bands.items():
                    freqs = np.asarray(tfr_corrected_avg.freqs)
                    fmask = (freqs >= fmin) & (freqs <= fmax)
                    if fmask.sum() == 0:
                        continue
                        
                    # Scalp-averaged band power over time
                    band_data = tfr_corrected_avg.data[eeg_picks, :, :][:, fmask, :].mean(axis=(0, 1))
                    axes[plot_idx].plot(times, band_data, label=f"{band_name} ({fmin:.0f}-{fmax:.0f}Hz)", alpha=0.8)
                
                axes[plot_idx].set_title(f"Baseline Correction: {mode}")
                axes[plot_idx].set_xlabel("Time (s)")
                axes[plot_idx].axvline(0, color='red', linestyle='--', alpha=0.5)
                axes[plot_idx].axhline(0 if mode in ['logratio', 'percent', 'zscore'] else 1, 
                                     color='gray', linestyle='-', alpha=0.5)
                
                if mode == "logratio":
                    axes[plot_idx].set_ylabel("log10(power/baseline)")
                elif mode == "ratio":
                    axes[plot_idx].set_ylabel("power/baseline")
                elif mode == "percent":
                    axes[plot_idx].set_ylabel("% change")
                elif mode == "zscore":
                    axes[plot_idx].set_ylabel("Z-score")
                axes[plot_idx].legend(fontsize=8)
                axes[plot_idx].grid(True, alpha=0.3)
                
            except Exception as e:
                axes[plot_idx].text(0.5, 0.5, f"{mode} failed: {e}", 
                                   transform=axes[plot_idx].transAxes, ha='center')
            
        # Summary statistics in last subplot
        axes[5].axis('off')
        stats_text = f"Baseline Window: {baseline[0]:.1f} to {baseline[1]:.1f} s\n"
        stats_text += f"Plateau Window: {plateau_window[0]:.1f} to {plateau_window[1]:.1f} s\n"
        stats_text += f"Number of epochs: {len(tfr)}\n"
        stats_text += f"EEG channels: {len(eeg_picks)}\n\n"
        stats_text += "Baseline Correction Modes:\n"
        stats_text += "• logratio: log10(power/baseline)\n"
        stats_text += "• ratio: power/baseline\n"
        stats_text += "• percent: 100*(power-baseline)/baseline\n"
        stats_text += "• zscore: (power-baseline_mean)/baseline_std"
        
        axes[5].text(0.05, 0.95, stats_text, transform=axes[5].transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        fig.suptitle("Baseline Correction Method Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        _save_fig(fig, diag_dir, "baseline_correction_methods_comparison.png", formats=["png", "svg"], logger=logger)
        
        if logger:
            logger.info(f"Saved baseline correction methods diagnostic plot")
        
    except Exception as e:
        msg = f"Baseline correction methods diagnostic failed: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)


def diagnostic_alternative_baselines(
    tfr,
    out_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Test alternative baseline windows to identify optimal baseline period.
    
    Tests different baseline windows to find one without anticipatory artifacts.
    """
    try:
        diag_dir = out_dir / "baseline_diagnostics"
        _ensure_dir(diag_dir)
        
        # Handle EpochsTFR vs AverageTFR
        if isinstance(tfr, mne.time_frequency.EpochsTFR):
            tfr_avg = tfr.average()
        else:
            tfr_avg = tfr.copy()
            
        # Get EEG channels for scalp-averaged analysis  
        eeg_picks = mne.pick_types(tfr_avg.info, eeg=True, exclude=[])
        if len(eeg_picks) == 0:
            raise RuntimeError("No EEG channels found for diagnostic analysis")
        eeg_ch_names = [tfr_avg.info['ch_names'][i] for i in eeg_picks]
        
        times = np.asarray(tfr_avg.times)
        
        # Test different baseline windows
        baseline_windows = [
            (-3.0, -2.0),  # Earlier baseline (avoid anticipation)
            (-2.5, -1.5),  # Shifted baseline
            (-2.0, -1.0),  # Current baseline
            (-1.5, -0.5),  # Later baseline
            (-4.0, -1.0),  # Longer baseline
            (-2.0, 0.0),   # Extended to stimulus
        ]
        
        # Get frequency bands
        bands = _get_consistent_bands()
        freqs = np.asarray(tfr_avg.freqs)
        
        # Create comparison figure
        n_bands = len(bands)
        fig, axes = plt.subplots(n_bands, 2, figsize=(12, 3 * n_bands), constrained_layout=True)
        if n_bands == 1:
            axes = axes[None, :]
            
        for band_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            fmax_eff = min(fmax, freqs.max())
            if fmin >= fmax_eff:
                for c in range(2):
                    axes[band_idx, c].axis('off')
                continue
            # Plot raw power trace - scalp averaged
            raw_data = tfr_avg.data[eeg_picks, :, :].mean(axis=0)  # Average across EEG channels
            # Frequency mask for current band
            fmask = (freqs >= fmin) & (freqs <= fmax_eff)
            band_power = raw_data[fmask, :].mean(axis=0)
            
            axes[band_idx, 0].plot(times, band_power, 'k-', alpha=0.8, label='Raw power')
            axes[band_idx, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Stimulus')
            
            # Show all baseline windows
            colors = plt.cm.tab10(np.linspace(0, 1, len(baseline_windows)))
            for i, (b_start, b_end) in enumerate(baseline_windows):
                # Check if baseline window is valid for this data
                if b_start < times.min() or b_end > times.max():
                    continue
                if b_end > 0:
                    continue  # Skip invalid baselines that extend into post-stimulus
                    
                axes[band_idx, 0].axvspan(b_start, b_end, alpha=0.3, color=colors[i], 
                                        label=f'BL {b_start:.1f}-{b_end:.1f}s')
                
            axes[band_idx, 0].set_title(f"Raw {band_name} Power - Scalp Average ({len(eeg_picks)} channels)")
            axes[band_idx, 0].set_xlabel("Time (s)")
            axes[band_idx, 0].set_ylabel("Power (a.u.)")
            axes[band_idx, 0].legend(fontsize=8)
            axes[band_idx, 0].grid(True, alpha=0.3)
            
            # Plot baseline-corrected traces for each window
            for i, (b_start, b_end) in enumerate(baseline_windows):
                # Check validity
                if b_start < times.min() or b_end > times.max() or b_end > 0:
                    continue
                    
                try:
                    tfr_test = tfr.copy()  # Use original epochs
                    _apply_baseline_safe(tfr_test, baseline=(b_start, b_end), mode="logratio", logger=None)
                    tfr_test_avg = tfr_test.average()  # Then average
                    corrected_data = tfr_test_avg.data[eeg_picks, :, :].mean(axis=0)  # Scalp average
                    # Use the same band frequency mask
                    band_corrected = corrected_data[fmask, :].mean(axis=0)
                    
                    axes[band_idx, 1].plot(times, band_corrected, color=colors[i], alpha=0.8,
                                         label=f'BL {b_start:.1f}-{b_end:.1f}s')
                except Exception:
                    continue
                    
            axes[band_idx, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[band_idx, 1].axhline(0, color='gray', linestyle='-', alpha=0.5)
            axes[band_idx, 1].set_title(f"Baseline-Corrected {band_name} Power")
            axes[band_idx, 1].set_xlabel("Time (s)")
            axes[band_idx, 1].set_ylabel("log10(power/baseline)")
            axes[band_idx, 1].legend(fontsize=8)
            axes[band_idx, 1].grid(True, alpha=0.3)
            
        fig.suptitle("Alternative Baseline Window Comparison", fontsize=14, fontweight='bold')
        _save_fig(fig, diag_dir, "alternative_baselines_comparison.png", formats=["png", "svg"], logger=logger)
        
        if logger:
            logger.info(f"Saved alternative baseline diagnostic plots to {diag_dir}")
        else:
            print(f"Saved alternative baseline diagnostic plots to {diag_dir}")
            
    except Exception as e:
        msg = f"Alternative baseline diagnostics failed: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)


def qc_baseline_plateau_power(
    tfr,
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
    logger: Optional[logging.Logger] = None,
) -> None:
    """QC plots and stats comparing baseline vs plateau band power on raw TFR.

    - Computes mean power within each band over baseline and plateau windows.
    - Saves per-band histograms of baseline power and log10(plateau/baseline).
    - Writes a TSV summary with basic stats per band.
    """
    try:
        qc_dir = out_dir / "qc"
        _ensure_dir(qc_dir)

        # Handle EpochsTFR vs AverageTFR
        data = getattr(tfr, "data", None)
        if data is None:
            return
        # Ensure 4D: (epochs, ch, f, t)
        if data.ndim == 3:
            data = data[None, ...]
        if data.ndim != 4:
            return

        freqs = np.asarray(tfr.freqs)
        times = np.asarray(tfr.times)

        # Build time masks and validate baseline coverage
        try:
            b_start, b_end, tmask_base = _validate_baseline_indices(times, baseline)
        except ValueError as e:
            msg = f"QC skipped: {e}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return

        tmask_plat = (times >= plateau_window[0]) & (times < plateau_window[1])

        if not np.any(tmask_plat):
            msg = f"QC skipped: plateau samples={int(tmask_plat.sum())}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return

        # Prepare a baseline-corrected AverageTFR for topomap-consistent summaries
        # We apply logratio baseline per-frequency, then average across epochs.
        tfr_topo_avg = None
        try:
            tfr_copy = tfr.copy()
            _apply_baseline_safe(tfr_copy, baseline=baseline, mode="logratio", logger=logger)
            if isinstance(tfr_copy, mne.time_frequency.EpochsTFR):
                tfr_topo_avg = tfr_copy.average()
            else:
                tfr_topo_avg = tfr_copy  # already AverageTFR
        except Exception as e:
            msg = (
                f"QC: failed to prepare baseline-corrected TFR for topomap-consistent summaries: {e}"
            )
            if logger:
                logger.warning(msg)
            else:
                print(msg)

        # Summaries TSV
        rows = []
        eps = 1e-20

        for band, (fmin, fmax) in BAND_BOUNDS.items():
            fmask = (freqs >= float(fmin)) & (freqs <= (float(fmax) if fmax is not None else freqs.max()))
            if not np.any(fmask):
                continue

            base = data[:, :, fmask, :][:, :, :, tmask_base].mean(axis=(2, 3))
            plat = data[:, :, fmask, :][:, :, :, tmask_plat].mean(axis=(2, 3))

            base_flat = base.reshape(-1)
            ratio_log = np.log10((plat.reshape(-1) + eps) / (base_flat + eps))

            # Save hist figure
            try:
                fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
                axes[0].hist(base_flat, bins=50, color="tab:blue", alpha=0.8)
                axes[0].set_title(f"Baseline power — {band}")
                axes[0].set_xlabel("Power (a.u.)")
                axes[0].set_ylabel("Count")
                axes[1].hist(ratio_log, bins=50, color="tab:orange", alpha=0.8)
                axes[1].set_title(f"log10(plateau/baseline) — {band}")
                axes[1].set_xlabel("log10 ratio")
                axes[1].set_ylabel("Count")
                try:
                    fig.suptitle(
                        f"Baseline vs Plateau QC — {band}\n(baseline={b_start:.2f}–{b_end:.2f}s; plateau={plateau_window[0]:.2f}–{plateau_window[1]:.2f}s)",
                        fontsize=10,
                    )
                except Exception:
                    pass
                _save_fig(fig, qc_dir, f"qc_baseline_plateau_hist_{band}.png", logger=logger)
            except Exception as e:
                if logger:
                    logger.warning(f"QC hist failed for band {band}: {e}")
                else:
                    print(f"QC hist failed for band {band}: {e}")

            # Per-channel histogram of topomap-consistent values (baseline logratio within plateau)
            topo_vals: Optional[np.ndarray] = None
            if tfr_topo_avg is not None:
                try:
                    fmin_eff = float(fmin)
                    fmax_eff = float(fmax) if fmax is not None else float(freqs.max())
                    topo_vals = _average_tfr_band(
                        tfr_topo_avg,
                        fmin=fmin_eff,
                        fmax=fmax_eff,
                        tmin=float(plateau_window[0]),
                        tmax=float(plateau_window[1]),
                    )
                except Exception as e:
                    if logger:
                        logger.warning(f"Topomap-consistent per-channel values failed for {band}: {e}")
                    else:
                        print(f"Topomap-consistent per-channel values failed for {band}: {e}")

            if topo_vals is not None and np.isfinite(topo_vals).any():
                try:
                    fig2, ax2 = plt.subplots(1, 1, figsize=(4.8, 3.2), constrained_layout=True)
                    ax2.hist(topo_vals, bins=50, color="tab:green", alpha=0.8)
                    ax2.set_title(f"Per-channel Δ (topomap-consistent) — {band}")
                    ax2.set_xlabel("log10 ratio")
                    ax2.set_ylabel("Count")
                    _save_fig(fig2, qc_dir, f"qc_band_topomap_values_hist_{band}.png", logger=logger)
                except Exception as e:
                    if logger:
                        logger.warning(f"QC per-channel hist failed for band {band}: {e}")
                    else:
                        print(f"QC per-channel hist failed for band {band}: {e}")

            # Aggregate stats
            row = {
                "band": band,
                "baseline_mean": float(np.nanmean(base_flat)),
                "baseline_median": float(np.nanmedian(base_flat)),
                "plateau_mean": float(np.nanmean(plat.reshape(-1))),
                "plateau_median": float(np.nanmedian(plat.reshape(-1))),
                "log10_ratio_mean": float(np.nanmean(ratio_log)),
                "log10_ratio_median": float(np.nanmedian(ratio_log)),
                "n_baseline_samples": int(tmask_base.sum()),
                "n_plateau_samples": int(tmask_plat.sum()),
            }
            if topo_vals is not None and np.isfinite(topo_vals).any():
                row["log10_ratio_mean_topomap"] = float(np.nanmean(topo_vals))
                row["log10_ratio_median_topomap"] = float(np.nanmedian(topo_vals))
            else:
                row["log10_ratio_mean_topomap"] = float("nan")
                row["log10_ratio_median_topomap"] = float("nan")
            rows.append(row)

        # Save summary TSV
        try:
            import pandas as pd  # local import to avoid global dependency if unused

            if rows:
                df = pd.DataFrame(rows)
                df_path = qc_dir / "qc_baseline_plateau_summary.tsv"
                df.to_csv(df_path, sep="\t", index=False)
                msg = f"Saved QC summary: {df_path}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
        except Exception as e:
            if logger:
                logger.warning(f"QC summary save failed: {e}")
            else:
                print(f"QC summary save failed: {e}")
    except Exception as e:
        if logger:
            logger.warning(f"QC baseline/plateau encountered an error: {e}")
        else:
            print(f"QC baseline/plateau encountered an error: {e}")


def contrast_pain_nonpain(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
    logger: Optional[logging.Logger] = None,
) -> None:
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        msg = "Contrast requires EpochsTFR (trial-level). Skipping contrasts and using only overall average."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return
    pain_col = _find_pain_binary_column(events_df)
    if pain_col is None:
        msg = f"Events with pain binary column {PAIN_BINARY_COLUMNS} required for contrast; skipping."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return

    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df)
    n = min(n_epochs, n_meta)
    # Enforce strictness when no TFR metadata is available
    if n_epochs != n_meta and not (getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns):
        msg = (
            f"Error: tfr epochs ({n_epochs}) != events rows ({n_meta}) and no matching pain column in TFR metadata. "
            f"Cannot guarantee alignment; skipping contrasts."
        )
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return

    # Prefer labels from TFR metadata if available to ensure perfect alignment
    if getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns:
        pain_vec = pd.to_numeric(tfr.metadata.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
    else:
        pain_vec = pd.to_numeric(events_df.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
    pain_mask = np.asarray(pain_vec == 1, dtype=bool)
    non_mask = np.asarray(pain_vec == 0, dtype=bool)

    # Debug counts before deciding to skip
    msg = f"Debug: n_epochs={n_epochs}, n_meta={n_meta}, n={n}, len_pain_vec={len(pain_vec)}"
    if logger:
        logger.debug(msg)
    else:
        print(msg)
    msg = f"Pain/non-pain counts (n={n}): pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())}."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        msg = "One of the groups has zero trials; skipping contrasts."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return

    # Subset and baseline-correct per-epoch before averaging
    tfr_sub = tfr.copy()[:n]
    if len(pain_mask) != len(tfr_sub):
        msg = f"Warning: mask length ({len(pain_mask)}) != TFR epochs ({len(tfr_sub)}); reslicing to match."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        n2 = min(len(tfr_sub), len(pain_mask))
        tfr_sub = tfr_sub[:n2]
        pain_mask = pain_mask[:n2]
        non_mask = non_mask[:n2]
        msg = f"Debug after reslice: len(tfr_sub)={len(tfr_sub)}, len(pain_mask)={len(pain_mask)}"
        if logger:
            logger.debug(msg)
        else:
            print(msg)
    _apply_baseline_safe(tfr_sub, baseline=baseline, mode="logratio", logger=logger)

    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    cz = _pick_central_channel(tfr_pain.info, preferred="Cz", logger=logger)

    # Plot Cz for both conditions
    try:
        ch_idx = tfr_pain.info["ch_names"].index(cz)
        arr_pain = np.asarray(tfr_pain.data[ch_idx])
        arr_non = np.asarray(tfr_non.data[ch_idx])
        vabs_pn = _robust_sym_vlim([arr_pain, arr_non])
        # Plateau mean annotations
        times = np.asarray(tfr_pain.times)
        tmin_req, tmax_req = plateau_window
        tmask = (times >= float(tmin_req)) & (times < float(tmax_req))
        if not np.any(tmask):
            if logger:
                logger.warning(
                    f"Plateau window [{tmin_req}, {tmax_req}] outside data range; using entire time span"
                )
            else:
                print(
                    f"Warning: Plateau window [{tmin_req}, {tmax_req}] outside data range; using entire time span"
                )
            tmask = np.ones_like(times, dtype=bool)
        mu_pain = float(np.nanmean(arr_pain[:, tmask]))
        pct_pain = (10.0 ** (mu_pain) - 1.0) * 100.0
        mu_non = float(np.nanmean(arr_non[:, tmask]))
        pct_non = (10.0 ** (mu_non) - 1.0) * 100.0
        fig = tfr_pain.plot(picks=cz, vlim=(-vabs_pn, +vabs_pn), show=False)
        try:
            fig.suptitle(
                f"Cz — Pain (baseline logratio)\nvlim ±{vabs_pn:.2f}; mean %Δ vs BL={pct_pain:+.0f}%",
                fontsize=12,
            )
        except Exception:
            pass
        _save_fig(fig, out_dir, "tfr_Cz_painful_baseline_logratio.png", logger=logger)
    except Exception as e:
        msg = f"Cz painful plot failed: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
    try:
        fig = tfr_non.plot(picks=cz, vlim=(-vabs_pn, +vabs_pn), show=False)
        try:
            fig.suptitle(
                f"Cz — Non-pain (baseline logratio)\nvlim ±{vabs_pn:.2f}; mean %Δ vs BL={pct_non:+.0f}%",
                fontsize=12,
            )
        except Exception:
            pass
        _save_fig(fig, out_dir, "tfr_Cz_nonpainful_baseline_logratio.png", logger=logger)
    except Exception as e:
        msg = f"Cz non-painful plot failed: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)

    # Difference (pain - non)
    try:
        tfr_diff = tfr_pain.copy()
        tfr_diff.data = tfr_pain.data - tfr_non.data
        tfr_diff.comment = "pain-minus-nonpain"
        # Robust symmetric vlim for diff
        arr_diff = np.asarray(arr_pain) - np.asarray(arr_non)
        vabs_diff = _robust_sym_vlim(arr_diff)
        mu_diff = float(np.nanmean(arr_diff[:, tmask]))
        pct_diff = (10.0 ** (mu_diff) - 1.0) * 100.0
        fig = tfr_diff.plot(picks=cz, vlim=(-vabs_diff, +vabs_diff), show=False)
        try:
            fig.suptitle(
                f"Cz — Pain minus Non (baseline logratio)\nvlim ±{vabs_diff:.2f}; Δ% vs BL={pct_diff:+.0f}%",
                fontsize=12,
            )
        except Exception:
            pass
        _save_fig(fig, out_dir, "tfr_Cz_pain_minus_nonpain_baseline_logratio.png", logger=logger)
    except Exception as e:
        msg = f"Cz difference plot failed: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)

    # Grouped topomap grid for alpha/beta/gamma x (pain/non/diff)
    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    tmin_eff = float(max(times.min(), tmin_req))
    tmax_eff = float(min(times.max(), tmax_req))
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = _get_consistent_bands(max_freq_available=fmax_available)
    tmin, tmax = tmin_eff, tmax_eff

    # Final counts after potential reslicing
    n_pain = int(pain_mask.sum())
    n_non = int(non_mask.sum())
    cond_labels = [f"Pain (n={n_pain})", f"Non-pain (n={n_non})", "", "Pain - Non"]
    n_rows = len(bands)
    # Insert a narrow spacer column between Non-pain and Diff
    n_cols = 4
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.25, 1.0], "wspace": 0.3},
    )
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis('off')
            continue
        # Average data for each condition
        pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        if pain_data is None or non_data is None:
            for c in range(n_cols):
                axes[r, c].axis('off')
            continue
        diff_data = pain_data - non_data
        # Scalp-averaged values for compact annotation (logratio and percent change) - EEG only
        pain_mu = float(np.nanmean(pain_data))
        non_mu = float(np.nanmean(non_data))
        diff_mu = float(np.nanmean(diff_data))
        # Robust symmetric scaling for pain/non and diff within band
        vabs_pn = _robust_sym_vlim([pain_data, non_data])
        diff_abs = _robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else 0.0
        # Plot Pain (col 0), Non-pain (col 1), leave col 2 empty, Diff (col 3)
        # Pain
        ax = axes[r, 0]
        try:
            mne.viz.plot_topomap(
                pain_data,
                tfr_pain.info,
                axes=ax,
                show=False,
                vlim=(-vabs_pn, +vabs_pn),
                cmap=TOPO_CMAP,
            )
        except Exception:
            _plot_topomap_on_ax(ax, pain_data, tfr_pain.info, vmin=-vabs_pn, vmax=+vabs_pn)
        # Annotate percent change vs baseline
        ax.text(0.5, 1.02, f"%Δ={(10**pain_mu - 1)*100:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        # Non-pain
        ax = axes[r, 1]
        try:
            mne.viz.plot_topomap(
                non_data,
                tfr_pain.info,
                axes=ax,
                show=False,
                vlim=(-vabs_pn, +vabs_pn),
                cmap=TOPO_CMAP,
            )
        except Exception:
            _plot_topomap_on_ax(ax, non_data, tfr_pain.info, vmin=-vabs_pn, vmax=+vabs_pn)
        # Annotate percent change vs baseline
        ax.text(0.5, 1.02, f"%Δ={(10**non_mu - 1)*100:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        # Spacer column (col 2)
        axes[r, 2].axis('off')
        # Diff
        ax = axes[r, 3]
        try:
            mne.viz.plot_topomap(
                diff_data,
                tfr_pain.info,
                axes=ax,
                show=False,
                vlim=((-diff_abs, +diff_abs) if diff_abs > 0 else None),
                cmap=TOPO_CMAP,
            )
        except Exception:
            _plot_topomap_on_ax(
                ax,
                diff_data,
                tfr_pain.info,
                vmin=(-diff_abs if diff_abs > 0 else None),
                vmax=(+diff_abs if diff_abs > 0 else None),
            )
        # Annotate percent change difference
        pct_mu = (10**(diff_mu) - 1.0) * 100.0
        ax.text(0.5, 1.02, f"Δ%={pct_mu:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        if r == 0:
            for c_title in (0, 1, 3):
                axes[r, c_title].set_title(cond_labels[c_title], fontsize=9, pad=4, y=1.04)
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)
        # Add compact colorbars per row
        try:
            sm_pn = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vabs_pn, vcenter=0.0, vmax=vabs_pn), cmap=TOPO_CMAP)
            sm_pn.set_array([])
            cbar_pn = fig.colorbar(
                sm_pn, ax=[axes[r, 0], axes[r, 1]], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD
            )
            try:
                cbar_pn.set_label("log10(power/baseline)")
            except Exception:
                pass
            if diff_abs > 0:
                sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=TOPO_CMAP)
                sm_diff.set_array([])
                cbar_diff = fig.colorbar(
                    sm_diff, ax=axes[r, 3], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD
                )
                try:
                    cbar_diff.set_label("log10(power/baseline)")
                except Exception:
                    pass
        except Exception:
            pass
    fig.suptitle(f"Topomaps (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)", fontsize=12)
    try:
        # Removed bottom 'Conditions' label per request; keep y-label for frequency bands
        fig.supylabel("Frequency bands", fontsize=10)
    except Exception:
        pass
    _save_fig(fig, out_dir, "topomap_grid_bands_pain_non_diff_baseline_logratio.png", formats=["png", "svg"])


def contrast_maxmin_temperature(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
) -> None:
    """Topomap grid comparing highest vs lowest temperature (Δ=log10(power/baseline)).

    Layout mirrors pain/non-pain grid: [Max temp, Min temp, spacer, Max - Min] across alpha/beta/gamma.
    """
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        print("Max-vs-min temperature contrast requires EpochsTFR; skipping.")
        return
    if events_df is None:
        print("Max-vs-min temperature contrast requires events_df; skipping.")
        return
    temp_col = _find_temperature_column(events_df)
    if temp_col is None:
        print("Max-vs-min temperature contrast: no temperature column found; skipping.")
        return

    # Align lengths and obtain temperature series (prefer TFR metadata if available)
    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df)
    n = min(n_epochs, n_meta)
    if getattr(tfr, "metadata", None) is not None and temp_col in tfr.metadata.columns:
        temp_series = tfr.metadata.iloc[:n][temp_col]
    else:
        temp_series = events_df.iloc[:n][temp_col]

    # Determine unique temperature levels (rounded to 1 decimal)
    try:
        temps = (
            pd.to_numeric(temp_series, errors="coerce").round(1).dropna().unique()
        )
        temps = sorted(map(float, temps))
    except Exception:
        temps = sorted(pd.Series(temp_series).dropna().unique())
    if len(temps) < 2:
        print("Max-vs-min temperature contrast: need at least 2 temperature levels; skipping.")
        return
    t_min = float(min(temps))
    t_max = float(max(temps))

    # Build masks for lowest and highest temperature
    try:
        mask_min = pd.to_numeric(temp_series, errors="coerce").round(1) == round(t_min, 1)
        mask_max = pd.to_numeric(temp_series, errors="coerce").round(1) == round(t_max, 1)
    except Exception:
        mask_min = temp_series == t_min
        mask_max = temp_series == t_max
    mask_min = np.asarray(mask_min, dtype=bool)
    mask_max = np.asarray(mask_max, dtype=bool)
    if mask_min.sum() == 0 or mask_max.sum() == 0:
        print(
            f"Max-vs-min temperature contrast: zero trials in one group (min n={int(mask_min.sum())}, max n={int(mask_max.sum())}); skipping."
        )
        return

    # Subset and baseline-correct per-epoch before averaging
    tfr_sub = tfr.copy()[:n]
    if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
        n2 = min(len(tfr_sub), len(mask_min), len(mask_max))
        tfr_sub = tfr_sub[:n2]
        mask_min = mask_min[:n2]
        mask_max = mask_max[:n2]
    _apply_baseline_safe(tfr_sub, baseline=baseline, mode="logratio")

    tfr_min = tfr_sub[mask_min].average()
    tfr_max = tfr_sub[mask_max].average()

    # Effective plateau window
    times = np.asarray(tfr_max.times)
    tmin_req, tmax_req = plateau_window
    tmin_eff = float(max(times.min(), tmin_req))
    tmax_eff = float(min(times.max(), tmax_req))
    tmin, tmax = tmin_eff, tmax_eff

    # Bands with consistent gamma limits across subjects
    fmax_available = float(np.max(tfr_max.freqs))
    bands = _get_consistent_bands(max_freq_available=fmax_available)

    # Grid like pain/non: [Max, Min, spacer, Max-Min]
    n_rows = len(bands)
    n_cols = 4
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.25, 1.0], "wspace": 0.3},
    )
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        max_data = _average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        min_data = _average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        if max_data is None or min_data is None:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        diff_data = max_data - min_data

        # Means for compact annotations (log10 ratio units)
        max_mu = float(np.nanmean(max_data))
        min_mu = float(np.nanmean(min_data))
        diff_mu = float(np.nanmean(diff_data))

        # Robust symmetric scaling for Max/Min within band and their diff
        vabs_pn = _robust_sym_vlim([max_data, min_data])
        diff_abs = _robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else 0.0

        # Max temp (col 0)
        ax = axes[r, 0]
        try:
            mne.viz.plot_topomap(
                max_data,
                tfr_max.info,
                axes=ax,
                show=False,
                vlim=(-vabs_pn, +vabs_pn),
                cmap=TOPO_CMAP,
            )
        except Exception:
            _plot_topomap_on_ax(ax, max_data, tfr_max.info, vmin=-vabs_pn, vmax=+vabs_pn)
        ax.text(0.5, 1.02, f"%Δ={(10**max_mu - 1)*100:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)

        # Min temp (col 1)
        ax = axes[r, 1]
        try:
            mne.viz.plot_topomap(
                min_data,
                tfr_min.info,
                axes=ax,
                show=False,
                vlim=(-vabs_pn, +vabs_pn),
                cmap=TOPO_CMAP,
            )
        except Exception:
            _plot_topomap_on_ax(ax, min_data, tfr_min.info, vmin=-vabs_pn, vmax=+vabs_pn)
        ax.text(0.5, 1.02, f"%Δ={(10**min_mu - 1)*100:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)

        # Spacer (col 2)
        axes[r, 2].axis("off")

        # Diff (Max - Min) (col 3)
        ax = axes[r, 3]
        try:
            mne.viz.plot_topomap(
                diff_data,
                tfr_max.info,
                axes=ax,
                show=False,
                vlim=((-diff_abs, +diff_abs) if diff_abs > 0 else None),
                cmap=TOPO_CMAP,
            )
        except Exception:
            _plot_topomap_on_ax(
                ax,
                diff_data,
                tfr_max.info,
                vmin=(-diff_abs if diff_abs > 0 else None),
                vmax=(+diff_abs if diff_abs > 0 else None),
            )
        pct_mu = (10**(diff_mu) - 1.0) * 100.0
        ax.text(0.5, 1.02, f"Δ%={pct_mu:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)

        if r == 0:
            axes[r, 0].set_title(f"Max {t_max:.1f}°C (n={int(mask_max.sum())})", fontsize=9, pad=4, y=1.04)
            axes[r, 1].set_title(f"Min {t_min:.1f}°C (n={int(mask_min.sum())})", fontsize=9, pad=4, y=1.04)
            axes[r, 3].set_title("Max - Min", fontsize=9, pad=4, y=1.04)
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)

        # Colorbars per row
        try:
            sm_pn = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs_pn, vmax=+vabs_pn), cmap=TOPO_CMAP)
            sm_pn.set_array([])
            cbar_pn = fig.colorbar(
                sm_pn, ax=[axes[r, 0], axes[r, 1]], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD
            )
            try:
                cbar_pn.set_label("log10(power/baseline)")
            except Exception:
                pass
            if diff_abs > 0:
                sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=TOPO_CMAP)
                sm_diff.set_array([])
                cbar_diff = fig.colorbar(
                    sm_diff, ax=axes[r, 3], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD
                )
                try:
                    cbar_diff.set_label("log10(power/baseline)")
                except Exception:
                    pass
        except Exception:
            pass

    try:
        fig.suptitle(
            f"Topomaps: Max vs Min temperature (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)",
            fontsize=12,
        )
        fig.supylabel("Frequency bands", fontsize=10)
    except Exception:
        pass
    _save_fig(
        fig,
        out_dir,
        "topomap_grid_bands_maxmin_temp_diff_baseline_logratio.png",
        formats=["png", "svg"],
    )


def contrast_pain_nonpain_topomaps_rois(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    roi_map: Dict[str, list[str]],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
) -> None:
    """Topomaps per ROI (highlighting ROI sensors) for pain/non-pain and difference.

    Uses full-head interpolation but highlights ROI sensors via mask.
    """
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        print("ROI topomap contrast requires EpochsTFR; skipping.")
        return
    pain_col = _find_pain_binary_column(events_df)
    if pain_col is None:
        print(f"Events with pain binary column {PAIN_BINARY_COLUMNS} required for ROI topomap contrasts; skipping.")
        return

    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df)
    n = min(n_epochs, n_meta)
    if n_epochs != n_meta and not (getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns):
        print(
            f"ROI topomaps: tfr epochs ({n_epochs}) != events rows ({n_meta}) and no matching pain column in TFR metadata; skipping."
        )
        return

    # Prefer labels from TFR metadata if available
    if getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns:
        pain_vec = pd.to_numeric(tfr.metadata.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
    else:
        pain_vec = pd.to_numeric(events_df.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
    pain_mask = np.asarray(pain_vec == 1, dtype=bool)
    non_mask = np.asarray(pain_vec == 0, dtype=bool)
    print(f"ROI topomaps pain/non-pain counts (n={n}): pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())}.")
    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        print("ROI topomaps: one of the groups has zero trials; skipping.")
        return

    tfr_sub = tfr.copy()[:n]
    if len(pain_mask) != len(tfr_sub):
        print(f"Warning (ROI topomaps): mask length ({len(pain_mask)}) != TFR epochs ({len(tfr_sub)}); reslicing to match.")
        n2 = min(len(tfr_sub), len(pain_mask))
        tfr_sub = tfr_sub[:n2]
        pain_mask = pain_mask[:n2]
        non_mask = non_mask[:n2]
        print(f"Debug after reslice (ROI topomaps): len(tfr_sub)={len(tfr_sub)}, len(pain_mask)={len(pain_mask)}")
    _apply_baseline_safe(tfr_sub, baseline=baseline, mode="logratio")
    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    fmax_available = float(np.max(tfr_pain.freqs))
    bands: Dict[str, Tuple[float, float]] = {}
    if "theta" in BAND_BOUNDS:
        bands["theta"] = BAND_BOUNDS["theta"]
    bands["alpha"] = BAND_BOUNDS["alpha"]
    bands["beta"] = BAND_BOUNDS["beta"]
    bands["gamma"] = (
        BAND_BOUNDS["gamma"][0],
        fmax_available if BAND_BOUNDS["gamma"][1] is None else BAND_BOUNDS["gamma"][1],
    )
    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    tmin_eff = float(max(times.min(), tmin_req))
    tmax_eff = float(min(times.max(), tmax_req))
    tmin, tmax = tmin_eff, tmax_eff

    ch_names = tfr_pain.info["ch_names"]
    # Add counts to condition labels
    n_pain = int(pain_mask.sum())
    n_non = int(non_mask.sum())
    cond_labels = [f"Pain (n={n_pain})", f"Non-pain (n={n_non})", "", "Pain - Non"]
    for roi, roi_chs in roi_map.items():
        # Build boolean mask for channels in this ROI
        mask_vec = np.array([ch in roi_chs for ch in ch_names], dtype=bool)
        mask_params = ROI_MASK_PARAMS_DEFAULT

        n_rows = len(bands)
        n_cols = 4
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.0 * n_cols, 3.5 * n_rows),
            squeeze=False,
            gridspec_kw={"width_ratios": [1.0, 1.0, 0.25, 1.0], "wspace": 0.3},
        )
        for r, (band, (fmin, fmax)) in enumerate(bands.items()):
            fmax_eff = min(fmax, fmax_available)
            if fmin >= fmax_eff:
                for c in range(n_cols):
                    axes[r, c].axis('off')
                continue
            pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
            non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
            if pain_data is None or non_data is None:
                for c in range(n_cols):
                    axes[r, c].axis('off')
                continue
            diff_data = pain_data - non_data
            # Robust symmetric scaling for pain/non and diff within band
            vabs_pn = _robust_sym_vlim([pain_data, non_data])
            vmin, vmax = -vabs_pn, +vabs_pn
            diff_abs = _robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else 0.0
            # ROI-averaged values for annotation (use ROI mask when available)
            if mask_vec.any():
                pain_mu = float(np.nanmean(pain_data[mask_vec]))
                non_mu = float(np.nanmean(non_data[mask_vec]))
                diff_mu = float(np.nanmean(diff_data[mask_vec]))
            else:
                pain_mu = float(np.nanmean(pain_data))
                non_mu = float(np.nanmean(non_data))
                diff_mu = float(np.nanmean(diff_data))
            # Pain (col 0)
            ax = axes[r, 0]
            try:
                mne.viz.plot_topomap(
                    pain_data,
                    tfr_pain.info,
                    axes=ax,
                    show=False,
                    vlim=(vmin, vmax) if vmin is not None and vmax is not None else None,
                    mask=mask_vec,
                    mask_params=mask_params,
                    cmap=TOPO_CMAP,
                )
            except Exception:
                _plot_topomap_on_ax(
                    ax,
                    pain_data,
                    tfr_pain.info,
                    mask=mask_vec,
                    mask_params=mask_params,
                    vmin=vmin,
                    vmax=vmax,
                )
            # Annotate ROI-mean (percent change)
            try:
                pct_mu = (10**(pain_mu) - 1.0) * 100.0
                ax.text(0.5, 1.02, f"%Δ_ROI={pct_mu:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=8)
            except Exception:
                pass
            # Non-pain (col 1)
            ax = axes[r, 1]
            try:
                mne.viz.plot_topomap(
                    non_data,
                    tfr_pain.info,
                    axes=ax,
                    show=False,
                    vlim=(vmin, vmax) if vmin is not None and vmax is not None else None,
                    mask=mask_vec,
                    mask_params=mask_params,
                    cmap=TOPO_CMAP,
                )
            except Exception:
                _plot_topomap_on_ax(
                    ax,
                    non_data,
                    tfr_pain.info,
                    mask=mask_vec,
                    mask_params=mask_params,
                    vmin=vmin,
                    vmax=vmax,
                )
            # Annotate ROI-mean (percent change)
            try:
                pct_mu = (10**(non_mu) - 1.0) * 100.0
                ax.text(0.5, 1.02, f"%Δ_ROI={pct_mu:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=8)
            except Exception:
                pass
            # Spacer (col 2)
            axes[r, 2].axis('off')
            # Diff (col 3)
            ax = axes[r, 3]
            try:
                mne.viz.plot_topomap(
                    diff_data,
                    tfr_pain.info,
                    axes=ax,
                    show=False,
                    vlim=((-diff_abs, +diff_abs) if diff_abs > 0 else None),
                    mask=mask_vec,
                    mask_params=mask_params,
                    cmap=TOPO_CMAP,
                )
            except Exception:
                _plot_topomap_on_ax(
                    ax,
                    diff_data,
                    tfr_pain.info,
                    mask=mask_vec,
                    mask_params=mask_params,
                    vmin=(-diff_abs if diff_abs > 0 else None),
                    vmax=(+diff_abs if diff_abs > 0 else None),
                )
            # Annotate ROI-mean difference (percent change)
            try:
                pct_mu = (10**(diff_mu) - 1.0) * 100.0
                ax.text(0.5, 1.02, f"Δ%_ROI={pct_mu:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=8)
            except Exception:
                pass
            if r == 0:
                for c_title in (0, 1, 3):
                    axes[r, c_title].set_title(cond_labels[c_title], fontsize=9, pad=4, y=1.04)
            axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)
            # Add compact colorbars per row
            try:
                sm_pn = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs_pn, vmax=+vabs_pn), cmap=TOPO_CMAP)
                sm_pn.set_array([])
                cbar_pn = fig.colorbar(
                    sm_pn, ax=[axes[r, 0], axes[r, 1]], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD
                )
                try:
                    cbar_pn.set_label("log10(power/baseline)")
                except Exception:
                    pass
                if diff_abs > 0:
                    sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=TOPO_CMAP)
                    sm_diff.set_array([])
                    cbar_diff = fig.colorbar(
                        sm_diff, ax=axes[r, 3], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD
                    )
                    try:
                        cbar_diff.set_label("log10(power/baseline)")
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            fig.suptitle(f"ROI: {roi} — Topomaps (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)", fontsize=12)
            # Removed bottom 'Conditions' label per request; keep frequency bands label
            fig.supylabel("Frequency bands", fontsize=10)
        except Exception:
            pass
        _save_fig(
            fig,
            out_dir,
            f"topomap_ROI-{_sanitize(roi)}_grid_bands_pain_non_diff_baseline_logratio.png",
            formats=["png", "svg"],
        )

def _epochs_mean_roi(epochs: mne.Epochs, roi_name: str, roi_chs: list[str]) -> Optional[mne.Epochs]:
    """Create an Epochs object with a single virtual channel as the mean of ROI channels."""
    if len(roi_chs) == 0:
        return None
    picks = mne.pick_channels(epochs.ch_names, include=roi_chs, ordered=True)
    if len(picks) == 0:
        return None
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    roi_data = data[:, picks, :].mean(axis=1, keepdims=True)
    info = mne.create_info([roi_name], sfreq=epochs.info['sfreq'], ch_types='eeg')
    epo_roi = mne.EpochsArray(
        roi_data,
        info,
        events=epochs.events,
        event_id=epochs.event_id,
        tmin=epochs.tmin,
        metadata=epochs.metadata,
        verbose=False,
    )
    return epo_roi


def compute_roi_tfrs(
    epochs: mne.Epochs,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> Dict[str, mne.time_frequency.EpochsTFR]:
    """Compute per-ROI EpochsTFR using MNE's tfr_morlet on ROI-averaged epochs."""
    if roi_map is None:
        roi_map = _build_rois(epochs.info)
    roi_tfrs: Dict[str, mne.time_frequency.EpochsTFR] = {}
    for roi, chs in roi_map.items():
        epo_roi = _epochs_mean_roi(epochs, roi, chs)
        if epo_roi is None:
            continue
        power = mne.time_frequency.tfr_morlet(
            epo_roi,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            average=False,
            decim=TFR_DECIM,
            picks=TFR_PICKS,
            n_jobs=-1,
        )
        roi_tfrs[roi] = power
    return roi_tfrs


def plot_rois_all_trials(
    roi_tfrs: Dict[str, mne.time_frequency.EpochsTFR],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot baseline-corrected TFR per ROI (epochs averaged) and save."""
    for roi, tfr in roi_tfrs.items():
        tfr_c = tfr.copy()
        _apply_baseline_safe(tfr_c, baseline=baseline, mode="logratio", logger=logger)
        tfr_avg = tfr_c.average()
        ch = tfr_avg.info['ch_names'][0]
        try:
            fig = tfr_avg.plot(picks=ch, show=False)
            try:
                fig.suptitle(f"ROI: {roi} — all trials (baseline logratio)", fontsize=12)
            except Exception:
                pass
            _save_fig(fig, out_dir, f"tfr_ROI-{_sanitize(roi)}_all_trials_baseline_logratio.png", logger=logger)
        except Exception as e:
            msg = f"ROI {roi} TFR plot failed: {e}"
            if logger:
                logger.error(msg)
            else:
                print(msg)

        # Also save band-limited TFRs per ROI
        try:
            fmax_available = float(np.max(tfr_avg.freqs))
            bands = _get_consistent_bands(max_freq_available=fmax_available)
            for band, (fmin, fmax) in bands.items():
                fmax_eff = min(fmax, fmax_available)
                if fmin >= fmax_eff:
                    continue
                try:
                    fig_b = tfr_avg.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False)
                    try:
                        fig_b.suptitle(f"ROI: {roi} — {band} band (baseline logratio)", fontsize=12)
                    except Exception:
                        pass
                    _save_fig(fig_b, out_dir, f"tfr_ROI-{_sanitize(roi)}_{band}_all_trials_baseline_logratio.png", logger=logger)
                except Exception as e:
                    msg = f"ROI {roi} band {band} TFR plot failed: {e}"
                    if logger:
                        logger.error(msg)
                    else:
                        print(msg)
        except Exception as e:
            msg = f"ROI {roi} banded TFR export skipped: {e}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)


def plot_topomaps_bands_all_trials(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
) -> None:
    """Consolidated topomaps for all trials averaged, baseline-corrected by frequency bands.

    Creates a single topomap grid showing full-scalp topomaps for different frequency bands 
    over a specified plateau window.
    """
    tfr_all = tfr.copy()
    _apply_baseline_safe(tfr_all, baseline=baseline, mode="logratio")
    if isinstance(tfr_all, mne.time_frequency.EpochsTFR):
        tfr_avg = tfr_all.average()
    else:
        tfr_avg = tfr_all

    fmax_available = float(np.max(tfr_avg.freqs))
    bands = _get_consistent_bands(max_freq_available=fmax_available)
    times = np.asarray(tfr_avg.times)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times.min(), tmin_req))
    tmax = float(min(times.max(), tmax_req))

    # tfr_avg is already baseline-corrected from line 1998, so just use it directly
    tfr_corrected = tfr_avg
    
    # Create single consolidated plot
    n_rows = len(bands)
    fig, axes = plt.subplots(n_rows, 1, figsize=(4.0, 3.5 * n_rows), squeeze=False)
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            axes[r, 0].axis('off')
            continue
        data = _average_tfr_band(tfr_corrected, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        if data is None:
            axes[r, 0].axis('off')
            continue
        # Use robust symmetric scaling per row
        vabs = _robust_sym_vlim(data)
        vmin, vmax = -vabs, +vabs
        try:
            mne.viz.plot_topomap(
                data,
                tfr_avg.info,
                axes=axes[r, 0],
                show=False,
                vlim=(vmin, vmax),
                cmap=TOPO_CMAP,
            )
        except Exception:
            _plot_topomap_on_ax(axes[r, 0], data, tfr_avg.info, vmin=vmin, vmax=vmax)
        # Annotate scalp-mean percent change over the plotted window (EEG only)
        try:
            eeg_picks = mne.pick_types(tfr_avg.info, eeg=True, exclude=[])
            mu = float(np.nanmean(data[eeg_picks]))  # EEG channels only
            pct = (10.0 ** (mu) - 1.0) * 100.0
            axes[r, 0].text(
                0.5,
                1.02,
                f"%Δ={pct:+.0f}%",
                transform=axes[r, 0].transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )
        except Exception:
            pass
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)
        try:
            sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=TOPO_CMAP)
            sm.set_array([])
            cbar = fig.colorbar(
                sm, ax=axes[r, 0], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD
            )
            try:
                cbar.set_label("log10(power/baseline)")
            except Exception:
                pass
        except Exception:
            pass
    fig.suptitle(f"Topomaps (all trials; baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)", fontsize=12)
    try:
        fig.supylabel("Frequency bands", fontsize=10)
        fig.supxlabel("All trials", fontsize=10)
    except Exception:
        pass
    _save_fig(fig, out_dir, f"topomap_grid_bands_all_trials_baseline_logratio.png", formats=["png", "svg"])


def _align_avg_tfrs(
    tfr_list: List["mne.time_frequency.AverageTFR"],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[mne.Info], Optional[np.ndarray]]:
    """Align AverageTFR objects across subjects to common channels/freqs/times.

    Returns (info_common, data_array) where data_array has shape
    (n_subjects, n_channels, n_freqs, n_times). Subjects that do not match
    the first subject's freqs/times are skipped.
    """
    if not tfr_list:
        return None, None
    # Filter out Nones just in case
    tfr_list = [t for t in tfr_list if t is not None]
    if not tfr_list:
        return None, None
    base = tfr_list[0]
    base_times = np.asarray(base.times)
    base_freqs = np.asarray(base.freqs)
    base_chs = list(base.info["ch_names"])
    keep: List[Tuple[str, "mne.time_frequency.AverageTFR"]] = [("S0", base)]
    for i, tfr in enumerate(tfr_list[1:], start=1):
        ok = np.allclose(tfr.times, base_times) and np.allclose(tfr.freqs, base_freqs)
        if not ok:
            if logger:
                logger.warning(f"Skipping subject {i}: times/freqs mismatch for group alignment")
            continue
        keep.append((f"S{i}", tfr))
    if len(keep) == 0:
        return None, None
    # Common channels intersection
    ch_sets = [set(t.info["ch_names"]) for _, t in keep]
    common = list(sorted(set.intersection(*ch_sets))) if ch_sets else []
    if len(common) == 0:
        if logger:
            logger.warning("No common channels across subjects; cannot align")
        return None, None
    # Reorder each subject to common channel order
    arrs = []
    for tag, t in keep:
        idxs = [t.info["ch_names"].index(ch) for ch in common]
        arrs.append(np.asarray(t.data)[idxs, :, :])  # (n_ch_common, n_freqs, n_times)
    data = np.stack(arrs, axis=0)
    # Build aligned Info from base by picking common channels
    pick_inds = [base_chs.index(ch) for ch in common]
    info_common = mne.pick_info(base.info, pick_inds)
    return info_common, data


def _avg_alltrials_to_avg_tfr(power: "mne.time_frequency.EpochsTFR") -> "mne.time_frequency.AverageTFR":
    """Return a baseline-corrected AverageTFR for all trials from an EpochsTFR."""
    tfr_all = power.copy()
    _apply_baseline_safe(tfr_all, baseline=BASELINE, mode="logratio")
    return tfr_all.average()


def _avg_by_mask_to_avg_tfr(
    power: "mne.time_frequency.EpochsTFR",
    mask: np.ndarray,
) -> Optional["mne.time_frequency.AverageTFR"]:
    try:
        t = power.copy()[mask]
        _apply_baseline_safe(t, baseline=BASELINE, mode="logratio")
        return t.average()
    except Exception:
        return None


def group_topomaps_bands_all_trials(
    powers: List["mne.time_frequency.EpochsTFR"],
    out_dir: Path,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Group-level topomaps for all trials averaged (baseline logratio) across subjects."""
    if not powers:
        return
    avg_list = []
    for p in powers:
        try:
            avg_list.append(_avg_alltrials_to_avg_tfr(p))
        except Exception:
            continue
    info_common, data = _align_avg_tfrs(avg_list, logger=logger)
    if info_common is None or data is None:
        if logger:
            logger.warning("Group all-trials: no aligned data across subjects")
        return
    # data shape: (n_subj, n_ch, n_freqs, n_times)
    mean_data = data.mean(axis=0)  # (n_ch, n_freqs, n_times)
    freqs = np.asarray(avg_list[0].freqs)
    times = np.asarray(avg_list[0].times)
    fmax_available = float(freqs.max())
    bands = _get_consistent_bands(max_freq_available=fmax_available)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times.min(), tmin_req))
    tmax = float(min(times.max(), tmax_req))
    n_rows = len(bands)
    fig, axes = plt.subplots(n_rows, 1, figsize=(4.0, 3.5 * n_rows), squeeze=False)
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            axes[r, 0].axis('off')
            continue
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        tmask = (times >= tmin) & (times < tmax)
        if fmask.sum() == 0 or tmask.sum() == 0:
            axes[r, 0].axis('off')
            continue
        vec = mean_data[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        vabs = _robust_sym_vlim(vec)
        try:
            mne.viz.plot_topomap(
                vec,
                info_common,
                axes=axes[r, 0],
                show=False,
                vlim=(-vabs, +vabs),
                cmap=TOPO_CMAP,
            )
        except Exception:
            _plot_topomap_on_ax(axes[r, 0], vec, info_common, vmin=-vabs, vmax=+vabs)
        # Annotate scalp-mean over EEG channels
        try:
            eeg_picks = mne.pick_types(info_common, eeg=True, exclude=[])
            mu = float(np.nanmean(vec[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(vec))
            pct = (10.0 ** (mu) - 1.0) * 100.0
            axes[r, 0].text(0.5, 1.02, f"%Δ={pct:+.0f}%",
                            transform=axes[r, 0].transAxes, ha="center", va="top", fontsize=9)
        except Exception:
            pass
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)
        try:
            sm = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs, vmax=+vabs), cmap=TOPO_CMAP)
            sm.set_array([])
            fig.colorbar(sm, ax=axes[r, 0], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
        except Exception:
            pass
    try:
        fig.suptitle(f"Group topomaps (all trials; baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)", fontsize=12)
    except Exception:
        pass
    _save_fig(fig, out_dir, f"group_topomap_grid_bands_all_trials_baseline_logratio.png", formats=["png", "svg"], logger=logger)


def group_topomaps_pain_nonpain(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Group-level pain vs non-pain topomaps (baseline logratio) and their difference.

    Uses subjects that have valid pain/non-pain splits with at least one trial in each.
    """
    if not powers:
        return
    avg_pain: List["mne.time_frequency.AverageTFR"] = []
    avg_non: List["mne.time_frequency.AverageTFR"] = []
    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        # Find pain column
        pain_col = None
        for c in PAIN_BINARY_COLUMNS:
            if c in ev.columns:
                pain_col = c
                break
        if pain_col is None:
            continue
        vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
        pain_mask = np.asarray(vals == 1, dtype=bool)
        non_mask = np.asarray(vals == 0, dtype=bool)
        if pain_mask.sum() == 0 or non_mask.sum() == 0:
            continue
        a_p = _avg_by_mask_to_avg_tfr(power, pain_mask)
        a_n = _avg_by_mask_to_avg_tfr(power, non_mask)
        if a_p is not None and a_n is not None:
            avg_pain.append(a_p)
            avg_non.append(a_n)
    if len(avg_pain) == 0 or len(avg_non) == 0:
        if logger:
            logger.warning("Group pain/non-pain: insufficient aligned subjects")
        return
    # Align separately (but freqs/times should match for both sets)
    info_p, data_p = _align_avg_tfrs(avg_pain, logger=logger)
    info_n, data_n = _align_avg_tfrs(avg_non, logger=logger)
    if info_p is None or data_p is None or info_n is None or data_n is None:
        if logger:
            logger.warning("Group pain/non-pain: could not align data")
        return
    # Ensure same channels and axes; if not, intersect again
    # For simplicity assume same as first alignment; otherwise skip
    mean_p = data_p.mean(axis=0)
    mean_n = data_n.mean(axis=0)
    # Create per-band topomap grid similar to contrast_pain_nonpain_topomaps_rois (but no ROI)
    fmax_available = float(np.max(avg_pain[0].freqs))
    bands = _get_consistent_bands(max_freq_available=fmax_available)
    times = np.asarray(avg_pain[0].times)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times.min(), tmin_req))
    tmax = float(min(times.max(), tmax_req))
    n_rows = len(bands)
    n_cols = 3  # pain, non, diff
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.5 * n_rows), squeeze=False)
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis('off')
            continue
        # Average over band/time
        freqs = np.asarray(avg_pain[0].freqs)
        times = np.asarray(avg_pain[0].times)
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        tmask = (times >= tmin) & (times < tmax)
        if fmask.sum() == 0 or tmask.sum() == 0:
            for c in range(n_cols):
                axes[r, c].axis('off')
            continue
        v_p = mean_p[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_n = mean_n[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_d = v_p - v_n
        vabs_pn = _robust_sym_vlim([v_p, v_n])
        vabs_d = _robust_sym_vlim(v_d)
        # Pain
        try:
            mne.viz.plot_topomap(v_p, info_p, axes=axes[r, 0], show=False, vlim=(-vabs_pn, +vabs_pn), cmap=TOPO_CMAP)
        except Exception:
            _plot_topomap_on_ax(axes[r, 0], v_p, info_p, vmin=-vabs_pn, vmax=+vabs_pn)
        try:
            eeg_picks = mne.pick_types(info_p, eeg=True, exclude=[])
            mu_p = float(np.nanmean(v_p[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(v_p))
            pct_p = (10.0 ** (mu_p) - 1.0) * 100.0
            axes[r, 0].text(0.5, 1.02, f"%Δ={pct_p:+.0f}%", transform=axes[r, 0].transAxes, ha="center", va="top", fontsize=9)
        except Exception:
            pass
        # Non-pain
        try:
            mne.viz.plot_topomap(v_n, info_n, axes=axes[r, 1], show=False, vlim=(-vabs_pn, +vabs_pn), cmap=TOPO_CMAP)
        except Exception:
            _plot_topomap_on_ax(axes[r, 1], v_n, info_n, vmin=-vabs_pn, vmax=+vabs_pn)
        try:
            eeg_picks = mne.pick_types(info_n, eeg=True, exclude=[])
            mu_n = float(np.nanmean(v_n[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(v_n))
            pct_n = (10.0 ** (mu_n) - 1.0) * 100.0
            axes[r, 1].text(0.5, 1.02, f"%Δ={pct_n:+.0f}%", transform=axes[r, 1].transAxes, ha="center", va="top", fontsize=9)
        except Exception:
            pass
        # Diff
        try:
            mne.viz.plot_topomap(v_d, info_p, axes=axes[r, 2], show=False, vlim=(-vabs_d, +vabs_d), cmap=TOPO_CMAP)
        except Exception:
            _plot_topomap_on_ax(axes[r, 2], v_d, info_p, vmin=-vabs_d, vmax=+vabs_d)
        try:
            mu_d = float(np.nanmean(v_d))
            pct_d = (10.0 ** (mu_d) - 1.0) * 100.0
            axes[r, 2].text(0.5, 1.02, f"Δ%={pct_d:+.1f}%", transform=axes[r, 2].transAxes, ha="center", va="top", fontsize=9)
        except Exception:
            pass
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)")
    # Per-row colorbars
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        freqs = np.asarray(avg_pain[0].freqs)
        times = np.asarray(avg_pain[0].times)
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        tmask = (times >= tmin) & (times < tmax)
        if fmask.sum() == 0 or tmask.sum() == 0:
            continue
        v_p = mean_p[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_n = mean_n[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_d = v_p - v_n
        vabs_pn = _robust_sym_vlim([v_p, v_n])
        vabs_d = _robust_sym_vlim(v_d)
        try:
            sm_pn = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs_pn, vmax=+vabs_pn), cmap=TOPO_CMAP)
            sm_pn.set_array([])
            fig.colorbar(sm_pn, ax=[axes[r, 0], axes[r, 1]], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
            sm_d = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vabs_d, vcenter=0.0, vmax=vabs_d), cmap=TOPO_CMAP)
            sm_d.set_array([])
            fig.colorbar(sm_d, ax=axes[r, 2], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
        except Exception:
            pass
    try:
        fig.suptitle("Group topomaps: Pain vs Non-pain (baseline logratio)", fontsize=12)
    except Exception:
        pass
    _ensure_dir(out_dir)
    _save_fig(fig, out_dir, "group_topomap_grid_bands_pain_vs_nonpain_baseline_logratio.png", formats=["png", "svg"], logger=logger)


def group_pain_nonpain_temporal_topomaps(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
    window_count: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Group temporal topomaps: pain vs non across multiple windows and bands.

    - Builds per-subject AverageTFR for pain and non, aligns across subjects, averages to group.
    - Splits the plateau interval into `window_count` equal windows.
    - Rows: Pain, Non-pain, Difference; Columns: windows; separate figure per band.
    """
    if not powers:
        return
    avg_pain: List["mne.time_frequency.AverageTFR"] = []
    avg_non: List["mne.time_frequency.AverageTFR"] = []
    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        pain_col = None
        for c in PAIN_BINARY_COLUMNS:
            if c in ev.columns:
                pain_col = c
                break
        if pain_col is None:
            continue
        vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
        pain_mask = np.asarray(vals == 1, dtype=bool)
        non_mask = np.asarray(vals == 0, dtype=bool)
        if pain_mask.sum() == 0 or non_mask.sum() == 0:
            continue
        a_p = _avg_by_mask_to_avg_tfr(power, pain_mask)
        a_n = _avg_by_mask_to_avg_tfr(power, non_mask)
        if a_p is not None and a_n is not None:
            avg_pain.append(a_p)
            avg_non.append(a_n)
    if len(avg_pain) < 2 or len(avg_non) < 2:
        if logger:
            logger.warning("Group temporal topomaps: insufficient subjects with pain/non trials")
        return
    info_p, data_p = _align_avg_tfrs(avg_pain, logger=logger)
    info_n, data_n = _align_avg_tfrs(avg_non, logger=logger)
    if info_p is None or data_p is None or info_n is None or data_n is None:
        if logger:
            logger.warning("Group temporal topomaps: could not align pain/non data")
        return
    mean_p = data_p.mean(axis=0)  # (n_ch, n_freqs, n_times)
    mean_n = data_n.mean(axis=0)
    freqs = np.asarray(avg_pain[0].freqs)
    times = np.asarray(avg_pain[0].times)
    # Plateau windows
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times.min(), tmin_req))
    tmax = float(min(times.max(), tmax_req))
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        if logger:
            logger.warning("Group temporal topomaps: invalid plateau window after clipping")
        return
    edges = np.linspace(tmin, tmax, int(window_count) + 1)
    win_starts = edges[:-1]
    win_ends = edges[1:]

    fmax_available = float(freqs.max())
    bands = _get_consistent_bands(max_freq_available=fmax_available)
    for band_name, (fmin, fmax) in bands.items():
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue
        # Precompute masks for performance
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        # Collect vectors per window for scaling
        pain_vecs = []
        non_vecs = []
        diff_vecs = []
        for w_start, w_end in zip(win_starts, win_ends):
            tmask = (times >= w_start) & (times < w_end)
            if fmask.sum() == 0 or tmask.sum() == 0:
                pain_vecs.append(None); non_vecs.append(None); diff_vecs.append(None)
                continue
            v_p = mean_p[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            v_n = mean_n[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            pain_vecs.append(v_p)
            non_vecs.append(v_n)
            diff_vecs.append(v_p - v_n)
        vals = [v for v in (pain_vecs + non_vecs) if v is not None]
        if len(vals) == 0:
            continue
        vabs_cond = _robust_sym_vlim(vals)
        diff_vals = [v for v in diff_vecs if v is not None]
        vabs_diff = _robust_sym_vlim(diff_vals) if len(diff_vals) > 0 else vabs_cond

        n_rows = 3
        n_cols = len(win_starts)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 9.0), squeeze=False, gridspec_kw={"hspace": 0.25, "wspace": 0.3})
        for col, (w_start, w_end) in enumerate(zip(win_starts, win_ends)):
            # Title per column
            try:
                axes[0, col].set_title(f"{w_start:.1f}-{w_end:.1f}s", fontsize=10, pad=25)
            except Exception:
                pass
            v_p = pain_vecs[col]
            v_n = non_vecs[col]
            v_d = diff_vecs[col]
            # Pain row 0
            if v_p is not None:
                try:
                    mne.viz.plot_topomap(v_p, info_p, axes=axes[0, col], show=False, vlim=(-vabs_cond, +vabs_cond), cmap=TOPO_CMAP)
                except Exception:
                    _plot_topomap_on_ax(axes[0, col], v_p, info_p, vmin=-vabs_cond, vmax=+vabs_cond)
                try:
                    eeg_picks = mne.pick_types(info_p, eeg=True, exclude=[])
                    mu = float(np.nanmean(v_p[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(v_p))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    axes[0, col].text(0.5, 1.08, f"%Δ={pct:+.0f}%", transform=axes[0, col].transAxes, ha="center", va="bottom", fontsize=8)
                except Exception:
                    pass
            else:
                axes[0, col].axis('off')
            # Non row 1
            if v_n is not None:
                try:
                    mne.viz.plot_topomap(v_n, info_n, axes=axes[1, col], show=False, vlim=(-vabs_cond, +vabs_cond), cmap=TOPO_CMAP)
                except Exception:
                    _plot_topomap_on_ax(axes[1, col], v_n, info_n, vmin=-vabs_cond, vmax=+vabs_cond)
                try:
                    eeg_picks = mne.pick_types(info_n, eeg=True, exclude=[])
                    mu = float(np.nanmean(v_n[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(v_n))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    axes[1, col].text(0.5, 1.08, f"%Δ={pct:+.0f}%", transform=axes[1, col].transAxes, ha="center", va="bottom", fontsize=8)
                except Exception:
                    pass
            else:
                axes[1, col].axis('off')
            # Diff row 2
            if v_d is not None:
                try:
                    mne.viz.plot_topomap(v_d, info_p, axes=axes[2, col], show=False, vlim=(-vabs_diff, +vabs_diff), cmap=TOPO_CMAP)
                except Exception:
                    _plot_topomap_on_ax(axes[2, col], v_d, info_p, vmin=-vabs_diff, vmax=+vabs_diff)
                try:
                    mu = float(np.nanmean(v_d))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    axes[2, col].text(0.5, 1.08, f"Δ%={pct:+.1f}%", transform=axes[2, col].transAxes, ha="center", va="bottom", fontsize=8)
                except Exception:
                    pass
            else:
                axes[2, col].axis('off')
        # Labels
        try:
            axes[0, 0].set_ylabel("Pain", fontsize=10)
            axes[1, 0].set_ylabel("Non-pain", fontsize=10)
            axes[2, 0].set_ylabel("Pain - Non", fontsize=10)
        except Exception:
            pass
        # Row colorbars
        try:
            sm_cond = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs_cond, vmax=+vabs_cond), cmap=TOPO_CMAP)
            sm_cond.set_array([])
            fig.colorbar(sm_cond, ax=axes[0:2, :].ravel().tolist(), fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
            sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vabs_diff, vcenter=0.0, vmax=vabs_diff), cmap=TOPO_CMAP)
            sm_diff.set_array([])
            fig.colorbar(sm_diff, ax=axes[2, :].ravel().tolist(), fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
        except Exception:
            pass
        # Title and save
        try:
            fig.suptitle(
                f"Group Temporal Topomaps: Pain vs Non-pain — {band_name} (t=[{tmin:.1f},{tmax:.1f}]s; {len(win_starts)} windows)",
                fontsize=12,
            )
        except Exception:
            pass
        band_suffix = band_name.lower()
        fname = f"group_temporal_topomaps_pain_vs_nonpain_{band_suffix}_plateau_{tmin:.0f}-{tmax:.0f}s_{len(win_starts)}windows.png"
        _save_fig(fig, out_dir, fname, formats=["png", "svg"], logger=logger)


def _collect_group_temperatures(events_by_subj: List[Optional[pd.DataFrame]]) -> list[float]:
    temps: set[float] = set()
    for ev in events_by_subj:
        if ev is None:
            continue
        # Find temperature column
        tcol = None
        for c in TEMPERATURE_COLUMNS:
            if c in ev.columns:
                tcol = c
                break
        if tcol is None:
            continue
        try:
            vals = pd.to_numeric(ev[tcol], errors="coerce").round(1).dropna().unique()
            for v in vals:
                try:
                    temps.add(float(v))
                except Exception:
                    continue
        except Exception:
            continue
    return sorted(temps)


def group_topomap_grid_baseline_temps(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Group Δ=log10(power/baseline) topomap grid: All trials + per-temperature across bands."""
    if not powers:
        return
    # All trials per-subject AverageTFR
    avg_all: List["mne.time_frequency.AverageTFR"] = []
    for p in powers:
        try:
            avg_all.append(_avg_alltrials_to_avg_tfr(p))
        except Exception:
            continue
    info_all, data_all = _align_avg_tfrs(avg_all, logger=logger)
    if info_all is None or data_all is None:
        if logger:
            logger.warning("Group temperature grid: could not align all-trials TFRs")
        return
    mean_all = data_all.mean(axis=0)  # (n_ch, n_freqs, n_times)

    # Determine temperature levels across group
    temps = _collect_group_temperatures(events_by_subj)
    # Build per-temp group mean Averages
    cond_map: Dict[str, Tuple[mne.Info, np.ndarray, int, float]] = {}
    # Add All trials first (label, info, data, N, nan tval)
    cond_map["All trials"] = (info_all, mean_all, data_all.shape[0], float("nan"))
    for tval in temps:
        avg_list: List["mne.time_frequency.AverageTFR"] = []
        for p, ev in zip(powers, events_by_subj):
            if ev is None:
                continue
            tcol = None
            for c in TEMPERATURE_COLUMNS:
                if c in ev.columns:
                    tcol = c
                    break
            if tcol is None:
                continue
            try:
                mask = pd.to_numeric(ev[tcol], errors="coerce").round(1) == round(float(tval), 1)
            except Exception:
                continue
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            a = _avg_by_mask_to_avg_tfr(p, mask)
            if a is not None:
                avg_list.append(a)
        if not avg_list:
            continue
        info_t, data_t = _align_avg_tfrs(avg_list, logger=logger)
        if info_t is None or data_t is None:
            continue
        mean_t = data_t.mean(axis=0)
        cond_map[f"{tval:.1f}°C"] = (info_t, mean_t, data_t.shape[0], float(tval))

    if len(cond_map) <= 1:
        if logger:
            logger.warning("Group temperature grid: no temperature-specific data available")
        return

    # Plot: rows=bands, cols=conditions (All + per-temperature)
    # Build ordered columns
    labels = list(cond_map.keys())
    # Use first entry to get freqs/times (should be consistent post-alignment)
    any_info, any_data, _, _ = next(iter(cond_map.values()))
    freqs = np.asarray(avg_all[0].freqs)
    times = np.asarray(avg_all[0].times)
    fmax_available = float(freqs.max())
    bands = _get_consistent_bands(max_freq_available=fmax_available)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times.min(), tmin_req))
    tmax = float(min(times.max(), tmax_req))

    n_rows = len(bands)
    n_cols = len(labels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 3.6 * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 0.30, "hspace": 0.55},
    )

    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis('off')
            continue
        # Gather vectors for vlim across columns
        col_vecs: List[np.ndarray] = []
        for label in labels:
            info_c, data_c, _, _ = cond_map[label]
            freqs_c = freqs  # aligned selections guarantee same freqs
            times_c = times
            fmask = (freqs_c >= fmin) & (freqs_c <= fmax_eff)
            tmask = (times_c >= tmin) & (times_c < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0:
                col_vecs.append(None)
                continue
            vec = data_c[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            col_vecs.append(vec)
        vals = [v for v in col_vecs if v is not None]
        if len(vals) == 0:
            for c in range(n_cols):
                axes[r, c].axis('off')
            continue
        vabs = _robust_sym_vlim(vals)
        for c, label in enumerate(labels):
            ax = axes[r, c]
            info_c, data_c, nsub, tval = cond_map[label]
            fmask = (freqs >= fmin) & (freqs <= fmax_eff)
            tmask = (times >= tmin) & (times < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0:
                ax.axis('off')
                continue
            vec = data_c[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            try:
                mne.viz.plot_topomap(
                    vec,
                    info_c,
                    axes=ax,
                    show=False,
                    vlim=(-vabs, +vabs),
                    cmap=TOPO_CMAP,
                )
            except Exception:
                _plot_topomap_on_ax(ax, vec, info_c, vmin=-vabs, vmax=+vabs)
            # Title with N subjects for this condition
            try:
                title = f"{label} (n={nsub})"
                ax.set_title(title, fontsize=9, pad=4)
            except Exception:
                pass
            # Annotate scalp-mean (percent change)
            try:
                eeg_picks = mne.pick_types(info_c, eeg=True, exclude=[])
                mu = float(np.nanmean(vec[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(vec))
                pct = (10.0 ** (mu) - 1.0) * 100.0
                ax.text(0.5, 1.02, f"%Δ={pct:+.0f}%", transform=ax.transAxes, ha="center", va="top", fontsize=8)
            except Exception:
                pass
            
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)")
        # Row colorbar
        try:
            sm = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs, vmax=+vabs), cmap=TOPO_CMAP)
            sm.set_array([])
            fig.colorbar(sm, ax=axes[r, :].ravel().tolist(), fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
        except Exception:
            pass
    try:
        fig.suptitle(
            f"Group Δ=log10(power/baseline) by temperature (t=[{tmin:.1f}, {tmax:.1f}] s)",
            fontsize=12,
        )
    except Exception:
        pass
    _save_fig(
        fig,
        out_dir,
        "group_topomap_grid_bands_by_temperature_baseline_logratio.png",
        formats=["png", "svg"],
        logger=logger,
    )


def group_contrast_maxmin_temperature(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Group topomap grid comparing highest vs lowest temperature (baseline logratio).

    - Columns: [Max temp, Min temp, spacer, Max - Min]
    - Rows: frequency bands from configuration
    - Uses global min/max temperature across subjects (rounded to 0.1°C); subjects
      without that level are excluded for that condition.
    """
    if not powers:
        return
    temps = _collect_group_temperatures(events_by_subj)
    if len(temps) < 2:
        if logger:
            logger.info("Group max/min: fewer than 2 temperature levels; skipping")
        return
    t_min = float(min(temps)); t_max = float(max(temps))

    # Build per-subject AverageTFR for min and max temps
    avg_min: List["mne.time_frequency.AverageTFR"] = []
    avg_max: List["mne.time_frequency.AverageTFR"] = []
    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        tcol = None
        for c in TEMPERATURE_COLUMNS:
            if c in ev.columns:
                tcol = c; break
        if tcol is None:
            continue
        try:
            vals = pd.to_numeric(ev[tcol], errors="coerce").round(1)
        except Exception:
            continue
        mask_min = np.asarray(vals == round(t_min, 1), dtype=bool)
        mask_max = np.asarray(vals == round(t_max, 1), dtype=bool)
        if mask_min.sum() > 0:
            a_min = _avg_by_mask_to_avg_tfr(power, mask_min)
            if a_min is not None:
                avg_min.append(a_min)
        if mask_max.sum() > 0:
            a_max = _avg_by_mask_to_avg_tfr(power, mask_max)
            if a_max is not None:
                avg_max.append(a_max)

    info_min, data_min = _align_avg_tfrs(avg_min, logger=logger)
    info_max, data_max = _align_avg_tfrs(avg_max, logger=logger)
    if info_min is None or data_min is None or info_max is None or data_max is None:
        if logger:
            logger.info("Group max/min: could not align min/max TFRs; skipping")
        return

    mean_min = data_min.mean(axis=0)  # (n_ch, n_freqs, n_times)
    mean_max = data_max.mean(axis=0)
    freqs = np.asarray(avg_min[0].freqs if avg_min else avg_max[0].freqs)
    times = np.asarray(avg_min[0].times if avg_min else avg_max[0].times)
    fmax_available = float(freqs.max())
    bands = _get_consistent_bands(max_freq_available=fmax_available)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times.min(), tmin_req)); tmax = float(min(times.max(), tmax_req))

    n_rows = len(bands)
    n_cols = 4  # Max, Min, spacer, Diff
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.25, 1.0], "wspace": 0.3},
    )
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis('off')
            continue
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        tmask = (times >= tmin) & (times < tmax)
        if fmask.sum() == 0 or tmask.sum() == 0:
            for c in range(n_cols):
                axes[r, c].axis('off')
            continue
        v_max = mean_max[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_min = mean_min[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_diff = v_max - v_min
        vabs_pn = _robust_sym_vlim([v_max, v_min])
        vabs_diff = _robust_sym_vlim(v_diff)
        # Max
        ax = axes[r, 0]
        try:
            mne.viz.plot_topomap(v_max, info_max, axes=ax, show=False, vlim=(-vabs_pn, +vabs_pn), cmap=TOPO_CMAP)
        except Exception:
            _plot_topomap_on_ax(ax, v_max, info_max, vmin=-vabs_pn, vmax=+vabs_pn)
        try:
            mu_max = float(np.nanmean(v_max))
            pct_max = (10.0 ** (mu_max) - 1.0) * 100.0
            ax.text(0.5, 1.02, f"%Δ={pct_max:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        except Exception:
            pass
        # Min
        ax = axes[r, 1]
        try:
            mne.viz.plot_topomap(v_min, info_min, axes=ax, show=False, vlim=(-vabs_pn, +vabs_pn), cmap=TOPO_CMAP)
        except Exception:
            _plot_topomap_on_ax(ax, v_min, info_min, vmin=-vabs_pn, vmax=+vabs_pn)
        try:
            mu_min = float(np.nanmean(v_min))
            pct_min = (10.0 ** (mu_min) - 1.0) * 100.0
            ax.text(0.5, 1.02, f"%Δ={pct_min:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        except Exception:
            pass
        # Spacer
        axes[r, 2].axis('off')
        # Diff
        ax = axes[r, 3]
        try:
            mne.viz.plot_topomap(v_diff, info_max, axes=ax, show=False, vlim=(-vabs_diff, +vabs_diff), cmap=TOPO_CMAP)
        except Exception:
            _plot_topomap_on_ax(ax, v_diff, info_max, vmin=-vabs_diff, vmax=+vabs_diff)
        try:
            mu_d = float(np.nanmean(v_diff))
            pct_d = (10.0 ** (mu_d) - 1.0) * 100.0
            ax.text(0.5, 1.02, f"Δ%={pct_d:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        except Exception:
            pass
        if r == 0:
            axes[r, 0].set_title(f"Max {t_max:.1f}°C (n={data_max.shape[0]})", fontsize=9, pad=4, y=1.04)
            axes[r, 1].set_title(f"Min {t_min:.1f}°C (n={data_min.shape[0]})", fontsize=9, pad=4, y=1.04)
            axes[r, 3].set_title("Max - Min", fontsize=9, pad=4, y=1.04)
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)
        # Per-row colorbars
        try:
            sm_pn = ScalarMappable(norm=mcolors.Normalize(vmin=-vabs_pn, vmax=+vabs_pn), cmap=TOPO_CMAP)
            sm_pn.set_array([])
            fig.colorbar(sm_pn, ax=[axes[r, 0], axes[r, 1]], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
            sm_d = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vabs_diff, vcenter=0.0, vmax=vabs_diff), cmap=TOPO_CMAP)
            sm_d.set_array([])
            fig.colorbar(sm_d, ax=axes[r, 3], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
        except Exception:
            pass
    try:
        fig.suptitle(
            f"Group Topomaps: Max vs Min temperature (baseline logratio; t=[{tmin:.1f}, {tmax:.1f}] s)",
            fontsize=12,
        )
    except Exception:
        pass
    _save_fig(fig, out_dir, "group_topomap_grid_bands_maxmin_temp_diff_baseline_logratio.png", formats=["png", "svg"], logger=logger)


def _avg_tfr_to_roi_average(
    tfr_avg: "mne.time_frequency.AverageTFR", 
    roi: str,
    roi_map_override: Optional[Dict[str, list[str]]] = None,
) -> Optional["mne.time_frequency.AverageTFR"]:
    """Collapse an AverageTFR to a single-channel ROI AverageTFR by averaging ROI channels.

    Returns None if no channels found for the ROI.
    """
    # Determine ROI channels (channel names, not regex patterns)
    # If an override is provided, it must contain concrete channel names.
    # Otherwise, compute channels by applying regex patterns from config to this subject's info.
    if roi_map_override is not None:
        roi_map = roi_map_override
        chs_override = roi_map.get(roi)
        if chs_override:
            chs_all = list(chs_override)
        else:
            # Fallback to pattern-based detection when override lacks this ROI
            pats = (_roi_definitions().get(roi) or [])
            chs_all = _find_roi_channels(tfr_avg.info, pats)
    else:
        pats = (_roi_definitions().get(roi) or [])
        chs_all = _find_roi_channels(tfr_avg.info, pats)
    # Intersect with available channels (robust to case/ref variants) and preserve subject order
    subj_chs = tfr_avg.info['ch_names']
    canon_subj = { _canonicalize_ch_name(ch).upper(): ch for ch in subj_chs }
    want = { _canonicalize_ch_name(ch).upper() for ch in chs_all }
    chs = [canon_subj[_canonicalize_ch_name(ch).upper()] for ch in subj_chs if _canonicalize_ch_name(ch).upper() in want]
    if len(chs) == 0:
        # Fallback 1: heuristic prefix-based ROI map built from this subject's channels
        try:
            subj_roi_map = _build_group_roi_map_from_channels(list(subj_chs))
            chs_subj_heur = list(subj_roi_map.get(roi, []))
        except Exception:
            chs_subj_heur = []
        if chs_subj_heur:
            # Preserve subject order
            chs = [ch for ch in subj_chs if ch in set(chs_subj_heur)]
    if len(chs) == 0:
        # Fallback 2: regex-based discovery against this subject's info (config ROIs)
        chs_rx = _find_roi_channels(tfr_avg.info, (_roi_definitions().get(roi) or []))
        if chs_rx:
            chs = [ch for ch in subj_chs if ch in set(chs_rx)]
    if len(chs) == 0:
        return None
    try:
        # Use MNE's robust channel picker (case-insensitive, handles ordering)
        picks = mne.pick_channels(subj_chs, include=chs, exclude=[])
    except Exception:
        # Fallback to exact index lookup
        picks = [subj_chs.index(ch) for ch in chs]
    if len(picks) == 0:
        return None
    data = np.asarray(tfr_avg.data)[picks, :, :].mean(axis=0, keepdims=True)  # (1, n_freqs, n_times)
    # Avoid constructor API issues by cloning and mutating the existing AverageTFR
    try:
        roi_tfr = tfr_avg.copy()
        roi_tfr.data = data
        roi_tfr.info = mne.create_info([f"ROI:{roi}"], sfreq=tfr_avg.info['sfreq'], ch_types='eeg')
        roi_tfr.nave = int(getattr(tfr_avg, 'nave', 1))
        roi_tfr.comment = f"ROI:{roi}"
        return roi_tfr
    except Exception:
        return None


def group_rois_all_trials(
    powers: List["mne.time_frequency.EpochsTFR"],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    logger: Optional[logging.Logger] = None,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> None:
    """Group-level ROI TFRs (all trials, baseline logratio)."""
    if not powers:
        return
    # Per-subject AverageTFR (all trials)
    avg_list = []
    for p in powers:
        try:
            t = p.copy()
            _apply_baseline_safe(t, baseline=baseline, mode="logratio", logger=logger)
            avg_list.append(t.average())
        except Exception:
            continue
    if not avg_list:
        return
    # For each ROI, collect per-subject ROI AverageTFR and aggregate
    rois = list((roi_map or _roi_definitions()).keys())
    for roi in rois:
        per_subj: List["mne.time_frequency.AverageTFR"] = []
        for a in avg_list:
            ra = _avg_tfr_to_roi_average(a, roi, roi_map_override=roi_map)
            if ra is not None:
                per_subj.append(ra)
        # Retry without override if override produced no contributions
        if len(per_subj) < 1 and roi_map is not None:
            for a in avg_list:
                ra = _avg_tfr_to_roi_average(a, roi, roi_map_override=None)
                if ra is not None:
                    per_subj.append(ra)
        if len(per_subj) < 1:
            if logger:
                logger.info(f"Group ROI all-trials: no subjects contributed to ROI '{roi}'")
            continue
        info_c, data_c = _align_avg_tfrs(per_subj, logger=logger)
        if info_c is None or data_c is None:
            continue
        mean_roi = data_c.mean(axis=0)  # (n_ch=1, n_freqs, n_times)
        # Build AverageTFR for plotting using clone to avoid constructor API differences
        try:
            grp = per_subj[0].copy()
            grp.data = mean_roi
            grp.info = info_c  # aligned single-channel info
            grp.nave = int(data_c.shape[0])
            grp.comment = f"Group ROI:{roi}"
            ch = grp.info['ch_names'][0]
            fig = grp.plot(picks=ch, show=False)
            try:
                fig.suptitle(f"Group ROI: {roi} — all trials (baseline logratio, n={data_c.shape[0]})", fontsize=12)
            except Exception:
                pass
            _save_fig(fig, out_dir, f"group_tfr_ROI-{_sanitize(roi)}_all_trials_baseline_logratio.png", logger=logger)
        except Exception as e:
            if logger:
                logger.warning(f"Group ROI all-trials plot failed for {roi}: {e}")


def group_contrast_pain_nonpain_rois(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    logger: Optional[logging.Logger] = None,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> None:
    """Group-level ROI TFRs for pain vs non-pain and their difference."""
    if not powers:
        return
    rois = list((roi_map or _roi_definitions()).keys())
    # Build per-ROI lists of per-subject (pain, non) AverageTFR collapsed to ROI
    for roi in rois:
        roi_p_list: List["mne.time_frequency.AverageTFR"] = []
        roi_n_list: List["mne.time_frequency.AverageTFR"] = []
        for power, ev in zip(powers, events_by_subj):
            if ev is None:
                continue
            pain_col = None
            for c in PAIN_BINARY_COLUMNS:
                if c in ev.columns:
                    pain_col = c
                    break
            if pain_col is None:
                continue
            vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
            pain_mask = np.asarray(vals == 1, dtype=bool)
            non_mask = np.asarray(vals == 0, dtype=bool)
            if pain_mask.sum() == 0 or non_mask.sum() == 0:
                continue
            a_p = _avg_by_mask_to_avg_tfr(power, pain_mask)
            a_n = _avg_by_mask_to_avg_tfr(power, non_mask)
            if a_p is None or a_n is None:
                continue
            r_p = _avg_tfr_to_roi_average(a_p, roi, roi_map_override=roi_map)
            r_n = _avg_tfr_to_roi_average(a_n, roi, roi_map_override=roi_map)
            if r_p is not None and r_n is not None:
                roi_p_list.append(r_p)
                roi_n_list.append(r_n)
        # Retry without override if no contributions under override
        if (len(roi_p_list) < 1 or len(roi_n_list) < 1) and roi_map is not None:
            roi_p_list = []
            roi_n_list = []
            for power, ev in zip(powers, events_by_subj):
                if ev is None:
                    continue
                pain_col = None
                for c in PAIN_BINARY_COLUMNS:
                    if c in ev.columns:
                        pain_col = c
                        break
                if pain_col is None:
                    continue
                vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
                pain_mask = np.asarray(vals == 1, dtype=bool)
                non_mask = np.asarray(vals == 0, dtype=bool)
                if pain_mask.sum() == 0 or non_mask.sum() == 0:
                    continue
                a_p = _avg_by_mask_to_avg_tfr(power, pain_mask)
                a_n = _avg_by_mask_to_avg_tfr(power, non_mask)
                if a_p is None or a_n is None:
                    continue
                r_p = _avg_tfr_to_roi_average(a_p, roi, roi_map_override=None)
                r_n = _avg_tfr_to_roi_average(a_n, roi, roi_map_override=None)
                if r_p is not None and r_n is not None:
                    roi_p_list.append(r_p)
                    roi_n_list.append(r_n)
        if len(roi_p_list) < 1 or len(roi_n_list) < 1:
            if logger:
                logger.info(f"Group ROI pain/non: no subjects contributed to ROI '{roi}'")
            continue
        info_p, data_p = _align_avg_tfrs(roi_p_list, logger=logger)
        info_n, data_n = _align_avg_tfrs(roi_n_list, logger=logger)
        if info_p is None or info_n is None or data_p is None or data_n is None:
            continue
        mean_p = data_p.mean(axis=0)  # (1, n_freqs, n_times)
        mean_n = data_n.mean(axis=0)
        # Build group AverageTFRs using clones
        try:
            grp_p = roi_p_list[0].copy()
            grp_p.data = mean_p
            grp_p.info = info_p
            grp_p.nave = int(data_p.shape[0])
            grp_p.comment = f"Group ROI:{roi} Pain"

            grp_n = roi_n_list[0].copy()
            grp_n.data = mean_n
            grp_n.info = info_n
            grp_n.nave = int(data_n.shape[0])
            grp_n.comment = f"Group ROI:{roi} Non"

            # Difference array
            diff = mean_p - mean_n
            grp_d = roi_p_list[0].copy()
            grp_d.data = diff
            grp_d.info = info_p
            grp_d.nave = int(min(data_p.shape[0], data_n.shape[0]))
            grp_d.comment = f"Group ROI:{roi} Diff"
            ch = grp_p.info['ch_names'][0]
            # Pain
            fig = grp_p.plot(picks=ch, show=False)
            try:
                fig.suptitle(f"Group ROI: {roi} — Pain (baseline logratio, n={data_p.shape[0]})", fontsize=12)
            except Exception:
                pass
            _save_fig(fig, out_dir, f"group_tfr_ROI-{_sanitize(roi)}_pain_baseline_logratio.png", logger=logger)
            # Non-pain
            fig = grp_n.plot(picks=ch, show=False)
            try:
                fig.suptitle(f"Group ROI: {roi} — Non-pain (baseline logratio, n={data_n.shape[0]})", fontsize=12)
            except Exception:
                pass
            _save_fig(fig, out_dir, f"group_tfr_ROI-{_sanitize(roi)}_nonpain_baseline_logratio.png", logger=logger)
            # Diff
            fig = grp_d.plot(picks=ch, show=False)
            try:
                fig.suptitle(f"Group ROI: {roi} — Pain minus Non (baseline logratio)", fontsize=12)
            except Exception:
                pass
            _save_fig(fig, out_dir, f"group_tfr_ROI-{_sanitize(roi)}_pain_minus_non_baseline_logratio.png", logger=logger)
        except Exception as e:
            if logger:
                logger.warning(f"Group ROI pain/non plot failed for {roi}: {e}")


def plot_topomap_grid_baseline_temps(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
) -> None:
    """Topomap grid of Δ=log10(power/baseline) by temperature (no baseline column).

    - Columns: All trials (Δ) followed by each temperature level (Δ), per frequency band.
    - Layout made more compact horizontally by reducing inter-column spacing.
    Saves PNG and SVG.
    """
    if events_df is None:
        print("Temperature grid: events_df is None; skipping.")
        return
    temp_col = _find_temperature_column(events_df)
    if temp_col is None:
        print("Temperature grid: no temperature column found; skipping.")
        return
    # Prepare baseline-corrected TFR for Δ columns only
    tfr_corr = tfr.copy()
    _apply_baseline_safe(tfr_corr, baseline=baseline, mode="logratio")
    tfr_avg_all_corr = tfr_corr.average() if isinstance(tfr_corr, mne.time_frequency.EpochsTFR) else tfr_corr

    # Determine unique temperatures (rounded to 1 decimal)
    try:
        temps = (
            pd.to_numeric(events_df[temp_col], errors="coerce")
            .round(1)
            .dropna()
            .unique()
        )
        temps = sorted(map(float, temps))
    except Exception:
        temps = sorted(events_df[temp_col].dropna().unique())
    if len(temps) == 0:
        print("Temperature grid: no temperature levels; skipping.")
        return

    # Clip plateau window to available times
    times_corr = np.asarray(tfr_avg_all_corr.times)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times_corr.min(), tmin_req))
    tmax = float(min(times_corr.max(), tmax_req))

    # Bands with consistent gamma limits across subjects
    fmax_available = float(np.max(tfr_avg_all_corr.freqs))
    bands = _get_consistent_bands(max_freq_available=fmax_available)

    # Build Δ condition averages from baseline-corrected TFR: All trials + temps
    cond_tfrs: list[tuple[str, "mne.time_frequency.AverageTFR", int, float]] = []
    n_all = len(tfr_corr) if isinstance(tfr_corr, mne.time_frequency.EpochsTFR) else 1
    cond_tfrs.append(("All trials", tfr_avg_all_corr, n_all, np.nan))

    if isinstance(tfr_corr, mne.time_frequency.EpochsTFR):
        for tval in temps:
            try:
                mask = pd.to_numeric(events_df[temp_col], errors="coerce").round(1) == round(float(tval), 1)
            except Exception:
                mask = events_df[temp_col] == tval
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            try:
                tfr_temp = tfr_corr.copy()[mask].average()
            except Exception as e:
                print(f"Temperature grid: averaging failed for temp={tval}: {e}")
                continue
            cond_tfrs.append((f"{tval:.1f}°C", tfr_temp, int(mask.sum()), float(tval)))
    else:
        print("Temperature grid: input is AverageTFR; cannot split by temperature; showing only All trials.")

    # Columns: All-trials + each temperature (Δ)
    n_cols = len(cond_tfrs)
    n_rows = len(bands)
    width_ratios = [1.0] * n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 3.8 * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 0.30, "hspace": 0.55, "width_ratios": width_ratios},
    )

    # Plot per-row: Δ columns (logratio) with symmetric scale
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        # Δ (logratio) data for all columns
        diff_datas: list[Optional[np.ndarray]] = []
        for _, tfr_cond, _, _ in cond_tfrs:
            d = _average_tfr_band(tfr_cond, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
            diff_datas.append(d)
        # Robust symmetric scaling around zero across all Δ columns
        vals = [v for v in diff_datas if v is not None and np.isfinite(v).any()]
        if len(vals) == 0:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        diff_abs = _robust_sym_vlim(vals)
        if not np.isfinite(diff_abs) or diff_abs == 0:
            diff_abs = 1e-6
        # Plot Δ columns
        for idx, (label, tfr_cond, n_cond, _tval) in enumerate(cond_tfrs, start=0):
            ax = axes[r, idx]
            data = diff_datas[idx]
            if data is None:
                ax.axis("off")
                continue
            try:
                mne.viz.plot_topomap(
                    data,
                    tfr_cond.info,
                    axes=ax,
                    show=False,
                    vlim=(-diff_abs, +diff_abs),
                    cmap=TOPO_CMAP,
                )
            except Exception:
                _plot_topomap_on_ax(ax, data, tfr_cond.info, vmin=-diff_abs, vmax=+diff_abs)
            mu = float(np.nanmean(data))
            pct = (10.0 ** (mu) - 1.0) * 100.0
            ax.text(0.5, 1.02, f"%Δ={pct:+.0f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)
            if r == 0:
                ax.set_title(f"{label} (n={n_cond})", fontsize=9, pad=4, y=1.04)
        # Label frequency band on the leftmost axis in the row
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)
        # Δ colorbar for all Δ columns in this row
        try:
            sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=TOPO_CMAP)
            sm_diff.set_array([])
            cbar_d = fig.colorbar(
                sm_diff, ax=axes[r, :].ravel().tolist(), fraction=0.045, pad=0.06, shrink=0.9
            )
            try:
                cbar_d.set_label("log10(power/baseline)")
            except Exception:
                pass
        except Exception:
            pass

    try:
        fig.suptitle(
            f"Topomaps by temperature: \u0394=log10(power/baseline) over plateau t=[{tmin:.1f}, {tmax:.1f}] s",
            fontsize=12,
        )
    except Exception:
        pass
    _save_fig(fig, out_dir, "topomap_grid_bands_alltrials_plus_temperatures_baseline_logratio.png", formats=["png", "svg"])

def plot_pain_nonpain_temporal_topomaps(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    window_size: float = 2.0,
    freq_band: Optional[Tuple[float, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Create temporal topomap grid: pain vs non-pain across 2-second windows.
    
    Creates a figure with 3 rows:
    - Row 1: Pain condition topomaps for each 2-second window
    - Row 2: Non-pain condition topomaps for each 2-second window  
    - Row 3: Difference (pain - non-pain) topomaps for each window
    
    Args:
        tfr: EpochsTFR object containing trial-level data
        events_df: Events dataframe with pain/non-pain labels
        out_dir: Output directory for saving figures
        baseline: Baseline correction window
        window_size: Size of temporal windows in seconds (default 2.0)
        freq_band: Optional frequency band (fmin, fmax). If None, averages all frequencies
        logger: Optional logger for messages
    """
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        msg = "Temporal topomaps require EpochsTFR (trial-level data). Skipping."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return
            
    pain_col = _find_pain_binary_column(events_df)
    if pain_col is None:
        msg = f"Events with pain binary column {PAIN_BINARY_COLUMNS} required for temporal topomaps; skipping."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return

    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df)
    n = min(n_epochs, n_meta)
    
    # Get pain/non-pain masks
    if getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns:
        pain_vec = pd.to_numeric(tfr.metadata.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
    else:
        pain_vec = pd.to_numeric(events_df.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
    
    pain_mask = np.asarray(pain_vec == 1, dtype=bool)
    non_mask = np.asarray(pain_vec == 0, dtype=bool)
    
    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        msg = "One of the groups has zero trials; skipping temporal topomaps."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return

    msg = f"Temporal topomaps: pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())} trials."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # Subset and baseline-correct
    tfr_sub = tfr.copy()[:n]
    if len(pain_mask) != len(tfr_sub):
        n2 = min(len(tfr_sub), len(pain_mask))
        tfr_sub = tfr_sub[:n2]
        pain_mask = pain_mask[:n2]
        non_mask = non_mask[:n2]
    
    _apply_baseline_safe(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
    
    # Split into pain/non-pain
    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()
    
    # Define temporal windows - always use 5 equal windows spanning the configured plateau interval
    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX)
    tmin_clip = float(max(times.min(), tmin_req))
    tmax_clip = float(min(times.max(), tmax_req))
    if not np.isfinite(tmin_clip) or not np.isfinite(tmax_clip) or (tmax_clip <= tmin_clip):
        msg = f"No valid plateau interval within data range; skipping temporal topomaps (requested [{tmin_req}, {tmax_req}] s, available [{times.min():.2f}, {times.max():.2f}] s)."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return
    # Build 5 equal windows across [tmin_clip, tmax_clip]
    n_windows = 5
    edges = np.linspace(tmin_clip, tmax_clip, n_windows + 1)
    window_starts = edges[:-1]
    window_ends = edges[1:]
    window_size_eff = float((tmax_clip - tmin_clip) / n_windows)
    msg = f"Creating temporal topomaps over plateau [{tmin_clip:.2f}, {tmax_clip:.2f}] s using {n_windows} windows (~{window_size_eff:.2f}s each)."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    # Get frequency bands for per-band analysis
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = _get_consistent_bands(max_freq_available=fmax_available)
    
    # Create temporal topomaps for each frequency band
    for band_name, (fmin, fmax) in bands.items():
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue
            
        freq_label = f"{band_name} ({fmin:.0f}-{fmax_eff:.0f}Hz)"
        
        # Extract data for each temporal window in this frequency band
        pain_data_windows = []
        non_data_windows = []
        diff_data_windows = []
        
        for tmin_win, tmax_win in zip(window_starts, window_ends):
            # Average over frequency band and time window
            pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            
            if pain_data is not None and non_data is not None:
                diff_data = pain_data - non_data
                pain_data_windows.append(pain_data)
                non_data_windows.append(non_data)
                diff_data_windows.append(diff_data)
            else:
                pain_data_windows.append(None)
                non_data_windows.append(None)
                diff_data_windows.append(None)
        
        # Determine consistent scaling across all windows and conditions
        all_data = [d for d in pain_data_windows + non_data_windows if d is not None]
        diff_data_valid = [d for d in diff_data_windows if d is not None]
        
        if len(all_data) == 0:
            msg = f"No valid data found for {band_name} temporal topomaps; skipping this band."
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            continue
        
        # Symmetric scaling for pain/non-pain conditions
        vabs_cond = _robust_sym_vlim(all_data)
        # Symmetric scaling for difference
        vabs_diff = _robust_sym_vlim(diff_data_valid) if len(diff_data_valid) > 0 else vabs_cond
        
        # Create figure: 3 rows (pain, non-pain, difference) x n_windows columns
        fig, axes = plt.subplots(
            3, n_windows,
            figsize=(3.0 * n_windows, 9.0),
            squeeze=False,
            gridspec_kw={"hspace": 0.25, "wspace": 0.3}
        )
        
        row_labels = [f"Pain (n={int(pain_mask.sum())})", f"Non-pain (n={int(non_mask.sum())})", "Pain - Non"]
        
        for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            # Column title - positioned higher
            axes[0, col].set_title(f"{tmin_win:.1f}-{tmax_win:.1f}s", fontsize=10, pad=25)
            
            # Row 0: Pain condition
            pain_data = pain_data_windows[col]
            if pain_data is not None:
                try:
                    _plot_topomap_on_ax(
                        axes[0, col], pain_data, tfr_pain.info,
                        vmin=-vabs_cond, vmax=+vabs_cond
                    )
                    # Add mean value with percentage change above topomap
                    mu = float(np.nanmean(pain_data))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    axes[0, col].text(0.5, 1.08, f"%Δ={pct:+.0f}%", 
                                    transform=axes[0, col].transAxes, ha="center", va="bottom", fontsize=8)
                except Exception as e:
                    axes[0, col].axis('off')
                    if logger:
                        logger.warning(f"Pain topomap failed for window {col}: {e}")
            else:
                axes[0, col].axis('off')
            
            # Row 1: Non-pain condition  
            non_data = non_data_windows[col]
            if non_data is not None:
                try:
                    _plot_topomap_on_ax(
                        axes[1, col], non_data, tfr_non.info,
                        vmin=-vabs_cond, vmax=+vabs_cond
                    )
                    # Add mean value with percentage change above topomap
                    mu = float(np.nanmean(non_data))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    axes[1, col].text(0.5, 1.08, f"%Δ={pct:+.0f}%", 
                                    transform=axes[1, col].transAxes, ha="center", va="bottom", fontsize=8)
                except Exception as e:
                    axes[1, col].axis('off')
                    if logger:
                        logger.warning(f"Non-pain topomap failed for window {col}: {e}")
            else:
                axes[1, col].axis('off')
            
            # Row 2: Difference (pain - non-pain)
            diff_data = diff_data_windows[col]
            if diff_data is not None:
                try:
                    _plot_topomap_on_ax(
                        axes[2, col], diff_data, tfr_pain.info,
                        vmin=-vabs_diff, vmax=+vabs_diff
                    )
                    # Add mean difference value with percentage change above topomap
                    mu = float(np.nanmean(diff_data))
                    pct = (10.0 ** (mu) - 1.0) * 100.0
                    axes[2, col].text(0.5, 1.08, f"Δ%={pct:+.1f}%", 
                                    transform=axes[2, col].transAxes, ha="center", va="bottom", fontsize=8)
                except Exception as e:
                    axes[2, col].axis('off')
                    if logger:
                        logger.warning(f"Difference topomap failed for window {col}: {e}")
            else:
                axes[2, col].axis('off')
        
        # Row labels
        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=11, labelpad=10)
        
        # Add colorbars
        try:
            # Colorbar for pain/non-pain conditions (rows 0-1)
            sm_cond = ScalarMappable(
                norm=mcolors.TwoSlopeNorm(vmin=-vabs_cond, vcenter=0.0, vmax=vabs_cond),
                cmap=TOPO_CMAP
            )
            sm_cond.set_array([])
            cbar_cond = fig.colorbar(
                sm_cond, ax=axes[:2, :].ravel().tolist(),
                fraction=0.03, pad=0.02, shrink=0.8, aspect=20
            )
            cbar_cond.set_label("log10(power/baseline)", fontsize=10)
            
            # Colorbar for difference (row 2)
            sm_diff = ScalarMappable(
                norm=mcolors.TwoSlopeNorm(vmin=-vabs_diff, vcenter=0.0, vmax=vabs_diff),
                cmap=TOPO_CMAP
            )
            sm_diff.set_array([])
            cbar_diff = fig.colorbar(
                sm_diff, ax=axes[2, :].ravel().tolist(),
                fraction=0.03, pad=0.02, shrink=0.8, aspect=20
            )
            cbar_diff.set_label("log10(power/baseline) difference", fontsize=10)
        except Exception as e:
            if logger:
                logger.warning(f"Colorbar creation failed: {e}")
        
        # Overall title - positioned higher to avoid overlap
        try:
            fig.suptitle(
                f"Temporal topomaps: Pain vs Non-pain - {freq_label} (plateau {tmin_clip:.1f}–{tmax_clip:.1f}s; 5 windows)\n"
                f"log10(power/baseline), vlim ±{vabs_cond:.2f} (conditions), ±{vabs_diff:.2f} (difference)",
                fontsize=12, y=1.08
            )
        except Exception:
            pass
        
        # Save figure
        band_suffix = band_name.lower()
        filename = (
            f"temporal_topomaps_pain_vs_nonpain_{band_suffix}_plateau_{tmin_clip:.0f}-{tmax_clip:.0f}s_5windows.png"
        )
        _save_fig(fig, out_dir, filename, formats=["png", "svg"], logger=logger)


def contrast_pain_nonpain_rois(
    roi_tfrs: Dict[str, mne.time_frequency.EpochsTFR],
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
) -> None:
    """Pain vs non-pain contrasts per ROI (plots ROI-averaged TFRs)."""
    pain_col = _find_pain_binary_column(events_df)
    if pain_col is None:
        print(f"Events with pain binary column {PAIN_BINARY_COLUMNS} required for ROI contrasts; skipping.")
        return

    for roi, tfr in roi_tfrs.items():
        try:
            n_epochs = tfr.data.shape[0]
            n_meta = len(events_df)
            n = min(n_epochs, n_meta)
            if n_epochs != n_meta:
                print(f"ROI {roi}: trimming to {n} epochs to match events.")

            if getattr(tfr, "metadata", None) is not None and pain_col in tfr.metadata.columns:
                pain_vec = pd.to_numeric(tfr.metadata.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
            else:
                pain_vec = pd.to_numeric(events_df.iloc[:n][pain_col], errors="coerce").fillna(0).astype(int).values
            pain_mask = np.asarray(pain_vec == 1, dtype=bool)
            non_mask = np.asarray(pain_vec == 0, dtype=bool)
            if pain_mask.sum() == 0 or non_mask.sum() == 0:
                print(f"ROI {roi}: one group has zero trials; skipping.")
                continue

            tfr_sub = tfr.copy()[:n]
            if len(pain_mask) != len(tfr_sub):
                print(f"Warning (ROI {roi}): mask length ({len(pain_mask)}) != TFR epochs ({len(tfr_sub)}); reslicing to match.")
                n2 = min(len(tfr_sub), len(pain_mask))
                tfr_sub = tfr_sub[:n2]
                pain_mask = pain_mask[:n2]
                non_mask = non_mask[:n2]
                print(f"Debug after reslice (ROI {roi}): len(tfr_sub)={len(tfr_sub)}, len(pain_mask)={len(pain_mask)}")
            _apply_baseline_safe(tfr_sub, baseline=baseline, mode="logratio")
            tfr_pain = tfr_sub[pain_mask].average()
            tfr_non = tfr_sub[non_mask].average()

            ch = tfr_pain.info['ch_names'][0]
            # Pain
            try:
                fig = tfr_pain.plot(picks=ch, show=False)
                try:
                    fig.suptitle(f"ROI: {roi} — Painful (baseline logratio)", fontsize=12)
                except Exception:
                    pass
                _save_fig(fig, out_dir, f"tfr_ROI-{_sanitize(roi)}_painful_baseline_logratio.png")
            except Exception as e:
                print(f"ROI {roi} pain plot failed: {e}")
            # Non-pain
            try:
                fig = tfr_non.plot(picks=ch, show=False)
                try:
                    fig.suptitle(f"ROI: {roi} — Non-pain (baseline logratio)", fontsize=12)
                except Exception:
                    pass
                _save_fig(fig, out_dir, f"tfr_ROI-{_sanitize(roi)}_nonpain_baseline_logratio.png")
            except Exception as e:
                print(f"ROI {roi} non-pain plot failed: {e}")
            # Difference
            try:
                tfr_diff = tfr_pain.copy()
                tfr_diff.data = tfr_pain.data - tfr_non.data
                fig = tfr_diff.plot(picks=ch, show=False)
                try:
                    fig.suptitle(f"ROI: {roi} — Pain minus Non-pain (baseline logratio)", fontsize=12)
                except Exception:
                    pass
                _save_fig(fig, out_dir, f"tfr_ROI-{_sanitize(roi)}_pain_minus_nonpain_baseline_logratio.png")
            except Exception as e:
                print(f"ROI {roi} diff plot failed: {e}")

            # Also save band-limited TFRs for pain/non-pain/diff
            try:
                fmax_available = float(np.max(tfr_pain.freqs))
                bands = _get_consistent_bands(max_freq_available=fmax_available)
                for band, (fmin, fmax) in bands.items():
                    fmax_eff = min(fmax, fmax_available)
                    if fmin >= fmax_eff:
                        continue
                    # Pain band-limited
                    try:
                        fig_b = tfr_pain.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False)
                        try:
                            fig_b.suptitle(f"ROI: {roi} — {band} Painful (baseline logratio)", fontsize=12)
                        except Exception:
                            pass
                        _save_fig(fig_b, out_dir, f"tfr_ROI-{_sanitize(roi)}_{band}_painful_baseline_logratio.png")
                    except Exception as e:
                        print(f"ROI {roi} band {band} painful plot failed: {e}")
                    # Non-pain band-limited
                    try:
                        fig_b = tfr_non.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False)
                        try:
                            fig_b.suptitle(f"ROI: {roi} — {band} Non-pain (baseline logratio)", fontsize=12)
                        except Exception:
                            pass
                        _save_fig(fig_b, out_dir, f"tfr_ROI-{_sanitize(roi)}_{band}_nonpain_baseline_logratio.png")
                    except Exception as e:
                        print(f"ROI {roi} band {band} non-pain plot failed: {e}")
                    # Diff band-limited
                    try:
                        fig_b = tfr_diff.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False)
                        try:
                            fig_b.suptitle(f"ROI: {roi} — {band} Pain minus Non-pain (baseline logratio)", fontsize=12)
                        except Exception:
                            pass
                        _save_fig(fig_b, out_dir, f"tfr_ROI-{_sanitize(roi)}_{band}_pain_minus_nonpain_baseline_logratio.png")
                    except Exception as e:
                        print(f"ROI {roi} band {band} diff plot failed: {e}")
            except Exception as e:
                print(f"ROI {roi} banded contrast TFR export skipped: {e}")
        except Exception as e:
            print(f"ROI {roi} contrast failed: {e}")
    

def _write_group_pain_counts_from_events(
    subjects: List[str],
    events_list: List[Optional[pd.DataFrame]],
    out_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    rows = []
    for subj, ev in zip(subjects, events_list):
        n_pain = 0
        n_non = 0
        if ev is not None:
            pain_col = None
            for c in PAIN_BINARY_COLUMNS:
                if c in ev.columns:
                    pain_col = c; break
            if pain_col is not None:
                vals = pd.to_numeric(ev[pain_col], errors="coerce")
                n_pain = int((vals == 1).sum())
                n_non = int((vals == 0).sum())
        rows.append({
            "subject": subj,
            "n_pain": n_pain,
            "n_nonpain": n_non,
            "n_total": n_pain + n_non,
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    total = df[["n_pain", "n_nonpain", "n_total"]].sum()
    total_row = {"subject": "TOTAL", **{k: int(v) for k, v in total.to_dict().items()}}
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    _ensure_dir(out_dir)
    out_path = out_dir / "counts_pain.tsv"
    try:
        df.to_csv(out_path, sep="\t", index=False)
        if logger:
            logger.info(f"Saved counts: {out_path}")
        else:
            print(f"Saved counts: {out_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to write counts TSV at {out_path}: {e}")
        else:
            print(f"Warning: Failed to write counts TSV at {out_path}: {e}")


def _write_group_band_summary(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Write concise TSV with per-band log10 ratio and % change for group conditions.

    Conditions: All-trials, Pain, Non-pain, Diff, and per temperature labels.
    Means are computed over EEG channels, plateau window, and band frequencies.
    """
    rows: list[dict] = []
    # Helper to append rows
    def add_row(cond: str, label: str, band: str, mu: float, n_subj: int):
        rows.append({
            "condition": cond,
            "label": label,
            "band": band,
            "mu_log10": float(mu),
            "pct_change": float((10.0 ** mu - 1.0) * 100.0),
            "n_subjects": int(n_subj),
        })

    # All-trials
    avg_list = []
    for p in powers:
        try:
            t = p.copy(); _apply_baseline_safe(t, baseline=BASELINE, mode="logratio")
            avg_list.append(t.average())
        except Exception:
            continue
    info_all, data_all = _align_avg_tfrs(avg_list, logger=logger)
    if info_all is not None and data_all is not None:
        freqs = np.asarray(avg_list[0].freqs)
        times = np.asarray(avg_list[0].times)
        fmax_available = float(freqs.max())
        bands = _get_consistent_bands(max_freq_available=fmax_available)
        tmin_req, tmax_req = plateau_window
        tmin = float(max(times.min(), tmin_req)); tmax = float(min(times.max(), tmax_req))
        mean_all = data_all.mean(axis=0)
        eeg_picks = mne.pick_types(info_all, eeg=True, exclude=[])
        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            fmask = (freqs >= fmin) & (freqs <= fmax_eff)
            tmask = (times >= tmin) & (times < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0:
                continue
            vec = mean_all[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            mu = float(np.nanmean(vec[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(vec))
            add_row("All", "All trials", band, mu, data_all.shape[0])

    # Pain vs Non
    avg_pain = []; avg_non = []
    for p, ev in zip(powers, events_by_subj):
        if ev is None: continue
        pain_col = None
        for c in PAIN_BINARY_COLUMNS:
            if c in ev.columns: pain_col = c; break
        if pain_col is None: continue
        vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
        mask_p = np.asarray(vals == 1, dtype=bool); mask_n = np.asarray(vals == 0, dtype=bool)
        if mask_p.sum() == 0 or mask_n.sum() == 0: continue
        a_p = _avg_by_mask_to_avg_tfr(p, mask_p); a_n = _avg_by_mask_to_avg_tfr(p, mask_n)
        if a_p is None or a_n is None: continue
        avg_pain.append(a_p); avg_non.append(a_n)
    info_p, data_p = _align_avg_tfrs(avg_pain, logger=logger)
    info_n, data_n = _align_avg_tfrs(avg_non, logger=logger)
    if info_p is not None and info_n is not None and data_p is not None and data_n is not None:
        freqs = np.asarray(avg_pain[0].freqs)
        times = np.asarray(avg_pain[0].times)
        fmax_available = float(freqs.max())
        bands = _get_consistent_bands(max_freq_available=fmax_available)
        tmin_req, tmax_req = plateau_window
        tmin = float(max(times.min(), tmin_req)); tmax = float(min(times.max(), tmax_req))
        mean_p = data_p.mean(axis=0); mean_n = data_n.mean(axis=0)
        eeg_picks_p = mne.pick_types(info_p, eeg=True, exclude=[])
        eeg_picks_n = mne.pick_types(info_n, eeg=True, exclude=[])
        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            fmask = (freqs >= fmin) & (freqs <= fmax_eff)
            tmask = (times >= tmin) & (times < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0:
                continue
            v_p = mean_p[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            v_n = mean_n[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            mu_p = float(np.nanmean(v_p[eeg_picks_p])) if len(eeg_picks_p) > 0 else float(np.nanmean(v_p))
            mu_n = float(np.nanmean(v_n[eeg_picks_n])) if len(eeg_picks_n) > 0 else float(np.nanmean(v_n))
            mu_d = float(np.nanmean(v_p - v_n))
            add_row("Pain", "pain", band, mu_p, data_p.shape[0])
            add_row("Non-pain", "non-pain", band, mu_n, data_n.shape[0])
            add_row("Diff", "pain-minus-non", band, mu_d, min(data_p.shape[0], data_n.shape[0]))

    # Temperatures
    # Reuse routine to collect temps per group
    temps = _collect_group_temperatures(events_by_subj)
    for tval in temps:
        avg_list = []
        ns = 0
        for p, ev in zip(powers, events_by_subj):
            if ev is None: continue
            tcol = None
            for c in TEMPERATURE_COLUMNS:
                if c in ev.columns: tcol = c; break
            if tcol is None: continue
            mask = pd.to_numeric(ev[tcol], errors="coerce").round(1) == round(float(tval), 1)
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0: continue
            a = _avg_by_mask_to_avg_tfr(p, mask)
            if a is not None:
                avg_list.append(a); ns += 1
        info_t, data_t = _align_avg_tfrs(avg_list, logger=logger)
        if info_t is None or data_t is None: continue
        freqs = np.asarray(avg_list[0].freqs)
        times = np.asarray(avg_list[0].times)
        fmax_available = float(freqs.max())
        bands = _get_consistent_bands(max_freq_available=fmax_available)
        tmin_req, tmax_req = plateau_window
        tmin = float(max(times.min(), tmin_req)); tmax = float(min(times.max(), tmax_req))
        mean_t = data_t.mean(axis=0)
        eeg_picks = mne.pick_types(info_t, eeg=True, exclude=[])
        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            fmask = (freqs >= fmin) & (freqs <= fmax_eff)
            tmask = (times >= tmin) & (times < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0: continue
            vec = mean_t[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
            mu = float(np.nanmean(vec[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(vec))
            add_row("Temperature", f"{float(tval):.1f}°C", band, mu, data_t.shape[0])

    if rows:
        _ensure_dir(out_dir)
        out_path = out_dir / "group_band_summary.tsv"
        try:
            pd.DataFrame(rows).to_csv(out_path, sep='\t', index=False)
            if logger:
                logger.info(f"Saved band summary: {out_path}")
            else:
                print(f"Saved band summary: {out_path}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to write band summary TSV: {e}")
            else:
                print(f"Warning: Failed to write band summary TSV: {e}")


def _write_group_roi_summary(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
    roi_map: Optional[Dict[str, list[str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Write TSV with per-ROI, per-band log10 ratio and % change for group conditions.

    Conditions: All-trials, Pain, Non-pain, Diff, and per-temperature labels.
    Means computed over ROI channel (single-channel AverageTFR), plateau window, and band freqs.
    """
    rows: list[dict] = []
    def add_row(roi: str, cond: str, label: str, band: str, mu: float, n_subj: int):
        rows.append({
            "roi": roi,
            "condition": cond,
            "label": label,
            "band": band,
            "mu_log10": float(mu),
            "pct_change": float((10.0 ** mu - 1.0) * 100.0),
            "n_subjects": int(n_subj),
        })

    if not powers:
        return

    # Build per-subject AverageTFR (all trials) once
    avg_all = []
    for p in powers:
        try:
            avg_all.append(_avg_alltrials_to_avg_tfr(p))
        except Exception:
            continue
    if not avg_all:
        return

    # Determine bands and plateau window from any AverageTFR
    freqs_ref = np.asarray(avg_all[0].freqs)
    times_ref = np.asarray(avg_all[0].times)
    fmax_available = float(freqs_ref.max())
    bands = _get_consistent_bands(max_freq_available=fmax_available)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times_ref.min(), tmin_req)); tmax = float(min(times_ref.max(), tmax_req))

    # ROIs
    rois = list((roi_map or _roi_definitions()).keys())

    # Helper: compute group mean ROI AverageTFR list -> (mean_data, n_subj)
    def group_mean_roi(avgs: List["mne.time_frequency.AverageTFR"], roi_name: str) -> Tuple[Optional[np.ndarray], int]:
        rois_list: List["mne.time_frequency.AverageTFR"] = []
        for a in avgs:
            r = _avg_tfr_to_roi_average(a, roi_name, roi_map_override=roi_map)
            if r is not None:
                rois_list.append(r)
        if not rois_list:
            return None, 0
        info_c, data_c = _align_avg_tfrs(rois_list, logger=logger)
        if info_c is None or data_c is None:
            return None, 0
        return data_c.mean(axis=0), data_c.shape[0]  # (1, n_freqs, n_times), N

    # All-trials per ROI
    for roi in rois:
        mean_roi, nsub = group_mean_roi(avg_all, roi)
        if mean_roi is None or nsub == 0:
            continue
        arr = np.asarray(mean_roi[0])  # (n_freqs, n_times)
        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            fmask = (freqs_ref >= fmin) & (freqs_ref <= fmax_eff)
            tmask = (times_ref >= tmin) & (times_ref < tmax)
            if fmask.sum() == 0 or tmask.sum() == 0:
                continue
            mu = float(np.nanmean(arr[fmask, :][:, tmask]))
            add_row(roi, "All", "All trials", band, mu, nsub)

    # Pain/Non/Diff per ROI
    avg_pain = []
    avg_non = []
    for p, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        pain_col = None
        for c in PAIN_BINARY_COLUMNS:
            if c in ev.columns:
                pain_col = c; break
        if pain_col is None:
            continue
        vals = pd.to_numeric(ev[pain_col], errors="coerce").fillna(0).astype(int).values
        mask_p = np.asarray(vals == 1, dtype=bool)
        mask_n = np.asarray(vals == 0, dtype=bool)
        if mask_p.sum() == 0 or mask_n.sum() == 0:
            continue
        a_p = _avg_by_mask_to_avg_tfr(p, mask_p)
        a_n = _avg_by_mask_to_avg_tfr(p, mask_n)
        if a_p is not None and a_n is not None:
            avg_pain.append(a_p)
            avg_non.append(a_n)
    if avg_pain and avg_non:
        freqs_p = np.asarray(avg_pain[0].freqs)
        times_p = np.asarray(avg_pain[0].times)
        fmax_available_p = float(freqs_p.max())
        bands_p = _get_consistent_bands(max_freq_available=fmax_available_p)
        tmin_p = float(max(times_p.min(), tmin_req)); tmax_p = float(min(times_p.max(), tmax_req))
        for roi in rois:
            mean_p, n_p = group_mean_roi(avg_pain, roi)
            mean_n, n_n = group_mean_roi(avg_non, roi)
            if mean_p is None or mean_n is None or n_p == 0 or n_n == 0:
                continue
            arr_p = np.asarray(mean_p[0]); arr_n = np.asarray(mean_n[0])
            for band, (fmin, fmax) in bands_p.items():
                fmax_eff = min(fmax, fmax_available_p)
                fmask = (freqs_p >= fmin) & (freqs_p <= fmax_eff)
                tmask = (times_p >= tmin_p) & (times_p < tmax_p)
                if fmask.sum() == 0 or tmask.sum() == 0:
                    continue
                mu_p = float(np.nanmean(arr_p[fmask, :][:, tmask]))
                mu_n = float(np.nanmean(arr_n[fmask, :][:, tmask]))
                mu_d = float(np.nanmean((arr_p - arr_n)[fmask, :][:, tmask]))
                add_row(roi, "Pain", "pain", band, mu_p, n_p)
                add_row(roi, "Non-pain", "non-pain", band, mu_n, n_n)
                add_row(roi, "Diff", "pain-minus-non", band, mu_d, min(n_p, n_n))

    # Temperatures per ROI
    temps = _collect_group_temperatures(events_by_subj)
    if temps:
        for roi in rois:
            for tval in temps:
                avgs_t = []
                for p, ev in zip(powers, events_by_subj):
                    if ev is None:
                        continue
                    tcol = None
                    for c in TEMPERATURE_COLUMNS:
                        if c in ev.columns:
                            tcol = c; break
                    if tcol is None:
                        continue
                    mask = pd.to_numeric(ev[tcol], errors="coerce").round(1) == round(float(tval), 1)
                    mask = np.asarray(mask, dtype=bool)
                    if mask.sum() == 0:
                        continue
                    a = _avg_by_mask_to_avg_tfr(p, mask)
                    if a is not None:
                        avgs_t.append(a)
                if not avgs_t:
                    continue
                freqs_t = np.asarray(avgs_t[0].freqs)
                times_t = np.asarray(avgs_t[0].times)
                fmax_available_t = float(freqs_t.max())
                bands_t = _get_consistent_bands(max_freq_available=fmax_available_t)
                tmin_t = float(max(times_t.min(), tmin_req)); tmax_t = float(min(times_t.max(), tmax_req))
                mean_t, n_t = group_mean_roi(avgs_t, roi)
                if mean_t is None or n_t == 0:
                    continue
                arr_t = np.asarray(mean_t[0])
                for band, (fmin, fmax) in bands_t.items():
                    fmax_eff = min(fmax, fmax_available_t)
                    fmask = (freqs_t >= fmin) & (freqs_t <= fmax_eff)
                    tmask = (times_t >= tmin_t) & (times_t < tmax_t)
                    if fmask.sum() == 0 or tmask.sum() == 0:
                        continue
                    mu = float(np.nanmean(arr_t[fmask, :][:, tmask]))
                    add_row(roi, "Temperature", f"{float(tval):.1f}°C", band, mu, n_t)

    if rows:
        _ensure_dir(out_dir)
        out_path = out_dir / "group_roi_summary.tsv"
        try:
            pd.DataFrame(rows).to_csv(out_path, sep='\t', index=False)
            if logger:
                logger.info(f"Saved ROI summary: {out_path}")
            else:
                print(f"Saved ROI summary: {out_path}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to write ROI summary TSV: {e}")
            else:
                print(f"Warning: Failed to write ROI summary TSV: {e}")


def main_group(
    subjects: List[str],
    task: str = DEFAULT_TASK,
    plateau_tmin: float = DEFAULT_PLATEAU_TMIN,
    plateau_tmax: float = DEFAULT_PLATEAU_TMAX,
    temperature_strategy: str = DEFAULT_TEMPERATURE_STRATEGY,
) -> None:
    logger = _setup_group_logging()
    logger.info(f"=== Time-frequency group analysis: {len(subjects)} subjects, task-{task} ===")
    logger.info(f"Subjects: {', '.join(subjects)}")
    out_dir = DERIV_ROOT / "group" / "eeg" / "plots" / "02_time_frequency_analysis"
    _ensure_dir(out_dir)

    powers: List[mne.time_frequency.EpochsTFR] = []
    events_list: List[Optional[pd.DataFrame]] = []
    ok_subjects: List[str] = []
    for s in subjects:
        logger.info(f"--- Computing TFR for subject {s} ---")
        power, ev = _compute_power_and_events(s, task, logger)
        if power is not None:
            powers.append(power)
            events_list.append(ev)
            ok_subjects.append(s)
        else:
            logger.warning(f"Skipping subject {s} due to errors")

    if len(powers) < 2:
        logger.warning(f"Only {len(powers)} subjects valid; skipping group-level plots")
        return

    # Build group ROI map from union of channels across subjects
    try:
        all_chs: list[str] = []
        seen = set()
        for p in powers:
            for ch in p.info['ch_names']:
                if ch not in seen:
                    seen.add(ch)
                    all_chs.append(ch)
        group_roi_map = _build_group_roi_map_from_channels(all_chs)
        # Summarize counts per ROI for transparency
        try:
            counts = {roi: len(chs) for roi, chs in group_roi_map.items()}
            counts_str = ", ".join([f"{k}={v}" for k, v in counts.items()])
            logger.info(f"Built group ROI map from union of subject channels; counts: {counts_str}")
        except Exception:
            logger.info("Built group ROI map from union of subject channels")
    except Exception as e:
        logger.warning(f"Failed to build group ROI map: {e}")
        group_roi_map = None

    # Group all-trials topomaps
    group_topomaps_bands_all_trials(
        powers,
        out_dir,
        plateau_window=(plateau_tmin, plateau_tmax),
        logger=logger,
    )

    # Group pain vs non-pain topomaps
    try:
        group_topomaps_pain_nonpain(
            powers,
            events_list,
            out_dir,
            plateau_window=(plateau_tmin, plateau_tmax),
            logger=logger,
        )
    except Exception as e:
        logger.warning(f"Group pain/non-pain topomaps failed: {e}")

    # Group temperature grid (pooled/both)
    if temperature_strategy in ("pooled", "both"):
        try:
            group_topomap_grid_baseline_temps(
                powers,
                events_list,
                out_dir,
                plateau_window=(plateau_tmin, plateau_tmax),
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"Group temperature grid failed: {e}")
        # Group max vs min temperature grid
        try:
            group_contrast_maxmin_temperature(
                powers,
                events_list,
                out_dir,
                plateau_window=(plateau_tmin, plateau_tmax),
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"Group max/min temperature grid failed: {e}")

    # Group ROI analyses (all trials and pain/non)
    try:
        group_rois_all_trials(powers, out_dir, baseline=BASELINE, logger=logger, roi_map=group_roi_map)
    except Exception as e:
        logger.warning(f"Group ROI all-trials failed: {e}")
    try:
        group_contrast_pain_nonpain_rois(powers, events_list, out_dir, baseline=BASELINE, logger=logger, roi_map=group_roi_map)
    except Exception as e:
        logger.warning(f"Group ROI pain/non failed: {e}")

    # Group temporal topomaps across windows (pain vs non)
    try:
        group_pain_nonpain_temporal_topomaps(
            powers,
            events_list,
            out_dir,
            plateau_window=(plateau_tmin, plateau_tmax),
            window_count=5,
            logger=logger,
        )
    except Exception as e:
        logger.warning(f"Group temporal topomaps failed: {e}")

    # Counts and band-summary TSVs
    _write_group_pain_counts_from_events(ok_subjects, events_list, out_dir, logger)
    try:
        _write_group_band_summary(powers, events_list, out_dir, plateau_window=(plateau_tmin, plateau_tmax), logger=logger)
    except Exception as e:
        logger.warning(f"Band summary TSV failed: {e}")
    try:
        _write_group_roi_summary(powers, events_list, out_dir, plateau_window=(plateau_tmin, plateau_tmax), roi_map=group_roi_map, logger=logger)
    except Exception as e:
        logger.warning(f"ROI summary TSV failed: {e}")
    logger.info(f"Group analysis completed. Results saved to: {out_dir}")
def main(subject: str = "001", task: str = DEFAULT_TASK,
         plateau_tmin: float = DEFAULT_PLATEAU_TMIN, plateau_tmax: float = DEFAULT_PLATEAU_TMAX,
         temperature_strategy: str = DEFAULT_TEMPERATURE_STRATEGY) -> None:
    logger = _setup_logging(subject)
    logger.info(f"=== Time-frequency analysis: sub-{subject}, task-{task} ===")
    plots_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots" / "02_time_frequency_analysis"
    _ensure_dir(plots_dir)

    # Load cleaned epochs
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None or not epo_path.exists():
        msg = f"Error: cleaned epochs file not found for sub-{subject}, task-{task} under {DERIV_ROOT}."
        if logger:
            logger.error(msg)
        else:
            print(msg)
        sys.exit(1)
    msg = f"Loading epochs: {epo_path}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)

    # Load events and align lengths; attach as metadata for consistency
    events_df = _load_events_df(subject, task, logger)
    if events_df is not None:
        # Robust alignment: try epochs.selection first, then sample-based, then fallback trim
        aligned = False
        sel = getattr(epochs, "selection", None)
        if sel is not None and len(sel) == len(epochs):
            try:
                if len(events_df) > int(np.max(sel)):
                    events_aligned = events_df.iloc[sel].reset_index(drop=True)
                    events_df = events_aligned
                    epochs.metadata = events_df.copy()
                    aligned = True
                    msg = f"SUCCESS: Aligned metadata using epochs.selection (epochs={len(epochs)}, events={len(events_df)})"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
            except Exception as e:
                msg = f"Selection-based alignment failed: {e}"
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)

        if not aligned and "sample" in events_df.columns and isinstance(getattr(epochs, "events", None), np.ndarray):
            try:
                samples = epochs.events[:, 0]
                events_by_sample = events_df.set_index("sample")
                events_aligned = events_by_sample.reindex(samples)
                if len(events_aligned) == len(epochs) and not events_aligned.isna().all(axis=1).any():
                    events_df = events_aligned.reset_index()
                    epochs.metadata = events_df.copy()
                    aligned = True
                    msg = f"SUCCESS: Aligned metadata using 'sample' column (epochs={len(epochs)}, events={len(events_df)})"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
            except Exception as e:
                msg = f"Sample-based alignment failed: {e}"
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)

        if not aligned:
            n = min(len(events_df), len(epochs))
            if len(events_df) != len(epochs):
                msg = (
                    f"Epochs ({len(epochs)}) and events ({len(events_df)}) length mismatch (cannot prove alignment)."
                )
                if logger:
                    logger.error(msg)
                else:
                    print(f"Error: {msg}")
                if not ALLOW_MISALIGNED_TRIM:
                    fail_msg = (
                        "Set time_frequency_analysis.allow_misaligned_trim=true in config to trim and proceed."
                    )
                    if logger:
                        logger.error(fail_msg)
                    else:
                        print(f"Error: {fail_msg}")
                    sys.exit(1)
                # Proceed with trimming only if explicitly allowed
                warn2 = (
                    f"FALLBACK (trim to n={n}): behavioral alignment may be unreliable — verify event filtering and epoch selection."
                )
                if logger:
                    logger.warning(warn2)
                else:
                    print(f"Warning: {warn2}")
            if len(epochs) != n:
                epochs = epochs[:n]
            events_df = events_df.iloc[:n].reset_index(drop=True)
            try:
                epochs.metadata = events_df.copy()
            except Exception as e:
                msg = f"CRITICAL: Failed to attach metadata to epochs: {e} - contrasts will be skipped"
                if logger:
                    logger.error(msg)
                else:
                    print(f"Error: {msg}")
    else:
        msg = "Warning: events.tsv missing; contrasts will be skipped if needed."
        if logger:
            logger.warning(msg)
        else:
            print(msg)

    # Compute per-trial TFR using Morlet wavelets
    # Frequencies covering alpha/beta/gamma using CONFIG
    freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQS)
    # Use adaptive n_cycles with a minimum floor for better low-frequency resolution
    # (consistent with our diagnostics and common practice)
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=N_CYCLES_FACTOR, min_cycles=3.0)
    msg = "Computing per-trial TFR (Morlet)..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=TFR_DECIM,
        n_jobs=-1,
        picks=TFR_PICKS,
        verbose=False,
    )
    # power is EpochsTFR
    msg = f"Computed TFR: type={type(power).__name__}, n_epochs={power.data.shape[0]}, n_freqs={len(power.freqs)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    # Diagnostics: raw Cz and baseline vs plateau QC on un-baselined TFR
    try:
        plot_cz_all_trials_raw(power, plots_dir, logger)
    except Exception as e:
        if logger:
            logger.warning(f"Raw Cz plot failed: {e}")
        else:
            print(f"Raw Cz plot failed: {e}")
    try:
        qc_baseline_plateau_power(
            power,
            plots_dir,
            baseline=BASELINE,
            plateau_window=(plateau_tmin, plateau_tmax),
            logger=logger,
        )
    except Exception as e:
        if logger:
            logger.warning(f"QC baseline/plateau failed: {e}")
        else:
            print(f"QC baseline/plateau failed: {e}")

    # Optionally run pooled (no temperature discrimination)
    if temperature_strategy in ("pooled", "both"):
        # All trials, Cz TFR (baseline-corrected using pre-stim interval)
        plot_cz_all_trials(power, plots_dir, baseline=BASELINE)

        # DIAGNOSTIC: Baseline correction investigation
        diagnostic_baseline_correction_methods(
            tfr=power,
            out_dir=plots_dir,
            baseline=BASELINE,
            plateau_window=(plateau_tmin, plateau_tmax),
            logger=logger,
        )
        
        diagnostic_alternative_baselines(
            tfr=power,
            out_dir=plots_dir,
            logger=logger,
        )

        # Pain vs Non-pain (if available)
        contrast_pain_nonpain(
            tfr=power,
            events_df=events_df,
            out_dir=plots_dir,
            baseline=BASELINE,
            plateau_window=(plateau_tmin, plateau_tmax),
            logger=logger,
        )

        # Contrasts and topomaps (handled above if events are available)

        # Temporal topomaps: pain vs non-pain across 2-second windows (per frequency band)
        try:
            plot_pain_nonpain_temporal_topomaps(
                tfr=power,
                events_df=events_df,
                out_dir=plots_dir,
                baseline=BASELINE,
                window_size=2.0,
                logger=logger,
            )
        except Exception as e:
            msg = f"Temporal topomaps failed: {e}"
            if logger:
                logger.error(msg)
            else:
                print(msg)

        # Per-ROI analysis: compute ROI EpochsTFRs from channel-averaged epochs
        msg = "Building ROIs and computing ROI TFRs (pooled)..."
        if logger:
            logger.info(msg)
        else:
            print(msg)
        roi_map = _build_rois(epochs.info)
        if len(roi_map) == 0:
            msg = "No ROI channels found in montage; skipping ROI analysis."
            if logger:
                logger.warning(msg)
            else:
                print(msg)
        else:
            roi_tfrs = compute_roi_tfrs(epochs, freqs=freqs, n_cycles=n_cycles, roi_map=roi_map)
            # Plot per-ROI all trials
            plot_rois_all_trials(roi_tfrs, plots_dir, baseline=BASELINE)
            # Per-ROI contrasts
            contrast_pain_nonpain_rois(roi_tfrs, events_df, plots_dir, baseline=BASELINE)
            # Full-scalp topomaps by frequency bands (all trials)
            plot_topomaps_bands_all_trials(power, plots_dir, baseline=BASELINE, plateau_window=(plateau_tmin, plateau_tmax))
            # Per-ROI topomap contrasts
            contrast_pain_nonpain_topomaps_rois(power, events_df, roi_map, plots_dir, baseline=BASELINE, plateau_window=(plateau_tmin, plateau_tmax))

    # Optionally run per-temperature
    temp_col = _find_temperature_column(events_df)
    # Multi-temperature topomap grid (All trials + per-temperature) at pooled level
    if temperature_strategy in ("pooled", "both") and events_df is not None and temp_col is not None:
        try:
            plot_topomap_grid_baseline_temps(
                tfr=power,
                events_df=events_df,
                out_dir=plots_dir,
                baseline=BASELINE,
                plateau_window=(plateau_tmin, plateau_tmax),
            )
        except Exception as e:
            msg = f"Temperature grid plot failed: {e}"
            if logger:
                logger.error(msg)
            else:
                print(msg)
        try:
            contrast_maxmin_temperature(
                tfr=power,
                events_df=events_df,
                out_dir=plots_dir,
                baseline=BASELINE,
                plateau_window=(plateau_tmin, plateau_tmax),
            )
        except Exception as e:
            msg = f"Max-min temperature contrast plot failed: {e}"
            if logger:
                logger.error(msg)
            else:
                print(msg)
    if temperature_strategy in ("per", "both"):
        if events_df is None or temp_col is None:
            msg = "Per-temperature analysis requested, but no temperature column found; skipping per-temperature plots."
            if logger:
                logger.warning(msg)
            else:
                print(msg)
        else:
            # Determine unique temperatures (rounded to 1 decimal to stabilize floats)
            try:
                temps = (
                    pd.to_numeric(events_df[temp_col], errors="coerce")
                    .round(1)
                    .dropna()
                    .unique()
                )
                temps = sorted(map(float, temps))
            except Exception:
                # Fallback: use raw unique values
                temps = sorted(events_df[temp_col].dropna().unique())
            if len(temps) == 0:
                msg = "No temperatures found in events; skipping per-temperature plots."
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
            else:
                msg = f"Running per-temperature analysis for {len(temps)} level(s): {temps}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                # Build ROI map once (channel set is constant across subsets)
                roi_map_all = _build_rois(epochs.info)
                for tval in temps:
                    # Build subset mask and slice epochs/events
                    try:
                        mask = pd.to_numeric(events_df[temp_col], errors="coerce").round(1) == round(float(tval), 1)
                    except Exception:
                        mask = events_df[temp_col] == tval
                    n_sel = int(mask.sum())
                    if n_sel == 0:
                        continue
                    epochs_t = epochs.copy()[mask.to_numpy()]
                    events_t = events_df.loc[mask].reset_index(drop=True)

                    # Output directory for this temperature
                    t_label = _format_temp_label(float(events_t[temp_col].iloc[0]))
                    plots_dir_t = plots_dir / f"temperature" / f"temp-{t_label}"
                    _ensure_dir(plots_dir_t)

                    # Compute per-trial TFR for this subset
                    msg = f"Computing TFR for temperature {tval} ({n_sel} trials)..."
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                    power_t = mne.time_frequency.tfr_morlet(
                        epochs_t,
                        freqs=freqs,
                        n_cycles=n_cycles,
                        use_fft=True,
                        return_itc=False,
                        average=False,
                        decim=TFR_DECIM,
                        picks=TFR_PICKS,
                        n_jobs=-1,
                    )

                    # Cz all-trials
                    plot_cz_all_trials(power_t, plots_dir_t, baseline=BASELINE)

                    # Pain vs Non-pain (if available)
                    contrast_pain_nonpain(
                        tfr=power_t,
                        events_df=events_t,
                        out_dir=plots_dir_t,
                        baseline=BASELINE,
                        plateau_window=(plateau_tmin, plateau_tmax),
                        logger=logger,
                    )

                    # ROI analyses for this temperature
                    if len(roi_map_all) == 0:
                        msg = "No ROI channels found; skipping ROI analyses for temperature subset."
                        if logger:
                            logger.warning(msg)
                        else:
                            print(msg)
                    else:
                        roi_tfrs_t = compute_roi_tfrs(epochs_t, freqs=freqs, n_cycles=n_cycles, roi_map=roi_map_all)
                        plot_rois_all_trials(roi_tfrs_t, plots_dir_t, baseline=BASELINE)
                        contrast_pain_nonpain_rois(roi_tfrs_t, events_t, plots_dir_t, baseline=BASELINE)
                        plot_topomaps_bands_all_trials(power_t, plots_dir_t, baseline=BASELINE, plateau_window=(plateau_tmin, plateau_tmax))
                        contrast_pain_nonpain_topomaps_rois(power_t, events_t, roi_map_all, plots_dir_t, baseline=BASELINE, plateau_window=(plateau_tmin, plateau_tmax))

    msg = "Done."
    if logger:
        logger.info(msg)
    else:
        print(msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Time-frequency analysis supporting single and multiple subjects")

    # Subject selection (mutually exclusive): --group, --subject(s), --all-subjects
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
        help="Process all available subjects with cleaned epochs files",
    )

    parser.add_argument("--task", "-t", type=str, default=DEFAULT_TASK, help="BIDS task label (default from config)")
    parser.add_argument("--plateau_tmin", type=float, default=DEFAULT_PLATEAU_TMIN, help="Plateau window start time in seconds (for topomaps and summaries)")
    parser.add_argument("--plateau_tmax", type=float, default=DEFAULT_PLATEAU_TMAX, help="Plateau window end time in seconds (for topomaps and summaries)")
    parser.add_argument(
        "--temperature_strategy", "-T",
        type=str,
        choices=["pooled", "per", "both"],
        default=DEFAULT_TEMPERATURE_STRATEGY,
        help="Temperature analysis strategy: pooled/per/both (default from config)",
    )

    args = parser.parse_args()

    # Resolve subjects
    subjects: Optional[List[str]] = None
    if args.group is not None:
        g = args.group.strip().lower()
        if g in {"all", "*", "@all"}:
            subjects = _get_available_subjects()
        else:
            cand = [s.strip() for s in g.replace(";", ",").replace(" ", ",").split(",") if s.strip()]
            subjects = []
            for s in cand:
                if _find_clean_epochs_path(s, args.task) is not None:
                    subjects.append(s)
                else:
                    print(f"Warning: --group subject '{s}' has no cleaned epochs; skipping")
    elif args.all_subjects:
        subjects = _get_available_subjects()
    elif args.subject:
        subjects = list(dict.fromkeys(args.subject))  # de-dup preserving order

    if subjects is None or len(subjects) == 0:
        # Single-subject mode requires --subject exactly one
        parser_single = argparse.ArgumentParser(add_help=False)
        # Just to provide a clearer message
        print("No subjects provided via --group/--all-subjects/--subject. For single subject, pass --subject <ID>.")
        sys.exit(2)

    if len(subjects) == 1:
        main(subject=subjects[0], task=args.task, plateau_tmin=args.plateau_tmin, plateau_tmax=args.plateau_tmax, temperature_strategy=args.temperature_strategy)
    else:
        main_group(subjects=subjects, task=args.task, plateau_tmin=args.plateau_tmin, plateau_tmax=args.plateau_tmax, temperature_strategy=args.temperature_strategy)
