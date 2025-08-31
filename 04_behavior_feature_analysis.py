from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import itertools
import logging

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import mne
from mne_bids import BIDSPath

# Note: Matplotlib setup is now handled by config.setup_matplotlib()

# ==========================
# CONFIG
# Load centralized configuration from YAML
# ==========================
from config_loader import load_config, get_legacy_constants

# Load configuration
config = load_config()
config.setup_matplotlib()

# Extract legacy constants for backward compatibility
_constants = get_legacy_constants(config)

PROJECT_ROOT = _constants["PROJECT_ROOT"]
BIDS_ROOT = _constants["BIDS_ROOT"] 
DERIV_ROOT = _constants["DERIV_ROOT"]
SUBJECTS = _constants["SUBJECTS"]
TASK = _constants["TASK"]
FEATURES_FREQ_BANDS = _constants["FEATURES_FREQ_BANDS"]
PSYCH_TEMP_COLUMNS = _constants["PSYCH_TEMP_COLUMNS"]
RATING_COLUMNS = _constants["RATING_COLUMNS"]
PAIN_BINARY_COLUMNS = _constants["PAIN_BINARY_COLUMNS"]
POWER_BANDS_TO_USE = _constants["POWER_BANDS_TO_USE"]
PLATEAU_WINDOW = _constants["PLATEAU_WINDOW"]
FIG_DPI = _constants["FIG_DPI"]
SAVE_FORMATS = _constants["SAVE_FORMATS"]
LOG_FILE_NAME = _constants["LOG_FILE_NAME"]
USE_SPEARMAN_DEFAULT = _constants["USE_SPEARMAN_DEFAULT"]
PARTIAL_COVARS_DEFAULT = _constants["PARTIAL_COVARS_DEFAULT"]
BOOTSTRAP_DEFAULT = _constants["BOOTSTRAP_DEFAULT"]
N_PERM_DEFAULT = _constants["N_PERM_DEFAULT"]
DO_GROUP_DEFAULT = _constants["DO_GROUP_DEFAULT"]
GROUP_ONLY_DEFAULT = _constants["GROUP_ONLY_DEFAULT"]
BUILD_REPORTS_DEFAULT = _constants["BUILD_REPORTS_DEFAULT"]
BAND_COLORS = _constants["BAND_COLORS"]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))


def _save_fig(fig: matplotlib.figure.Figure, path_base: Path | str, formats: Tuple[str, ...] = SAVE_FORMATS) -> None:
    """Save figure to multiple formats at FIG_DPI and close.

    If path_base already has a suffix, it will be stripped and replaced.
    """
    base = Path(path_base)
    if base.suffix:
        base = base.with_suffix("")
    for ext in formats:
        fig.savefig(base.with_suffix(f".{ext}"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _logratio_to_pct(v):
    """Transform log10(power/baseline) to percent change.

    Accepts scalar or array-like. Returns values in percent.
    """
    return (np.power(10.0, v) - 1.0) * 100.0


def _pct_to_logratio(p):
    """Inverse transform: percent change to log10(power/baseline).

    Accepts scalar or array-like. Clips 1 + p/100 to a small positive
    minimum (1e-9) to avoid log10 of non-positive values, which previously
    caused runtime warnings when percent was <= -100.
    """
    p_arr = np.asarray(p, dtype=float)
    return np.log10(np.clip(1.0 + (p_arr / 100.0), 1e-9, None))


def _find_connectivity_path(subject: str, task: str) -> Path:
    """Find connectivity file path for subject and task."""
    # First try the expected parquet file
    parquet_path = DERIV_ROOT / f'sub-{subject}' / 'eeg' / 'connectivity_features.parquet'
    if parquet_path.exists():
        return parquet_path
    
    # Fall back to TSV file in features directory
    tsv_path = DERIV_ROOT / f'sub-{subject}' / 'eeg' / 'features' / 'features_connectivity.tsv'
    return tsv_path


def _find_clean_epochs_path(subject: str, task: str) -> Optional[Path]:
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
    p2 = DERIV_ROOT / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_proc-clean_epo.fif"
    if p2.exists():
        return p2
    subj_eeg_dir = DERIV_ROOT / f"sub-{subject}" / "eeg"
    if subj_eeg_dir.exists():
        cands = sorted(subj_eeg_dir.glob(f"sub-{subject}_task-{task}*epo.fif"))
        for c in cands:
            if "proc-clean" in c.name or "proc-cleaned" in c.name or "clean" in c.name:
                return c
        if cands:
            return cands[0]
    subj_dir = DERIV_ROOT / f"sub-{subject}"
    if subj_dir.exists():
        for c in sorted(subj_dir.rglob(f"sub-{subject}_task-{task}*epo.fif")):
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
    p = ebp.fpath or (BIDS_ROOT / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_events.tsv")
    try:
        return pd.read_csv(p, sep="\t") if p.exists() else None
    except Exception as e:
        print(f"Warning: failed to read events TSV at {p}: {e}")
        return None


def _align_events_to_epochs(events_df: Optional[pd.DataFrame], epochs: mne.Epochs) -> Optional[pd.DataFrame]:
    """Align events rows to epochs order using selection/sample heuristics.

    Mirrors logic from feature engineering to ensure behavioral vectors align to
    epochs even if there are row-count mismatches or ordering differences."""
    if events_df is None or len(events_df) == 0:
        return None
    if len(epochs) == 0:
        return pd.DataFrame()

    logger = logging.getLogger("align_events")

    # 1) Try epochs.selection to reindex
    if hasattr(epochs, "selection") and epochs.selection is not None:
        sel = epochs.selection
        try:
            if len(events_df) > int(np.max(sel)):
                logger.debug(f"Successfully aligned events using epochs.selection ({len(sel)} epochs)")
                return events_df.iloc[sel].reset_index(drop=True)
        except (IndexError, ValueError, TypeError) as e:
            logger.warning(f"Failed to align events using epochs.selection: {e}")
            
    # 2) Use sample column to reindex to epochs.events
    if "sample" in events_df.columns and isinstance(getattr(epochs, "events", None), np.ndarray):
        try:
            samples = epochs.events[:, 0]
            out = events_df.set_index("sample").reindex(samples)
            if len(out) == len(epochs) and not out.isna().all(axis=1).any():
                logger.debug(f"Successfully aligned events using sample column ({len(out)} epochs)")
                return out.reset_index()
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Failed to align events using sample column: {e}")
            
    # 3) Critical failure: cannot guarantee alignment
    logger.critical(f"CRITICAL: Unable to align events to epochs reliably. "
                   f"Events: {len(events_df)} rows, Epochs: {len(epochs)} epochs. "
                   f"This could result in completely invalid correlations due to trial misalignment.")
    raise ValueError(f"Cannot guarantee events-to-epochs alignment for reliable analysis. "
                    f"Events DataFrame ({len(events_df)} rows) cannot be reliably aligned to "
                    f"epochs ({len(epochs)} epochs). This is a critical failure that would "
                    f"invalidate all behavioral correlations.")


def _pick_first_column(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[str]:
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _features_dir(subject: str) -> Path:
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "features"


def _plots_dir(subject: str) -> Path:
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots" / "04_behavior_feature_analysis"


def _stats_dir(subject: str) -> Path:
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "stats"


def _group_stats_dir() -> Path:
    return DERIV_ROOT / "group" / "eeg" / "stats"


def _group_plots_dir() -> Path:
    return DERIV_ROOT / "group" / "eeg" / "plots"


def _load_features_and_targets(subject: str, task: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.Series, mne.Info]:
    """Load power features, optional connectivity features, targets, and channel info.

    Returns: (pow_df, conn_df_or_None, y, info)
    """
    feats_dir = _features_dir(subject)
    pow_path = feats_dir / "features_eeg_direct.tsv"
    conn_path = feats_dir / "features_connectivity.tsv"
    y_path = feats_dir / "target_vas_ratings.tsv"

    if not pow_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing features or targets for sub-{subject}. Expected at {feats_dir}")

    pow_df = pd.read_csv(pow_path, sep="\t")
    conn_df = pd.read_csv(conn_path, sep="\t") if conn_path.exists() else None
    y_df = pd.read_csv(y_path, sep="\t")
    # y column may vary; use the only column
    if y_df.shape[1] != 1:
        # Fallback: pick the first non-object column
        col = y_df.select_dtypes(exclude=["object"]).columns[0]
        y = pd.to_numeric(y_df[col], errors="coerce")
    else:
        y = pd.to_numeric(y_df.iloc[:, 0], errors="coerce")

    # Load epochs info for channel locations
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None:
        raise FileNotFoundError(f"Could not find cleaned epochs for sub-{subject}, task-{task}")
    epochs = mne.read_epochs(epo_path, preload=False, verbose=False)

    return pow_df, conn_df, y, epochs.info


def _setup_logging(subject: str) -> logging.Logger:
    """Set up logging with console and file handlers for behavior feature analysis."""
    logger = logging.getLogger(f"behavior_analysis_sub_{subject}")
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
    log_dir = _plots_dir(subject).parent / "logs"  # e.g., derivatives/sub-001/eeg/logs/
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME
    file_handler = logging.FileHandler(log_file, mode='w')  # Overwrite each run
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# ROI utilities (replicate minimal logic to avoid importing numbered module names)

def _roi_definitions() -> Dict[str, List[str]]:
    return {
        "Frontal": [r"^(Fpz|Fp[12]|AFz|AF[3-8]|Fz|F[1-8])$"],
        "Central": [r"^(Cz|C[1-6])$"],
        "Parietal": [r"^(Pz|P[1-8])$"],
        "Occipital": [r"^(Oz|O[12]|POz|PO[3-8])$"],
        "Temporal": [r"^(T7|T8|TP7|TP8|FT7|FT8)$"],
        "Sensorimotor": [r"^(FC[234]|FCz)$", r"^(C[234]|Cz)$", r"^(CP[234]|CPz)$"],
    }


def _build_rois(info: mne.Info) -> Dict[str, List[str]]:
    import re
    chs = info["ch_names"]
    roi_map: Dict[str, List[str]] = {}
    for roi, pats in _roi_definitions().items():
        found: List[str] = []
        for pat in pats:
            rx = re.compile(pat, flags=re.IGNORECASE)
            found.extend([ch for ch in chs if rx.match(ch)])
        # preserve order, deduplicate
        seen = set()
        ordered: List[str] = []
        for ch in chs:
            if ch in found and ch not in seen:
                seen.add(ch)
                ordered.append(ch)
        if ordered:
            roi_map[roi] = ordered
    return roi_map


# -----------------------------------------------------------------------------
# Psychometric curves
# -----------------------------------------------------------------------------

def plot_psychometrics(subject: str, task: str = TASK) -> None:
    logger = _setup_logging(subject)
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    events = _load_events_df(subject, task)
    if events is None or len(events) == 0:
        logger.warning(f"No events for psychometrics: sub-{subject}")
        return

    temp_col = _pick_first_column(events, PSYCH_TEMP_COLUMNS)
    rating_col = _pick_first_column(events, RATING_COLUMNS)
    pain_col = _pick_first_column(events, PAIN_BINARY_COLUMNS)

    if temp_col is None:
        logger.warning(f"Psychometrics: no temperature column found; skipping for sub-{subject}.")
        return

    # Clean columns
    temp = pd.to_numeric(events[temp_col], errors="coerce")
    # Plot continuous rating vs temperature if available
    if rating_col is not None:
        rating = pd.to_numeric(events[rating_col], errors="coerce")
        mask = temp.notna() & rating.notna()
        if mask.sum() >= 5:
            t = temp[mask]
            r = rating[mask]
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.regplot(x=t, y=r, scatter_kws={"s": 25, "alpha": 0.7}, line_kws={"color": "k"}, ax=ax)
            ax.set_xlabel(f"Temperature ({temp_col})")
            ax.set_ylabel(f"Rating ({rating_col})")
            ax.set_title("Rating vs Temperature")
            _save_fig(fig, plots_dir / "psychometric_rating_vs_temp")
            # Save Pearson and Spearman
            pr, pp = stats.pearsonr(t, r)
            sr, sp = stats.spearmanr(t, r, nan_policy="omit")
            pd.DataFrame({
                "metric": ["pearson_r", "pearson_p", "spearman_r", "spearman_p"],
                "value": [pr, pp, sr, sp],
            }).to_csv(stats_dir / "psychometric_rating_vs_temp_stats.tsv", sep="\t", index=False)

# -----------------------------------------------------------------------------
# Correlation: power features -> topographic maps
# -----------------------------------------------------------------------------

def _fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR.
    Returns: (reject_mask, crit_p)
    """
    p = np.asarray(pvals)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool), np.nan
    order = np.argsort(p)
    ranked = np.arange(1, n + 1)
    thresh = (ranked / n) * alpha
    passed = p[order] <= thresh
    if not np.any(passed):
        return np.zeros_like(p, dtype=bool), np.nan
    k_max = np.max(np.where(passed)[0])
    crit = p[order][k_max]
    reject = p <= crit
    return reject, float(crit)

def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg adjusted p-values (q-values).

    Returns an array of the same shape with BH q-values in [0, 1].
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.arange(1, n + 1, dtype=float)
    p_sorted = p[order]
    q_raw = p_sorted * n / ranks
    # Enforce monotonicity of q-values
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q



# -----------------------------------------------------------------------------
# Correlation: ROI-averaged power vs behavior (rating and temperature)
# -----------------------------------------------------------------------------

def correlate_power_roi_stats(
    subject: str,
    task: str = TASK,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    logger = _setup_logging(subject)
    logger.info(f"Starting ROI power correlation analysis for sub-{subject}")
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    # Initialize RNG if not provided
    if rng is None:
        rng = np.random.default_rng(42)

    # Load power features, target ratings, and sensor info
    pow_df, _conn_df, y, info = _load_features_and_targets(subject, task)
    y = pd.to_numeric(y, errors="coerce")

    # Load epochs for alignment and events for temperature
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None:
        logger.error(f"Could not find epochs for ROI correlations: sub-{subject}")
        return
    epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
    events = _load_events_df(subject, task)
    aligned_events = _align_events_to_epochs(events, epochs) if events is not None else None
    temp_series: Optional[pd.Series] = None
    if aligned_events is not None:
        temp_col = _pick_first_column(aligned_events, PSYCH_TEMP_COLUMNS)
        if temp_col is not None:
            temp_series = pd.to_numeric(aligned_events[temp_col], errors="coerce")

    # Build ROIs present in this recording
    roi_map = _build_rois(info)

    # Prepare outputs
    recs_rating: List[Dict[str, object]] = []
    recs_temp: List[Dict[str, object]] = []

    # Helper: build multi-covariate design matrix Z from aligned events
    def _build_Z(df_events: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df_events is None:
            return None
        # Default covariates if not provided: temperature and trial index if available
        covars = list(partial_covars) if partial_covars is not None else []
        if not covars:
            # Try to infer a reasonable default set
            tcol = _pick_first_column(df_events, PSYCH_TEMP_COLUMNS)
            if tcol is not None:
                covars.append(tcol)
            for c in ["trial", "trial_number", "trial_index", "run", "block"]:
                if c in df_events.columns:
                    covars.append(c)
                    break
        if not covars:
            return None
        Z = pd.DataFrame()
        for c in covars:
            if c in df_events.columns:
                Z[c] = pd.to_numeric(df_events[c], errors="coerce")
        return Z if not Z.empty else None

    Z_df_full = _build_Z(aligned_events)
    # For temperature targets, drop the temperature column from Z to avoid conditioning on the outcome variable
    Z_df_temp = None
    if Z_df_full is not None:
        try:
            Z_df_temp = Z_df_full.drop(columns=[temp_col], errors="ignore") if temp_col else Z_df_full.copy()
            if Z_df_temp.shape[1] == 0:
                Z_df_temp = None
        except (KeyError, AttributeError, ValueError):
            Z_df_temp = Z_df_full

    def _partial_corr_xy_given_Z(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str) -> Tuple[float, float, int]:
        # Align and drop missing jointly
        df_full = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1)
        df = df_full.dropna()
        if len(df_full) > len(df):
            logger.warning(f"Partial correlation dropped {len(df_full) - len(df)} rows due to missing data (kept {len(df)}/{len(df_full)})")
        if len(df) < 5 or df["y"].nunique() <= 1:
            return np.nan, np.nan, 0
        if method == "spearman":
            xr = stats.rankdata(df["x"].to_numpy())
            yr = stats.rankdata(df["y"].to_numpy())
            Zr = np.column_stack([stats.rankdata(df[c].to_numpy()) for c in Z.columns]) if len(Z.columns) else np.empty((len(df), 0))
            Xd = np.column_stack([np.ones(len(df)), Zr])
            bx = np.linalg.lstsq(Xd, xr, rcond=None)[0]
            by = np.linalg.lstsq(Xd, yr, rcond=None)[0]
            x_res = xr - Xd.dot(bx)
            y_res = yr - Xd.dot(by)
            r_p, p_p = stats.pearsonr(x_res, y_res)
        else:
            Xd = np.column_stack([np.ones(len(df)), df[Z.columns].to_numpy()])
            bx = np.linalg.lstsq(Xd, df["x"].to_numpy(), rcond=None)[0]
            by = np.linalg.lstsq(Xd, df["y"].to_numpy(), rcond=None)[0]
            x_res = df["x"].to_numpy() - Xd.dot(bx)
            y_res = df["y"].to_numpy() - Xd.dot(by)
            r_p, p_p = stats.pearsonr(x_res, y_res)
        return float(r_p), float(p_p), int(len(df))

    def _fisher_ci(r: float, n: int) -> Tuple[float, float]:
        if not np.isfinite(r) or n < 4:
            return np.nan, np.nan
        r = float(np.clip(r, -0.999999, 0.999999))
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        z_lo = z - 1.96 * se
        z_hi = z + 1.96 * se
        return float(np.tanh(z_lo)), float(np.tanh(z_hi))

    def _perm_pval_simple(x: pd.Series, y: pd.Series, method: str, n_perm: int, rng: np.random.Generator) -> float:
        df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(df) < 5:
            return np.nan
        if method == "spearman" and df["y"].nunique() > 5:
            obs, _ = stats.spearmanr(df["x"], df["y"], nan_policy="omit")
        else:
            obs, _ = stats.pearsonr(df["x"], df["y"])
        ge = 1
        y_vals = df["y"].to_numpy()
        for _ in range(int(n_perm)):
            y_pi = y_vals[rng.permutation(len(y_vals))]
            if method == "spearman" and df["y"].nunique() > 5:
                rp, _ = stats.spearmanr(df["x"], y_pi, nan_policy="omit")
            else:
                rp, _ = stats.pearsonr(df["x"], y_pi)
            if np.abs(rp) >= np.abs(obs) - 1e-12:
                ge += 1
        return ge / (int(n_perm) + 1)

    def _perm_pval_partial_FL(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str, n_perm: int, rng: np.random.Generator) -> float:
        df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
        if len(df) < 5:
            return np.nan
        if method == "spearman":
            xr = stats.rankdata(df["x"].to_numpy())
            yr = stats.rankdata(df["y"].to_numpy())
            Zr = np.column_stack([stats.rankdata(df[c].to_numpy()) for c in Z.columns]) if len(Z.columns) else np.empty((len(df), 0))
            Xd = np.column_stack([np.ones(len(df)), Zr])
            bx = np.linalg.lstsq(Xd, xr, rcond=None)[0]
            by = np.linalg.lstsq(Xd, yr, rcond=None)[0]
            rx = xr - Xd.dot(bx)
            ry = yr - Xd.dot(by)
            obs, _ = stats.pearsonr(rx, ry)
        else:
            Xd = np.column_stack([np.ones(len(df)), df[Z.columns].to_numpy()])
            bx = np.linalg.lstsq(Xd, df["x"].to_numpy(), rcond=None)[0]
            by = np.linalg.lstsq(Xd, df["y"].to_numpy(), rcond=None)[0]
            rx = df["x"].to_numpy() - Xd.dot(bx)
            ry = df["y"].to_numpy() - Xd.dot(by)
            obs, _ = stats.pearsonr(rx, ry)
        ge = 1
        for _ in range(int(n_perm)):
            ry_pi = ry[rng.permutation(len(ry))]
            if method == "spearman":
                rp, _ = stats.pearsonr(rx, ry_pi)
            else:
                rp, _ = stats.pearsonr(rx, ry_pi)
            if np.abs(rp) >= np.abs(obs) - 1e-12:
                ge += 1
        return ge / (int(n_perm) + 1)

    # For each band and ROI, average channels and correlate with behavior
    for band in POWER_BANDS_TO_USE:
        # Identify available columns for this band
        band_cols_available = {c for c in pow_df.columns if c.startswith(f"pow_{band}_")}
        if not band_cols_available:
            continue
        band_rng = FEATURES_FREQ_BANDS.get(band)
        band_range_str = f"{band_rng[0]:g}\u2013{band_rng[1]:g} Hz" if band_rng is not None else ""

        for roi, chs in roi_map.items():
            roi_cols = [f"pow_{band}_{ch}" for ch in chs if f"pow_{band}_{ch}" in band_cols_available]
            if not roi_cols:
                continue

            # Convert columns to numeric individually, then average across ROI channels
            roi_frame = pow_df[roi_cols].apply(pd.to_numeric, errors="coerce")
            roi_vals = roi_frame.mean(axis=1)

            # Align lengths with rating target
            n_len = min(len(roi_vals), len(y))
            x = roi_vals.iloc[:n_len]
            y_r = y.iloc[:n_len]
            m = x.notna() & y_r.notna()
            n_eff = int(m.sum())
            if n_eff >= 5:
                if use_spearman and y_r.nunique() > 5:
                    r, p = stats.spearmanr(x[m], y_r[m], nan_policy="omit")
                    method = "spearman"
                else:
                    r, p = stats.pearsonr(x[m], y_r[m])
                    method = "pearson"

                # Partial correlation given multi-covariates (if available)
                r_part = np.nan
                p_part = np.nan
                n_part = 0
                r_part_temp = np.nan
                p_part_temp = np.nan
                n_part_temp = 0
                if Z_df_full is not None and len(Z_df_full) > 0:
                    n_len_pt = min(len(roi_vals), len(y), len(Z_df_full))
                    r_part, p_part, n_part = _partial_corr_xy_given_Z(
                        roi_vals.iloc[:n_len_pt], y.iloc[:n_len_pt], Z_df_full.iloc[:n_len_pt], method
                    )
                # Back-compat: partial only for temperature (if available)
                if temp_series is not None and len(temp_series) > 0:
                    n_len_tmp = min(len(roi_vals), len(y), len(temp_series))
                    df_tmp = pd.DataFrame({"temp": temp_series.iloc[:n_len_tmp]})
                    r_part_temp, p_part_temp, n_part_temp = _partial_corr_xy_given_Z(
                        roi_vals.iloc[:n_len_tmp], y.iloc[:n_len_tmp], df_tmp, method
                    )

                # CIs for r: Fisher z for Pearson; bootstrap for Spearman
                ci_low = np.nan
                ci_high = np.nan
                if method == "pearson" and n_eff >= 4:
                    ci_low, ci_high = _fisher_ci(r, n_eff)
                elif bootstrap and n_eff >= 5:
                    idx = np.where(m.to_numpy())[0]
                    boots: List[float] = []
                    for _ in range(int(bootstrap)):
                        bidx = rng.choice(idx, size=len(idx), replace=True)
                        xb = x.iloc[bidx]
                        yb = y_r.iloc[bidx]
                        if method == "spearman" and yb.nunique() > 5:
                            rb, _ = stats.spearmanr(xb, yb, nan_policy="omit")
                        else:
                            rb, _ = stats.pearsonr(xb, yb)
                        boots.append(rb)
                    if boots:
                        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

                # Permutation p-values (simple and partial)
                p_perm = np.nan
                p_partial_perm = np.nan
                p_partial_given_temp_perm = np.nan
                if n_perm and n_eff >= 5:
                    p_perm = _perm_pval_simple(x, y_r, method, int(n_perm), rng)
                    if Z_df_full is not None and len(Z_df_full) > 0:
                        n_len_pt = min(len(x), len(y_r), len(Z_df_full))
                        p_partial_perm = _perm_pval_partial_FL(
                            x.iloc[:n_len_pt], y_r.iloc[:n_len_pt], Z_df_full.iloc[:n_len_pt], method, int(n_perm), rng
                        )
                    if temp_series is not None and len(temp_series) > 0:
                        n_len_tmp = min(len(x), len(y_r), len(temp_series))
                        df_tmp = pd.DataFrame({"temp": temp_series.iloc[:n_len_tmp]})
                        p_partial_given_temp_perm = _perm_pval_partial_FL(
                            x.iloc[:n_len_tmp], y_r.iloc[:n_len_tmp], df_tmp.iloc[:n_len_tmp], method, int(n_perm), rng
                        )

                recs_rating.append({
                    "roi": roi,
                    "band": band,
                    "band_range": band_range_str,
                    "r": float(r),
                    "p": float(p),
                    "n": n_eff,
                    "method": method,
                    "r_ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                    "r_ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                    # General partials
                    "r_partial": float(r_part) if np.isfinite(r_part) else np.nan,
                    "p_partial": float(p_part) if np.isfinite(p_part) else np.nan,
                    "n_partial": n_part,
                    "partial_covars": ",".join(Z_df_full.columns.tolist()) if Z_df_full is not None else "",
                    # Back-compat single-covariate (temperature)
                    "r_partial_given_temp": float(r_part_temp) if np.isfinite(r_part_temp) else np.nan,
                    "p_partial_given_temp": float(p_part_temp) if np.isfinite(p_part_temp) else np.nan,
                    "n_partial_given_temp": n_part_temp,
                    # Permutation p-values
                    "p_perm": float(p_perm) if np.isfinite(p_perm) else np.nan,
                    "p_partial_perm": float(p_partial_perm) if np.isfinite(p_partial_perm) else np.nan,
                    "p_partial_given_temp_perm": float(p_partial_given_temp_perm) if np.isfinite(p_partial_given_temp_perm) else np.nan,
                    "n_perm": int(n_perm),
                })

            # Temperature correlation if available
            if temp_series is not None and len(temp_series) > 0:
                n_len_t = min(len(roi_vals), len(temp_series))
                x2 = roi_vals.iloc[:n_len_t]
                t2 = temp_series.iloc[:n_len_t]
                m2 = x2.notna() & t2.notna()
                n_eff2 = int(m2.sum())
                if n_eff2 >= 5:
                    if use_spearman and t2.nunique() > 5:
                        r2, p2 = stats.spearmanr(x2[m2], t2[m2], nan_policy="omit")
                        method2 = "spearman"
                    else:
                        r2, p2 = stats.pearsonr(x2[m2], t2[m2])
                        method2 = "pearson"
                    # CI: Fisher for Pearson, bootstrap for Spearman
                    ci2_low = np.nan
                    ci2_high = np.nan
                    if method2 == "pearson" and n_eff2 >= 4:
                        ci2_low, ci2_high = _fisher_ci(r2, n_eff2)
                    elif bootstrap and n_eff2 >= 5:
                        idx2 = np.where(m2.to_numpy())[0]
                        boots2: List[float] = []
                        for _ in range(int(bootstrap)):
                            bidx2 = rng.choice(idx2, size=len(idx2), replace=True)
                            xb = x2.iloc[bidx2]
                            tb = t2.iloc[bidx2]
                            if method2 == "spearman" and tb.nunique() > 5:
                                rb, _ = stats.spearmanr(xb, tb, nan_policy="omit")
                            else:
                                rb, _ = stats.pearsonr(xb, tb)
                            boots2.append(rb)
                        if boots2:
                            ci2_low, ci2_high = np.percentile(boots2, [2.5, 97.5])

                    # Partial correlation controlling covariates excluding temperature itself
                    r2_part = np.nan
                    p2_part = np.nan
                    n2_part = 0
                    if Z_df_temp is not None and len(Z_df_temp) > 0:
                        n_len_pt2 = min(len(x2), len(t2), len(Z_df_temp))
                        r2_part, p2_part, n2_part = _partial_corr_xy_given_Z(
                            x2.iloc[:n_len_pt2], t2.iloc[:n_len_pt2], Z_df_temp.iloc[:n_len_pt2], method2
                        )

                    # Permutation p-values (simple and partial)
                    p2_perm = np.nan
                    p2_partial_perm = np.nan
                    if n_perm and n_eff2 >= 5:
                        p2_perm = _perm_pval_simple(x2, t2, method2, int(n_perm), rng)
                        if Z_df_temp is not None and len(Z_df_temp) > 0:
                            n_len_pt2 = min(len(x2), len(t2), len(Z_df_temp))
                            p2_partial_perm = _perm_pval_partial_FL(
                                x2.iloc[:n_len_pt2], t2.iloc[:n_len_pt2], Z_df_temp.iloc[:n_len_pt2], method2, int(n_perm), rng
                            )

                    recs_temp.append({
                        "roi": roi,
                        "band": band,
                        "band_range": band_range_str,
                        "r": float(r2),
                        "p": float(p2),
                        "n": n_eff2,
                        "method": method2,
                        "r_ci_low": float(ci2_low) if np.isfinite(ci2_low) else np.nan,
                        "r_ci_high": float(ci2_high) if np.isfinite(ci2_high) else np.nan,
                        "r_partial": float(r2_part) if np.isfinite(r2_part) else np.nan,
                        "p_partial": float(p2_part) if np.isfinite(p2_part) else np.nan,
                        "n_partial": n2_part,
                        "partial_covars": ",".join(Z_df_temp.columns.tolist()) if Z_df_temp is not None else "",
                        "p_perm": float(p2_perm) if np.isfinite(p2_perm) else np.nan,
                        "p_partial_perm": float(p2_partial_perm) if np.isfinite(p2_partial_perm) else np.nan,
                        "n_perm": int(n_perm),
                    })

    # Save TSVs with FDR across all ROI-band tests per target
    if recs_rating:
        df_r = pd.DataFrame(recs_rating)
        pvec = df_r["p_perm"].to_numpy() if "p_perm" in df_r.columns and np.isfinite(df_r["p_perm"]).any() else df_r["p"].to_numpy()
        rej, crit = _fdr_bh(pvec, alpha=0.05)
        df_r["fdr_reject"] = rej
        df_r["fdr_crit_p"] = crit
        df_r.to_csv(stats_dir / "corr_stats_pow_roi_vs_rating.tsv", sep="\t", index=False)

    if recs_temp:
        df_t = pd.DataFrame(recs_temp)
        pvec_t = df_t["p_perm"].to_numpy() if "p_perm" in df_t.columns and np.isfinite(df_t["p_perm"]).any() else df_t["p"].to_numpy()
        rej_t, crit_t = _fdr_bh(pvec_t, alpha=0.05)
        df_t["fdr_reject"] = rej_t
        df_t["fdr_crit_p"] = crit_t
        df_t.to_csv(stats_dir / "corr_stats_pow_roi_vs_temp.tsv", sep="\t", index=False)

# -----------------------------------------------------------------------------
# Visualization: ROI-averaged power vs behavior scatter plots
# -----------------------------------------------------------------------------

def plot_power_roi_scatter(
    subject: str,
    task: str = TASK,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Per-ROI scatter plots of band power vs behavior with regression and stats.

    - Plots ROI-averaged power vs target rating for each band and ROI.
    - Optionally also plots vs temperature if aligned behavioral events exist.
    - Annotates r, p, n and (if available) partial r/p given covariates.
    - 95% CI of r shown: Fisher z for Pearson; optional bootstrap for Spearman if
      ``bootstrap_ci > 0`` (otherwise Spearman CI omitted).
    - Regression line CI band rendered via seaborn.
    - Significance-aware aesthetics: regression line colored by p<0.05.
    - Saves PNG and SVG via _save_fig().
    """
    logger = _setup_logging(subject)
    logger.info(f"Starting ROI power scatter plotting for sub-{subject}")
    plots_dir = _plots_dir(subject)
    _ensure_dir(plots_dir)

    # Initialize RNG if not provided
    if rng is None:
        rng = np.random.default_rng(42)

    # Load features/targets and sensor info
    pow_df, _conn_df, y, info = _load_features_and_targets(subject, task)
    y = pd.to_numeric(y, errors="coerce")

    # Load epochs/events to align covariates and temperature
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None:
        logger.error(f"Could not find epochs for ROI scatter plots: sub-{subject}")
        return
    epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
    events = _load_events_df(subject, task)
    aligned_events = _align_events_to_epochs(events, epochs) if events is not None else None

    temp_series: Optional[pd.Series] = None
    temp_col: Optional[str] = None
    if aligned_events is not None:
        tcol = _pick_first_column(aligned_events, PSYCH_TEMP_COLUMNS)
        if tcol is not None:
            temp_col = tcol
            temp_series = pd.to_numeric(aligned_events[tcol], errors="coerce")

    # Helper to build covariate design matrix Z
    def _build_Z(df_events: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df_events is None:
            return None
        covars = list(partial_covars) if partial_covars is not None else []
        if not covars:
            # Reasonable defaults: temperature and a trial index if present
            tcol_loc = _pick_first_column(df_events, PSYCH_TEMP_COLUMNS)
            if tcol_loc is not None:
                covars.append(tcol_loc)
            for c in ["trial", "trial_number", "trial_index", "run", "block"]:
                if c in df_events.columns:
                    covars.append(c)
                    break
        if not covars:
            return None
        Z = pd.DataFrame()
        for c in covars:
            if c in df_events.columns:
                Z[c] = pd.to_numeric(df_events[c], errors="coerce")
        return Z if not Z.empty else None

    Z_df_full = _build_Z(aligned_events)
    Z_df_temp = None
    if Z_df_full is not None:
        try:
            Z_df_temp = Z_df_full.drop(columns=[temp_col], errors="ignore") if temp_col else Z_df_full.copy()
            if Z_df_temp.shape[1] == 0:
                Z_df_temp = None
        except (KeyError, AttributeError, ValueError):
            Z_df_temp = Z_df_full

    # Helpers: significance formatting, CIs, bootstrap CIs, and stats textbox
    def _p_to_stars(p: float) -> str:
        if not np.isfinite(p):
            return ""
        if p < 1e-3:
            return "***"
        if p < 1e-2:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    def _sig_color(p: float) -> str:
        return "#C42847" if (np.isfinite(p) and p < 0.05) else "#666666"

    def _fisher_ci_r(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
        if not np.isfinite(r) or n < 4:
            return np.nan, np.nan
        r = float(np.clip(r, -0.999999, 0.999999))
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        zcrit = float(stats.norm.ppf(1 - alpha / 2.0))
        lo = z - zcrit * se
        hi = z + zcrit * se
        return float(np.tanh(lo)), float(np.tanh(hi))

    def _bootstrap_corr_ci(
        x: pd.Series,
        y: pd.Series,
        method: str,
        n_boot: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, float]:
        dfb = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if n_boot <= 0 or len(dfb) < 5:
            return np.nan, np.nan
        if rng is None:
            rng = np.random.default_rng(42)
        Xv = dfb["x"].to_numpy()
        Yv = dfb["y"].to_numpy()
        N = len(dfb)
        vals: List[float] = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, N, size=N)
            xb = Xv[idx]
            yb = Yv[idx]
            if method == "spearman" and len(np.unique(yb)) > 5:
                rb, _ = stats.spearmanr(xb, yb, nan_policy="omit")
            else:
                rb, _ = stats.pearsonr(xb, yb)
            if np.isfinite(rb):
                vals.append(float(rb))
        if not vals:
            return np.nan, np.nan
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    def _fmt_stats_text(
        r: float,
        p: float,
        n: int,
        method_code: str,
        ci: Optional[Tuple[float, float]] = None,
        r_part: float = np.nan,
        p_part: float = np.nan,
        n_part: int = 0,
        zcols: Optional[List[str]] = None,
    ) -> str:
        label = "ρ" if method_code == "spearman" else "r"
        stars = _p_to_stars(p)
        ci_str = ""
        if ci is not None and np.all(np.isfinite(ci)):
            ci_str = f" [{ci[0]:.2f}, {ci[1]:.2f}]"
        line1 = f"{label}={r:.2f}{ci_str}, p={p:.3g} {stars}, n={n}"
        line2 = ""
        if np.isfinite(r_part) and np.isfinite(p_part) and n_part:
            stars2 = _p_to_stars(p_part)
            line2 = f"Partial {label}={r_part:.2f}, p={p_part:.3g} {stars2}, n={n_part}"
            if zcols:
                zlab = (", ".join(zcols[:3]) + ("…" if len(zcols) > 3 else "")) if zcols else ""
                if zlab:
                    line2 += f"\nControlling for: {zlab}"
        return line1 + ("\n" + line2 if line2 else "")

    # Partial correlation helper
    def _partial_corr_xy_given_Z(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str) -> Tuple[float, float, int]:
        df_full = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1)
        df = df_full.dropna()
        if len(df_full) > len(df):
            logger.warning(f"Partial correlation dropped {len(df_full) - len(df)} rows due to missing data (kept {len(df)}/{len(df_full)})")
        if len(df) < 5 or df["y"].nunique() <= 1:
            return np.nan, np.nan, 0
        if method == "spearman":
            xr = stats.rankdata(df["x"].to_numpy())
            yr = stats.rankdata(df["y"].to_numpy())
            Zr = np.column_stack([stats.rankdata(df[c].to_numpy()) for c in Z.columns]) if len(Z.columns) else np.empty((len(df), 0))
            Xd = np.column_stack([np.ones(len(df)), Zr])
            bx = np.linalg.lstsq(Xd, xr, rcond=None)[0]
            by = np.linalg.lstsq(Xd, yr, rcond=None)[0]
            x_res = xr - Xd.dot(bx)
            y_res = yr - Xd.dot(by)
            r_p, p_p = stats.pearsonr(x_res, y_res)
        else:
            Xd = np.column_stack([np.ones(len(df)), df[Z.columns].to_numpy()])
            bx = np.linalg.lstsq(Xd, df["x"].to_numpy(), rcond=None)[0]
            by = np.linalg.lstsq(Xd, df["y"].to_numpy(), rcond=None)[0]
            x_res = df["x"].to_numpy() - Xd.dot(bx)
            y_res = df["y"].to_numpy() - Xd.dot(by)
            r_p, p_p = stats.pearsonr(x_res, y_res)
        return float(r_p), float(p_p), int(len(df))

    def _partial_residuals_xy_given_Z(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str) -> Tuple[pd.Series, pd.Series, int]:
        df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
        if len(df) < 5 or df["y"].nunique() <= 1:
            return pd.Series(dtype=float), pd.Series(dtype=float), 0
        if method == "spearman":
            xr = stats.rankdata(df["x"].to_numpy())
            yr = stats.rankdata(df["y"].to_numpy())
            Zr = np.column_stack([stats.rankdata(df[c].to_numpy()) for c in Z.columns]) if len(Z.columns) else np.empty((len(df), 0))
            Xd = np.column_stack([np.ones(len(df)), Zr])
            bx = np.linalg.lstsq(Xd, xr, rcond=None)[0]
            by = np.linalg.lstsq(Xd, yr, rcond=None)[0]
            x_res = xr - Xd.dot(bx)
            y_res = yr - Xd.dot(by)
        else:
            Xd = np.column_stack([np.ones(len(df)), df[Z.columns].to_numpy()])
            bx = np.linalg.lstsq(Xd, df["x"].to_numpy(), rcond=None)[0]
            by = np.linalg.lstsq(Xd, df["y"].to_numpy(), rcond=None)[0]
            x_res = df["x"].to_numpy() - Xd.dot(bx)
            y_res = df["y"].to_numpy() - Xd.dot(by)
        return pd.Series(x_res), pd.Series(y_res), int(len(df))

    def _fmt_stats_text_partial(
        method_code: str,
        r_part: float,
        p_part: float,
        n_part: int,
        zcols: Optional[List[str]] = None,
        ci_part: Optional[Tuple[float, float]] = None,
    ) -> str:
        label = "ρ" if method_code == "spearman" else "r"
        stars2 = _p_to_stars(p_part) if np.isfinite(p_part) else ""
        ci_str = ""
        if ci_part is not None and np.all(np.isfinite(ci_part)):
            ci_str = f" [{ci_part[0]:.2f}, {ci_part[1]:.2f}]"
        line1 = f"Partial {label}={r_part:.2f}{ci_str}, p={p_part:.3g} {stars2}, n={n_part}"
        line2 = ""
        if zcols:
            zlab = (", ".join(zcols[:3]) + ("…" if len(zcols) > 3 else "")) if zcols else ""
            if zlab:
                line2 = f"Controlling for: {zlab}"
        return line1 + ("\n" + line2 if line2 else "")

    # Figure helpers
    def _new_fig_axes() -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        # Kept for backward-compat but not used for new figures
        fig = plt.figure(figsize=(6.8, 3.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[4.8, 2.2], wspace=0.45)
        ax = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1])
        ax_info.axis("off")
        return fig, ax, ax_info

    def _new_single_ax() -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        return fig, ax

    def _generate_correlation_scatter(
        x_data: pd.Series,
        y_data: pd.Series,
        x_label: str,
        y_label: str,
        title_prefix: str,
        band_color: str,
        output_path: Path,
        method_code: str = "spearman",
        Z_covars: Optional[pd.DataFrame] = None,
        covar_names: Optional[List[str]] = None,
        bootstrap_ci: int = 0,
        rng: Optional[np.random.Generator] = None,
        is_partial_residuals: bool = False,
    ) -> None:
        """Generic helper to generate correlation scatter plots with stats.
        
        Args:
            x_data: X variable (power values)
            y_data: Y variable (rating/temperature)
            x_label: X-axis label
            y_label: Y-axis label  
            title_prefix: Title prefix (e.g., "Alpha power vs rating")
            band_color: Color for scatter points
            output_path: Output file path (without extension)
            method_code: "spearman" or "pearson"
            Z_covars: Covariate matrix for partial correlation
            covar_names: Names of covariates for annotations
            bootstrap_ci: Number of bootstrap samples for CI (0 = Fisher CI)
            rng: Random number generator
            is_partial_residuals: Whether x_data/y_data are already residuals
        """
        # Align and filter data
        n_len = min(len(x_data), len(y_data))
        x = x_data.iloc[:n_len] if hasattr(x_data, 'iloc') else x_data[:n_len]
        y = y_data.iloc[:n_len] if hasattr(y_data, 'iloc') else y_data[:n_len]
        
        if is_partial_residuals:
            # Data is already residuals, use as-is
            m = pd.Series([True] * len(x), index=x.index if hasattr(x, 'index') else range(len(x)))
            n_eff = len(x)
            x_plot = x
            y_plot = y
        else:
            # Filter missing values
            m = x.notna() & y.notna()
            n_eff = int(m.sum())
            x_plot = x
            y_plot = y
            
        if n_eff < 5:
            return
            
        # Calculate correlation stats
        if is_partial_residuals:
            # For residuals, just compute correlation directly
            if method_code == "spearman":
                r, p = stats.spearmanr(x, y, nan_policy="omit")
            else:
                r, p = stats.pearsonr(x, y)
            r_part, p_part, n_part = np.nan, np.nan, 0
        else:
            # Regular correlation
            if method_code == "spearman" and y.nunique() > 5:
                r, p = stats.spearmanr(x[m], y[m], nan_policy="omit")
            else:
                r, p = stats.pearsonr(x[m], y[m])
                
            # Optional partial correlation
            r_part, p_part, n_part = np.nan, np.nan, 0
            if Z_covars is not None and len(Z_covars) > 0:
                n_len_pt = min(len(x), len(y), len(Z_covars))
                r_part, p_part, n_part = _partial_corr_xy_given_Z(
                    x.iloc[:n_len_pt], y.iloc[:n_len_pt], Z_covars.iloc[:n_len_pt], method_code
                )

        # Calculate confidence intervals
        if method_code == "pearson" and n_eff >= 4:
            ci = _fisher_ci_r(r, n_eff)
        elif method_code == "spearman" and bootstrap_ci > 0:
            if is_partial_residuals:
                ci = _bootstrap_corr_ci(x, y, method_code, n_boot=int(bootstrap_ci), rng=rng)
            else:
                ci = _bootstrap_corr_ci(x[m], y[m], method_code, n_boot=int(bootstrap_ci), rng=rng)
        else:
            ci = (np.nan, np.nan)

        # Create and style figure
        fig, ax = _new_single_ax()
        line_color = _sig_color(p)
        
        sns.regplot(
            x=x_plot,
            y=y_plot,
            ax=ax,
            ci=95,
            scatter_kws={"s": 26, "alpha": 0.75, "color": band_color, "edgecolor": "white", "linewidths": 0.2},
            line_kws={"color": line_color, "lw": 1.2},
        )
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Add secondary x-axis for non-residual power plots
        if not is_partial_residuals and "log10(power" in x_label:
            try:
                secax_top = ax.secondary_xaxis('top', functions=(_logratio_to_pct, _pct_to_logratio))
                secax_top.set_xlabel("Percent change from baseline (%)")
                secax_top.xaxis.set_major_locator(MaxNLocator(nbins=5))
            except (AttributeError, TypeError, ValueError):
                pass
        elif is_partial_residuals and method_code == "pearson" and "residuals of log10(power" in x_label:
            try:
                secax_top = ax.secondary_xaxis('top', functions=(_logratio_to_pct, _pct_to_logratio))
                secax_top.set_xlabel("Percent change from baseline (%)")
                secax_top.xaxis.set_major_locator(MaxNLocator(nbins=5))
            except (AttributeError, TypeError, ValueError):
                pass
        
        # Format title with stats
        label = "ρ" if method_code == "spearman" else "r"
        if is_partial_residuals:
            title_stats = f"{label}={r:.2f}, p={p:.3g}, n={n_eff}"
        else:
            title_stats = f"{label}={r:.2f}, p={p:.3g}, n={n_eff}"
        ax.set_title(f"{title_prefix} ({title_stats})", fontsize=10)
        
        # Apply styling
        if "Rating" in y_label and not is_partial_residuals:
            try:
                if y.min() >= 0 and y.max() <= 100:
                    ax.set_ylim(0, 100)
            except (AttributeError, TypeError, ValueError):
                pass
        elif "Temperature" in y_label and not is_partial_residuals:
            # Align y-ticks to actual temperature values
            unique_temps = np.sort(np.unique(np.asarray(y)))
            if np.allclose(unique_temps, unique_temps.astype(int)):
                unique_temps = unique_temps.astype(int)
            if len(unique_temps) >= 2:
                ax.set_yticks(unique_temps.tolist())
                ymin = unique_temps.min() - 0.5
                ymax = unique_temps.max() + 0.5
                ax.set_ylim(ymin, ymax)
        elif is_partial_residuals:
            # Reduce y-axis tick clutter for residuals
            try:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            except (AttributeError, TypeError, ValueError):
                pass
                
        ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.8)
        try:
            sns.despine(ax=ax)
        except (AttributeError, TypeError):
            pass
            
        fig.tight_layout()
        _save_fig(fig, output_path)

    # Build ROI map and iterate
    roi_map = _build_rois(info)
    for band in POWER_BANDS_TO_USE:
        band_cols = {c for c in pow_df.columns if c.startswith(f"pow_{band}_")}
        if not band_cols:
            continue
        band_rng = FEATURES_FREQ_BANDS.get(band)
        band_title = band.capitalize()
        band_color = BAND_COLORS.get(band, "#4C4C4C")

        # --- Overall (all sensors) scatter ---
        try:
            overall_vals = pow_df[list(band_cols)].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        except (KeyError, ValueError, TypeError):
            overall_vals = pd.Series(np.nan, index=pow_df.index)

        # Determine correlation method
        do_spear = bool(use_spearman and y.nunique() > 5)
        method_code = "spearman" if do_spear else "pearson"
        covar_names = list(Z_df_full.columns) if Z_df_full is not None else None

        # Rating target scatter (overall)
        _generate_correlation_scatter(
            x_data=overall_vals,
            y_data=y,
            x_label="log10(power/baseline [-5–0 s])",
            y_label="Rating",
            title_prefix=f"{band_title} power vs rating — Overall",
            band_color=band_color,
            output_path=plots_dir / f"scatter_pow_overall_{_sanitize(band)}_vs_rating",
            method_code=method_code,
            Z_covars=Z_df_full,
            covar_names=covar_names,
            bootstrap_ci=bootstrap_ci,
            rng=rng,
        )
        
        # Separate partial-residuals figure if covariates available
        if Z_df_full is not None and len(Z_df_full) > 0:
            n_len_pt = min(len(overall_vals), len(y), len(Z_df_full))
            x_part = overall_vals.iloc[:n_len_pt]
            y_part = y.iloc[:n_len_pt] 
            Z_part = Z_df_full.iloc[:n_len_pt]
            x_res_sr, y_res_sr, n_res = _partial_residuals_xy_given_Z(x_part, y_part, Z_part, method_code)
            if n_res >= 5:
                residual_xlabel = "Partial residuals (ranked) of log10(power/baseline)" if method_code == "spearman" else "Partial residuals of log10(power/baseline)"
                residual_ylabel = "Partial residuals (ranked) of rating" if method_code == "spearman" else "Partial residuals of rating"
                
                _generate_correlation_scatter(
                    x_data=x_res_sr,
                    y_data=y_res_sr,
                    x_label=residual_xlabel,
                    y_label=residual_ylabel,
                    title_prefix=f"Partial residuals — {band_title} vs rating — Overall",
                    band_color=band_color,
                    output_path=plots_dir / f"scatter_pow_overall_{_sanitize(band)}_vs_rating_partial",
                    method_code=method_code,
                    bootstrap_ci=bootstrap_ci,
                    rng=rng,
                    is_partial_residuals=True,
                )

        # Temperature target scatter (overall)
        if do_temp and temp_series is not None and len(temp_series) > 0:
            # Determine method for temperature (may be different if discrete)
            do_spear_t = bool(use_spearman and temp_series.nunique() > 5)
            method2_code = "spearman" if do_spear_t else "pearson"
            covar_names_temp = list(Z_df_temp.columns) if Z_df_temp is not None else None
            
            _generate_correlation_scatter(
                x_data=overall_vals,
                y_data=temp_series,
                x_label="log10(power/baseline [-5–0 s])",
                y_label="Temperature (°C)",
                title_prefix=f"{band_title} power vs temperature — Overall",
                band_color=band_color,
                output_path=plots_dir / f"scatter_pow_overall_{_sanitize(band)}_vs_temp",
                method_code=method2_code,
                Z_covars=Z_df_temp,
                covar_names=covar_names_temp,
                bootstrap_ci=bootstrap_ci,
                rng=rng,
            )
            
            # Separate partial-residuals figure if covariates available
            if Z_df_temp is not None and len(Z_df_temp) > 0:
                n_len_pt2 = min(len(overall_vals), len(temp_series), len(Z_df_temp))
                x_part2 = overall_vals.iloc[:n_len_pt2]
                y_part2 = temp_series.iloc[:n_len_pt2]
                Z_part2 = Z_df_temp.iloc[:n_len_pt2]
                x2_res_sr, y2_res_sr, n2_res = _partial_residuals_xy_given_Z(x_part2, y_part2, Z_part2, method2_code)
                if n2_res >= 5:
                    residual_xlabel = "Partial residuals (ranked) of log10(power/baseline)" if method2_code == "spearman" else "Partial residuals of log10(power/baseline)"
                    residual_ylabel = "Partial residuals (ranked) of temperature (°C)" if method2_code == "spearman" else "Partial residuals of temperature (°C)"
                    
                    _generate_correlation_scatter(
                        x_data=x2_res_sr,
                        y_data=y2_res_sr,
                        x_label=residual_xlabel,
                        y_label=residual_ylabel,
                        title_prefix=f"Partial residuals — {band_title} vs temperature — Overall",
                        band_color=band_color,
                        output_path=plots_dir / f"scatter_pow_overall_{_sanitize(band)}_vs_temp_partial",
                        method_code=method2_code,
                        bootstrap_ci=bootstrap_ci,
                        rng=rng,
                        is_partial_residuals=True,
                    )

        for roi, chs in roi_map.items():
            roi_cols = [f"pow_{band}_{ch}" for ch in chs if f"pow_{band}_{ch}" in band_cols]
            if not roi_cols:
                continue
            roi_vals = pow_df[roi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

            # ROI-specific rating scatter
            _generate_correlation_scatter(
                x_data=roi_vals,
                y_data=y,
                x_label="log10(power/baseline [-5–0 s])",
                y_label="Rating",
                title_prefix=f"{band_title} power vs rating — {roi}",
                band_color=band_color,
                output_path=plots_dir / f"scatter_pow_roi_{_sanitize(roi)}_{_sanitize(band)}_vs_rating",
                method_code=method_code,
                Z_covars=Z_df_full,
                covar_names=covar_names,
                bootstrap_ci=bootstrap_ci,
                rng=rng,
            )
            
            # ROI partial-residuals figure if covariates available
            if Z_df_full is not None and len(Z_df_full) > 0:
                n_len_pt = min(len(roi_vals), len(y), len(Z_df_full))
                x_part = roi_vals.iloc[:n_len_pt]
                y_part = y.iloc[:n_len_pt]
                Z_part = Z_df_full.iloc[:n_len_pt]
                x_res_sr, y_res_sr, n_res = _partial_residuals_xy_given_Z(x_part, y_part, Z_part, method_code)
                if n_res >= 5:
                    residual_xlabel = "Partial residuals (ranked) of log10(power/baseline)" if method_code == "spearman" else "Partial residuals of log10(power/baseline)"
                    residual_ylabel = "Partial residuals (ranked) of rating" if method_code == "spearman" else "Partial residuals of rating"
                    
                    _generate_correlation_scatter(
                        x_data=x_res_sr,
                        y_data=y_res_sr,
                        x_label=residual_xlabel,
                        y_label=residual_ylabel,
                        title_prefix=f"Partial residuals — {band_title} vs rating — {roi}",
                        band_color=band_color,
                        output_path=plots_dir / f"scatter_pow_roi_{_sanitize(roi)}_{_sanitize(band)}_vs_rating_partial",
                        method_code=method_code,
                        bootstrap_ci=bootstrap_ci,
                        rng=rng,
                        is_partial_residuals=True,
                    )

            # ROI-specific temperature scatter (optional)
            if do_temp and temp_series is not None and len(temp_series) > 0:
                # Determine method for temperature (may be different if discrete)
                do_spear_t = bool(use_spearman and temp_series.nunique() > 5)
                method2_code = "spearman" if do_spear_t else "pearson"
                covar_names_temp = list(Z_df_temp.columns) if Z_df_temp is not None else None
                
                _generate_correlation_scatter(
                    x_data=roi_vals,
                    y_data=temp_series,
                    x_label="log10(power/baseline [-5–0 s])",
                    y_label="Temperature (°C)",
                    title_prefix=f"{band_title} power vs temperature — {roi}",
                    band_color=band_color,
                    output_path=plots_dir / f"scatter_pow_roi_{_sanitize(roi)}_{_sanitize(band)}_vs_temp",
                    method_code=method2_code,
                    Z_covars=Z_df_temp,
                    covar_names=covar_names_temp,
                    bootstrap_ci=bootstrap_ci,
                    rng=rng,
                )
                
                # ROI temperature partial-residuals figure if covariates available
                if Z_df_temp is not None and len(Z_df_temp) > 0:
                    n_len_pt2 = min(len(roi_vals), len(temp_series), len(Z_df_temp))
                    x_part2 = roi_vals.iloc[:n_len_pt2]
                    y_part2 = temp_series.iloc[:n_len_pt2]
                    Z_part2 = Z_df_temp.iloc[:n_len_pt2]
                    x2_res_sr, y2_res_sr, n2_res = _partial_residuals_xy_given_Z(x_part2, y_part2, Z_part2, method2_code)
                    if n2_res >= 5:
                        residual_xlabel = "Partial residuals (ranked) of log10(power/baseline)" if method2_code == "spearman" else "Partial residuals of log10(power/baseline)"
                        residual_ylabel = "Partial residuals (ranked) of temperature (°C)" if method2_code == "spearman" else "Partial residuals of temperature (°C)"
                        
                        _generate_correlation_scatter(
                            x_data=x2_res_sr,
                            y_data=y2_res_sr,
                            x_label=residual_xlabel,
                            y_label=residual_ylabel,
                            title_prefix=f"Partial residuals — {band_title} vs temperature — {roi}",
                            band_color=band_color,
                            output_path=plots_dir / f"scatter_pow_roi_{_sanitize(roi)}_{_sanitize(band)}_vs_temp_partial",
                            method_code=method2_code,
                            bootstrap_ci=bootstrap_ci,
                            rng=rng,
                            is_partial_residuals=True,
                        )

# Power-Behavior Correlation Visualization (transferred from 03_feature_engineering.py)
# -----------------------------------------------------------------------------

def plot_power_behavior_correlation(pow_df: pd.DataFrame, y: pd.Series, bands: List[str], 
                                   subject: str, save_dir: Path, logger: logging.Logger):
    """Plot correlations between band power and behavioral ratings."""
    try:
        if y is None or len(y) == 0 or y.isna().all():
            logger.warning("No valid behavioral data for correlation plots")
            return
            
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
                continue
                
            # Average power across all channels for this band
            band_power_avg = pow_df[band_cols].mean(axis=1)
            
            # Remove NaN pairs
            valid_mask = ~(band_power_avg.isna() | y.isna())
            if valid_mask.sum() < 5:  # Need at least 5 valid points
                logger.warning(f"Too few valid points for {band} band correlation")
                continue
                
            x_valid = band_power_avg[valid_mask]
            y_valid = y[valid_mask]
            
            # Create scatter plot
            band_color = getattr(config.visualization.band_colors, band, '#1f77b4') if hasattr(config.visualization, 'band_colors') else '#1f77b4'
            axes[i].scatter(x_valid, y_valid, alpha=0.6, s=30, color=band_color)
            
            # Add regression line
            z = np.polyfit(x_valid, y_valid, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
            axes[i].plot(x_line, p(x_line), 'r--', alpha=0.8)
            
            axes[i].set_xlabel(f'{band.capitalize()} Power\n(log10(power/baseline))')
            axes[i].set_ylabel('Behavioral Rating')
            axes[i].set_title(f'{band.capitalize()} Power vs Behavior')
            axes[i].grid(True, alpha=0.3)
            
            # Add correlation statistics
            r, p_val = stats.pearsonr(x_valid, y_valid)
            rho, p_spear = stats.spearmanr(x_valid, y_valid)
            
            stats_text = f'Pearson r={r:.3f} (p={p_val:.3f})\nSpearman ρ={rho:.3f} (p={p_spear:.3f})\nn={len(x_valid)}'
            axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for j in range(len(bands), len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        _save_fig(fig, save_dir / f'sub-{subject}_power_behavior_correlation')
        logger.info(f"Saved power-behavior correlations: {save_dir / f'sub-{subject}_power_behavior_correlation.png'}")
        
    except Exception as e:
        logger.error(f"Failed to create power-behavior correlation plots: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_power_behavior_correlation_matrix(pow_df: pd.DataFrame, y: pd.Series, bands: List[str], 
                                          subject: str, save_dir: Path, logger: logging.Logger):
    """Plot separate correlation figures for each frequency band."""
    try:
        for band in bands:
            # Get columns for this band
            band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
            if not band_cols:
                continue
            
            # Calculate correlations and p-values
            correlations = []
            p_values = []
            channel_names = []
            
            for col in band_cols:
                valid_mask = ~(pow_df[col].isna() | y.isna())
                if valid_mask.sum() > 5:
                    r, p = stats.spearmanr(pow_df[col][valid_mask], y[valid_mask])
                    correlations.append(r)
                    p_values.append(p)
                    channel_names.append(col.replace(f'pow_{band}_', ''))
                else:
                    correlations.append(0)
                    p_values.append(1.0)
                    channel_names.append(col.replace(f'pow_{band}_', ''))
            
            if len(correlations) > 0:
                # Create separate figure for this band
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create bar plot of correlations
                bars = ax.bar(range(len(correlations)), correlations, 
                             color=['red' if p < 0.05 else 'lightblue' for p in p_values])
                
                ax.set_xlabel('Channel', fontweight='bold')
                ax.set_ylabel('Spearman ρ', fontweight='bold')
                ax.set_title(f'{band.upper()} Band - Channel-wise Correlations with Behavior\nSubject {subject}', 
                           fontweight='bold', fontsize=14)
                ax.set_xticks(range(len(channel_names)))
                ax.set_xticklabels(channel_names, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # Add significance threshold lines
                ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Moderate correlation')
                ax.axhline(y=-0.3, color='green', linestyle='--', alpha=0.7)
                
                # Add legend for significance
                significant_count = sum(1 for p in p_values if p < 0.05)
                ax.text(0.02, 0.98, f'Significant channels: {significant_count}/{len(correlations)}', 
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                _save_fig(fig, save_dir / f'sub-{subject}_power_behavior_correlation_{band}')
                plt.close(fig)
        
        logger.info(f"Saved separate power-behavior correlation plots for {len(bands)} bands in: {save_dir}")
        
    except Exception as e:
        logger.error(f"Failed to create power-behavior correlation plots: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_behavioral_response_patterns(y: pd.Series, aligned_events: Optional[pd.DataFrame], 
                                    subject: str, save_dir: Path, logger: logging.Logger):
    """Plot behavioral response patterns and distributions as separate figures."""
    try:
        # 1. Rating distribution
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.hist(y.dropna(), bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Pain Rating')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Rating Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_rating = y.mean()
        std_rating = y.std()
        ax1.axvline(mean_rating, color='red', linestyle='--', label=f'Mean: {mean_rating:.2f}')
        ax1.axvline(mean_rating + std_rating, color='orange', linestyle=':', alpha=0.7, label=f'±SD: {std_rating:.2f}')
        ax1.axvline(mean_rating - std_rating, color='orange', linestyle=':', alpha=0.7)
        ax1.legend()
        
        plt.tight_layout()
        _save_fig(fig1, save_dir / f'sub-{subject}_rating_distribution')
        plt.close(fig1)
        
        
        
        # 4. Rating reliability (autocorrelation)
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        if len(y) > 10:
            # Calculate lag-1 autocorrelation
            y_shifted = y.shift(1)
            valid_mask = ~(y.isna() | y_shifted.isna())
            
            if valid_mask.sum() > 5:
                ax4.scatter(y_shifted[valid_mask], y[valid_mask], alpha=0.6, s=30)
                ax4.set_xlabel('Previous Trial Rating')
                ax4.set_ylabel('Current Trial Rating')
                ax4.set_title('Rating Consistency (Lag-1)')
                ax4.grid(True, alpha=0.3)
                
                # Add correlation
                r, p = stats.spearmanr(y_shifted[valid_mask], y[valid_mask])
                ax4.text(0.05, 0.95, f'ρ={r:.3f} (p={p:.3f})', 
                        transform=ax4.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'Insufficient trials for autocorrelation', 
                    ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Rating Consistency (Lag-1)')
        
        plt.tight_layout()
        _save_fig(fig4, save_dir / f'sub-{subject}_rating_consistency')
        plt.close(fig4)
        
        logger.info(f"Saved behavioral response patterns as separate figures in: {save_dir}")
        
    except Exception as e:
        logger.error(f"Failed to create behavioral response patterns: {e}")
        for fig_var in ['fig1', 'fig2', 'fig3', 'fig4']:
            if fig_var in locals():
                plt.close(locals()[fig_var])




def plot_behavioral_predictor_importance(pow_df: pd.DataFrame, y: pd.Series, bands: List[str],
                                       subject: str, save_dir: Path, logger: logging.Logger):
    """Plot behavioral predictor importance ranking across channels and bands."""
    try:
        # Calculate correlations for all power features
        correlations = []
        p_values = []
        feature_names = []
        band_labels = []
        
        for band in bands:
            band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
            
            for col in band_cols:
                valid_mask = ~(pow_df[col].isna() | y.isna())
                if valid_mask.sum() < 5:
                    continue
                    
                x_valid = pow_df[col][valid_mask]
                y_valid = y[valid_mask]
                
                r, p = stats.spearmanr(x_valid, y_valid)
                correlations.append(abs(r))  # Use absolute correlation for importance
                p_values.append(p)
                feature_names.append(col.replace(f'pow_{band}_', ''))
                band_labels.append(band)
        
        if not correlations:
            logger.warning("No valid correlations found for predictor importance")
            return
        
        # Filter to only include significant correlations (p < 0.05)
        significant_mask = np.array(p_values) < 0.05
        significant_correlations = [correlations[i] for i in range(len(correlations)) if significant_mask[i]]
        significant_features = [feature_names[i] for i in range(len(feature_names)) if significant_mask[i]]
        significant_bands = [band_labels[i] for i in range(len(band_labels)) if significant_mask[i]]
        significant_p_values = [p_values[i] for i in range(len(p_values)) if significant_mask[i]]
        
        if not significant_correlations:
            logger.warning("No significant correlations (p < 0.05) found for predictor importance")
            return
            
        # Sort by importance (absolute correlation) among significant results
        sorted_indices = np.argsort(significant_correlations)[::-1]  # Descending order
        top_n = min(20, len(significant_correlations))  # Show top 20 or all if fewer
        
        top_correlations = [significant_correlations[i] for i in sorted_indices[:top_n]]
        top_features = [significant_features[i] for i in sorted_indices[:top_n]]
        top_bands = [significant_bands[i] for i in sorted_indices[:top_n]]
        top_p_values = [significant_p_values[i] for i in sorted_indices[:top_n]]
        
        # 1. Horizontal bar plot of top predictors
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(top_n)
        colors = [BAND_COLORS.get(band, '#4C4C4C') for band in top_bands]
        
        bars = ax1.barh(y_pos, top_correlations, color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{feat} ({band})" for feat, band in zip(top_features, top_bands)])
        ax1.set_xlabel('|Spearman ρ| with Behavior (p < 0.05)')
        ax1.set_title(f'Top {top_n} Significant Behavioral Predictors')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add correlation values and p-values on bars
        for i, (bar, corr, p_val) in enumerate(zip(bars, top_correlations, top_p_values)):
            ax1.text(corr + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f} (p={p_val:.3f})', va='center', fontsize=8)
        
        plt.tight_layout()
        _save_fig(fig1, save_dir / f'sub-{subject}_top_behavioral_predictors')
        plt.close(fig1)
        
        # 2. Band-wise importance distribution (only for significant correlations)
        band_importance = {}
        for band in bands:
            band_correlations = [significant_correlations[i] for i, b in enumerate(significant_bands) if b == band]
            if band_correlations:
                band_importance[band] = {
                    'mean': np.mean(band_correlations),
                    'std': np.std(band_correlations),
                    'max': np.max(band_correlations)
                }
        
        if band_importance:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            bands_plot = list(band_importance.keys())
            means = [band_importance[b]['mean'] for b in bands_plot]
            stds = [band_importance[b]['std'] for b in bands_plot]
            colors_band = [BAND_COLORS.get(band, '#4C4C4C') for band in bands_plot]
            
            x_pos = np.arange(len(bands_plot))
            bars2 = ax2.bar(x_pos, means, yerr=stds, capsize=5, 
                           color=colors_band, alpha=0.7, error_kw={'alpha': 0.8})
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([b.capitalize() for b in bands_plot])
            ax2.set_ylabel('Mean |Spearman ρ|')
            ax2.set_title('Band-wise Predictor Importance')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add values on bars
            for bar, mean, std in zip(bars2, means, stds):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            _save_fig(fig2, save_dir / f'sub-{subject}_band_predictor_importance')
            plt.close(fig2)
        
        logger.info(f"Saved behavioral predictor importance plots as separate figures in: {save_dir}")
        
    except Exception as e:
        logger.error(f"Failed to create behavioral predictor importance: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_power_spectrogram_with_behavior(pow_df: pd.DataFrame, y: pd.Series, bands: List[str],
                                       subject: str, save_dir: Path, logger: logging.Logger):
    """Create publication-quality spectrograms showing EEG power dynamics with behavior overlay."""
    try:
        # Calculate average power across all channels per band
        band_powers = {}
        for band in bands:
            band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
            if band_cols:
                band_powers[band] = pow_df[band_cols].mean(axis=1)
        
        if not band_powers:
            logger.warning("No band power data found for spectrogram")
            return
        
        # Create separate figure for each band
        n_trials = len(y)
        trial_indices = np.arange(n_trials)
        
        # Normalize behavior for color mapping
        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else np.zeros_like(y)
        
        for band in bands:
            if band not in band_powers:
                continue
                
            power = band_powers[band]
            
            # Create individual figure for this band
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create power trace
            ax.plot(trial_indices, power, 'k-', alpha=0.3, linewidth=1, label='EEG Power')
            
            # Overlay behavior as colored scatter
            scatter = ax.scatter(trial_indices, power, c=y_norm, cmap='RdYlBu_r', 
                               s=40, alpha=0.8, edgecolor='black', linewidth=0.5,
                               label='Power (colored by rating)')
            
            # Add smooth trend lines
            from scipy import interpolate
            if len(power) > 10:
                # Smooth power trend
                power_smooth = interpolate.interp1d(trial_indices, power, kind='cubic')(trial_indices)
                ax.plot(trial_indices, power_smooth, 'b-', alpha=0.6, linewidth=2, label='Power trend')
            
            ax.set_ylabel(f'{band.capitalize()} Power (log)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Trial Number', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{band.capitalize()} Band Power Dynamics with Behavioral Ratings\nSubject {subject}', 
                        fontweight='bold', fontsize=14)
            
            # Add correlation info
            valid_mask = ~(power.isna() | y.isna())
            if valid_mask.sum() > 5:
                r, p = stats.spearmanr(power[valid_mask], y[valid_mask])
                ax.text(0.02, 0.95, f'ρ = {r:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontweight='bold')
            
            # Add colorbar for behavior
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Pain Rating (normalized)', fontweight='bold')
            
            # Set colorbar ticks to actual rating values
            y_min, y_max = y.min(), y.max()
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels([f'{y_min:.1f}', f'{(y_min+y_max)/2:.1f}', f'{y_max:.1f}'])
            
            plt.tight_layout()
            _save_fig(fig, save_dir / f'sub-{subject}_power_spectrogram_behavior_{band}')
            plt.close(fig)
        
        logger.info(f"Saved EEG power spectrograms with behavior for {len(bands)} bands in: {save_dir}")
        
    except Exception as e:
        logger.error(f"Failed to create power spectrograms with temperature: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_power_spectrogram_temperature_band(pow_df: pd.DataFrame, aligned_events: pd.DataFrame, bands: List[str],
                                           subject: str, save_dir: Path, logger: logging.Logger):
    """Create publication-quality spectrograms showing EEG power dynamics colored by temperature values."""
    try:
        # Extract temperature data from aligned events
        temp_col = _pick_first_column(aligned_events, PSYCH_TEMP_COLUMNS) if aligned_events is not None else None
        if temp_col is None or aligned_events is None:
            logger.warning("No temperature data found for spectrogram")
            return
            
        temp_series = pd.to_numeric(aligned_events[temp_col], errors="coerce")
        
        # Remove trials with missing temperature data
        valid_mask = ~temp_series.isna()
        if valid_mask.sum() < 5:
            logger.warning("Insufficient valid temperature data for spectrogram")
            return
            
        # Filter data to valid trials only
        pow_df_filtered = pow_df[valid_mask]
        temp_filtered = temp_series[valid_mask]
        
        # Calculate average power across all channels per band
        band_powers = {}
        for band in bands:
            band_cols = [col for col in pow_df_filtered.columns if col.startswith(f'pow_{band}_')]
            if band_cols:
                band_powers[band] = pow_df_filtered[band_cols].mean(axis=1)
        
        if not band_powers:
            logger.warning("No band power data found for temperature spectrogram")
            return
        
        # Create separate figure for each band
        n_trials = len(temp_filtered)
        trial_indices = np.arange(n_trials)
        
        # Normalize temperature for color mapping
        temp_norm = (temp_filtered - temp_filtered.min()) / (temp_filtered.max() - temp_filtered.min()) if temp_filtered.max() > temp_filtered.min() else np.zeros_like(temp_filtered)
        
        for band in bands:
            if band not in band_powers:
                continue
                
            power = band_powers[band]
            
            # Create individual figure for this band
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create power trace
            ax.plot(trial_indices, power, 'k-', alpha=0.3, linewidth=1, label='EEG Power')
            
            # Overlay temperature as colored scatter - use temperature-appropriate colormap
            scatter = ax.scatter(trial_indices, power, c=temp_norm, cmap='coolwarm', 
                               s=40, alpha=0.8, edgecolor='black', linewidth=0.5,
                               label='Power (colored by temperature)')
            
            # Add smooth trend lines
            from scipy import interpolate
            if len(power) > 10:
                # Smooth power trend
                power_smooth = interpolate.interp1d(trial_indices, power, kind='cubic')(trial_indices)
                ax.plot(trial_indices, power_smooth, 'b-', alpha=0.6, linewidth=2, label='Power trend')
            
            ax.set_ylabel(f'{band.capitalize()} Power (log)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Trial Number', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{band.capitalize()} Band Power Dynamics with Temperature\nSubject {subject}', 
                        fontweight='bold', fontsize=14)
            
            # Add correlation info
            valid_corr_mask = ~(power.isna() | temp_filtered.isna())
            if valid_corr_mask.sum() > 5:
                r, p = stats.spearmanr(power[valid_corr_mask], temp_filtered[valid_corr_mask])
                ax.text(0.02, 0.95, f'ρ = {r:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontweight='bold')
            
            # Add colorbar for temperature
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Temperature (°C)', fontweight='bold')
            
            # Set colorbar ticks to actual temperature values
            temp_min, temp_max = temp_filtered.min(), temp_filtered.max()
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels([f'{temp_min:.1f}', f'{(temp_min+temp_max)/2:.1f}', f'{temp_max:.1f}'])
            
            plt.tight_layout()
            _save_fig(fig, save_dir / f'sub-{subject}_power_spectrogram_temperature_{band}')
            plt.close(fig)
        
        logger.info(f"Saved EEG power spectrograms with temperature for {len(bands)} bands in: {save_dir}")
        
    except Exception as e:
        logger.error(f"Failed to create power spectrograms with temperature: {e}")
        if 'fig' in locals():
            plt.close(fig)


# Topographic correlation maps function removed due to MNE compatibility issues
# with show_names parameter in plot_topomap()


def plot_behavior_modulated_connectivity(subject: str, task: str, y: pd.Series,
                                       save_dir: Path, logger: logging.Logger):
    """Create behavior-modulated connectivity network visualization."""
    try:
        # Load connectivity data
        conn_path = _find_connectivity_path(subject, task)
        if not conn_path.exists():
            logger.warning(f"No connectivity data found for {subject}")
            return
        
        # Read connectivity data based on file extension
        if conn_path.suffix == '.parquet':
            conn_df = pd.read_parquet(conn_path)
        elif conn_path.suffix == '.tsv':
            conn_df = pd.read_csv(conn_path, sep='\t')
        else:
            logger.warning(f"Unsupported connectivity file format: {conn_path.suffix}")
            return
        
        # Get connectivity measures and frequency bands
        conn_measures = ['coh', 'plv', 'pli', 'wpli']
        available_measures = [m for m in conn_measures if any(m in col for col in conn_df.columns)]
        
        if not available_measures:
            logger.warning("No connectivity measures found")
            return
        
        # Focus on one measure for visualization (coherence preferred)
        measure = 'coh' if 'coh' in available_measures else available_measures[0]
        bands = ['alpha', 'beta', 'gamma']
        
        for band in bands:
            measure_cols = [col for col in conn_df.columns if f'{measure}_{band}' in col]
            if not measure_cols:
                continue
            
            # Calculate correlations with behavior for each connection
            correlations = []
            connections = []
            
            for col in measure_cols:
                valid_mask = ~(conn_df[col].isna() | y.isna())
                if valid_mask.sum() > 5:
                    r, p = stats.spearmanr(conn_df[col][valid_mask], y[valid_mask])
                    if abs(r) > 0.3 and p < 0.05:  # Only strong significant correlations
                        correlations.append(r)
                        # Extract channel pair from column name
                        pair = col.replace(f'{measure}_{band}_', '').replace('conn_', '')
                        connections.append(pair)
            
            if len(connections) < 3:
                continue
            
            # Create network visualization
            import networkx as nx
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create graph
            G = nx.Graph()
            
            # Add nodes and edges
            edge_weights = []
            for i, (conn, corr) in enumerate(zip(connections, correlations)):
                if '-' in conn:
                    ch1, ch2 = conn.split('-')
                    G.add_edge(ch1, ch2, weight=abs(corr), correlation=corr)
                    edge_weights.append(abs(corr))
            
            if G.number_of_nodes() == 0:
                continue
            
            # Create layout
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes
            node_sizes = [G.degree(node) * 100 for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                 node_color='lightblue', alpha=0.7, ax=ax)
            
            # Draw edges colored by correlation strength
            edges = G.edges()
            weights = [G[u][v]['correlation'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights,
                                 edge_cmap=plt.cm.RdBu_r, edge_vmin=-max(abs(w) for w in weights),
                                 edge_vmax=max(abs(w) for w in weights), width=2, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, 
                                     norm=plt.Normalize(vmin=-max(abs(w) for w in weights),
                                                      vmax=max(abs(w) for w in weights)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Correlation with Behavior', fontweight='bold')
            
            ax.set_title(f'Behavior-Modulated {measure.upper()} Connectivity\n{band.capitalize()} Band - Subject {subject}',
                        fontweight='bold', fontsize=14)
            ax.axis('off')
            
            plt.tight_layout()
            _save_fig(fig, save_dir / f'sub-{subject}_connectivity_network_{measure}_{band}')
            plt.close(fig)
        
        logger.info(f"Saved behavior-modulated connectivity networks")
        
    except Exception as e:
        logger.error(f"Failed to create behavior-modulated connectivity network: {e}")
        if 'fig' in locals():
            plt.close(fig)





def plot_neural_state_pca(pow_df: pd.DataFrame, y: pd.Series, 
                         subject: str, save_dir: Path, logger: logging.Logger):
    """Create neural state space visualization using PCA."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data for PCA
        pow_clean = pow_df.dropna()
        if len(pow_clean) < 10:
            logger.warning("Insufficient data for PCA analysis")
            return
            
        scaler = StandardScaler()
        pow_scaled = scaler.fit_transform(pow_clean)
        
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(pow_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                           c=y.loc[pow_clean.index], cmap='RdYlBu_r', 
                           s=60, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
        ax.set_title(f'Neural State Space (PCA)\nSubject {subject}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pain Rating', fontweight='bold')
        
        # Add explained variance text
        total_var = sum(pca.explained_variance_ratio_[:2])
        ax.text(0.02, 0.98, f'Total Variance Explained: {total_var*100:.1f}%',
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontweight='bold')
        
        plt.tight_layout()
        _save_fig(fig, save_dir / f'sub-{subject}_neural_state_pca')
        plt.close(fig)
        
        logger.info(f"Saved neural state PCA visualization")
        
    except Exception as e:
        logger.error(f"Failed to create neural state PCA: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_band_power_summary(pow_df: pd.DataFrame, bands: List[str],
                           subject: str, save_dir: Path, logger: logging.Logger):
    """Create band power summary visualization."""
    try:
        band_means = []
        band_stds = []
        valid_bands = []
        
        for band in bands:
            band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
            if band_cols:
                band_power = pow_df[band_cols].mean(axis=1)
                band_means.append(band_power.mean())
                band_stds.append(band_power.std())
                valid_bands.append(band)
        
        if not valid_bands:
            logger.warning("No valid bands for power summary")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [BAND_COLORS.get(band, '#4C4C4C') for band in valid_bands]
        bars = ax.bar(range(len(valid_bands)), band_means, yerr=band_stds,
                     capsize=5, color=colors, alpha=0.7, error_kw={'alpha': 0.8})
        ax.set_xticks(range(len(valid_bands)))
        ax.set_xticklabels([b.capitalize() for b in valid_bands])
        ax.set_ylabel('Mean Log Power', fontweight='bold')
        ax.set_title(f'EEG Band Power Summary\nSubject {subject}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, mean, std in zip(bars, band_means, band_stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                   f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        _save_fig(fig, save_dir / f'sub-{subject}_band_power_summary')
        plt.close(fig)
        
        logger.info(f"Saved band power summary visualization")
        
    except Exception as e:
        logger.error(f"Failed to create band power summary: {e}")
        if 'fig' in locals():
            plt.close(fig)


def plot_statistical_effect_size_summary(pow_df: pd.DataFrame, y: pd.Series, bands: List[str],
                                        subject: str, save_dir: Path, logger: logging.Logger):
    """Create statistical summary with effect sizes for publication."""
    try:
        from scipy.stats import pearsonr
        
        # Collect statistical metrics
        stats_data = []
        
        for band in bands:
            band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
            
            for col in band_cols:
                ch_name = col.replace(f'pow_{band}_', '')
                valid_mask = ~(pow_df[col].isna() | y.isna())
                
                if valid_mask.sum() > 10:
                    x_vals = pow_df[col][valid_mask]
                    y_vals = y[valid_mask]
                    
                    # Spearman correlation
                    r_spear, p_spear = stats.spearmanr(x_vals, y_vals)
                    
                    # Pearson correlation
                    r_pears, p_pears = pearsonr(x_vals, y_vals)
                    
                    # Effect size (Cohen's r interpretation)
                    effect_size = 'Small' if abs(r_spear) < 0.3 else ('Medium' if abs(r_spear) < 0.5 else 'Large')
                    
                    stats_data.append({
                        'Band': band.capitalize(),
                        'Channel': ch_name,
                        'Spearman_r': r_spear,
                        'Spearman_p': p_spear,
                        'Pearson_r': r_pears,
                        'Pearson_p': p_pears,
                        'Effect_Size': effect_size,
                        'N': valid_mask.sum()
                    })
        
        if not stats_data:
            logger.warning("No statistical data available")
            return
        
        # Convert to DataFrame for easier handling
        stats_df = pd.DataFrame(stats_data)
        
        # Create separate statistical summary figures
        
        # Figure 1: Effect size distribution by band
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        effect_counts = stats_df.groupby(['Band', 'Effect_Size']).size().unstack(fill_value=0)
        effect_counts.plot(kind='bar', ax=ax1, color=['lightcoral', 'khaki', 'lightgreen'])
        ax1.set_title(f'Effect Size Distribution by Band\nSubject {subject}', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Number of Channels', fontweight='bold')
        ax1.set_xlabel('Frequency Band', fontweight='bold')
        ax1.legend(title='Effect Size')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        _save_fig(fig1, save_dir / f'sub-{subject}_effect_size_distribution')
        plt.close(fig1)
        
        
        # Figure 3: Significance by band
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sig_summary = stats_df.groupby('Band').apply(
            lambda x: (x['Spearman_p'] < 0.05).sum() / len(x) * 100,
            include_groups=False
        ).reset_index()
        sig_summary.columns = ['Band', 'Significant_Percent']
        
        colors = [BAND_COLORS.get(band.lower(), '#4C4C4C') for band in sig_summary['Band']]
        bars = ax3.bar(sig_summary['Band'], sig_summary['Significant_Percent'], 
                      color=colors, alpha=0.7)
        ax3.set_ylabel('% Significant Channels', fontweight='bold')
        ax3.set_xlabel('Frequency Band', fontweight='bold')
        ax3.set_title(f'Significant Brain-Behavior Correlations by Band (p < 0.05)\nSubject {subject}', 
                     fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add percentages on bars
        for bar, pct in zip(bars, sig_summary['Significant_Percent']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        _save_fig(fig3, save_dir / f'sub-{subject}_significance_by_band')
        plt.close(fig3)
        
        # Figure 4: Top correlations table-style visualization (significant only)
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        # Filter to only significant correlations first
        significant_stats = stats_df[stats_df['Spearman_p'] < 0.05]
        
        if len(significant_stats) == 0:
            logger.warning("No significant correlations (p < 0.05) found for top correlations plot")
            plt.close(fig4)
        else:
            top_correlations = significant_stats.nlargest(min(15, len(significant_stats)), 'Spearman_r')
            
            y_pos = np.arange(len(top_correlations))
            colors_top = [BAND_COLORS.get(band.lower(), '#4C4C4C') for band in top_correlations['Band']]
            
            bars = ax4.barh(y_pos, top_correlations['Spearman_r'], color=colors_top, alpha=0.7)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([f"{row['Channel']} ({row['Band']})" for _, row in top_correlations.iterrows()])
            ax4.set_xlabel('Spearman ρ (p < 0.05)', fontweight='bold')
            ax4.set_title(f'Top {len(top_correlations)} Significant Brain-Behavior Correlations\nSubject {subject}', fontweight='bold', fontsize=14)
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Add correlation values and p-values
            for i, (bar, _, row) in enumerate(zip(bars, top_correlations['Spearman_r'], top_correlations.itertuples())):
                r_val = row.Spearman_r
                p_val = row.Spearman_p
                ax4.text(r_val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{r_val:.3f} (p={p_val:.3f})', va='center', fontsize=8)
            
            plt.tight_layout()
            _save_fig(fig4, save_dir / f'sub-{subject}_top_correlations')
            plt.close(fig4)
        
        # Save statistical summary as CSV
        stats_df.to_csv(save_dir / f'sub-{subject}_statistical_summary.csv', index=False)
        
        logger.info(f"Saved statistical effect size summary and CSV")
        
    except Exception as e:
        logger.error(f"Failed to create statistical effect size summary: {e}")
        if 'fig' in locals():
            plt.close(fig)


# Connectivity ROI summary correlations (within/between ROI averages)
# -----------------------------------------------------------------------------

def correlate_connectivity_roi_summaries(
    subject: str,
    task: str = TASK,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    # Initialize RNG if not provided
    if rng is None:
        rng = np.random.default_rng(42)

    feats_dir = _features_dir(subject)
    conn_path = feats_dir / "features_connectivity.tsv"
    y_path = feats_dir / "target_vas_ratings.tsv"
    if not conn_path.exists() or not y_path.exists():
        return

    # Load info for ROI definitions and align events for covariates
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None:
        return
    epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
    info = epochs.info
    roi_map = _build_rois(info)
    events = _load_events_df(subject, task)
    aligned_events = _align_events_to_epochs(events, epochs) if events is not None else None
    # Temperature series aligned to epochs (if available)
    temp_series: Optional[pd.Series] = None
    temp_col: Optional[str] = None
    if aligned_events is not None:
        tcol = _pick_first_column(aligned_events, PSYCH_TEMP_COLUMNS)
        if tcol is not None:
            temp_col = tcol
            temp_series = pd.to_numeric(aligned_events[tcol], errors="coerce")

    # Helper: build multi-covariate design matrix Z from aligned events
    def _build_Z(df_events: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df_events is None:
            return None
        covars = list(partial_covars) if partial_covars is not None else []
        if not covars:
            # Try to infer a reasonable default set similar to power ROI stats
            tcol = _pick_first_column(df_events, PSYCH_TEMP_COLUMNS)
            if tcol is not None:
                covars.append(tcol)
            for c in ["trial", "trial_number", "trial_index", "run", "block"]:
                if c in df_events.columns:
                    covars.append(c)
                    break
        if not covars:
            return None
        Z = pd.DataFrame()
        for c in covars:
            if c in df_events.columns:
                Z[c] = pd.to_numeric(df_events[c], errors="coerce")
        return Z if not Z.empty else None

    Z_df_full = _build_Z(aligned_events)
    # For temperature targets, drop the temperature column from Z to avoid conditioning on the outcome variable
    Z_df_temp = None
    if Z_df_full is not None:
        try:
            Z_df_temp = Z_df_full.drop(columns=[temp_col], errors="ignore") if temp_col else Z_df_full.copy()
            if Z_df_temp.shape[1] == 0:
                Z_df_temp = None
        except (KeyError, AttributeError, ValueError):
            Z_df_temp = Z_df_full

    def _partial_corr_xy_given_Z(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str) -> Tuple[float, float, int]:
        df_full = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1)
        df = df_full.dropna()
        if len(df_full) > len(df):
            logger.warning(f"Partial correlation dropped {len(df_full) - len(df)} rows due to missing data (kept {len(df)}/{len(df_full)})")
        if len(df) < 5 or df["y"].nunique() <= 1:
            return np.nan, np.nan, 0
        Xd = np.column_stack([np.ones(len(df)), df[Z.columns].to_numpy()])
        bx = np.linalg.lstsq(Xd, df["x"].to_numpy(), rcond=None)[0]
        by = np.linalg.lstsq(Xd, df["y"].to_numpy(), rcond=None)[0]
        x_res = df["x"].to_numpy() - Xd.dot(bx)
        y_res = df["y"].to_numpy() - Xd.dot(by)
        if method == "spearman" and df["y"].nunique() > 5:
            r_p, p_p = stats.spearmanr(x_res, y_res, nan_policy="omit")
        else:
            r_p, p_p = stats.pearsonr(x_res, y_res)
        return float(r_p), float(p_p), int(len(df))
    # Note: Do not require sensor ROI map here; atlas-based grouping may still proceed.

    def _fisher_ci(r: float, n: int) -> Tuple[float, float]:
        if not np.isfinite(r) or n < 4:
            return np.nan, np.nan
        r = float(np.clip(r, -0.999999, 0.999999))
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        z_lo = z - 1.96 * se
        z_hi = z + 1.96 * se
        return float(np.tanh(z_lo)), float(np.tanh(z_hi))

    def _perm_pval_simple(x: pd.Series, y: pd.Series, method: str, n_perm: int, rng: np.random.Generator) -> float:
        df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(df) < 5:
            return np.nan
        if method == "spearman" and df["y"].nunique() > 5:
            obs, _ = stats.spearmanr(df["x"], df["y"], nan_policy="omit")
        else:
            obs, _ = stats.pearsonr(df["x"], df["y"])
        ge = 1
        y_vals = df["y"].to_numpy()
        for _ in range(int(n_perm)):
            y_pi = y_vals[rng.permutation(len(y_vals))]
            if method == "spearman" and df["y"].nunique() > 5:
                rp, _ = stats.spearmanr(df["x"], y_pi, nan_policy="omit")
            else:
                rp, _ = stats.pearsonr(df["x"], y_pi)
            if np.abs(rp) >= np.abs(obs) - 1e-12:
                ge += 1
        return ge / (int(n_perm) + 1)

    def _perm_pval_partial_FL(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str, n_perm: int, rng: np.random.Generator) -> float:
        df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
        if len(df) < 5:
            return np.nan
        Xd = np.column_stack([np.ones(len(df)), df[Z.columns].to_numpy()])
        bx = np.linalg.lstsq(Xd, df["x"].to_numpy(), rcond=None)[0]
        by = np.linalg.lstsq(Xd, df["y"].to_numpy(), rcond=None)[0]
        rx = df["x"].to_numpy() - Xd.dot(bx)
        ry = df["y"].to_numpy() - Xd.dot(by)
        if method == "spearman" and df["y"].nunique() > 5:
            obs, _ = stats.spearmanr(rx, ry, nan_policy="omit")
        else:
            obs, _ = stats.pearsonr(rx, ry)
        ge = 1
        for _ in range(int(n_perm)):
            ry_pi = ry[rng.permutation(len(ry))]
            if method == "spearman" and df["y"].nunique() > 5:
                rp, _ = stats.spearmanr(rx, ry_pi, nan_policy="omit")
            else:
                rp, _ = stats.pearsonr(rx, ry_pi)
            if np.abs(rp) >= np.abs(obs) - 1e-12:
                ge += 1
        return ge / (int(n_perm) + 1)

    X = pd.read_csv(conn_path, sep="\t")
    y_df = pd.read_csv(y_path, sep="\t")
    y = pd.to_numeric(y_df.iloc[:, 0], errors="coerce")

    cols = list(X.columns)
    prefixes = sorted({"_".join(c.split("_")[:2]) for c in cols})

    # Build node list per prefix
    for pref in prefixes:
        cols_pref = [c for c in cols if c.startswith(pref + "_")]
        if not cols_pref:
            continue
        pair_names = [c.split(pref + "_", 1)[-1] for c in cols_pref]
        nodes = sorted({nm for pair in pair_names for nm in pair.split("__")})
        node_to_idx = {nm: i for i, nm in enumerate(nodes)}

        # Determine measure name for Fisher z averaging of edges (e.g., AEC)
        meas_name = pref.split("_", 1)[0].lower()
        apply_fisher_edges = meas_name in ("aec", "aec_orth", "corr", "pearsonr")

        # --- Helper: build atlas-based ROI map from node labels ---
        def _build_atlas_rois_from_nodes(node_list: List[str], hemisphere_split: bool = True) -> Dict[str, List[str]]:
            import re
            # Known 7-Network system tokens (Schaefer/Yeo)
            systems = {"Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"}
            roi_nodes: Dict[str, List[str]] = {}
            for nm in node_list:
                toks = nm.split("_")
                hemi = None
                system = None
                # Heuristic: find hemisphere token and system token among parts
                for t in toks:
                    if t in ("LH", "RH"):
                        hemi = t
                    if t in systems:
                        system = t
                if system is None:
                    # Try regex fallback like '.*_(LH|RH)_(?P<sys>[A-Za-z]+)' patterns
                    m = re.search(r"(?:^|_)(LH|RH)_([A-Za-z]+)", nm)
                    if m:
                        hemi = m.group(1)
                        cand = m.group(2)
                        if cand in systems:
                            system = cand
                if system is None:
                    continue
                roi = system + (f"_{hemi}" if hemisphere_split and hemi else "")
                roi_nodes.setdefault(roi, []).append(nm)
            return roi_nodes

        # --- Helper: generic builder of within/between summary map from an ROI->nodes map ---
        def _build_summary_map_from_roi_nodes(roi_nodes: Dict[str, List[str]]) -> Dict[Tuple[str, str], List[str]]:
            summary: Dict[Tuple[str, str], List[str]] = {}
            # Within-ROI
            for roi_name, members in roi_nodes.items():
                ch_set = set(members)
                key = (roi_name, roi_name)
                cols_within: List[str] = []
                for col in cols_pref:
                    pair = col.split(pref + "_", 1)[-1]
                    try:
                        a, b = pair.split("__")
                    except ValueError:
                        continue
                    if a in ch_set and b in ch_set and a != b:
                        cols_within.append(col)
                if cols_within:
                    summary[key] = cols_within
            # Between-ROI
            rois_local = sorted(roi_nodes.keys())
            for i in range(len(rois_local)):
                for j in range(i + 1, len(rois_local)):
                    r1, r2 = rois_local[i], rois_local[j]
                    set1, set2 = set(roi_nodes[r1]), set(roi_nodes[r2])
                    cols_between: List[str] = []
                    for col in cols_pref:
                        pair = col.split(pref + "_", 1)[-1]
                        try:
                            a, b = pair.split("__")
                        except ValueError:
                            continue
                        if (a in set1 and b in set2) or (a in set2 and b in set1):
                            cols_between.append(col)
                    if cols_between:
                        summary[(r1, r2)] = cols_between
            return summary

        # 1) Try atlas-based ROI grouping first
        atlas_roi_map = _build_atlas_rois_from_nodes(nodes, hemisphere_split=True)
        summary_map: Dict[Tuple[str, str], List[str]] = {}
        if atlas_roi_map:
            summary_map = _build_summary_map_from_roi_nodes(atlas_roi_map)

        # 2) Fallback to sensor ROI grouping (will usually yield empty with atlas nodes)
        if not summary_map and roi_map:
            summary_map = _build_summary_map_from_roi_nodes(roi_map)

        if not summary_map:
            continue

        # Correlate each summary with behavior (optionally with partials and bootstrap CIs)
        recs: List[Dict[str, object]] = []
        recs_temp: List[Dict[str, object]] = []
        for (roi_i, roi_j), cols_list in summary_map.items():
            edge_df = X[cols_list].apply(pd.to_numeric, errors="coerce")
            if apply_fisher_edges:
                arr = edge_df.to_numpy(dtype=float)
                arr = np.clip(arr, -0.999999, 0.999999)
                z = np.arctanh(arr)
                z_mean = np.nanmean(z, axis=1)
                xi = pd.Series(np.tanh(z_mean), index=edge_df.index)
            else:
                xi = edge_df.mean(axis=1)
            mask = xi.notna() & y.notna()
            n_eff = int(mask.sum())
            if n_eff < 5:
                continue
            if use_spearman and y.nunique() > 5:
                r, p = stats.spearmanr(xi[mask], y[mask], nan_policy="omit")
                method = "spearman"
            else:
                r, p = stats.pearsonr(xi[mask], y[mask])
                method = "pearson"

            # Partial correlations given multi-covariates (if available)
            r_part = np.nan
            p_part = np.nan
            n_part = 0
            if Z_df_full is not None and len(Z_df_full) > 0:
                n_len_pt = min(len(xi), len(y), len(Z_df_full))
                r_part, p_part, n_part = _partial_corr_xy_given_Z(
                    xi.iloc[:n_len_pt], y.iloc[:n_len_pt], Z_df_full.iloc[:n_len_pt], method
                )

            # CIs for r: Fisher z for Pearson; bootstrap for Spearman
            ci_low = np.nan
            ci_high = np.nan
            if method == "pearson" and n_eff >= 4:
                ci_low, ci_high = _fisher_ci(r, n_eff)
            elif bootstrap and n_eff >= 5:
                idx = np.where(mask.to_numpy())[0]
                boots: List[float] = []
                for _ in range(int(bootstrap)):
                    bidx = rng.choice(idx, size=len(idx), replace=True)
                    xb = xi.iloc[bidx]
                    yb = y.iloc[bidx]
                    if method == "spearman" and yb.nunique() > 5:
                        rb, _ = stats.spearmanr(xb, yb, nan_policy="omit")
                    else:
                        rb, _ = stats.pearsonr(xb, yb)
                    boots.append(rb)
                if boots:
                    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

            # Permutation p-values (simple and partial)
            p_perm = np.nan
            p_partial_perm = np.nan
            if n_perm and n_eff >= 5:
                p_perm = _perm_pval_simple(xi, y, method, int(n_perm), rng)
                if Z_df_full is not None and len(Z_df_full) > 0:
                    n_len_pt = min(len(xi), len(y), len(Z_df_full))
                    p_partial_perm = _perm_pval_partial_FL(
                        xi.iloc[:n_len_pt], y.iloc[:n_len_pt], Z_df_full.iloc[:n_len_pt], method, int(n_perm), rng
                    )

            recs.append({
                "measure_band": pref,
                "roi_i": roi_i,
                "roi_j": roi_j,
                "summary_type": "within" if roi_i == roi_j else "between",
                "n_edges": len(cols_list),
                "r": float(r),
                "p": float(p),
                "n": n_eff,
                "method": method,
                # CIs and partials
                "r_ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                "r_ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                "r_partial": float(r_part) if np.isfinite(r_part) else np.nan,
                "p_partial": float(p_part) if np.isfinite(p_part) else np.nan,
                "n_partial": n_part,
                "partial_covars": ",".join(Z_df_full.columns.tolist()) if Z_df_full is not None else "",
                # Permutations
                "p_perm": float(p_perm) if np.isfinite(p_perm) else np.nan,
                "p_partial_perm": float(p_partial_perm) if np.isfinite(p_partial_perm) else np.nan,
                "n_perm": int(n_perm),
            })

            # Temperature correlations (if available)
            if temp_series is not None and len(temp_series) > 0:
                n_len_t = min(len(xi), len(temp_series))
                xt = xi.iloc[:n_len_t]
                tt = temp_series.iloc[:n_len_t]
                m2 = xt.notna() & tt.notna()
                n_eff2 = int(m2.sum())
                if n_eff2 >= 5:
                    if use_spearman and tt.nunique() > 5:
                        r2, p2 = stats.spearmanr(xt[m2], tt[m2], nan_policy="omit")
                        method2 = "spearman"
                    else:
                        r2, p2 = stats.pearsonr(xt[m2], tt[m2])
                        method2 = "pearson"
                    # Partial correlation controlling covariates excluding temperature itself
                    r2_part = np.nan
                    p2_part = np.nan
                    n2_part = 0
                    if Z_df_temp is not None and len(Z_df_temp) > 0:
                        n_len_pt2 = min(len(xt), len(tt), len(Z_df_temp))
                        r2_part, p2_part, n2_part = _partial_corr_xy_given_Z(
                            xt.iloc[:n_len_pt2], tt.iloc[:n_len_pt2], Z_df_temp.iloc[:n_len_pt2], method2
                        )
                    # CIs for r2: Fisher z for Pearson; bootstrap for Spearman
                    ci2_low = np.nan
                    ci2_high = np.nan
                    if method2 == "pearson" and n_eff2 >= 4:
                        ci2_low, ci2_high = _fisher_ci(r2, n_eff2)
                    elif bootstrap and n_eff2 >= 5:
                        idx2 = np.where(m2.to_numpy())[0]
                        boots2: List[float] = []
                        for _ in range(int(bootstrap)):
                            bidx2 = rng.choice(idx2, size=len(idx2), replace=True)
                            xb = xt.iloc[bidx2]
                            tb = tt.iloc[bidx2]
                            if method2 == "spearman" and tb.nunique() > 5:
                                rb, _ = stats.spearmanr(xb, tb, nan_policy="omit")
                            else:
                                rb, _ = stats.pearsonr(xb, tb)
                            boots2.append(rb)
                        if boots2:
                            ci2_low, ci2_high = np.percentile(boots2, [2.5, 97.5])
                    # Permutation p-values for temperature correlations (simple and partial)
                    p2_perm = np.nan
                    p2_partial_perm = np.nan
                    if n_perm and n_eff2 >= 5:
                        p2_perm = _perm_pval_simple(xt, tt, method2, int(n_perm), rng)
                        if Z_df_temp is not None and len(Z_df_temp) > 0:
                            n_len_pt2 = min(len(xt), len(tt), len(Z_df_temp))
                            p2_partial_perm = _perm_pval_partial_FL(
                                xt.iloc[:n_len_pt2], tt.iloc[:n_len_pt2], Z_df_temp.iloc[:n_len_pt2], method2, int(n_perm), rng
                            )

                    recs_temp.append({
                        "measure_band": pref,
                        "roi_i": roi_i,
                        "roi_j": roi_j,
                        "summary_type": "within" if roi_i == roi_j else "between",
                        "n_edges": len(cols_list),
                        "r": float(r2),
                        "p": float(p2),
                        "n": n_eff2,
                        "method": method2,
                        "r_ci_low": float(ci2_low) if np.isfinite(ci2_low) else np.nan,
                        "r_ci_high": float(ci2_high) if np.isfinite(ci2_high) else np.nan,
                        "r_partial": float(r2_part) if np.isfinite(r2_part) else np.nan,
                        "p_partial": float(p2_part) if np.isfinite(p2_part) else np.nan,
                        "n_partial": n2_part,
                        "partial_covars": ",".join(Z_df_temp.columns.tolist()) if Z_df_temp is not None else "",
                        "p_perm": float(p2_perm) if np.isfinite(p2_perm) else np.nan,
                        "p_partial_perm": float(p2_partial_perm) if np.isfinite(p2_partial_perm) else np.nan,
                        "n_perm": int(n_perm),
                    })

        if recs:
            df = pd.DataFrame(recs)
            # FDR per measure_band, prefer permutation p-values when available
            pvec = df["p_perm"].to_numpy() if "p_perm" in df.columns and np.isfinite(df["p_perm"]).any() else df["p"].to_numpy()
            rej, crit = _fdr_bh(pvec, alpha=0.05)
            df["fdr_reject"] = rej
            df["fdr_crit_p"] = crit
            df.to_csv(stats_dir / f"corr_stats_conn_roi_summary_{_sanitize(pref)}_vs_rating.tsv", sep="\t", index=False)

        if recs_temp:
            df_t = pd.DataFrame(recs_temp)
            pvec_t = df_t["p_perm"].to_numpy() if "p_perm" in df_t.columns and np.isfinite(df_t["p_perm"]).any() else df_t["p"].to_numpy()
            rej_t, crit_t = _fdr_bh(pvec_t, alpha=0.05)
            df_t["fdr_reject"] = rej_t
            df_t["fdr_crit_p"] = crit_t
            df_t.to_csv(stats_dir / f"corr_stats_conn_roi_summary_{_sanitize(pref)}_vs_temp.tsv", sep="\t", index=False)


# -----------------------------------------------------------------------------
# Connectivity correlations -> heatmaps (optional if connectivity features exist)
# -----------------------------------------------------------------------------

def correlate_connectivity_heatmaps(subject: str, task: str = TASK, use_spearman: bool = True) -> None:
    logger = _setup_logging(subject)
    logger.info(f"Starting connectivity correlation analysis for sub-{subject}")
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    # Load features/targets
    feats_dir = _features_dir(subject)
    conn_path = feats_dir / "features_connectivity.tsv"
    y_path = feats_dir / "target_vas_ratings.tsv"
    if not conn_path.exists() or not y_path.exists():
        logger.warning(f"Connectivity features or targets missing for sub-{subject}; skipping connectivity correlations.")
        return

    X = pd.read_csv(conn_path, sep="\t")
    y_df = pd.read_csv(y_path, sep="\t")
    y = pd.to_numeric(y_df.iloc[:, 0], errors="coerce")

    # Determine available measures and bands from columns (e.g., 'aec_alpha_F3__Cz')
    cols = list(X.columns)
    # Parse unique prefixes like 'aec_alpha' or 'wpli_beta'
    prefixes = sorted({"_".join(c.split("_")[:2]) for c in cols})

    for pref in prefixes:
        cols_pref = [c for c in cols if c.startswith(pref + "_")]
        # Estimate node labels from pair names if possible
        pair_names = [c.split(pref + "_", 1)[-1] for c in cols_pref]
        # Extract node names as union of tokens split by '__'
        nodes = sorted({nm for pair in pair_names for nm in pair.split("__")})
        n_nodes = len(nodes)
        if n_nodes < 3:
            print(f"Could not infer nodes for {pref}; skipping heatmap.")
            continue
        node_idx = {nm: i for i, nm in enumerate(nodes)}

        # Correlate each edge with target using pairwise deletion
        rvals = np.full((n_nodes, n_nodes), np.nan, float)
        pvals = np.full((n_nodes, n_nodes), np.nan, float)
        for col in cols_pref:
            pair = col.split(pref + "_", 1)[-1]
            try:
                a, b = pair.split("__")
            except ValueError:
                continue
            i, j = node_idx[a], node_idx[b]
            xi = pd.to_numeric(X[col], errors="coerce")
            mask = xi.notna() & y.notna()
            if mask.sum() < 5:
                continue
            if use_spearman and y.nunique() > 5:
                r, p = stats.spearmanr(xi[mask], y[mask], nan_policy="omit")
            else:
                r, p = stats.pearsonr(xi[mask], y[mask])
            rvals[i, j] = r
            rvals[j, i] = r
            pvals[i, j] = p
            pvals[j, i] = p
        
        # We still compute and save TSV edge-level stats below.

        # Save TSV of edge stats (upper triangle i>j)
        # FDR across valid edges
        iu = np.triu_indices(n_nodes, k=1)
        p_flat = pvals[iu]
        valid_idx = np.isfinite(p_flat)
        p_valid = p_flat[valid_idx]
        rej_valid, crit_p = _fdr_bh(p_valid, alpha=0.05)
        # Map back to edges
        valid_pairs = [(iu[0][k], iu[1][k]) for k in np.where(valid_idx)[0]]
        reject_map = {pair: bool(rej_valid[k]) for k, pair in enumerate(valid_pairs)}
        crit_val = float(np.max(p_valid[rej_valid])) if np.any(rej_valid) else np.nan

        recs: List[Dict[str, object]] = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                recs.append({
                    "node_i": nodes[i],
                    "node_j": nodes[j],
                    "r": rvals[i, j],
                    "p": pvals[i, j],
                    "fdr_reject": reject_map.get((i, j), False),
                    "fdr_crit_p": crit_val,
                })
        edges_df = pd.DataFrame(recs)
        edges_df.to_csv(stats_dir / f"corr_stats_edges_{_sanitize(pref)}_vs_rating.tsv", sep="\t", index=False)
        # Convenience: write top-20 edges by |r|
        try:
            topn = edges_df.dropna(subset=["r"]).assign(abs_r=lambda d: d["r"].abs()).nlargest(20, "abs_r")
            topn.to_csv(stats_dir / f"corr_stats_edges_{_sanitize(pref)}_vs_rating_top20.tsv", sep="\t", index=False)
        except (KeyError, ValueError, OSError):
            pass


# -----------------------------------------------------------------------------
# Export: Combined per-band power correlation stats
# -----------------------------------------------------------------------------

def export_combined_power_corr_stats(subject: str) -> None:
    """Combine per-band per-channel power correlation TSVs into consolidated files.

    For each target in {rating, temp}, reads files named
    'corr_stats_pow_{band}_vs_{target}.tsv' for bands in POWER_BANDS_TO_USE,
    concatenates them (adding a 'band' column if missing), and writes
    'corr_stats_pow_combined_vs_{target}.tsv' and '.csv' in the subject stats dir.
    Missing input files are skipped gracefully.
    """
    stats_dir = _stats_dir(subject)
    _ensure_dir(stats_dir)

    bands = list(POWER_BANDS_TO_USE)
    for target in ("rating", "temp"):
        frames: List[pd.DataFrame] = []
        for band in bands:
            f = stats_dir / f"corr_stats_pow_{band}_vs_{target}.tsv"
            if not f.exists():
                continue
            try:
                df = pd.read_csv(f, sep="\t")
            except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if df is None or df.empty:
                continue
            # Ensure a 'band' column exists and is filled
            if "band" not in df.columns:
                df["band"] = band
            else:
                try:
                    df["band"] = df["band"].fillna(band)
                except (KeyError, AttributeError):
                    pass
            frames.append(df)

        if frames:
            cat = pd.concat(frames, ignore_index=True)
            out_base = stats_dir / f"corr_stats_pow_combined_vs_{target}"
            cat.to_csv(out_base.with_suffix(".tsv"), sep="\t", index=False)
            try:
                cat.to_csv(out_base.with_suffix(".csv"), index=False)
            except (OSError, PermissionError):
                # CSV export is optional; ignore failures
                pass


# -----------------------------------------------------------------------------
# Global FDR across all tests (subject-level)
# -----------------------------------------------------------------------------

def apply_global_fdr(subject: str, alpha: float = 0.05) -> None:
    """Apply a single Benjamini–Hochberg FDR across all subject tests.

    - Aggregates p-values from all relevant stats TSVs for the subject.
    - Prefers permutation p-values (column 'p_perm') when available; else uses 'p'.
    - Computes global BH q-values and rejection at ``alpha``.
    - Updates each TSV with new columns:
        * 'p_used_for_global_fdr'
        * 'q_fdr_global'
        * 'fdr_reject_global'
        * 'fdr_crit_p_global' (same value across all rows/files)
    - Writes a summary TSV 'global_fdr_summary.tsv' to the subject stats dir.
    """
    logger = _setup_logging(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(stats_dir)

    # Collect candidate files
    patterns = [
        "corr_stats_pow_roi_vs_rating.tsv",
        "corr_stats_pow_roi_vs_temp.tsv",
        "corr_stats_conn_roi_summary_*_vs_rating.tsv",
        "corr_stats_conn_roi_summary_*_vs_temp.tsv",
        "corr_stats_edges_*_vs_rating.tsv",
        "corr_stats_edges_*_vs_temp.tsv",
    ]

    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(stats_dir.glob(pat)))

    if not files:
        logger.info(f"No stats TSVs found for global FDR in {stats_dir}")
        return

    # Helpers to parse metadata from filenames
    def _parse_analysis_type(name: str) -> str:
        if name.startswith("corr_stats_pow_roi"):
            return "pow_roi"
        if name.startswith("corr_stats_conn_roi_summary"):
            return "conn_roi_summary"
        if name.startswith("corr_stats_edges"):
            return "conn_edges"
        return "other"

    def _parse_target(name: str) -> str:
        if "_vs_" in name:
            return name.split("_vs_", 1)[1].split(".", 1)[0]
        return ""

    def _parse_measure_band(analysis_type: str, name: str) -> str:
        if analysis_type == "conn_edges" and name.startswith("corr_stats_edges_"):
            tail = name[len("corr_stats_edges_"):]
            return tail.split("_vs_", 1)[0]
        if analysis_type == "conn_roi_summary" and name.startswith("corr_stats_conn_roi_summary_"):
            tail = name[len("corr_stats_conn_roi_summary_"):]
            return tail.split("_vs_", 1)[0]
        return ""

    # Aggregate p-values and keep references for back-writing
    all_p: List[float] = []
    refs: List[Tuple[Path, int]] = []  # (file_path, row_idx)
    metas: List[Dict[str, object]] = []

    for f in files:
        try:
            df = pd.read_csv(f, sep="\t")
        except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if df is None or df.empty:
            continue

        name = f.name
        analysis_type = _parse_analysis_type(name)
        target = _parse_target(name)
        measure_band_from_name = _parse_measure_band(analysis_type, name)

        # Choose per-row p-value: prefer permutation p when available, else raw p
        p_perm_ser = pd.to_numeric(df["p_perm"], errors="coerce") if "p_perm" in df.columns else pd.Series(np.nan, index=df.index)
        p_raw_ser = pd.to_numeric(df["p"], errors="coerce") if "p" in df.columns else pd.Series(np.nan, index=df.index)
        # Use permutation p when finite, otherwise fallback to raw p
        pser = p_perm_ser.where(np.isfinite(p_perm_ser), p_raw_ser)
        mask = np.isfinite(pser.to_numpy())
        if not np.any(mask):
            continue

        # Build per-row metadata and aggregate
        for idx, used in enumerate(mask):
            if not used:
                continue
            pval = float(pser.iloc[idx])
            all_p.append(pval)
            refs.append((f, idx))

            meta: Dict[str, object] = {
                "source_file": f.name,
                "analysis_type": analysis_type,
                "target": target,
                "measure_band": measure_band_from_name,
                "row_index": int(idx),
            }
            # Track the source of the p-value for this row
            try:
                src = "p_perm" if np.isfinite(p_perm_ser.iloc[idx]) else ("p" if np.isfinite(p_raw_ser.iloc[idx]) else "")
            except (IndexError, KeyError):
                src = ""
            if src:
                meta["p_used_source"] = src
            # Optional rich labels depending on file contents
            try:
                if analysis_type == "pow_roi":
                    roi = df.get("roi", pd.Series([""] * len(df))).iloc[idx]
                    band = df.get("band", pd.Series([""] * len(df))).iloc[idx]
                    meta.update({"roi": roi, "band": band})
                    meta["test_label"] = f"pow_{band}_ROI {roi} vs {target}"
                elif analysis_type == "conn_roi_summary":
                    roi_i = df.get("roi_i", pd.Series([""] * len(df))).iloc[idx]
                    roi_j = df.get("roi_j", pd.Series([""] * len(df))).iloc[idx]
                    summ = df.get("summary_type", pd.Series([""] * len(df))).iloc[idx]
                    meas = df.get("measure_band", pd.Series([measure_band_from_name] * len(df))).iloc[idx]
                    meta.update({"roi_i": roi_i, "roi_j": roi_j, "summary_type": summ, "measure_band": meas})
                    meta["test_label"] = f"{meas} {roi_i}-{roi_j} ({summ}) vs {target}"
                elif analysis_type == "conn_edges":
                    ni = df.get("node_i", pd.Series([""] * len(df))).iloc[idx]
                    nj = df.get("node_j", pd.Series([""] * len(df))).iloc[idx]
                    meta.update({"node_i": ni, "node_j": nj})
                    meas = measure_band_from_name
                    meta["test_label"] = f"{meas} edge {ni}-{nj} vs {target}"
                else:
                    meta["test_label"] = f"{name}[{idx}]"
            except (IndexError, KeyError, ValueError):
                meta["test_label"] = f"{name}[{idx}]"
            metas.append(meta)

    # Nothing to correct
    if not all_p:
        logger.info("No valid p-values found for global FDR; skipping.")
        return

    p_arr = np.asarray(all_p, dtype=float)
    # Compute q-values and rejection using BH
    q_arr = _bh_adjust(p_arr)
    rej_arr, crit_p = _fdr_bh(p_arr, alpha=float(alpha))

    # Prepare updates per file
    updates: Dict[Path, List[Tuple[int, float, bool, float]]] = {}
    for k, (f, row_idx) in enumerate(refs):
        updates.setdefault(f, []).append((row_idx, float(q_arr[k]), bool(rej_arr[k]), float(p_arr[k])))

    # Write back to each file
    for f, items in updates.items():
        try:
            df = pd.read_csv(f, sep="\t")
        except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if df is None or df.empty:
            continue
        nrows = len(df)
        qcol = np.full(nrows, np.nan, dtype=float)
        rcol = np.zeros(nrows, dtype=bool)
        pused = np.full(nrows, np.nan, dtype=float)
        for (row_idx, qv, rj, pu) in items:
            if 0 <= int(row_idx) < nrows:
                qcol[int(row_idx)] = qv
                rcol[int(row_idx)] = rj
                pused[int(row_idx)] = pu
        df["p_used_for_global_fdr"] = pused
        df["q_fdr_global"] = qcol
        df["fdr_reject_global"] = rcol
        df["fdr_crit_p_global"] = float(crit_p)
        try:
            df.to_csv(f, sep="\t", index=False)
        except (OSError, PermissionError):
            pass

    # Build and save a global summary TSV
    summary_rows: List[Dict[str, object]] = []
    for k, meta in enumerate(metas):
        row = dict(meta)
        row["p_used_for_global_fdr"] = float(p_arr[k])
        row["q_fdr_global"] = float(q_arr[k])
        row["fdr_reject_global"] = bool(rej_arr[k])
        row["fdr_crit_p_global"] = float(crit_p)
        summary_rows.append(row)

    try:
        df_sum = pd.DataFrame(summary_rows)
        df_sum.to_csv(stats_dir / "global_fdr_summary.tsv", sep="\t", index=False)
    except (OSError, PermissionError):
        pass


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def process_subject(
    subject: str,
    task: str = TASK,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    build_report: bool = False,
    rng_seed: int = 42,
) -> None:
    logger = _setup_logging(subject)
    logger.info(f"=== Behavior-feature analyses: sub-{subject}, task-{task} ===")
    
    # Initialize shared RNG for consistent but independent randomization across functions
    rng = np.random.default_rng(rng_seed)
    try:
        plot_psychometrics(subject, task)
    except Exception as e:
        logger.error(f"Psychometric plotting failed for sub-{subject}: {e}")
    try:
        correlate_power_roi_stats(
            subject,
            task,
            use_spearman=use_spearman,
            partial_covars=partial_covars,
            bootstrap=bootstrap,
            n_perm=n_perm,
            rng=rng,
        )
    except Exception as e:
        logger.error(f"ROI power correlations failed for sub-{subject}: {e}")
    try:
        plot_power_roi_scatter(
            subject,
            task,
            use_spearman=use_spearman,
            partial_covars=partial_covars,
            bootstrap_ci=bootstrap,
            rng=rng,
        )
    except Exception as e:
        logger.error(f"ROI power scatter plotting failed for sub-{subject}: {e}")
    
    # Power-behavior correlation plots (transferred from 03_feature_engineering.py)
    try:
        pow_df, _, y, _ = _load_features_and_targets(subject, task)
        plots_dir = _plots_dir(subject)
        plot_power_behavior_correlation(pow_df, y, POWER_BANDS_TO_USE, subject, plots_dir, logger)
    except Exception as e:
        logger.error(f"Power-behavior correlation plotting failed for sub-{subject}: {e}")

    # Advanced feature vs behavior visualizations
    try:
        pow_df, _, y, info = _load_features_and_targets(subject, task)
        plots_dir = _plots_dir(subject)
        
        # Load events for behavioral pattern analysis
        events = _load_events_df(subject, task)
        aligned_events = _align_events_to_epochs(events, mne.read_epochs(_find_clean_epochs_path(subject, task), preload=False, verbose=False)) if events is not None else None
        
        # 1. Power-behavior correlation matrix
        plot_power_behavior_correlation_matrix(pow_df, y, POWER_BANDS_TO_USE, subject, plots_dir, logger)
        
        # 2. Behavioral response patterns
        plot_behavioral_response_patterns(y, aligned_events, subject, plots_dir, logger)
        
        # 4. Behavioral predictor importance ranking
        plot_behavioral_predictor_importance(pow_df, y, POWER_BANDS_TO_USE, subject, plots_dir, logger)
        
        # 5. Publication-quality EEG power spectrogram with behavior overlay
        plot_power_spectrogram_with_behavior(pow_df, y, POWER_BANDS_TO_USE, subject, plots_dir, logger)
        
        # 5b. Temperature-based spectrograms
        plot_power_spectrogram_temperature_band(pow_df, aligned_events, POWER_BANDS_TO_USE, subject, plots_dir, logger)
        
        # 6. Topographic correlation maps (removed - MNE compatibility issues)
        # plot_topographic_correlation_maps(pow_df, y, POWER_BANDS_TO_USE, subject, plots_dir, logger)
        
        # 7. Neural state space analysis (PCA)
        plot_neural_state_pca(pow_df, y, subject, plots_dir, logger)
        
        # 8. Band power summary
        plot_band_power_summary(pow_df, POWER_BANDS_TO_USE, subject, plots_dir, logger)
        
        # 9. Statistical effect size summary (4 separate figures)
        plot_statistical_effect_size_summary(pow_df, y, POWER_BANDS_TO_USE, subject, plots_dir, logger)
        
    except Exception as e:
        logger.error(f"Advanced behavior visualization plots failed for sub-{subject}: {e}")
    
    # Advanced connectivity analyses
    try:
        plots_dir = _plots_dir(subject)
        
        # Behavior-modulated connectivity networks
        plot_behavior_modulated_connectivity(subject, task, y, plots_dir, logger)
        
    except Exception as e:
        logger.error(f"Advanced connectivity visualization plots failed for sub-{subject}: {e}")
    
    try:
        correlate_connectivity_heatmaps(subject, task, use_spearman=use_spearman)
    except Exception as e:
        logger.error(f"Connectivity correlations failed for sub-{subject}: {e}")
    try:
        correlate_connectivity_roi_summaries(
            subject,
            task,
            use_spearman=use_spearman,
            partial_covars=partial_covars,
            bootstrap=bootstrap,
            n_perm=n_perm,
            rng=rng,
        )
    except Exception as e:
        logger.error(f"Connectivity ROI summaries failed for sub-{subject}: {e}")

    # Combine per-band power correlation stats into consolidated TSV/CSV
    try:
        export_combined_power_corr_stats(subject)
    except Exception as e:
        logger.error(f"Combined power corr stats export failed for sub-{subject}: {e}")

    # Apply a subject-level global FDR across all tests
    try:
        apply_global_fdr(subject)
    except Exception as e:
        logger.error(f"Global FDR application failed for sub-{subject}: {e}")

    if build_report:
        try:
            build_subject_report(subject, task)
        except Exception as e:
            logger.error(f"Report build failed for sub-{subject}: {e}")


def _fisher_aggregate(rs: List[float]) -> Tuple[float, float, float, int]:
    """Aggregate correlations across subjects via Fisher z.

    Returns: (r_group, ci_low, ci_high, n)
    """
    vals = np.array([r for r in rs if np.isfinite(r)])
    vals = np.clip(vals, -0.999999, 0.999999)
    n = vals.size
    if n < 2:
        return np.nan, np.nan, np.nan, n
    z = np.arctanh(vals)
    mean_z = float(np.mean(z))
    sd_z = float(np.std(z, ddof=1))
    se = sd_z / np.sqrt(n) if sd_z > 0 else np.nan
    if np.isnan(se) or se == 0:
        return float(np.tanh(mean_z)), np.nan, np.nan, n
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    ci_low_z = mean_z - tcrit * se
    ci_high_z = mean_z + tcrit * se
    return float(np.tanh(mean_z)), float(np.tanh(ci_low_z)), float(np.tanh(ci_high_z)), n


def aggregate_group_level(subjects: Optional[List[str]] = None, task: str = TASK) -> None:
    if subjects is None or subjects == ["all"]:
        subjects = SUBJECTS
    gstats = _group_stats_dir()
    gplots = _group_plots_dir()
    _ensure_dir(gstats)
    _ensure_dir(gplots)

    # 1) ROI power vs rating
    by_key: Dict[Tuple[str, str], List[float]] = {}
    for sub in subjects:
        f = _stats_dir(sub) / "corr_stats_pow_roi_vs_rating.tsv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t")
        for _, row in df.iterrows():
            key = (str(row.get("roi")), str(row.get("band")))
            r = row.get("r")
            try:
                r = float(r)
            except (ValueError, TypeError):
                r = np.nan
            by_key.setdefault(key, []).append(r)
    if by_key:
        recs = []
        for (roi, band), rs in by_key.items():
            r_grp, ci_l, ci_h, n = _fisher_aggregate(rs)
            recs.append({
                "roi": roi,
                "band": band,
                "r_group": r_grp,
                "r_ci_low": ci_l,
                "r_ci_high": ci_h,
                "n_subjects": n,
            })
        dfg = pd.DataFrame(recs)
        # p-values via one-sample t-test on Fisher z against 0
        pvals = []
        for (roi, band), rs in by_key.items():
            vals = np.array([r for r in rs if np.isfinite(r)])
            vals = np.clip(vals, -0.999999, 0.999999)
            n = vals.size
            if n < 2:
                pvals.append(np.nan)
                continue
            z = np.arctanh(vals)
            tstat, p = stats.ttest_1samp(z, popmean=0.0)
            pvals.append(float(p))
        dfg["p_group"] = pvals
        # FDR per band
        out_rows = []
        for band in sorted(dfg["band"].unique()):
            dfb = dfg[dfg["band"] == band].copy()
            rej, crit = _fdr_bh(dfb["p_group"].to_numpy(), alpha=0.05)
            dfb["fdr_reject"] = rej
            dfb["fdr_crit_p"] = crit
            out_rows.append(dfb)
        dfg2 = pd.concat(out_rows, ignore_index=True)
        dfg2.to_csv(gstats / "group_corr_pow_roi_vs_rating.tsv", sep="\t", index=False)

        # Simple plot: bar of r_group by ROI per band
        try:
            for band in sorted(dfg2["band"].unique()):
                dfb = dfg2[dfg2["band"] == band]
                fig, ax = plt.subplots(figsize=(6, 3.2))
                order = sorted(dfb["roi"].unique())
                sns.barplot(data=dfb, x="roi", y="r_group", order=order, color="steelblue", ax=ax)
                # Add error bars using CI
                for i, roi in enumerate(order):
                    row = dfb[dfb["roi"] == roi].iloc[0]
                    yv = row["r_group"]
                    yerr_low = yv - row["r_ci_low"] if np.isfinite(row["r_ci_low"]) else 0
                    yerr_high = row["r_ci_high"] - yv if np.isfinite(row["r_ci_high"]) else 0
                    ax.errorbar(i, yv, yerr=[[yerr_low], [yerr_high]], fmt="none", ecolor="k", capsize=3)
                ax.set_ylabel("Group r (Fisher back-transformed)")
                ax.set_xlabel("ROI")
                band_rng = FEATURES_FREQ_BANDS.get(band)
                band_label = f"{band} ({band_rng[0]:g}\u2013{band_rng[1]:g} Hz)" if band_rng is not None else band
                ax.set_title(f"Group ROI power vs rating: {band_label}")
                ax.axhline(0, color="k", linewidth=0.8)
                fig.tight_layout()
                _save_fig(fig, gplots / f"group_roi_power_vs_rating_{_sanitize(band)}")
        except (ValueError, KeyError, OSError):
            pass

    # 2) Connectivity ROI summaries per measure_band
    # Collect per subject files by measure_band
    per_pref: Dict[str, List[pd.DataFrame]] = {}
    for sub in subjects:
        subj_stats = _stats_dir(sub)
        if not subj_stats.exists():
            continue
        for f in subj_stats.glob("corr_stats_conn_roi_summary_*_vs_rating.tsv"):
            try:
                df = pd.read_csv(f, sep="\t")
            except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if df.empty or "measure_band" not in df.columns:
                continue
            per_pref.setdefault(str(df["measure_band"].iloc[0]), []).append(df)
    for pref, dfs in per_pref.items():
        # Concatenate and aggregate per (roi_i, roi_j)
        cat = pd.concat(dfs, ignore_index=True)
        out_rows = []
        for (roi_i, roi_j), grp in cat.groupby(["roi_i", "roi_j"], dropna=False):
            rs = grp["r"].to_numpy(dtype=float)
            r_grp, ci_l, ci_h, n = _fisher_aggregate(rs.tolist())
            # p-value via t-test on Fisher z
            vals = np.clip(rs[np.isfinite(rs)], -0.999999, 0.999999)
            if vals.size >= 2:
                tstat, p = stats.ttest_1samp(np.arctanh(vals), popmean=0.0)
                pval = float(p)
            else:
                pval = np.nan
            out_rows.append({
                "measure_band": pref,
                "roi_i": roi_i,
                "roi_j": roi_j,
                "summary_type": "within" if roi_i == roi_j else "between",
                "r_group": r_grp,
                "r_ci_low": ci_l,
                "r_ci_high": ci_h,
                "n_subjects": n,
                "p_group": pval,
            })
        out_df = pd.DataFrame(out_rows)
        rej, crit = _fdr_bh(out_df["p_group"].to_numpy(), alpha=0.05)
        out_df["fdr_reject"] = rej
        out_df["fdr_crit_p"] = crit
        out_df.to_csv(gstats / f"group_corr_conn_roi_summary_{_sanitize(pref)}_vs_rating.tsv", sep="\t", index=False)


def build_subject_report(subject: str, task: str = TASK) -> None:
    try:
        Report = getattr(mne, "Report")
    except AttributeError:
        Report = None
    if Report is None:
        print("mne.Report not available; skipping report build.")
        return
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    report_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "report"
    _ensure_dir(report_dir)
    rep = Report(title=f"Behavior & EEG features: sub-{subject}, task-{task}")
    # Psychometric plots
    for fn in ["psychometric_rating_vs_temp.png", "psychometric_pain_vs_temp.png"]:
        p = plots_dir / fn
        if p.exists():
            rep.add_image(p, title=fn, section="Psychometrics")
    # Save stats TSVs as links
    for fn in [
        "corr_stats_pow_roi_vs_rating.tsv",
        "corr_stats_pow_roi_vs_temp.tsv",
    ]:
        p = stats_dir / fn
        if p.exists():
            rep.add_html(f"<p><a href='{p.as_posix()}'>Download {fn}</a></p>", title=fn, section="Stats")
    # Add global FDR summary if present
    p = stats_dir / "global_fdr_summary.tsv"
    if p.exists():
        rep.add_html(f"<p><a href='{p.as_posix()}'>Download global_fdr_summary.tsv</a></p>", title="global_fdr_summary.tsv", section="Stats")
    # Add connectivity ROI summary TSVs (atlas-aware) if present
    try:
        for p in sorted(stats_dir.glob("corr_stats_conn_roi_summary_*_vs_rating.tsv")):
            fn = p.name
            rep.add_html(f"<p><a href='{p.as_posix()}'>Download {fn}</a></p>", title=fn, section="Stats")
        # Add connectivity ROI summary temperature TSVs if present
        for p in sorted(stats_dir.glob("corr_stats_conn_roi_summary_*_vs_temp.tsv")):
            fn = p.name
            rep.add_html(f"<p><a href='{p.as_posix()}'>Download {fn}</a></p>", title=fn, section="Stats")
        # Add top-20 edge result TSVs if present
        for p in sorted(stats_dir.glob("corr_stats_edges_*_vs_rating_top20.tsv")):
            fn = p.name
            rep.add_html(f"<p><a href='{p.as_posix()}'>Download {fn}</a></p>", title=fn, section="Stats")
    except (OSError, AttributeError, ValueError):
        pass
    out_html = report_dir / f"report_sub-{subject}_task-{task}_behavior_features.html"
    rep.save(out_html, overwrite=True, open_browser=False)


def main(
    subjects: Optional[List[str]] = None,
    task: str = TASK,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    do_group: bool = False,
    group_only: bool = False,
    build_reports: bool = False,
    rng_seed: int = 42,
):
    if subjects is None or subjects == ["all"]:
        subjects = SUBJECTS
    if not group_only:
        for sub in subjects:
            process_subject(
                sub,
                task,
                use_spearman=use_spearman,
                partial_covars=partial_covars,
                bootstrap=bootstrap,
                n_perm=n_perm,
                build_report=build_reports,
                rng_seed=rng_seed,
            )
    if do_group or group_only:
        aggregate_group_level(subjects, task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Behavioral psychometrics and EEG feature correlations")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject IDs to process (e.g., 001 002) or 'all' for all configured subjects")
    parser.add_argument("--task", default=TASK, help="Task label (default from config)")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=BOOTSTRAP_DEFAULT,
        help="Number of bootstrap resamples for Spearman CI in scatter plots (0 disables)",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Random seed for reproducible bootstrap and permutation tests",
    )
    args = parser.parse_args()

    subs = None if args.subjects in (None, [], ["all"]) else args.subjects
    main(
        subs,
        task=args.task,
        use_spearman=USE_SPEARMAN_DEFAULT,
        partial_covars=PARTIAL_COVARS_DEFAULT,
        bootstrap=args.bootstrap,
        n_perm=N_PERM_DEFAULT,
        do_group=DO_GROUP_DEFAULT,
        group_only=GROUP_ONLY_DEFAULT,
        build_reports=BUILD_REPORTS_DEFAULT,
        rng_seed=args.rng_seed,
    )
