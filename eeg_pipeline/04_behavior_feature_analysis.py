from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, StrMethodFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import mne
from mne_bids import BIDSPath

# Note: Matplotlib setup is now handled by config.setup_matplotlib()

# ==========================
# CONFIG
# Load centralized configuration from YAML
# ==========================
from utils.config_loader import load_config, get_legacy_constants
from utils.logging_utils import get_subject_logger, get_group_logger
from utils.io_utils import (
    _find_clean_epochs_path as _find_clean_epochs_path,
    _load_events_df as _load_events_df,
    _align_events_to_epochs as _align_events_to_epochs,
)

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
BAND_COLORS = _constants["BAND_COLORS"]

# Global strict mode: abort on misalignment or length mismatches instead of trimming
STRICT_MODE = bool(config.get("analysis.strict_mode", True))

# Behavior analysis config (important parameters)
BEHAV_FDR_ALPHA = float(config.get("behavior_analysis.statistics.fdr_alpha", 0.05))
BEHAV_MIN_CORR_SAMPLES = int(config.get("behavior_analysis.statistics.min_correlation_samples", 10))
BEHAV_BOOTSTRAP_N = int(config.get("behavior_analysis.statistics.bootstrap_n", 1000))

# Visualization annotation toggles
ANNOTATE_FDR = bool(config.get("behavior_analysis.visualization.annotate_fdr_note", True))

# Spline smoothing parameters for diagnostics
SPLINE_SMOOTHING_FRAC = float(config.get("behavior_analysis.spline.smoothing_factor", 0.3))
SPLINE_MIN_SMOOTHING = float(config.get("behavior_analysis.spline.min_smoothing", 1.0))
SPLINE_MAX_ITER = int(config.get("behavior_analysis.spline.max_iter", 50))
SPLINE_TOL = float(config.get("behavior_analysis.spline.tolerance", 1e-3))

# Visualization smoothing for time series
VIZ_SMOOTHING_SIGMA = float(config.get("behavior_analysis.visualization.smoothing_sigma", 2.0))

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))


def _save_fig(fig: matplotlib.figure.Figure, path_base: Path | str, formats: Tuple[str, ...] = SAVE_FORMATS) -> None:
    """Save figure to multiple formats at FIG_DPI and close with footer annotations."""
    base = Path(path_base)
    if base.suffix:
        base = base.with_suffix("")
    # Add footer with baseline window and FDR alpha for transparency
    try:
        bwin = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))
        foot = f"Baseline: [{float(bwin[0]):.2f}, {float(bwin[1]):.2f}] s | FDR alpha: {BEHAV_FDR_ALPHA}"
        try:
            fig.text(0.01, 0.01, foot, fontsize=8, alpha=0.8)
        except Exception:
            pass
    except Exception:
        pass
    for ext in formats:
        try:
            fig.savefig(base.with_suffix(f".{ext}"), dpi=FIG_DPI, bbox_inches="tight")
        except Exception:
            try:
                fig.savefig(base.with_suffix(f".{ext}"), dpi=FIG_DPI)
            except Exception:
                pass
    plt.close(fig)


def _get_band_color(band: str) -> str:
    """Return color for a band with safe defaults.

    Prefers BAND_COLORS from legacy constants; falls back to a sensible palette.
    """
    try:
        if isinstance(BAND_COLORS, dict) and band in BAND_COLORS:
            return str(BAND_COLORS[band])
    except Exception:
        pass
    fallback = {"delta": "#4169e1", "theta": "purple", "alpha": "green", "beta": "orange", "gamma": "red"}
    return fallback.get(band, "#1f77b4")

def _logratio_to_pct(v):
    """Transform logratio log10(power/baseline) to percent change (%).

    Accepts scalar or array-like. Returns values in percent.
    """
    v_arr = np.asarray(v, dtype=float)
    return (np.power(10.0, v_arr) - 1.0) * 100.0


def _pct_to_logratio(p):
    """Inverse transform: percent change (%) to logratio log10(power/baseline).

    Accepts scalar or array-like. Clips 1 + p/100 to a small positive
    minimum (1e-9) to avoid log10 of non-positive values.
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


## _find_clean_epochs_path imported from io_utils


## _load_events_df imported from io_utils


## _align_events_to_epochs imported from io_utils


def _pick_first_column(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[str]:
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None



def _canonical_covariate_name(name: Optional[str]) -> Optional[str]:
    """Map covariate column variants to canonical labels.

    Standardizes known aliases such that pooled design matrices share the same
    column names across subjects.
    """
    if name is None:
        return None
    n = str(name).lower()
    temp_aliases = {c.lower() for c in PSYCH_TEMP_COLUMNS}
    trial_aliases = {"trial", "trial_number", "trial_index"}
    if n in temp_aliases:
        return "temperature"
    if n in trial_aliases:
        return "trial"
    return n



def _build_covariate_matrices(
    df_events: Optional[pd.DataFrame],
    partial_covars: Optional[List[str]],
    temp_col: Optional[str],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Construct design matrices for partial correlations.

    Returns ``(Z_full, Z_temp)`` where ``Z_full`` includes all covariates and
    ``Z_temp`` drops the temperature column (when present) for temperature
    correlations.
    """
    if df_events is None:
        return None, None

    covars = []
    name_map: Dict[str, str] = {}
    if partial_covars:
        for c in partial_covars:
            if c in df_events.columns:
                covars.append(c)
                name_map[c] = _canonical_covariate_name(c)
            else:
                canon = _canonical_covariate_name(c)
                if canon == "temperature":
                    tcol = _pick_first_column(df_events, PSYCH_TEMP_COLUMNS)
                    if tcol is not None:
                        covars.append(tcol)
                        name_map[tcol] = canon
                elif canon == "trial":
                    tcol = _pick_first_column(df_events, ["trial", "trial_number", "trial_index"])
                    if tcol is not None:
                        covars.append(tcol)
                        name_map[tcol] = canon
    else:
        tcol = _pick_first_column(df_events, PSYCH_TEMP_COLUMNS)
        if tcol is not None:
            covars.append(tcol)
            name_map[tcol] = "temperature"
        trialc = _pick_first_column(df_events, ["trial", "trial_number", "trial_index", "run", "block"])
        if trialc is not None:
            covars.append(trialc)
            name_map[trialc] = _canonical_covariate_name(trialc)

    if not covars:
        return None, None
    Z = pd.DataFrame()
    for c in covars:
        if c in df_events.columns:

            Z[name_map.get(c, c)] = pd.to_numeric(df_events[c], errors="coerce")
    if Z.empty:
        return None, None
    temp_col_can = _canonical_covariate_name(temp_col) if temp_col else None
    Z_temp = Z.drop(columns=[temp_col_can], errors="ignore") if temp_col_can else Z.copy()

    if Z_temp.shape[1] == 0:
        Z_temp = None
    return Z, Z_temp


def _partial_corr_xy_given_Z(
    x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str
) -> Tuple[float, float, int]:
    """Compute partial correlation of ``x`` and ``y`` controlling for ``Z``."""
    df_full = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1)
    df = df_full.dropna()
    if len(df) < 5 or df["y"].nunique() <= 1:
        return np.nan, np.nan, 0
    if method == "spearman":
        xr = stats.rankdata(df["x"].to_numpy())
        yr = stats.rankdata(df["y"].to_numpy())
        Zr = (
            np.column_stack([stats.rankdata(df[c].to_numpy()) for c in Z.columns])
            if len(Z.columns)
            else np.empty((len(df), 0))
        )
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


def _partial_residuals_xy_given_Z(
    x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str
) -> Tuple[pd.Series, pd.Series, int]:
    """Compute partial residuals of ``x`` and ``y`` after regressing out ``Z``."""
    df_full = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1)
    df = df_full.dropna()
    if len(df) < 5 or df["y"].nunique() <= 1:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    if method == "spearman":
        xr = stats.rankdata(df["x"].to_numpy())
        yr = stats.rankdata(df["y"].to_numpy())
        Zr = (
            np.column_stack([stats.rankdata(df[c].to_numpy()) for c in Z.columns])
            if len(Z.columns)
            else np.empty((len(df), 0))
        )
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
    return (
        pd.Series(x_res, index=df.index),
        pd.Series(y_res, index=df.index),
        int(len(df)),
    )


def _features_dir(subject: str) -> Path:
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "features"


def _plots_dir(subject: str) -> Path:
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots" / "04_behavior_feature_analysis"


def _stats_dir(subject: str) -> Path:
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "stats"


def _group_stats_dir() -> Path:
    return DERIV_ROOT / "group" / "eeg" / "stats"


def _group_plots_dir() -> Path:
    # Keep plots organized under a script-specific subfolder (mirrors subject-level)
    return DERIV_ROOT / "group" / "eeg" / "plots" / "04_behavior_feature_analysis"


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
    """Backward-compatible wrapper using centralized logging utils."""
    return get_subject_logger("behavior_analysis", subject, LOG_FILE_NAME)


def _setup_group_logging() -> logging.Logger:
    """Backward-compatible wrapper using centralized logging utils."""
    return get_group_logger("behavior_analysis", LOG_FILE_NAME)


# ROI utilities centralized to avoid drift across scripts
from utils.roi_utils import build_rois_from_info as _build_rois


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
            
            # Create presentable axis labels without underscores
            temp_label = temp_col.replace('_', ' ').title()
            rating_label = rating_col.replace('_', ' ').title()
            
            ax.set_xlabel(f"Temperature")
            ax.set_ylabel(f"Rating")
            ax.set_title("Rating vs Temperature")
            ax.grid(True, alpha=0.3)
            
            # Calculate and format statistics
            sr, sp = stats.spearmanr(t, r, nan_policy="omit")
            
            # Format p-value properly
            if sp < 0.001:
                p_text = "p < .001"
            elif sp < 0.01:
                p_text = f"p < .01"
            elif sp < 0.05:
                p_text = f"p < .05"
            else:
                p_text = f"p = {sp:.3f}"
            
            # Add statistical annotation to the plot
            stats_text = f'ρ = {sr:.3f}, {p_text}'
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontweight='bold', fontsize=10)

            # Align x-axis ticks with actual temperature values to match dots
            try:
                t_vals = np.asarray(t, dtype=float)
                uniq = np.unique(np.round(t_vals, 6))
                # If temperatures take on a small number of discrete values, use them as ticks
                if uniq.size <= 12:
                    ax.set_xticks(uniq)
                    # If all are integers, force integer locator to prevent half-step misalignment
                    if np.allclose(uniq, uniq.astype(int)):
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
                    else:
                        # Otherwise show up to 2 decimals for readability
                        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
                # Set tight x-limits with a small margin to avoid tick/point edge clipping
                if np.isfinite(t_vals).any():
                    xmin = np.nanmin(t_vals)
                    xmax = np.nanmax(t_vals)
                    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                        pad = 0.02 * (xmax - xmin) if xmax - xmin > 0 else 0.5
                        ax.set_xlim(xmin - pad, xmax + pad)
            except Exception:
                pass
            
            _save_fig(fig, plots_dir / "psychometric_rating_vs_temp")
            plt.close(fig)
            
            # Save Spearman only (consistent metric)
            pd.DataFrame({
                "metric": ["spearman_r", "spearman_p"],
                "value": [sr, sp],
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
# Shared visualization utilities
# -----------------------------------------------------------------------------


def _sig_color(p: float) -> str:
    """Return a color based on statistical significance."""
    return "#C42847" if (np.isfinite(p) and p < 0.05) else "#666666"


def _fisher_ci_r(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Fisher z-transform based confidence interval for correlations."""
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
    """Bootstrap confidence interval for Spearman correlation (consistent metric)."""
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
        # Always compute Spearman correlation
        rb, _ = stats.spearmanr(xb, yb, nan_policy="omit")
        if np.isfinite(rb):
            vals.append(float(rb))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _generate_correlation_scatter(
    x_data: pd.Series,
    y_data: pd.Series,
    x_label: str,
    y_label: str,
    title_prefix: str,
    band_color: str,
    output_path: Path,
    *,
    method_code: str = "spearman",
    Z_covars: Optional[pd.DataFrame] = None,
    covar_names: Optional[List[str]] = None,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    is_partial_residuals: bool = False,
    roi_channels: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
    annotated_stats: Optional[Tuple[float, float, int]] = None,
    annot_ci: Optional[Tuple[float, float]] = None,
    stats_tag: Optional[str] = None,
) -> Tuple[float, float, int]:
    """Generate correlation scatter plots with marginal histograms and stats.

    Returns the correlation coefficient, p-value and effective sample size.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if len(x_data) != len(y_data):
        if STRICT_MODE:
            raise ValueError(
                f"Strict mode: length mismatch in correlation inputs (x={len(x_data)}, y={len(y_data)})."
            )
        logger.warning(
            f"Data length mismatch: x={len(x_data)}, y={len(y_data)}. Using overlap for correlation."
        )
    n_len = min(len(x_data), len(y_data))
    x = x_data.iloc[:n_len] if hasattr(x_data, "iloc") else x_data[:n_len]
    y = y_data.iloc[:n_len] if hasattr(y_data, "iloc") else y_data[:n_len]

    if is_partial_residuals:
        m = pd.Series([True] * len(x), index=x.index if hasattr(x, "index") else range(len(x)))
        n_eff = len(x)
        x_clean = x
        y_clean = y
    else:
        m = x.notna() & y.notna()
        n_eff = int(m.sum())
        x_clean = x[m]
        y_clean = y[m]

    if n_eff < 5:
        return np.nan, np.nan, n_eff

    if is_partial_residuals:
        # Always use Spearman for direct correlations
        r, p = stats.spearmanr(x_clean, y_clean, nan_policy="omit")
        r_part, p_part, n_part = np.nan, np.nan, 0
    else:
        # Always use Spearman for direct correlations
        r, p = stats.spearmanr(x_clean, y_clean, nan_policy="omit")
        r_part, p_part, n_part = np.nan, np.nan, 0
        if Z_covars is not None and len(Z_covars) > 0:
            n_len_pt = min(len(x), len(y), len(Z_covars))
            r_part, p_part, n_part = _partial_corr_xy_given_Z(
                x.iloc[:n_len_pt], y.iloc[:n_len_pt], Z_covars.iloc[:n_len_pt], method_code
            )

    # Use bootstrap CI for Spearman when requested
    if bootstrap_ci > 0:
        ci = _bootstrap_corr_ci(x_clean, y_clean, method_code, n_boot=int(bootstrap_ci), rng=rng)
    else:
        ci = (np.nan, np.nan)

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.15,
        wspace=0.15,
        left=0.1,
        right=0.95,
        top=0.80,
        bottom=0.12,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_histx.tick_params(labelbottom=False)
    ax_histy.tick_params(labelleft=False)

    line_color = _sig_color(p)
    sns.regplot(
        x=x_clean,
        y=y_clean,
        ax=ax_main,
        ci=95,
        scatter_kws={"s": 30, "alpha": 0.7, "color": band_color, "edgecolor": "white", "linewidths": 0.3},
        line_kws={"color": line_color, "lw": 1.5},
    )

    ax_histx.hist(x_clean, bins=15, color=band_color, alpha=0.7, edgecolor="white", linewidth=0.5)
    try:
        from scipy.stats import gaussian_kde

        if len(x_clean) > 3:
            kde_x = gaussian_kde(x_clean)
            x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
            kde_vals = kde_x(x_range)
            hist_counts, _ = np.histogram(x_clean, bins=15)
            kde_scale = hist_counts.max() / kde_vals.max() if kde_vals.max() > 0 else 1
            ax_histx.plot(x_range, kde_vals * kde_scale, color="darkblue", linewidth=1.5, alpha=0.8)
    except (ImportError, ValueError, ZeroDivisionError):
        pass

    ax_histy.hist(
        y_clean,
        bins=15,
        orientation="horizontal",
        color=band_color,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    try:
        from scipy.stats import gaussian_kde

        if len(y_clean) > 3:
            kde_y = gaussian_kde(y_clean)
            y_range = np.linspace(y_clean.min(), y_clean.max(), 100)
            kde_vals_y = kde_y(y_range)
            hist_counts_y, _ = np.histogram(y_clean, bins=15)
            kde_scale_y = hist_counts_y.max() / kde_vals_y.max() if kde_vals_y.max() > 0 else 1
            ax_histy.plot(kde_vals_y * kde_scale_y, y_range, color="darkblue", linewidth=1.5, alpha=0.8)
    except (ImportError, ValueError, ZeroDivisionError):
        pass

    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)

    show_pct_axis = False
    if not is_partial_residuals and "log10(power" in x_label:
        show_pct_axis = True
    elif is_partial_residuals and "residuals of log10(power" in x_label:
        show_pct_axis = True

    if show_pct_axis:
        try:
            ax_pct = ax_histx.secondary_xaxis("top", functions=(_logratio_to_pct, _pct_to_logratio))
            ax_pct.set_xlabel("Power Change (%)", fontsize=9)
            # Use a few more major ticks and clean integer labels; add minor ticks for readability
            ax_pct.xaxis.set_major_locator(MaxNLocator(nbins=7))
            ax_pct.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            ax_pct.xaxis.set_minor_locator(AutoMinorLocator(2))
        except (AttributeError, TypeError, ValueError):
            pass

    # Stats box and channel annotation are handled below (figure-level)

    # Figure-level title and stats box for top-of-figure placement
    # Decide which stats to annotate (allows group-level overrides)
    r_disp, p_disp, n_disp = r, p, n_eff
    if annotated_stats is not None:
        try:
            r_disp, p_disp, n_disp = annotated_stats
        except Exception:
            pass
    try:
        label = "Spearman \u03c1"
    except Exception:
        label = "r"
    # Prefer externally provided CI if available
    ci_disp = annot_ci if (annot_ci is not None) else ci
    ci_str = ""
    if ci_disp is not None and np.all(np.isfinite(ci_disp)):
        ci_str = f"\nCI [{ci_disp[0]:.2f}, {ci_disp[1]:.2f}]"
    tag_str = f" {stats_tag}" if stats_tag else ""
    stats_text = f"{label}{tag_str} = {r_disp:.3f}\np = {p_disp:.3f}\nn = {n_disp}{ci_str}"
    fig.text(
        0.98,
        0.94,
        stats_text,
        fontsize=10,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    # Optional ROI channels annotation (if provided)
    if roi_channels:
        try:
            chan_text = "Channels: " + ", ".join(roi_channels[:10])
            fig.text(
                0.02,
                0.94,
                chan_text,
                fontsize=8,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        except Exception:
            pass
    # True figure-level title at top
    fig.suptitle(title_prefix, fontsize=12, fontweight="bold", y=0.975)
    fig.tight_layout()
    _save_fig(fig, output_path)
    return float(r), float(p), int(n_eff)


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
        # Always compute Spearman correlation for permutations
        obs, _ = stats.spearmanr(df["x"], df["y"], nan_policy="omit")
        ge = 1
        y_vals = df["y"].to_numpy()
        for _ in range(int(n_perm)):
            y_pi = y_vals[rng.permutation(len(y_vals))]
            rp, _ = stats.spearmanr(df["x"], y_pi, nan_policy="omit")
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

            # Align lengths for correlation analysis (using overlap)
            if len(roi_vals) != len(y):
                logger.warning(f"Length mismatch for {roi} {band}: power={len(roi_vals)}, rating={len(y)}. "
                              f"Using overlapping trials for correlation.")
            n_len = min(len(roi_vals), len(y))
            x = roi_vals.iloc[:n_len]
            y_r = y.iloc[:n_len]
            m = x.notna() & y_r.notna()
            n_eff = int(m.sum())
            if n_eff >= 5:
                # Always use Spearman
                r, p = stats.spearmanr(x[m], y_r[m], nan_policy="omit")
                method = "spearman"

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
                if bootstrap and n_eff >= 5:
                    idx = np.where(m.to_numpy())[0]
                    boots: List[float] = []
                    for _ in range(int(bootstrap)):
                        bidx = rng.choice(idx, size=len(idx), replace=True)
                        xb = x.iloc[bidx]
                        yb = y_r.iloc[bidx]
                        rb, _ = stats.spearmanr(xb, yb, nan_policy="omit")
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
                if len(roi_vals) != len(temp_series):
                    logger.warning(f"Length mismatch for {roi} {band} vs temperature: "
                                  f"power={len(roi_vals)}, temp={len(temp_series)}. Using overlap.")
                n_len_t = min(len(roi_vals), len(temp_series))
                x2 = roi_vals.iloc[:n_len_t]
                t2 = temp_series.iloc[:n_len_t]
                m2 = x2.notna() & t2.notna()
                n_eff2 = int(m2.sum())
                if n_eff2 >= 5:
                    # Always use Spearman
                    r2, p2 = stats.spearmanr(x2[m2], t2[m2], nan_policy="omit")
                    method2 = "spearman"
                    # CI: bootstrap for Spearman
                    ci2_low = np.nan
                    ci2_high = np.nan
                    if bootstrap and n_eff2 >= 5:
                        idx2 = np.where(m2.to_numpy())[0]
                        boots2: List[float] = []
                        for _ in range(int(bootstrap)):
                            bidx2 = rng.choice(idx2, size=len(idx2), replace=True)
                            xb = x2.iloc[bidx2]
                            tb = t2.iloc[bidx2]
                            rb, _ = stats.spearmanr(xb, tb, nan_policy="omit")
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
        fdr_alpha = config.get('analysis', {}).get('behavior_analysis', {}).get('statistics', {}).get('fdr_alpha', 0.05)
        rej, crit = _fdr_bh(pvec, alpha=fdr_alpha)
        df_r["fdr_reject"] = rej
        df_r["fdr_crit_p"] = crit
        df_r.to_csv(stats_dir / "corr_stats_pow_roi_vs_rating.tsv", sep="\t", index=False)

    if recs_temp:
        df_t = pd.DataFrame(recs_temp)
        pvec_t = df_t["p_perm"].to_numpy() if "p_perm" in df_t.columns and np.isfinite(df_t["p_perm"]).any() else df_t["p"].to_numpy()
        rej_t, crit_t = _fdr_bh(pvec_t, alpha=fdr_alpha)
        df_t["fdr_reject"] = rej_t
        df_t["fdr_crit_p"] = crit_t
        df_t.to_csv(stats_dir / "corr_stats_pow_roi_vs_temp.tsv", sep="\t", index=False)

# -----------------------------------------------------------------------------
# Correlation: Channel-level power vs behavior (per-band correlation TSVs)
# -----------------------------------------------------------------------------

def correlate_power_topomaps(
    subject: str,
    task: str = TASK,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Correlate channel-level power with behavior and export per-band TSVs.
    
    Creates individual correlation files for each frequency band:
    - corr_stats_pow_{band}_vs_rating.tsv  
    - corr_stats_pow_{band}_vs_temp.tsv
    
    These files are then combined by export_combined_power_corr_stats().
    
    Parameters
    ----------
    subject : str
        Subject identifier
    task : str
        Task name
    use_spearman : bool
        Use Spearman (True) or Pearson (False) correlation
    partial_covars : list of str, optional
        Covariates for partial correlation
    bootstrap : int
        Number of bootstrap samples for CI (default 0)
    n_perm : int
        Number of permutation tests (default 0)
    rng : np.random.Generator, optional
        Random number generator
    """
    logger = _setup_logging(subject)
    logger.info(f"Starting channel-level power correlation analysis for sub-{subject}")
    stats_dir = _stats_dir(subject)
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
        logger.error(f"Could not find epochs for channel correlations: sub-{subject}")
        return
    epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
    events = _load_events_df(subject, task)
    aligned_events = _align_events_to_epochs(events, epochs) if events is not None else None
    temp_series: Optional[pd.Series] = None
    if aligned_events is not None:
        temp_col = _pick_first_column(aligned_events, PSYCH_TEMP_COLUMNS)
        if temp_col is not None:
            temp_series = pd.to_numeric(aligned_events[temp_col], errors="coerce")

    # Helper: build multi-covariate design matrix Z from aligned events
    def _build_covariates_matrix() -> Optional[np.ndarray]:
        if partial_covars is None or aligned_events is None:
            return None
        Z_cols = []
        for cov in partial_covars:
            if cov == "temperature" and temp_series is not None:
                Z_cols.append(temp_series.values)
            elif cov == "trial_number":
                Z_cols.append(np.arange(len(y)))
            elif cov == "subject_mean_rating":
                mean_rating = y.mean()
                Z_cols.append(np.full(len(y), mean_rating))
            elif cov in aligned_events.columns:
                Z_cols.append(pd.to_numeric(aligned_events[cov], errors="coerce").values)
        return np.column_stack(Z_cols) if Z_cols else None

    Z = _build_covariates_matrix()

    # Process each frequency band
    for band in POWER_BANDS_TO_USE:
        logger.info(f"Processing {band} band correlations")
        
        # Get channels for this band
        band_cols = [c for c in pow_df.columns if f"pow_{band}_" in c]
        if not band_cols:
            logger.warning(f"No power columns found for {band} band")
            continue
            
        recs_rating: List[Dict[str, object]] = []
        recs_temp: List[Dict[str, object]] = []
        
        for col in band_cols:
            # Extract channel name from column (e.g., "pow_alpha_Fp1" -> "Fp1")
            channel = col.replace(f"pow_{band}_", "")
            
            # Get power values for this channel-band
            x = pd.to_numeric(pow_df[col], errors="coerce")
            
            # Skip if insufficient valid data
            valid_mask = x.notna() & y.notna()
            if valid_mask.sum() < 10:
                continue
                
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            Z_valid = Z[valid_mask] if Z is not None else None
            
            # Compute correlation vs rating
            corr_method = "spearman"
            try:
                r, p = stats.spearmanr(x_valid, y_valid)
                
                # Confidence interval
                ci_lo, ci_hi = np.nan, np.nan
                if bootstrap > 0:  # Spearman bootstrap CI
                    def _boot_spearman(x, y, rng):
                        idx = rng.choice(len(x), size=len(x), replace=True)
                        return stats.spearmanr(x[idx], y[idx])[0]
                    boot_rs = [_boot_spearman(x_valid.values, y_valid.values, rng) for _ in range(bootstrap)]
                    ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])
                
                # Partial correlation if covariates provided
                r_partial, p_partial = np.nan, np.nan
                if Z_valid is not None and Z_valid.shape[1] > 0:
                    try:
                        r_partial, p_partial = _partial_corr_xy_given_Z(x_valid.values, y_valid.values, Z_valid, 'spearman')
                    except Exception:
                        pass
                
                # Permutation test
                p_perm = np.nan
                if n_perm > 0:
                    null_rs = []
                    for _ in range(n_perm):
                        y_perm = rng.permutation(y_valid)
                        r_perm = stats.spearmanr(x_valid, y_perm)[0]
                        null_rs.append(r_perm)
                    p_perm = (np.sum(np.abs(null_rs) >= np.abs(r)) + 1) / (n_perm + 1)
                
                # Store record
                rec = {
                    "channel": channel,
                    "band": band,
                    "r": r,
                    "p": p,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "r_partial": r_partial,
                    "p_partial": p_partial,
                    "p_perm": p_perm,
                    "n": len(x_valid),
                    "method": corr_method,
                }
                recs_rating.append(rec)
                
            except Exception as e:
                logger.warning(f"Correlation failed for {channel} {band} vs rating: {e}")
                continue
            
            # Compute correlation vs temperature if available
            if temp_series is not None:
                temp_valid = temp_series[valid_mask]
                if temp_valid.notna().sum() >= 10:
                    try:
                        r_temp, p_temp = stats.spearmanr(x_valid, temp_valid)
                        
                        # Temperature CI
                        ci_lo_temp, ci_hi_temp = np.nan, np.nan
                        # Spearman CI via bootstrap only (if desired elsewhere)
                        
                        rec_temp = {
                            "channel": channel,
                            "band": band,
                            "r": r_temp,
                            "p": p_temp,
                            "ci_lo": ci_lo_temp,
                            "ci_hi": ci_hi_temp,
                            "n": temp_valid.notna().sum(),
                            "method": corr_method,
                        }
                        recs_temp.append(rec_temp)
                        
                    except Exception as e:
                        logger.warning(f"Correlation failed for {channel} {band} vs temperature: {e}")
        
        # Save per-band TSVs
        if recs_rating:
            df_rating = pd.DataFrame(recs_rating)
            # Apply FDR correction within band
            pvec = df_rating["p_perm"].to_numpy() if "p_perm" in df_rating.columns and np.isfinite(df_rating["p_perm"]).any() else df_rating["p"].to_numpy()
            rej, crit = _fdr_bh(pvec, alpha=BEHAV_FDR_ALPHA)
            df_rating["fdr_reject"] = rej
            df_rating["fdr_crit_p"] = crit
            df_rating.to_csv(stats_dir / f"corr_stats_pow_{band}_vs_rating.tsv", sep="\t", index=False)
            logger.info(f"Saved {len(df_rating)} {band} band correlations vs rating")
            
        if recs_temp:
            df_temp = pd.DataFrame(recs_temp)
            # Apply FDR correction within band
            pvec_temp = df_temp["p"].to_numpy()
            rej_temp, crit_temp = _fdr_bh(pvec_temp, alpha=BEHAV_FDR_ALPHA)
            df_temp["fdr_reject"] = rej_temp
            df_temp["fdr_crit_p"] = crit_temp
            df_temp.to_csv(stats_dir / f"corr_stats_pow_{band}_vs_temp.tsv", sep="\t", index=False)
            logger.info(f"Saved {len(df_temp)} {band} band correlations vs temperature")

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
    Z_df_full, Z_df_temp = _build_covariate_matrices(aligned_events, partial_covars, temp_col)

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

    def _generate_correlation_scatter_local(
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
        roi_channels: Optional[List[str]] = None,
    ) -> None:
        """Generate correlation scatter plots with marginal histograms and stats.
        
        Creates publication-ready scatter plots with:
        - Main scatter plot with regression line and CI
        - Marginal histograms on top and right
        - Statistical annotations
        - Secondary x-axis for power plots
        
        Args:
            x_data: X variable (power values)
            y_data: Y variable (rating/temperature)
            x_label: X-axis label
            y_label: Y-axis label  
            title_prefix: Title prefix (e.g., "Alpha power vs rating")
            band_color: Color for scatter points and histograms
            output_path: Output file path (without extension)
            method_code: "spearman" or "pearson"
            Z_covars: Covariate matrix for partial correlation
            covar_names: Names of covariates for annotations
            bootstrap_ci: Number of bootstrap samples for CI (0 = Fisher CI)
            rng: Random number generator
            is_partial_residuals: Whether x_data/y_data are already residuals
        """
        # Align data for correlation analysis (using overlap)
        if len(x_data) != len(y_data):
            logger.warning(f"Data length mismatch: x={len(x_data)}, y={len(y_data)}. Using overlap for correlation.")
        n_len = min(len(x_data), len(y_data))
        x = x_data.iloc[:n_len] if hasattr(x_data, 'iloc') else x_data[:n_len]
        y = y_data.iloc[:n_len] if hasattr(y_data, 'iloc') else y_data[:n_len]
        
        if is_partial_residuals:
            # Data is already residuals, use as-is
            m = pd.Series([True] * len(x), index=x.index if hasattr(x, 'index') else range(len(x)))
            n_eff = len(x)
            x_clean = x
            y_clean = y
        else:
            # Filter missing values
            m = x.notna() & y.notna()
            n_eff = int(m.sum())
            x_clean = x[m]
            y_clean = y[m]
            
        if n_eff < 5:
            return
            
        # Calculate correlation stats (always Spearman)
        if is_partial_residuals:
            r, p = stats.spearmanr(x_clean, y_clean, nan_policy="omit")
            r_part, p_part, n_part = np.nan, np.nan, 0
        else:
            r, p = stats.spearmanr(x_clean, y_clean, nan_policy="omit")
            # Optional partial correlation
            r_part, p_part, n_part = np.nan, np.nan, 0
            if Z_covars is not None and len(Z_covars) > 0:
                n_len_pt = min(len(x), len(y), len(Z_covars))
                r_part, p_part, n_part = _partial_corr_xy_given_Z(
                    x.iloc[:n_len_pt], y.iloc[:n_len_pt], Z_covars.iloc[:n_len_pt], method_code
                )

        # Calculate confidence intervals (bootstrap for Spearman if requested)
        if bootstrap_ci > 0:
            ci = _bootstrap_corr_ci(x_clean, y_clean, method_code, n_boot=int(bootstrap_ci), rng=rng)
        else:
            ci = (np.nan, np.nan)

        # Create figure with marginal plots using gridspec
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], 
                             hspace=0.15, wspace=0.15, 
                             left=0.1, right=0.95, top=0.80, bottom=0.12)
        
        # Main scatter plot
        ax_main = fig.add_subplot(gs[1, 0])
        # Top histogram (x-axis marginal)
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
        # Right histogram (y-axis marginal)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
        
        # Configure histogram axes
        ax_histx.tick_params(labelbottom=False)
        ax_histy.tick_params(labelleft=False)
        
        # Plot main scatter with regression
        line_color = _sig_color(p)
        
        sns.regplot(
            x=x_clean,
            y=y_clean,
            ax=ax_main,
            ci=95,
            scatter_kws={"s": 30, "alpha": 0.7, "color": band_color, "edgecolor": "white", "linewidths": 0.3},
            line_kws={"color": line_color, "lw": 1.5},
        )
        
        # Plot marginal histograms
        # X marginal (top)
        ax_histx.hist(x_clean, bins=15, color=band_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        # Add KDE curve on top histogram
        try:
            from scipy.stats import gaussian_kde
            if len(x_clean) > 3:
                kde_x = gaussian_kde(x_clean)
                x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                kde_vals = kde_x(x_range)
                # Scale KDE to match histogram
                hist_counts, _ = np.histogram(x_clean, bins=15)
                kde_scale = hist_counts.max() / kde_vals.max() if kde_vals.max() > 0 else 1
                ax_histx.plot(x_range, kde_vals * kde_scale, color='darkblue', linewidth=1.5, alpha=0.8)
        except (ImportError, ValueError, ZeroDivisionError):
            pass
            
        # Y marginal (right)
        ax_histy.hist(y_clean, bins=15, orientation='horizontal', color=band_color, alpha=0.7, 
                     edgecolor='white', linewidth=0.5)
        # Add KDE curve on right histogram
        try:
            if len(y_clean) > 3:
                kde_y = gaussian_kde(y_clean)
                y_range = np.linspace(y_clean.min(), y_clean.max(), 100)
                kde_vals_y = kde_y(y_range)
                # Scale KDE to match histogram
                hist_counts_y, _ = np.histogram(y_clean, bins=15)
                kde_scale_y = hist_counts_y.max() / kde_vals_y.max() if kde_vals_y.max() > 0 else 1
                ax_histy.plot(kde_vals_y * kde_scale_y, y_range, color='darkblue', linewidth=1.5, alpha=0.8)
        except (ImportError, ValueError, ZeroDivisionError):
            pass
        
        # Set labels
        ax_main.set_xlabel(x_label)
        ax_main.set_ylabel(y_label)
        
        # Add secondary x-axis for power plots (only if no histogram above)
        show_pct_axis = False
        if not is_partial_residuals and "log10(power" in x_label:
            show_pct_axis = True
        elif is_partial_residuals and "residuals of log10(power" in x_label:
            show_pct_axis = True
            
        if show_pct_axis:
            try:
                # Add percentage axis on top histogram
                ax_pct = ax_histx.secondary_xaxis('top', functions=(_logratio_to_pct, _pct_to_logratio))
                ax_pct.set_xlabel("Power Change (%)", fontsize=9)
                # Increase tick density and format as integers; add minor ticks
                ax_pct.xaxis.set_major_locator(MaxNLocator(nbins=7))
                ax_pct.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
                ax_pct.xaxis.set_minor_locator(AutoMinorLocator(2))
            except (AttributeError, TypeError, ValueError):
                pass
        
        # Format title and stats annotation
        label = "Spearman ρ"
        title_parts = [title_prefix]
        
        # Add stats text box to main plot
        stats_text = f"{label} = {r:.3f}\np = {p:.3f}\nn = {n_eff}"
        
        # Position stats box at the top-right of the FIGURE (not the axis)
        # Using figure fraction coordinates avoids crowding the scatter/hist axes
        fig.text(0.98, 0.94, stats_text,
                 fontsize=10, va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Set figure title, appending ROI channels if provided on a new line
        base_title = " vs ".join(title_parts)
        if roi_channels:
            ch_list = ", ".join(roi_channels)
            # Split into two lines for readability
            full_title = f"{base_title}\n({ch_list})"
        else:
            full_title = base_title
        fig.suptitle(full_title, fontsize=11.5, fontweight='bold', y=0.975)
        # Optional figure note about FDR
        if ANNOTATE_FDR:
            try:
                fig.text(
                    0.02, 0.02,
                    f"Note: correlation p shown here is uncorrected; see stats TSVs for FDR (alpha={BEHAV_FDR_ALPHA}).",
                    fontsize=8, ha="left", va="bottom", alpha=0.75,
                )
            except Exception:
                pass
        
        # Apply styling to main plot
        if "Rating" in y_label and not is_partial_residuals:
            try:
                if y.min() >= 0 and y.max() <= 100:
                    ax_main.set_ylim(0, 200)  # Give some breathing room
            except (AttributeError, TypeError, ValueError):
                pass
        elif "Temperature" in y_label and not is_partial_residuals:
            # Align y-ticks to actual temperature values
            unique_temps = np.sort(np.unique(np.asarray(y_clean)))
            if np.allclose(unique_temps, unique_temps.astype(int)):
                unique_temps = unique_temps.astype(int)
            if len(unique_temps) >= 2:
                ax_main.set_yticks(unique_temps.tolist())
                ymin = unique_temps.min() - 0.5
                ymax = unique_temps.max() + 0.5
                ax_main.set_ylim(ymin, ymax)
        elif is_partial_residuals:
            # Reduce tick clutter for residuals
            try:
                ax_main.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax_main.xaxis.set_major_locator(MaxNLocator(nbins=6))
            except (AttributeError, TypeError, ValueError):
                pass
                
        # Style all axes
        for ax in [ax_main, ax_histx, ax_histy]:
            ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.8)
            try:
                sns.despine(ax=ax)
            except (AttributeError, TypeError):
                pass
        
        # Clean up marginal plot spines
        ax_histx.spines['top'].set_visible(False)
        ax_histx.spines['right'].set_visible(False)
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
                
        _save_fig(fig, output_path)

    # Build ROI map and iterate
    roi_map = _build_rois(info)
    for band in POWER_BANDS_TO_USE:
        band_cols = {c for c in pow_df.columns if c.startswith(f"pow_{band}_")}
        if not band_cols:
            continue
        band_rng = FEATURES_FREQ_BANDS.get(band)
        band_title = band.capitalize()
        band_color = _get_band_color(band)

        # --- Overall (all sensors) scatter ---
        try:
            overall_vals = pow_df[list(band_cols)].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        except (KeyError, ValueError, TypeError):
            overall_vals = pd.Series(np.nan, index=pow_df.index)

        # Determine correlation method (always Spearman)
        do_spear = True
        method_code = "spearman"
        covar_names = list(Z_df_full.columns) if Z_df_full is not None else None

        # Create overall subfolder for overall (non-ROI) plots
        overall_plots_dir = plots_dir / "overall"
        _ensure_dir(overall_plots_dir)
        
        # Rating target scatter (overall)
        _generate_correlation_scatter_local(
            x_data=overall_vals,
            y_data=y,
            x_label="log10(power/baseline [-5–0 s])",
            y_label="Rating",
            title_prefix=f"{band_title} power vs rating — Overall",
            band_color=band_color,
            output_path=overall_plots_dir / f"scatter_pow_overall_{_sanitize(band)}_vs_rating",
            method_code=method_code,
            Z_covars=Z_df_full,
            covar_names=covar_names,
            bootstrap_ci=bootstrap_ci,
            rng=rng,
        )
        
        # Generate residual diagnostics for overall rating correlation
        try:
            plot_regression_residual_diagnostics(
                x_data=overall_vals,
                y_data=y,
                title_prefix=f"{band_title} power vs rating — Overall",
                output_path=overall_plots_dir / f"residual_diagnostics_pow_overall_{_sanitize(band)}_vs_rating",
                band_color=band_color,
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"Failed to create residual diagnostics for overall {band} vs rating: {e}")
        
        
        # Separate partial-residuals figure if covariates available
        if Z_df_full is not None and len(Z_df_full) > 0:
            n_len_pt = min(len(overall_vals), len(y), len(Z_df_full))
            x_part = overall_vals.iloc[:n_len_pt]
            y_part = y.iloc[:n_len_pt] 
            Z_part = Z_df_full.iloc[:n_len_pt]
            x_res_sr, y_res_sr, n_res = _partial_residuals_xy_given_Z(x_part, y_part, Z_part, method_code)
            if n_res >= 5:
                residual_xlabel = (
                    "Partial residuals (ranked) of log10(power/baseline)"
                    if method_code == "spearman"
                    else "Partial residuals of log10(power/baseline)"
                )
                residual_ylabel = "Partial residuals (ranked) of rating" if method_code == "spearman" else "Partial residuals of rating"
                
                _generate_correlation_scatter_local(
                    x_data=x_res_sr,
                    y_data=y_res_sr,
                    x_label=residual_xlabel,
                    y_label=residual_ylabel,
                    title_prefix=f"Partial residuals — {band_title} vs rating — Overall",
                    band_color=band_color,
                    output_path=overall_plots_dir / f"scatter_pow_overall_{_sanitize(band)}_vs_rating_partial",
                    method_code=method_code,
                    bootstrap_ci=bootstrap_ci,
                    rng=rng,
                    is_partial_residuals=True,
                )

        # Temperature target scatter (overall)
        if do_temp and temp_series is not None and len(temp_series) > 0:
            # Determine method for temperature (may be different if discrete)
            do_spear_t = True
            method2_code = "spearman"
            covar_names_temp = list(Z_df_temp.columns) if Z_df_temp is not None else None
            
            _generate_correlation_scatter_local(
                x_data=overall_vals,
                y_data=temp_series,
                x_label="log10(power/baseline [-5–0 s])",
                y_label="Temperature (°C)",
                title_prefix=f"{band_title} power vs temperature — Overall",
                band_color=band_color,
                output_path=overall_plots_dir / f"scatter_pow_overall_{_sanitize(band)}_vs_temp",
                method_code=method2_code,
                Z_covars=Z_df_temp,
                covar_names=covar_names_temp,
                bootstrap_ci=bootstrap_ci,
                rng=rng,
            )
            
            # Generate residual diagnostics for overall temperature correlation
            try:
                plot_regression_residual_diagnostics(
                    x_data=overall_vals,
                    y_data=temp_series,
                    title_prefix=f"{band_title} power vs temperature — Overall",
                    output_path=overall_plots_dir / f"residual_diagnostics_pow_overall_{_sanitize(band)}_vs_temp",
                    band_color=band_color,
                    logger=logger,
                )
            except Exception as e:
                logger.warning(f"Failed to create residual diagnostics for overall {band} vs temperature: {e}")
            
            # Overall temperature partial-residuals figure if covariates available
            if Z_df_temp is not None and len(Z_df_temp) > 0:
                n_len_pt2 = min(len(overall_vals), len(temp_series), len(Z_df_temp))
                x_part2 = overall_vals.iloc[:n_len_pt2]
                y_part2 = temp_series.iloc[:n_len_pt2]
                Z_part2 = Z_df_temp.iloc[:n_len_pt2]
                x2_res_sr, y2_res_sr, n2_res = _partial_residuals_xy_given_Z(x_part2, y_part2, Z_part2, method2_code)
                if n2_res >= 5:
                    residual_xlabel = (
                        "Partial residuals (ranked) of log10(power/baseline)"
                    )
                    residual_ylabel = "Partial residuals (ranked) of temperature (°C)"
                    
                    _generate_correlation_scatter_local(
                        x_data=x2_res_sr,
                        y_data=y2_res_sr,
                        x_label=residual_xlabel,
                        y_label=residual_ylabel,
                        title_prefix=f"Partial residuals — {band_title} vs temperature — Overall",
                        band_color=band_color,
                        output_path=overall_plots_dir / f"scatter_pow_overall_{_sanitize(band)}_vs_temp_partial",
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

            # Create ROI-specific subfolder
            roi_plots_dir = plots_dir / "roi_scatters" / _sanitize(roi)
            _ensure_dir(roi_plots_dir)

            # ROI-specific rating scatter
            _generate_correlation_scatter_local(
                x_data=roi_vals,
                y_data=y,
                x_label="log10(power/baseline [-5–0 s])",
                y_label="Rating",
                title_prefix=f"{band_title} power vs rating — {roi}",
                band_color=band_color,
                output_path=roi_plots_dir / f"scatter_pow_{_sanitize(band)}_vs_rating",
                method_code=method_code,
                Z_covars=Z_df_full,
                covar_names=covar_names,
                bootstrap_ci=bootstrap_ci,
                rng=rng,
                roi_channels=chs,
            )
            
            # Generate residual diagnostics for ROI rating correlation
            try:
                plot_regression_residual_diagnostics(
                    x_data=roi_vals,
                    y_data=y,
                    title_prefix=f"{band_title} power vs rating — {roi}",
                    output_path=roi_plots_dir / f"residual_diagnostics_pow_{_sanitize(band)}_vs_rating",
                    band_color=band_color,
                    logger=logger,
                )
            except Exception as e:
                logger.warning(f"Failed to create residual diagnostics for {roi} {band} vs rating: {e}")
            
            
            # ROI partial-residuals figure if covariates available
            if Z_df_full is not None and len(Z_df_full) > 0:
                n_len_pt = min(len(roi_vals), len(y), len(Z_df_full))
                x_part = roi_vals.iloc[:n_len_pt]
                y_part = y.iloc[:n_len_pt]
                Z_part = Z_df_full.iloc[:n_len_pt]
                x_res_sr, y_res_sr, n_res = _partial_residuals_xy_given_Z(x_part, y_part, Z_part, method_code)
                if n_res >= 5:
                    residual_xlabel = (
                        "Partial residuals (ranked) of log10(power/baseline)"
                        if method_code == "spearman"
                        else "Partial residuals of log10(power/baseline)"
                    )
                    residual_ylabel = "Partial residuals (ranked) of rating" if method_code == "spearman" else "Partial residuals of rating"
                    
                    _generate_correlation_scatter_local(
                    x_data=x_res_sr,
                    y_data=y_res_sr,
                    x_label=residual_xlabel,
                    y_label=residual_ylabel,
                    title_prefix=f"Partial residuals — {band_title} vs rating — {roi}",
                    band_color=band_color,
                    output_path=roi_plots_dir / f"scatter_pow_{_sanitize(band)}_vs_rating_partial",
                    method_code=method_code,
                    bootstrap_ci=bootstrap_ci,
                    rng=rng,
                    is_partial_residuals=True,
                    roi_channels=chs,
                )

            # ROI-specific temperature scatter (optional)
            if do_temp and temp_series is not None and len(temp_series) > 0:
                # Always use Spearman for temperature correlations
                do_spear_t = True
                method2_code = "spearman"
                covar_names_temp = list(Z_df_temp.columns) if Z_df_temp is not None else None
                
                _generate_correlation_scatter_local(
                    x_data=roi_vals,
                    y_data=temp_series,
                    x_label="log10(power/baseline [-5–0 s])",
                    y_label="Temperature (°C)",
                    title_prefix=f"{band_title} power vs temperature — {roi}",
                    band_color=band_color,
                    output_path=roi_plots_dir / f"scatter_pow_{_sanitize(band)}_vs_temp",
                    method_code=method2_code,
                    Z_covars=Z_df_temp,
                    covar_names=covar_names_temp,
                    bootstrap_ci=bootstrap_ci,
                    rng=rng,
                    roi_channels=chs,
                )
                
                # Generate residual diagnostics for ROI temperature correlation
                try:
                    plot_regression_residual_diagnostics(
                        x_data=roi_vals,
                        y_data=temp_series,
                        title_prefix=f"{band_title} power vs temperature — {roi}",
                        output_path=roi_plots_dir / f"residual_diagnostics_pow_{_sanitize(band)}_vs_temp",
                        band_color=band_color,
                        logger=logger,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create residual diagnostics for {roi} {band} vs temperature: {e}")
                
                # ROI temperature partial-residuals figure if covariates available
                if Z_df_temp is not None and len(Z_df_temp) > 0:
                    n_len_pt2 = min(len(roi_vals), len(temp_series), len(Z_df_temp))
                    x_part2 = roi_vals.iloc[:n_len_pt2]
                    y_part2 = temp_series.iloc[:n_len_pt2]
                    Z_part2 = Z_df_temp.iloc[:n_len_pt2]
                    x2_res_sr, y2_res_sr, n2_res = _partial_residuals_xy_given_Z(x_part2, y_part2, Z_part2, method2_code)
                    if n2_res >= 5:
                        residual_xlabel = "Partial residuals (ranked) of log10(power/baseline)"
                        residual_ylabel = "Partial residuals (ranked) of temperature (°C)"
                        _generate_correlation_scatter_local(
                            x_data=x2_res_sr,
                            y_data=y2_res_sr,
                            x_label=residual_xlabel,
                            y_label=residual_ylabel,
                            title_prefix=f"Partial residuals — {band_title} vs temperature — {roi}",
                            band_color=band_color,
                            output_path=roi_plots_dir / f"scatter_pow_{_sanitize(band)}_vs_temp_partial",
                            method_code=method2_code,
                            bootstrap_ci=bootstrap_ci,
                            rng=rng,
                            is_partial_residuals=True,
                            roi_channels=chs,
                        )

# -----------------------------------------------------------------------------
# Group-level visualization: pooled ROI power scatter plots
# -----------------------------------------------------------------------------

def plot_group_power_roi_scatter(
    subjects: List[str],
    task: str = TASK,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    do_temp: bool = True,
    bootstrap_ci: int = 0,
    rng: Optional[np.random.Generator] = None,
    pooling_strategy: str = "within_subject_centered",
    cluster_bootstrap: int = 0,
    subject_fixed_effects: bool = True,
) -> None:
    """Create pooled ROI power scatter plots across subjects.

    Logs concise progress messages and writes pooled correlation stats.

    pooling_strategy:
        - 'pooled_trials': concatenate trials across subjects.
        - 'within_subject_centered': mean-center x and y within each subject, then pool.
        - 'fisher_by_subject': compute per-subject r and aggregate via Fisher z.
    """
    logger = _setup_group_logging()
    plots_dir = _group_plots_dir()
    stats_dir = _group_stats_dir()
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    if rng is None:
        rng = np.random.default_rng(42)

    try:
        logger.info(
            f"Starting group pooled ROI scatters for {len(subjects)} subjects (task={task}, strategy={pooling_strategy}, FE={'on' if subject_fixed_effects else 'off'})"
        )
    except Exception:
        pass

    allowed_pool = {"pooled_trials", "within_subject_centered", "fisher_by_subject"}
    if pooling_strategy not in allowed_pool:
        logger.warning(f"Unknown pooling_strategy '{pooling_strategy}', falling back to 'pooled_trials'")
        pooling_strategy = "pooled_trials"

    tag_map = {
        "pooled_trials": "[Pooled]",
        "within_subject_centered": "[Centered]",
        "within_subject_zscored": "[Z-scored]",
        "fisher_by_subject": "[Fisher]",
    }

    def _compute_group_stats(
        x_lists: List[np.ndarray],
        y_lists: List[np.ndarray],
        method_code: str,
        *,
        strategy: str,
        cb_n: int,
        rng_local: np.random.Generator,
    ) -> Tuple[float, float, int, int, Tuple[float, float]]:
        """Return (r, p, n_trials, n_subjects, ci_95) under the chosen strategy.

        - For pooled strategies, p is the trial-level correlation p.
        - For fisher_by_subject, p is from a one-sample t-test on Fisher z.
        - CI is a subject-level (cluster) bootstrap CI if cb_n > 0 and >=2 subjects; else NaNs.
        """
        # Pre-filter subjects for valid data (>=5 non-NaN pairs)
        pairs = []
        for xi, yi in zip(x_lists, y_lists):
            xi = np.asarray(xi)
            yi = np.asarray(yi)
            n = min(len(xi), len(yi))
            xi = xi[:n]
            yi = yi[:n]
            m = np.isfinite(xi) & np.isfinite(yi)
            if m.sum() >= 5:
                pairs.append((xi[m], yi[m]))
        S = len(pairs)
        if S == 0:
            return np.nan, np.nan, 0, 0, (np.nan, np.nan)

        def _r_xy(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
            # Always Spearman for consistency
            r, p = stats.spearmanr(x, y, nan_policy="omit")
            return float(r), float(p)

        if strategy in {"pooled_trials", "within_subject_centered", "within_subject_zscored"}:
            xs = []
            ys = []
            for (xi, yi) in pairs:
                if strategy == "within_subject_centered":
                    xi = xi - np.nanmean(xi)
                    yi = yi - np.nanmean(yi)
                elif strategy == "within_subject_zscored":
                    sx = np.nanstd(xi, ddof=1)
                    sy = np.nanstd(yi, ddof=1)
                    if sx <= 0 or sy <= 0:
                        # Skip subjects with no variance
                        continue
                    xi = (xi - np.nanmean(xi)) / sx
                    yi = (yi - np.nanmean(yi)) / sy
                xs.append(xi)
                ys.append(yi)
            X = np.concatenate(xs)
            Y = np.concatenate(ys)
            r_obs, p_obs = _r_xy(X, Y)
            n_trials = int(len(X))

            # Cluster bootstrap over subjects
            ci = (np.nan, np.nan)
            if cb_n and S >= 2:
                boots = []
                idxs = np.arange(S)
                for _ in range(int(cb_n)):
                    pick = rng_local.choice(idxs, size=S, replace=True)
                    bx = []
                    by = []
                    for j in pick:
                        xi, yi = pairs[j]
                        if strategy == "within_subject_centered":
                            xi = xi - np.nanmean(xi)
                            yi = yi - np.nanmean(yi)
                        elif strategy == "within_subject_zscored":
                            sx = np.nanstd(xi, ddof=1)
                            sy = np.nanstd(yi, ddof=1)
                            if sx <= 0 or sy <= 0:
                                continue
                            xi = (xi - np.nanmean(xi)) / sx
                            yi = (yi - np.nanmean(yi)) / sy
                        bx.append(xi)
                        by.append(yi)
                    Xb = np.concatenate(bx)
                    Yb = np.concatenate(by)
                    rb, _ = _r_xy(Xb, Yb)
                    if np.isfinite(rb):
                        boots.append(rb)
                if boots:
                    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
            return float(r_obs), float(p_obs), n_trials, S, ci

        # fisher_by_subject
        r_subj = []
        for (xi, yi) in pairs:
            r_i, _ = _r_xy(xi, yi)
            if np.isfinite(r_i):
                r_subj.append(float(np.clip(r_i, -0.999999, 0.999999)))
        if not r_subj:
            return np.nan, np.nan, 0, S, (np.nan, np.nan)
        z = np.arctanh(np.array(r_subj))
        r_grp = float(np.tanh(np.mean(z)))
        # p via t-test of Fisher z against 0
        if len(z) >= 2:
            tstat, p_grp = stats.ttest_1samp(z, popmean=0.0)
            p_grp = float(p_grp)
        else:
            p_grp = np.nan
        n_trials = int(sum(len(x) for (x, _y) in pairs))
        ci = (np.nan, np.nan)
        if cb_n and S >= 2:
            boots = []
            idxs = np.arange(S)
            for _ in range(int(cb_n)):
                pick = rng_local.choice(idxs, size=S, replace=True)
                zb = np.mean(z[pick])
                boots.append(float(np.tanh(zb)))
            if boots:
                ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        return r_grp, p_grp, n_trials, S, ci

    rating_x_by_key: Dict[Tuple[str, str], List[np.ndarray]] = {}
    rating_y_by_key: Dict[Tuple[str, str], List[np.ndarray]] = {}
    rating_Z_by_key: Dict[Tuple[str, str], List[pd.DataFrame]] = {}
    rating_hasZ_by_key: Dict[Tuple[str, str], List[bool]] = {}
    rating_subj_by_key: Dict[Tuple[str, str], List[str]] = {}
    temp_x_by_key: Dict[Tuple[str, str], List[np.ndarray]] = {}
    temp_y_by_key: Dict[Tuple[str, str], List[np.ndarray]] = {}
    temp_Z_by_key: Dict[Tuple[str, str], List[pd.DataFrame]] = {}
    temp_hasZ_by_key: Dict[Tuple[str, str], List[bool]] = {}
    temp_subj_by_key: Dict[Tuple[str, str], List[str]] = {}
    have_temp = False

    for sub in subjects:
        try:
            pow_df, _conn_df, y, info = _load_features_and_targets(sub, task)
        except Exception as e:
            logger.warning(f"Skipping sub-{sub} for group scatter: {e}")
            continue
        y = pd.to_numeric(y, errors="coerce")

        # Load events for covariates and temperature
        try:
            epo_path = _find_clean_epochs_path(sub, task)
            if epo_path is None:
                raise FileNotFoundError("epochs not found")
            epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
            events = _load_events_df(sub, task)
            aligned_events = _align_events_to_epochs(events, epochs) if events is not None else None
        except Exception as e:
            logger.warning(f"Skipping sub-{sub} due to events/epochs issue: {e}")
            continue

        temp_series: Optional[pd.Series] = None
        temp_col: Optional[str] = None
        if aligned_events is not None:
            tcol = _pick_first_column(aligned_events, PSYCH_TEMP_COLUMNS)
            if tcol is not None:
                temp_col = tcol
                temp_series = pd.to_numeric(aligned_events[tcol], errors="coerce")
                have_temp = True

        Z_df_full, Z_df_temp = _build_covariate_matrices(aligned_events, partial_covars, temp_col)

        roi_map = _build_rois(info)
        for band in POWER_BANDS_TO_USE:
            band_cols = [c for c in pow_df.columns if c.startswith(f"pow_{band}_")]
            if not band_cols:
                continue
            band_vals = pow_df[band_cols].apply(pd.to_numeric, errors="coerce")
            overall_vals = band_vals.mean(axis=1).to_numpy()
            key_overall = (band, "All")
            rating_x_by_key.setdefault(key_overall, []).append(overall_vals)
            rating_y_by_key.setdefault(key_overall, []).append(y.to_numpy())
            rating_subj_by_key.setdefault(key_overall, []).append(sub)
            if Z_df_full is not None:
                rating_Z_by_key.setdefault(key_overall, []).append(Z_df_full)
                rating_hasZ_by_key.setdefault(key_overall, []).append(True)
            else:
                rating_hasZ_by_key.setdefault(key_overall, []).append(False)
            if do_temp and temp_series is not None:
                temp_x_by_key.setdefault(key_overall, []).append(overall_vals)
                temp_y_by_key.setdefault(key_overall, []).append(temp_series.to_numpy())
                temp_subj_by_key.setdefault(key_overall, []).append(sub)
                if Z_df_temp is not None:
                    temp_Z_by_key.setdefault(key_overall, []).append(Z_df_temp)
                    temp_hasZ_by_key.setdefault(key_overall, []).append(True)
                else:
                    temp_hasZ_by_key.setdefault(key_overall, []).append(False)
            for roi, chs in roi_map.items():
                cols = [f"pow_{band}_{ch}" for ch in chs if f"pow_{band}_{ch}" in pow_df.columns]
                if not cols:
                    continue
                roi_vals = pow_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
                key = (band, roi)
                rating_x_by_key.setdefault(key, []).append(roi_vals)
                rating_y_by_key.setdefault(key, []).append(y.to_numpy())
                rating_subj_by_key.setdefault(key, []).append(sub)
                if Z_df_full is not None:
                    rating_Z_by_key.setdefault(key, []).append(Z_df_full)
                    rating_hasZ_by_key.setdefault(key, []).append(True)
                else:
                    rating_hasZ_by_key.setdefault(key, []).append(False)
                if do_temp and temp_series is not None:
                    temp_x_by_key.setdefault(key, []).append(roi_vals)
                    temp_y_by_key.setdefault(key, []).append(temp_series.to_numpy())
                    temp_subj_by_key.setdefault(key, []).append(sub)
                    if Z_df_temp is not None:
                        temp_Z_by_key.setdefault(key, []).append(Z_df_temp)
                        temp_hasZ_by_key.setdefault(key, []).append(True)
                    else:
                        temp_hasZ_by_key.setdefault(key, []).append(False)

    rating_records: List[Dict[str, object]] = []
    method_code = "spearman"
    for (band, roi), x_lists in rating_x_by_key.items():
        y_lists = rating_y_by_key.get((band, roi))
        if not y_lists:
            continue
        # Keep Z handling for passing to the plotting helper, but group r/p are computed above
        Z_lists = rating_Z_by_key.get((band, roi))
        if Z_lists:
            common_cols = set(Z_lists[0].columns)
            for df in Z_lists[1:]:
                common_cols &= set(df.columns)
            if common_cols:
                Z_all = pd.concat([df[sorted(common_cols)] for df in Z_lists], ignore_index=True)
            else:
                Z_all = None
        else:
            Z_all = None

        band_rng = FEATURES_FREQ_BANDS.get(band)
        band_title = f"{band.capitalize()} ({band_rng[0]:g}\u2013{band_rng[1]:g} Hz)" if band_rng else band.capitalize()
        title_roi = "Overall" if roi == "All" else roi
        title_prefix = f"{band_title} power vs rating — {title_roi}"

        out_dir = plots_dir / (
            "overall" if roi == "All" else Path("roi_scatters") / _sanitize(roi)
        )

        _ensure_dir(out_dir)
        base_name = (
            f"scatter_pow_overall_{_sanitize(band)}_vs_rating" if roi == "All" else f"scatter_pow_{_sanitize(band)}_vs_rating"
        )
        out_path = out_dir / base_name
        # Build visualization data and aligned design matrix for partials
        vis_x = []
        vis_y = []
        vis_subj_ids = []
        subj_order = rating_subj_by_key.get((band, roi), [])
        for i, (xi_arr, yi_arr) in enumerate(zip(x_lists, y_lists)):
            xi = pd.Series(xi_arr)
            yi = pd.Series(yi_arr)
            n = min(len(xi), len(yi))
            xi = xi.iloc[:n]
            yi = yi.iloc[:n]
            m = xi.notna() & yi.notna()
            xi = xi[m]
            yi = yi[m]
            if pooling_strategy == "within_subject_centered":
                xi = xi - xi.mean()
                yi = yi - yi.mean()
            elif pooling_strategy == "within_subject_zscored":
                sx = xi.std(ddof=1)
                sy = yi.std(ddof=1)
                if sx <= 0 or sy <= 0:
                    continue
                xi = (xi - xi.mean()) / sx
                yi = (yi - yi.mean()) / sy
            elif pooling_strategy == "fisher_by_subject":
                # Center for visualization under Fisher strategy
                xi = xi - xi.mean()
                yi = yi - yi.mean()
            vis_x.append(xi)
            vis_y.append(yi)
            sid = subj_order[i] if i < len(subj_order) else str(i)
            vis_subj_ids.extend([sid] * len(xi))
        x_all = pd.concat(vis_x, ignore_index=True) if vis_x else pd.Series(dtype=float)
        y_all = pd.concat(vis_y, ignore_index=True) if vis_y else pd.Series(dtype=float)

        # Build aligned Z for partials + optional subject fixed-effects
        Z_all_vis = None
        x_all_partial = None
        y_all_partial = None
        hasZ_list = rating_hasZ_by_key.get((band, roi))
        if hasZ_list is not None and (subject_fixed_effects or rating_Z_by_key.get((band, roi))):
            vis_Z = []
            partial_x = []
            partial_y = []
            j = 0
            common_cols = []
            Z_lists = rating_Z_by_key.get((band, roi))
            if Z_lists:
                common_cols = sorted(set(Z_lists[0].columns))
                for df in Z_lists[1:]:
                    common_cols = sorted(set(common_cols) & set(df.columns))
            # Build Z rows aligned to vis_x/vis_y entries
            for i, (xi_arr, yi_arr) in enumerate(zip(x_lists, y_lists)):
                if not hasZ_list[i]:
                    continue
                Zi = Z_lists[j]
                j += 1
                # Align Zi to xi/yi filtering
                xi = pd.Series(xi_arr)
                yi = pd.Series(yi_arr)
                n = min(len(xi), len(yi), len(Zi))
                xi = xi.iloc[:n]
                yi = yi.iloc[:n]
                Zi = Zi.iloc[:n]
                m = xi.notna() & yi.notna()
                xi = xi[m]
                yi = yi[m]
                Zi = Zi.loc[m]
                # Apply same transformation as main visualization for consistency
                if pooling_strategy == "within_subject_centered":
                    xi = xi - xi.mean()
                    yi = yi - yi.mean()
                elif pooling_strategy == "within_subject_zscored":
                    sx = xi.std(ddof=1)
                    sy = yi.std(ddof=1)
                    if sx <= 0 or sy <= 0:
                        continue
                    xi = (xi - xi.mean()) / sx
                    yi = (yi - yi.mean()) / sy
                # Keep only common covariates
                Zi = Zi[common_cols] if common_cols else Zi
                # Optionally add subject id for FE dummies
                if subject_fixed_effects:
                    sid = subj_order[i] if i < len(subj_order) else str(i)
                    Zi = Zi.copy()
                    Zi["__subject_id__"] = sid
                vis_Z.append(Zi)
                partial_x.append(xi)
                partial_y.append(yi)
            if vis_Z:
                Z_all_vis = pd.concat(vis_Z, ignore_index=True)
                x_all_partial = pd.concat(partial_x, ignore_index=True)
                y_all_partial = pd.concat(partial_y, ignore_index=True)
                if subject_fixed_effects and "__subject_id__" in Z_all_vis.columns:
                    dummies = pd.get_dummies(Z_all_vis["__subject_id__"].astype(str), prefix="sub", drop_first=True)
                    Z_all_vis = pd.concat([Z_all_vis.drop(columns=["__subject_id__"]), dummies], axis=1)

        r_g, p_g, n_trials, n_subj, ci95 = _compute_group_stats(
            [np.asarray(v) for v in x_lists],
            [np.asarray(v) for v in y_lists],
            method_code,
            strategy=pooling_strategy,
            cb_n=int(cluster_bootstrap),
            rng_local=rng,
        )

        tag = tag_map.get(pooling_strategy)

        r, p, n_eff = _generate_correlation_scatter(
            x_data=x_all,
            y_data=y_all,
            x_label="log10(power/baseline [-5–0 s])",
            # Use pooling-aware y-axis label for rating
            y_label=(
                "Rating (centered)" if pooling_strategy == "within_subject_centered"
                else ("Rating (z-scored)" if pooling_strategy == "within_subject_zscored" else "Rating")
            ),
            title_prefix=title_prefix,
            band_color=_get_band_color(band),
            output_path=out_path,
            method_code=method_code,
            Z_covars=None,
            covar_names=None,
            bootstrap_ci=bootstrap_ci,
            rng=rng,
            logger=logger,
            annotated_stats=(r_g, p_g, n_trials),
            annot_ci=ci95,
            stats_tag=tag,
        )
        rating_records.append({
            "roi": roi,
            "band": band,
            "r_pooled": r_g,
            "p_pooled": p_g,
            "n_total": n_trials,
            "n_subjects": n_subj,
            "pooling_strategy": pooling_strategy,
            "ci_low": ci95[0],
            "ci_high": ci95[1],
        })

        try:
            logger.info(
                f"Pooled rating scatter saved: {out_path} "
                f"(band={band}, roi={title_roi}, r={r_g:.3f}, p={p_g:.3g}, n_trials={n_trials}, n_subj={n_subj}, strategy={pooling_strategy})"
            )
        except Exception:
            pass

        if Z_all_vis is None and subject_fixed_effects and len(vis_subj_ids) == len(x_all):
            # Build FE-only design matrix aligned to all pooled rows
            Z_all_vis = pd.get_dummies(pd.Series(vis_subj_ids, name="__subject_id__").astype(str), prefix="sub", drop_first=True)
            x_all_partial = x_all
            y_all_partial = y_all

        if Z_all_vis is not None and len(Z_all_vis) > 0 and x_all_partial is not None and y_all_partial is not None:
            # Use aligned subset for partial residuals
            x_res, y_res, n_res = _partial_residuals_xy_given_Z(x_all_partial, y_all_partial, Z_all_vis, method_code)
            if n_res >= 5:
                residual_xlabel = (
                    "Partial residuals (ranked) of log10(power/baseline)"
                )
                residual_ylabel = (
                    "Partial residuals (ranked) of rating"
                )
                _generate_correlation_scatter(
                    x_data=x_res,
                    y_data=y_res,
                    x_label=residual_xlabel,
                    y_label=residual_ylabel,
                    title_prefix=f"Partial residuals — {band_title} power vs rating — {title_roi}",
                    band_color=_get_band_color(band),
                    output_path=out_dir / f"{base_name}_partial",
                    method_code=method_code,
                    is_partial_residuals=True,
                    rng=rng,
                    logger=logger,
                )

    if rating_records:
        df = pd.DataFrame(rating_records)
        rej, crit = _fdr_bh(df["p_pooled"].to_numpy(), alpha=BEHAV_FDR_ALPHA)
        df["fdr_reject"] = rej
        df["fdr_crit_p"] = crit
        out_stats = stats_dir / "group_pooled_corr_pow_roi_vs_rating.tsv"
        df.to_csv(out_stats, sep="\t", index=False)
        try:
            logger.info(f"Wrote pooled ROI vs rating stats: {out_stats}")
        except Exception:
            pass

    if do_temp and have_temp:
        temp_records: List[Dict[str, object]] = []
        for (band, roi), x_lists in temp_x_by_key.items():
            y_lists = temp_y_by_key.get((band, roi))
            if not y_lists:
                continue
            # Keep Z for plotting; stats computed separately
            Z_lists = temp_Z_by_key.get((band, roi))
            if Z_lists:
                common_cols = set(Z_lists[0].columns)
                for df in Z_lists[1:]:
                    common_cols &= set(df.columns)
                if common_cols:
                    Z_all = pd.concat([df[sorted(common_cols)] for df in Z_lists], ignore_index=True)
                else:
                    Z_all = None
            else:
                Z_all = None

            band_rng = FEATURES_FREQ_BANDS.get(band)
            band_title = f"{band.capitalize()} ({band_rng[0]:g}\u2013{band_rng[1]:g} Hz)" if band_rng else band.capitalize()
            title_roi = "Overall" if roi == "All" else roi
            title_prefix = f"{band_title} power vs temperature — {title_roi}"

            out_dir = plots_dir / (
                "overall" if roi == "All" else Path("roi_scatters") / _sanitize(roi)
            )

            _ensure_dir(out_dir)
            base_name = (
                f"scatter_pow_overall_{_sanitize(band)}_vs_temp" if roi == "All" else f"scatter_pow_{_sanitize(band)}_vs_temp"
            )
            out_path = out_dir / base_name
            try:
                _y_all_concat = np.concatenate([np.asarray(v) for v in y_lists])
                _y_all_concat = _y_all_concat[np.isfinite(_y_all_concat)]
                _ny = len(np.unique(_y_all_concat))
            except Exception:
                _ny = 0
            method2_code = "spearman"
            # Visualization data according to strategy
            vis_x = []
            vis_y = []
            vis_subj_ids = []
            subj_order = temp_subj_by_key.get((band, roi), [])
            for i, (xi_arr, yi_arr) in enumerate(zip(x_lists, y_lists)):
                xi = pd.Series(xi_arr)
                yi = pd.Series(yi_arr)
                n = min(len(xi), len(yi))
                xi = xi.iloc[:n]
                yi = yi.iloc[:n]
                m = xi.notna() & yi.notna()
                xi = xi[m]
                yi = yi[m]
                if pooling_strategy == "within_subject_centered":
                    xi = xi - xi.mean()
                    yi = yi - yi.mean()
                elif pooling_strategy == "within_subject_zscored":
                    sx = xi.std(ddof=1)
                    sy = yi.std(ddof=1)
                    if sx <= 0 or sy <= 0:
                        continue
                    xi = (xi - xi.mean()) / sx
                    yi = (yi - yi.mean()) / sy
                elif pooling_strategy == "fisher_by_subject":
                    xi = xi - xi.mean()
                    yi = yi - yi.mean()
                vis_x.append(xi)
                vis_y.append(yi)
                sid = subj_order[i] if i < len(subj_order) else str(i)
                vis_subj_ids.extend([sid] * len(xi))
            x_all = pd.concat(vis_x, ignore_index=True) if vis_x else pd.Series(dtype=float)
            y_all = pd.concat(vis_y, ignore_index=True) if vis_y else pd.Series(dtype=float)

            # Build aligned Z for partials + optional subject fixed-effects
            Z_all_vis = None
            x_all_partial = None
            y_all_partial = None
            hasZ_list = temp_hasZ_by_key.get((band, roi))
            if hasZ_list is not None and (subject_fixed_effects or temp_Z_by_key.get((band, roi))):
                vis_Z = []
                partial_x = []
                partial_y = []
                j = 0
                common_cols = []
                Z_lists2 = temp_Z_by_key.get((band, roi))
                if Z_lists2:
                    common_cols = sorted(set(Z_lists2[0].columns))
                    for df in Z_lists2[1:]:
                        common_cols = sorted(set(common_cols) & set(df.columns))
                for i, (xi_arr, yi_arr) in enumerate(zip(x_lists, y_lists)):
                    if not hasZ_list[i]:
                        continue
                    Zi = Z_lists2[j]
                    j += 1
                    xi = pd.Series(xi_arr)
                    yi = pd.Series(yi_arr)
                    n = min(len(xi), len(yi), len(Zi))
                    xi = xi.iloc[:n]
                    yi = yi.iloc[:n]
                    Zi = Zi.iloc[:n]
                    m = xi.notna() & yi.notna()
                    xi = xi[m]
                    yi = yi[m]
                    Zi = Zi.loc[m]
                    if pooling_strategy == "within_subject_centered":
                        xi = xi - xi.mean()
                        yi = yi - yi.mean()
                    elif pooling_strategy == "within_subject_zscored":
                        sx = xi.std(ddof=1)
                        sy = yi.std(ddof=1)
                        if sx <= 0 or sy <= 0:
                            continue
                        xi = (xi - xi.mean()) / sx
                        yi = (yi - yi.mean()) / sy
                    Zi = Zi[common_cols] if common_cols else Zi
                    if subject_fixed_effects:
                        sid = subj_order[i] if i < len(subj_order) else str(i)
                        Zi = Zi.copy()
                        Zi["__subject_id__"] = sid
                    vis_Z.append(Zi)
                    partial_x.append(xi)
                    partial_y.append(yi)
                if vis_Z:
                    Z_all_vis = pd.concat(vis_Z, ignore_index=True)
                    x_all_partial = pd.concat(partial_x, ignore_index=True)
                    y_all_partial = pd.concat(partial_y, ignore_index=True)
                    if subject_fixed_effects and "__subject_id__" in Z_all_vis.columns:
                        dummies = pd.get_dummies(Z_all_vis["__subject_id__"].astype(str), prefix="sub", drop_first=True)
                        Z_all_vis = pd.concat([Z_all_vis.drop(columns=["__subject_id__"]), dummies], axis=1)

            r_g, p_g, n_trials, n_subj, ci95 = _compute_group_stats(
                [np.asarray(v) for v in x_lists],
                [np.asarray(v) for v in y_lists],
                method2_code,
                strategy=pooling_strategy,
                cb_n=int(cluster_bootstrap),
                rng_local=rng,
            )

            tag = tag_map.get(pooling_strategy)

            # Use pooling-aware y-axis label for temperature
            if pooling_strategy == "within_subject_centered":
                _y_label_temp = "Temperature (°C, centered)"
            elif pooling_strategy == "within_subject_zscored":
                _y_label_temp = "Temperature (z-scored)"
            else:
                _y_label_temp = "Temperature (°C)"

            r, p, n_eff = _generate_correlation_scatter(
                x_data=x_all,
                y_data=y_all,
                x_label="log10(power/baseline [-5–0 s])",
                y_label=_y_label_temp,
                title_prefix=title_prefix,
                band_color=_get_band_color(band),
                output_path=out_path,
                method_code=method2_code,
                Z_covars=None,
                covar_names=None,
                bootstrap_ci=bootstrap_ci,
                rng=rng,
                logger=logger,
                annotated_stats=(r_g, p_g, n_trials),
                annot_ci=ci95,
                stats_tag=tag,
            )
            temp_records.append({
                "roi": roi,
                "band": band,
                "r_pooled": r_g,
                "p_pooled": p_g,
                "n_total": n_trials,
                "n_subjects": n_subj,
                "pooling_strategy": pooling_strategy,
                "ci_low": ci95[0],
                "ci_high": ci95[1],
            })

            try:
                logger.info(
                    f"Pooled temperature scatter saved: {out_path} "
                    f"(band={band}, roi={title_roi}, r={r_g:.3f}, p={p_g:.3g}, n_trials={n_trials}, n_subj={n_subj}, strategy={pooling_strategy})"
                )
            except Exception:
                pass

            if Z_all_vis is None and subject_fixed_effects and len(vis_subj_ids) == len(x_all):
                Z_all_vis = pd.get_dummies(pd.Series(vis_subj_ids, name="__subject_id__").astype(str), prefix="sub", drop_first=True)
                x_all_partial = x_all
                y_all_partial = y_all
            if Z_all_vis is not None and len(Z_all_vis) > 0 and x_all_partial is not None and y_all_partial is not None:
                x_res2, y_res2, n_res2 = _partial_residuals_xy_given_Z(x_all_partial, y_all_partial, Z_all_vis, method2_code)
                if n_res2 >= 5:
                    residual_xlabel = (
                        "Partial residuals (ranked) of log10(power/baseline)"
                        if method2_code == "spearman"
                        else "Partial residuals of log10(power/baseline)"
                    )
                    residual_ylabel = (
                        "Partial residuals (ranked) of temperature (°C)"
                        if method2_code == "spearman"
                        else "Partial residuals of temperature (°C)"
                    )
                    _generate_correlation_scatter(
                        x_data=x_res2,
                        y_data=y_res2,
                        x_label=residual_xlabel,
                        y_label=residual_ylabel,
                        title_prefix=f"Partial residuals — {band_title} power vs temperature — {title_roi}",
                        band_color=_get_band_color(band),
                        output_path=out_dir / f"{base_name}_partial",
                        method_code=method2_code,
                        is_partial_residuals=True,
                        rng=rng,
                        logger=logger,
                    )

        if temp_records:
            df_t = pd.DataFrame(temp_records)
            rej_t, crit_t = _fdr_bh(df_t["p_pooled"].to_numpy(), alpha=BEHAV_FDR_ALPHA)
            df_t["fdr_reject"] = rej_t
            df_t["fdr_crit_p"] = crit_t
            out_stats_t = stats_dir / "group_pooled_corr_pow_roi_vs_temp.tsv"
            df_t.to_csv(out_stats_t, sep="\t", index=False)
            try:
                logger.info(f"Wrote pooled ROI vs temperature stats: {out_stats_t}")
            except Exception:
                pass

# Residual Diagnostic Visualization for Regression Analysis
# -----------------------------------------------------------------------------

def plot_regression_residual_diagnostics(
    x_data: pd.Series,
    y_data: pd.Series,
    title_prefix: str,
    output_path: Path,
    band_color: str = "#4C72B0",
    logger: Optional[logging.Logger] = None,
) -> None:
    """Create comprehensive residual diagnostic plots for regression analysis.
    
    Generates a 4-panel diagnostic figure with:
    1. Residuals vs Fitted Values (heteroscedasticity check)
    2. Q-Q Plot for normality assessment
    3. Scale-Location plot (sqrt of standardized residuals vs fitted)
    4. Residuals vs Leverage (Cook's distance contours)
    
    Args:
        x_data: Independent variable (power values)
        y_data: Dependent variable (ratings/temperature)
        title_prefix: Base title for the diagnostic plots
        output_path: Output file path (without extension)
        band_color: Color for plot elements
        logger: Optional logger for warnings
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Clean and align data
    df = pd.concat([x_data.rename("x"), y_data.rename("y")], axis=1).dropna()
    if len(df) < 10:
        logger.warning(f"Too few valid points ({len(df)}) for meaningful residual diagnostics")
        return
        
    x_clean = df["x"].to_numpy()
    y_clean = df["y"].to_numpy()
    
    # Fit linear regression
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    X = x_clean.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X, y_clean)
    
    # Calculate diagnostic metrics
    y_pred = reg.predict(X)
    residuals = y_clean - y_pred
    std_residuals = residuals / np.std(residuals, ddof=1)
    
    # Leverage (hat values) - diagonal of hat matrix
    X_design = np.column_stack([np.ones(len(X)), X.flatten()])
    try:
        H = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
        leverage = np.diag(H)
    except np.linalg.LinAlgError:
        # Fallback if matrix is singular
        leverage = np.full(len(X), 1.0 / len(X))
    
    # Cook's distance
    n_params = 2  # intercept + slope
    cooks_d = (std_residuals**2 / n_params) * (leverage / (1 - leverage)**2)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Regression Diagnostics — {title_prefix}", fontsize=14, fontweight='bold')
    
    # 1. Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.6, color=band_color, s=25, edgecolors='white', linewidths=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1)
    
    # Add LOESS smooth line for trend detection
    try:
        from scipy.interpolate import UnivariateSpline
        if len(np.unique(y_pred)) > 3:
            sort_idx = np.argsort(y_pred)
            
            # Robust spline fitting with configurable parameters
            smoothing_factor = max(len(y_pred) * SPLINE_SMOOTHING_FRAC, SPLINE_MIN_SMOOTHING)
            max_iter = SPLINE_MAX_ITER
            tolerance = SPLINE_TOL
            
            try:
                spline = UnivariateSpline(
                    y_pred[sort_idx], 
                    residuals[sort_idx], 
                    s=smoothing_factor,
                    k=3  # Cubic spline
                )
                y_pred_smooth = np.linspace(y_pred.min(), y_pred.max(), 100)
                residuals_smooth = spline(y_pred_smooth)
                ax1.plot(y_pred_smooth, residuals_smooth, color='darkblue', linewidth=2, alpha=0.8)
            except Exception as spline_error:
                # Fallback to simple moving average if spline fails
                logger.warning(f"Spline smoothing failed, using moving average: {spline_error}")
                window_size = max(3, len(y_pred) // 10)
                y_pred_smooth = np.linspace(y_pred.min(), y_pred.max(), min(50, len(y_pred)))
                residuals_smooth = np.interp(y_pred_smooth, y_pred[sort_idx], residuals[sort_idx])
                ax1.plot(y_pred_smooth, residuals_smooth, color='darkblue', linewidth=2, alpha=0.8, linestyle='--')
    except (ImportError, ValueError):
        pass
        
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q Plot for normality
    ax2 = axes[0, 1]
    from scipy import stats as scipy_stats
    scipy_stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor(band_color)
    ax2.get_lines()[0].set_markeredgecolor('white')
    ax2.get_lines()[0].set_markersize(6)
    ax2.get_lines()[0].set_alpha(0.7)
    ax2.get_lines()[1].set_color('red')
    ax2.get_lines()[1].set_linewidth(2)
    ax2.set_title("Normal Q-Q Plot")
    ax2.grid(True, alpha=0.3)
    
    # Add Shapiro-Wilk test result
    if len(residuals) <= 5000:  # Shapiro-Wilk limit
        shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals)
        ax2.text(0.05, 0.95, f"Shapiro-Wilk: W={shapiro_stat:.3f}, p={shapiro_p:.3f}", 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Scale-Location (sqrt of |standardized residuals| vs fitted)
    ax3 = axes[1, 0]
    sqrt_abs_std_resid = np.sqrt(np.abs(std_residuals))
    ax3.scatter(y_pred, sqrt_abs_std_resid, alpha=0.6, color=band_color, s=25, edgecolors='white', linewidths=0.3)
    
    # Add smooth trend line
    try:
        if len(np.unique(y_pred)) > 3:
            sort_idx = np.argsort(y_pred)
            
            # Robust spline fitting
            smoothing_factor = max(len(y_pred) * 0.1, 1.0)
            try:
                spline = UnivariateSpline(
                    y_pred[sort_idx], 
                    sqrt_abs_std_resid[sort_idx], 
                    s=smoothing_factor,
                    k=3
                )
                y_pred_smooth = np.linspace(y_pred.min(), y_pred.max(), 100)
                sqrt_smooth = spline(y_pred_smooth)
                ax3.plot(y_pred_smooth, sqrt_smooth, color='darkblue', linewidth=2, alpha=0.8)
            except Exception as spline_error:
                # Fallback to interpolation
                logger.warning(f"Spline smoothing failed for scale-location plot: {spline_error}")
                y_pred_smooth = np.linspace(y_pred.min(), y_pred.max(), min(50, len(y_pred)))
                sqrt_smooth = np.interp(y_pred_smooth, y_pred[sort_idx], sqrt_abs_std_resid[sort_idx])
                ax3.plot(y_pred_smooth, sqrt_smooth, color='darkblue', linewidth=2, alpha=0.8, linestyle='--')
    except (ImportError, ValueError, Exception):
        pass
        
    ax3.set_xlabel("Fitted Values")
    ax3.set_ylabel("√|Standardized Residuals|")
    ax3.set_title("Scale-Location")
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals vs Leverage (with Cook's distance contours)
    ax4 = axes[1, 1]
    ax4.scatter(leverage, std_residuals, alpha=0.6, color=band_color, s=25, edgecolors='white', linewidths=0.3)
    
    # Add Cook's distance contours
    try:
        xlim = ax4.get_xlim()
        ylim = ax4.get_ylim()
        
        # Create grid for Cook's distance contours
        lev_range = np.linspace(0.001, max(leverage) * 1.1, 100)
        
        # Cook's distance thresholds to show
        cook_levels = [0.5, 1.0]
        for cook_thresh in cook_levels:
            # Cook's D = (std_resid^2 / p) * (leverage / (1-leverage)^2)
            # Solve for std_resid: std_resid = ±sqrt(cook_thresh * p * (1-leverage)^2 / leverage)
            valid_lev = lev_range[lev_range < 0.99]  # Avoid division by zero
            pos_resid = np.sqrt(cook_thresh * n_params * (1 - valid_lev)**2 / valid_lev)
            neg_resid = -pos_resid
            
            ax4.plot(valid_lev, pos_resid, 'r--', alpha=0.6, linewidth=1, 
                    label=f"Cook's D = {cook_thresh}" if cook_thresh == cook_levels[0] else "")
            ax4.plot(valid_lev, neg_resid, 'r--', alpha=0.6, linewidth=1)
            
    except (ValueError, ZeroDivisionError, FloatingPointError):
        pass
    
    # Highlight points with high Cook's distance
    high_cook_mask = cooks_d > 4.0 / len(x_clean)  # Common threshold: 4/n
    if np.any(high_cook_mask):
        ax4.scatter(leverage[high_cook_mask], std_residuals[high_cook_mask], 
                   color='red', s=40, alpha=0.8, edgecolors='darkred', linewidths=1,
                   label=f'High Cook\'s D (>{4.0/len(x_clean):.3f})')
    
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_xlabel("Leverage")
    ax4.set_ylabel("Standardized Residuals")
    ax4.set_title("Residuals vs Leverage")
    ax4.grid(True, alpha=0.3)
    if ax4.get_legend_handles_labels()[0]:  # Only show legend if there are labeled elements
        ax4.legend(fontsize=8)
    
    # Add summary statistics text box
    summary_stats = f"""Diagnostic Summary:
n = {len(residuals)}
RMSE = {np.sqrt(np.mean(residuals**2)):.3f}
Max |Cook's D| = {np.max(cooks_d):.3f}
Max Leverage = {np.max(leverage):.3f}"""
    
    fig.text(0.02, 0.98, summary_stats, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.08)
    _save_fig(fig, output_path)


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
            axes[i].scatter(x_valid, y_valid, alpha=0.6, s=30, color=_get_band_color(band))
            
            # Add regression line
            z = np.polyfit(x_valid, y_valid, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
            axes[i].plot(x_line, p(x_line), 'r--', alpha=0.8)
            
            axes[i].set_xlabel(f'{band.capitalize()} Power\nlog10(power/baseline)')
            axes[i].set_ylabel('Behavioral Rating')
            axes[i].set_title(f'{band.capitalize()} Power vs Behavior')
            axes[i].grid(True, alpha=0.3)
            
            # Add correlation statistics
            rho, p_spear = stats.spearmanr(x_valid, y_valid)
            
            stats_text = f'Spearman ρ={rho:.3f} (p={p_spear:.3f})\nn={len(x_valid)}'
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


def plot_significant_correlations_topomap(pow_df: pd.DataFrame, y: pd.Series, bands: List[str], 
                                         info: mne.Info, subject: str, save_dir: Path, 
                                         logger: logging.Logger, alpha: float = 0.05):
    """Create topographical maps showing significant correlations for each frequency band.
    
    Uses MNE's plot_topomap with mask parameter to highlight significant electrodes.
    """
    try:
        # Calculate correlations for all bands
        bands_with_data = []
        
        for band in bands:
            band_cols = [col for col in pow_df.columns if col.startswith(f'pow_{band}_')]
            if not band_cols:
                continue
                
            # Extract channel names and calculate correlations
            ch_names = [col.replace(f'pow_{band}_', '') for col in band_cols]
            correlations = []
            p_values = []
            
            for col in band_cols:
                # Clean correlation calculation
                valid_data = pd.concat([pow_df[col], y], axis=1).dropna()
                if len(valid_data) >= 5:
                    r, p = stats.spearmanr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
                else:
                    r, p = np.nan, 1.0
                correlations.append(r)
                p_values.append(p)
            
            # Significance mask (may be all False if no significant channels)
            sig_mask = np.array(p_values) < alpha
            bands_with_data.append({
                'band': band,
                'channels': ch_names,
                'correlations': np.array(correlations),
                'p_values': np.array(p_values),
                'significant_mask': sig_mask
            })
        
        if not bands_with_data:
            logger.warning("No significant correlations found across any frequency band")
            return
        
        # Create figure
        n_bands = len(bands_with_data)
        fig, axes = plt.subplots(1, n_bands, figsize=(4.8 * n_bands, 4.8))
        if n_bands == 1:
            axes = [axes]
        # Make heads larger and reduce whitespace between them; reserve more space below for colorbar
        plt.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.20, wspace=0.08)
        
        # Calculate global color limits based on correlations
        all_sig_corrs = []
        for band_data in bands_with_data:
            sig_corrs = band_data['correlations'][band_data['significant_mask']]
            all_sig_corrs.extend(sig_corrs[np.isfinite(sig_corrs)])
        if all_sig_corrs:
            vmax = max(abs(np.min(all_sig_corrs)), abs(np.max(all_sig_corrs)))
        else:
            # If none significant, fall back to all correlations' robust max
            all_corrs = []
            for band_data in bands_with_data:
                all_corrs.extend(band_data['correlations'][np.isfinite(band_data['correlations'])])
            vmax = max(abs(np.min(all_corrs)), abs(np.max(all_corrs))) if all_corrs else 0.5
        
        successful_plots = []
        
        for i, band_data in enumerate(bands_with_data):
            ax = axes[i]
            
            try:
                # Create data arrays matching info channel order
                n_info_chs = len(info['ch_names'])
                topo_data = np.zeros(n_info_chs)
                topo_mask = np.zeros(n_info_chs, dtype=bool)
                
                # Map band data to info channels
                for j, info_ch in enumerate(info['ch_names']):
                    if info_ch in band_data['channels']:
                        ch_idx = band_data['channels'].index(info_ch)
                        topo_data[j] = band_data['correlations'][ch_idx] if np.isfinite(band_data['correlations'][ch_idx]) else 0
                        topo_mask[j] = band_data['significant_mask'][ch_idx]
                
                # Select EEG channels
                picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
                if len(picks) == 0:
                    raise ValueError("No EEG channels found")
                
                # Plot topomap
                im, _ = mne.viz.plot_topomap(
                    topo_data[picks],
                    mne.pick_info(info, picks),
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vlim=(-vmax, vmax),
                    contours=6,
                    mask=topo_mask[picks],
                    mask_params=dict(
                        marker='o', 
                        markerfacecolor='white', 
                        markeredgecolor='black', 
                        linewidth=1, 
                        markersize=6
                    )
                )
                
                successful_plots.append(im)
                
                # Add title
                n_sig = topo_mask[picks].sum()
                n_total = len([ch for ch in band_data['channels'] if ch in info['ch_names']])
                ax.set_title(
                    f'{band_data["band"].upper()}\n{n_sig}/{n_total} significant',
                    fontweight='bold', fontsize=12, pad=10
                )
                
            except Exception as e:
                logger.warning(f"Failed to plot {band_data['band']} topomap: {e}")
                ax.text(0.5, 0.5, f'{band_data["band"].upper()}\nPlot failed', 
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.set_title(f'{band_data["band"].upper()}\n(Error)', fontweight='bold', pad=10)
        
        # Main title
        plt.suptitle(
            f'Significant EEG-Pain Correlations (p < {alpha})\nSubject {subject}',
            fontweight='bold', fontsize=14, y=1.02
        )
        
        # Add colorbar if successful plots exist
        if successful_plots:
            # Build a dynamic, centered colorbar under the span of the topomap axes
            # Use axes positions (in figure fraction) to define cbar extents
            left = min(ax.get_position().x0 for ax in axes)
            right = max(ax.get_position().x1 for ax in axes)
            bottom = min(ax.get_position().y0 for ax in axes)
            span = right - left
            # Colorbar occupies 55% of head span and sits with a fixed gap below
            cb_width = 0.55 * span
            cb_left = left + 0.225 * span  # center it
            cb_bottom = max(0.04, bottom - 0.06)  # keep a bit away from heads and above bottom edge
            cax = fig.add_axes([cb_left, cb_bottom, cb_width, 0.028])
            cbar = fig.colorbar(successful_plots[-1], cax=cax, orientation='horizontal')
            cbar.set_label('Spearman correlation (ρ)', fontweight='bold', fontsize=11)
            cbar.ax.tick_params(pad=2, labelsize=9)
        
        # Save
        _save_fig(fig, save_dir / f'sub-{subject}_significant_correlations_topomap')
        plt.close(fig)
        
        logger.info(f"Created topomaps for {len(bands_with_data)} frequency bands: "
                   f"{[bd['band'] for bd in bands_with_data]}")
        
    except Exception as e:
        logger.error(f"Failed to create significant correlations topomap: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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


def plot_top_behavioral_predictors(
    subject: str,
    task: str = TASK,
    alpha: float = None,
    top_n: int = None
) -> None:
    """Plot top N significant behavioral predictors as horizontal bar chart.
    
    Creates a publication-ready horizontal bar chart showing the channel-band combinations
    with the strongest significant correlations with behavioral ratings, similar to the
    format shown in your example figure.
    
    Parameters
    ----------
    subject : str
        Subject identifier
    task : str
        Task name
    alpha : float
        Significance threshold (default 0.05)
    top_n : int
        Number of top predictors to show (default 20)
    """
    if alpha is None:
        alpha = BEHAV_FDR_ALPHA
    if top_n is None:
        top_n = int(config.get("behavior_analysis.predictors.top_n", 20))
    logger = _setup_logging(subject)
    logger.info(f"Creating top {top_n} behavioral predictors plot for sub-{subject}")
    
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    
    # Load combined correlation statistics for available targets (rating, temperature)
    candidate_files = [
        ("rating", stats_dir / "corr_stats_pow_combined_vs_rating.tsv"),
        ("temp", stats_dir / "corr_stats_pow_combined_vs_temp.tsv"),
        ("temperature", stats_dir / "corr_stats_pow_combined_vs_temperature.tsv"),
    ]
    frames = []
    for target_label, path in candidate_files:
        if path.exists():
            try:
                _df = pd.read_csv(path, sep="\t")
                _df["target"] = target_label
                frames.append(_df)
            except Exception as e:
                logger.warning(f"Failed reading stats file {path}: {e}")
    if not frames:
        logger.warning(
            "No combined correlation stats found for rating or temperature. Expected one of: "
            + ", ".join(str(p) for _, p in candidate_files)
        )
        return
    
    try:
        # Combine correlation statistics across available targets
        df = pd.concat(frames, axis=0, ignore_index=True)
        
        # Filter for significant correlations and valid data
        df_sig = df[
            (df['p'] <= alpha) & 
            df['r'].notna() & 
            df['p'].notna() &
            df['channel'].notna() &
            df['band'].notna()
        ].copy()
        
        if len(df_sig) == 0:
            logger.warning(f"No significant correlations found (p <= {alpha})")
            return
        
        # Calculate absolute correlation and sort by it
        df_sig['abs_r'] = df_sig['r'].abs()
        df_top = df_sig.nlargest(top_n, 'abs_r')
        
        if len(df_top) == 0:
            logger.warning("No top correlations to plot")
            return
        
        # Create predictor labels (channel + band) with target for clarity
        # Example: CP6 (alpha) [rating] or CP6 (alpha) [temp]
        if 'target' in df_top.columns:
            df_top['predictor'] = df_top['channel'] + ' (' + df_top['band'] + ') [' + df_top['target'].astype(str) + ']'
        else:
            df_top['predictor'] = df_top['channel'] + ' (' + df_top['band'] + ')'
        
        # Sort by absolute correlation (ascending for horizontal bar plot)
        df_top = df_top.sort_values('abs_r', ascending=True)
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
        
        # Create color mapping for bands
        band_colors = {}
        for band in df_top['band'].unique():
            if band in BAND_COLORS:
                band_colors[band] = BAND_COLORS[band]
            else:
                # Fallback colors if band not in BAND_COLORS
                fallback_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                band_idx = list(df_top['band'].unique()).index(band)
                band_colors[band] = fallback_colors[band_idx % len(fallback_colors)]
        
        # Create colors list for each bar
        colors = [band_colors[band] for band in df_top['band']]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(df_top))
        bars = ax.barh(y_pos, df_top['abs_r'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_top['predictor'], fontsize=11)
        ax.set_xlabel(f"|Spearman ρ| with Behavior (p < {alpha})", fontweight='bold', fontsize=12)
        ax.set_title(f'Top {top_n} Significant Behavioral Predictors', fontweight='bold', fontsize=14, pad=20)
        
        # Add correlation values and p-values as text annotations
        for i, (_, row) in enumerate(df_top.iterrows()):
            r_val = row['r']
            p_val = row['p']
            abs_r_val = row['abs_r']
            
            # Position text slightly to the right of the bar
            x_pos = abs_r_val + 0.01
            ax.text(x_pos, i, f'{abs_r_val:.3f} (p={p_val:.3f})', 
                   va='center', ha='left', fontsize=10, fontweight='normal')
        
        # Set x-axis limits with some padding
        max_r = df_top['abs_r'].max()
        ax.set_xlim(0, max_r * 1.25)
        
        # Add grid for better readability
        ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        output_path = plots_dir / f'sub-{subject}_top_{top_n}_behavioral_predictors'
        _save_fig(fig, output_path)
        plt.close(fig)
        
        logger.info(f"Saved top {top_n} behavioral predictors plot: {output_path}.png")
        # Summarize counts by target if present
        if 'target' in df.columns:
            counts_by_tgt = df_sig['target'].value_counts().to_dict() if len(df_sig) else {}
            logger.info(f"Found {len(df_top)} significant predictors across targets {counts_by_tgt} (out of {len(df)} total correlations)")
        else:
            logger.info(f"Found {len(df_top)} significant predictors (out of {len(df)} total correlations)")
        
        # Export the top predictors data for reference
        top_predictors_file = stats_dir / f"top_{top_n}_behavioral_predictors.tsv"
        # Build export, including target when available
        export_cols = ['predictor', 'channel', 'band', 'r', 'abs_r', 'p', 'n']
        if 'target' in df_top.columns and 'target' not in export_cols:
            export_cols = ['target'] + export_cols
        df_top_export = df_top[export_cols].copy()
        df_top_export = df_top_export.sort_values('abs_r', ascending=False)  # Sort descending for export
        df_top_export.to_csv(top_predictors_file, sep="\t", index=False)
        logger.info(f"Exported top predictors data: {top_predictors_file}")
        
    except Exception as e:
        logger.error(f"Failed to create top behavioral predictors plot: {e}")
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
        obs, _ = stats.spearmanr(df["x"], df["y"], nan_policy="omit")
        ge = 1
        y_vals = df["y"].to_numpy()
        for _ in range(int(n_perm)):
            y_pi = y_vals[rng.permutation(len(y_vals))]
            rp, _ = stats.spearmanr(df["x"], y_pi, nan_policy="omit")
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
            ge = 1
            for _ in range(int(n_perm)):
                ry_pi = ry[rng.permutation(len(ry))]
                rp, _ = stats.pearsonr(rx, ry_pi)
                if np.abs(rp) >= np.abs(obs) - 1e-12:
                    ge += 1
            return ge / (int(n_perm) + 1)
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
            # Always use Spearman
            r, p = stats.spearmanr(xi[mask], y[mask], nan_policy="omit")
            method = "spearman"

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
            if bootstrap and n_eff >= 5:
                idx = np.where(mask.to_numpy())[0]
                boots: List[float] = []
                for _ in range(int(bootstrap)):
                    bidx = rng.choice(idx, size=len(idx), replace=True)
                    xb = xi.iloc[bidx]
                    yb = y.iloc[bidx]
                    rb, _ = stats.spearmanr(xb, yb, nan_policy="omit")
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
                if len(xi) != len(temp_series):
                    logger.warning(f"Channel vs temp length mismatch: power={len(xi)}, temp={len(temp_series)}. Using overlap.")
                n_len_t = min(len(xi), len(temp_series))
                xt = xi.iloc[:n_len_t]
                tt = temp_series.iloc[:n_len_t]
                m2 = xt.notna() & tt.notna()
                n_eff2 = int(m2.sum())
                if n_eff2 >= 5:
                    # Always use Spearman
                    r2, p2 = stats.spearmanr(xt[m2], tt[m2], nan_policy="omit")
                    method2 = "spearman"
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
                    if bootstrap and n_eff2 >= 5:
                        idx2 = np.where(m2.to_numpy())[0]
                        boots2: List[float] = []
                        for _ in range(int(bootstrap)):
                            bidx2 = rng.choice(idx2, size=len(idx2), replace=True)
                            xb = xt.iloc[bidx2]
                            tb = tt.iloc[bidx2]
                            rb, _ = stats.spearmanr(xb, tb, nan_policy="omit")
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
            rej, crit = _fdr_bh(pvec, alpha=BEHAV_FDR_ALPHA)
            df["fdr_reject"] = rej
            df["fdr_crit_p"] = crit
            df.to_csv(stats_dir / f"corr_stats_conn_roi_summary_{_sanitize(pref)}_vs_rating.tsv", sep="\t", index=False)

        if recs_temp:
            df_t = pd.DataFrame(recs_temp)
            pvec_t = df_t["p_perm"].to_numpy() if "p_perm" in df_t.columns and np.isfinite(df_t["p_perm"]).any() else df_t["p"].to_numpy()
            rej_t, crit_t = _fdr_bh(pvec_t, alpha=BEHAV_FDR_ALPHA)
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
            # Always use Spearman
            r, p = stats.spearmanr(xi[mask], y[mask], nan_policy="omit")
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
        rej_valid, crit_p = _fdr_bh(p_valid, alpha=BEHAV_FDR_ALPHA)
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
# Export: All significant predictors (ROI + channel-level)
# -----------------------------------------------------------------------------

def export_all_significant_predictors(subject: str, alpha: float = 0.05, use_fdr: bool = True) -> None:
    """Export all significant EEG predictors (ROI + channel-level) to single CSV.
    
    Combines ROI-level and channel-level power correlations with behavioral ratings,
    filters for significant predictors, and exports to a consolidated CSV file.
    
    Parameters
    ----------
    subject : str
        Subject identifier
    alpha : float
        Significance threshold (default 0.05)
    use_fdr : bool
        Use FDR-corrected p-values if available (default True)
    """
    stats_dir = _stats_dir(subject)
    _ensure_dir(stats_dir)
    
    logger = _setup_logging(subject)
    logger.info(f"Exporting all significant predictors for sub-{subject} (alpha={alpha})")
    
    all_predictors = []
    
    # 1. Load ROI-level correlations for available targets
    for target in ("rating", "temp", "temperature"):
        roi_file = stats_dir / f"corr_stats_pow_roi_vs_{target}.tsv"
        if not roi_file.exists():
            continue
        try:
            roi_df = pd.read_csv(roi_file, sep="\t")
            # Select significant using FDR when available and requested
            if use_fdr and "fdr_reject" in roi_df.columns:
                significant_roi = roi_df[roi_df["fdr_reject"] == True].copy()
            elif use_fdr and "fdr_crit_p" in roi_df.columns and "p" in roi_df.columns:
                significant_roi = roi_df[roi_df["p"] <= roi_df["fdr_crit_p"]].copy()
            else:
                significant_roi = roi_df[roi_df["p"] <= alpha].copy()
            if len(significant_roi) > 0:
                # Add predictor type, target, and format predictor name
                significant_roi["predictor_type"] = "ROI"
                significant_roi["target"] = target
                significant_roi["predictor"] = significant_roi["roi"] + " (" + significant_roi["band"] + ")"
                # Select and rename columns for consistency
                roi_cols = {
                    "predictor": "predictor",
                    "roi": "region",
                    "band": "band",
                    "r": "r",
                    "p": "p",
                    "n": "n",
                    "predictor_type": "type",
                    "target": "target",
                }
                if "fdr_reject" in significant_roi.columns:
                    roi_cols["fdr_reject"] = "fdr_significant"
                if "fdr_crit_p" in significant_roi.columns:
                    roi_cols["fdr_crit_p"] = "fdr_critical_p"
                roi_subset = significant_roi[list(roi_cols.keys())].rename(columns=roi_cols)
                all_predictors.append(roi_subset)
                logger.info(f"Found {len(significant_roi)} significant ROI predictors for target '{target}'")
        except Exception as e:
            logger.warning(f"Failed to load ROI correlations from {roi_file}: {e}")
    
    # 2. Load combined channel-level correlations for available targets
    for target in ("rating", "temp", "temperature"):
        combined_file = stats_dir / f"corr_stats_pow_combined_vs_{target}.tsv"
        if not combined_file.exists():
            continue
        try:
            chan_df = pd.read_csv(combined_file, sep="\t")
            # Select significant using FDR when available and requested
            if use_fdr and "fdr_reject" in chan_df.columns:
                significant_chan = chan_df[chan_df["fdr_reject"] == True].copy()
            elif use_fdr and "fdr_crit_p" in chan_df.columns and "p" in chan_df.columns:
                significant_chan = chan_df[chan_df["p"] <= chan_df["fdr_crit_p"]].copy()
            else:
                significant_chan = chan_df[chan_df["p"] <= alpha].copy()
            if len(significant_chan) > 0:
                # Add predictor type, target, and format predictor name
                significant_chan["predictor_type"] = "Channel"
                significant_chan["target"] = target
                significant_chan["predictor"] = significant_chan["channel"] + " (" + significant_chan["band"] + ")"
                # Select and rename columns for consistency
                chan_cols = {
                    "predictor": "predictor",
                    "channel": "region",
                    "band": "band",
                    "r": "r",
                    "p": "p",
                    "n": "n",
                    "predictor_type": "type",
                    "target": "target",
                }
                if "fdr_reject" in significant_chan.columns:
                    chan_cols["fdr_reject"] = "fdr_significant"
                if "fdr_crit_p" in significant_chan.columns:
                    chan_cols["fdr_crit_p"] = "fdr_critical_p"
                chan_subset = significant_chan[list(chan_cols.keys())].rename(columns=chan_cols)
                all_predictors.append(chan_subset)
                logger.info(f"Found {len(significant_chan)} significant channel predictors for target '{target}'")
        except Exception as e:
            logger.warning(f"Failed to load channel correlations from {combined_file}: {e}")
    
    # 3. Combine and export
    if all_predictors:
        # Concatenate all significant predictors
        combined_df = pd.concat(all_predictors, ignore_index=True)
        
        # Add absolute correlation for reference
        combined_df["abs_r"] = combined_df["r"].abs()
        
        # Sort by p-value (lowest first)
        combined_df = combined_df.sort_values("p", ascending=True)
        
        # Export to CSV
        output_file = stats_dir / "all_significant_predictors.csv"
        combined_df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(combined_df)} total significant predictors to: {output_file}")
        logger.info(f"  - ROI predictors: {len([p for p in all_predictors if 'ROI' in str(p.get('type', '').iloc[0] if len(p) > 0 else '')])}")
        logger.info(f"  - Channel predictors: {len([p for p in all_predictors if 'Channel' in str(p.get('type', '').iloc[0] if len(p) > 0 else '')])}")
        
        # Summary statistics
        n_roi = len(combined_df[combined_df["type"] == "ROI"])
        n_chan = len(combined_df[combined_df["type"] == "Channel"])
        max_r = combined_df["abs_r"].max()
        strongest = combined_df.iloc[0]
        
        logger.info(f"Summary: {n_roi} ROI + {n_chan} channel predictors")
        logger.info(f"Strongest predictor: {strongest['predictor']} (r={strongest['r']:.3f})")
        
    else:
        logger.warning("No significant predictors found")
        # Create empty file for consistency
        empty_df = pd.DataFrame(columns=["predictor", "region", "band", "r", "p", "n", "type", "target", "abs_r"])
        output_file = stats_dir / "all_significant_predictors.csv"
        empty_df.to_csv(output_file, index=False)

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
# REMOVED: plot_neural_behavioral_state_space function


# -----------------------------------------------------------------------------
# Time-Frequency Power-Behavior Correlation Heatmap
# -----------------------------------------------------------------------------

def plot_time_frequency_correlation_heatmap(
    subject: str,
    task: str = TASK,
    use_spearman: bool = None,
) -> None:
    """Create time-frequency correlation heatmap showing power-behavior relationships.
    
    This visualization reveals how correlations between EEG power and behavioral ratings
    vary across both time and frequency dimensions, providing insights into the temporal
    dynamics of brain-behavior coupling.
    
    Parameters
    ----------
    subject : str
        Subject identifier
    task : str
        Task name
    use_spearman : bool
        Use Spearman (True) or Pearson (False) correlation
    time_resolution : float
        Time bin size in seconds
    freq_resolution : float
        Frequency bin size in Hz
    time_window : tuple
        (tmin, tmax) in seconds relative to event onset
    freq_range : tuple
        (fmin, fmax) frequency range in Hz
    alpha : float
        Significance threshold for FDR correction
    roi_selection : str or None
        ROI name to focus on (None for all channels)
    """
    logger = _setup_logging(subject)
    logger.info(f"Creating time-frequency correlation heatmap for sub-{subject}")
    
    # Get behavior analysis parameters from centralized config
    behavior_config = config.get('behavior_analysis', {})
    heatmap_config = behavior_config.get('time_frequency_heatmap', {})
    viz_config = behavior_config.get('visualization', {})
    spline_config = behavior_config.get('spline', {})
    stats_config = behavior_config.get('statistics', {})
    
    # Extract parameters with defaults
    time_resolution = heatmap_config.get('time_resolution', 0.1)
    freq_resolution = heatmap_config.get('freq_resolution', 2.0)
    time_window = tuple(heatmap_config.get('time_window', [-0.5, 2.0]))
    freq_range = tuple(heatmap_config.get('freq_range', [4.0, 40.0]))
    alpha = heatmap_config.get('alpha', BEHAV_FDR_ALPHA)
    roi_selection = heatmap_config.get('roi_selection')
    if roi_selection == "null":
        roi_selection = None
    n_cycles_factor = heatmap_config.get('n_cycles_factor', 2.0)
    decim = heatmap_config.get('decim', 3)
    min_valid_points = heatmap_config.get('min_valid_points', 5)
    
    # Correlation method default from global statistics setting
    if use_spearman is None:
        use_spearman = bool(config.get("statistics.use_spearman_default", True))

    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)
    
    # Load epochs and behavioral data
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None:
        logger.error(f"Could not find epochs for sub-{subject}")
        return
        
    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
    events = _load_events_df(subject, task)
    aligned_events = _align_events_to_epochs(events, epochs) if events is not None else None
    
    if aligned_events is None:
        logger.error(f"Could not align events for sub-{subject}")
        return
        
    # Get behavioral ratings
    rating_col = _pick_first_column(aligned_events, RATING_COLUMNS)
    if rating_col is None:
        logger.error(f"No rating column found for sub-{subject}")
        return
        
    y = pd.to_numeric(aligned_events[rating_col], errors="coerce")
    if y.isna().all():
        logger.error(f"All behavioral ratings are NaN for sub-{subject}")
        return
    
    # Select channels based on ROI if specified
    if roi_selection is not None:
        roi_map = _build_rois(epochs.info)
        if roi_selection in roi_map:
            channels = roi_map[roi_selection]
            epochs = epochs.pick_channels(channels)
            logger.info(f"Selected {len(channels)} channels from {roi_selection} ROI")
        else:
            logger.warning(f"ROI {roi_selection} not found, using all channels")
    
    # Create time-frequency representation
    freqs = np.arange(freq_range[0], freq_range[1] + freq_resolution, freq_resolution)
    n_cycles = freqs / n_cycles_factor  # Adaptive number of cycles
    
    logger.info(f"Computing time-frequency decomposition: {len(freqs)} frequencies, {len(epochs)} epochs")
    tfr = mne.time_frequency.tfr_morlet(
        epochs, 
        freqs=freqs, 
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=decim,
        n_jobs=1,
        average=False,  # IMPORTANT: keep epoch dimension for trial-wise correlations
        verbose=False
    )

    # Validate epoch-resolved TFR shape
    try:
        ndim = getattr(getattr(tfr, 'data', None), 'ndim', None)
        if ndim != 4:
            logger.error(f"TFR must be epoch-resolved with shape (epochs, channels, freqs, times); got ndim={ndim}")
            return
        logger.info(f"TFR shape: {tfr.data.shape}; channels={len(tfr.ch_names)}; freqs={len(freqs)}; times={len(tfr.times)}")
    except Exception as e:
        logger.error(f"Failed to validate TFR shape: {e}")
        return

    # Apply baseline correction to obtain log10(power/baseline) prior to correlations
    baseline_applied = False
    baseline_window_used: Optional[Tuple[float, float]] = None
    try:
        bl = config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01])
        b_start = float(bl[0]) if bl[0] is not None else float(tfr.times[0])
        b_end = float(bl[1]) if bl[1] is not None else 0.0
        # Clip baseline window to available time range and ≤ 0 s
        tmin_avail, tmax_avail = float(tfr.times[0]), float(tfr.times[-1])
        b_start_clip = max(b_start, tmin_avail)
        b_end_clip = min(b_end, 0.0, tmax_avail)
        # Ensure enough baseline samples as in other scripts
        min_bl_samp = int(config.get("time_frequency_analysis.min_baseline_samples", 5))
        times_arr = np.asarray(tfr.times)
        bl_mask = (times_arr >= b_start_clip) & (times_arr <= b_end_clip)
        if b_start_clip >= b_end_clip or int(bl_mask.sum()) < max(1, min_bl_samp):
            logger.warning(
                f"Baseline window [{b_start}, {b_end}] invalid/insufficient for available times "
                f"[{tmin_avail}, {tmax_avail}] (samples={int(bl_mask.sum())}); skipping baseline for TF heatmap."
            )
        else:
            if (b_start_clip != b_start) or (b_end_clip != b_end):
                logger.info(
                    f"Clipped baseline window from [{b_start}, {b_end}] to "
                    f"[{b_start_clip}, {b_end_clip}] to fit data range."
                )
            tfr.apply_baseline(baseline=(b_start_clip, b_end_clip), mode="logratio")
            baseline_applied = True
            baseline_window_used = (b_start_clip, b_end_clip)
    except Exception as e:
        logger.warning(f"TF heatmap baseline correction failed: {e}")

    # Crop to desired time window, clipping to available range to avoid warnings
    tmin_req, tmax_req = float(time_window[0]), float(time_window[1])
    tmin_avail, tmax_avail = float(tfr.times[0]), float(tfr.times[-1])
    tmin_clip = max(tmin_req, tmin_avail)
    tmax_clip = min(tmax_req, tmax_avail)
    if tmin_clip > tmax_clip:
        # Fallback to full available window
        logger.warning(
            f"Requested crop window [{tmin_req}, {tmax_req}] invalid for available times "
            f"[{tmin_avail}, {tmax_avail}]; using full range."
        )
        tmin_clip, tmax_clip = tmin_avail, tmax_avail
    elif tmin_clip != tmin_req or tmax_clip != tmax_req:
        logger.info(
            f"Clipped crop window from [{tmin_req}, {tmax_req}] to "
            f"[{tmin_clip}, {tmax_clip}] to fit data range."
        )
    tfr.crop(tmin=tmin_clip, tmax=tmax_clip)

    # Create time bins within the effective cropped range
    times = tfr.times
    tb_start = float(times[0])
    tb_end = float(times[-1])
    time_bins = np.arange(tb_start, tb_end + time_resolution, time_resolution)
    if time_bins.size < 2:
        # Ensure at least one bin edge pair
        time_bins = np.array([tb_start, tb_end], dtype=float)
    time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # Initialize correlation arrays
    n_time_bins = len(time_bin_centers)
    n_freqs = len(freqs)
    n_channels = len(tfr.ch_names)
    
    correlations = np.full((n_freqs, n_time_bins), np.nan)
    p_values = np.full((n_freqs, n_time_bins), np.nan)
    n_valid = np.full((n_freqs, n_time_bins), 0)
    
    # Compute correlations for each time-frequency bin
    logger.info("Computing correlations across time-frequency grid...")
    
    for t_idx, t_center in enumerate(time_bin_centers):
        # Find time indices for this bin
        t_start = t_center - time_resolution / 2
        t_end = t_center + time_resolution / 2
        time_mask = (times >= t_start) & (times <= t_end)
        
        if not np.any(time_mask):
            continue
            
        for f_idx, freq in enumerate(freqs):
            # Get power data for this frequency
            # Expect TFR data shape: (epochs, channels, freqs, times)
            if tfr.data.ndim != 4:
                logger.error(f"Unexpected TFR data dimensionality: {tfr.data.ndim}; expected 4")
                continue
            # 4D: (epochs, channels, freqs, times)
            power_freq = tfr.data[:, :, f_idx, :]  # (epochs, channels, times)
            
            # Apply time mask to get data for this time bin
            power_data = power_freq[:, :, time_mask]  # (epochs, channels, time_points_in_bin)
            
            # Average across channels and time points
            if roi_selection is not None or n_channels == 1:
                # Single ROI or single channel: average across channels and time
                power_values = np.mean(power_data, axis=(1, 2))  # Average across channels and time
            else:
                # Multiple channels: average across all channels and time points  
                power_values = np.mean(power_data, axis=(1, 2))
            
            # Align with behavioral data
            n_common = min(len(power_values), len(y))
            power_vals = power_values[:n_common]
            behavior_vals = y.iloc[:n_common]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(power_vals) | behavior_vals.isna())
            if np.sum(valid_mask) < min_valid_points:  # Need minimum valid points
                continue
                
            power_clean = power_vals[valid_mask]
            behavior_clean = behavior_vals[valid_mask]
            
            # Compute correlation
            try:
                # Always use Spearman for time-frequency correlations
                r, p = stats.spearmanr(power_clean, behavior_clean)
                    
                correlations[f_idx, t_idx] = r
                p_values[f_idx, t_idx] = p
                n_valid[f_idx, t_idx] = np.sum(valid_mask)
                
            except Exception as e:
                logger.warning(f"Correlation failed for freq={freq:.1f}Hz, time={t_center:.2f}s: {e}")
                continue
    
    # FDR correction across all time-frequency points
    valid_p_mask = ~np.isnan(p_values)
    if np.any(valid_p_mask):
        p_flat = p_values[valid_p_mask]
        # Use internal BH adjust to get q-values (adjusted p-values)
        p_corrected_flat = _bh_adjust(p_flat)

        p_corrected = np.full_like(p_values, np.nan)
        p_corrected[valid_p_mask] = p_corrected_flat

        significant_mask = p_corrected < alpha
    else:
        significant_mask = np.zeros_like(p_values, dtype=bool)
    
    # Create separate visualizations
    correlation_vmin = viz_config.get('correlation_vmin', -0.6)
    correlation_vmax = viz_config.get('correlation_vmax', 0.6)
    
    roi_suffix = f"_{roi_selection.lower()}" if roi_selection else ""
    method_suffix = "_spearman"
    
    # Figure 1: Time-frequency correlation heatmap
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    im1 = ax1.imshow(correlations, aspect='auto', origin='lower', 
                     extent=[tb_start, tb_end, freq_range[0], freq_range[1]],
                     cmap='RdBu_r', vmin=correlation_vmin, vmax=correlation_vmax)
    ax1.set_xlabel('Time (s)', fontweight='bold')
    ax1.set_ylabel('Frequency (Hz)', fontweight='bold')
    title_main = 'Time-Frequency Power-Behavior Correlations'
    method_name = 'Spearman'
    roi_name = roi_selection or 'All Channels'
    metric = 'log10(power/baseline)' if baseline_applied else 'raw power'
    bl_txt = ''
    if baseline_applied and baseline_window_used is not None:
        bl_txt = f" | BL: [{baseline_window_used[0]:.2f}, {baseline_window_used[1]:.2f}] s"
    ax1.set_title(
        f"{title_main}\nSubject: {subject} | Method: {method_name} | ROI: {roi_name} | Metric: {metric}{bl_txt}",
        fontsize=14,
        fontweight='bold',
    )
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Correlation (r)')
    cbar1.ax.tick_params(labelsize=12)
    
    # Add frequency band horizontal lines
    for band, (fmin, fmax) in FEATURES_FREQ_BANDS.items():
        if fmin >= freq_range[0] and fmax <= freq_range[1]:
            ax1.axhline(fmin, color='white', linestyle='-', alpha=0.3, linewidth=0.5)
            ax1.axhline(fmax, color='white', linestyle='-', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    fig1_name = f"time_frequency_correlation_heatmap{roi_suffix}{method_suffix}"
    _save_fig(fig1, plots_dir / fig1_name)
    
    # Figure 2: Frequency profile of strongest correlations
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    max_corr_per_freq = np.nanmax(np.abs(correlations), axis=1)
    freq_with_data = freqs[~np.isnan(max_corr_per_freq)]
    max_corr_freq_clean = max_corr_per_freq[~np.isnan(max_corr_per_freq)]
    
    if len(freq_with_data) > 0:
        ax2.plot(freq_with_data, max_corr_freq_clean, 'o-', linewidth=2, markersize=4, color='black')
        ax2.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax2.set_ylabel('Max |Correlation|', fontweight='bold')
        ax2.set_title(f'Frequency Profile of Strongest Power-Behavior Correlations\n'
                     f'Subject: {subject} | Method: Spearman | '
                     f'ROI: {roi_selection or "All Channels"}', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add frequency band annotations using configured colors (includes delta/theta/...)
        for band, (fmin, fmax) in FEATURES_FREQ_BANDS.items():
            if fmin >= freq_range[0] and fmax <= freq_range[1]:
                ax2.axvspan(fmin, fmax, alpha=0.2, color=_get_band_color(band),
                            label=f'{band} ({fmin:g}–{fmax:g}Hz)')
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    fig2_name = f"frequency_profile_correlation{roi_suffix}{method_suffix}"
    _save_fig(fig2, plots_dir / fig2_name)
    
    # Save statistical results
    results_df = pd.DataFrame({
        'frequency': np.repeat(freqs, n_time_bins),
        'time': np.tile(time_bin_centers, n_freqs),
        'correlation': correlations.flatten(),
        'p_value': p_values.flatten(),
        'p_corrected': p_corrected.flatten() if 'p_corrected' in locals() else np.nan,
        'significant': significant_mask.flatten() if 'significant_mask' in locals() else False,
        'n_valid': n_valid.flatten()
    })
    
    # Remove rows with all NaN correlations
    results_df = results_df.dropna(subset=['correlation'])
    
    stats_file = stats_dir / f"time_frequency_correlation_stats{roi_suffix}{method_suffix}.tsv"
    results_df.to_csv(stats_file, sep='\t', index=False)
    
    # Log summary statistics
    if not results_df.empty:
        n_significant = np.sum(results_df['significant'])
        max_r = results_df['correlation'].abs().max()
        best_result = results_df.loc[results_df['correlation'].abs().idxmax()]
        
        logger.info(f"Time-frequency correlation analysis completed:")
        logger.info(f"  - Total time-frequency points: {len(results_df)}")
        logger.info(f"  - Significant correlations (FDR < {alpha}): {n_significant}")
        logger.info(f"  - Maximum |correlation|: {max_r:.3f}")
        logger.info(f"  - Best correlation: r={best_result['correlation']:.3f} at "
                   f"{best_result['frequency']:.1f}Hz, {best_result['time']:.2f}s")
        logger.info(f"  - Results saved to: {stats_file}")
    else:
        logger.warning("No valid correlations computed")


# -----------------------------------------------------------------------------
# Trial-by-Trial Power Evolution with Behavioral Adaptation
# -----------------------------------------------------------------------------

def plot_power_behavior_evolution_across_trials(
    subject: str,
    task: str = TASK,
    window_size: int = 20,
    bands_to_plot: Optional[List[str]] = None
) -> None:
    """
    Plot how power-behavior correlations evolve across trials within the session.
    
    Creates a sliding window analysis showing:
    1. Running correlations between power and behavior across trials
    2. Behavioral trend line across trials  
    3. Power trend lines for key ROIs/bands across trials
    4. Significance bands for correlations
    
    Parameters:
    -----------
    subject : str
        Subject identifier
    task : str
        Task name
    window_size : int
        Number of trials for sliding window correlation (default: 20)
    bands_to_plot : Optional[List[str]] 
        Frequency bands to analyze (default: uses first 3 from POWER_BANDS_TO_USE)
    """
    logger = _setup_logging(subject)
    logger.info(f"Creating power-behavior evolution analysis for sub-{subject}")
    
    plots_dir = _plots_dir(subject)
    _ensure_dir(plots_dir)
    
    try:
        # Load data
        pow_df, _, y, info = _load_features_and_targets(subject, task)
        y = pd.to_numeric(y, errors="coerce")
        
        if bands_to_plot is None:
            bands_to_plot = POWER_BANDS_TO_USE[:3]  # Limit to avoid clutter
        
        # Build ROIs
        roi_map = _build_rois(info)
        key_rois = ['Frontal', 'Central', 'Parietal', 'Occipital']  # Focus on key regions
        available_rois = [roi for roi in key_rois if roi in roi_map]
        
        if not available_rois:
            logger.warning(f"No key ROIs available for evolution analysis: sub-{subject}")
            return
        
        n_trials = len(y)
        if n_trials < window_size * 2:
            logger.warning(f"Not enough trials ({n_trials}) for evolution analysis (need at least {window_size * 2})")
            return
            
        trial_numbers = np.arange(1, n_trials + 1)
        
        # Create separate behavioral response evolution figure
        fig_behav, ax_behav = plt.subplots(1, 1, figsize=(10, 6))
        
        # Remove NaN values for smoothing
        y_clean = y.fillna(y.mean())
        y_smooth = gaussian_filter1d(y_clean, sigma=VIZ_SMOOTHING_SIGMA)
        
        ax_behav.scatter(trial_numbers, y, alpha=0.4, s=15, color='gray', label='Raw ratings')
        ax_behav.plot(trial_numbers, y_smooth, color='red', linewidth=3, 
                     label='Smoothed trend', alpha=0.8)
        ax_behav.set_xlabel('Trial Number', fontweight='bold')
        ax_behav.set_ylabel('Pain Rating', fontweight='bold')
        ax_behav.set_title(f'Behavioral Response Evolution - sub-{subject}', fontweight='bold', fontsize=12)
        ax_behav.legend(fontsize=10)
        ax_behav.grid(True, alpha=0.3)
        
        plt.tight_layout()
        _save_fig(fig_behav, plots_dir / f"behavioral_response_evolution")
        
        # Create separate figure for each band with 2 subplots
        for band in bands_to_plot:
            band_color = BAND_COLORS.get(band, 'blue')
            band_range = FEATURES_FREQ_BANDS.get(band)
            band_label = f"{band.title()} ({band_range[0]:g}–{band_range[1]:g} Hz)" if band_range else band.title()
            
            # Get ROI-averaged power for this band
            roi_power = {}
            for roi in available_rois:
                roi_cols = [f"pow_{band}_{ch}" for ch in roi_map[roi] 
                          if f"pow_{band}_{ch}" in pow_df.columns]
                if roi_cols:
                    roi_power[roi] = pow_df[roi_cols].apply(
                        pd.to_numeric, errors='coerce').mean(axis=1)
            
            if not roi_power:
                logger.warning(f"No power data available for {band} band")
                continue
            
            # Create figure with 2 subplots for this band
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            
            # Top subplot: Running correlations
            ax_corr = axes[0]
            mid_points = []
            correlations = {roi: [] for roi in roi_power.keys()}
            
            for start in range(n_trials - window_size + 1):
                end = start + window_size
                mid_points.append(start + window_size // 2 + 1)  # +1 for 1-based trial numbers
                y_window = y.iloc[start:end]
                
                for roi in roi_power.keys():
                    x_window = roi_power[roi].iloc[start:end]
                    mask = x_window.notna() & y_window.notna()
                    if mask.sum() >= 5:
                        r, _ = stats.spearmanr(x_window[mask], y_window[mask])
                        correlations[roi].append(r)
                    else:
                        correlations[roi].append(np.nan)
            
            # Plot running correlations with different colors for each ROI
            roi_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for roi_idx, roi in enumerate(roi_power.keys()):
                color = roi_colors[roi_idx % len(roi_colors)]
                ax_corr.plot(mid_points, correlations[roi], 
                            label=f'{roi}', linewidth=2.5, alpha=0.8, color=color)
            
            ax_corr.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax_corr.set_ylabel('Running Correlation (r)', fontweight='bold')
            ax_corr.set_title(f'{band_label}: Power-Behavior Correlation Evolution', fontweight='bold', fontsize=12)
            ax_corr.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
            ax_corr.grid(True, alpha=0.3)
            ax_corr.set_ylim(-1, 1)
            
            # Bottom subplot: Power trends for key ROI
            ax_power = axes[1]
            
            for roi_idx, roi in enumerate(roi_power.keys()):
                color = roi_colors[roi_idx % len(roi_colors)]
                power_clean = roi_power[roi].fillna(roi_power[roi].mean())
                power_smooth = gaussian_filter1d(power_clean, sigma=VIZ_SMOOTHING_SIGMA)
                ax_power.plot(trial_numbers, power_smooth, 
                             label=f'{roi}', linewidth=2.5, alpha=0.8, color=color)
            
            ax_power.set_xlabel('Trial Number', fontweight='bold')
            ax_power.set_ylabel('log10(power/baseline)', fontweight='bold')
            ax_power.set_title(f'{band_label} Power Evolution', fontweight='bold', fontsize=12)
            ax_power.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
            ax_power.grid(True, alpha=0.3)
            
            plt.suptitle(f'Power Evolution: {band_label} - sub-{subject}', 
                         fontsize=14, fontweight='bold', y=0.95)
            plt.tight_layout()
            _save_fig(fig, plots_dir / f"power_evolution_{band}")
        
        logger.info(f"Saved behavioral evolution figure and power evolution analysis for {len(bands_to_plot)} bands for sub-{subject}")
        
    except Exception as e:
        logger.error(f"Failed to create power-behavior evolution analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if 'fig' in locals():
            plt.close(fig)


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
        
        # 1.5. Topographical maps of significant correlations  
        plot_significant_correlations_topomap(pow_df, y, POWER_BANDS_TO_USE, info, subject, plots_dir, logger)
        
        # 2. Behavioral response patterns
        plot_behavioral_response_patterns(y, aligned_events, subject, plots_dir, logger)
        
        
        # 5. Publication-quality EEG power spectrogram with behavior overlay
        plot_power_spectrogram_with_behavior(pow_df, y, POWER_BANDS_TO_USE, subject, plots_dir, logger)
        
        # 5b. Temperature-based spectrograms
        plot_power_spectrogram_temperature_band(pow_df, aligned_events, POWER_BANDS_TO_USE, subject, plots_dir, logger)
        
        # 6. Topographic correlation maps (removed - MNE compatibility issues)
        # plot_topographic_correlation_maps(pow_df, y, POWER_BANDS_TO_USE, subject, plots_dir, logger)
        
        
        # 8. Band power summary - REMOVED
        # plot_band_power_summary(pow_df, POWER_BANDS_TO_USE, subject, plots_dir, logger)
        
        # 9. Neural-behavioral state space visualization - REMOVED
        # plot_neural_behavioral_state_space(subject, task, method='pca', n_components=2, rng=rng)
        
        # 10. Trial-by-trial power-behavior evolution analysis
        plot_power_behavior_evolution_across_trials(subject, task, window_size=20, bands_to_plot=None)
        
        # 11. Time-frequency correlation heatmap - NEW VISUALIZATION
        plot_time_frequency_correlation_heatmap(
            subject,
            task,
            use_spearman=use_spearman,
        )
        
        
        
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

    # Generate per-band channel-level power correlation stats
    try:
        correlate_power_topomaps(
            subject,
            task,
            use_spearman=use_spearman,
            partial_covars=partial_covars,
            bootstrap=bootstrap,
            n_perm=0,
            rng=rng,
        )
    except Exception as e:
        logger.error(f"Channel-level power correlations failed for sub-{subject}: {e}")

    # Combine per-band power correlation stats into consolidated TSV/CSV
    try:
        export_combined_power_corr_stats(subject)
    except Exception as e:
        logger.error(f"Combined power corr stats export failed for sub-{subject}: {e}")

    # Plot top behavioral predictors based on combined correlation stats
    try:
        plot_top_behavioral_predictors(subject, task)
    except Exception as e:
        logger.error(f"Top behavioral predictors plot failed for sub-{subject}: {e}")

    # Export all significant predictors to single CSV file
    try:
        export_all_significant_predictors(subject, alpha=0.05, use_fdr=True)
    except Exception as e:
        logger.error(f"All significant predictors export failed for sub-{subject}: {e}")

    # Apply a subject-level global FDR across all tests
    try:
        apply_global_fdr(subject)
    except Exception as e:
        logger.error(f"Global FDR application failed for sub-{subject}: {e}")

    # Report building removed per user request


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


def aggregate_group_level(
    subjects: Optional[List[str]] = None,
    task: str = TASK,
    *,
    pooling_strategy: str = "within_subject_centered",
    cluster_bootstrap: int = 0,
    subject_fixed_effects: bool = True,
) -> None:
    if subjects is None or subjects == ["all"]:
        subjects = SUBJECTS
    gstats = _group_stats_dir()
    gplots = _group_plots_dir()
    _ensure_dir(gstats)
    _ensure_dir(gplots)

    logger = _setup_group_logging()
    try:
        logger.info(f"Starting group aggregation for {len(subjects)} subjects (task={task})")
    except Exception:
        pass

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
            rej, crit = _fdr_bh(dfb["p_group"].to_numpy(), alpha=BEHAV_FDR_ALPHA)
            dfb["fdr_reject"] = rej
            dfb["fdr_crit_p"] = crit
            out_rows.append(dfb)
        dfg2 = pd.concat(out_rows, ignore_index=True)
        out_path_rating = gstats / "group_corr_pow_roi_vs_rating.tsv"
        dfg2.to_csv(out_path_rating, sep="\t", index=False)
        try:
            logger.info(f"Wrote group ROI power vs rating summary: {out_path_rating}")
        except Exception:
            pass

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
        rej, crit = _fdr_bh(out_df["p_group"].to_numpy(), alpha=BEHAV_FDR_ALPHA)
        out_df["fdr_reject"] = rej
        out_df["fdr_crit_p"] = crit
        out_conn = gstats / f"group_corr_conn_roi_summary_{_sanitize(pref)}_vs_rating.tsv"
        out_df.to_csv(out_conn, sep="\t", index=False)
        try:
            logger.info(f"Wrote group connectivity ROI summary: {out_conn}")
        except Exception:
            pass

    # Generate pooled scatter plots across subjects
    try:
        try:
            logger.info("Generating pooled ROI scatters across subjects…")
        except Exception:
            pass
        plot_group_power_roi_scatter(
            subjects,
            task=task,
            use_spearman=True,
            partial_covars=PARTIAL_COVARS_DEFAULT,
            do_temp=True,
            bootstrap_ci=0,
            rng=None,
            pooling_strategy=pooling_strategy,
            cluster_bootstrap=int(cluster_bootstrap),
            subject_fixed_effects=bool(subject_fixed_effects),
        )
    except Exception as e:
        logging.getLogger("behavior_analysis_group").error(
            f"Group scatter plotting failed: {e}"
        )

    # Group channel-level aggregation and visualizations
    try:
        _group_channel_level_visuals(subjects, task, logger)
    except Exception as e:
        logger.error(f"Group channel-level visuals failed: {e}")

    # Group overall multi-band summary figures (pooled overall power vs rating/temp)
    try:
        _group_overall_band_summary(
            subjects,
            task,
            logger=logger,
            use_spearman=True,
            pooling_strategy=pooling_strategy,
            cluster_bootstrap=int(cluster_bootstrap),
        )
    except Exception as e:
        logger.error(f"Group overall multi-band summary failed: {e}")

def _group_channel_level_visuals(subjects: List[str], task: str, logger: logging.Logger) -> None:
    """Aggregate channel-level correlations across subjects and create group plots.

    - Aggregates per-band channel-level r via Fisher z; p via t-test on Fisher z.
    - Saves per-band TSVs and a combined TSV in the group stats directory.
    - Plots per-band bar charts of r_group and a top-N predictors chart.
    - Plots group significant topomaps across bands using a representative subject's montage.
    """
    gstats = _group_stats_dir()
    gplots = _group_plots_dir()
    _ensure_dir(gstats)
    _ensure_dir(gplots)

    # Helper: load subject channel-level stats for a band
    def _load_sub_band_df(sub: str, band: str) -> Optional[pd.DataFrame]:
        f = _stats_dir(sub) / f"corr_stats_pow_{band}_vs_rating.tsv"
        if not f.exists():
            return None
        try:
            df = pd.read_csv(f, sep="\t")
        except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            return None
        if df is None or df.empty or "channel" not in df.columns or "r" not in df.columns:
            return None
        return df

    bands_to_df: Dict[str, pd.DataFrame] = {}
    bands_to_df_temp: Dict[str, pd.DataFrame] = {}
    for band in POWER_BANDS_TO_USE:
        chan_to_r: Dict[str, List[float]] = {}
        chan_to_r_temp: Dict[str, List[float]] = {}
        # Collect r per subject for each channel
        for sub in subjects:
            df = _load_sub_band_df(sub, band)
            if df is None:
                continue
            for _, row in df.iterrows():
                ch = str(row.get("channel"))
                try:
                    r = float(row.get("r"))
                except (TypeError, ValueError):
                    r = np.nan
                if np.isfinite(r):
                    chan_to_r.setdefault(ch, []).append(float(np.clip(r, -0.999999, 0.999999)))
        # Temperature per subject (optional)
        for sub in subjects:
            ftemp = _stats_dir(sub) / f"corr_stats_pow_{band}_vs_temp.tsv"
            if not ftemp.exists():
                continue
            try:
                df_t = pd.read_csv(ftemp, sep="\t")
            except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if df_t is None or df_t.empty or "channel" not in df_t.columns or "r" not in df_t.columns:
                continue
            for _, row in df_t.iterrows():
                ch = str(row.get("channel"))
                try:
                    r = float(row.get("r"))
                except (TypeError, ValueError):
                    r = np.nan
                if np.isfinite(r):
                    chan_to_r_temp.setdefault(ch, []).append(float(np.clip(r, -0.999999, 0.999999)))

        if not chan_to_r and not chan_to_r_temp:
            continue

        # Aggregate per channel via Fisher (rating)
        out_rows: List[Dict[str, object]] = []
        for ch, rs in sorted(chan_to_r.items()):
            vals = np.array(rs, dtype=float)
            vals = vals[np.isfinite(vals)]
            vals = np.clip(vals, -0.999999, 0.999999)
            if vals.size == 0:
                continue
            z = np.arctanh(vals)
            r_grp = float(np.tanh(np.mean(z)))
            # p via t-test on Fisher z against 0
            if len(z) >= 2:
                tstat, p = stats.ttest_1samp(z, popmean=0.0)
                p = float(p)
            else:
                p = np.nan
            # CI via t-critical on mean z
            if len(z) >= 2:
                sd = float(np.std(z, ddof=1))
                se = sd / np.sqrt(len(z)) if sd > 0 else np.nan
                if np.isfinite(se) and se > 0:
                    tcrit = float(stats.t.ppf(0.975, df=len(z) - 1))
                    lo = float(np.tanh(np.mean(z) - tcrit * se))
                    hi = float(np.tanh(np.mean(z) + tcrit * se))
                else:
                    lo = np.nan
                    hi = np.nan
            else:
                lo = np.nan
                hi = np.nan
            out_rows.append({
                "channel": ch,
                "band": band,
                "r_group": r_grp,
                "p_group": p,
                "r_ci_low": lo,
                "r_ci_high": hi,
                "n_subjects": int(len(z)),
            })

        df_band = pd.DataFrame(out_rows)
        if not df_band.empty:
            rej, crit = _fdr_bh(df_band["p_group"].to_numpy(), alpha=BEHAV_FDR_ALPHA)
            df_band["fdr_reject"] = rej
            df_band["fdr_crit_p"] = crit
            df_band = df_band.sort_values("channel").reset_index(drop=True)
            df_band.to_csv(gstats / f"group_corr_pow_{_sanitize(band)}_vs_rating.tsv", sep="\t", index=False)
            bands_to_df[band] = df_band

        # Aggregate per channel via Fisher (temperature)
        out_rows_t: List[Dict[str, object]] = []
        for ch, rs in sorted(chan_to_r_temp.items()):
            vals = np.array(rs, dtype=float)
            vals = vals[np.isfinite(vals)]
            vals = np.clip(vals, -0.999999, 0.999999)
            if vals.size == 0:
                continue
            z = np.arctanh(vals)
            r_grp = float(np.tanh(np.mean(z)))
            if len(z) >= 2:
                tstat, p = stats.ttest_1samp(z, popmean=0.0)
                p = float(p)
            else:
                p = np.nan
            if len(z) >= 2:
                sd = float(np.std(z, ddof=1))
                se = sd / np.sqrt(len(z)) if sd > 0 else np.nan
                if np.isfinite(se) and se > 0:
                    tcrit = float(stats.t.ppf(0.975, df=len(z) - 1))
                    lo = float(np.tanh(np.mean(z) - tcrit * se))
                    hi = float(np.tanh(np.mean(z) + tcrit * se))
                else:
                    lo = np.nan
                    hi = np.nan
            else:
                lo = np.nan
                hi = np.nan
            out_rows_t.append({
                "channel": ch,
                "band": band,
                "r_group": r_grp,
                "p_group": p,
                "r_ci_low": lo,
                "r_ci_high": hi,
                "n_subjects": int(len(z)),
            })

        df_band_t = pd.DataFrame(out_rows_t)
        if not df_band_t.empty:
            rej_t, crit_t = _fdr_bh(df_band_t["p_group"].to_numpy(), alpha=BEHAV_FDR_ALPHA)
            df_band_t["fdr_reject"] = rej_t
            df_band_t["fdr_crit_p"] = crit_t
            df_band_t = df_band_t.sort_values("channel").reset_index(drop=True)
            df_band_t.to_csv(gstats / f"group_corr_pow_{_sanitize(band)}_vs_temp.tsv", sep="\t", index=False)
            bands_to_df_temp[band] = df_band_t

    if not bands_to_df and not bands_to_df_temp:
        logger.warning("No group channel-level stats aggregated (missing subject inputs?)")
        return

    # Combined across bands for top predictors plot
    if bands_to_df:
        combined = pd.concat([bands_to_df[b] for b in bands_to_df.keys()], ignore_index=True)
        combined.to_csv(gstats / "group_corr_pow_combined_vs_rating.tsv", sep="\t", index=False)
    else:
        combined = pd.DataFrame()

    if bands_to_df_temp:
        combined_t = pd.concat([bands_to_df_temp[b] for b in bands_to_df_temp.keys()], ignore_index=True)
        combined_t.to_csv(gstats / "group_corr_pow_combined_vs_temp.tsv", sep="\t", index=False)
    else:
        combined_t = pd.DataFrame()

    # Per-band bar plots (group_power_behavior_correlation_{band})
    for band, dfb in bands_to_df.items():
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            xs = np.arange(len(dfb))
            colors = ["red" if (np.isfinite(p) and p < BEHAV_FDR_ALPHA) else "lightblue" for p in dfb["p_group"]]
            ax.bar(xs, dfb["r_group"], color=colors)
            ax.set_xlabel("Channel", fontweight="bold")
            ax.set_ylabel("Spearman ρ", fontweight="bold")
            ax.set_title(f"{band.upper()} Band - Channel-wise Correlations with Behavior\nGroup", fontweight="bold", fontsize=14)
            ax.set_xticks(xs)
            ax.set_xticklabels(dfb["channel"].tolist(), rotation=45, ha="right")
            ax.grid(True, alpha=0.3, axis="y")
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
            # Moderate threshold lines (visual aid)
            ax.axhline(y=0.3, color="green", linestyle="--", alpha=0.7)
            ax.axhline(y=-0.3, color="green", linestyle="--", alpha=0.7)
            sig_count = int((dfb["p_group"] < BEHAV_FDR_ALPHA).sum())
            ax.text(0.02, 0.98, f"Significant channels: {sig_count}/{len(dfb)}",
                    transform=ax.transAxes, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            plt.tight_layout()
            _save_fig(fig, gplots / f"group_power_behavior_correlation_{_sanitize(band)}")
        except Exception as e:
            logger.warning(f"Failed group bar plot for {band}: {e}")

    # Per-band bar plots for temperature (group_power_temperature_correlation_{band})
    for band, dfb in bands_to_df_temp.items():
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            xs = np.arange(len(dfb))
            colors = ["red" if (np.isfinite(p) and p < BEHAV_FDR_ALPHA) else "lightblue" for p in dfb["p_group"]]
            ax.bar(xs, dfb["r_group"], color=colors)
            ax.set_xlabel("Channel", fontweight="bold")
            ax.set_ylabel("Spearman ρ", fontweight="bold")
            ax.set_title(f"{band.upper()} Band - Channel-wise Correlations with Temperature\nGroup", fontweight="bold", fontsize=14)
            ax.set_xticks(xs)
            ax.set_xticklabels(dfb["channel"].tolist(), rotation=45, ha="right")
            ax.grid(True, alpha=0.3, axis="y")
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
            ax.axhline(y=0.3, color="green", linestyle="--", alpha=0.7)
            ax.axhline(y=-0.3, color="green", linestyle="--", alpha=0.7)
            sig_count = int((dfb["p_group"] < BEHAV_FDR_ALPHA).sum())
            ax.text(0.02, 0.98, f"Significant channels: {sig_count}/{len(dfb)}",
                    transform=ax.transAxes, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            plt.tight_layout()
            _save_fig(fig, gplots / f"group_power_temperature_correlation_{_sanitize(band)}")
        except Exception as e:
            logger.warning(f"Failed group temperature bar plot for {band}: {e}")

    # Group top-N behavioral predictors
    try:
        top_n = int(config.get("behavior_analysis.predictors.top_n", 20))
        df_sig = combined[(~combined.empty) & (combined["p_group"] <= BEHAV_FDR_ALPHA) & combined["r_group"].notna()].copy() if not combined.empty else pd.DataFrame()
        if len(df_sig) > 0:
            df_sig["abs_r"] = df_sig["r_group"].abs()
            df_top = df_sig.nlargest(top_n, "abs_r").copy()
            df_top = df_top.sort_values("abs_r", ascending=True)

            fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
            labels = [f"{ch} ({band})" for ch, band in zip(df_top["channel"], df_top["band"])]
            y_pos = np.arange(len(df_top))
            colors = [_get_band_color(b) for b in df_top["band"]]
            ax.barh(y_pos, df_top["abs_r"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=11)
            ax.set_xlabel(f"|Spearman ρ| (p < {BEHAV_FDR_ALPHA})", fontweight='bold', fontsize=12)
            ax.set_title(f"Top {top_n} Significant Behavioral Predictors — Group", fontweight='bold', fontsize=14, pad=20)
            for i, (_, row) in enumerate(df_top.iterrows()):
                ax.text(row["abs_r"] + 0.01, i, f"{row['abs_r']:.3f} (p={row['p_group']:.3f})", va='center', ha='left', fontsize=10)
            ax.set_xlim(0, float(df_top["abs_r"].max()) * 1.25)
            ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            plt.tight_layout()
            _save_fig(fig, gplots / f"group_top_{top_n}_behavioral_predictors")
            # Export TSV
            df_top_export = df_top[["channel", "band", "r_group", "p_group", "n_subjects", "abs_r"]].sort_values("abs_r", ascending=False)
            df_top_export.to_csv(gstats / f"group_top_{top_n}_behavioral_predictors.tsv", sep="\t", index=False)
        else:
            logger.info("No significant group-level predictors to plot for top-N.")
    except Exception as e:
        logger.warning(f"Top-N group predictors plotting failed: {e}")

    # Group top-N temperature predictors
    try:
        top_n = int(config.get("behavior_analysis.predictors.top_n", 20))
        df_sig_t = combined_t[(~combined_t.empty) & (combined_t["p_group"] <= BEHAV_FDR_ALPHA) & combined_t["r_group"].notna()].copy() if not combined_t.empty else pd.DataFrame()
        if len(df_sig_t) > 0:
            df_sig_t["abs_r"] = df_sig_t["r_group"].abs()
            df_top_t = df_sig_t.nlargest(top_n, "abs_r").copy()
            df_top_t = df_top_t.sort_values("abs_r", ascending=True)
            fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
            labels = [f"{ch} ({band})" for ch, band in zip(df_top_t["channel"], df_top_t["band"])]
            y_pos = np.arange(len(df_top_t))
            colors = [_get_band_color(b) for b in df_top_t["band"]]
            ax.barh(y_pos, df_top_t["abs_r"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=11)
            ax.set_xlabel(f"|Spearman ρ| (p < {BEHAV_FDR_ALPHA})", fontweight='bold', fontsize=12)
            ax.set_title(f"Top {top_n} Significant Temperature Predictors — Group", fontweight='bold', fontsize=14, pad=20)
            for i, (_, row) in enumerate(df_top_t.iterrows()):
                ax.text(row["abs_r"] + 0.01, i, f"{row['abs_r']:.3f} (p={row['p_group']:.3f})", va='center', ha='left', fontsize=10)
            ax.set_xlim(0, float(df_top_t["abs_r"].max()) * 1.25)
            ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            plt.tight_layout()
            _save_fig(fig, gplots / f"group_top_{top_n}_temperature_predictors")
            df_top_t[["channel", "band", "r_group", "p_group", "n_subjects", "abs_r"]].sort_values("abs_r", ascending=False).to_csv(
                gstats / f"group_top_{top_n}_temperature_predictors.tsv", sep="\t", index=False
            )
        else:
            logger.info("No significant group-level temperature predictors to plot for top-N.")
    except Exception as e:
        logger.warning(f"Top-N temperature predictors plotting failed: {e}")

    # Group significant correlations topomap (rating)
    try:
        # Choose representative info from the first subject with epochs
        info = None
        for sub in subjects:
            epo_path = _find_clean_epochs_path(sub, task)
            if epo_path is not None and Path(epo_path).exists():
                info = mne.read_epochs(epo_path, preload=False, verbose=False).info
                break
        if info is None:
            logger.warning("No epochs found to build group topomap; skipping.")
            return

        # Prepare band data for topomap
        bands_with_data = []
        for band, dfb in bands_to_df.items():
            chs = dfb["channel"].astype(str).tolist()
            corrs = dfb["r_group"].to_numpy()
            pvals = dfb["p_group"].to_numpy()
            sig_mask = np.isfinite(pvals) & (pvals < BEHAV_FDR_ALPHA)
            bands_with_data.append({
                "band": band,
                "channels": chs,
                "correlations": corrs,
                "p_values": pvals,
                "significant_mask": sig_mask,
            })
        if not bands_with_data:
            return
        n_bands = len(bands_with_data)
        fig, axes = plt.subplots(1, n_bands, figsize=(4.8 * n_bands, 4.8))
        if n_bands == 1:
            axes = [axes]
        plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.16, wspace=0.08)

        # Determine vlim
        all_sig = []
        for bd in bands_with_data:
            all_sig.extend(bd["correlations"][bd["significant_mask"]])
        finite_sig = np.array(all_sig, dtype=float)
        finite_sig = finite_sig[np.isfinite(finite_sig)]
        if finite_sig.size > 0:
            vmax = float(np.max(np.abs(finite_sig)))
        else:
            all_vals = []
            for bd in bands_with_data:
                all_vals.extend(bd["correlations"][np.isfinite(bd["correlations"])])
            vmax = float(np.max(np.abs(all_vals))) if all_vals else 0.5

        successful = []
        for i, bd in enumerate(bands_with_data):
            ax = axes[i]
            try:
                picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
                if len(picks) == 0:
                    raise ValueError("No EEG channels found in info")
                n_info_chs = len(info['ch_names'])
                topo_data = np.zeros(n_info_chs)
                topo_mask = np.zeros(n_info_chs, dtype=bool)
                for j, ch in enumerate(info['ch_names']):
                    if ch in bd['channels']:
                        idx = bd['channels'].index(ch)
                        val = bd['correlations'][idx]
                        topo_data[j] = val if np.isfinite(val) else 0.0
                        topo_mask[j] = bool(bd['significant_mask'][idx])
                im, _ = mne.viz.plot_topomap(
                    topo_data[picks],
                    mne.pick_info(info, picks),
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vlim=(-vmax, vmax),
                    contours=6,
                    mask=topo_mask[picks],
                    mask_params=dict(marker='o', markerfacecolor='white', markeredgecolor='black', linewidth=1, markersize=6)
                )
                successful.append(im)
                n_sig = int(topo_mask[picks].sum())
                n_total = int(np.sum([1 for ch in bd['channels'] if ch in info['ch_names']]))
                ax.set_title(f"{bd['band'].upper()}\n{n_sig}/{n_total} significant", fontweight='bold', fontsize=12, pad=10)
            except Exception as e:
                logger.warning(f"Group topomap failed for band {bd['band']}: {e}")
                ax.text(0.5, 0.5, f"{bd['band'].upper()}\nPlot failed", ha='center', va='center', transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.set_title(f"{bd['band'].upper()}\n(Error)", fontweight='bold', pad=10)

        plt.suptitle(f"Group Significant EEG-Pain Correlations (p < {BEHAV_FDR_ALPHA})", fontweight='bold', fontsize=14, y=0.97)
        if successful:
            left = min(ax.get_position().x0 for ax in axes)
            right = max(ax.get_position().x1 for ax in axes)
            bottom = min(ax.get_position().y0 for ax in axes)
            span = right - left
            cb_width = 0.55 * span
            cb_left = left + 0.225 * span
            cb_bottom = max(0.04, bottom - 0.06)
            cax = fig.add_axes([cb_left, cb_bottom, cb_width, 0.028])
            cbar = fig.colorbar(successful[-1], cax=cax, orientation='horizontal')
            cbar.set_label('Correlation (ρ)', fontweight='bold', fontsize=11)
            cbar.ax.tick_params(pad=2, labelsize=9)
        _save_fig(fig, gplots / "group_significant_correlations_topomap")
    except Exception as e:
        logger.warning(f"Group significant topomap plotting failed: {e}")

    # Group significant correlations topomap (temperature)
    try:
        if not bands_to_df_temp:
            return
        info = None
        for sub in subjects:
            epo_path = _find_clean_epochs_path(sub, task)
            if epo_path is not None and Path(epo_path).exists():
                info = mne.read_epochs(epo_path, preload=False, verbose=False).info
                break
        if info is None:
            return
        bands_with_data = []
        for band, dfb in bands_to_df_temp.items():
            chs = dfb["channel"].astype(str).tolist()
            corrs = dfb["r_group"].to_numpy()
            pvals = dfb["p_group"].to_numpy()
            sig_mask = np.isfinite(pvals) & (pvals < BEHAV_FDR_ALPHA)
            bands_with_data.append({
                "band": band,
                "channels": chs,
                "correlations": corrs,
                "p_values": pvals,
                "significant_mask": sig_mask,
            })
        n_bands = len(bands_with_data)
        fig, axes = plt.subplots(1, n_bands, figsize=(4.8 * n_bands, 4.8))
        if n_bands == 1:
            axes = [axes]
        plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.16, wspace=0.08)
        all_sig = []
        for bd in bands_with_data:
            all_sig.extend(bd["correlations"][bd["significant_mask"]])
        finite_sig = np.array(all_sig, dtype=float)
        finite_sig = finite_sig[np.isfinite(finite_sig)]
        if finite_sig.size > 0:
            vmax = float(np.max(np.abs(finite_sig)))
        else:
            all_vals = []
            for bd in bands_with_data:
                all_vals.extend(bd["correlations"][np.isfinite(bd["correlations"])])
            vmax = float(np.max(np.abs(all_vals))) if all_vals else 0.5
        successful = []
        for i, bd in enumerate(bands_with_data):
            ax = axes[i]
            try:
                picks = mne.pick_types(info, meg=False, eeg=True, exclude='bads')
                if len(picks) == 0:
                    raise ValueError("No EEG channels found in info")
                n_info_chs = len(info['ch_names'])
                topo_data = np.zeros(n_info_chs)
                topo_mask = np.zeros(n_info_chs, dtype=bool)
                for j, ch in enumerate(info['ch_names']):
                    if ch in bd['channels']:
                        idx = bd['channels'].index(ch)
                        val = bd['correlations'][idx]
                        topo_data[j] = val if np.isfinite(val) else 0.0
                        topo_mask[j] = bool(bd['significant_mask'][idx])
                im, _ = mne.viz.plot_topomap(
                    topo_data[picks],
                    mne.pick_info(info, picks),
                    axes=ax,
                    show=False,
                    cmap='RdBu_r',
                    vlim=(-vmax, vmax),
                    contours=6,
                    mask=topo_mask[picks],
                    mask_params=dict(marker='o', markerfacecolor='white', markeredgecolor='black', linewidth=1, markersize=6)
                )
                successful.append(im)
                n_sig = int(topo_mask[picks].sum())
                n_total = int(np.sum([1 for ch in bd['channels'] if ch in info['ch_names']]))
                ax.set_title(f"{bd['band'].upper()}\n{n_sig}/{n_total} significant", fontweight='bold', fontsize=12, pad=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"{bd['band'].upper()}\nPlot failed", ha='center', va='center', transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                ax.set_title(f"{bd['band'].upper()}\n(Error)", fontweight='bold', pad=10)
        plt.suptitle(f"Group Significant EEG-Temperature Correlations (p < {BEHAV_FDR_ALPHA})", fontweight='bold', fontsize=14, y=0.97)
        if successful:
            left = min(ax.get_position().x0 for ax in axes)
            right = max(ax.get_position().x1 for ax in axes)
            bottom = min(ax.get_position().y0 for ax in axes)
            span = right - left
            cb_width = 0.55 * span
            cb_left = left + 0.225 * span
            cb_bottom = max(0.04, bottom - 0.06)
            cax = fig.add_axes([cb_left, cb_bottom, cb_width, 0.028])
            cbar = fig.colorbar(successful[-1], cax=cax, orientation='horizontal')
            cbar.set_label('Correlation (ρ)', fontweight='bold', fontsize=11)
            cbar.ax.tick_params(pad=2, labelsize=9)
        _save_fig(fig, gplots / "group_significant_correlations_topomap_temperature")
    except Exception:
        pass


def _group_overall_band_summary(
    subjects: List[str],
    task: str,
    logger: logging.Logger,
    use_spearman: bool,
    pooling_strategy: str,
    cluster_bootstrap: int,
) -> None:
    """Create group scatter summaries (overall power vs rating and temperature).

    For each band, pools trials across subjects using the requested strategy and shows
    a scatter with regression line. Saves one figure per band for each target:
    - group_power_behavior_correlation_{band}
    - group_power_temperature_correlation_{band} (when temperature is available)
    """
    # Build pooled 'overall' values (mean across channels per band) per subject
    rating_x: Dict[str, List[np.ndarray]] = {}
    rating_y: Dict[str, List[np.ndarray]] = {}
    temp_x: Dict[str, List[np.ndarray]] = {}
    temp_y: Dict[str, List[np.ndarray]] = {}
    have_temp = False

    for sub in subjects:
        try:
            pow_df, _conn_df, y, info = _load_features_and_targets(sub, task)
            y = pd.to_numeric(y, errors="coerce")
            epo_path = _find_clean_epochs_path(sub, task)
            if epo_path is None:
                continue
            epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
            events = _load_events_df(sub, task)
            aligned = _align_events_to_epochs(events, epochs) if events is not None else None
        except Exception:
            continue
        ts = None
        if aligned is not None:
            tcol = _pick_first_column(aligned, PSYCH_TEMP_COLUMNS)
            if tcol is not None:
                ts = pd.to_numeric(aligned[tcol], errors="coerce")
                have_temp = True
        for band in POWER_BANDS_TO_USE:
            cols = [c for c in pow_df.columns if c.startswith(f"pow_{band}_")]
            if not cols:
                continue
            vals = pow_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
            rating_x.setdefault(band, []).append(vals)
            rating_y.setdefault(band, []).append(y.to_numpy())
            if ts is not None:
                temp_x.setdefault(band, []).append(vals)
                temp_y.setdefault(band, []).append(ts.to_numpy())

    def _pool_xy(x_lists, y_lists, strategy: str):
        vis_x = []
        vis_y = []
        for xi, yi in zip(x_lists, y_lists):
            xi = pd.Series(xi)
            yi = pd.Series(yi)
            n = min(len(xi), len(yi))
            xi = xi.iloc[:n]
            yi = yi.iloc[:n]
            m = xi.notna() & yi.notna()
            xi = xi[m]
            yi = yi[m]
            if strategy == "within_subject_centered":
                xi = xi - xi.mean()
                yi = yi - yi.mean()
            elif strategy == "within_subject_zscored":
                sx = xi.std(ddof=1)
                sy = yi.std(ddof=1)
                if sx <= 0 or sy <= 0:
                    continue
                xi = (xi - xi.mean()) / sx
                yi = (yi - yi.mean()) / sy
            elif strategy == "fisher_by_subject":
                xi = xi - xi.mean()
                yi = yi - yi.mean()
            vis_x.append(xi)
            vis_y.append(yi)
        X = pd.concat(vis_x, ignore_index=True) if vis_x else pd.Series(dtype=float)
        Y = pd.concat(vis_y, ignore_index=True) if vis_y else pd.Series(dtype=float)
        return X, Y

    # Rating summary figures: one scatter per band per figure
    n_bands = len(POWER_BANDS_TO_USE)
    if n_bands > 0:
        for band in POWER_BANDS_TO_USE:
            x_lists = rating_x.get(band, [])
            y_lists = rating_y.get(band, [])
            if not x_lists or not y_lists:
                continue
            X, Y = _pool_xy(x_lists, y_lists, pooling_strategy)
            if len(X) < 5:
                continue
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            sns.regplot(
                x=X,
                y=Y,
                ax=ax,
                ci=95,
                scatter_kws={
                    "s": 25,
                    "alpha": 0.7,
                    "color": _get_band_color(band),
                    "edgecolor": "white",
                    "linewidths": 0.3,
                },
                line_kws={"color": "#666666", "lw": 1.5},
            )
            ax.set_xlabel(f"{band.capitalize()} Power\nlog10(power/baseline)")
            ax.set_ylabel("Rating")
            ax.set_title(f"{band.capitalize()} vs Rating")
            try:
                r, p = stats.spearmanr(X, Y, nan_policy="omit")
                ax.text(
                    0.02,
                    0.98,
                    f"Spearman ρ={r:.3f}\np={p:.3f}\nn={len(X)}",
                    transform=ax.transAxes,
                    va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                )
            except Exception:
                pass
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            _save_fig(fig, _group_plots_dir() / f"group_power_behavior_correlation_{_sanitize(band)}")
            plt.close(fig)

    # Temperature summary figures: one scatter per band per figure
    if have_temp and len(temp_x) > 0:
        for band in POWER_BANDS_TO_USE:
            x_lists = temp_x.get(band, [])
            y_lists = temp_y.get(band, [])
            if not x_lists or not y_lists:
                continue
            X, Y = _pool_xy(x_lists, y_lists, pooling_strategy)
            if len(X) < 5:
                continue
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            sns.regplot(
                x=X,
                y=Y,
                ax=ax,
                ci=95,
                scatter_kws={
                    "s": 25,
                    "alpha": 0.7,
                    "color": _get_band_color(band),
                    "edgecolor": "white",
                    "linewidths": 0.3,
                },
                line_kws={"color": "#666666", "lw": 1.5},
            )
            ax.set_xlabel(f"{band.capitalize()} Power\nlog10(power/baseline)")
            # Clarify y-axis based on pooling strategy
            if pooling_strategy == "within_subject_centered":
                ax.set_ylabel("Temperature (°C, centered)")
            elif pooling_strategy == "within_subject_zscored":
                ax.set_ylabel("Temperature (z-scored)")
            else:
                ax.set_ylabel("Temperature (°C)")
            ax.set_title(f"{band.capitalize()} vs Temp")
            try:
                r, p = stats.spearmanr(X, Y, nan_policy="omit")
                ax.text(
                    0.02,
                    0.98,
                    f"Spearman ρ={r:.3f}\np={p:.3f}\nn={len(X)}",
                    transform=ax.transAxes,
                    va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                )
            except Exception:
                pass
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            _save_fig(fig, _group_plots_dir() / f"group_power_temperature_correlation_{_sanitize(band)}")
            plt.close(fig)

    # Group channel-level aggregation and visualizations
    try:
        _group_channel_level_visuals(subjects, task, logger)
    except Exception as e:
        logger.error(f"Group channel-level visuals failed: {e}")


def _collect_subject_ids_with_features(root: Path) -> List[str]:
    subs: List[str] = []
    for sub_dir in sorted(root.glob("sub-*/eeg/features")):
        feat_ok = (sub_dir / "features_eeg_direct.tsv").exists()
        tgt_ok = (sub_dir / "target_vas_ratings.tsv").exists()
        if feat_ok and tgt_ok:
            sid = sub_dir.parent.parent.name.replace("sub-", "")
            subs.append(sid)
    return subs


def main(
    subjects: Optional[List[str]] = None,
    task: str = TASK,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
    do_group: bool = False,
    group_only: bool = False,
    rng_seed: int = 42,
    all_subjects: bool = False,
    pooling_strategy: str = "within_subject_centered",
    cluster_bootstrap: int = 0,
    group_subject_fixed_effects: bool = True,
):
    # Enforce CLI-provided subjects; allow --all-subjects to auto-detect from derivatives
    if all_subjects:
        subjects = _collect_subject_ids_with_features(Path(DERIV_ROOT))
        if not subjects:
            raise ValueError(f"No subjects with features found in {DERIV_ROOT}")
    elif subjects is None or len(subjects) == 0:
        raise ValueError("No subjects specified. Use --group all|A,B,C, or --subject (can repeat), or --all-subjects.")
    if not group_only:
        for sub in subjects:
            process_subject(
                sub,
                task,
                use_spearman=use_spearman,
                partial_covars=partial_covars,
                bootstrap=bootstrap,
                n_perm=n_perm,
                rng_seed=rng_seed,
            )
    if do_group or group_only:
        aggregate_group_level(
            subjects,
            task,
            pooling_strategy=pooling_strategy,
            cluster_bootstrap=int(cluster_bootstrap),
            subject_fixed_effects=bool(group_subject_fixed_effects),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Behavioral psychometrics and EEG feature correlations (single or multiple subjects)")

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
        help="Process all available subjects with features and targets",
    )

    # Deprecated alias (kept for backward compatibility). Used only if other selectors absent.
    parser.add_argument(
        "--subjects", nargs="*", default=None,
        help="[Deprecated] Subject IDs list. Prefer --subject repeated or --group."
    )

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
    parser.add_argument(
        "--pooling-strategy",
        choices=["pooled", "centered", "zscored", "fisher"],
        default="centered",
        help=(
            "Group pooling strategy for pooled scatters: "
            "'pooled' (concatenate trials), 'centered' (within-subject mean-center then pool), "
            "'zscored' (within-subject z-score then pool), or 'fisher' (aggregate subject-wise r via Fisher z)."
        ),
    )
    parser.add_argument(
        "--cluster-bootstrap",
        type=int,
        default=BEHAV_BOOTSTRAP_N,
        help=(
            "Number of subject-level bootstrap resamples for pooled group r (0 disables)."
        ),
    )
    parser.add_argument(
        "--no-group-subject-fixed-effects",
        action="store_false",
        dest="group_subject_fixed_effects",
        help="Disable subject fixed-effects (dummies) in group-level partial residuals",
    )

    args = parser.parse_args()

    # Resolve subjects according to 01/02 pattern
    subjects: Optional[List[str]] = None
    if args.group is not None:
        g = args.group.strip()
        if g.lower() in {"all", "*", "@all"}:
            subjects = _collect_subject_ids_with_features(Path(DERIV_ROOT))
        else:
            cand = [s.strip() for s in g.replace(";", ",").replace(" ", ",").split(",") if s.strip()]
            subjects = []
            for s in cand:
                feats = _features_dir(s)
                if (feats / "features_eeg_direct.tsv").exists() and (feats / "target_vas_ratings.tsv").exists():
                    subjects.append(s)
                else:
                    print(f"Warning: --group subject '{s}' has no features/targets; skipping")
    elif args.all_subjects:
        subjects = _collect_subject_ids_with_features(Path(DERIV_ROOT))
    elif args.subject:
        # De-duplicate while preserving order
        _seen = set()
        subjects = []
        for s in args.subject:
            if s not in _seen:
                _seen.add(s)
                subjects.append(s)
    elif args.subjects:
        # Deprecated alias fallback
        subjects = list(dict.fromkeys(args.subjects))

    if subjects is None or len(subjects) == 0:
        print("No subjects provided. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
        sys.exit(2)

    # Auto-enable group aggregation when multiple subjects provided (mirrors 02)
    if len(subjects) == 1:
        main(
            subjects,
            task=args.task,
            use_spearman=True,
            partial_covars=PARTIAL_COVARS_DEFAULT,
            bootstrap=args.bootstrap,
            n_perm=N_PERM_DEFAULT,
            do_group=False,
            group_only=False,
            rng_seed=args.rng_seed,
            all_subjects=False,
            pooling_strategy=(
                "within_subject_centered" if args.pooling_strategy == "centered" else (
                    "within_subject_zscored" if args.pooling_strategy == "zscored" else (
                        "pooled_trials" if args.pooling_strategy == "pooled" else "fisher_by_subject"
                    )
                )
            ),
            cluster_bootstrap=int(args.cluster_bootstrap),
            group_subject_fixed_effects=bool(getattr(args, "group_subject_fixed_effects", True)),
        )
    else:
        main(
            subjects,
            task=args.task,
            use_spearman=True,
            partial_covars=PARTIAL_COVARS_DEFAULT,
            bootstrap=args.bootstrap,
            n_perm=N_PERM_DEFAULT,
            do_group=True,
            group_only=GROUP_ONLY_DEFAULT,
            rng_seed=args.rng_seed,
            all_subjects=False,
            pooling_strategy=(
                "within_subject_centered" if args.pooling_strategy == "centered" else (
                    "within_subject_zscored" if args.pooling_strategy == "zscored" else (
                        "pooled_trials" if args.pooling_strategy == "pooled" else "fisher_by_subject"
                    )
                )
            ),
            cluster_bootstrap=int(args.cluster_bootstrap),
            group_subject_fixed_effects=bool(getattr(args, "group_subject_fixed_effects", True)),
        )
