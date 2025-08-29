from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import itertools

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import mne
from mne_bids import BIDSPath

# -----------------------------------------------------------------------------
# Resolve project config (paths, subjects, task)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "eeg_pipeline"))
try:
    import config as cfg  # type: ignore
except Exception:
    class _Fallback:
        project_root = PROJECT_ROOT
        bids_root = str(PROJECT_ROOT / "eeg_pipeline" / "bids_output")
        deriv_root = str((PROJECT_ROOT / "eeg_pipeline" / "bids_output" / "derivatives"))
        task = "thermalactive"
        subjects = ["001"]
        # freq bands used in features (align with 03_feature_engineering)
        features_freq_bands = {
            "theta": (4.0, 8.0 - 0.1),
            "alpha": (8.0, 13.0 - 0.1),
            "beta": (13.0, 30.0),
            "gamma": (30.0 + 0.1, 80.0),
        }
    cfg = _Fallback()  # type: ignore

BIDS_ROOT = Path(getattr(cfg, "bids_root", PROJECT_ROOT / "eeg_pipeline" / "bids_output"))
DERIV_ROOT = Path(getattr(cfg, "deriv_root", BIDS_ROOT / "derivatives"))
TASK = getattr(cfg, "task", "thermalactive")
SUBJECTS = getattr(cfg, "subjects", ["001"])  # may be updated in CLI
FEATURES_FREQ_BANDS = getattr(cfg, "features_freq_bands", {
    "theta": (4.0, 8.0 - 0.1),
    "alpha": (8.0, 13.0 - 0.1),
    "beta": (13.0, 30.0),
    "gamma": (30.0 + 0.1, 80.0),
})

# -----------------------------------------------------------------------------
# Analysis parameters
# -----------------------------------------------------------------------------
PSYCH_TEMP_COLUMNS = ["stimulus_temp", "stimulus_temperature", "temp", "temperature", "temp_level"]
RATING_COLUMNS = [
    "vas_final_coded_rating", "vas_final_rating", "vas_rating",
    "pain_intensity", "pain_rating", "rating"
]
PAIN_BINARY_COLUMNS = ["pain_binary_coded", "pain_binary", "pain"]
POWER_BANDS_TO_USE = ["alpha", "beta", "gamma"]  # match 03_feature_engineering POWER_BANDS
PLATEAU_WINDOW = (3.0, 10.5)  # seconds, for doc only (power features already reflect this window)
FIG_DPI = 300
SAVE_FORMATS = ("png", "svg")
COLORBAR_FRACTION = 0.03
COLORBAR_PAD = 0.02
TOPO_CMAP = "RdBu_r"
TOPO_CONTOURS = 6

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
    per-trial features computed from the same epochs object.
    """
    if events_df is None:
        return None
    # 1) Use epochs.selection as row indices if available
    sel = getattr(epochs, "selection", None)
    if sel is not None and len(sel) == len(epochs):
        try:
            if len(events_df) > int(np.max(sel)):
                return events_df.iloc[sel].reset_index(drop=True)
        except Exception:
            pass
    # 2) Use sample column to reindex to epochs.events
    if "sample" in events_df.columns and isinstance(getattr(epochs, "events", None), np.ndarray):
        try:
            samples = epochs.events[:, 0]
            out = events_df.set_index("sample").reindex(samples)
            if len(out) == len(epochs) and not out.isna().all(axis=1).any():
                return out.reset_index()
        except Exception:
            pass
    # 3) Fallback: naive trim to min length
    n = min(len(events_df), len(epochs))
    if n == 0:
        return None
    return events_df.iloc[:n].reset_index(drop=True)


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
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots"


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
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    events = _load_events_df(subject, task)
    if events is None or len(events) == 0:
        print(f"No events for psychometrics: sub-{subject}")
        return

    temp_col = _pick_first_column(events, PSYCH_TEMP_COLUMNS)
    rating_col = _pick_first_column(events, RATING_COLUMNS)
    pain_col = _pick_first_column(events, PAIN_BINARY_COLUMNS)

    if temp_col is None:
        print(f"Psychometrics: no temperature column found; skipping for sub-{subject}.")
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

    # Plot binary pain psychometric if available
    if pain_col is not None:
        pain = pd.to_numeric(events[pain_col], errors="coerce")
        mask = temp.notna() & pain.isin([0, 1])
        if mask.sum() >= 5 and pain[mask].nunique() > 1:
            t = temp[mask]
            y = pain[mask]
            # Bin by temperature and compute P(pain)
            try:
                n_bins = min(8, max(4, int(np.sqrt(len(t)))))
                bins = np.linspace(t.min(), t.max(), n_bins + 1)
                cats = pd.cut(t, bins=bins, include_lowest=True)
                df = pd.DataFrame({"temp": t, "pain": y, "bin": cats})
                grp = df.groupby("bin", observed=False)
                x_bin = grp["temp"].mean()
                p_pain = grp["pain"].mean()
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                ax.plot(x_bin, p_pain, "o-", color="crimson")
                ax.set_xlabel(f"Temperature ({temp_col})")
                ax.set_ylabel("P(pain)")
                ax.set_ylim(-0.02, 1.02)
                ax.set_title("Psychometric curve: pain vs temperature")
                _save_fig(fig, plots_dir / "psychometric_pain_vs_temp")
            except Exception as e:
                print(f"Psychometric binning failed: {e}")


# -----------------------------------------------------------------------------
# Scatter: ROI-averaged power vs rating
# -----------------------------------------------------------------------------

def scatter_roi_power_vs_rating(
    subject: str,
    task: str = TASK,
    roi: str = "central",
    band: str = "beta",
) -> None:
    """Scatter plot of ROI-averaged power vs rating with temperature hue."""

    plots_dir = _plots_dir(subject)
    _ensure_dir(plots_dir)

    # Load power features and sensor info
    pow_df, _conn_df, _y, info = _load_features_and_targets(subject, task)

    # Identify ROI channels
    roi_map = _build_rois(info)
    roi_key = None
    for key in roi_map:
        if key.lower() == roi.lower():
            roi_key = key
            break
    if roi_key is None:
        print(f"ROI '{roi}' not found for sub-{subject}. Available: {list(roi_map)}")
        return
    chs = roi_map[roi_key]
    cols = [f"pow_{band}_{ch}" for ch in chs if f"pow_{band}_{ch}" in pow_df.columns]
    if not cols:
        print(f"No power columns for band '{band}' and ROI '{roi_key}' in sub-{subject}")
        return
    roi_power = pow_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Load epochs and align events for ratings and temperature
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None:
        print(f"Could not find epochs for sub-{subject}")
        return
    epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
    events = _load_events_df(subject, task)
    aligned_events = _align_events_to_epochs(events, epochs) if events is not None else None
    if aligned_events is None or len(aligned_events) == 0:
        print(f"No aligned events for sub-{subject}")
        return

    rating_col = _pick_first_column(aligned_events, RATING_COLUMNS)
    temp_col = _pick_first_column(aligned_events, PSYCH_TEMP_COLUMNS)
    if rating_col is None:
        print(f"No rating column found for sub-{subject}")
        return

    rating = pd.to_numeric(aligned_events[rating_col], errors="coerce")
    temp = pd.to_numeric(aligned_events[temp_col], errors="coerce") if temp_col is not None else None

    # Align lengths and drop missing
    n = min(len(roi_power), len(rating))
    data = {
        "power": roi_power.iloc[:n],
        "rating": rating.iloc[:n],
    }
    if temp is not None:
        data["temp"] = temp.iloc[:n]
    df = pd.DataFrame(data).dropna(subset=["power", "rating"])
    if df.empty:
        print(f"Not enough valid data for scatter plot: sub-{subject}")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    hue = "temp" if "temp" in df.columns else None
    sns.scatterplot(data=df, x="power", y="rating", hue=hue, ax=ax)
    sns.regplot(data=df, x="power", y="rating", scatter=False, ax=ax, color="k")
    band_rng = FEATURES_FREQ_BANDS.get(band)
    band_label = f"{band} ({band_rng[0]:g}\u2013{band_rng[1]:g} Hz)" if band_rng else band
    ax.set_xlabel(f"{roi_key} {band_label} power")
    ax.set_ylabel(f"Rating ({rating_col})")
    ax.set_title(f"{roi_key} {band} power vs rating")
    _save_fig(fig, plots_dir / f"scatter_{_sanitize(roi_key)}_{band}_power_vs_rating_temp")


# Trial-wise trajectory: ROI power and rating
# -----------------------------------------------------------------------------

def plot_trialwise_power_and_rating(
    subject: str,
    task: str = TASK,
    roi: str = "central",
    band: str = "alpha",
) -> None:
    """Plot trial-wise trajectories of ROI-averaged power and rating.

    Uses the target rating vector exported alongside features, ensuring the
    ratings and power features remain synchronized. A simple rolling-average
    smoothing is applied; adjust ``roll_window`` to modify the smoothing window
    or set it to ``None``/``1`` to disable.
    """

    plots_dir = _plots_dir(subject)
    _ensure_dir(plots_dir)

    # Load power features, pre-aligned target ratings, and channel info
    pow_df, _conn_df, y, info = _load_features_and_targets(subject, task)

    # Identify ROI channels
    roi_map = _build_rois(info)
    roi_key = None
    for key in roi_map:
        if key.lower() == roi.lower():
            roi_key = key
            break
    if roi_key is None:
        print(f"ROI '{roi}' not found for sub-{subject}. Available: {list(roi_map)}")
        return
    chs = roi_map[roi_key]
    cols = [f"pow_{band}_{ch}" for ch in chs if f"pow_{band}_{ch}" in pow_df.columns]
    if not cols:
        print(f"No power columns for band '{band}' and ROI '{roi_key}' in sub-{subject}")
        return
    roi_power = pow_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Use target ratings returned by _load_features_and_targets, which are
    # already aligned to pow_df. Fallback to events alignment only if ratings
    # are entirely missing.
    rating = pd.to_numeric(y, errors="coerce")
    rating_label = "Rating"
    if rating.isna().all():
        events = _load_events_df(subject, task)
        epo_path = _find_clean_epochs_path(subject, task)
        if events is None or epo_path is None:
            print(f"No rating data for sub-{subject}")
            return
        epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
        aligned_events = _align_events_to_epochs(events, epochs)
        if aligned_events is None or len(aligned_events) == 0:
            print(f"No aligned events for sub-{subject}")
            return
        rating_col = _pick_first_column(aligned_events, RATING_COLUMNS)
        if rating_col is None:
            print(f"No rating column found for sub-{subject}")
            return
        rating = pd.to_numeric(aligned_events[rating_col], errors="coerce")
        rating_label = f"Rating ({rating_col})"

    # Build DataFrame aligning trials
    n = min(len(roi_power), len(rating))
    df = pd.DataFrame(
        {
            "trial_index": np.arange(n),
            "power": roi_power.iloc[:n].to_numpy(),
            "rating": rating.iloc[:n].to_numpy(),
        }
    ).dropna(subset=["power", "rating"])
    if df.empty:
        print(f"Not enough valid data for trajectory plot: sub-{subject}")
        return

    # Optional rolling-average smoothing
    roll_window: Optional[int] = 3  # set to None or 1 to disable
    if roll_window and roll_window > 1:
        df["power_plot"] = df["power"].rolling(roll_window, center=True, min_periods=1).mean()
        df["rating_plot"] = df["rating"].rolling(roll_window, center=True, min_periods=1).mean()
    else:
        df["power_plot"] = df["power"]
        df["rating_plot"] = df["rating"]

    # Plot dual-axis line plot
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax2 = ax1.twinx()
    ax1.plot(df["trial_index"], df["power_plot"], color="tab:blue", label="Power")
    ax2.plot(df["trial_index"], df["rating_plot"], color="tab:orange", label="Rating")
    ax1.set_xlabel("Trial index")
    band_rng = FEATURES_FREQ_BANDS.get(band)
    band_label = f"{band} ({band_rng[0]:g}\u2013{band_rng[1]:g} Hz)" if band_rng else band
    ax1.set_ylabel(f"{roi_key} {band_label} power", color="tab:blue")
    ax2.set_ylabel(rating_label, color="tab:orange")
    ax1.set_title(f"{roi_key} {band} power and rating over trials")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    _save_fig(
        fig,
        plots_dir / f"trajectory_{_sanitize(roi_key)}_{band}_power_and_rating",
    )


# Power high vs low rating topographic difference
# -----------------------------------------------------------------------------

def plot_power_topomap_high_vs_low(subject: str, task: str = TASK, band: str = "alpha") -> None:
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    # Load power features and ratings
    pow_df, _conn_df, y, info = _load_features_and_targets(subject, task)

    cols = [c for c in pow_df.columns if c.startswith(f"pow_{band}_")]
    if not cols:
        print(f"No power features found for band {band} in sub-{subject}")
        return

    # Order channels according to info
    chan_order = [ch for ch in info["ch_names"] if f"pow_{band}_{ch}" in cols]
    if not chan_order:
        print(f"No matching channels for band {band} in sub-{subject}")
        return

    X = pow_df[[f"pow_{band}_{ch}" for ch in chan_order]].apply(pd.to_numeric, errors="coerce")

    # Align lengths and drop NaN ratings
    n = min(len(X), len(y))
    X = X.iloc[:n, :]
    y = pd.to_numeric(y.iloc[:n], errors="coerce")
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    if len(y) < 4:
        print(f"Too few trials for high/low split: sub-{subject}, band {band}")
        return

    med = float(np.nanmedian(y))
    hi_mask = y > med
    lo_mask = y <= med
    if hi_mask.sum() == 0 or lo_mask.sum() == 0:
        print(f"High/low groups empty for sub-{subject}, band {band}")
        return

    hi_mean = X.loc[hi_mask].mean(axis=0)
    lo_mean = X.loc[lo_mask].mean(axis=0)
    diff = hi_mean - lo_mean

    tvals: List[float] = []
    pvals: List[float] = []
    for ch in X.columns:
        xh = X.loc[hi_mask, ch]
        xl = X.loc[lo_mask, ch]
        if xh.notna().sum() >= 2 and xl.notna().sum() >= 2:
            t, p = stats.ttest_ind(xh, xl, equal_var=False, nan_policy="omit")
        else:
            t, p = np.nan, np.nan
        tvals.append(float(t))
        pvals.append(float(p))

    channel_names = [c.split(f"pow_{band}_", 1)[-1] for c in X.columns]
    stats_df = pd.DataFrame(
        {
            "channel": channel_names,
            "mean_high": hi_mean.to_numpy(),
            "mean_low": lo_mean.to_numpy(),
            "diff": diff.to_numpy(),
            "t": tvals,
            "p": pvals,
        }
    )
    stats_df.to_csv(stats_dir / f"topomap_diff_{band}_high_vs_low.tsv", sep="\t", index=False)

    # Plot topomap
    picks = mne.pick_channels(info["ch_names"], include=chan_order)
    info_sub = mne.pick_info(info, picks, copy=True)
    data = diff.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    mask_arr = np.array(pvals) < 0.05
    mask_plot = mask_arr if mask_arr.any() else None
    mask_params = (
        dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0.5, markersize=4)
        if mask_arr.any()
        else None
    )
    fig, ax = plt.subplots()
    try:
        mne.viz.plot_topomap(
            data,
            info_sub,
            axes=ax,
            show=False,
            vlim=(-vmax, vmax),
            cmap=TOPO_CMAP,
            contours=TOPO_CONTOURS,
            mask=mask_plot,
            mask_params=mask_params,
        )
    except TypeError:
        mne.viz.plot_topomap(
            data,
            info_sub,
            axes=ax,
            show=False,
            vmin=-vmax,
            vmax=vmax,
            cmap=TOPO_CMAP,
            contours=TOPO_CONTOURS,
            mask=mask_plot,
            mask_params=mask_params,
        )
    ax.set_title(f"{band} high\u2013low power")
    _save_fig(fig, plots_dir / f"topomap_diff_{band}_high_vs_low")

# Trial-level TFR sorted by rating
# -----------------------------------------------------------------------------

def plot_tfr_sorted_by_rating(subject: str, task: str = TASK, channel: str = "Cz") -> None:
    """Plot per-trial TFR power for a channel sorted by behavioral ratings."""

    plots_dir = _plots_dir(subject)
    _ensure_dir(plots_dir)

    # ------------------------------------------------------------------
    # Load epochs
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None:
        print(f"No cleaned epochs found for sub-{subject}, task-{task}")
        return
    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)

    # ------------------------------------------------------------------
    # Load and align ratings to epochs
    events = _load_events_df(subject, task)
    events = _align_events_to_epochs(events, epochs)
    rating_col = _pick_first_column(events, RATING_COLUMNS) if events is not None else None
    if rating_col is None:
        print(f"TFR sorted by rating: no rating column found for sub-{subject}")
        return
    ratings = pd.to_numeric(events[rating_col], errors="coerce").to_numpy()
    valid = ~np.isnan(ratings)
    if valid.sum() == 0:
        print(f"TFR sorted by rating: no valid ratings for sub-{subject}")
        return
    epochs = epochs[valid]
    ratings = ratings[valid]

    # ------------------------------------------------------------------
    # Compute trial-level TFR
    freqs = np.linspace(4.0, 40.0, 20)
    n_cycles = freqs / 2.0
    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=2,
        n_jobs=1,
        picks="eeg",
        verbose=False,
    )

    if channel not in power.ch_names:
        print(f"Channel {channel} not found for sub-{subject}")
        return

    ch_idx = power.ch_names.index(channel)
    data = power.data[:, ch_idx, :, :]  # (n_trials, n_freqs, n_times)
    data = data.mean(axis=1)  # average over frequency -> (n_trials, n_times)

    order = np.argsort(ratings)
    data = data[order]
    ratings_sorted = ratings[order]

    # ------------------------------------------------------------------
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(data, ax=ax, cmap="viridis", cbar_kws={"label": "Power"})

    times = power.times
    xticks = np.linspace(0, len(times) - 1, 5, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{times[i]:.1f}" for i in xticks])

    yticks = np.linspace(0, len(ratings_sorted) - 1, min(10, len(ratings_sorted)), dtype=int)
    ax.set_yticks(yticks + 0.5)
    ax.set_yticklabels([f"{ratings_sorted[i]:.1f}" for i in yticks])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rating (sorted)")
    ax.set_title(f"{channel} power sorted by rating")

    fname = f"tfr_sorted_by_rating_{_sanitize(channel)}"
    _save_fig(fig, plots_dir / fname)

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


def correlate_power_topomaps(subject: str, task: str = TASK, use_spearman: bool = True) -> None:
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    # Load features, targets, and sensor info
    pow_df, _conn_df, y, info = _load_features_and_targets(subject, task)

    # Drop only NaN targets; keep feature NaNs and handle per-channel pairwise deletion
    y = pd.to_numeric(y, errors="coerce")
    target_mask = y.notna()
    if target_mask.sum() < 5:
        print(f"Too few valid targets for correlations: sub-{subject}")
        return

    # Determine EEG channels in info (exclude EOG/MEG/etc.)
    eeg_idx = mne.pick_types(info, eeg=True, meg=False, seeg=False, eog=False, stim=False, exclude="bads")
    ch_names = [info["ch_names"][i] for i in eeg_idx]

    # For each band, collect per-channel columns like 'pow_alpha_Cz'
    for band in POWER_BANDS_TO_USE:
        cols = [c for c in pow_df.columns if c.startswith(f"pow_{band}_")]
        # Ensure channel ordering matches info
        def _chan_from_col(col: str) -> str:
            return col.split(f"pow_{band}_", 1)[-1]
        chan_order = [c for c in ch_names if f"pow_{band}_{c}" in cols]
        if not chan_order:
            print(f"No power features found for band {band} in sub-{subject}")
            continue
        X = pow_df[[f"pow_{band}_{ch}" for ch in chan_order]]
        # Align lengths to y
        n = min(len(X), len(y))
        X = X.iloc[:n, :]
        y_n = y.iloc[:n]
        do_spearman = bool(use_spearman and y_n.nunique() > 5)
        method_str = "Spearman" if do_spearman else "Pearson"
        band_rng = FEATURES_FREQ_BANDS.get(band)
        band_label = f"{band} ({band_rng[0]:g}\u2013{band_rng[1]:g} Hz)" if band_rng is not None else band

        # Compute correlation per channel with pairwise non-missing
        rs: List[float] = []
        ps: List[float] = []
        n_effs: List[int] = []
        for ch in chan_order:
            xi = pd.to_numeric(X[f"pow_{band}_{ch}"], errors="coerce")
            mask = xi.notna() & y_n.notna()
            if mask.sum() < 5:
                rs.append(np.nan)
                ps.append(1.0)
                n_effs.append(int(mask.sum()))
                continue
            if use_spearman and y_n.nunique() > 5:
                r, p = stats.spearmanr(xi[mask], y_n[mask], nan_policy="omit")
            else:
                r, p = stats.pearsonr(xi[mask], y_n[mask])
            rs.append(r)
            ps.append(p)
            n_effs.append(int(mask.sum()))
        r_arr = np.array(rs, dtype=float)
        p_arr = np.array(ps, dtype=float)

        # FDR across channels
        rej, crit = _fdr_bh(np.where(np.isnan(p_arr), 1.0, p_arr), alpha=0.05)
        sig_mask = rej & np.isfinite(r_arr)

        # Symmetric color limits
        vmax = np.nanmax(np.abs(r_arr)) if np.isfinite(r_arr).any() else 1.0
        vlim = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0

        # Plot topomap (subselect info to channels present in features to match lengths)
        try:
            # Info subset to only the channels we have correlations for
            picks_sub = mne.pick_channels(info["ch_names"], include=chan_order)
            info_sub = mne.pick_info(info, picks_sub, copy=True)
            # --- Cluster-based permutation across sensors ---
            # Build per-trial contribution matrix whose mean equals (scaled) correlation
            # Use rank-based transform if Spearman requested
            cluster_sig_mask = None
            try:
                X_all = pow_df[[f"pow_{band}_{ch}" for ch in chan_order]].apply(pd.to_numeric, errors="coerce")
                # Align to y length
                X_all = X_all.iloc[:n, :]
                y_vec = y_n.iloc[:n]
                # Trials with all channels and y present
                m_all = y_vec.notna() & X_all.notna().all(axis=1)
                if int(m_all.sum()) >= 8:
                    Xo = X_all[m_all]
                    yo = y_vec[m_all]
                    # Rank-transform for Spearman
                    if use_spearman and yo.nunique() > 5:
                        Xo = Xo.rank(method="average")
                        yo = yo.rank(method="average")
                    # z-score per channel and y
                    Xo = (Xo - Xo.mean(axis=0)) / Xo.std(axis=0, ddof=1)
                    yo = (yo - yo.mean()) / yo.std(ddof=1)
                    # Drop channels with zero variance (std NaN/inf)
                    keep = np.isfinite(Xo.std(axis=0, ddof=1)).to_numpy()
                    Xo = Xo.iloc[:, keep]
                    kept_chans = [ch for ch, k in zip(chan_order, keep) if k]
                    # Update info/adjacency for kept channels
                    if kept_chans:
                        picks_keep = mne.pick_channels(info_sub["ch_names"], include=kept_chans)
                        info_keep = mne.pick_info(info_sub, picks_keep, copy=True)
                        adjacency, ch_names_adj = mne.channels.find_ch_adjacency(info_keep, ch_type="eeg")
                        # Per-trial products shape (n_trials, n_kept_channels)
                        Z = Xo.to_numpy(dtype=float) * yo.to_numpy(dtype=float)[:, None]
                        # One-sample cluster permutation test on mean>0 (two-tailed)
                        T_obs, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(
                            Z, adjacency=adjacency, tail=0, n_permutations=2000, seed=42, n_jobs=1, out_type="mask"
                        )
                        # Build significant mask in kept channel space
                        sig_alpha = 0.05
                        sig_any = np.zeros(len(kept_chans), dtype=bool)
                        # Normalize clusters to boolean masks depending on MNE version
                        for cl, pv in zip(clusters, cluster_pv):
                            if pv < sig_alpha:
                                if isinstance(cl, np.ndarray) and cl.dtype == bool:
                                    sig_any |= cl
                                else:
                                    # out_type="mask" should already give boolean masks
                                    try:
                                        sig_any |= cl.astype(bool)
                                    except Exception:
                                        pass
                        # Map back to full chan_order mask
                        cluster_sig_mask = np.zeros(len(chan_order), dtype=bool)
                        for idx_keep, ch in enumerate(kept_chans):
                            if sig_any[idx_keep]:
                                try:
                                    i_full = chan_order.index(ch)
                                    cluster_sig_mask[i_full] = True
                                except ValueError:
                                    pass
                        # Save clusters TSV
                        try:
                            recs_clu = []
                            for cid, (cl, pv) in enumerate(zip(clusters, cluster_pv)):
                                if isinstance(cl, np.ndarray) and cl.dtype == bool:
                                    idxs = np.where(cl)[0]
                                else:
                                    # Fallback: attempt boolean cast
                                    idxs = np.where(np.asarray(cl).astype(bool))[0]
                                chs = [kept_chans[i] for i in idxs]
                                recs_clu.append({
                                    "cluster_id": cid,
                                    "size": int(len(idxs)),
                                    "p_cluster": float(pv),
                                    "sum_t": float(np.nansum(T_obs[idxs])) if np.size(T_obs) >= np.max(idxs)+1 else np.nan,
                                    "channels": ",".join(chs),
                                })
                            if recs_clu:
                                pd.DataFrame(recs_clu).to_csv(
                                    stats_dir / f"corr_stats_pow_topo_clusters_{band}.tsv", sep="\t", index=False
                                )
                        except Exception:
                            pass
                # else: too few valid trials; skip cluster test silently
            except Exception:
                # Robust to any cluster test failure, continue with plain topomap
                cluster_sig_mask = None

            fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.8))
            try:
                # Newer MNE versions use vlim for color limits
                mne.viz.plot_topomap(
                    r_arr,
                    info_sub,
                    axes=ax,
                    show=False,
                    vlim=(-vlim, +vlim),
                    cmap=TOPO_CMAP,
                    contours=TOPO_CONTOURS,
                    mask=(cluster_sig_mask if cluster_sig_mask is not None else sig_mask),
                    mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0.5, markersize=4),
                )
            except TypeError:
                # Fallback for older MNE versions expecting vmin/vmax
                mne.viz.plot_topomap(
                    r_arr,
                    info_sub,
                    axes=ax,
                    show=False,
                    vmin=-vlim,
                    vmax=+vlim,
                    cmap=TOPO_CMAP,
                    contours=TOPO_CONTOURS,
                    mask=(cluster_sig_mask if cluster_sig_mask is not None else sig_mask),
                    mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0.5, markersize=4),
                )
            ax.set_title(f"{method_str} r: power {band_label} vs rating\nFDR q<0.05")
            # Colorbar
            import matplotlib.colors as mcolors
            from matplotlib.cm import ScalarMappable
            sm = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim), cmap=TOPO_CMAP)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
            cbar.set_label(f"{method_str} r")
            _save_fig(fig, plots_dir / f"topomap_corr_pow_{band}_vs_rating")
        except Exception as e:
            print(f"Topomap plotting failed for band {band}: {e}")

        # Save TSV of stats
        out = pd.DataFrame({
            "channel": chan_order,
            "r": r_arr,
            "p": p_arr,
            "n_eff": np.array(n_effs, dtype=float),
            "fdr_reject": sig_mask,
            "fdr_crit_p": [crit] * len(chan_order),
            "method": [method_str] * len(chan_order),
            "band": [band] * len(chan_order),
            "band_range": [f"{band_rng[0]:g}-{band_rng[1]:g} Hz" if band_rng is not None else ""] * len(chan_order),
        })
        out.to_csv(stats_dir / f"corr_stats_pow_{band}_vs_rating.tsv", sep="\t", index=False)


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
) -> None:
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    # Load power features, target ratings, and sensor info
    pow_df, _conn_df, y, info = _load_features_and_targets(subject, task)
    y = pd.to_numeric(y, errors="coerce")

    # Load epochs for alignment and events for temperature
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None:
        print(f"Could not find epochs for ROI correlations: sub-{subject}")
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
        except Exception:
            Z_df_temp = Z_df_full

    def _partial_corr_xy_given_Z(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str) -> Tuple[float, float, int]:
        # Align and drop missing jointly
        df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
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
                    rng = np.random.default_rng(42)
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
                    rng = np.random.default_rng(123)
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
                        rng = np.random.default_rng(42)
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
                        rng = np.random.default_rng(123)
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

# Connectivity ROI summary correlations (within/between ROI averages)
# -----------------------------------------------------------------------------

def correlate_connectivity_roi_summaries(
    subject: str,
    task: str = TASK,
    use_spearman: bool = True,
    partial_covars: Optional[List[str]] = None,
    bootstrap: int = 0,
    n_perm: int = 0,
) -> None:
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

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
        except Exception:
            Z_df_temp = Z_df_full

    def _partial_corr_xy_given_Z(x: pd.Series, y: pd.Series, Z: pd.DataFrame, method: str) -> Tuple[float, float, int]:
        df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
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
                rng = np.random.default_rng(42)
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
                rng = np.random.default_rng(123)
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
                        rng = np.random.default_rng(42)
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
                        rng = np.random.default_rng(123)
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
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    # Load features/targets
    feats_dir = _features_dir(subject)
    conn_path = feats_dir / "features_connectivity.tsv"
    y_path = feats_dir / "target_vas_ratings.tsv"
    if not conn_path.exists() or not y_path.exists():
        print(f"Connectivity features or targets missing for sub-{subject}; skipping connectivity correlations.")
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

        # Heatmap plotting intentionally removed due to feature count and file size.
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
        except Exception:
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
) -> None:
    print(f"=== Behavior-feature analyses: sub-{subject}, task-{task} ===")
    try:
        plot_psychometrics(subject, task)
    except Exception as e:
        print(f"Psychometric plotting failed for sub-{subject}: {e}")
    try:
        correlate_power_topomaps(subject, task, use_spearman=use_spearman)
    except Exception as e:
        print(f"Power correlations failed for sub-{subject}: {e}")
    try:
        for b in POWER_BANDS_TO_USE:
            plot_power_topomap_high_vs_low(subject, task, band=b)
    except Exception as e:
        print(f"High/low power topomap failed for sub-{subject}: {e}")
    try:
        correlate_power_roi_stats(
            subject,
            task,
            use_spearman=use_spearman,
            partial_covars=partial_covars,
            bootstrap=bootstrap,
            n_perm=n_perm,
        )
    except Exception as e:
        print(f"ROI power correlations failed for sub-{subject}: {e}")
    try:
        correlate_connectivity_heatmaps(subject, task, use_spearman=use_spearman)
    except Exception as e:
        print(f"Connectivity correlations failed for sub-{subject}: {e}")
    try:
        correlate_connectivity_roi_summaries(
            subject,
            task,
            use_spearman=use_spearman,
            partial_covars=partial_covars,
            bootstrap=bootstrap,
            n_perm=n_perm,
        )
    except Exception as e:
        print(f"Connectivity ROI summaries failed for sub-{subject}: {e}")

    if build_report:
        try:
            build_subject_report(subject, task)
        except Exception as e:
            print(f"Report build failed for sub-{subject}: {e}")


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
            except Exception:
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
        except Exception:
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
            except Exception:
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
    except Exception:
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
    # Topomaps
    for band in POWER_BANDS_TO_USE:
        p = plots_dir / f"topomap_corr_pow_{band}_vs_rating.png"
        if p.exists():
            rep.add_image(p, title=f"Topomap corr {band}", section="Topomaps")
    # Save stats TSVs as links
    for fn in [
        "corr_stats_pow_roi_vs_rating.tsv",
        "corr_stats_pow_roi_vs_temp.tsv",
    ]:
        p = stats_dir / fn
        if p.exists():
            rep.add_html(f"<p><a href='{p.as_posix()}'>Download {fn}</a></p>", title=fn, section="Stats")
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
    except Exception:
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
            )
    if do_group or group_only:
        aggregate_group_level(subjects, task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Behavioral psychometrics and EEG feature correlations")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject IDs to process (e.g., 001 002) or 'all' for all configured subjects")
    parser.add_argument("--task", default=TASK, help="Task label (default from config)")
    parser.add_argument("--pearson", action="store_true", help="Use Pearson correlations (default: Spearman when appropriate)")
    parser.add_argument("--partial-covars", nargs="*", default=None, help="Event columns to control for in partial correlations (e.g., temperature trial_number)")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap resamples for per-subject ROI r 95% CI (0 to disable)")
    parser.add_argument("--n-perm", type=int, default=0, help="Number of permutations for permutation p-values (0 to disable)")
    parser.add_argument("--group", action="store_true", help="Also aggregate group-level results across subjects")
    parser.add_argument("--group-only", action="store_true", help="Only run group-level aggregation (skip per-subject)")
    parser.add_argument("--report", action="store_true", help="Build per-subject MNE HTML report")
    args = parser.parse_args()

    subs = None if args.subjects in (None, [], ["all"]) else args.subjects
    main(
        subs,
        task=args.task,
        use_spearman=not args.pearson,
        partial_covars=args.partial_covars,
        bootstrap=args.bootstrap,
        n_perm=args.n_perm,
        do_group=args.group,
        group_only=args.group_only,
        build_reports=args.report,
    )
