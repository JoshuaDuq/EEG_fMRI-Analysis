from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
FIG_DPI = 200
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
            fig.savefig(plots_dir / "psychometric_rating_vs_temp.png", dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig)
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
                fig.savefig(plots_dir / "psychometric_pain_vs_temp.png", dpi=FIG_DPI, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"Psychometric binning failed: {e}")


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

        # Compute correlation per channel with pairwise non-missing
        rs: List[float] = []
        ps: List[float] = []
        for ch in chan_order:
            xi = pd.to_numeric(X[f"pow_{band}_{ch}"], errors="coerce")
            mask = xi.notna() & y_n.notna()
            if mask.sum() < 5:
                rs.append(np.nan)
                ps.append(1.0)
                continue
            if use_spearman and y_n.nunique() > 5:
                r, p = stats.spearmanr(xi[mask], y_n[mask], nan_policy="omit")
            else:
                r, p = stats.pearsonr(xi[mask], y_n[mask])
            rs.append(r)
            ps.append(p)
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
                    mask=sig_mask,
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
                    mask=sig_mask,
                    mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0.5, markersize=4),
                )
            ax.set_title(f"r (power {band} vs rating)\nFDR q<0.05")
            # Colorbar
            import matplotlib.colors as mcolors
            from matplotlib.cm import ScalarMappable
            sm = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim), cmap=TOPO_CMAP)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
            fig.savefig(plots_dir / f"topomap_corr_pow_{band}_vs_rating.png", dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"Topomap plotting failed for band {band}: {e}")

        # Save TSV of stats
        out = pd.DataFrame({
            "channel": chan_order,
            "r": r_arr,
            "p": p_arr,
            "fdr_reject": sig_mask,
            "fdr_crit_p": [crit] * len(chan_order),
        })
        out.to_csv(stats_dir / f"corr_stats_pow_{band}_vs_rating.tsv", sep="\t", index=False)


# -----------------------------------------------------------------------------
# Correlation: ROI-averaged power vs behavior (rating and temperature)
# -----------------------------------------------------------------------------

def correlate_power_roi_stats(subject: str, task: str = TASK, use_spearman: bool = True) -> None:
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

    # For each band and ROI, average channels and correlate with behavior
    for band in POWER_BANDS_TO_USE:
        # Identify available columns for this band
        band_cols_available = {c for c in pow_df.columns if c.startswith(f"pow_{band}_")}
        if not band_cols_available:
            continue

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
                recs_rating.append({
                    "roi": roi,
                    "band": band,
                    "r": float(r),
                    "p": float(p),
                    "n": n_eff,
                    "method": method,
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
                    recs_temp.append({
                        "roi": roi,
                        "band": band,
                        "r": float(r2),
                        "p": float(p2),
                        "n": n_eff2,
                        "method": method2,
                    })

    # Save TSVs with FDR across all ROI-band tests per target
    if recs_rating:
        df_r = pd.DataFrame(recs_rating)
        rej, crit = _fdr_bh(df_r["p"].to_numpy(), alpha=0.05)
        df_r["fdr_reject"] = rej
        df_r["fdr_crit_p"] = crit
        df_r.to_csv(stats_dir / "corr_stats_pow_roi_vs_rating.tsv", sep="\t", index=False)

    if recs_temp:
        df_t = pd.DataFrame(recs_temp)
        rej_t, crit_t = _fdr_bh(df_t["p"].to_numpy(), alpha=0.05)
        df_t["fdr_reject"] = rej_t
        df_t["fdr_crit_p"] = crit_t
        df_t.to_csv(stats_dir / "corr_stats_pow_roi_vs_temp.tsv", sep="\t", index=False)


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
        recs: List[Dict[str, object]] = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                recs.append({
                    "node_i": nodes[i],
                    "node_j": nodes[j],
                    "r": rvals[i, j],
                    "p": pvals[i, j],
                })
        pd.DataFrame(recs).to_csv(stats_dir / f"corr_stats_edges_{_sanitize(pref)}_vs_rating.tsv", sep="\t", index=False)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def process_subject(subject: str, task: str = TASK) -> None:
    print(f"=== Behavior-feature analyses: sub-{subject}, task-{task} ===")
    try:
        plot_psychometrics(subject, task)
    except Exception as e:
        print(f"Psychometric plotting failed for sub-{subject}: {e}")
    try:
        correlate_power_topomaps(subject, task)
    except Exception as e:
        print(f"Power correlations failed for sub-{subject}: {e}")
    try:
        correlate_power_roi_stats(subject, task)
    except Exception as e:
        print(f"ROI power correlations failed for sub-{subject}: {e}")
    try:
        correlate_connectivity_heatmaps(subject, task)
    except Exception as e:
        print(f"Connectivity correlations failed for sub-{subject}: {e}")


def main(subjects: Optional[List[str]] = None, task: str = TASK):
    if subjects is None or subjects == ["all"]:
        subjects = SUBJECTS
    for sub in subjects:
        process_subject(sub, task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Behavioral psychometrics and EEG feature correlations")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject IDs (e.g., 001 002) or 'all'")
    parser.add_argument("--task", default=TASK, help="Task label (default from config)")
    args = parser.parse_args()

    subs = None if args.subjects in (None, [], ["all"]) else args.subjects
    main(subs, task=args.task)
