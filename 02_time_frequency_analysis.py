import os
import sys
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath

# Resolve project config similar to 01_foundational_analysis.py
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
    cfg = _Fallback()  # type: ignore

BIDS_ROOT = Path(getattr(cfg, "bids_root", PROJECT_ROOT / "eeg_pipeline" / "bids_output"))
DERIV_ROOT = Path(getattr(cfg, "deriv_root", BIDS_ROOT / "derivatives"))
TASK = getattr(cfg, "task", "thermalactive")


# =========================
# CONFIG — tweak here
# =========================
# Analysis scope
DEFAULT_TASK = TASK
DEFAULT_TEMPERATURE_STRATEGY = "pooled"  # one of {"pooled", "per", "both"}

# TFR computation
FREQ_MIN = 4.0
FREQ_MAX = 100.0
N_FREQS = 40
# n_cycles rule: proportional to frequency (e.g., f/2)
N_CYCLES_FACTOR = 0.5  # n_cycles = freqs * N_CYCLES_FACTOR
TFR_DECIM = 2
TFR_PICKS = "eeg"

# Baseline and windows
BASELINE = (None, 0.0)  # pre-stim
DEFAULT_PLATEAU_TMIN = 3.0
DEFAULT_PLATEAU_TMAX = 10.0

# Bands (upper bound None => capped at data's max freq)
BAND_BOUNDS = {
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, None),
}

# Plot styling
FIG_DPI = 300
FIG_PAD_INCH = 0.2
TOPO_CONTOURS = 6
TOPO_CMAP = "RdBu_r"
COLORBAR_FRACTION = 0.03
COLORBAR_PAD = 0.02

# ROI mask styling
ROI_MASK_PARAMS_DEFAULT = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0.5, markersize=4)

# Columns that may store stimulus temperature
TEMPERATURE_COLUMNS = ["stimulus_temp", "stimulus_temperature", "temp", "temperature"]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _roi_definitions() -> Dict[str, list[str]]:
    """Scientifically reasonable sensor-space ROIs using 10-10 labels.

    Patterns are regexes matched case-insensitively against channel names.
    """
    return {
        # Frontal pole + frontal
        "Frontal": [r"^(Fpz|Fp[12]|AFz|AF[3-8]|Fz|F[1-8])$"],
        # Central strip only (C-series)
        "Central": [r"^(Cz|C[1-6])$"],
        # Parietal (P-series)
        "Parietal": [r"^(Pz|P[1-8])$"],
        # Occipital and parieto-occipital
        "Occipital": [r"^(Oz|O[12]|POz|PO[3-8])$"],
        # Lateral temporal regions
        "Temporal": [r"^(T7|T8|TP7|TP8|FT7|FT8)$"],
        # Bilateral sensorimotor strip (FC/C/CP around hand knob)
        "Sensorimotor": [r"^(FC[234]|FCz)$", r"^(C[234]|Cz)$", r"^(CP[234]|CPz)$"],
    }


def _find_roi_channels(info: mne.Info, patterns: list[str]) -> list[str]:
    chs = info["ch_names"]
    out: list[str] = []
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        out.extend([ch for ch in chs if rx.match(ch)])
    # Preserve original channel order and deduplicate
    seen = set()
    ordered = []
    for ch in chs:
        if ch in out and ch not in seen:
            seen.add(ch)
            ordered.append(ch)
    return ordered


def _build_rois(info: mne.Info) -> Dict[str, list[str]]:
    roi_map = {}
    for roi, pats in _roi_definitions().items():
        chans = _find_roi_channels(info, pats)
        if len(chans) > 0:
            roi_map[roi] = chans
    return roi_map


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


def _format_temp_label(val: float) -> str:
    """Format a temperature value as a safe label for paths, e.g., 47.3 -> '47p3'."""
    try:
        v = float(val)
    except Exception:
        return _sanitize(str(val))
    # Keep one decimal if present in data; replace '.' with 'p'
    s = f"{v:.1f}"
    return s.replace(".", "p")


def _find_tfr_path(subject: str, task: str) -> Optional[Path]:
    # Prefer exact brief-provided filename
    p1 = DERIV_ROOT / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_power_epo-tfr.h5"
    if p1.exists():
        return p1
    # Fallback: any *_epo-tfr.h5 under subject eeg dir
    eeg_dir = DERIV_ROOT / f"sub-{subject}" / "eeg"
    if eeg_dir.exists():
        cands = sorted(eeg_dir.glob(f"sub-{subject}_task-{task}*_epo-tfr.h5"))
        if cands:
            return cands[0]
    # Last resort: recursive search in subject dir
    subj_dir = DERIV_ROOT / f"sub-{subject}"
    if subj_dir.exists():
        for c in sorted(subj_dir.rglob(f"sub-{subject}_task-{task}*_epo-tfr.h5")):
            return c
    return None


def _apply_baseline_safe(tfr_obj, baseline: Tuple[Optional[float], Optional[float]] = BASELINE, mode: str = "logratio"):
    try:
        tfr_obj.apply_baseline(baseline=baseline, mode=mode)
        print(f"Applied baseline {baseline} with mode='{mode}'.")
    except Exception as e:
        print(f"Baseline correction skipped (no pre-stim interval or error): {e}")


def _pick_central_channel(info: mne.Info, preferred: str = "Cz") -> str:
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
    print(f"Channel '{preferred}' not found; using '{fallback}' instead.")
    return fallback


def _save_fig(fig_obj: Any, out_dir: Path, name: str, formats: Optional[list[str]] = None) -> None:
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
        # Robust save without constrained_layout to avoid layout warnings
        for ext_i in exts:
            out_name = (f"{stem}.{ext_i}" if i == 0 else f"{stem}_{i+1}.{ext_i}")
            out_path = out_dir / out_name
            try:
                try:
                    f.set_constrained_layout(False)
                except Exception:
                    pass
                try:
                    f.tight_layout()
                except Exception:
                    try:
                        f.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08)
                    except Exception:
                        pass
                try:
                    f.canvas.draw()
                except Exception:
                    pass
                f.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=FIG_PAD_INCH)
                saved_any = True
                print(f"Saved: {out_path}")
            except Exception as e:
                # Fallback: try saving without tight bbox (some backends raise 'float division by zero')
                try:
                    f.savefig(out_path, dpi=FIG_DPI)
                    saved_any = True
                    print(f"Saved (no tight bbox) due to layout error for: {out_path}. Reason: {e}")
                except Exception as e2:
                    print(f"Failed to save figure to {out_path}: {e2} (original error: {e})")
        plt.close(f)
        if not saved_any:
            print(f"Warning: no output saved for figure named '{name}'")


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
        t_mask = (times >= tmin) & (times <= tmax)
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
        )


def plot_cz_all_trials(tfr, out_dir: Path, baseline: Tuple[Optional[float], Optional[float]] = BASELINE) -> None:
    # Work on a copy for baseline correction
    tfr_copy = tfr.copy()
    _apply_baseline_safe(tfr_copy, baseline=baseline, mode="logratio")

    # Average across trials if EpochsTFR
    if isinstance(tfr_copy, mne.time_frequency.EpochsTFR):
        tfr_avg = tfr_copy.average()
    else:
        tfr_avg = tfr_copy  # already AverageTFR

    cz = _pick_central_channel(tfr_avg.info, preferred="Cz")
    try:
        fig = tfr_avg.plot(picks=cz, show=False)
        try:
            fig.suptitle("Cz TFR — all trials (baseline logratio)", fontsize=12)
        except Exception:
            pass
        _save_fig(fig, out_dir, f"tfr_Cz_all_trials_baseline_logratio.png")
    except Exception as e:
        print(f"Cz TFR plot failed: {e}")


def contrast_pain_nonpain(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
) -> None:
    if not isinstance(tfr, mne.time_frequency.EpochsTFR):
        print("Contrast requires EpochsTFR (trial-level). Skipping contrasts and using only overall average.")
        return
    if events_df is None or "pain_binary_coded" not in events_df.columns:
        print("Events with 'pain_binary_coded' required for contrast; skipping.")
        return

    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df)
    n = min(n_epochs, n_meta)
    if n_epochs != n_meta:
        print(f"Warning: tfr epochs ({n_epochs}) != events rows ({n_meta}); trimming to {n}.")

    # Prefer labels from TFR metadata if available to ensure perfect alignment
    if getattr(tfr, "metadata", None) is not None and "pain_binary_coded" in tfr.metadata.columns:
        pain_vec = pd.to_numeric(tfr.metadata.iloc[:n]["pain_binary_coded"], errors="coerce").fillna(0).astype(int).values
    else:
        pain_vec = pd.to_numeric(events_df.iloc[:n]["pain_binary_coded"], errors="coerce").fillna(0).astype(int).values
    pain_mask = np.asarray(pain_vec == 1, dtype=bool)
    non_mask = np.asarray(pain_vec == 0, dtype=bool)

    # Debug counts before deciding to skip
    print(f"Debug: n_epochs={n_epochs}, n_meta={n_meta}, n={n}, len_pain_vec={len(pain_vec)}")
    print(f"Pain/non-pain counts (n={n}): pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())}.")
    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        print("One of the groups has zero trials; skipping contrasts.")
        return

    # Subset and baseline-correct per-epoch before averaging
    tfr_sub = tfr.copy()[:n]
    if len(pain_mask) != len(tfr_sub):
        print(f"Warning: mask length ({len(pain_mask)}) != TFR epochs ({len(tfr_sub)}); reslicing to match.")
        n2 = min(len(tfr_sub), len(pain_mask))
        tfr_sub = tfr_sub[:n2]
        pain_mask = pain_mask[:n2]
        non_mask = non_mask[:n2]
        print(f"Debug after reslice: len(tfr_sub)={len(tfr_sub)}, len(pain_mask)={len(pain_mask)}")
    _apply_baseline_safe(tfr_sub, baseline=baseline, mode="logratio")

    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    cz = _pick_central_channel(tfr_pain.info, preferred="Cz")

    # Plot Cz for both conditions
    try:
        fig = tfr_pain.plot(picks=cz, show=False)
        _save_fig(fig, out_dir, "tfr_Cz_painful_baseline_logratio.png")
    except Exception as e:
        print(f"Cz painful plot failed: {e}")
    try:
        fig = tfr_non.plot(picks=cz, show=False)
        _save_fig(fig, out_dir, "tfr_Cz_nonpainful_baseline_logratio.png")
    except Exception as e:
        print(f"Cz non-painful plot failed: {e}")

    # Difference (pain - non)
    try:
        tfr_diff = tfr_pain.copy()
        tfr_diff.data = tfr_pain.data - tfr_non.data
        tfr_diff.comment = "pain-minus-nonpain"
        fig = tfr_diff.plot(picks=cz, show=False)
        _save_fig(fig, out_dir, "tfr_Cz_pain_minus_nonpain_baseline_logratio.png")
    except Exception as e:
        print(f"Cz difference plot failed: {e}")

    # Grouped topomap grid for alpha/beta/gamma x (pain/non/diff)
    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    tmin_eff = float(max(times.min(), tmin_req))
    tmax_eff = float(min(times.max(), tmax_req))
    fmax_available = float(np.max(tfr_pain.freqs))
    bands: Dict[str, Tuple[float, float]] = {
        "alpha": BAND_BOUNDS["alpha"],
        "beta": BAND_BOUNDS["beta"],
        "gamma": (BAND_BOUNDS["gamma"][0], fmax_available if BAND_BOUNDS["gamma"][1] is None else BAND_BOUNDS["gamma"][1]),
    }
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
        # Common scaling for pain/non within band
        vmin = float(min(pain_data.min(), non_data.min()))
        vmax = float(max(pain_data.max(), non_data.max()))
        diff_abs = float(np.nanmax(np.abs(diff_data))) if np.isfinite(diff_data).any() else 0.0
        # Plot Pain (col 0), Non-pain (col 1), leave col 2 empty, Diff (col 3)
        # Pain
        ax = axes[r, 0]
        try:
            mne.viz.plot_topomap(pain_data, tfr_pain.info, axes=ax, show=False, vmin=vmin, vmax=vmax, cmap=TOPO_CMAP)
        except Exception:
            _plot_topomap_on_ax(ax, pain_data, tfr_pain.info)
        # Non-pain
        ax = axes[r, 1]
        try:
            mne.viz.plot_topomap(non_data, tfr_pain.info, axes=ax, show=False, vmin=vmin, vmax=vmax, cmap=TOPO_CMAP)
        except Exception:
            _plot_topomap_on_ax(ax, non_data, tfr_pain.info)
        # Spacer column (col 2)
        axes[r, 2].axis('off')
        # Diff
        ax = axes[r, 3]
        try:
            mne.viz.plot_topomap(diff_data, tfr_pain.info, axes=ax, show=False,
                                 vmin=-diff_abs if diff_abs > 0 else None,
                                 vmax=+diff_abs if diff_abs > 0 else None,
                                 cmap=TOPO_CMAP)
        except Exception:
            _plot_topomap_on_ax(ax, diff_data, tfr_pain.info)
        if r == 0:
            for c_title in (0, 1, 3):
                axes[r, c_title].set_title(cond_labels[c_title], fontsize=9, pad=4)
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)
        # Add compact colorbars per row
        try:
            sm_pn = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=TOPO_CMAP)
            sm_pn.set_array([])
            cbar_pn = fig.colorbar(sm_pn, ax=[axes[r, 0], axes[r, 1]], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
            try:
                cbar_pn.set_label("log10(power/baseline)")
            except Exception:
                pass
            if diff_abs > 0:
                sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=TOPO_CMAP)
                sm_diff.set_array([])
                cbar_diff = fig.colorbar(sm_diff, ax=axes[r, 3], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
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
    if events_df is None or "pain_binary_coded" not in events_df.columns:
        print("Events with 'pain_binary_coded' required for ROI topomap contrasts; skipping.")
        return

    n_epochs = tfr.data.shape[0]
    n_meta = len(events_df)
    n = min(n_epochs, n_meta)
    if n_epochs != n_meta:
        print(f"ROI topomaps: tfr epochs ({n_epochs}) != events rows ({n_meta}); trimming to {n}.")

    # Prefer labels from TFR metadata if available
    if getattr(tfr, "metadata", None) is not None and "pain_binary_coded" in tfr.metadata.columns:
        pain_vec = pd.to_numeric(tfr.metadata.iloc[:n]["pain_binary_coded"], errors="coerce").fillna(0).astype(int).values
    else:
        pain_vec = pd.to_numeric(events_df.iloc[:n]["pain_binary_coded"], errors="coerce").fillna(0).astype(int).values
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
    bands: Dict[str, Tuple[float, float]] = {
        "alpha": BAND_BOUNDS["alpha"],
        "beta": BAND_BOUNDS["beta"],
        "gamma": (BAND_BOUNDS["gamma"][0], fmax_available if BAND_BOUNDS["gamma"][1] is None else BAND_BOUNDS["gamma"][1]),
    }
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
            gridspec_kw={"width_ratios": [1.0, 1.0, 0.25, 1.0], "wspace": 0.4},
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
            vmin = float(min(pain_data.min(), non_data.min()))
            vmax = float(max(pain_data.max(), non_data.max()))
            diff_abs = float(np.nanmax(np.abs(diff_data))) if np.isfinite(diff_data).any() else 0.0
            # Pain (col 0)
            ax = axes[r, 0]
            try:
                mne.viz.plot_topomap(pain_data, tfr_pain.info, axes=ax, show=False, vmin=vmin, vmax=vmax,
                                     mask=mask_vec, mask_params=mask_params, cmap=TOPO_CMAP)
            except Exception:
                _plot_topomap_on_ax(ax, pain_data, tfr_pain.info, mask=mask_vec, mask_params=mask_params)
            # Non-pain (col 1)
            ax = axes[r, 1]
            try:
                mne.viz.plot_topomap(non_data, tfr_pain.info, axes=ax, show=False, vmin=vmin, vmax=vmax,
                                     mask=mask_vec, mask_params=mask_params, cmap=TOPO_CMAP)
            except Exception:
                _plot_topomap_on_ax(ax, non_data, tfr_pain.info, mask=mask_vec, mask_params=mask_params)
            # Spacer (col 2)
            axes[r, 2].axis('off')
            # Diff (col 3)
            ax = axes[r, 3]
            try:
                mne.viz.plot_topomap(diff_data, tfr_pain.info, axes=ax, show=False,
                                     vmin=-diff_abs if diff_abs > 0 else None,
                                     vmax=+diff_abs if diff_abs > 0 else None,
                                     mask=mask_vec, mask_params=mask_params, cmap=TOPO_CMAP)
            except Exception:
                _plot_topomap_on_ax(ax, diff_data, tfr_pain.info, mask=mask_vec, mask_params=mask_params)
            if r == 0:
                for c_title in (0, 1, 3):
                    axes[r, c_title].set_title(cond_labels[c_title], fontsize=9, pad=4)
            axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)
            # Add compact colorbars per row
            try:
                sm_pn = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=TOPO_CMAP)
                sm_pn.set_array([])
                cbar_pn = fig.colorbar(sm_pn, ax=[axes[r, 0], axes[r, 1]], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
                try:
                    cbar_pn.set_label("log10(power/baseline)")
                except Exception:
                    pass
                if diff_abs > 0:
                    sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=TOPO_CMAP)
                    sm_diff.set_array([])
                    cbar_diff = fig.colorbar(sm_diff, ax=axes[r, 3], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
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
        _save_fig(fig, out_dir, f"topomap_ROI-{_sanitize(roi)}_grid_bands_pain_non_diff_baseline_logratio.png", formats=["png", "svg"])

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
) -> None:
    """Plot baseline-corrected TFR per ROI (epochs averaged) and save."""
    for roi, tfr in roi_tfrs.items():
        tfr_c = tfr.copy()
        _apply_baseline_safe(tfr_c, baseline=baseline, mode="logratio")
        tfr_avg = tfr_c.average()
        ch = tfr_avg.info['ch_names'][0]
        try:
            fig = tfr_avg.plot(picks=ch, show=False)
            try:
                fig.suptitle(f"ROI: {roi} — all trials (baseline logratio)", fontsize=12)
            except Exception:
                pass
            _save_fig(fig, out_dir, f"tfr_ROI-{_sanitize(roi)}_all_trials_baseline_logratio.png")
        except Exception as e:
            print(f"ROI {roi} TFR plot failed: {e}")

        # Also save band-limited TFRs per ROI
        try:
            fmax_available = float(np.max(tfr_avg.freqs))
            bands: Dict[str, Tuple[float, float]] = {
                "alpha": BAND_BOUNDS["alpha"],
                "beta": BAND_BOUNDS["beta"],
                "gamma": (BAND_BOUNDS["gamma"][0], fmax_available if BAND_BOUNDS["gamma"][1] is None else BAND_BOUNDS["gamma"][1]),
            }
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
                    _save_fig(fig_b, out_dir, f"tfr_ROI-{_sanitize(roi)}_{band}_all_trials_baseline_logratio.png")
                except Exception as e:
                    print(f"ROI {roi} band {band} TFR plot failed: {e}")
        except Exception as e:
            print(f"ROI {roi} banded TFR export skipped: {e}")


def plot_topomaps_rois_all_trials(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    roi_map: Dict[str, list[str]],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = BASELINE,
    plateau_window: Tuple[float, float] = (DEFAULT_PLATEAU_TMIN, DEFAULT_PLATEAU_TMAX),
) -> None:
    """Topomaps per ROI for all trials averaged, baseline-corrected.

    Highlights ROI sensors via mask, and plots per-band topomaps over a
    specified plateau window.
    """
    tfr_all = tfr.copy()
    _apply_baseline_safe(tfr_all, baseline=baseline, mode="logratio")
    if isinstance(tfr_all, mne.time_frequency.EpochsTFR):
        tfr_avg = tfr_all.average()
    else:
        tfr_avg = tfr_all

    fmax_available = float(np.max(tfr_avg.freqs))
    bands: Dict[str, Tuple[float, float]] = {
        "alpha": BAND_BOUNDS["alpha"],
        "beta": BAND_BOUNDS["beta"],
        "gamma": (BAND_BOUNDS["gamma"][0], fmax_available if BAND_BOUNDS["gamma"][1] is None else BAND_BOUNDS["gamma"][1]),
    }
    times = np.asarray(tfr_avg.times)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times.min(), tmin_req))
    tmax = float(min(times.max(), tmax_req))

    ch_names = tfr_avg.info["ch_names"]
    for roi, roi_chs in roi_map.items():
        mask_vec = np.array([ch in roi_chs for ch in ch_names], dtype=bool)
        mask_params = ROI_MASK_PARAMS_DEFAULT

        n_rows = len(bands)
        fig, axes = plt.subplots(n_rows, 1, figsize=(4.0, 3.5 * n_rows), squeeze=False)
        for r, (band, (fmin, fmax)) in enumerate(bands.items()):
            fmax_eff = min(fmax, fmax_available)
            if fmin >= fmax_eff:
                axes[r, 0].axis('off')
                continue
            data = _average_tfr_band(tfr_avg, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
            if data is None:
                axes[r, 0].axis('off')
                continue
            # Use per-row scaling and add colorbar
            vmin = float(np.nanmin(data))
            vmax = float(np.nanmax(data))
            try:
                mne.viz.plot_topomap(data, tfr_avg.info, axes=axes[r, 0], show=False,
                                     vmin=vmin, vmax=vmax,
                                     mask=mask_vec, mask_params=mask_params, cmap=TOPO_CMAP)
            except Exception:
                _plot_topomap_on_ax(axes[r, 0], data, tfr_avg.info, mask=mask_vec, mask_params=mask_params)
            axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)
            try:
                sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=TOPO_CMAP)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=axes[r, 0], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
                try:
                    cbar.set_label("log10(power/baseline)")
                except Exception:
                    pass
            except Exception:
                pass
        fig.suptitle(f"ROI: {roi} — Topomaps (all trials; baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s)", fontsize=12)
        try:
            fig.supylabel("Frequency bands", fontsize=10)
            fig.supxlabel("All trials", fontsize=10)
        except Exception:
            pass
        _save_fig(fig, out_dir, f"topomap_ROI-{_sanitize(roi)}_grid_bands_all_trials_baseline_logratio.png", formats=["png", "svg"])


def contrast_pain_nonpain_rois(
    roi_tfrs: Dict[str, mne.time_frequency.EpochsTFR],
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    baseline: Tuple[Optional[float], Optional[float]] = (None, 0.0),
) -> None:
    """Pain vs non-pain contrasts per ROI (plots ROI-averaged TFRs)."""
    if events_df is None or "pain_binary_coded" not in events_df.columns:
        print("Events with 'pain_binary_coded' required for ROI contrasts; skipping.")
        return

    for roi, tfr in roi_tfrs.items():
        try:
            n_epochs = tfr.data.shape[0]
            n_meta = len(events_df)
            n = min(n_epochs, n_meta)
            if n_epochs != n_meta:
                print(f"ROI {roi}: trimming to {n} epochs to match events.")

            if getattr(tfr, "metadata", None) is not None and "pain_binary_coded" in tfr.metadata.columns:
                pain_vec = pd.to_numeric(tfr.metadata.iloc[:n]["pain_binary_coded"], errors="coerce").fillna(0).astype(int).values
            else:
                pain_vec = pd.to_numeric(events_df.iloc[:n]["pain_binary_coded"], errors="coerce").fillna(0).astype(int).values
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
                bands: Dict[str, Tuple[float, float]] = {
                    "alpha": BAND_BOUNDS["alpha"],
                    "beta": BAND_BOUNDS["beta"],
                    "gamma": (BAND_BOUNDS["gamma"][0], fmax_available if BAND_BOUNDS["gamma"][1] is None else BAND_BOUNDS["gamma"][1]),
                }
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


def main(subject: str = "001", task: str = DEFAULT_TASK,
         plateau_tmin: float = DEFAULT_PLATEAU_TMIN, plateau_tmax: float = DEFAULT_PLATEAU_TMAX,
         temperature_strategy: str = DEFAULT_TEMPERATURE_STRATEGY) -> None:
    plots_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots"
    _ensure_dir(plots_dir)

    # Load cleaned epochs
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None or not epo_path.exists():
        print(f"Error: cleaned epochs file not found for sub-{subject}, task-{task} under {DERIV_ROOT}.")
        sys.exit(1)
    print(f"Loading epochs: {epo_path}")
    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)

    # Load events and align lengths; attach as metadata for consistency
    events_df = _load_events_df(subject, task)
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
                    if len(events_df) != len(epochs):
                        print("Aligned metadata using epochs.selection.")
            except Exception as e:
                print(f"Selection-based alignment failed: {e}")

        if not aligned and "sample" in events_df.columns and isinstance(getattr(epochs, "events", None), np.ndarray):
            try:
                samples = epochs.events[:, 0]
                events_by_sample = events_df.set_index("sample")
                events_aligned = events_by_sample.reindex(samples)
                if len(events_aligned) == len(epochs) and not events_aligned.isna().all(axis=1).any():
                    events_df = events_aligned.reset_index()
                    epochs.metadata = events_df.copy()
                    aligned = True
                    print("Aligned metadata using 'sample' column to epochs.events.")
            except Exception as e:
                print(f"Sample-based alignment failed: {e}")

        if not aligned:
            n = min(len(events_df), len(epochs))
            if len(events_df) != len(epochs):
                print(f"Warning: events rows ({len(events_df)}) != epochs ({len(epochs)}); trimming to {n}.")
            if len(epochs) != n:
                epochs = epochs[:n]
            events_df = events_df.iloc[:n].reset_index(drop=True)
            try:
                epochs.metadata = events_df.copy()
            except Exception as e:
                print(f"Warning: failed to attach metadata to epochs: {e}")
    else:
        print("Warning: events.tsv missing; contrasts will be skipped if needed.")

    # Compute per-trial TFR using Morlet wavelets
    # Frequencies covering alpha/beta/gamma using CONFIG
    freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQS)
    n_cycles = freqs * N_CYCLES_FACTOR  # proportional cycles per frequency
    print("Computing per-trial TFR (Morlet)...")
    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=TFR_DECIM,
        picks=TFR_PICKS,
        n_jobs=-1,
    )
    # power is EpochsTFR
    print(f"Computed TFR: type={type(power).__name__}, n_epochs={power.data.shape[0]}, n_freqs={len(power.freqs)}")

    # Optionally run pooled (no temperature discrimination)
    if temperature_strategy in ("pooled", "both"):
        # All trials, Cz TFR (baseline-corrected using pre-stim interval)
        plot_cz_all_trials(power, plots_dir, baseline=BASELINE)

        # Contrasts and topomaps (only if we have events)
        contrast_pain_nonpain(
            tfr=power,
            events_df=events_df,
            out_dir=plots_dir,
            baseline=BASELINE,
            plateau_window=(plateau_tmin, plateau_tmax),
        )

        # Per-ROI analysis: compute ROI EpochsTFRs from channel-averaged epochs
        print("Building ROIs and computing ROI TFRs (pooled)...")
        roi_map = _build_rois(epochs.info)
        if len(roi_map) == 0:
            print("No ROI channels found in montage; skipping ROI analysis.")
        else:
            roi_tfrs = compute_roi_tfrs(epochs, freqs=freqs, n_cycles=n_cycles, roi_map=roi_map)
            # Plot per-ROI all trials
            plot_rois_all_trials(roi_tfrs, plots_dir, baseline=BASELINE)
            # Per-ROI contrasts
            contrast_pain_nonpain_rois(roi_tfrs, events_df, plots_dir, baseline=BASELINE)
            # Per-ROI topomaps (all trials)
            plot_topomaps_rois_all_trials(power, roi_map, plots_dir, baseline=BASELINE, plateau_window=(plateau_tmin, plateau_tmax))
            # Per-ROI topomap contrasts
            contrast_pain_nonpain_topomaps_rois(power, events_df, roi_map, plots_dir, baseline=BASELINE, plateau_window=(plateau_tmin, plateau_tmax))

    # Optionally run per-temperature
    temp_col = _find_temperature_column(events_df)
    if temperature_strategy in ("per", "both"):
        if events_df is None or temp_col is None:
            print("Per-temperature analysis requested, but no temperature column found; skipping per-temperature plots.")
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
                print("No temperatures found in events; skipping per-temperature plots.")
            else:
                print(f"Running per-temperature analysis for {len(temps)} level(s): {temps}")
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
                    print(f"Computing TFR for temperature {tval} ({n_sel} trials)...")
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
                    )

                    # ROI analyses for this temperature
                    if len(roi_map_all) == 0:
                        print("No ROI channels found; skipping ROI analyses for temperature subset.")
                    else:
                        roi_tfrs_t = compute_roi_tfrs(epochs_t, freqs=freqs, n_cycles=n_cycles, roi_map=roi_map_all)
                        plot_rois_all_trials(roi_tfrs_t, plots_dir_t, baseline=BASELINE)
                        contrast_pain_nonpain_rois(roi_tfrs_t, events_t, plots_dir_t, baseline=BASELINE)
                        plot_topomaps_rois_all_trials(power_t, roi_map_all, plots_dir_t, baseline=BASELINE, plateau_window=(plateau_tmin, plateau_tmax))
                        contrast_pain_nonpain_topomaps_rois(power_t, events_t, roi_map_all, plots_dir_t, baseline=BASELINE, plateau_window=(plateau_tmin, plateau_tmax))

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Time-frequency analysis and contrasts (alpha/beta/gamma) for one subject")
    parser.add_argument("--subject", "-s", type=str, default="001", help="BIDS subject label without 'sub-' prefix (e.g., 001)")
    parser.add_argument("--task", "-t", type=str, default=DEFAULT_TASK, help="BIDS task label (default from config)")
    parser.add_argument("--plateau_tmin", type=float, default=DEFAULT_PLATEAU_TMIN, help="Plateau window start in seconds relative to stimulus onset")
    parser.add_argument("--plateau_tmax", type=float, default=DEFAULT_PLATEAU_TMAX, help="Plateau window end in seconds relative to stimulus onset")
    parser.add_argument(
        "--temperature_strategy",
        type=str,
        choices=["pooled", "per", "both"],
        default=DEFAULT_TEMPERATURE_STRATEGY,
        help="Whether to analyze pooled across temperatures, per temperature, or both",
    )
    args = parser.parse_args()

    main(subject=args.subject, task=args.task, plateau_tmin=args.plateau_tmin, plateau_tmax=args.plateau_tmax,
         temperature_strategy=args.temperature_strategy)
