from __future__ import annotations

"""
Aperiodic (1/f) control and corrected band power features.

Per-subject:
- Compute per-trial pre-stimulus PSD (Welch) per channel.
- Fit log10(PSD) ~ a + b*log10(f) to estimate aperiodic offset (a) and slope (b).
- Compute aperiodic-corrected band power as residual mean within each band.
- Save ML-ready features to features folder and basic stats/plots.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy import stats

from utils.config_loader import load_config, get_legacy_constants
from utils.io_utils import (
    _find_clean_epochs_path as _find_clean_epochs_path,
    _load_events_df as _load_events_df,
    _align_events_to_epochs as _align_events_to_epochs,
    _pick_target_column as _pick_target_column,
)
_C = get_legacy_constants(config)

DERIV_ROOT: Path = _C["DERIV_ROOT"]
TASK: str = _C["TASK"]
FEATURES_FREQ_BANDS: Dict[str, Tuple[float, float]] = _C["FEATURES_FREQ_BANDS"]
RATING_COLUMNS: List[str] = _C["RATING_COLUMNS"]
SAVE_FORMATS = tuple(config.get("output.save_formats", ["png"]))
FIG_DPI = int(config.get("output.fig_dpi", 300))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plots_dir(subject: str) -> Path:
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots" / "08_aperiodic"


def _stats_dir(subject: str) -> Path:
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "stats" / "08_aperiodic"


def _features_dir(subject: str) -> Path:
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "features"


def _save_fig(fig: plt.Figure, base: Path) -> None:
    base = base.with_suffix("")
    for ext in SAVE_FORMATS:
        try:
            fig.savefig(base.with_suffix(f".{ext}"), dpi=FIG_DPI, bbox_inches="tight")
        except Exception:
            pass
    plt.close(fig)


def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _fit_aperiodic(logf: np.ndarray, logpsd: np.ndarray) -> Tuple[float, float]:
    # returns (offset a, slope b) for y = a + b*x
    try:
        b, a = np.polyfit(logf, logpsd, 1)  # note: np.polyfit returns [slope, intercept]
        return float(a), float(b)
    except Exception:
        return float("nan"), float("nan")


def analyze_subject(
    subject: str,
    task: str = TASK,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    fmin: float = 2.0,
    fmax: float = 40.0,
    n_fft: Optional[int] = None,
) -> Optional[Path]:
    plots_dir = _plots_dir(subject)
    stats_dir = _stats_dir(subject)
    feats_dir = _features_dir(subject)
    _ensure_dir(plots_dir); _ensure_dir(stats_dir); _ensure_dir(feats_dir)

    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None or not Path(epo_path).exists():
        print(f"No epochs for sub-{subject}, task-{task}")
        return None
    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)

    if baseline is None:
        bwin = config.get("time_frequency_analysis.baseline_window", [-0.5, -0.01])
        b_start = float(bwin[0])
        b_end = float(bwin[1]) if float(bwin[1]) <= 0 else 0.0
        baseline = (b_start, b_end)
    b_start, b_end = baseline

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        print("No EEG channels after picks")
        return None

    # Compute PSD per epoch/channel in baseline window
    psds, freqs = mne.time_frequency.psd_welch(
        epochs,
        picks=picks,
        fmin=float(fmin), fmax=float(fmax),
        tmin=b_start, tmax=b_end,
        n_fft=n_fft,
        n_overlap=None,
        average='mean',  # average over windows, not epochs
        verbose=False,
    )  # psds shape: (n_epochs, n_channels, n_freqs)
    if psds.ndim != 3:
        print("Unexpected PSD shape")
        return None
    eps = 1e-20
    logf = np.log10(freqs)
    logpsd = np.log10(np.maximum(psds, eps))

    n_ep, n_ch, n_fr = logpsd.shape
    # Fit aperiodic params per (epoch, channel)
    offset = np.full((n_ep, n_ch), np.nan)
    slope = np.full((n_ep, n_ch), np.nan)
    for i in range(n_ep):
        for j in range(n_ch):
            a, b = _fit_aperiodic(logf, logpsd[i, j, :])
            offset[i, j] = a
            slope[i, j] = b

    # Residual (aperiodic-corrected) spectra
    resid = np.empty_like(logpsd)
    for i in range(n_ep):
        for j in range(n_ch):
            resid[i, j, :] = logpsd[i, j, :] - (offset[i, j] + slope[i, j] * logf)

    # Band masks
    band_masks = {b: (freqs >= lo) & (freqs <= hi) for b, (lo, hi) in FEATURES_FREQ_BANDS.items()}
    # Build features per trial
    ch_names = [epochs.info["ch_names"][p] for p in picks]
    rows = []
    for i in range(n_ep):
        rec = {}
        # aperiodic params per channel
        for j, ch in enumerate(ch_names):
            rec[f"aper_slope_{ch}"] = float(slope[i, j])
            rec[f"aper_offset_{ch}"] = float(offset[i, j])
        # pow raw and corrected per band/channel
        for b, mask in band_masks.items():
            if not np.any(mask):
                for j, ch in enumerate(ch_names):
                    rec[f"powrawpre_{b}_{ch}"] = np.nan
                    rec[f"powcorr_{b}_{ch}"] = np.nan
            else:
                for j, ch in enumerate(ch_names):
                    rec[f"powrawpre_{b}_{ch}"] = float(np.mean(logpsd[i, j, mask]))
                    rec[f"powcorr_{b}_{ch}"] = float(np.mean(resid[i, j, mask]))
        rows.append(rec)
    feat_df = pd.DataFrame(rows)
    out_feats = feats_dir / "features_aperiodic.tsv"
    feat_df.to_csv(out_feats, sep="\t", index=False)

    # Behavior correlations (channel-averaged summaries)
    beh_corr = None
    events = _load_events_df(subject, task)
    aligned = _align_events_to_epochs(events, epochs)
    if aligned is not None:
        rating_col = _pick_first(aligned, RATING_COLUMNS)
        if rating_col is not None:
            y = pd.to_numeric(aligned[rating_col], errors="coerce").to_numpy()
            mask_valid = np.isfinite(y)
            # Slope/offset averaged across channels
            slope_mean = np.nanmean(slope, axis=1)
            offset_mean = np.nanmean(offset, axis=1)
            rows2 = []
            def _corr(a):
                try:
                    r, p = stats.spearmanr(a[mask_valid], y[mask_valid])
                    return float(r), float(p), int(mask_valid.sum())
                except Exception:
                    return np.nan, np.nan, 0
            r,p,n = _corr(slope_mean)
            rows2.append({"metric": "slope_mean", "r": r, "p": p, "n": n})
            r,p,n = _corr(offset_mean)
            rows2.append({"metric": "offset_mean", "r": r, "p": p, "n": n})
            # powcorr band avg across channels
            for b, mask in band_masks.items():
                if not np.any(mask):
                    continue
                band_resid = np.nanmean(resid[:, :, mask], axis=(1, 2))  # (n_ep,)
                r,p,n = _corr(band_resid)
                rows2.append({"metric": f"powcorr_{b}_mean", "r": r, "p": p, "n": n})
            beh_corr = pd.DataFrame(rows2)
            beh_corr["q"] = _fdr_bh(beh_corr["p"].to_numpy())
            out_beh = stats_dir / "aperiodic_behavior_correlations.tsv"
            beh_corr.to_csv(out_beh, sep="\t", index=False)
            # Plot bar chart of correlations
            try:
                fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(rows2)), 4.0))
                xs = np.arange(len(rows2))
                vals = [r["r"] for r in rows2]
                ax.bar(xs, vals, color="#1f77b4")
                ax.set_xticks(xs)
                ax.set_xticklabels([str(r["metric"]) for r in rows2], rotation=45, ha='right')
                ax.set_ylabel("Spearman r")
                ax.set_title("Aperiodic/Corrected vs rating (channel-avg)")
                # Mark significant
                for i, rec in enumerate(rows2):
                    if np.isfinite(rec["r"]) and beh_corr.iloc[i]["q"] < 0.05:
                        ax.text(i, vals[i] + 0.02*np.sign(vals[i] or 1), "*", ha="center", va="bottom")
                ax.axhline(0, color='k', lw=0.8)
                plt.tight_layout()
                _save_fig(fig, plots_dir / "aperiodic_behavior_bar")
            except Exception:
                pass

    # Topomaps for mean slope/offset
    picks_idx = picks
    try:
        slope_mean_ch = np.nanmean(slope, axis=0)
        offset_mean_ch = np.nanmean(offset, axis=0)
        fig1, ax1 = plt.subplots(1, 2, figsize=(9.0, 3.8))
        mne.viz.plot_topomap(slope_mean_ch, epochs.info, picks=picks_idx, axes=ax1[0], show=False, contours=6, cmap='RdBu_r')
        ax1[0].set_title('Aperiodic slope (mean)')
        mne.viz.plot_topomap(offset_mean_ch, epochs.info, picks=picks_idx, axes=ax1[1], show=False, contours=6, cmap='RdBu_r')
        ax1[1].set_title('Aperiodic offset (mean)')
        plt.tight_layout()
        _save_fig(fig1, plots_dir / "aperiodic_slope_offset_topomap")
    except Exception:
        pass

    print(f"Saved aperiodic features: {out_feats}")
    return out_feats


def _fdr_bh(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    try:
        from statsmodels.stats.multitest import fdrcorrection as _fdrcorrection
        _, q = _fdrcorrection(p, alpha=alpha)
        return q
    except Exception:
        # fallback BH
        p = np.asarray(p, dtype=float)
        n = p.size
        order = np.argsort(p)
        ranked = p[order]
        q = ranked * n / (np.arange(1, n + 1))
        for i in range(n - 2, -1, -1):
            q[i] = min(q[i], q[i + 1])
        out = np.empty_like(p)
        out[order] = q
        return out


def main():
    import argparse
    p = argparse.ArgumentParser(description="Aperiodic (1/f) analysis and corrected band power features.")
    p.add_argument("--subject", required=True, help="Subject ID (no 'sub-' prefix)")
    p.add_argument("--task", type=str, default=TASK, help="BIDS task label")
    p.add_argument("--baseline", nargs=2, type=float, default=None, help="Baseline window [start end] in seconds (end <= 0)")
    p.add_argument("--fmin", type=float, default=2.0)
    p.add_argument("--fmax", type=float, default=40.0)
    p.add_argument("--n-fft", type=int, default=None)
    args = p.parse_args()
    baseline = tuple(args.baseline) if args.baseline is not None else None
    analyze_subject(
        subject=args.subject,
        task=args.task,
        baseline=baseline,
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        n_fft=(int(args.n_fft) if args.n_fft is not None else None),
    )


if __name__ == "__main__":
    main()

