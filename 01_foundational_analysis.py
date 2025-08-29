import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath

# Resolve project config
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


# ==========================
# CONFIG
# Centralized, user-tunable parameters
# ==========================
# Default task for CLI and main()
DEFAULT_TASK = TASK

# Figure saving
FIG_DPI = 200
FIG_PAD_INCH = 0.2

# Plotting picks
ERP_PICKS = "eeg"

# Event/metadata column candidates
PAIN_COLUMNS = ["pain_binary_coded", "pain_binary", "pain"]
TEMPERATURE_COLUMNS = ["stimulus_temp", "temperature", "temp_level"]

# Colors for ERP contrasts
PAIN_COLOR = "crimson"
NONPAIN_COLOR = "navy"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
    # Use BIDSPath to resolve events.tsv
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
        # Fallback to literal
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


def _save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    _ensure_dir(out_dir)
    out_path = out_dir / name
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=FIG_PAD_INCH)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _maybe_crop_epochs(
    epochs: mne.Epochs, crop_tmin: Optional[float], crop_tmax: Optional[float]
) -> mne.Epochs:
    """Optionally crop epochs in time.

    Notes
    -----
    - Uses include_tmax=False to drop the terminal sample, which helps avoid
      boundary artifacts at the exact epoch end (e.g., stimulus offset).
    - If a bound is None, the existing epochs.tmin/tmax is kept.
    """
    if crop_tmin is None and crop_tmax is None:
        return epochs
    tmin = epochs.tmin if crop_tmin is None else float(crop_tmin)
    tmax = epochs.tmax if crop_tmax is None else float(crop_tmax)
    # Ensure valid order
    if tmax <= tmin:
        raise ValueError(f"Invalid crop window: tmin={tmin}, tmax={tmax}")
    print(f"Cropping epochs to [{tmin:.3f}, {tmax:.3f}] s (include_tmax=False)")
    ep = epochs.copy()
    # Cropping modifies epoch data; ensure it is loaded into memory
    if not getattr(ep, "preload", False):
        ep.load_data()
    return ep.crop(tmin=tmin, tmax=tmax, include_tmax=False)


def erp_contrast_pain(epochs: mne.Epochs, out_dir: Path) -> None:
    # Prefer MNE metadata-based selection
    if epochs.metadata is None:
        print("ERP pain contrast: epochs.metadata is missing; skipping.")
        return
    col = None
    for candidate in PAIN_COLUMNS:
        if candidate in epochs.metadata.columns:
            col = candidate
            break
    if col is None:
        print("ERP pain contrast: No pain column found in metadata. Skipping.")
        return

    # Build selections using MNE metadata query
    try:
        ep_pain = epochs[f"{col} == 1"]
    except Exception:
        # In case types require casting
        ep_pain = epochs[np.asarray(pd.to_numeric(epochs.metadata[col], errors="coerce") == 1)]
    try:
        ep_non = epochs[f"{col} == 0"]
    except Exception:
        ep_non = epochs[np.asarray(pd.to_numeric(epochs.metadata[col], errors="coerce") == 0)]

    if len(ep_pain) == 0 or len(ep_non) == 0:
        print("ERP pain contrast: one of the groups has zero trials; skipping.")
        return

    ev_pain = ep_pain.average(picks=ERP_PICKS)
    ev_non = ep_non.average(picks=ERP_PICKS)

    # GFP contrast
    try:
        fig = mne.viz.plot_compare_evokeds(
            {"painful": ev_pain, "non-painful": ev_non},
            picks=ERP_PICKS,
            combine="gfp",
            show=False,
            colors={"painful": PAIN_COLOR, "non-painful": NONPAIN_COLOR},
        )
        if isinstance(fig, list):
            fig = fig[0]
        _save_fig(fig, out_dir, "erp_pain_binary_gfp.png")
    except Exception as e:
        print(f"ERP pain contrast (GFP) failed: {e}")

    # Butterfly overlay
    try:
        fig = mne.viz.plot_compare_evokeds(
            {"painful": ev_pain, "non-painful": ev_non},
            picks=ERP_PICKS,
            combine=None,
            show=False,
            colors={"painful": PAIN_COLOR, "non-painful": NONPAIN_COLOR},
        )
        if isinstance(fig, list):
            fig = fig[0]
        _save_fig(fig, out_dir, "erp_pain_binary_butterfly.png")
    except Exception as e:
        print(f"ERP pain contrast (butterfly) failed: {e}")


def erp_by_temperature(epochs: mne.Epochs, out_dir: Path) -> None:
    if epochs.metadata is None:
        print("ERP by temperature: epochs.metadata is missing; skipping.")
        return
    col = None
    for candidate in TEMPERATURE_COLUMNS:
        if candidate in epochs.metadata.columns:
            col = candidate
            break
    if col is None:
        print("ERP by temperature: No temperature column found in metadata. Skipping.")
        return

    # Determine unique, sensibly sorted levels
    levels_series = epochs.metadata[col]
    try:
        numeric_levels = pd.to_numeric(levels_series, errors="coerce")
        if numeric_levels.notna().all():
            uniq_sorted = np.sort(numeric_levels.unique())
            represent = {v: str(int(v)) if float(v).is_integer() else str(v) for v in uniq_sorted}
            use_numeric = True
        else:
            raise ValueError
    except Exception:
        uniq_sorted = sorted(levels_series.astype(str).unique())
        use_numeric = False


    evokeds: Dict[str, mne.Evoked] = {}
    for lvl in uniq_sorted:
        if use_numeric:
            query = f"{col} == {lvl}"
            label = represent[lvl]
        else:
            # Escape quotes if present in string
            lvl_str = str(lvl).replace('"', '\\"')
            query = f"{col} == \"{lvl_str}\""
            label = str(lvl)
        try:
            evokeds[label] = epochs[query].average(picks=ERP_PICKS)
        except Exception as e:
            print(f"Temperature level {lvl}: selection/averaging failed: {e}")

    if len(evokeds) == 0:
        print("ERP by temperature: No evokeds computed; skipping plot.")
        return

    # One plot per temperature level (Evoked butterfly) with title
    for label, evk in evokeds.items():
        try:
            fig = evk.plot(picks=ERP_PICKS, spatial_colors=True, show=False)
            try:
                fig.suptitle(f"ERP - Temperature {label}")
            except Exception:
                pass
            safe_label = (
                str(label)
                .replace(" ", "_")
                .replace("/", "-")
                .replace("\\", "-")
                .replace(".", "p")
                .replace(":", "-")
            )
            _save_fig(fig, out_dir, f"erp_temperature_{safe_label}_butterfly.png")
        except Exception as e:
            print(f"Per-temperature plot failed for {label}: {e}")

    # Plot GFP across levels
    try:
        fig = mne.viz.plot_compare_evokeds(evokeds, picks=ERP_PICKS, combine="gfp", show=False)
        if isinstance(fig, list):
            fig = fig[0]
        _save_fig(fig, out_dir, "erp_by_temperature_gfp.png")
    except Exception as e:
        print(f"ERP by temperature (GFP) failed: {e}")


def main(
    subject: str = "001",
    task: str = DEFAULT_TASK,
    crop_tmin: Optional[float] = None,
    crop_tmax: Optional[float] = None,
) -> None:
    # Resolve paths
    plots_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots"
    _ensure_dir(plots_dir)

    # Load epochs
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None or not epo_path.exists():
        print(f"Error: could not find cleaned epochs file for sub-{subject}, task-{task} under {DERIV_ROOT}.")
        sys.exit(1)
    print(f"Loading epochs: {epo_path}")
    epochs = mne.read_epochs(epo_path, preload=False, verbose=False)

    # Load events dataframe and attach as metadata (MNE-native selection uses this)
    events_df = _load_events_df(subject, task)
    if events_df is None:
        print("Warning: events TSV not found; ERP contrasts will be skipped.")
    else:
        print(f"Loaded events: {len(events_df)} rows")
        # Align events to epochs using selection or sample index; fallback to trimming
        aligned = False
        sel = getattr(epochs, "selection", None)
        if sel is not None and len(sel) == len(epochs):
            try:
                if len(events_df) > int(np.max(sel)):
                    events_aligned = events_df.iloc[sel].reset_index(drop=True)
                    epochs.metadata = events_aligned
                    aligned = True
                    if len(events_df) != len(epochs):
                        print(
                            f"Aligned metadata using epochs.selection (kept {len(epochs)} of {len(events_df)} events)."
                        )
            except Exception as e:
                print(f"Selection-based alignment failed: {e}")

        if not aligned and "sample" in events_df.columns and isinstance(getattr(epochs, "events", None), np.ndarray):
            try:
                samples = epochs.events[:, 0]
                events_by_sample = events_df.set_index("sample")
                # Reindex to epochs' event sample order
                events_aligned = events_by_sample.reindex(samples)
                # If any rows are completely missing, abort this method
                if len(events_aligned) == len(epochs) and not events_aligned.isna().all(axis=1).any():
                    epochs.metadata = events_aligned.reset_index()
                    aligned = True
                    print("Aligned metadata using 'sample' column to epochs.events.")
            except Exception as e:
                print(f"Sample-based alignment failed: {e}")

        if not aligned:
            # Fallback: naive min-length trimming in original order
            n = min(len(events_df), len(epochs))
            if len(events_df) != len(epochs):
                print(f"Warning: events rows ({len(events_df)}) != epochs ({len(epochs)}); trimming to {n}.")
            if len(epochs) != n:
                epochs = epochs[:n]
            epochs.metadata = events_df.iloc[:n].reset_index(drop=True)

        # Also save counts for pain if available
        for candidate in ["pain_binary_coded", "pain_binary", "pain"]:
            if candidate in epochs.metadata.columns:
                try:
                    _save_counts_tsv(pd.to_numeric(epochs.metadata[candidate], errors="ignore"), plots_dir, "counts_pain.tsv")
                except Exception as e:
                    print(f"Saving counts_pain.tsv failed: {e}")
                break

    # Optional epoch time cropping prior to averaging/plotting
    if crop_tmin is not None or crop_tmax is not None:
        epochs = _maybe_crop_epochs(epochs, crop_tmin, crop_tmax)

    # ERP: pain vs non-pain and by temperature (requires metadata)
    if events_df is not None and epochs.metadata is not None:
        erp_contrast_pain(epochs, plots_dir)
        erp_by_temperature(epochs, plots_dir)

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Foundational EEG QC and ERP analysis for one subject")
    parser.add_argument("--subject", "-s", type=str, default="001", help="BIDS subject label without 'sub-' prefix (e.g., 001)")
    parser.add_argument("--task", "-t", type=str, default=DEFAULT_TASK, help="BIDS task label (default from config)")
    parser.add_argument("--crop-tmin", type=float, default=None, help="Optional epoch crop start time in seconds")
    parser.add_argument("--crop-tmax", type=float, default=None, help="Optional epoch crop end time in seconds (excluded)")
    args = parser.parse_args()

    main(subject=args.subject, task=args.task, crop_tmin=args.crop_tmin, crop_tmax=args.crop_tmax)
