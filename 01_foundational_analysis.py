import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath

# Load centralized configuration
from config_loader import load_config, get_legacy_constants

config = load_config()
config.setup_matplotlib()

# Extract legacy constants
_constants = get_legacy_constants(config)

PROJECT_ROOT = _constants["PROJECT_ROOT"]
BIDS_ROOT = _constants["BIDS_ROOT"]
DERIV_ROOT = _constants["DERIV_ROOT"]
TASK = _constants["TASK"]
DEFAULT_TASK = TASK
FIG_DPI = _constants["FIG_DPI"]
ERP_PICKS = _constants["ERP_PICKS"]
PAIN_COLUMNS = _constants["PAIN_COLUMNS"]
TEMPERATURE_COLUMNS = _constants["TEMPERATURE_COLUMNS"]

# Extract parameters from config
FIG_PAD_INCH = config.visualization.pad_inches
BBOX_INCHES = config.visualization.bbox_inches
PAIN_COLOR = config.analysis.erp.pain_color
NONPAIN_COLOR = config.analysis.erp.nonpain_color
INCLUDE_TMAX_IN_CROP = config.analysis.erp.include_tmax_in_crop
DEFAULT_CROP_TMIN = config.analysis.erp.default_crop_tmin
DEFAULT_CROP_TMAX = config.analysis.erp.default_crop_tmax
ERP_COMBINE = config.analysis.erp.combine
PLOTS_SUBDIR = config.analysis.erp.plots_subdir
LOG_FILE_NAME = config.logging.file_names.foundational
COUNTS_FILE_NAME = config.analysis.erp.counts_file_name
ERP_OUTPUT_FILES = dict(config.analysis.erp.output_files)


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


def _load_events_df(subject: str, task: str, logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
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
    """Set up logging with console and file handlers for foundational analysis."""
    logger = logging.getLogger(f"foundational_analysis_sub_{subject}")
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


def _save_fig(fig: plt.Figure, out_dir: Path, name: str, logger: Optional[logging.Logger] = None) -> None:
    _ensure_dir(out_dir)
    out_path = out_dir / name
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches=BBOX_INCHES, pad_inches=FIG_PAD_INCH)
    plt.close(fig)
    msg = f"Saved: {out_path}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def _maybe_crop_epochs(
    epochs: mne.Epochs, crop_tmin: Optional[float], crop_tmax: Optional[float], logger: Optional[logging.Logger] = None
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
    msg = f"Cropping epochs to [{tmin:.3f}, {tmax:.3f}] s (include_tmax={INCLUDE_TMAX_IN_CROP})"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    ep = epochs.copy()
    # Cropping modifies epoch data; ensure it is loaded into memory
    if not getattr(ep, "preload", False):
        ep.load_data()
    return ep.crop(tmin=tmin, tmax=tmax, include_tmax=INCLUDE_TMAX_IN_CROP)


def erp_contrast_pain(
    epochs: mne.Epochs, out_dir: Path, logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, mne.Evoked]]:
    # Prefer MNE metadata-based selection
    if epochs.metadata is None:
        msg = "ERP pain contrast: epochs.metadata is missing; skipping."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None
    col = None
    for candidate in PAIN_COLUMNS:
        if candidate in epochs.metadata.columns:
            col = candidate
            break
    if col is None:
        msg = "ERP pain contrast: No pain column found in metadata. Skipping."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None

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
        msg = "ERP pain contrast: one of the groups has zero trials; skipping."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None

    ev_pain = ep_pain.average(picks=ERP_PICKS)
    ev_non = ep_non.average(picks=ERP_PICKS)

    # GFP contrast
    try:
        fig = mne.viz.plot_compare_evokeds(
            {"painful": ev_pain, "non-painful": ev_non},
            picks=ERP_PICKS,
            combine=ERP_COMBINE,
            show=False,
            colors={"painful": PAIN_COLOR, "non-painful": NONPAIN_COLOR},
        )
        if isinstance(fig, list):
            fig = fig[0]
        _save_fig(fig, out_dir, ERP_OUTPUT_FILES["pain_gfp"], logger)
    except Exception as e:
        msg = f"ERP pain contrast (GFP) failed: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)

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
        _save_fig(fig, out_dir, ERP_OUTPUT_FILES["pain_butterfly"], logger)
    except Exception as e:
        msg = f"ERP pain contrast (butterfly) failed: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)

    return {"painful": ev_pain, "non-painful": ev_non}


def erp_by_temperature(
    epochs: mne.Epochs, out_dir: Path, logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, mne.Evoked]]:
    if epochs.metadata is None:
        msg = "ERP by temperature: epochs.metadata is missing; skipping."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None
    col = None
    for candidate in TEMPERATURE_COLUMNS:
        if candidate in epochs.metadata.columns:
            col = candidate
            break
    if col is None:
        msg = "ERP by temperature: No temperature column found in metadata. Skipping."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None

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
            msg = f"Temperature level {lvl}: selection/averaging failed: {e}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)

    if len(evokeds) == 0:
        msg = "ERP by temperature: No evokeds computed; skipping plot."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None

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
            _save_fig(fig, out_dir, ERP_OUTPUT_FILES["temp_butterfly_template"].format(label=safe_label), logger)
        except Exception as e:
            msg = f"Per-temperature plot failed for {label}: {e}"
            if logger:
                logger.error(msg)
            else:
                print(msg)

    # Plot GFP across levels
    try:
        fig = mne.viz.plot_compare_evokeds(evokeds, picks=ERP_PICKS, combine=ERP_COMBINE, show=False)
        if isinstance(fig, list):
            fig = fig[0]
        _save_fig(fig, out_dir, ERP_OUTPUT_FILES["temp_gfp"], logger)
    except Exception as e:
        msg = f"ERP by temperature (GFP) failed: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)

    return evokeds


def process_subject(
    subject: str,
    task: str = DEFAULT_TASK,
    crop_tmin: Optional[float] = DEFAULT_CROP_TMIN,
    crop_tmax: Optional[float] = DEFAULT_CROP_TMAX,
) -> Optional[Dict[str, Any]]:
    logger = _setup_logging(subject)
    logger.info(f"=== Foundational analysis: sub-{subject}, task-{task} ===")
    # Resolve paths
    plots_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots" / PLOTS_SUBDIR
    _ensure_dir(plots_dir)

    # Load epochs
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None or not epo_path.exists():
        msg = f"Error: could not find cleaned epochs file for sub-{subject}, task-{task} under {DERIV_ROOT}."
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
    epochs = mne.read_epochs(epo_path, preload=False, verbose=False)

    # Load events dataframe and attach as metadata (MNE-native selection uses this)
    events_df = _load_events_df(subject, task, logger)
    if events_df is None:
        msg = "Warning: events TSV not found; ERP contrasts will be skipped."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
    else:
        msg = f"Loaded events: {len(events_df)} rows"
        if logger:
            logger.info(msg)
        else:
            print(msg)
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
                        msg = f"Aligned metadata using epochs.selection (kept {len(epochs)} of {len(events_df)} events)."
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
                # Reindex to epochs' event sample order
                events_aligned = events_by_sample.reindex(samples)
                # If any rows are completely missing, abort this method
                if len(events_aligned) == len(epochs) and not events_aligned.isna().all(axis=1).any():
                    epochs.metadata = events_aligned.reset_index()
                    aligned = True
                    msg = "Aligned metadata using 'sample' column to epochs.events."
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
            # Fallback: naive min-length trimming in original order
            n = min(len(events_df), len(epochs))
            if len(events_df) != len(epochs):
                msg = f"Warning: events rows ({len(events_df)}) != epochs ({len(epochs)}); trimming to {n}."
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
            if len(epochs) != n:
                epochs = epochs[:n]
            epochs.metadata = events_df.iloc[:n].reset_index(drop=True)


    # Optional epoch time cropping prior to averaging/plotting
    if crop_tmin is not None or crop_tmax is not None:
        epochs = _maybe_crop_epochs(epochs, crop_tmin, crop_tmax, logger)

    pain_evokeds = None
    temp_evokeds = None
    if events_df is not None and epochs.metadata is not None:
        pain_evokeds = erp_contrast_pain(epochs, plots_dir, logger)
        temp_evokeds = erp_by_temperature(epochs, plots_dir, logger)

    msg = "Done."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return {"subject": subject, "pain_evokeds": pain_evokeds, "temp_evokeds": temp_evokeds}


def aggregate_group_level(results: List[Dict[str, Any]]) -> None:
    """Grand-average ERP data across subjects and generate group plots."""
    if not results:
        return
    out_dir = DERIV_ROOT / "group" / "eeg" / "plots" / PLOTS_SUBDIR
    _ensure_dir(out_dir)

    pain_painful: List[mne.Evoked] = []
    pain_non: List[mne.Evoked] = []
    temp_map: Dict[str, List[mne.Evoked]] = {}
    for res in results:
        pe = res.get("pain_evokeds") or {}
        if "painful" in pe and "non-painful" in pe:
            pain_painful.append(pe["painful"])
            pain_non.append(pe["non-painful"])
        te = res.get("temp_evokeds") or {}
        for label, evk in te.items():
            temp_map.setdefault(label, []).append(evk)

    if pain_painful and pain_non:
        g_pain = mne.grand_average(pain_painful)
        g_non = mne.grand_average(pain_non)
        try:
            fig = mne.viz.plot_compare_evokeds(
                {"painful": g_pain, "non-painful": g_non},
                picks=ERP_PICKS,
                combine=ERP_COMBINE,
                show=False,
                colors={"painful": PAIN_COLOR, "non-painful": NONPAIN_COLOR},
            )
            if isinstance(fig, list):
                fig = fig[0]
            _save_fig(fig, out_dir, "group_pain_gfp.png")
        except Exception:
            pass
        try:
            fig = mne.viz.plot_compare_evokeds(
                {"painful": g_pain, "non-painful": g_non},
                picks=ERP_PICKS,
                combine=None,
                show=False,
                colors={"painful": PAIN_COLOR, "non-painful": NONPAIN_COLOR},
            )
            if isinstance(fig, list):
                fig = fig[0]
            _save_fig(fig, out_dir, "group_pain_butterfly.png")
        except Exception:
            pass

    if temp_map:
        grand_temps: Dict[str, mne.Evoked] = {
            label: mne.grand_average(evs) for label, evs in temp_map.items()
        }
        for label, evk in grand_temps.items():
            try:
                fig = evk.plot(picks=ERP_PICKS, spatial_colors=True, show=False)
                try:
                    fig.suptitle(f"ERP - Temperature {label}")
                except Exception:
                    pass
                safe_label = (
                    str(label).replace(" ", "_").replace("/", "-").replace("\\", "-")
                )
                _save_fig(fig, out_dir, f"group_temp_{safe_label}_butterfly.png")
            except Exception:
                pass
        try:
            fig = mne.viz.plot_compare_evokeds(
                grand_temps,
                picks=ERP_PICKS,
                combine=ERP_COMBINE,
                show=False,
            )
            if isinstance(fig, list):
                fig = fig[0]
            _save_fig(fig, out_dir, "group_temp_gfp.png")
        except Exception:
            pass


def main(
    subjects: Optional[List[str]] = None,
    task: str = DEFAULT_TASK,
    crop_tmin: Optional[float] = DEFAULT_CROP_TMIN,
    crop_tmax: Optional[float] = DEFAULT_CROP_TMAX,
    do_group: bool = False,
) -> None:
    if subjects is None or subjects == ["all"]:
        subs = getattr(config, "subjects", [])
        if not subs:
            raise ValueError(
                "No subjects provided and config.project.subjects is empty."
            )
        subjects = subs

    results: List[Dict[str, Any]] = []
    for sub in subjects:
        res = process_subject(sub, task, crop_tmin=crop_tmin, crop_tmax=crop_tmax)
        if res:
            results.append(res)

    if do_group or len(results) > 1:
        aggregate_group_level(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Foundational EEG ERP analysis")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject IDs (e.g., 001 002) or 'all' for all configured subjects",
    )
    parser.add_argument("--task", "-t", type=str, default=DEFAULT_TASK, help="BIDS task label")
    parser.add_argument("--crop-tmin", type=float, default=DEFAULT_CROP_TMIN, help="Epoch crop start")
    parser.add_argument("--crop-tmax", type=float, default=DEFAULT_CROP_TMAX, help="Epoch crop end")
    parser.add_argument(
        "--do-group",
        action="store_true",
        help="Force grand-average ERPs even for a single subject (default when multiple subjects)",
    )
    args = parser.parse_args()

    main(subjects=args.subjects, task=args.task, crop_tmin=args.crop_tmin, crop_tmax=args.crop_tmax, do_group=args.do_group)
