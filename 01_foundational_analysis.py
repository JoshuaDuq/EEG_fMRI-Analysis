import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath

# Strict alignment utilities
from alignment_utils import align_events_to_epochs_strict, validate_alignment

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
FIG_PAD_INCH = float(config.get("output.pad_inches", 0.02))
BBOX_INCHES = config.get("output.bbox_inches", "tight")
PAIN_COLOR = config.get("foundational_analysis.erp.pain_color", "crimson")
NONPAIN_COLOR = config.get("foundational_analysis.erp.nonpain_color", "navy")
INCLUDE_TMAX_IN_CROP = bool(config.get("foundational_analysis.erp.include_tmax_in_crop", False))
DEFAULT_CROP_TMIN = config.get("foundational_analysis.erp.default_crop_tmin", None)
DEFAULT_CROP_TMAX = config.get("foundational_analysis.erp.default_crop_tmax", None)
ERP_COMBINE = config.get("foundational_analysis.erp.combine", "gfp")
PLOTS_SUBDIR = config.get("foundational_analysis.erp.plots_subdir", "01_foundational_analysis")
LOG_FILE_NAME = config.get("logging.file_names.foundational", "01_foundational_analysis.log")
COUNTS_FILE_NAME = config.get("foundational_analysis.erp.counts_file_name", "counts_pain.tsv")
ERP_OUTPUT_FILES = dict(config.get("foundational_analysis.erp.output_files", {
    "pain_gfp": "erp_pain_binary_gfp.png",
    "pain_butterfly": "erp_pain_binary_butterfly.png",
    "temp_gfp": "erp_by_temperature_gfp.png",
    "temp_butterfly": "erp_by_temperature_butterfly.png",
    "temp_butterfly_template": "erp_by_temperature_butterfly_{label}.png",
}))


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


def _setup_logging(subject: Optional[str] = None) -> logging.Logger:
    """Set up logging with console and file handlers for foundational analysis."""
    if subject is None:
        logger_name = "foundational_analysis_group"
        log_dir = DERIV_ROOT / "group" / "eeg" / "logs"
    else:
        logger_name = f"foundational_analysis_sub_{subject}"
        log_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "logs"
    
    logger = logging.getLogger(logger_name)
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


def _get_available_subjects() -> List[str]:
    """Find all available subjects in DERIV_ROOT with cleaned epochs files."""
    subjects = []
    if not DERIV_ROOT.exists():
        return subjects
    
    for subj_dir in DERIV_ROOT.glob("sub-*"):
        if subj_dir.is_dir():
            subject_id = subj_dir.name[4:]  # Remove 'sub-' prefix
            # Check if cleaned epochs exist for this subject
            if _find_clean_epochs_path(subject_id, DEFAULT_TASK) is not None:
                subjects.append(subject_id)
    
    return sorted(subjects)


def process_single_subject(
    subject: str,
    task: str = DEFAULT_TASK,
    crop_tmin: Optional[float] = DEFAULT_CROP_TMIN,
    crop_tmax: Optional[float] = DEFAULT_CROP_TMAX,
    logger: Optional[logging.Logger] = None
) -> Tuple[Optional[mne.Epochs], bool]:
    """Process a single subject and return epochs and success status.
    
    Returns
    -------
    epochs : mne.Epochs or None
        The processed epochs, or None if processing failed
    success : bool
        True if processing completed successfully
    """
    if logger is None:
        logger = _setup_logging(subject)
    
    logger.info(f"=== Processing sub-{subject}, task-{task} ===")
    
    # Resolve paths
    plots_dir = DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots" / PLOTS_SUBDIR
    _ensure_dir(plots_dir)

    # Load epochs
    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None or not epo_path.exists():
        logger.error(f"Could not find cleaned epochs file for sub-{subject}, task-{task}")
        return None, False
    
    logger.info(f"Loading epochs: {epo_path}")
    
    try:
        epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
    except Exception as e:
        logger.error(f"Failed to load epochs for sub-{subject}: {e}")
        return None, False

    # Load events dataframe and attach as metadata
    events_df = _load_events_df(subject, task, logger)
    if events_df is None:
        logger.warning("Events TSV not found; ERP contrasts will be skipped.")
    else:
        logger.info(f"Loaded events: {len(events_df)} rows")
        try:
            events_aligned = align_events_to_epochs_strict(events_df, epochs, logger)
            if events_aligned is not None:
                epochs.metadata = events_aligned
                validate_alignment(events_aligned, epochs, logger)
        except ValueError as e:
            logger.error(f"Event alignment failed strictly: {e}")
            return None, False

    # Optional epoch time cropping prior to averaging/plotting
    if crop_tmin is not None or crop_tmax is not None:
        epochs = _maybe_crop_epochs(epochs, crop_tmin, crop_tmax, logger)

    # ERP: pain vs non-pain and by temperature (requires metadata)
    if events_df is not None and epochs.metadata is not None:
        erp_contrast_pain(epochs, plots_dir, logger)
        erp_by_temperature(epochs, plots_dir, logger)

    logger.info("Single subject processing completed.")
    return epochs, True


def group_erp_contrast_pain(all_epochs: List[mne.Epochs], out_dir: Path, logger: Optional[logging.Logger] = None) -> None:
    """Create group-level pain vs non-pain ERP contrasts using grand averaging."""
    if not all_epochs:
        return
    
    pain_evokeds = []
    nonpain_evokeds = []
    
    for epochs in all_epochs:
        if epochs.metadata is None:
            continue
            
        # Find pain column
        col = None
        for candidate in PAIN_COLUMNS:
            if candidate in epochs.metadata.columns:
                col = candidate
                break
        if col is None:
            continue
            
        # Build selections
        try:
            ep_pain = epochs[f"{col} == 1"]
            ep_non = epochs[f"{col} == 0"]
        except Exception:
            try:
                ep_pain = epochs[np.asarray(pd.to_numeric(epochs.metadata[col], errors="coerce") == 1)]
                ep_non = epochs[np.asarray(pd.to_numeric(epochs.metadata[col], errors="coerce") == 0)]
            except Exception:
                continue
        
        if len(ep_pain) > 0:
            pain_evokeds.append(ep_pain.average(picks=ERP_PICKS))
        if len(ep_non) > 0:
            nonpain_evokeds.append(ep_non.average(picks=ERP_PICKS))
    
    if len(pain_evokeds) == 0 or len(nonpain_evokeds) == 0:
        if logger:
            logger.warning("Group ERP pain contrast: insufficient data across subjects")
        return
    
    # Create grand averages (fallback visualization)
    grand_pain = mne.grand_average(pain_evokeds, interpolate_bads=True)
    grand_nonpain = mne.grand_average(nonpain_evokeds, interpolate_bads=True)
    
    # GFP contrast
    try:
        # Prefer CI shading across subjects when available (newer MNE)
        try:
            fig = mne.viz.plot_compare_evokeds(
                {"painful": pain_evokeds, "non-painful": nonpain_evokeds},
                picks=ERP_PICKS,
                combine=ERP_COMBINE,
                show=False,
                colors={"painful": PAIN_COLOR, "non-painful": NONPAIN_COLOR},
                ci=0.95,
            )
        except TypeError:
            # Older MNE without 'ci' or list handling for CI
            fig = mne.viz.plot_compare_evokeds(
                {"painful": grand_pain, "non-painful": grand_nonpain},
                picks=ERP_PICKS,
                combine=ERP_COMBINE,
                show=False,
                colors={"painful": PAIN_COLOR, "non-painful": NONPAIN_COLOR},
            )
        if isinstance(fig, list):
            fig = fig[0]
        fig.suptitle(f"Group ERP: Pain vs Non-Pain (N={len(pain_evokeds)} subjects)", fontsize=14, fontweight='bold')
        _save_fig(fig, out_dir, "group_" + ERP_OUTPUT_FILES["pain_gfp"], logger)
    except Exception as e:
        if logger:
            logger.error(f"Group ERP pain contrast (GFP) failed: {e}")

    # Butterfly overlay
    try:
        fig = mne.viz.plot_compare_evokeds(
            {"painful": grand_pain, "non-painful": grand_nonpain},
            picks=ERP_PICKS,
            combine=None,
            show=False,
            colors={"painful": PAIN_COLOR, "non-painful": NONPAIN_COLOR},
        )
        if isinstance(fig, list):
            fig = fig[0]
        fig.suptitle(f"Group ERP: Pain vs Non-Pain (N={len(pain_evokeds)} subjects)", fontsize=14, fontweight='bold')
        _save_fig(fig, out_dir, "group_" + ERP_OUTPUT_FILES["pain_butterfly"], logger)
    except Exception as e:
        if logger:
            logger.error(f"Group ERP pain contrast (butterfly) failed: {e}")


def group_erp_by_temperature(all_epochs: List[mne.Epochs], out_dir: Path, logger: Optional[logging.Logger] = None) -> None:
    """Create group-level temperature ERP contrasts using grand averaging."""
    if not all_epochs:
        return
    
    # Collect temperature levels across all subjects
    temp_evokeds_by_level: Dict[str, List[mne.Evoked]] = {}
    
    for epochs in all_epochs:
        if epochs.metadata is None:
            continue
            
        # Find temperature column
        col = None
        for candidate in TEMPERATURE_COLUMNS:
            if candidate in epochs.metadata.columns:
                col = candidate
                break
        if col is None:
            continue
            
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
        
        # Process each temperature level
        for lvl in uniq_sorted:
            if use_numeric:
                query = f"{col} == {lvl}"
                label = represent[lvl]
            else:
                lvl_str = str(lvl).replace('"', '\\"')
                query = f"{col} == \"{lvl_str}\""
                label = str(lvl)
            
            try:
                ep_temp = epochs[query]
                if len(ep_temp) > 0:
                    evoked = ep_temp.average(picks=ERP_PICKS)
                    if label not in temp_evokeds_by_level:
                        temp_evokeds_by_level[label] = []
                    temp_evokeds_by_level[label].append(evoked)
            except Exception as e:
                if logger:
                    logger.warning(f"Temperature level {lvl}: selection/averaging failed: {e}")
    
    if not temp_evokeds_by_level:
        if logger:
            logger.warning("Group ERP by temperature: No evokeds computed across subjects")
        return
    
    # Create grand averages for each temperature level
    grand_temp_evokeds: Dict[str, mne.Evoked] = {}
    for label, evokeds_list in temp_evokeds_by_level.items():
        if len(evokeds_list) > 0:
            grand_temp_evokeds[label] = mne.grand_average(evokeds_list, interpolate_bads=True)

    # Plot per-temperature butterfly (grand average across subjects for each level)
    try:
        for label, evk in grand_temp_evokeds.items():
            try:
                fig = evk.plot(picks=ERP_PICKS, spatial_colors=True, show=False)
                try:
                    fig.suptitle(f"Group ERP - Temperature {label}")
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
                _save_fig(fig, out_dir, "group_" + ERP_OUTPUT_FILES["temp_butterfly_template"].format(label=safe_label), logger)
            except Exception as e:
                if logger:
                    logger.warning(f"Group per-temperature butterfly failed for {label}: {e}")
    except Exception as e:
        if logger:
            logger.warning(f"Group per-temperature butterfly plotting failed: {e}")

    # Combined butterfly overlay across all temperature levels (grand averages)
    try:
        if len(grand_temp_evokeds) >= 2:
            fig = mne.viz.plot_compare_evokeds(
                grand_temp_evokeds, picks=ERP_PICKS, combine=None, show=False
            )
            if isinstance(fig, list):
                fig = fig[0]
            try:
                n_info = ", ".join([f"{k}: N={len(temp_evokeds_by_level[k])}" for k in grand_temp_evokeds.keys()])
                fig.suptitle(f"Group ERP by Temperature (Butterfly) — {n_info}", fontsize=14, fontweight='bold')
            except Exception:
                pass
            _save_fig(fig, out_dir, "group_" + ERP_OUTPUT_FILES["temp_butterfly"], logger)
    except Exception as e:
        if logger:
            logger.warning(f"Group combined temperature butterfly failed: {e}")

    # Plot GFP across temperature levels
    try:
        # Prefer CI shading across subjects when available (newer MNE)
        try:
            fig = mne.viz.plot_compare_evokeds(
                temp_evokeds_by_level, picks=ERP_PICKS, combine=ERP_COMBINE, show=False, ci=0.95
            )
        except TypeError:
            fig = mne.viz.plot_compare_evokeds(grand_temp_evokeds, picks=ERP_PICKS, combine=ERP_COMBINE, show=False)
        if isinstance(fig, list):
            fig = fig[0]
        n_subjects_info = ", ".join([f"{k}: N={len(v)}" for k, v in temp_evokeds_by_level.items()])
        fig.suptitle(f"Group ERP by Temperature ({n_subjects_info} subjects)", fontsize=14, fontweight='bold')
        _save_fig(fig, out_dir, "group_" + ERP_OUTPUT_FILES["temp_gfp"], logger)
    except Exception as e:
        if logger:
            logger.error(f"Group ERP by temperature (GFP) failed: {e}")


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


def _write_group_pain_counts(
    all_epochs: List[mne.Epochs], subjects: List[str], out_dir: Path, logger: Optional[logging.Logger] = None
) -> None:
    """Write per-subject trial counts for pain vs non-pain to TSV.

    The output includes columns: subject, n_pain, n_nonpain, n_total.
    Subjects missing metadata or pain columns are recorded with zeros.
    """
    rows = []
    for subj, epochs in zip(subjects, all_epochs):
        n_pain = 0
        n_non = 0
        if epochs.metadata is not None:
            col = None
            for candidate in PAIN_COLUMNS:
                if candidate in epochs.metadata.columns:
                    col = candidate
                    break
            if col is not None:
                vals = pd.to_numeric(epochs.metadata[col], errors="coerce")
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
    # Add summary row
    total = df[["n_pain", "n_nonpain", "n_total"]].sum()
    total_row = {"subject": "TOTAL", **{k: int(v) for k, v in total.to_dict().items()}}
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    _ensure_dir(out_dir)
    out_path = out_dir / COUNTS_FILE_NAME
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


def erp_contrast_pain(epochs: mne.Epochs, out_dir: Path, logger: Optional[logging.Logger] = None) -> None:
    # Prefer MNE metadata-based selection
    if epochs.metadata is None:
        msg = "ERP pain contrast: epochs.metadata is missing; skipping."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return
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
        msg = "ERP pain contrast: one of the groups has zero trials; skipping."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return

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


def erp_by_temperature(epochs: mne.Epochs, out_dir: Path, logger: Optional[logging.Logger] = None) -> None:
    if epochs.metadata is None:
        msg = "ERP by temperature: epochs.metadata is missing; skipping."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return
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
            _save_fig(fig, out_dir, ERP_OUTPUT_FILES["temp_butterfly_template"].format(label=safe_label), logger)
        except Exception as e:
            msg = f"Per-temperature plot failed for {label}: {e}"
            if logger:
                logger.error(msg)
            else:
                print(msg)

    # Combined butterfly overlay across all temperature levels
    try:
        if len(evokeds) >= 2:
            fig = mne.viz.plot_compare_evokeds(evokeds, picks=ERP_PICKS, combine=None, show=False)
            if isinstance(fig, list):
                fig = fig[0]
            try:
                fig.suptitle("ERP by Temperature (Butterfly)")
            except Exception:
                pass
            _save_fig(fig, out_dir, ERP_OUTPUT_FILES["temp_butterfly"], logger)
    except Exception as e:
        msg = f"ERP by temperature (combined butterfly) failed: {e}"
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


def main(
    subjects: Optional[List[str]] = None,
    all_subjects: bool = False,
    task: str = DEFAULT_TASK,
    crop_tmin: Optional[float] = DEFAULT_CROP_TMIN,
    crop_tmax: Optional[float] = DEFAULT_CROP_TMAX,
    group: Optional[str] = None,
) -> None:
    """Main function for foundational EEG ERP analysis.
    
    Supports both single-subject and multi-subject analysis.
    Creates individual subject plots plus group-level grand average ERPs.
    """
    # Determine subjects to process from CLI
    if group is not None:
        if group.strip().lower() in {"all", "*", "@all"}:
            subjects = _get_available_subjects()
            if not subjects:
                raise ValueError(f"No subjects with cleaned epochs found in {DERIV_ROOT}")
        else:
            # Parse comma/space separated labels
            cand = [s.strip() for s in group.replace(";", ",").replace(" ", ",").split(",") if s.strip()]
            if not cand:
                raise ValueError("--group provided but no valid subject labels parsed")
            # Filter to those that have data; warn for missing
            valid = []
            for s in cand:
                if _find_clean_epochs_path(s, task) is not None:
                    valid.append(s)
                else:
                    print(f"Warning: --group subject '{s}' has no cleaned epochs; skipping")
            subjects = valid
            if not subjects:
                raise ValueError("--group provided but no valid subjects with data found")
    else:
        if all_subjects:
            subjects = _get_available_subjects()
            if not subjects:
                raise ValueError(f"No subjects with cleaned epochs found in {DERIV_ROOT}")
        elif subjects is None or len(subjects) == 0:
            raise ValueError(
                "No subjects specified. Use --group all|A,B,C, or --subject (can repeat) or --all-subjects."
            )
    
    # Setup group logging
    group_logger = _setup_logging()
    group_logger.info(f"=== Multi-subject foundational analysis: {len(subjects)} subjects, task-{task} ===")
    group_logger.info(f"Subjects: {', '.join(subjects)}")
    
    # Process each subject individually
    all_epochs = []
    successful_subjects = []
    
    for subject in subjects:
        group_logger.info(f"--- Processing subject: {subject} ---")
        epochs, success = process_single_subject(subject, task, crop_tmin, crop_tmax, group_logger)
        if success and epochs is not None:
            all_epochs.append(epochs)
            successful_subjects.append(subject)
        else:
            group_logger.warning(f"Failed to process subject {subject}, excluding from group analysis")
    
    # Group-level analysis
    if len(all_epochs) >= 2:  # Need at least 2 subjects for meaningful group analysis
        group_logger.info(f"=== Group analysis: {len(successful_subjects)} successful subjects ===")
        group_plots_dir = DERIV_ROOT / "group" / "eeg" / "plots" / PLOTS_SUBDIR
        _ensure_dir(group_plots_dir)
        
        # Group ERP analyses
        group_erp_contrast_pain(all_epochs, group_plots_dir, group_logger)
        group_erp_by_temperature(all_epochs, group_plots_dir, group_logger)
        # Trial counts summary
        _write_group_pain_counts(all_epochs, successful_subjects, group_plots_dir, group_logger)
        
        group_logger.info(f"Group analysis completed. Results saved to: {group_plots_dir}")
    else:
        group_logger.warning(f"Only {len(all_epochs)} subjects processed successfully. Skipping group analysis.")
    
    group_logger.info(f"=== Analysis complete ===")
    if successful_subjects:
        group_logger.info(f"Successfully processed: {', '.join(successful_subjects)}")
    failed_subjects = set(subjects) - set(successful_subjects)
    if failed_subjects:
        group_logger.warning(f"Failed to process: {', '.join(failed_subjects)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Foundational EEG ERP analysis supporting single and multiple subjects")
    
    # Subject selection arguments (mutually exclusive)
    subject_group = parser.add_mutually_exclusive_group()
    subject_group.add_argument(
        "--group", type=str,
        help=(
            "Group of subjects to process: either 'all' or a comma/space-separated "
            "list of BIDS labels without 'sub-' (e.g., '0001,0002,0003')."
        ),
    )
    subject_group.add_argument(
        "--subject", "-s", type=str, action="append",
        help=(
            "BIDS subject label(s) without 'sub-' prefix (e.g., 0000). "
            "Can be specified multiple times for multiple subjects."
        ),
    )
    subject_group.add_argument(
        "--all-subjects", action="store_true",
        help="Process all available subjects with cleaned epochs files"
    )
    
    parser.add_argument("--task", "-t", type=str, default=DEFAULT_TASK, help="BIDS task label (default from config)")
    parser.add_argument("--crop-tmin", type=float, default=DEFAULT_CROP_TMIN, help="ERP epoch crop start time (s)")
    parser.add_argument("--crop-tmax", type=float, default=DEFAULT_CROP_TMAX, help="ERP epoch crop end time (s)")
    
    args = parser.parse_args()

    main(
        subjects=args.subject, 
        all_subjects=args.all_subjects, 
        task=args.task,
        crop_tmin=args.crop_tmin,
        crop_tmax=args.crop_tmax,
        group=args.group,
    )
