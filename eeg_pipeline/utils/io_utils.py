"""
Shared I/O and alignment utilities for EEG pipeline scripts.

Centralizes common helpers that were previously duplicated across modules
or dynamically imported from numbered scripts.

Functions here avoid side effects and prefer configuration-driven paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from mne_bids import BIDSPath
import json

# Load centralized configuration and legacy constants
try:  # when imported as part of the package
    from .config_loader import load_config, get_legacy_constants
    from .alignment_utils import align_events_to_epochs_strict
except Exception:  # pragma: no cover - when run as a script in the same folder
    from config_loader import load_config, get_legacy_constants
    from alignment_utils import align_events_to_epochs_strict


_config = load_config()
_constants = get_legacy_constants(_config)
_STRICT_MODE = bool(_config.get("analysis.strict_mode", True))

PROJECT_ROOT: Path = Path(_constants["PROJECT_ROOT"]).resolve()
BIDS_ROOT: Path = Path(_constants["BIDS_ROOT"]).resolve()
DERIV_ROOT: Path = Path(_constants["DERIV_ROOT"]).resolve()

# Optional targets for column picking (e.g., VAS/rating) used in some steps
TARGET_COLUMNS = tuple(_constants.get("TARGET_COLUMNS", ()))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_clean_epochs_path(subject: str, task: str, deriv_root: Optional[Path] = None) -> Optional[Path]:
    """Locate cleaned epochs file under derivatives for a subject and task.

    Order of resolution:
    1) BIDSPath with processing='clean' and suffix='epo'
    2) Literal proc-clean filename under sub-*/eeg
    3) Any task-matching epo file under sub-*/eeg preferring names containing 'clean'
    4) Recursive search under sub-*
    """
    root = Path(deriv_root) if deriv_root is not None else DERIV_ROOT

    # 1) Try BIDSPath
    bp = BIDSPath(
        subject=subject,
        task=task,
        datatype="eeg",
        processing="clean",
        suffix="epo",
        extension=".fif",
        root=root,
        check=False,
    )
    p1 = bp.fpath
    if p1 and p1.exists():
        return p1

    # 2) Literal fallback
    p2 = root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_proc-clean_epo.fif"
    if p2.exists():
        return p2

    # 3) Glob within eeg dir
    subj_eeg_dir = root / f"sub-{subject}" / "eeg"
    if subj_eeg_dir.exists():
        cands = sorted(subj_eeg_dir.glob(f"sub-{subject}_task-{task}*epo.fif"))
        for c in cands:
            if any(tok in c.name for tok in ("proc-clean", "proc-cleaned", "clean")):
                return c
        if cands:
            return cands[0]

    # 4) Recursive search within subject
    subj_dir = root / f"sub-{subject}"
    if subj_dir.exists():
        for c in sorted(subj_dir.rglob(f"sub-{subject}_task-{task}*epo.fif")):
            return c
    return None


def _load_events_df(subject: str, task: str, bids_root: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load BIDS events.tsv for subject and task with robust fallbacks."""
    root = Path(bids_root) if bids_root is not None else BIDS_ROOT
    ebp = BIDSPath(
        subject=subject,
        task=task,
        datatype="eeg",
        suffix="events",
        extension=".tsv",
        root=root,
        check=False,
    )
    p = ebp.fpath
    if p is None:
        p = root / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_events.tsv"
    if p.exists():
        try:
            return pd.read_csv(p, sep="\t")
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            if _STRICT_MODE:
                raise
            return None
    return None


def _align_events_to_epochs(events_df: Optional[pd.DataFrame], epochs, logger=None) -> Optional[pd.DataFrame]:
    """Strictly align events to epochs using centralized alignment utilities.

    This enforces safety and avoids heuristic trimming that could mislabel trials.
    """
    try:
        return align_events_to_epochs_strict(events_df, epochs, logger)
    except ValueError:
        # Critical alignment failures should surface in strict mode; otherwise, return None.
        if _STRICT_MODE:
            raise
        return None


def _pick_target_column(df: pd.DataFrame) -> Optional[str]:
    """Pick a sensible behavioral target column from a DataFrame.

    Prefers configured TARGET_COLUMNS; falls back to any numeric column whose
    name contains 'vas' or 'rating'. Returns None if none found.
    """
    # Prefer configured list
    for c in TARGET_COLUMNS:
        if c in df.columns:
            return c
    # Heuristic fallback
    for c in df.columns:
        cl = str(c).lower()
        if ("vas" in cl or "rating" in cl) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


__all__ = [
    "_ensure_dir",
    "_find_clean_epochs_path",
    "_load_events_df",
    "_align_events_to_epochs",
    "_pick_target_column",
    "_ensure_derivatives_dataset_description",
]


def _ensure_derivatives_dataset_description() -> None:
    """Ensure derivatives/dataset_description.json exists with minimal metadata.

    Creates the file under DERIV_ROOT if missing, following BIDS Derivatives spec.
    """
    desc_path = DERIV_ROOT / "dataset_description.json"
    if desc_path.exists():
        return
    meta = {
        "Name": "EEG Pipeline Derivatives",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "EEG_fMRI_Analysis Pipeline",
                "Version": "unknown",
                "Description": "Custom EEG analysis (ERP, TFR, features, decoding)",
            }
        ],
    }
    try:
        _ensure_dir(DERIV_ROOT)
        with open(desc_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
