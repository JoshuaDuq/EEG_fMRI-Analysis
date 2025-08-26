import os
import re
import sys
import argparse
import inspect
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description

# Allow UTF-8 console output on Windows
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Try to import project config if present
DEFAULT_TASK = "thermalactive"
DEFAULT_MONTAGE = "easycap-M1"
DEFAULT_LINE_FREQ = 60.0

try:
    # Resolve to project root: this file sits in eeg_pipeline/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(PROJECT_ROOT / "eeg_pipeline"))
    import config as project_cfg  # type: ignore

    BIDS_ROOT = Path(getattr(project_cfg, "bids_root", PROJECT_ROOT / "eeg_pipeline" / "bids_output"))
    TASK = getattr(project_cfg, "task", DEFAULT_TASK)
    MONTAGE_NAME = getattr(project_cfg, "eeg_template_montage", DEFAULT_MONTAGE)
    LINE_FREQ = float(getattr(project_cfg, "zapline_fline", DEFAULT_LINE_FREQ) or DEFAULT_LINE_FREQ)
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    BIDS_ROOT = PROJECT_ROOT / "eeg_pipeline" / "bids_output"
    TASK = DEFAULT_TASK
    MONTAGE_NAME = DEFAULT_MONTAGE
    LINE_FREQ = DEFAULT_LINE_FREQ


def find_brainvision_vhdrs(source_root: Path) -> List[Path]:
    """Find all BrainVision .vhdr files under sub-*/eeg.

    Expected layout: source_root/sub-XXX/eeg/*.vhdr
    """
    vhdrs = sorted(source_root.glob("sub-*/eeg/*.vhdr"))
    return [p for p in vhdrs if p.is_file()]


def parse_subject_id(path: Path) -> str:
    """Extract BIDS subject label (without the 'sub-' prefix) from a path.

    Returns e.g. '001'.
    """
    m = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    if not m:
        raise ValueError(f"Could not parse subject from path: {path}")
    return m.group(1)


def events_from_raw_annotations(raw: mne.io.BaseRaw) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """Derive events and an auto event_id from raw annotations.

    If no events can be formed, returns (None, None).
    """
    try:
        # Let MNE infer mapping automatically
        events, event_id = mne.events_from_annotations(raw, event_id=None)
        if events is None or len(events) == 0 or not event_id:
            return None, None
        return events, event_id
    except Exception:
        return None, None


def ensure_dataset_description(bids_root: Path, name: str = "EEG BIDS dataset") -> None:
    bids_root.mkdir(parents=True, exist_ok=True)
    make_dataset_description(
        path=bids_root,
        name=name,
        # These metadata can be customized later
        dataset_type="raw",
        overwrite=True,
    )


def convert_one(vhdr_path: Path, bids_root: Path, task: str, montage_name: Optional[str], line_freq: Optional[float], overwrite: bool = False, merge_behavior: bool = False, zero_base_onsets: bool = False) -> BIDSPath:
    """Convert a single BrainVision file to BIDS using MNE-BIDS.

    Returns the BIDSPath that was written.
    """
    sub_label = parse_subject_id(vhdr_path)

    # Determine run index by sorting all runs for this subject
    all_runs = sorted(vhdr_path.parent.glob("*.vhdr"))
    run_idx = all_runs.index(vhdr_path) + 1 if len(all_runs) > 1 else None

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=False)

    # Set types for non-EEG channels if present
    non_eeg_types = {"HEOG": "eog", "VEOG": "eog", "ECG": "ecg"}
    present_types = {k: v for k, v in non_eeg_types.items() if k in raw.ch_names}
    if present_types:
        raw.set_channel_types(present_types)

    # Optional: set standard montage if names match template
    if montage_name:
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            raw.set_montage(montage, on_missing="warn")
        except Exception:
            # Continue without montage if template not applicable
            pass

    # Add helpful metadata
    raw.info["line_freq"] = line_freq

    # Filter annotations in-place to rely on annotations only (no explicit events passed)
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip()

    target = "Stim_on/S 1"
    anns = raw.annotations
    try:
        if len(anns) > 0:
            keep_idx = [i for i, d in enumerate(anns.description)
                        if _norm(d) == target or str(d).startswith("Stim_on")]
            if keep_idx:
                new_onset = [anns.onset[i] for i in keep_idx]
                new_duration = [anns.duration[i] for i in keep_idx]
                new_desc = [anns.description[i] for i in keep_idx]
                # Optionally zero-base onsets
                if zero_base_onsets and len(new_onset) > 0:
                    base = new_onset[0]
                    new_onset = [o - base for o in new_onset]
                new_anns = mne.Annotations(
                    onset=new_onset,
                    duration=new_duration,
                    description=new_desc,
                    orig_time=anns.orig_time,
                )
                raw.set_annotations(new_anns)
            else:
                # No Stim_on found: clear annotations so nothing gets written
                raw.set_annotations(mne.Annotations([], [], [], orig_time=anns.orig_time))
    except Exception:
        # If annotation filtering fails, proceed with whatever annotations exist
        pass

    bids_path = BIDSPath(
        subject=sub_label,
        task=task,
        run=run_idx,
        datatype="eeg",
        suffix="eeg",
        root=bids_root,
    )

    # Build kwargs for write_raw_bids (no events passed; rely on annotations only)
    sig = inspect.signature(write_raw_bids)
    params = sig.parameters
    kwargs = {}
    # Keep minimal compatibility toggles only
    if "allow_preload" in params:
        kwargs["allow_preload"] = False
    if "format" in params:
        kwargs["format"] = "BrainVision"
    if "verbose" in params:
        kwargs["verbose"] = False

    write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        overwrite=overwrite,
        **kwargs,
    )

    # Post-process: merge behavioral TrialSummary.csv onto Stim_on events if requested
    if merge_behavior:
        try:
            # Locate events.tsv using BIDSPath utilities
            ebp = bids_path.copy()
            ebp.update(suffix="events", extension=".tsv")
            events_tsv_path = ebp.fpath

            if events_tsv_path and events_tsv_path.exists():
                # Psychopy data assumed under sibling folder 'Psychopy_Data' of the subject source folder
                subj_source_dir = vhdr_path.parent.parent  # .../sub-XXX
                psychopy_dir = subj_source_dir / "Psychopy_Data"
                csvs = sorted(psychopy_dir.glob("*TrialSummary.csv")) if psychopy_dir.exists() else []
                if csvs:
                    behav_df = pd.read_csv(csvs[0])
                    ev_df = pd.read_csv(events_tsv_path, sep="\t")

                    # Keep only Stim_on rows (in case upstream didn't filter)
                    def _norm(s: str) -> str:
                        return re.sub(r"\s+", " ", str(s)).strip()

                    ev_df = ev_df[ev_df["trial_type"].map(_norm) == "Stim_on/S 1"].reset_index(drop=True)

                    # Align lengths, warn on mismatch
                    if len(behav_df) != len(ev_df):
                        print(f"Warning: Stim_on events ({len(ev_df)}) != behavioral rows ({len(behav_df)}). Columns will be trimmed to min length.")
                    n = min(len(behav_df), len(ev_df))
                    if n > 0:
                        behav_sub = behav_df.iloc[:n].reset_index(drop=True)
                        ev_df = ev_df.iloc[:n].reset_index(drop=True)
                        # Add behavioral columns
                        for col in behav_sub.columns:
                            ev_df[col] = behav_sub[col].values
                        ev_df.to_csv(events_tsv_path, sep="\t", index=False)
                # else: no behavioral CSV; skip silently
        except Exception as e:
            print(f"Warning: behavioral merge failed for {bids_path}: {e}")

    return bids_path


def main():
    parser = argparse.ArgumentParser(description="Convert BrainVision EEG to BIDS using MNE-BIDS")
    parser.add_argument("--source_root", type=str, default=str(PROJECT_ROOT / "eeg_pipeline" / "source_data"), help="Path to source_data root containing sub-*/eeg")
    parser.add_argument("--bids_root", type=str, default=str(BIDS_ROOT), help="Output BIDS root directory")
    parser.add_argument("--task", type=str, default=TASK, help="BIDS task label")
    parser.add_argument("--subjects", type=str, nargs="*", default=None, help="Optional list of subject labels to include (e.g., 001 002). If omitted, all found are used.")
    parser.add_argument("--montage", type=str, default=MONTAGE_NAME, help="Standard montage name to set on raw (e.g., easycap-M1). Use '' to skip.")
    parser.add_argument("--line_freq", type=float, default=LINE_FREQ, help="Line noise frequency (Hz) metadata for sidecar.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing BIDS files")
    parser.add_argument("--merge_behavior", action="store_true", help="Merge Psychopy TrialSummary.csv into Stim_on events.tsv (disabled by default)")
    parser.add_argument("--zero_base_onsets", action="store_true", help="Zero-base kept annotation onsets so events start at 0.0")

    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    bids_root = Path(args.bids_root).resolve()
    task = args.task
    montage_name = args.montage if args.montage else None

    print(f"Scanning for BrainVision files in: {source_root}")
    vhdrs = find_brainvision_vhdrs(source_root)
    if not vhdrs:
        print("No .vhdr files found under sub-*/eeg/. Nothing to convert.")
        sys.exit(1)

    # Filter subjects if requested
    if args.subjects:
        subj_set = set(args.subjects)
        vhdrs = [p for p in vhdrs if parse_subject_id(p) in subj_set]
        if not vhdrs:
            print(f"No matching .vhdr files for subjects: {sorted(subj_set)}")
            sys.exit(1)

    ensure_dataset_description(bids_root, name=f"{task} EEG")

    written: List[BIDSPath] = []
    for i, vhdr in enumerate(vhdrs, 1):
        try:
            bp = convert_one(
                vhdr_path=vhdr,
                bids_root=bids_root,
                task=task,
                montage_name=montage_name,
                line_freq=args.line_freq,
                overwrite=args.overwrite,
                merge_behavior=args.merge_behavior,
                zero_base_onsets=args.zero_base_onsets,
            )
            written.append(bp)
            rel = str(bp.fpath).replace(str(bids_root) + os.sep, "") if bp.fpath else str(bp)
            print(f"[{i}/{len(vhdrs)}] Wrote: {rel}")
        except Exception as e:
            print(f"[{i}/{len(vhdrs)}] Failed: {vhdr} -> {e}")

    print(f"Done. Converted {len(written)} file(s) to BIDS in: {bids_root}")


if __name__ == "__main__":
    main()
