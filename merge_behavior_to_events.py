import os
import re
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd

# Ensure UTF-8 on Windows consoles
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _norm_trial_type(s: str) -> str:
    # Collapse all whitespace runs to single spaces, strip ends
    return re.sub(r"\s+", " ", str(s)).strip()


def _find_behavior_csv(source_sub_dir: Path) -> Optional[Path]:
    # PsychoPy data expected here: sub-XXX/PsychoPy_Data/*TrialSummary.csv
    psychopy_dir = source_sub_dir / "PsychoPy_Data"
    if not psychopy_dir.exists():
        return None
    csvs: List[Path] = sorted(psychopy_dir.glob("*TrialSummary.csv"))
    return csvs[0] if csvs else None


def merge_one_subject_events(events_tsv: Path, source_root: Path, dry_run: bool = False) -> bool:
    """Merge behavioral TrialSummary.csv into Stim_on rows of an events.tsv.

    Returns True on success, False if no merge performed.
    """
    # Parse subject label from path
    m = re.search(r"sub-([A-Za-z0-9]+)", str(events_tsv))
    if not m:
        print(f"[skip] Could not parse subject from: {events_tsv}")
        return False
    sub_label = m.group(1)

    # Locate corresponding behavior CSV
    beh_csv = _find_behavior_csv(source_root / f"sub-{sub_label}")
    if not beh_csv or not beh_csv.exists():
        print(f"[warn] No TrialSummary.csv found for sub-{sub_label} under {source_root}/sub-{sub_label}/PsychoPy_Data")
        return False

    # Load data
    try:
        ev_df = pd.read_csv(events_tsv, sep="\t")
    except Exception as e:
        print(f"[error] Failed reading events: {events_tsv} -> {e}")
        return False

    try:
        beh_df = pd.read_csv(beh_csv)
    except Exception as e:
        print(f"[error] Failed reading behavior: {beh_csv} -> {e}")
        return False

    # Identify Stim_on rows
    if "trial_type" not in ev_df.columns:
        print(f"[warn] 'trial_type' column missing in events: {events_tsv}")
        return False

    stim_mask = ev_df["trial_type"].map(_norm_trial_type).str.startswith("Stim_on")
    stim_idx = ev_df.index[stim_mask].tolist()
    if len(stim_idx) == 0:
        print(f"[warn] No Stim_on events in: {events_tsv}")
        return False

    n_ev = len(stim_idx)
    n_beh = len(beh_df)

    if n_ev != n_beh:
        print(f"[warn] Count mismatch for sub-{sub_label}: Stim_on events = {n_ev}, behavior rows = {n_beh}. Will trim to min length and continue.")
    n = min(n_ev, n_beh)
    if n == 0:
        print(f"[warn] Nothing to merge for sub-{sub_label} (n=0)")
        return False

    # Subset to aligned lengths
    ev_target_rows = stim_idx[:n]
    beh_sub = beh_df.iloc[:n].reset_index(drop=True)

    # Add/overwrite behavioral columns onto targeted event rows
    for col in beh_sub.columns:
        # Initialize column if not present
        if col not in ev_df.columns:
            ev_df[col] = pd.NA
        # Assign values to Stim_on rows in order
        ev_df.loc[ev_target_rows, col] = beh_sub[col].values

    # Write back
    if dry_run:
        print(f"[dry-run] Would update: {events_tsv} with columns: {list(beh_sub.columns)}")
        return True

    try:
        ev_df.to_csv(events_tsv, sep="\t", index=False)
        print(f"[ok] Merged behavior -> events for sub-{sub_label}: {events_tsv}")
        return True
    except Exception as e:
        print(f"[error] Failed writing events: {events_tsv} -> {e}")
        return False


def main():
    import argparse

    # Try to import project config to get defaults
    default_project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(default_project_root / "eeg_pipeline"))
    try:
        import config as project_cfg  # type: ignore
        bids_root = Path(getattr(project_cfg, "bids_root", default_project_root / "eeg_pipeline" / "bids_output"))
        source_root = Path(default_project_root / "eeg_pipeline" / "source_data")
        task = str(getattr(project_cfg, "task", "thermalactive"))
    except Exception:
        bids_root = default_project_root / "eeg_pipeline" / "bids_output"
        source_root = default_project_root / "eeg_pipeline" / "source_data"
        task = "thermalactive"

    parser = argparse.ArgumentParser(description="Merge behavioral TrialSummary.csv into BIDS events.tsv for each subject")
    parser.add_argument("--bids_root", type=str, default=str(bids_root), help="BIDS root containing sub-*/eeg/*_events.tsv")
    parser.add_argument("--source_root", type=str, default=str(source_root), help="Source root containing sub-*/PsychoPy_Data/*TrialSummary.csv")
    parser.add_argument("--task", type=str, default=task, help="Task label used in events filenames")
    parser.add_argument("--dry_run", action="store_true", help="Do not write files; just report planned changes")

    args = parser.parse_args()

    bids_root = Path(args.bids_root).resolve()
    source_root = Path(args.source_root).resolve()
    task = args.task

    # Find events.tsv files for the task
    pattern = f"sub-*/eeg/*_task-{task}_events.tsv"
    ev_paths = sorted(bids_root.glob(pattern))
    if not ev_paths:
        print(f"[info] No events found under {bids_root} for task '{task}' with pattern {pattern}")
        sys.exit(0)

    n_ok = 0
    for ev in ev_paths:
        ok = merge_one_subject_events(ev, source_root=source_root, dry_run=args.dry_run)
        n_ok += int(ok)

    print(f"Done. Processed {len(ev_paths)} event file(s), merged successfully: {n_ok}.")


if __name__ == "__main__":
    main()
