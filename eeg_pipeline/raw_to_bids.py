import os
import re
import sys
import argparse
import inspect
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import mne
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description

# Allow UTF-8 console output on Windows
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Load centralized configuration
try:  # support both `python -m eeg_pipeline.raw_to_bids` and direct execution
    from .config_loader import load_config, get_legacy_constants
except Exception:  # pragma: no cover
    from config_loader import load_config, get_legacy_constants

config = load_config()
_constants = get_legacy_constants(config)

PROJECT_ROOT = _constants["PROJECT_ROOT"]
BIDS_ROOT = _constants["BIDS_ROOT"]
TASK = _constants["TASK"]
# Prefer new per-script section; fall back to legacy constants
MONTAGE_NAME = config.get("raw_to_bids.default_montage", _constants.get("DEFAULT_MONTAGE", "easycap-M1"))
LINE_FREQ = float(config.get("raw_to_bids.default_line_freq", _constants.get("DEFAULT_LINE_FREQ", 60.0)))


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


def _extract_run_label(name: str) -> Optional[int]:
    """Extract run number from a filename (e.g., '..._run-02_...')."""
    m = re.search(r"run-?(\d+)", name, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def events_from_raw_annotations(raw: mne.io.BaseRaw) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """Derive events and an auto event_id from raw annotations.

    If no events can be formed, returns (None, None).
    """
    # Let MNE infer mapping automatically
    events, event_id = mne.events_from_annotations(raw, event_id=None)
    if events is None or len(events) == 0 or not event_id:
        return None, None
    return events, event_id


def ensure_dataset_description(bids_root: Path, name: str = "EEG BIDS dataset") -> None:
    bids_root.mkdir(parents=True, exist_ok=True)
    make_dataset_description(
        path=bids_root,
        name=name,
        # These metadata can be customized later
        dataset_type="raw",
        overwrite=True,
    )


def convert_one(
    vhdr_path: Path,
    bids_root: Path,
    task: str,
    montage_name: Optional[str],
    line_freq: Optional[float],
    overwrite: bool = False,
    zero_base_onsets: bool = False,
    trim_to_first_volume: bool = False,
    event_prefixes: Optional[List[str]] = None,
    keep_all_annotations: bool = False,
) -> BIDSPath:
    """Convert a single BrainVision file to BIDS using MNE-BIDS.

    Returns the BIDSPath that was written.
    """
    sub_label = parse_subject_id(vhdr_path)

    # Determine run index, prefer parsing from filename (e.g., run1, run-02)
    run_idx: Optional[int] = None
    m_run = re.search(r"run[-_]?(\d+)", vhdr_path.stem, flags=re.IGNORECASE)
    if m_run:
        try:
            run_idx = int(m_run.group(1))
        except ValueError:
            run_idx = None
    if run_idx is None:
        # Fallback: position in sorted list
        all_runs = sorted(vhdr_path.parent.glob("*.vhdr"))
        run_idx = all_runs.index(vhdr_path) + 1 if len(all_runs) > 1 else None
        if run_idx is not None:
            print(
                f"Warning: No explicit run found in filename '{vhdr_path.name}'. "
                f"Inferring run={run_idx} by alphabetical order among {len(all_runs)} files. "
                f"Prefer 'run-01' style filenames to guarantee correct run IDs."
            )

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
            # Fix common naming mismatch for EasyCap: 'FPz' -> 'Fpz'
            if "FPz" in raw.ch_names and "Fpz" not in raw.ch_names:
                raw.rename_channels({"FPz": "Fpz"})
            raw.set_montage(montage, on_missing="warn")
        except Exception:
            # Continue without montage if template not applicable
            pass

    # Add helpful metadata
    raw.info["line_freq"] = line_freq

    # Filter annotations in-place to rely on annotations only (no explicit events passed)
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip()

    # Optionally crop the raw to the first MRI volume trigger (e.g., "Volume/V 1" or "V  1") to remove dummy-scan period
    did_trim = False
    if trim_to_first_volume:
        anns0 = raw.annotations
        if len(anns0) > 0:
            # Match either 'Volume/V 1', 'Volume,V 1', or a bare 'V  1' (spaces normalized to single space)
            _pat_v1 = re.compile(r"(^|[/,])V\s*1(\D|$)")
            vol_idx = [
                i
                for i, d in enumerate(anns0.description)
                if _norm(d).startswith("Volume/V") or _pat_v1.search(_norm(d)) is not None
            ]
            if vol_idx:
                t0 = min(anns0.onset[i] for i in vol_idx)
                if isinstance(t0, (int, float)) and t0 > 0:
                    # Crop so that time 0 aligns with the first detected volume trigger
                    print(
                        f"Trimming raw to first volume trigger at {t0:.3f}s relative to recording start."
                    )
                    raw.crop(tmin=float(t0), tmax=None)
                    did_trim = True

    # If we trimmed/cropped the raw, preload to memory so write_raw_bids can write the modified data
    if did_trim and not raw.preload:
        raw.load_data()

    # Normalize prefixes (default to ["Trig_therm/T 1"] if not provided)
    prefixes = event_prefixes if event_prefixes is not None else ["Trig_therm"]
    norm_prefixes = [_norm(p) for p in prefixes if str(p).strip() != ""]
    anns = raw.annotations
    if len(anns) > 0 and not keep_all_annotations:
        keep_idx = [
            i
            for i, d in enumerate(anns.description)
            if any(_norm(d).startswith(tp) for tp in norm_prefixes)
        ]
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
            # No matches found: clear annotations so nothing gets written
            print(
                "Warning: No annotations matched provided prefixes. "
                f"Prefixes={norm_prefixes}. Found {len(anns)} annotations but will drop all, resulting in no events.tsv. "
                "Use --keep_all_annotations or adjust --event_prefix to keep the desired events."
            )
            raw.set_annotations(mne.Annotations([], [], [], orig_time=anns.orig_time))

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
        # Allow writing when data are preloaded (needed after cropping)
        kwargs["allow_preload"] = bool(getattr(raw, "preload", False))
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

    # Post-process events.tsv to add a 'sample' column for robust later alignment
    try:
        sf = float(raw.info.get("sfreq", 0.0))
        if sf > 0:
            ev_bp = bids_path.copy().update(suffix="events", extension=".tsv")
            ev_path = ev_bp.fpath
            if ev_path is not None and ev_path.exists():
                import pandas as _pd, numpy as _np
                ev = _pd.read_csv(ev_path, sep="\t")
                # Add 'sample' if missing
                if "onset" in ev.columns and "sample" not in ev.columns:
                    samp = _pd.to_numeric(ev["onset"], errors="coerce") * sf
                    samp = samp.round().astype("Int64")
                    ev["sample"] = samp
                    ev.to_csv(ev_path, sep="\t", index=False)
                # Write/update per-subject integrity TSV summarizing this run's events
                try:
                    sub = parse_subject_id(ev_path)
                except Exception:
                    sub = parse_subject_id(vhdr_path)
                # Determine run label
                run_label = run_idx if run_idx is not None else _extract_run_label(ev_path.name) if 'run' in ev.columns else None
                # Metrics
                n_rows = int(len(ev))
                has_sample = int("sample" in ev.columns)
                n_miss_onset = int(ev.get("onset").isna().sum()) if "onset" in ev.columns else n_rows
                n_miss_duration = int(ev.get("duration").isna().sum()) if "duration" in ev.columns else n_rows
                n_miss_tt = int(ev.get("trial_type").isna().sum()) if "trial_type" in ev.columns else n_rows
                uniq_tt = []
                if "trial_type" in ev.columns:
                    try:
                        uniq_tt = sorted({str(x) for x in ev["trial_type"].dropna().unique().tolist()})
                    except Exception:
                        uniq_tt = []
                on_min = float(_pd.to_numeric(ev.get("onset"), errors="coerce").min()) if "onset" in ev.columns else _np.nan
                on_max = float(_pd.to_numeric(ev.get("onset"), errors="coerce").max()) if "onset" in ev.columns else _np.nan
                nonmono = False
                if "onset" in ev.columns:
                    try:
                        ons = _pd.to_numeric(ev["onset"], errors="coerce").to_numpy()
                        dif = _np.diff(ons)
                        nonmono = bool(_np.any(dif < -1e-9))
                    except Exception:
                        nonmono = False
                neg_on = int((_pd.to_numeric(ev.get("onset"), errors="coerce") < 0).sum()) if "onset" in ev.columns else 0
                # Append row to subject integrity file in the eeg directory
                integ_path = ev_path.parent / f"sub-{sub}_task-{task}_events_integrity.tsv"
                row = _pd.DataFrame([
                    {
                        "subject": sub,
                        "run": run_label,
                        "n_rows": n_rows,
                        "has_sample": has_sample,
                        "n_missing_onset": n_miss_onset,
                        "n_missing_duration": n_miss_duration,
                        "n_missing_trial_type": n_miss_tt,
                        "unique_trial_types": "|".join(uniq_tt),
                        "onset_min": on_min,
                        "onset_max": on_max,
                        "onset_nonmonotonic": int(nonmono),
                        "negative_onsets": int(neg_on),
                    }
                ])
                header = not integ_path.exists()
                row.to_csv(integ_path, sep="\t", index=False, mode=("w" if header else "a"), header=header)

                # Create minimal events JSON sidecar describing columns
                try:
                    ev_json_bp = ev_bp.copy().update(extension=".json")
                    ev_json_path = ev_json_bp.fpath
                    if ev_json_path is not None:
                        ev_desc = {
                            "onset": {
                                "LongName": "Event onset",
                                "Description": "Onset of the event relative to the start of the recording",
                                "Units": "s",
                            },
                            "duration": {
                                "LongName": "Event duration",
                                "Description": "Duration of the event",
                                "Units": "s",
                            },
                            "trial_type": {
                                "LongName": "Event label",
                                "Description": "Categorical event label indicating experimental condition",
                            },
                            "sample": {
                                "LongName": "Sample index",
                                "Description": "Sample index corresponding to the event onset",
                                "Units": "samples",
                            },
                        }
                        import json as _json
                        with open(ev_json_path, "w", encoding="utf-8") as f:
                            _json.dump(ev_desc, f, indent=2)
                except Exception:
                    pass

                # Normalize channels.tsv types and status
                try:
                    ch_bp = bids_path.copy().update(suffix="channels", extension=".tsv")
                    ch_path = ch_bp.fpath
                    if ch_path is not None and ch_path.exists():
                        ch = _pd.read_csv(ch_path, sep="\t")
                        # Canonicalize Type: uppercase known entries; map GSR to MISC
                        if "type" in ch.columns:
                            types = ch["type"].astype(str).str.upper()
                            # If channel name equals GSR (case-insensitive), set type to MISC
                            name_is_gsr = ch.get("name", _pd.Series([None]*len(ch))).astype(str).str.upper().eq("GSR")
                            types = types.where(~name_is_gsr, other="MISC")
                            # Normalize common types
                            types = types.replace({
                                "EEG": "EEG",
                                "EOG": "EOG",
                                "EMG": "EMG",
                                "ECG": "ECG",
                                "GSR": "MISC",
                                "MISC": "MISC",
                            })
                            ch["type"] = types
                        # Fill missing status with 'good' and normalize case
                        if "status" in ch.columns:
                            status = ch["status"].astype(str).str.lower()
                            status = status.where(~status.isin(["", "nan", "none"]), other="good")
                            status = status.replace({"ok": "good"})
                            ch["status"] = status
                        else:
                            ch["status"] = "good"
                        ch.to_csv(ch_path, sep="\t", index=False)
                except Exception:
                    pass
    except Exception:
        # Do not fail conversion if event patching fails
        pass

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
    parser.add_argument("--zero_base_onsets", action="store_true", help="Zero-base kept annotation onsets so events start at 0.0")
    parser.add_argument(
        "--trim_to_first_volume",
        action="store_true",
        help=(
            "Crop raw to start at the first MRI volume trigger (e.g., 'Volume/V 1', 'Volume,V 1', or bare 'V  1') "
            "to remove the initial dummy-scan period where the scanner may not send triggers."
        ),
    )
    parser.add_argument(
        "--event_prefix",
        action="append",
        default=None,
        help=(
            "Keep only annotations whose normalized label starts with this prefix. "
            "Repeat to keep multiple prefixes, e.g., --event_prefix Trig_therm --event_prefix Reward_on. "
            "If omitted, defaults to Trig_therm. Use --keep_all_annotations to keep all annotations."
        ),
    )
    parser.add_argument("--keep_all_annotations", action="store_true", help="If set, do not filter annotations at all; write whatever exists.")

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
                zero_base_onsets=args.zero_base_onsets,
                trim_to_first_volume=args.trim_to_first_volume,
                event_prefixes=args.event_prefix,
                keep_all_annotations=args.keep_all_annotations,
            )
            written.append(bp)
            rel = str(bp.fpath).replace(str(bids_root) + os.sep, "") if bp.fpath else str(bp)
            print(f"[{i}/{len(vhdrs)}] Wrote: {rel}")
        except Exception as e:
            print(f"[{i}/{len(vhdrs)}] Failed: {vhdr} -> {e}")

    # Optionally summarize per-subject integrity files per subject
    try:
        by_sub = {}
        for bp in written:
            if bp is None:
                continue
            s = bp.subject
            if not s:
                continue
            by_sub.setdefault(s, True)
        for s in sorted(by_sub.keys()):
            subj_eeg = bids_root / f"sub-{s}" / "eeg"
            integ = subj_eeg / f"sub-{s}_task-{task}_events_integrity.tsv"
            if integ.exists():
                print(f"Integrity summary for sub-{s}: {integ}")
    except Exception:
        pass

    print(f"Done. Converted {len(written)} file(s) to BIDS in: {bids_root}")


if __name__ == "__main__":
    main()
