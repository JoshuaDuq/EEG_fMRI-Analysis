#!/usr/bin/env python3
"""
02_build_confounds_24HMP_outliers.py - Build clean confound regressors for GLM.

Purpose:
    Extract 24-parameter motion model + motion outliers from fMRIPrep confounds
    for use in first-level GLM. Ensures clean matrices with no NaNs and proper
    alignment with BOLD timeseries.

Inputs:
    - work/index/sub-<ID>_files.json: File inventory from 01_discover_inputs.py
    - 00_config.yaml: Configuration file

Outputs:
    - work/firstlevel/sub-<ID>/run-0<r>_confounds_24hmp_outliers.tsv: Clean confounds
    - work/firstlevel/sub-<ID>/run-0<r>_confounds_summary.json: Metadata

Acceptance Criteria:
    - Column count = 24 + #outliers
    - Rows = n_volumes from BOLD
    - No NaNs (filled with 0)
    - All 24 HMP columns present
    
Exit codes:
    0 - All runs processed successfully
    1 - Processing failures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import nibabel as nib

from config_loader import load_config


def log(msg: str, level: str = "INFO"):
    """Print log message with level prefix."""
    print(f"[{level}] {msg}", flush=True)


def load_inventory(inventory_path: Path) -> Dict:
    """
    Load file inventory JSON from 01_discover_inputs.py.
    
    Parameters
    ----------
    inventory_path : Path
        Path to inventory JSON file
    
    Returns
    -------
    dict
        File inventory
    """
    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")
    
    with open(inventory_path, 'r') as f:
        inventory = json.load(f)
    
    return inventory


def get_bold_n_volumes(bold_path: Path) -> int:
    """
    Get number of volumes from BOLD file.
    
    Parameters
    ----------
    bold_path : Path
        Path to BOLD NIfTI file
    
    Returns
    -------
    int
        Number of volumes
    """
    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD file not found: {bold_path}")
    
    img = nib.load(str(bold_path))
    shape = img.shape
    
    if len(shape) == 4:
        return shape[3]
    elif len(shape) == 3:
        return 1
    else:
        raise ValueError(f"Unexpected BOLD shape: {shape}")


def extract_confounds(confounds_path: Path, 
                     motion_24_cols: List[str],
                     motion_outlier_prefix: str,
                     n_volumes: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Extract 24-parameter motion model + outliers from fMRIPrep confounds.
    
    Parameters
    ----------
    confounds_path : Path
        Path to fMRIPrep confounds TSV
    motion_24_cols : list of str
        24 motion parameter column names
    motion_outlier_prefix : str
        Prefix for motion outlier columns
    n_volumes : int
        Expected number of volumes (rows)
    
    Returns
    -------
    pd.DataFrame
        Clean confounds matrix (n_volumes × (24 + n_outliers))
    dict
        Metadata about extraction
    """
    if not confounds_path.exists():
        raise FileNotFoundError(f"Confounds file not found: {confounds_path}")
    
    # Load confounds
    confounds = pd.read_csv(confounds_path, sep='\t')
    
    metadata = {
        'original_rows': len(confounds),
        'original_columns': len(confounds.columns),
        'expected_rows': n_volumes,
        'motion_24_missing': [],
        'motion_24_present': 0,
        'motion_outliers_found': 0,
        'motion_outlier_columns': [],
        'nan_counts': {},
        'filled_nans': False
    }
    
    # Check row count matches BOLD
    if len(confounds) != n_volumes:
        raise ValueError(
            f"Confounds rows ({len(confounds)}) != BOLD volumes ({n_volumes})"
        )
    
    metadata['rows_match_bold'] = True
    
    # Extract 24 motion parameters
    motion_24_present = []
    motion_24_missing = []
    
    for col in motion_24_cols:
        if col in confounds.columns:
            motion_24_present.append(col)
        else:
            motion_24_missing.append(col)
    
    metadata['motion_24_present'] = len(motion_24_present)
    metadata['motion_24_missing'] = motion_24_missing
    
    if motion_24_missing:
        raise ValueError(f"Missing required motion columns: {motion_24_missing}")
    
    # Extract motion outlier columns
    outlier_cols = [col for col in confounds.columns 
                   if col.startswith(motion_outlier_prefix)]
    outlier_cols = sorted(outlier_cols)  # Sort for reproducibility
    
    metadata['motion_outliers_found'] = len(outlier_cols)
    metadata['motion_outlier_columns'] = outlier_cols
    
    # Combine 24 HMP + outliers
    selected_cols = motion_24_present + outlier_cols
    confounds_clean = confounds[selected_cols].copy()
    
    # Check for NaNs
    nan_counts = confounds_clean.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    
    if len(nan_cols) > 0:
        metadata['nan_counts'] = nan_cols.to_dict()
        metadata['filled_nans'] = True
        
        # Fill NaNs with 0 (standard practice for first volume derivatives)
        confounds_clean = confounds_clean.fillna(0)
        
        log(f"  Filled NaNs in {len(nan_cols)} column(s): {list(nan_cols.index)}")
    
    # Final validation
    assert confounds_clean.shape[0] == n_volumes, \
        f"Output rows ({confounds_clean.shape[0]}) != BOLD volumes ({n_volumes})"
    assert confounds_clean.shape[1] == (24 + len(outlier_cols)), \
        f"Output columns ({confounds_clean.shape[1]}) != 24 + {len(outlier_cols)}"
    assert not confounds_clean.isna().any().any(), \
        "NaNs still present after filling"
    
    metadata['output_rows'] = confounds_clean.shape[0]
    metadata['output_columns'] = confounds_clean.shape[1]
    
    return confounds_clean, metadata


def save_confounds(confounds: pd.DataFrame, output_path: Path):
    """
    Save confounds to TSV.
    
    Parameters
    ----------
    confounds : pd.DataFrame
        Confounds matrix
    output_path : Path
        Output TSV path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    confounds.to_csv(output_path, sep='\t', index=False, float_format='%.10f')
    log(f"  Saved confounds: {output_path.name} ({confounds.shape[0]} rows × {confounds.shape[1]} cols)")


def save_metadata(metadata: Dict, output_path: Path):
    """
    Save extraction metadata to JSON.
    
    Parameters
    ----------
    metadata : dict
        Metadata dictionary
    output_path : Path
        Output JSON path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def process_subject(config: Dict, inventory: Dict, 
                   work_dir: Path) -> Tuple[int, int]:
    """
    Process all runs for a subject.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    inventory : dict
        File inventory
    work_dir : Path
        Working directory
    
    Returns
    -------
    tuple of (int, int)
        (n_success, n_total) run counts
    """
    subject = inventory['subject']
    log(f"Processing {subject}")
    
    # Get config parameters
    motion_24_cols = config['confounds']['motion_24_params']
    motion_outlier_prefix = config['confounds']['motion_outlier_prefix']
    
    # Output directory
    subject_dir = work_dir / "firstlevel" / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    n_success = 0
    n_total = 0
    
    for run_key, run_data in inventory['runs'].items():
        run_num = run_data['run_number']
        n_total += 1
        
        log(f"  Run {run_num}:")
        
        # Check if run is complete
        if not run_data['complete']:
            log(f"    ✗ Skipping incomplete run", "WARNING")
            continue
        
        try:
            # Get file paths
            bold_path = Path(run_data['files']['bold']['path'])
            confounds_path = Path(run_data['files']['confounds']['path'])
            
            # Get BOLD volume count
            n_volumes = get_bold_n_volumes(bold_path)
            log(f"    BOLD volumes: {n_volumes}")
            
            # Extract confounds
            confounds_clean, metadata = extract_confounds(
                confounds_path,
                motion_24_cols,
                motion_outlier_prefix,
                n_volumes
            )
            
            # Log extraction details
            log(f"    24 HMP: {metadata['motion_24_present']}/24 present")
            log(f"    Motion outliers: {metadata['motion_outliers_found']}")
            log(f"    Output shape: {metadata['output_rows']} × {metadata['output_columns']}")
            
            # Save confounds
            output_tsv = subject_dir / f"run-{run_num:02d}_confounds_24hmp_outliers.tsv"
            save_confounds(confounds_clean, output_tsv)
            
            # Save metadata
            metadata['subject'] = subject
            metadata['run'] = run_num
            metadata['bold_path'] = str(bold_path)
            metadata['confounds_path'] = str(confounds_path)
            metadata['output_path'] = str(output_tsv)
            
            output_json = subject_dir / f"run-{run_num:02d}_confounds_summary.json"
            save_metadata(metadata, output_json)
            
            log(f"    ✓ Success")
            n_success += 1
            
        except Exception as e:
            log(f"    ✗ Failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    return n_success, n_total


def create_subject_summary(subject_dir: Path, subject: str) -> Dict:
    """
    Create summary report for subject.
    
    Parameters
    ----------
    subject_dir : Path
        Subject output directory
    subject : str
        Subject ID
    
    Returns
    -------
    dict
        Summary statistics
    """
    summary_files = sorted(subject_dir.glob("run-*_confounds_summary.json"))
    
    if not summary_files:
        return {'subject': subject, 'n_runs': 0}
    
    # Aggregate metadata
    all_meta = []
    for sf in summary_files:
        with open(sf, 'r') as f:
            all_meta.append(json.load(f))
    
    summary = {
        'subject': subject,
        'n_runs': len(all_meta),
        'total_columns': [m['output_columns'] for m in all_meta],
        'total_outliers': [m['motion_outliers_found'] for m in all_meta],
        'total_volumes': sum(m['output_rows'] for m in all_meta),
        'any_nans_filled': any(m['filled_nans'] for m in all_meta),
        'runs_processed': [m['run'] for m in all_meta]
    }
    
    # Calculate outlier statistics
    outlier_counts = summary['total_outliers']
    if outlier_counts:
        summary['outlier_stats'] = {
            'min': min(outlier_counts),
            'max': max(outlier_counts),
            'mean': np.mean(outlier_counts),
            'total': sum(outlier_counts)
        }
    
    # Save summary
    summary_path = subject_dir / f"{subject}_confounds_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log(f"  Subject summary: {summary_path.name}")
    
    return summary


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Build clean confound regressors (24HMP + outliers) for GLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects from config
  python 02_build_confounds_24HMP_outliers.py
  
  # Process specific subject
  python 02_build_confounds_24HMP_outliers.py --subject sub-0001
  
  # Custom work directory
  python 02_build_confounds_24HMP_outliers.py --work-dir custom_work
        """
    )
    
    parser.add_argument('--config', default='00_config.yaml',
                       help='Path to configuration file (default: 00_config.yaml)')
    parser.add_argument('--subject', default=None,
                       help='Process specific subject (default: all from config)')
    parser.add_argument('--work-dir', default='work',
                       help='Working directory (default: work)')
    
    args = parser.parse_args()
    
    # Load configuration
    log("=" * 70)
    log("BUILD CONFOUND REGRESSORS (24HMP + OUTLIERS)")
    log("=" * 70)
    
    try:
        config = load_config(args.config)
        log(f"Loaded config: {args.config}")
    except Exception as e:
        log(f"Failed to load config: {e}", "ERROR")
        return 1
    
    # Determine subjects to process
    if args.subject:
        subjects = [args.subject]
        log(f"Processing single subject: {args.subject}")
    else:
        subjects = config['subjects']
        log(f"Processing {len(subjects)} subject(s) from config")
    
    # Setup directories
    work_dir = Path(args.work_dir)
    index_dir = work_dir / "index"
    
    if not index_dir.exists():
        log(f"Index directory not found: {index_dir}", "ERROR")
        log("Run 01_discover_inputs.py first", "ERROR")
        return 1
    
    all_success = True
    subject_summaries = []
    
    # Process each subject
    for subject in subjects:
        log("")
        log("=" * 70)
        log(f"SUBJECT: {subject}")
        log("=" * 70)
        
        # Load inventory
        inventory_path = index_dir / f"{subject}_files.json"
        
        if not inventory_path.exists():
            log(f"Inventory not found: {inventory_path}", "ERROR")
            log("Run 01_discover_inputs.py first", "ERROR")
            all_success = False
            continue
        
        try:
            inventory = load_inventory(inventory_path)
        except Exception as e:
            log(f"Failed to load inventory: {e}", "ERROR")
            all_success = False
            continue
        
        # Process subject
        try:
            n_success, n_total = process_subject(config, inventory, work_dir)
        except Exception as e:
            log(f"Processing failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            all_success = False
            continue
        
        # Create subject summary
        try:
            subject_dir = work_dir / "firstlevel" / subject
            summary = create_subject_summary(subject_dir, subject)
            subject_summaries.append(summary)
        except Exception as e:
            log(f"Failed to create summary: {e}", "WARNING")
        
        # Subject summary
        log("")
        log(f"SUMMARY for {subject}:")
        log(f"  Runs processed: {n_success}/{n_total}")
        
        if n_success == n_total:
            log(f"  ✓ All runs successful")
        else:
            log(f"  ✗ {n_total - n_success} run(s) failed", "WARNING")
            all_success = False
        
        if subject_summaries:
            summary = subject_summaries[-1]
            if 'outlier_stats' in summary:
                stats = summary['outlier_stats']
                log(f"  Motion outliers: {stats['total']} total across runs")
                log(f"    Range: {stats['min']}-{stats['max']} per run")
                log(f"    Mean: {stats['mean']:.1f} per run")
    
    # Final summary
    log("")
    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    
    if subject_summaries:
        total_runs = sum(s['n_runs'] for s in subject_summaries)
        total_volumes = sum(s['total_volumes'] for s in subject_summaries)
        total_outliers = sum(s.get('outlier_stats', {}).get('total', 0) 
                           for s in subject_summaries)
        
        log(f"Subjects processed: {len(subject_summaries)}")
        log(f"Total runs: {total_runs}")
        log(f"Total volumes: {total_volumes}")
        log(f"Total motion outliers: {total_outliers}")
        log(f"Outlier rate: {100 * total_outliers / total_volumes:.2f}% of volumes")
    
    if all_success:
        log("")
        log("✓ All subjects processed successfully")
        log(f"Confounds ready in: {work_dir}/firstlevel/")
        return 0
    else:
        log("")
        log("✗ Some subjects failed processing", "WARNING")
        log("Review error messages above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
