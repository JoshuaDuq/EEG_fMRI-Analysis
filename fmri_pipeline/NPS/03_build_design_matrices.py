#!/usr/bin/env python3
"""
03_build_design_matrices.py - Build first-level GLM design matrices.

Purpose:
    Generate per-run design matrices with temperature regressors + nuisance events,
    properly convolved with SPM canonical HRF and sampled at TR. Compute correlation
    diagnostics to detect multicollinearity issues.

Inputs:
    - Events TSVs (from split_events_to_runs.py)
    - work/index/sub-<ID>_files.json (from 01_discover_inputs.py)
    - 00_config.yaml: Configuration file

Outputs:
    - work/firstlevel/sub-<ID>/run-0<r>_design_preview.tsv: Design matrix (QC)
    - qc/sub-<ID>_design_corr_run-0<r>.tsv: Correlation diagnostics
    - work/firstlevel/sub-<ID>/run-0<r>_design_metadata.json: Design info

Acceptance Criteria:
    - Exactly 6 temperature task regressors + nuisance events
    - Low inter-task correlations (|r| < 0.2 threshold)
    - Proper HRF convolution (SPM canonical)
    - Correct TR sampling
    
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
from nilearn.glm.first_level import make_first_level_design_matrix

from config_loader import load_config


def log(msg: str, level: str = "INFO"):
    """Print log message with level prefix."""
    print(f"[{level}] {msg}", flush=True)


def load_inventory(inventory_path: Path) -> Dict:
    """Load file inventory from 01_discover_inputs.py."""
    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")
    
    with open(inventory_path, 'r') as f:
        return json.load(f)


def get_bold_info(bold_path: Path) -> Tuple[int, float]:
    """
    Get BOLD volume count and TR.
    
    Parameters
    ----------
    bold_path : Path
        Path to BOLD NIfTI
    
    Returns
    -------
    tuple of (int, float)
        (n_volumes, tr_seconds)
    """
    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD not found: {bold_path}")
    
    img = nib.load(str(bold_path))
    n_volumes = img.shape[3] if len(img.shape) == 4 else 1
    
    # Try to get TR from header
    try:
        tr = img.header.get_zooms()[3]
    except Exception:
        tr = None
    
    return n_volumes, tr


def _load_drop_log(subject: str, drop_root: Path) -> pd.DataFrame:
    """Load EEG drop log for subject if available."""

    # Subject may already have 'sub-' prefix
    subject_id = subject if subject.startswith('sub-') else f"sub-{subject}"
    drop_path = drop_root / subject_id / "eeg" / "features" / "dropped_trials.tsv"
    if not drop_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(drop_path, sep="\t")
        if "original_index" not in df.columns:
            log(f"    ⚠ drop log missing 'original_index' at {drop_path}; ignoring", "WARNING")
            return pd.DataFrame()
        return df
    except Exception as exc:  # pragma: no cover
        log(f"    ⚠ Failed to read drop log {drop_path}: {exc}", "WARNING")
        return pd.DataFrame()


def _filter_events_with_drop_log(events_df: pd.DataFrame,
                                drop_log: pd.DataFrame,
                                run_num: int) -> pd.DataFrame:
    """Remove EEG-rejected trials from events DataFrame if drop log is provided."""

    if drop_log.empty:
        return events_df

    filtered = events_df.copy()

    if {'run', 'trial_number'}.issubset(drop_log.columns) and 'trial_number' in filtered.columns:
        drop_pairs = set(
            (int(row['run']), int(row['trial_number']))
            for _, row in drop_log.iterrows()
            if not pd.isna(row.get('run')) and not pd.isna(row.get('trial_number'))
        )
        if drop_pairs:
            mask = filtered.apply(
                lambda r: (int(r.get('run', run_num)), int(r.get('trial_number', r.name + 1))) in drop_pairs,
                axis=1,
            )
            if mask.any():
                log(f"    Run {run_num}: removed {mask.sum()} events per EEG drop log", "INFO")
                filtered = filtered[~mask].reset_index(drop=True)

    if 'original_index' in drop_log.columns and len(filtered) > 0:
        drop_indices = set(int(idx) for idx in drop_log['original_index'].tolist())
        mask = filtered.index.to_series().isin(drop_indices)
        if mask.any():
            log(
                f"    Run {run_num}: removed {mask.sum()} events per EEG drop log (index match)",
                "INFO",
            )
            filtered = filtered[~mask].reset_index(drop=True)

    return filtered


def create_design_matrix(events_df: pd.DataFrame,
                        n_volumes: int,
                        tr: float,
                        temp_labels: List[str],
                        nuisance_events: List[str],
                        hrf_model: str = 'spm') -> pd.DataFrame:
    """
    Create first-level design matrix with HRF convolution.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe with onset, duration, trial_type
    n_volumes : int
        Number of BOLD volumes
    tr : float
        Repetition time in seconds
    temp_labels : list of str
        Temperature condition labels
    nuisance_events : list of str
        Nuisance event labels (decision, rating, delay)
    hrf_model : str
        HRF model ('spm', 'glover', etc.)
    
    Returns
    -------
    pd.DataFrame
        Design matrix (n_volumes × n_regressors)
    """
    # Calculate frame times (middle of each TR)
    frame_times = np.arange(n_volumes) * tr
    
    # Filter events to task and nuisance types
    task_types = set(temp_labels)
    nuisance_types = set(nuisance_events)
    all_types = task_types | nuisance_types
    
    # Keep only relevant events
    events_filtered = events_df[events_df['trial_type'].isin(all_types)].copy()
    
    if len(events_filtered) == 0:
        raise ValueError("No relevant events found in events file")
    
    # Ensure required columns
    required_cols = ['onset', 'duration', 'trial_type']
    for col in required_cols:
        if col not in events_filtered.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Create design matrix with Nilearn
    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_filtered[required_cols],
        hrf_model=hrf_model,
        drift_model=None,  # Don't add drift here (will add confounds separately)
        high_pass=None,
        add_regs=None,
        add_reg_names=None,
        min_onset=0
    )
    
    # Remove constant column if present
    if 'constant' in design_matrix.columns:
        design_matrix = design_matrix.drop(columns=['constant'])
    
    return design_matrix


def compute_design_correlations(design_matrix: pd.DataFrame,
                               temp_labels: List[str]) -> pd.DataFrame:
    """
    Compute pairwise correlations among design matrix columns.
    
    Parameters
    ----------
    design_matrix : pd.DataFrame
        Design matrix
    temp_labels : list of str
        Temperature labels (for task regressors)
    
    Returns
    -------
    pd.DataFrame
        Correlation matrix with diagnostics
    """
    # Compute correlation matrix
    corr_matrix = design_matrix.corr()
    
    # Extract task regressors only
    task_cols = [col for col in design_matrix.columns if any(temp in col for temp in temp_labels)]
    
    if not task_cols:
        log("  ⚠ No task columns found for correlation analysis", "WARNING")
        return pd.DataFrame(), corr_matrix
    
    # Get task-task correlations
    task_corr = corr_matrix.loc[task_cols, task_cols]
    
    # Create summary with off-diagonal correlations
    corr_summary = []
    
    for i, col1 in enumerate(task_cols):
        for col2 in task_cols[i+1:]:
            r = task_corr.loc[col1, col2]
            corr_summary.append({
                'regressor_1': col1,
                'regressor_2': col2,
                'correlation': r,
                'abs_correlation': abs(r),
                'flag_high': abs(r) > 0.2
            })
    
    summary_df = pd.DataFrame(corr_summary)

    return summary_df, corr_matrix


def validate_design(design_matrix: pd.DataFrame,
                   temp_labels: List[str],
                   nuisance_events: List[str],
                   events_df: pd.DataFrame = None) -> Dict:
    """
    Validate design matrix structure.
    
    Parameters
    ----------
    design_matrix : pd.DataFrame
        Design matrix
    temp_labels : list of str
        All possible temperature labels (from config)
    nuisance_events : list of str
        Expected nuisance event labels
    events_df : pd.DataFrame, optional
        Events dataframe to check which temps are actually present
    
    Returns
    -------
    dict
        Validation results
    """
    validation = {
        'n_rows': len(design_matrix),
        'n_columns': len(design_matrix.columns),
        'column_names': list(design_matrix.columns),
        'temp_regressors_found': [],
        'temp_regressors_missing': [],
        'temp_regressors_present_in_events': [],
        'nuisance_regressors_found': [],
        'nuisance_regressors_missing': [],
        'has_constant': False,
        'valid': True,
        'warnings': []
    }
    
    # Check for constant
    if 'constant' in design_matrix.columns:
        validation['has_constant'] = True
        validation['warnings'].append("Design matrix contains constant column")
    
    # Determine which temps are actually present in events
    if events_df is not None:
        temp_types_in_events = set(events_df['trial_type'].unique()) & set(temp_labels)
        validation['temp_regressors_present_in_events'] = sorted(temp_types_in_events)
    else:
        # If no events provided, assume all temps should be present
        temp_types_in_events = set(temp_labels)
        validation['temp_regressors_present_in_events'] = temp_labels
    
    # Check temperature regressors (only validate those present in events)
    for temp in temp_labels:
        found = any(temp in col for col in design_matrix.columns)
        if found:
            validation['temp_regressors_found'].append(temp)
        else:
            if temp in temp_types_in_events:
                # Missing a temp that should be there
                validation['temp_regressors_missing'].append(temp)
                validation['warnings'].append(f"Missing expected temperature regressor: {temp}")
                validation['valid'] = False
            else:
                # Not present in events for this run (OK for balanced incomplete designs)
                validation['temp_regressors_missing'].append(temp)
    
    # Check nuisance regressors (these should always be present)
    for nuisance in nuisance_events:
        found = any(nuisance in col for col in design_matrix.columns)
        if found:
            validation['nuisance_regressors_found'].append(nuisance)
        else:
            validation['nuisance_regressors_missing'].append(nuisance)
            validation['warnings'].append(f"Missing nuisance regressor: {nuisance}")
            validation['valid'] = False
    
    # Expected count (only for temps present in events + all nuisance)
    expected_cols = len(temp_types_in_events) + len(nuisance_events)
    if validation['n_columns'] != expected_cols:
        validation['warnings'].append(
            f"Column count ({validation['n_columns']}) != expected ({expected_cols} = {len(temp_types_in_events)} temps + {len(nuisance_events)} nuisance)"
        )
    
    return validation


def save_design_matrix(design_matrix: pd.DataFrame, output_path: Path):
    """Save design matrix to TSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    design_matrix.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
    log(f"  Saved design matrix: {output_path.name}")


def save_correlations(corr_df: pd.DataFrame, output_path: Path):
    """Save correlation summary to TSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
    log(f"  Saved correlations: {output_path.name}")


def save_metadata(metadata: Dict, output_path: Path):
    """Save design metadata to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def process_run(config: Dict,
               events_path: Path,
               bold_path: Path,
               run_num: int,
               subject: str,
               work_dir: Path,
               qc_dir: Path,
               drop_log: pd.DataFrame) -> bool:
    """
    Process single run to create design matrix.
    
    Parameters
    ----------
    config : dict
        Configuration
    events_path : Path
        Events TSV path
    bold_path : Path
        BOLD NIfTI path
    run_num : int
        Run number
    subject : str
        Subject ID
    work_dir : Path
        Working directory
    qc_dir : Path
        QC directory
    
    Returns
    -------
    bool
        Success status
    """
    log(f"  Run {run_num}:")
    
    try:
        # Load events
        events = pd.read_csv(events_path, sep='\t')
        events = _filter_events_with_drop_log(events, drop_log, run_num)
        log(f"    Events loaded: {len(events)} events")
        
        # Get BOLD info
        n_volumes, bold_tr = get_bold_info(bold_path)
        
        # Get TR from config
        config_tr = config['glm']['tr']
        
        # Use config TR (more reliable than header)
        tr = config_tr
        if bold_tr is not None and abs(bold_tr - config_tr) > 0.01:
            log(f"    ⚠ TR mismatch: header={bold_tr:.3f}s, config={config_tr:.3f}s. Using config.", "WARNING")
        
        log(f"    BOLD: {n_volumes} volumes, TR={tr:.3f}s")
        
        # Get parameters from config
        temp_labels = config['glm']['temp_labels']
        nuisance_events = config['glm']['nuisance_events']
        hrf_model = config['glm']['hrf']['model']
        
        # Create design matrix
        design_matrix = create_design_matrix(
            events,
            n_volumes,
            tr,
            temp_labels,
            nuisance_events,
            hrf_model=hrf_model
        )
        
        log(f"    Design matrix: {design_matrix.shape[0]} × {design_matrix.shape[1]}")
        
        # Validate design (pass events to check which temps are actually present)
        validation = validate_design(design_matrix, temp_labels, nuisance_events, events)
        
        if validation['warnings']:
            for warning in validation['warnings']:
                log(f"    ⚠ {warning}", "WARNING")
        
        if not validation['valid']:
            log(f"    ✗ Design validation failed", "ERROR")
            return False
        
        # Report temperatures found vs those present in events (not total possible)
        n_temps_in_events = len(validation['temp_regressors_present_in_events'])
        n_temps_found = len(validation['temp_regressors_found'])
        log(f"    ✓ Temperature regressors: {n_temps_found}/{n_temps_in_events} present in this run")
        log(f"    ✓ Nuisance regressors: {len(validation['nuisance_regressors_found'])}/{len(nuisance_events)}")
        
        # Compute correlations
        corr_summary, corr_matrix = compute_design_correlations(design_matrix, temp_labels)
        
        if not corr_summary.empty:
            # Flag high correlations
            high_corr = corr_summary[corr_summary['flag_high']]
            
            if len(high_corr) > 0:
                log(f"    ⚠ High correlations detected: {len(high_corr)} pairs |r| > 0.2", "WARNING")
                for _, row in high_corr.iterrows():
                    log(f"      {row['regressor_1']} <-> {row['regressor_2']}: r={row['correlation']:.3f}", "WARNING")
            else:
                log(f"    ✓ All task correlations |r| < 0.2")
            
            # Summary statistics
            max_corr = corr_summary['abs_correlation'].max()
            mean_corr = corr_summary['abs_correlation'].mean()
            log(f"    Correlation stats: max={max_corr:.3f}, mean={mean_corr:.3f}")
        
        # Prepare metadata
        metadata = {
            'subject': subject,
            'run': run_num,
            'n_volumes': n_volumes,
            'tr': tr,
            'n_events': len(events),
            'hrf_model': hrf_model,
            'design_shape': list(design_matrix.shape),
            'temp_labels': temp_labels,
            'nuisance_events': nuisance_events,
            'validation': validation,
            'correlation_summary': {
                'n_pairs': len(corr_summary),
                'n_high_corr': int(corr_summary['flag_high'].sum()) if not corr_summary.empty else 0,
                'max_abs_corr': float(corr_summary['abs_correlation'].max()) if not corr_summary.empty else None,
                'mean_abs_corr': float(corr_summary['abs_correlation'].mean()) if not corr_summary.empty else None
            }
        }
        
        # Save outputs
        subject_dir = work_dir / "firstlevel" / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Design matrix preview
        design_preview_path = subject_dir / f"run-{run_num:02d}_design_preview.tsv"
        save_design_matrix(design_matrix, design_preview_path)
        
        # Metadata
        metadata_path = subject_dir / f"run-{run_num:02d}_design_metadata.json"
        save_metadata(metadata, metadata_path)
        
        # Correlations
        if not corr_summary.empty:
            corr_path = qc_dir / f"{subject}_design_corr_run-{run_num:02d}.tsv"
            save_correlations(corr_summary, corr_path)
            
            # Save full correlation matrix for detailed inspection
            corr_matrix_path = subject_dir / f"run-{run_num:02d}_design_corr_matrix.tsv"
            corr_matrix.to_csv(corr_matrix_path, sep='\t', float_format='%.6f')
        
        log(f"    ✓ Success")
        return True
        
    except Exception as e:
        log(f"    ✗ Failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


def process_subject(config: Dict,
                   inventory: Dict,
                   work_dir: Path,
                   qc_dir: Path) -> Tuple[int, int]:
    """
    Process all runs for a subject.
    
    Returns
    -------
    tuple of (int, int)
        (n_success, n_total)
    """
    subject = inventory['subject']
    log(f"Processing {subject}")
    
    n_success = 0
    n_total = 0
    
    for run_key, run_data in inventory['runs'].items():
        run_num = run_data['run_number']
        n_total += 1
        
        if not run_data['complete']:
            log(f"  Run {run_num}: Skipping incomplete run", "WARNING")
            continue
        
        events_path = Path(run_data['files']['events']['path'])
        bold_path = Path(run_data['files']['bold']['path'])
        
        # Note: EEG-rejected trials are already filtered in source events files
        # by split_events_to_runs.py, so no need to load drop log here
        drop_log = pd.DataFrame()

        success = process_run(
            config,
            events_path,
            bold_path,
            run_num,
            subject,
            work_dir,
            qc_dir,
            drop_log
        )
        
        if success:
            n_success += 1
    
    return n_success, n_total


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Build first-level GLM design matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects from config
  python 03_build_design_matrices.py
  
  # Process specific subject
  python 03_build_design_matrices.py --subject sub-0001
  
  # Custom directories
  python 03_build_design_matrices.py --work-dir custom_work --qc-dir custom_qc
        """
    )
    
    parser.add_argument('--config', default='00_config.yaml',
                       help='Path to configuration file (default: 00_config.yaml)')
    parser.add_argument('--subject', default=None,
                       help='Process specific subject (default: all from config)')
    parser.add_argument('--work-dir', default='work',
                       help='Working directory (default: work)')
    parser.add_argument('--qc-dir', default='qc',
                       help='QC directory (default: qc)')
    
    args = parser.parse_args()
    
    # Load configuration
    log("=" * 70)
    log("BUILD FIRST-LEVEL DESIGN MATRICES")
    log("=" * 70)
    
    try:
        config = load_config(args.config)
        log(f"Loaded config: {args.config}")
    except Exception as e:
        log(f"Failed to load config: {e}", "ERROR")
        return 1
    
    # Determine subjects
    if args.subject:
        subjects = [args.subject]
        log(f"Processing single subject: {args.subject}")
    else:
        subjects = config['subjects']
        log(f"Processing {len(subjects)} subject(s) from config")
    
    # Setup directories
    work_dir = Path(args.work_dir)
    qc_dir = Path(args.qc_dir)
    index_dir = work_dir / "index"
    
    if not index_dir.exists():
        log(f"Index directory not found: {index_dir}", "ERROR")
        log("Run 01_discover_inputs.py first", "ERROR")
        return 1
    
    all_success = True
    
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
            n_success, n_total = process_subject(config, inventory, work_dir, qc_dir)
        except Exception as e:
            log(f"Processing failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            all_success = False
            continue
        
        # Summary
        log("")
        log(f"SUMMARY for {subject}:")
        log(f"  Runs processed: {n_success}/{n_total}")
        
        if n_success == n_total:
            log(f"  ✓ All runs successful")
        else:
            log(f"  ✗ {n_total - n_success} run(s) failed", "WARNING")
            all_success = False
    
    # Final summary
    log("")
    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    
    if all_success:
        log("✓ All subjects processed successfully")
        log(f"Design matrices ready in: {work_dir}/firstlevel/")
        log(f"QC reports in: {qc_dir}/")
        return 0
    else:
        log("✗ Some subjects failed processing", "WARNING")
        return 1


if __name__ == '__main__':
    sys.exit(main())
