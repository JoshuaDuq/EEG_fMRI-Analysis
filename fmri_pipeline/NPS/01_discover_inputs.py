#!/usr/bin/env python3
"""
01_discover_inputs.py - Discover and validate all input files for fMRI GLM analysis.

Purpose:
    Enumerate all required input files (BOLD, masks, confounds, events) for each
    subject/run and perform rigorous validation checks before GLM analysis.

Inputs:
    - 00_config.yaml: Central configuration file

Outputs:
    - work/index/sub-<ID>_files.json: Resolved file paths per subject
    - qc/sub-<ID>_events_check.tsv: Per-run event validation
    - qc/sub-<ID>_confounds_check.tsv: Confound column validation
    
Validation Criteria:
    - Events: 11 heat trials per run, 66 total across 6 runs
    - Temperature balance: 11 trials per temp (44.3-49.3°C) 
    - Durations: heat ~12.5s, decision/rating present
    - Confounds: 24 HMP columns + motion outlier detection
    
Exit codes:
    0 - All files found and validated
    1 - Missing files or validation failures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import nibabel as nib

from config_loader import load_config, get_subject_files


def log(msg: str, level: str = "INFO"):
    """Print log message with level prefix."""
    print(f"[{level}] {msg}", flush=True)


def discover_subject_files(config: Dict, subject: str) -> Dict[str, any]:
    """
    Discover all input files for a subject across all runs.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject ID (e.g., 'sub-0001')
    
    Returns
    -------
    dict
        File inventory with paths and validation status
    """
    task = config['task']
    space = config['space']
    runs = config['runs']
    
    inventory = {
        'subject': subject,
        'task': task,
        'space': space,
        'runs': {},
        'anatomical': {},
        'summary': {
            'total_runs': len(runs),
            'runs_complete': 0,
            'missing_files': []
        }
    }
    
    log(f"Discovering files for {subject}")
    
    # Anatomical files (run-independent)
    try:
        t1w_preproc = get_subject_files(config, subject, run=1, file_type='anat')
    except Exception as e:
        log(f"  ⚠ Error building anatomical path: {e}", "WARNING")
        t1w_preproc = None
    
    if t1w_preproc:
        inventory['anatomical']['t1w_preproc'] = {
            'path': str(t1w_preproc),
            'exists': t1w_preproc.exists()
        }
        
        if not t1w_preproc.exists():
            log(f"  ⚠ Anatomical T1w not found: {t1w_preproc.name}", "WARNING")
    
    # Functional files (per run)
    for run in runs:
        run_key = f"run-{run:02d}"
        run_data = {'run_number': run, 'files': {}, 'complete': False}
        
        # Get file paths using config_loader helper
        try:
            bold_file = get_subject_files(config, subject, run, 'bold')
            mask_file = get_subject_files(config, subject, run, 'mask')
            confounds_file = get_subject_files(config, subject, run, 'confounds')
            events_file = get_subject_files(config, subject, run, 'events')
        except Exception as e:
            log(f"  ✗ Error building paths for {run_key}: {e}", "ERROR")
            inventory['runs'][run_key] = run_data
            continue
        
        # Check BOLD
        run_data['files']['bold'] = {
            'path': str(bold_file),
            'exists': bold_file.exists()
        }
        
        if bold_file.exists():
            try:
                img = nib.load(str(bold_file))
                run_data['files']['bold']['shape'] = img.shape
                run_data['files']['bold']['n_volumes'] = img.shape[3] if len(img.shape) == 4 else 1
            except Exception as e:
                log(f"  ✗ Cannot read BOLD: {bold_file.name} - {e}", "ERROR")
                run_data['files']['bold']['error'] = str(e)
        else:
            log(f"  ✗ Missing BOLD: {bold_file.name}", "ERROR")
            inventory['summary']['missing_files'].append(str(bold_file))
        
        # Check mask
        run_data['files']['mask'] = {
            'path': str(mask_file),
            'exists': mask_file.exists()
        }
        
        if not mask_file.exists():
            log(f"  ✗ Missing mask: {mask_file.name}", "ERROR")
            inventory['summary']['missing_files'].append(str(mask_file))
        
        # Check confounds
        run_data['files']['confounds'] = {
            'path': str(confounds_file),
            'exists': confounds_file.exists()
        }
        
        if confounds_file.exists():
            try:
                df = pd.read_csv(confounds_file, sep='\t')
                run_data['files']['confounds']['n_rows'] = len(df)
                run_data['files']['confounds']['n_columns'] = len(df.columns)
            except Exception as e:
                log(f"  ✗ Cannot read confounds: {confounds_file.name} - {e}", "ERROR")
                run_data['files']['confounds']['error'] = str(e)
        else:
            log(f"  ✗ Missing confounds: {confounds_file.name}", "ERROR")
            inventory['summary']['missing_files'].append(str(confounds_file))
        
        # Check events
        run_data['files']['events'] = {
            'path': str(events_file),
            'exists': events_file.exists()
        }
        
        if events_file.exists():
            try:
                df = pd.read_csv(events_file, sep='\t')
                run_data['files']['events']['n_events'] = len(df)
                run_data['files']['events']['event_types'] = df['trial_type'].unique().tolist()
            except Exception as e:
                log(f"  ✗ Cannot read events: {events_file.name} - {e}", "ERROR")
                run_data['files']['events']['error'] = str(e)
        else:
            log(f"  ✗ Missing events: {events_file.name}", "ERROR")
            inventory['summary']['missing_files'].append(str(events_file))
        
        # Mark run as complete if all critical files exist
        run_data['complete'] = all([
            run_data['files']['bold']['exists'],
            run_data['files']['mask']['exists'],
            run_data['files']['confounds']['exists'],
            run_data['files']['events']['exists']
        ])
        
        if run_data['complete']:
            inventory['summary']['runs_complete'] += 1
            log(f"  ✓ {run_key}: All files present")
        else:
            log(f"  ✗ {run_key}: Incomplete", "WARNING")
        
        inventory['runs'][run_key] = run_data
    
    return inventory


def validate_events(config: Dict, subject: str, inventory: Dict) -> pd.DataFrame:
    """
    Validate event files for all runs.
    
    Checks:
    - 11 heat trials per run
    - Temperature labels match config
    - Heat duration ~12.5s
    - Decision and rating events present
    - Temperature balance: 11 trials per temp across all runs
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject ID
    inventory : dict
        File inventory from discover_subject_files
    
    Returns
    -------
    pd.DataFrame
        Per-run validation results
    """
    log(f"Validating events for {subject}")
    
    temp_labels = set(config['glm']['temp_labels'])
    expected_heat_duration = config['glm']['stim_dur_sec']
    nuisance_events = set(config['glm']['nuisance_events'])
    
    validation_rows = []
    all_heat_events = []
    
    for run_key, run_data in inventory['runs'].items():
        run_num = run_data['run_number']
        
        row = {
            'subject': subject,
            'run': run_num,
            'events_file_exists': False,
            'n_total_events': 0,
            'n_heat_trials': 0,
            'n_decision': 0,
            'n_rating': 0,
            'n_delay': 0,
            'heat_duration_mean': np.nan,
            'heat_duration_std': np.nan,
            'heat_duration_min': np.nan,
            'heat_duration_max': np.nan,
            'onset_min': np.nan,
            'onset_max': np.nan,
            'valid_heat_count': False,
            'valid_temp_labels': False,
            'valid_durations': False,
            'valid_nuisance': False,
            'errors': []
        }
        
        if not run_data['files']['events']['exists']:
            row['errors'].append('Events file missing')
            validation_rows.append(row)
            continue
        
        row['events_file_exists'] = True
        
        try:
            events = pd.read_csv(run_data['files']['events']['path'], sep='\t')
            row['n_total_events'] = len(events)
            row['onset_min'] = events['onset'].min()
            row['onset_max'] = events['onset'].max()
            
            # Heat trials
            heat_events = events[events['trial_type'].isin(temp_labels)]
            row['n_heat_trials'] = len(heat_events)
            
            if len(heat_events) > 0:
                row['heat_duration_mean'] = heat_events['duration'].mean()
                row['heat_duration_std'] = heat_events['duration'].std()
                row['heat_duration_min'] = heat_events['duration'].min()
                row['heat_duration_max'] = heat_events['duration'].max()
                
                all_heat_events.append(heat_events)
            
            # Nuisance events
            row['n_decision'] = len(events[events['trial_type'] == 'decision'])
            row['n_rating'] = len(events[events['trial_type'] == 'rating'])
            row['n_delay'] = len(events[events['trial_type'] == 'delay'])
            
            # Validation checks
            row['valid_heat_count'] = (row['n_heat_trials'] == 11)
            
            # Check all temp labels are valid
            invalid_temps = set(heat_events['trial_type'].unique()) - temp_labels
            row['valid_temp_labels'] = (len(invalid_temps) == 0)
            if invalid_temps:
                row['errors'].append(f"Invalid temp labels: {invalid_temps}")
            
            # Check heat durations (allow ±1s tolerance)
            duration_ok = (
                abs(row['heat_duration_mean'] - expected_heat_duration) < 1.0
                and row['heat_duration_std'] < 0.5
            )
            row['valid_durations'] = duration_ok
            if not duration_ok:
                row['errors'].append(
                    f"Heat duration mean={row['heat_duration_mean']:.2f}s "
                    f"(expected ~{expected_heat_duration}s)"
                )
            
            # Check nuisance events
            if 'decision' in nuisance_events and row['n_decision'] != 11:
                row['errors'].append(f"Expected 11 decision events, got {row['n_decision']}")
            if 'rating' in nuisance_events and row['n_rating'] != 11:
                row['errors'].append(f"Expected 11 rating events, got {row['n_rating']}")
            if 'delay' in nuisance_events and row['n_delay'] < 1:
                row['errors'].append(f"Expected delay events, got {row['n_delay']}")
            
            row['valid_nuisance'] = (
                (row['n_decision'] == 11 if 'decision' in nuisance_events else True)
                and (row['n_rating'] == 11 if 'rating' in nuisance_events else True)
            )
            
            if not row['valid_heat_count']:
                row['errors'].append(f"Expected 11 heat trials, got {row['n_heat_trials']}")
            
        except Exception as e:
            row['errors'].append(f"Error reading events: {e}")
            log(f"  ✗ Run {run_num}: {e}", "ERROR")
        
        validation_rows.append(row)
    
    # Overall temperature balance check
    if all_heat_events:
        combined_heat = pd.concat(all_heat_events, ignore_index=True)
        temp_counts = combined_heat['trial_type'].value_counts()
        
        log(f"  Temperature distribution across all runs:")
        for temp in sorted(temp_labels):
            count = temp_counts.get(temp, 0)
            status = "✓" if count == 11 else "✗"
            log(f"    {status} {temp}: {count} trials (expected 11)")
        
        total_heat = len(combined_heat)
        expected_total = len(temp_labels) * 11
        if total_heat == expected_total:
            log(f"  ✓ Total heat trials: {total_heat}/{expected_total}")
        else:
            log(f"  ✗ Total heat trials: {total_heat}/{expected_total}", "ERROR")
    
    df = pd.DataFrame(validation_rows)
    
    # Summary
    n_valid = df['valid_heat_count'].sum()
    n_total = len(df)
    if n_valid == n_total:
        log(f"  ✓ Events validation: {n_valid}/{n_total} runs passed")
    else:
        log(f"  ✗ Events validation: {n_valid}/{n_total} runs passed", "WARNING")
    
    return df


def validate_confounds(config: Dict, subject: str, inventory: Dict) -> pd.DataFrame:
    """
    Validate confound files for all runs.
    
    Checks:
    - 24 motion parameters exist
    - Motion outlier columns detected
    - Number of rows matches BOLD volumes
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject ID
    inventory : dict
        File inventory
    
    Returns
    -------
    pd.DataFrame
        Per-run confound validation
    """
    log(f"Validating confounds for {subject}")
    
    motion_24 = set(config['confounds']['motion_24_params'])
    motion_outlier_prefix = config['confounds']['motion_outlier_prefix']
    
    validation_rows = []
    
    for run_key, run_data in inventory['runs'].items():
        run_num = run_data['run_number']
        
        row = {
            'subject': subject,
            'run': run_num,
            'confounds_file_exists': False,
            'n_rows': 0,
            'n_columns': 0,
            'n_motion_24': 0,
            'missing_motion_params': [],
            'n_motion_outliers': 0,
            'motion_outlier_columns': [],
            'n_bold_volumes': 0,
            'rows_match_volumes': False,
            'valid_motion_24': False,
            'errors': []
        }
        
        if not run_data['files']['confounds']['exists']:
            row['errors'].append('Confounds file missing')
            validation_rows.append(row)
            continue
        
        row['confounds_file_exists'] = True
        
        # Get expected number of volumes from BOLD
        if run_data['files']['bold']['exists']:
            row['n_bold_volumes'] = run_data['files']['bold'].get('n_volumes', 0)
        
        try:
            confounds = pd.read_csv(run_data['files']['confounds']['path'], sep='\t')
            row['n_rows'] = len(confounds)
            row['n_columns'] = len(confounds.columns)
            
            columns = set(confounds.columns)
            
            # Check 24 motion parameters
            missing_motion = motion_24 - columns
            row['n_motion_24'] = 24 - len(missing_motion)
            row['missing_motion_params'] = list(missing_motion)
            row['valid_motion_24'] = (len(missing_motion) == 0)
            
            if missing_motion:
                row['errors'].append(f"Missing motion params: {missing_motion}")
            
            # Find motion outlier columns
            outlier_cols = [col for col in confounds.columns 
                          if col.startswith(motion_outlier_prefix)]
            row['n_motion_outliers'] = len(outlier_cols)
            row['motion_outlier_columns'] = outlier_cols
            
            # Check row count matches BOLD volumes
            if row['n_bold_volumes'] > 0:
                row['rows_match_volumes'] = (row['n_rows'] == row['n_bold_volumes'])
                if not row['rows_match_volumes']:
                    row['errors'].append(
                        f"Confounds rows ({row['n_rows']}) != "
                        f"BOLD volumes ({row['n_bold_volumes']})"
                    )
            
        except Exception as e:
            row['errors'].append(f"Error reading confounds: {e}")
            log(f"  ✗ Run {run_num}: {e}", "ERROR")
        
        validation_rows.append(row)
    
    df = pd.DataFrame(validation_rows)
    
    # Summary
    n_valid = df['valid_motion_24'].sum()
    n_total = len(df)
    if n_valid == n_total:
        log(f"  ✓ Confounds validation: {n_valid}/{n_total} runs passed")
    else:
        log(f"  ✗ Confounds validation: {n_valid}/{n_total} runs passed", "WARNING")
    
    # Motion outlier summary
    total_outliers = df['n_motion_outliers'].sum()
    if total_outliers > 0:
        log(f"  ℹ Total motion outlier columns detected: {total_outliers}")
    
    return df


def save_inventory(inventory: Dict, output_path: Path):
    """Save file inventory to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(inventory, f, indent=2)
    log(f"Saved inventory: {output_path}")


def save_validation_report(df: pd.DataFrame, output_path: Path):
    """Save validation DataFrame to TSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert list columns to strings for TSV output
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    df.to_csv(output_path, sep='\t', index=False, float_format='%.3f')
    log(f"Saved validation report: {output_path}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Discover and validate input files for fMRI GLM pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects from config
  python 01_discover_inputs.py
  
  # Process specific subject
  python 01_discover_inputs.py --subject sub-0001
  
  # Custom config file
  python 01_discover_inputs.py --config custom_config.yaml
        """
    )
    
    parser.add_argument('--config', default='00_config.yaml',
                       help='Path to configuration file (default: 00_config.yaml)')
    parser.add_argument('--subject', default=None,
                       help='Process specific subject (default: all from config)')
    parser.add_argument('--work-dir', default='work',
                       help='Working directory for outputs (default: work)')
    parser.add_argument('--qc-dir', default='qc',
                       help='QC directory for validation reports (default: qc)')
    
    args = parser.parse_args()
    
    # Load configuration
    log("=" * 70)
    log("FMRI INPUT DISCOVERY AND VALIDATION")
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
    
    # Setup output directories
    work_dir = Path(args.work_dir)
    qc_dir = Path(args.qc_dir)
    index_dir = work_dir / "index"
    
    all_success = True
    
    # Process each subject
    for subject in subjects:
        log("")
        log("=" * 70)
        log(f"SUBJECT: {subject}")
        log("=" * 70)
        
        # Discover files
        try:
            inventory = discover_subject_files(config, subject)
        except Exception as e:
            log(f"File discovery failed: {e}", "ERROR")
            all_success = False
            continue
        
        # Save inventory
        inventory_path = index_dir / f"{subject}_files.json"
        save_inventory(inventory, inventory_path)
        
        # Validate events
        try:
            events_validation = validate_events(config, subject, inventory)
            events_report_path = qc_dir / f"{subject}_events_check.tsv"
            save_validation_report(events_validation, events_report_path)
        except Exception as e:
            log(f"Events validation failed: {e}", "ERROR")
            all_success = False
        
        # Validate confounds
        try:
            confounds_validation = validate_confounds(config, subject, inventory)
            confounds_report_path = qc_dir / f"{subject}_confounds_check.tsv"
            save_validation_report(confounds_validation, confounds_report_path)
        except Exception as e:
            log(f"Confounds validation failed: {e}", "ERROR")
            all_success = False
        
        # Summary for this subject
        log("")
        log(f"SUMMARY for {subject}:")
        log(f"  Runs complete: {inventory['summary']['runs_complete']}/{inventory['summary']['total_runs']}")
        
        if inventory['summary']['missing_files']:
            log(f"  ✗ Missing {len(inventory['summary']['missing_files'])} file(s)", "WARNING")
            all_success = False
        else:
            log(f"  ✓ All files present")
        
        # Check validation status
        events_pass = events_validation['valid_heat_count'].all()
        confounds_pass = confounds_validation['valid_motion_24'].all()
        
        if events_pass and confounds_pass:
            log(f"  ✓ All validations passed")
        else:
            if not events_pass:
                log(f"  ✗ Events validation failed", "WARNING")
            if not confounds_pass:
                log(f"  ✗ Confounds validation failed", "WARNING")
            all_success = False
    
    # Final summary
    log("")
    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    
    if all_success:
        log("✓ All subjects passed discovery and validation")
        log("Ready to proceed with GLM analysis")
        return 0
    else:
        log("✗ Some subjects failed validation", "WARNING")
        log("Review QC reports before proceeding")
        return 1


if __name__ == '__main__':
    sys.exit(main())
