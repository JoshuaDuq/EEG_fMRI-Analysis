#!/usr/bin/env python3
"""
04_fit_first_level_glm.py - Fit first-level GLMs and extract beta maps.

Purpose:
    Fit run-wise GLMs with canonical HRF convolution for task regressors,
    24-parameter motion model + outliers as confounds, and high-pass filtering.
    Extract beta maps for each temperature condition.

Inputs:
    - work/index/sub-<ID>_files.json: File inventory
    - work/firstlevel/sub-<ID>/run-0<r>_confounds_24hmp_outliers.tsv: Confounds
    - 00_config.yaml: Configuration file

Outputs:
    - work/firstlevel/sub-<ID>/run-0<r>_beta_temp44p3.nii.gz: Beta maps (one per temp)
    - work/firstlevel/sub-<ID>/run-0<r>_modeldiag.json: Model diagnostics
    - qc/sub-<ID>_run-0<r>_regressor_snr.tsv: SNR per regressor (optional)
    
Acceptance Criteria:
    - 6 beta maps per run (one per temperature)
    - Reasonable DOF (n_volumes - n_regressors)
    - Clean residuals (low autocorrelation)
    
Exit codes:
    0 - All runs processed successfully
    1 - Processing failures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import apply_mask, unmask

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


def load_confounds(confounds_path: Path) -> pd.DataFrame:
    """Load confounds TSV from 02_build_confounds_24HMP_outliers.py."""
    if not confounds_path.exists():
        raise FileNotFoundError(f"Confounds not found: {confounds_path}")
    
    confounds = pd.read_csv(confounds_path, sep='\t')
    return confounds


def _load_drop_log(subject: str, drop_log_root: Path) -> pd.DataFrame:
    """Load EEG drop log for subject if available."""

    # Subject may already have 'sub-' prefix
    subject_id = subject if subject.startswith('sub-') else f"sub-{subject}"
    drop_log_path = drop_log_root / subject_id / "eeg" / "features" / "dropped_trials.tsv"
    if not drop_log_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(drop_log_path, sep="\t")
        if "original_index" not in df.columns:
            log(
                f"  ⚠ drop log missing 'original_index' column at {drop_log_path}; ignoring",
                "WARNING",
            )
            return pd.DataFrame()
        return df
    except Exception as exc:  # pragma: no cover
        log(f"  ⚠ Failed to read drop log {drop_log_path}: {exc}", "WARNING")
        return pd.DataFrame()


def prepare_events(
    events_path: Path,
    temp_labels: list,
    nuisance_events: list,
    drop_log: pd.DataFrame,
    run_num: int,
) -> pd.DataFrame:
    """
    Load and prepare events for GLM.
    
    Parameters
    ----------
    events_path : Path
        Path to events TSV
    temp_labels : list
        Temperature condition labels
    nuisance_events : list
        Nuisance event labels
    
    Returns
    -------
    pd.DataFrame
        Events dataframe with onset, duration, trial_type
    """
    events = pd.read_csv(events_path, sep='\t')

    # Remove EEG-rejected trials if a drop log was provided
    if not drop_log.empty:
        # Temperature trials typically have run/trial info
        if {'trial_number', 'run'}.issubset(drop_log.columns) and 'trial_number' in events.columns:
            drop_pairs = set(
                (int(row['run']), int(row['trial_number']))
                for _, row in drop_log.iterrows()
                if not pd.isna(row.get('run')) and not pd.isna(row.get('trial_number'))
            )
            if drop_pairs:
                mask = events.apply(
                    lambda r: (int(r.get('run', run_num)), int(r.get('trial_number', r.name + 1))) in drop_pairs,
                    axis=1,
                )
                if mask.any():
                    log(
                        f"    Run {run_num}: removing {mask.sum()} events per EEG drop log",
                        "INFO",
                    )
                    events = events[~mask].reset_index(drop=True)
        elif 'original_index' in drop_log.columns:
            drop_indices = set(int(idx) for idx in drop_log['original_index'].tolist())
            mask = events.index.to_series().isin(drop_indices)
            if mask.any():
                log(
                    f"    Run {run_num}: removing {mask.sum()} events per EEG drop log (index match)",
                    "INFO",
                )
                events = events[~mask].reset_index(drop=True)

    # Filter to relevant event types
    all_types = set(temp_labels) | set(nuisance_events)
    events_filtered = events[events['trial_type'].isin(all_types)].copy()
    
    # Ensure required columns
    required_cols = ['onset', 'duration', 'trial_type']
    for col in required_cols:
        if col not in events_filtered.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return events_filtered[required_cols]


def fit_glm(bold_path: Path,
           mask_path: Path,
           events: pd.DataFrame,
           confounds: pd.DataFrame,
           tr: float,
           hrf_model: str,
           high_pass: float) -> Tuple[FirstLevelModel, Dict]:
    """
    Fit first-level GLM for a single run.
    
    Parameters
    ----------
    bold_path : Path
        Path to BOLD NIfTI
    mask_path : Path
        Path to brain mask
    events : pd.DataFrame
        Events dataframe
    confounds : pd.DataFrame
        Confounds matrix
    tr : float
        Repetition time
    hrf_model : str
        HRF model ('spm', 'glover', etc.)
    high_pass : float
        High-pass filter cutoff in Hz
    
    Returns
    -------
    FirstLevelModel
        Fitted GLM model
    dict
        Fit diagnostics
    """
    # Load BOLD and mask
    bold_img = nib.load(str(bold_path))
    mask_img = nib.load(str(mask_path))
    
    n_volumes = bold_img.shape[3] if len(bold_img.shape) == 4 else 1
    
    # Validate confounds match BOLD
    if len(confounds) != n_volumes:
        raise ValueError(
            f"Confounds rows ({len(confounds)}) != BOLD volumes ({n_volumes})"
        )
    
    # Create FirstLevelModel
    glm = FirstLevelModel(
        t_r=tr,
        hrf_model=hrf_model,
        drift_model='cosine',
        high_pass=high_pass,
        mask_img=mask_img,
        smoothing_fwhm=None,  # No spatial smoothing
        minimize_memory=False,  # Keep results in memory
        n_jobs=1,  # Single-threaded for stability
        verbose=0
    )
    
    log(f"      Fitting GLM...")
    log(f"        Events: {len(events)} events")
    log(f"        Confounds: {confounds.shape[1]} columns")
    log(f"        High-pass: {high_pass:.6f} Hz ({1/high_pass:.1f} s cutoff)")
    
    # Fit model
    glm.fit(bold_img, events=events, confounds=confounds)
    
    # Compute diagnostics
    n_regressors = glm.design_matrices_[0].shape[1]
    dof = n_volumes - n_regressors
    
    # Get R² from model
    try:
        r_squared = glm.r_square_[0]
        if isinstance(r_squared, np.ndarray):
            mean_r2 = float(np.mean(r_squared[r_squared > 0]))
            median_r2 = float(np.median(r_squared[r_squared > 0]))
        else:
            mean_r2 = float(r_squared)
            median_r2 = float(r_squared)
    except Exception:
        mean_r2 = None
        median_r2 = None
    
    diagnostics = {
        'n_volumes': n_volumes,
        'n_regressors': n_regressors,
        'n_events': len(events),
        'n_confounds': confounds.shape[1],
        'dof': dof,
        'mean_r_squared': mean_r2,
        'median_r_squared': median_r2,
        'design_matrix_shape': list(glm.design_matrices_[0].shape),
        'design_columns': list(glm.design_matrices_[0].columns)
    }
    
    log(f"      GLM fitted successfully")
    log(f"        Regressors: {n_regressors} (task + confounds + drift)")
    log(f"        DOF: {dof}")
    if mean_r2 is not None:
        log(f"        Mean R²: {mean_r2:.4f}")
    
    return glm, diagnostics


def extract_beta_maps(glm: FirstLevelModel,
                     temp_labels: list,
                     output_dir: Path,
                     run_num: int) -> Dict[str, Path]:
    """
    Extract beta maps for each temperature condition.
    
    Parameters
    ----------
    glm : FirstLevelModel
        Fitted GLM model
    temp_labels : list
        Temperature condition labels
    output_dir : Path
        Output directory
    run_num : int
        Run number
    
    Returns
    -------
    dict
        Mapping of condition -> beta map path
    """
    beta_paths = {}
    
    log(f"      Extracting beta maps...")
    
    # Get design matrix columns
    design_cols = glm.design_matrices_[0].columns.tolist()
    
    # Build mapping of temp labels to design matrix columns
    temp_col_map = {}
    for temp in temp_labels:
        # Find columns that contain this temp label
        matching_cols = [col for col in design_cols if temp in col]
        if matching_cols:
            # Use the first match (should be the main effect, not drift/confound)
            temp_col_map[temp] = matching_cols[0]
    
    log(f"        Found {len(temp_col_map)}/{len(temp_labels)} temperature columns in design")
    
    for temp in temp_labels:
        # Check if this temp is in the design for this run
        if temp not in temp_col_map:
            log(f"        {temp}: Not present in this run (by design)", "INFO")
            continue
        
        try:
            # Get the actual column name from design matrix
            col_name = temp_col_map[temp]
            
            # Build contrast vector manually (safer than passing string)
            contrast_vec = np.zeros(len(design_cols))
            col_idx = design_cols.index(col_name)
            contrast_vec[col_idx] = 1.0
            
            # Compute contrast (raw beta effect)
            beta_map = glm.compute_contrast(contrast_vec, output_type='effect_size')
            
            # Save beta map
            beta_path = output_dir / f"run-{run_num:02d}_beta_{temp}.nii.gz"
            nib.save(beta_map, str(beta_path))
            
            beta_paths[temp] = beta_path
            
            # Get some statistics
            beta_data = beta_map.get_fdata()
            beta_nonzero = beta_data[beta_data != 0]
            
            if len(beta_nonzero) > 0:
                log(f"        {temp}: mean={beta_nonzero.mean():.4f}, "
                    f"std={beta_nonzero.std():.4f}, "
                    f"range=[{beta_nonzero.min():.4f}, {beta_nonzero.max():.4f}]")
            else:
                log(f"        {temp}: No non-zero voxels (check events/design)", "WARNING")
            
        except Exception as e:
            log(f"        ✗ Failed to extract {temp}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            continue
    
    log(f"      Saved {len(beta_paths)} beta maps")
    
    return beta_paths


def compute_regressor_snr(glm: FirstLevelModel,
                         bold_path: Path,
                         mask_path: Path) -> pd.DataFrame:
    """
    Compute signal-to-noise ratio for each regressor.
    
    Parameters
    ----------
    glm : FirstLevelModel
        Fitted GLM model
    bold_path : Path
        Path to BOLD NIfTI
    mask_path : Path
        Path to brain mask
    
    Returns
    -------
    pd.DataFrame
        SNR per regressor
    """
    try:
        # Load BOLD and mask
        bold_img = nib.load(str(bold_path))
        mask_img = nib.load(str(mask_path))
        
        # Extract data (timepoints x voxels)
        bold_data = apply_mask(bold_img, mask_img)

        # Standardize BOLD signals once to enable fast correlation estimates
        bold_data = np.asarray(bold_data, dtype=np.float64)
        n_timepoints = bold_data.shape[0]

        # Guard against degenerate time series
        if n_timepoints < 2:
            raise ValueError("At least two time points are required to compute regressor SNR")

        bold_mean = bold_data.mean(axis=0)
        bold_centered = bold_data - bold_mean
        bold_std = bold_centered.std(axis=0, ddof=1)
        valid_voxels = bold_std > 0

        bold_z = np.zeros_like(bold_centered)
        if np.any(valid_voxels):
            bold_z[:, valid_voxels] = bold_centered[:, valid_voxels] / bold_std[valid_voxels]

        bold_z_valid = bold_z[:, valid_voxels]
        n_valid_voxels = bold_z_valid.shape[1]

        # Get design matrix
        design = glm.design_matrices_[0]

        # Compute variance explained by each regressor
        snr_data = []

        for col in design.columns:
            if col.startswith('drift') or col == 'constant':
                continue  # Skip drift and constant

            # Get regressor
            regressor = np.asarray(design[col].values, dtype=np.float64)
            reg_mean = regressor.mean()
            reg_std_unbiased = regressor.std(ddof=1)
            reg_range = float(regressor.max() - regressor.min())

            if (
                np.isclose(reg_range, 0.0)
                or not np.isfinite(reg_std_unbiased)
                or reg_std_unbiased <= 0
                or np.isclose(reg_std_unbiased, 0.0)
            ):
                mean_corr = 0.0
            else:
                reg_z = (regressor - reg_mean) / reg_std_unbiased
                if n_valid_voxels == 0:
                    mean_corr = 0.0
                else:
                    corr_with_bold = reg_z @ bold_z_valid / (n_timepoints - 1)
                    mean_corr = float(np.nanmean(np.abs(corr_with_bold))) if corr_with_bold.size else 0.0

            snr_data.append({
                'regressor': col,
                'mean_abs_correlation': mean_corr,
                'regressor_std': float(regressor.std()),
                'regressor_range': reg_range
            })

        snr_df = pd.DataFrame(snr_data)
        if not snr_df.empty:
            snr_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            snr_df.fillna(0.0, inplace=True)
        return snr_df
        
    except Exception as e:
        log(f"      ⚠ SNR computation failed: {e}", "WARNING")
        return pd.DataFrame()


def save_diagnostics(diagnostics: Dict, output_path: Path):
    """Save model diagnostics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    log(f"      Saved diagnostics: {output_path.name}")


def save_snr(snr_df: pd.DataFrame, output_path: Path):
    """Save SNR table to TSV."""
    if snr_df.empty:
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snr_df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
    log(f"      Saved SNR: {output_path.name}")


def process_run(config: Dict,
               inventory: Dict,
               run_key: str,
               subject: str,
               work_dir: Path,
               qc_dir: Path,
               compute_snr: bool = False) -> bool:
    """
    Process single run: fit GLM and extract beta maps.
    
    Parameters
    ----------
    config : dict
        Configuration
    inventory : dict
        File inventory
    run_key : str
        Run key (e.g., 'run-01')
    subject : str
        Subject ID
    work_dir : Path
        Working directory
    qc_dir : Path
        QC directory
    compute_snr : bool
        Whether to compute regressor SNR
    
    Returns
    -------
    bool
        Success status
    """
    run_data = inventory['runs'][run_key]
    run_num = run_data['run_number']
    
    log(f"  Run {run_num}:")
    
    # Check if run is complete
    if not run_data['complete']:
        log(f"    ✗ Skipping incomplete run", "WARNING")
        return False
    
    try:
        # Get file paths
        bold_path = Path(run_data['files']['bold']['path'])
        mask_path = Path(run_data['files']['mask']['path'])
        events_path = Path(run_data['files']['events']['path'])
        
        # Load confounds
        subject_dir = work_dir / "firstlevel" / subject
        confounds_path = subject_dir / f"run-{run_num:02d}_confounds_24hmp_outliers.tsv"
        
        if not confounds_path.exists():
            log(f"    ✗ Confounds not found: {confounds_path.name}", "ERROR")
            log(f"    Run 02_build_confounds_24HMP_outliers.py first", "ERROR")
            return False
        
        confounds = load_confounds(confounds_path)
        log(f"    Confounds: {confounds.shape[1]} columns")
        
        # Prepare events
        # Note: EEG-rejected trials are already filtered out in the source events files
        # by split_events_to_runs.py, so no need to filter again here
        temp_labels = config['glm']['temp_labels']
        nuisance_events = config['glm']['nuisance_events']
        
        events = prepare_events(events_path, temp_labels, nuisance_events, pd.DataFrame(), run_num)
        log(f"    Events: {len(events)} events")
        
        # Get GLM parameters
        tr = config['glm']['tr']
        hrf_model = config['glm']['hrf']['model']
        high_pass_sec = config['glm']['high_pass_sec']
        high_pass_hz = 1.0 / high_pass_sec
        
        log(f"    GLM settings:")
        log(f"      TR: {tr}s")
        log(f"      HRF: {hrf_model}")
        log(f"      High-pass: {high_pass_sec}s")
        
        # Fit GLM
        glm, diagnostics = fit_glm(
            bold_path,
            mask_path,
            events,
            confounds,
            tr,
            hrf_model,
            high_pass_hz
        )
        
        # Extract beta maps
        beta_paths = extract_beta_maps(glm, temp_labels, subject_dir, run_num)
        
        if len(beta_paths) == 0:
            log(f"    ✗ No beta maps extracted", "ERROR")
            return False
        
        # Add beta paths to diagnostics
        diagnostics['beta_maps'] = {temp: str(path) for temp, path in beta_paths.items()}
        diagnostics['subject'] = subject
        diagnostics['run'] = run_num
        
        # Save diagnostics
        diag_path = subject_dir / f"run-{run_num:02d}_modeldiag.json"
        save_diagnostics(diagnostics, diag_path)
        
        # Optional: Compute regressor SNR
        if compute_snr:
            log(f"      Computing regressor SNR...")
            snr_df = compute_regressor_snr(glm, bold_path, mask_path)
            if not snr_df.empty:
                snr_path = qc_dir / f"{subject}_run-{run_num:02d}_regressor_snr.tsv"
                save_snr(snr_df, snr_path)
        
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
                   qc_dir: Path,
                   compute_snr: bool = False) -> Tuple[int, int]:
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
    
    for run_key in sorted(inventory['runs'].keys()):
        n_total += 1
        success = process_run(
            config,
            inventory,
            run_key,
            subject,
            work_dir,
            qc_dir,
            compute_snr
        )
        if success:
            n_success += 1
    
    return n_success, n_total


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Fit first-level GLMs and extract beta maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects
  python 04_fit_first_level_glm.py
  
  # Process specific subject
  python 04_fit_first_level_glm.py --subject sub-0001
  
  # Compute regressor SNR (slower)
  python 04_fit_first_level_glm.py --subject sub-0001 --compute-snr
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
    parser.add_argument('--compute-snr', action='store_true',
                       help='Compute regressor SNR (adds processing time)')
    
    args = parser.parse_args()
    
    # Load configuration
    log("=" * 70)
    log("FIT FIRST-LEVEL GLMs")
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
            n_success, n_total = process_subject(
                config,
                inventory,
                work_dir,
                qc_dir,
                args.compute_snr
            )
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
        log(f"Beta maps ready in: {work_dir}/firstlevel/")
        return 0
    else:
        log("✗ Some subjects failed processing", "WARNING")
        return 1


if __name__ == '__main__':
    sys.exit(main())
