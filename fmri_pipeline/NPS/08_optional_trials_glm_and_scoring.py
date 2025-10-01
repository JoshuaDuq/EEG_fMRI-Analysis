#!/usr/bin/env python3
"""
08_optional_trials_glm_and_scoring.py - Trial-wise GLM and NPS scoring.

Purpose:
    Fit trial-wise GLMs (one regressor per trial) and compute NPS scores for
    each individual trial. Enables ROC/AUC analysis and forced-choice modeling.

Inputs:
    - work/index/sub-<ID>_files.json: File inventory
    - work/firstlevel/sub-<ID>/run-0<r>_confounds_24hmp_outliers.tsv: Confounds
    - NPS weights file: weights_NSF_grouppred_cvpcr.nii.gz
    - 00_config.yaml: Configuration file

Outputs:
    - outputs/nps_scores/sub-<ID>/trial_br.tsv: Trial-wise BR scores
    - outputs/nps_scores/sub-<ID>/trial_scoring_metadata.json: Metadata
    - qc/sub-<ID>_trial_glm_qc.tsv: QC metrics

Acceptance Criteria:
    - 66 trials per subject (11 per temperature × 6 temps)
    - All BR values finite
    - AUC > 0.5 for pain discrimination
    - Reasonable BR distributions

Exit codes:
    0 - All subjects processed successfully
    1 - Processing failures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from config_loader import load_config


def log(msg: str, level: str = "INFO"):
    """Print log message with level prefix."""
    print(f"[{level}] {msg}", flush=True)

def _load_drop_log(subject: str, drop_log_dir: Path) -> pd.DataFrame:
    """Load EEG drop log for a subject, returning a DataFrame of dropped events.

    The drop log is expected at
    `drop_log_dir/sub-<subject>/eeg/features/dropped_trials.tsv`, with an
    'original_index' column. Additional trial metadata (e.g., run, trial_number) is optional but LEVERAGED when present.
    """

    # Subject may already have 'sub-' prefix
    subject_id = subject if subject.startswith('sub-') else f"sub-{subject}"
    drop_log_path = drop_log_dir / subject_id / "eeg" / "features" / "dropped_trials.tsv"
    if not drop_log_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(drop_log_path, sep="\t")
        # Ensure required column exists
        if "original_index" not in df.columns:
            log(
                f"    ⚠ drop log at {drop_log_path} missing 'original_index'; ignoring",
                "WARNING",
            )
            return pd.DataFrame()
        return df
    except Exception as exc:  # pragma: no cover - safety guard
        log(f"    ⚠ Failed to read drop log {drop_log_path}: {exc}", "WARNING")
        return pd.DataFrame()


def create_trial_events(
    events_path: Path,
    run_num: int,
    global_trial_offset: int,
    drop_log: pd.DataFrame,
    subject: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create trial-wise events with unique regressor per trial.
    
    Parameters
    ----------
    events_path : Path
        Path to events TSV
    run_num : int
        Run number
    global_trial_offset : int
        Offset for global trial indexing across runs
    
    Returns
    -------
    pd.DataFrame
        Events with trial_type = 'trial_<idx>'
    """
    events = pd.read_csv(events_path, sep='\t')
    
    # Filter to temperature trials only
    temp_trials = events[events['trial_type'].str.startswith('temp')].copy()
    
    # Filter out trials that were rejected in EEG preprocessing
    if not drop_log.empty:
        # Two strategies depending on available metadata
        dropped_mask = pd.Series([False] * len(temp_trials), index=temp_trials.index)

        if {"run", "trial_number"}.issubset(drop_log.columns):
            # Strategy 1: Match by (run, trial_number) pairs
            drop_pairs = set(
                (int(row["run"]), int(row["trial_number"]))
                for _, row in drop_log.iterrows()
                if not pd.isna(row.get("run")) and not pd.isna(row.get("trial_number"))
            )
            if drop_pairs:
                # Assign sequential trial numbers within run (1-indexed)
                temp_trials['_temp_trial_num'] = range(1, len(temp_trials) + 1)
                dropped_mask = temp_trials.apply(
                    lambda r: (run_num, int(r['_temp_trial_num'])) in drop_pairs,
                    axis=1,
                )
                temp_trials.drop(columns=['_temp_trial_num'], inplace=True)

        if not dropped_mask.any() and "original_index" in drop_log.columns:
            # Fallback: use original index aligned to global trial offset
            # original_index in drop_log is global (0-65), need to check against global offset
            drop_indices = set(int(idx) for idx in drop_log["original_index"].tolist())
            # Create global indices for current run's trials
            temp_trials['_temp_global_idx'] = range(global_trial_offset, global_trial_offset + len(temp_trials))
            dropped_mask = temp_trials['_temp_global_idx'].isin(drop_indices)
            temp_trials.drop(columns=['_temp_global_idx'], inplace=True)

        if dropped_mask.any():
            n_drop = int(dropped_mask.sum())
            log(
                f"    Run {run_num}: excluding {n_drop} trial(s) based on EEG drop log",
                "INFO",
            )
            drop_subset = temp_trials[dropped_mask].copy()
            drop_subset["drop_reason"] = drop_subset.index.map(
                lambda idx: ";".join(
                    str(reason)
                    for reason in drop_log.loc[
                        drop_log["original_index"] == idx, "drop_reason"
                    ].tolist()
                    if reason
                )
            )
            # Save diagnostic info for combined summary later
            drop_summary_path = events_path.parent / f"sub-{subject}_run-{run_num:02d}_dropped_trials.tsv"
            drop_subset.to_csv(drop_summary_path, sep="\t", index=False)
            temp_trials = temp_trials[~dropped_mask].reset_index(drop=True)

    # Assign unique trial indices
    temp_trials['trial_idx_global'] = range(global_trial_offset, global_trial_offset + len(temp_trials))
    temp_trials['trial_idx_run'] = range(len(temp_trials))
    
    # Create trial-specific regressor names
    temp_trials['trial_regressor'] = temp_trials['trial_idx_global'].apply(lambda x: f'trial_{x:03d}')
    
    # Create trial events dataframe
    trial_events = pd.DataFrame({
        'onset': temp_trials['onset'],
        'duration': temp_trials['duration'],
        'trial_type': temp_trials['trial_regressor']
    })
    
    # Add nuisance events (decision, rating, delay)
    nuisance_events = events[events['trial_type'].isin(['decision', 'rating', 'delay'])].copy()
    nuisance_events = nuisance_events[['onset', 'duration', 'trial_type']]
    
    # Combine
    all_events = pd.concat([trial_events, nuisance_events], ignore_index=True)
    all_events = all_events.sort_values('onset').reset_index(drop=True)
    
    return all_events, temp_trials


def fit_trial_glm(bold_path: Path,
                 mask_path: Path,
                 events: pd.DataFrame,
                 confounds: pd.DataFrame,
                 tr: float,
                 hrf_model: str,
                 high_pass: float) -> FirstLevelModel:
    """
    Fit trial-wise GLM.
    
    Parameters
    ----------
    bold_path : Path
        Path to BOLD NIfTI
    mask_path : Path
        Path to brain mask
    events : pd.DataFrame
        Trial-wise events
    confounds : pd.DataFrame
        Confounds matrix
    tr : float
        Repetition time
    hrf_model : str
        HRF model
    high_pass : float
        High-pass filter cutoff in Hz
    
    Returns
    -------
    FirstLevelModel
        Fitted GLM model
    """
    # Load BOLD and mask
    bold_img = nib.load(str(bold_path))
    mask_img = nib.load(str(mask_path))
    
    n_volumes = bold_img.shape[3] if len(bold_img.shape) == 4 else 1
    
    # Validate confounds
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
        smoothing_fwhm=None,
        minimize_memory=False,
        n_jobs=1,
        verbose=0
    )
    
    # Fit model
    glm.fit(bold_img, events=events, confounds=confounds)
    
    return glm


def extract_trial_betas(glm: FirstLevelModel,
                       trial_regressors: List[str],
                       nps_weights: nib.Nifti1Image) -> Dict[str, np.ndarray]:
    """
    Extract and resample trial betas to NPS grid.
    
    Parameters
    ----------
    glm : FirstLevelModel
        Fitted GLM
    trial_regressors : list of str
        List of trial regressor names
    nps_weights : Nifti1Image
        NPS weights (target grid)
    
    Returns
    -------
    dict
        Mapping of trial_regressor -> resampled beta data (1D array on NPS mask)
    """
    design_cols = glm.design_matrices_[0].columns.tolist()
    
    # NPS mask
    nps_data = np.squeeze(nps_weights.get_fdata())
    nps_mask = nps_data != 0
    
    trial_betas = {}
    
    for trial_reg in trial_regressors:
        # Find column in design matrix
        matching_cols = [col for col in design_cols if trial_reg in col]
        if not matching_cols:
            log(f"    ⚠ Trial {trial_reg} not found in design", "WARNING")
            continue
        
        col_name = matching_cols[0]
        col_idx = design_cols.index(col_name)
        
        # Build contrast vector
        contrast_vec = np.zeros(len(design_cols))
        contrast_vec[col_idx] = 1.0
        
        # Compute beta map
        beta_map = glm.compute_contrast(contrast_vec, output_type='effect_size')
        
        # Resample to NPS grid
        beta_resampled = resample_to_img(
            beta_map,
            nps_weights,
            interpolation='linear',
            copy=True,
            force_resample=True
        )
        
        # Squeeze and extract on mask
        beta_data = np.squeeze(beta_resampled.get_fdata())
        beta_masked = beta_data[nps_mask]
        
        # Replace NaN/Inf
        if np.any(~np.isfinite(beta_masked)):
            beta_masked = np.nan_to_num(beta_masked, nan=0.0, posinf=0.0, neginf=0.0)
        
        trial_betas[trial_reg] = beta_masked
    
    return trial_betas


def score_trial_betas(trial_betas: Dict[str, np.ndarray],
                     nps_weights: nib.Nifti1Image) -> Dict[str, float]:
    """
    Compute NPS scores for each trial.
    
    Parameters
    ----------
    trial_betas : dict
        Mapping of trial_regressor -> beta data on NPS mask
    nps_weights : Nifti1Image
        NPS weights
    
    Returns
    -------
    dict
        Mapping of trial_regressor -> BR score
    """
    # Get weights on mask
    nps_data = np.squeeze(nps_weights.get_fdata())
    nps_mask = nps_data != 0
    weights_masked = nps_data[nps_mask]
    
    trial_scores = {}
    
    for trial_reg, beta_masked in trial_betas.items():
        # Compute dot product
        br_score = np.dot(weights_masked, beta_masked)
        trial_scores[trial_reg] = float(br_score)
    
    return trial_scores


def process_subject(config: Dict,
                   inventory: Dict,
                   subject: str,
                   work_dir: Path,
                   output_dir: Path,
                   qc_dir: Path,
                   nps_weights: nib.Nifti1Image) -> bool:
    """
    Process single subject: trial-wise GLM and scoring.
    
    Parameters
    ----------
    config : dict
        Configuration
    inventory : dict
        File inventory
    subject : str
        Subject ID
    work_dir : Path
        Working directory
    output_dir : Path
        Output directory
    qc_dir : Path
        QC directory
    nps_weights : Nifti1Image
        NPS weights
    
    Returns
    -------
    bool
        Success status
    """
    log(f"Processing {subject}")
    
    temp_mapping = config['glm'].get('temp_celsius_mapping', {})
    tr = config['glm']['tr']
    hrf_model = config['glm']['hrf']['model']
    high_pass_sec = config['glm']['high_pass_sec']
    high_pass_hz = 1.0 / high_pass_sec
    pain_threshold = config.get('behavior', {}).get('vas_pain_threshold', 100.0)
    
    # Output directory
    subject_output_dir = output_dir / "nps_scores" / subject
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process runs sequentially
    # Note: EEG-rejected trials are already filtered in source events files
    # by split_events_to_runs.py, so no need to load drop log here
    drop_log_df = pd.DataFrame()

    all_trial_results = []
    global_trial_idx = 0
    
    for run_key in sorted(inventory['runs'].keys()):
        run_data = inventory['runs'][run_key]
        run_num = run_data['run_number']
        
        if not run_data['complete']:
            log(f"  Run {run_num}: Skipping incomplete run", "WARNING")
            continue
        
        log(f"  Run {run_num}:")
        
        try:
            # Get file paths
            bold_path = Path(run_data['files']['bold']['path'])
            mask_path = Path(run_data['files']['mask']['path'])
            events_path = Path(run_data['files']['events']['path'])
            
            # Load confounds
            confounds_path = work_dir / "firstlevel" / subject / f"run-{run_num:02d}_confounds_24hmp_outliers.tsv"
            if not confounds_path.exists():
                log(f"    ✗ Confounds not found", "ERROR")
                continue
            
            confounds = pd.read_csv(confounds_path, sep='\t')
            
            # Load original events for behavioral data
            original_events = pd.read_csv(events_path, sep='\t')
            
            # Create trial-wise events, removing EEG-dropped trials when applicable
            trial_events, trial_info = create_trial_events(
                events_path=events_path,
                run_num=run_num,
                global_trial_offset=global_trial_idx,
                drop_log=drop_log_df,
                subject=subject,
            )

            if trial_info.empty:
                log(
                    f"    ⚠ No remaining temperature trials after applying EEG drop log; skipping run",
                    "WARNING",
                )
                continue
            
            n_trials = len(trial_info)
            log(f"    Trials: {n_trials}")
            log(f"    Fitting trial-wise GLM...")
            
            # Fit GLM
            glm = fit_trial_glm(
                bold_path,
                mask_path,
                trial_events,
                confounds,
                tr,
                hrf_model,
                high_pass_hz
            )
            
            n_regressors = glm.design_matrices_[0].shape[1]
            log(f"    Regressors: {n_regressors} (trials + nuisance + confounds + drift)")
            
            # Extract trial betas
            log(f"    Extracting and resampling {n_trials} trial betas...")
            trial_regressors = trial_info['trial_regressor'].tolist()
            trial_betas = extract_trial_betas(glm, trial_regressors, nps_weights)
            
            log(f"    Extracted {len(trial_betas)}/{n_trials} trials")
            
            # Score trials
            log(f"    Computing NPS scores...")
            trial_scores = score_trial_betas(trial_betas, nps_weights)
            
            # Link to behavioral data
            for idx, row in trial_info.iterrows():
                trial_reg = row['trial_regressor']
                
                if trial_reg not in trial_scores:
                    continue
                
                # Extract behavioral data
                temp_label = row['trial_type']
                temp_celsius = temp_mapping.get(temp_label, np.nan)
                
                # Get VAS rating
                vas_rating = row.get('vas_0_200', np.nan)
                if pd.isna(vas_rating) and 'rating' in row:
                    vas_rating = row['rating']
                
                # Pain binary (could be from events or threshold VAS)
                pain_binary = int(vas_rating > pain_threshold) if not pd.isna(vas_rating) else None
                
                trial_result = {
                    'subject': subject,
                    'run': run_num,
                    'trial_idx_global': row['trial_idx_global'],
                    'trial_idx_run': row['trial_idx_run'],
                    'temp_label': temp_label,
                    'temp_celsius': temp_celsius,
                    'vas_rating': vas_rating,
                    'pain_binary': pain_binary,
                    'br_score': trial_scores[trial_reg]
                }
                
                all_trial_results.append(trial_result)
            
            log(f"    ✓ Success: {len(trial_scores)} trials scored")
            
            # Update global trial index
            global_trial_idx += n_trials
            
        except Exception as e:
            log(f"    ✗ Failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_trial_results) == 0:
        log(f"  ✗ No trials processed for {subject}", "ERROR")
        return False
    
    # Create results dataframe
    results_df = pd.DataFrame(all_trial_results)
    
    log(f"  Total trials: {len(results_df)}")
    
    # Compute QC metrics
    log(f"  Computing QC metrics...")
    
    # Check for finite values
    n_finite = results_df['br_score'].notna().sum()
    log(f"    Finite BR scores: {n_finite}/{len(results_df)}")
    
    # Temperature-BR correlation
    valid_temp = results_df.dropna(subset=['temp_celsius', 'br_score'])
    if len(valid_temp) >= 10:
        corr_temp, p_temp = spearmanr(valid_temp['temp_celsius'], valid_temp['br_score'])
        log(f"    Temperature-BR correlation: r={corr_temp:.3f}, p={p_temp:.4f}")
    else:
        corr_temp, p_temp = np.nan, np.nan
    
    # VAS-BR correlation
    valid_vas = results_df.dropna(subset=['vas_rating', 'br_score'])
    if len(valid_vas) >= 10:
        corr_vas, p_vas = spearmanr(valid_vas['vas_rating'], valid_vas['br_score'])
        log(f"    VAS-BR correlation: r={corr_vas:.3f}, p={p_vas:.4f}")
    else:
        corr_vas, p_vas = np.nan, np.nan
    
    # ROC AUC for pain classification
    valid_roc = results_df.dropna(subset=['pain_binary', 'br_score'])
    if len(valid_roc) >= 10 and valid_roc['pain_binary'].nunique() == 2:
        try:
            auc = roc_auc_score(valid_roc['pain_binary'], valid_roc['br_score'])
            log(f"    ROC AUC (pain vs no-pain): {auc:.3f}")
        except Exception:
            auc = np.nan
            log(f"    ⚠ Could not compute AUC", "WARNING")
    else:
        auc = np.nan
        log(f"    ⚠ Insufficient data for AUC", "WARNING")
    
    # Save trial results
    results_path = subject_output_dir / "trial_br.tsv"
    results_df.to_csv(results_path, sep='\t', index=False, float_format='%.6f')
    log(f"  Saved: {results_path.name}")
    
    # Save metadata
    metadata = {
        'subject': subject,
        'n_trials': len(results_df),
        'n_runs': results_df['run'].nunique(),
        'pain_threshold': float(pain_threshold),
        'qc': {
            'temp_br_correlation': float(corr_temp) if not np.isnan(corr_temp) else None,
            'temp_br_pvalue': float(p_temp) if not np.isnan(p_temp) else None,
            'vas_br_correlation': float(corr_vas) if not np.isnan(corr_vas) else None,
            'vas_br_pvalue': float(p_vas) if not np.isnan(p_vas) else None,
            'roc_auc': float(auc) if not np.isnan(auc) else None
        },
        'br_stats': {
            'mean': float(results_df['br_score'].mean()),
            'std': float(results_df['br_score'].std()),
            'min': float(results_df['br_score'].min()),
            'max': float(results_df['br_score'].max())
        }
    }
    
    metadata_path = subject_output_dir / "trial_scoring_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    log(f"  Saved: {metadata_path.name}")
    
    # QC report
    qc_data = {
        'subject': subject,
        'n_trials': len(results_df),
        'n_finite': n_finite,
        **metadata['qc'],
        **metadata['br_stats']
    }
    
    qc_path = qc_dir / f"{subject}_trial_glm_qc.tsv"
    pd.DataFrame([qc_data]).to_csv(qc_path, sep='\t', index=False, float_format='%.6f')
    log(f"  Saved QC: {qc_path.name}")
    
    return True


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Trial-wise GLM and NPS scoring (optional)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects
  python 08_optional_trials_glm_and_scoring.py
  
  # Process specific subject
  python 08_optional_trials_glm_and_scoring.py --subject sub-0001

Note: This is computationally intensive (fits GLM with ~66 trial regressors).
      Expect 10-30 minutes per subject depending on hardware.
        """
    )
    
    parser.add_argument('--config', default='00_config.yaml',
                       help='Path to configuration file (default: 00_config.yaml)')
    parser.add_argument('--subject', default=None,
                       help='Process specific subject (default: all from config)')
    parser.add_argument('--work-dir', default='work',
                       help='Working directory (default: work)')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--qc-dir', default='qc',
                       help='QC directory (default: qc)')
    parser.add_argument('--nps-weights', default=None,
                       help='Path to NPS weights (default: from config)')
    
    args = parser.parse_args()
    
    # Load configuration
    log("=" * 70)
    log("TRIAL-WISE GLM AND NPS SCORING (OPTIONAL)")
    log("=" * 70)
    log("⚠ This is computationally intensive - expect 10-30 min per subject", "WARNING")
    log("")
    
    try:
        config = load_config(args.config)
        log(f"Loaded config: {args.config}")
    except Exception as e:
        log(f"Failed to load config: {e}", "ERROR")
        return 1
    
    # Get NPS weights path
    if args.nps_weights:
        nps_weights_path = Path(args.nps_weights)
    else:
        if 'resources' in config and 'nps_weights_path' in config['resources']:
            nps_weights_path = Path(config['resources']['nps_weights_path'])
        else:
            log("NPS weights path not found in config", "ERROR")
            return 1
    
    # Load NPS weights
    try:
        nps_weights = nib.load(str(nps_weights_path))
        log(f"Loaded NPS weights: {nps_weights_path.name}")
    except Exception as e:
        log(f"Failed to load NPS weights: {e}", "ERROR")
        return 1
    
    log("")
    
    # Determine subjects
    if args.subject:
        subjects = [args.subject]
        log(f"Processing single subject: {args.subject}")
    else:
        subjects = config['subjects']
        log(f"Processing {len(subjects)} subject(s) from config")
    
    # Setup directories
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    qc_dir = Path(args.qc_dir)
    index_dir = work_dir / "index"
    
    qc_dir.mkdir(parents=True, exist_ok=True)
    
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
            with open(inventory_path, 'r') as f:
                inventory = json.load(f)
        except Exception as e:
            log(f"Failed to load inventory: {e}", "ERROR")
            all_success = False
            continue
        
        # Process subject
        try:
            success = process_subject(
                config,
                inventory,
                subject,
                work_dir,
                output_dir,
                qc_dir,
                nps_weights
            )
            
            if not success:
                all_success = False
                
        except Exception as e:
            log(f"Processing failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            all_success = False
            continue
    
    # Final summary
    log("")
    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    
    if all_success:
        log("✓ All subjects processed successfully")
        log(f"Trial-wise scores in: {output_dir}/nps_scores/")
        log(f"QC reports in: {qc_dir}/")
        return 0
    else:
        log("✗ Some subjects failed processing", "WARNING")
        return 1


if __name__ == '__main__':
    sys.exit(main())
