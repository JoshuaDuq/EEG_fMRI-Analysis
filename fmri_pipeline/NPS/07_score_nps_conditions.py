#!/usr/bin/env python3
"""
07_score_nps_conditions.py - Compute NPS signature responses per temperature.

Purpose:
    Apply NPS weights to harmonized beta maps to compute signature responses
    (brain response scores) for each subject × temperature condition. Extract
    behavioral ratings and validate monotonic relationship with temperature.

Inputs:
    - outputs/nps_ready/sub-<ID>/beta_temp*_onNPSgrid.nii.gz: Harmonized betas
    - NPS weights file: weights_NSF_grouppred_cvpcr.nii.gz
    - Events files: For VAS ratings extraction
    - 00_config.yaml: Configuration file

Outputs:
    - outputs/nps_scores/sub-<ID>/level_br.tsv: Signature scores per condition
    - outputs/nps_scores/sub-<ID>/scoring_metadata.json: Scoring details
    - qc/sub-<ID>_nps_scores_qc.tsv: QC metrics

Acceptance Criteria:
    - All BR values finite (no NaN/Inf)
    - Monotonic trend: BR increases with temperature
    - Monotonic trend: BR correlates with VAS ratings
    - Proper masking to non-zero NPS voxels

Exit codes:
    0 - All subjects processed successfully
    1 - Processing failures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import re

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr

from config_loader import load_config


def log(msg: str, level: str = "INFO"):
    """Print log message with level prefix."""
    print(f"[{level}] {msg}", flush=True)


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
            log(f"  ⚠ drop log missing 'original_index' at {drop_path}; ignoring", "WARNING")
            return pd.DataFrame()
        return df
    except Exception as exc:  # pragma: no cover
        log(f"  ⚠ Failed to read drop log {drop_path}: {exc}", "WARNING")
        return pd.DataFrame()


def _parse_run_number(events_path: Path) -> int:
    match = re.search(r"run-(\d+)", events_path.name)
    if match:
        return int(match.group(1))
    return -1


def _filter_events_with_drop_log(events: pd.DataFrame,
                                drop_log: pd.DataFrame,
                                run_num: int) -> pd.DataFrame:
    if drop_log.empty:
        return events

    filtered = events.copy()

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
                log(f"    Run {run_num:02d}: removing {mask.sum()} events per EEG drop log", "INFO")
                filtered = filtered[~mask].reset_index(drop=True)

    if 'original_index' in drop_log.columns and len(filtered) > 0:
        drop_indices = set(int(idx) for idx in drop_log['original_index'].tolist())
        mask = filtered.index.to_series().isin(drop_indices)
        if mask.any():
            log(
                f"    Run {run_num:02d}: removing {mask.sum()} events per EEG drop log (index match)",
                "INFO",
            )
            filtered = filtered[~mask].reset_index(drop=True)

    return filtered


def load_nps_weights(weights_path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    """
    Load NPS weights and create mask.
    
    Parameters
    ----------
    weights_path : Path
        Path to NPS weights NIfTI
    
    Returns
    -------
    tuple of (Nifti1Image, np.ndarray)
        (weights_img, mask_array)
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"NPS weights not found: {weights_path}")
    
    weights_img = nib.load(str(weights_path))
    weights_data = weights_img.get_fdata()
    
    # Squeeze to 3D if needed
    weights_data = np.squeeze(weights_data)
    
    # Create mask (non-zero voxels)
    mask = weights_data != 0
    
    n_voxels = np.sum(mask)
    log(f"Loaded NPS weights: {weights_path.name}")
    log(f"  Shape: {weights_data.shape}")
    log(f"  Non-zero voxels: {n_voxels:,}")
    log(f"  Weight range: [{weights_data[mask].min():.6f}, {weights_data[mask].max():.6f}]")
    log(f"  Weight sum: {weights_data[mask].sum():.6f}")
    
    return weights_img, mask


def compute_signature_response(beta_img: nib.Nifti1Image,
                               weights_img: nib.Nifti1Image,
                               mask: np.ndarray) -> Dict:
    """
    Compute NPS signature response (dot product).
    
    Parameters
    ----------
    beta_img : Nifti1Image
        Beta map (on NPS grid)
    weights_img : Nifti1Image
        NPS weights
    mask : np.ndarray
        Mask of non-zero weight voxels
    
    Returns
    -------
    dict
        Signature response and diagnostics
    """
    # Load data
    beta_data = beta_img.get_fdata()
    weights_data = weights_img.get_fdata()
    
    # Squeeze to 3D if needed
    beta_data = np.squeeze(beta_data)
    weights_data = np.squeeze(weights_data)
    
    # Validate shapes match
    if beta_data.shape != weights_data.shape:
        raise ValueError(
            f"Shape mismatch: beta={beta_data.shape}, weights={weights_data.shape}"
        )
    
    # Extract masked values
    beta_masked = beta_data[mask]
    weights_masked = weights_data[mask]
    
    # Check for invalid values in beta
    n_beta_nan = np.sum(np.isnan(beta_masked))
    n_beta_inf = np.sum(np.isinf(beta_masked))
    
    if n_beta_nan > 0 or n_beta_inf > 0:
        log(f"    ⚠ Beta has {n_beta_nan} NaNs and {n_beta_inf} Infs in mask", "WARNING")
        # Replace with 0 for robustness
        beta_masked = np.nan_to_num(beta_masked, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute dot product (brain response)
    br_score = np.dot(weights_masked, beta_masked)
    
    # Diagnostics
    result = {
        'br_score': float(br_score),
        'n_voxels': int(np.sum(mask)),
        'n_beta_nonzero': int(np.sum(beta_masked != 0)),
        'beta_mean': float(beta_masked.mean()),
        'beta_std': float(beta_masked.std()),
        'beta_min': float(beta_masked.min()),
        'beta_max': float(beta_masked.max()),
        'n_beta_nan': int(n_beta_nan),
        'n_beta_inf': int(n_beta_inf),
        'br_is_finite': bool(np.isfinite(br_score))
    }
    
    return result


def extract_vas_ratings(events_paths: List[Path],
                       temp_label: str,
                       temp_celsius: float,
                       drop_log: pd.DataFrame) -> Dict:
    """
    Extract VAS ratings for a specific temperature from events files.
    
    Parameters
    ----------
    events_paths : list of Path
        Paths to events TSV files (across runs)
    temp_label : str
        Temperature label (e.g., 'temp44p3')
    temp_celsius : float
        Temperature in Celsius
    
    Returns
    -------
    dict
        VAS statistics
    """
    vas_values = []
    
    for events_path in events_paths:
        if not events_path.exists():
            continue
        
        try:
            events = pd.read_csv(events_path, sep='\t')
            run_num = _parse_run_number(events_path)
            events = _filter_events_with_drop_log(events, drop_log, run_num)
            
            # Filter to this temperature
            temp_events = events[events['trial_type'] == temp_label]
            
            # Extract VAS ratings (column might be 'vas_0_200' or 'rating')
            if 'vas_0_200' in events.columns:
                vas_col = 'vas_0_200'
            elif 'rating' in events.columns:
                vas_col = 'rating'
            else:
                continue
            
            # Get ratings for this temp
            temp_vas = temp_events[vas_col].dropna()
            vas_values.extend(temp_vas.tolist())
            
        except Exception as e:
            log(f"    ⚠ Failed to extract VAS from {events_path.name}: {e}", "WARNING")
            continue
    
    # Compute statistics
    if len(vas_values) > 0:
        vas_stats = {
            'mean_vas': float(np.mean(vas_values)),
            'std_vas': float(np.std(vas_values)),
            'median_vas': float(np.median(vas_values)),
            'min_vas': float(np.min(vas_values)),
            'max_vas': float(np.max(vas_values)),
            'n_trials': len(vas_values)
        }
    else:
        # No VAS data available
        vas_stats = {
            'mean_vas': np.nan,
            'std_vas': np.nan,
            'median_vas': np.nan,
            'min_vas': np.nan,
            'max_vas': np.nan,
            'n_trials': 0
        }
    
    return vas_stats


def get_events_paths(inventory: Dict) -> List[Path]:
    """Extract events file paths from inventory."""
    events_paths = []
    for run_key in sorted(inventory['runs'].keys()):
        run_data = inventory['runs'][run_key]
        if run_data['complete'] and 'events' in run_data['files']:
            events_path = Path(run_data['files']['events']['path'])
            if events_path.exists():
                events_paths.append(events_path)
    return events_paths


def process_subject(config: Dict,
                   inventory: Dict,
                   subject: str,
                   nps_ready_dir: Path,
                   output_dir: Path,
                   qc_dir: Path,
                   weights_img: nib.Nifti1Image,
                   mask: np.ndarray) -> bool:
    """
    Process single subject: score all temperatures.
    
    Parameters
    ----------
    config : dict
        Configuration
    inventory : dict
        File inventory
    subject : str
        Subject ID
    nps_ready_dir : Path
        Directory with harmonized betas
    output_dir : Path
        Output directory
    qc_dir : Path
        QC directory
    weights_img : Nifti1Image
        NPS weights
    mask : np.ndarray
        NPS mask
    
    Returns
    -------
    bool
        Success status
    """
    log(f"Processing {subject}")
    
    temp_labels = config['glm']['temp_labels']
    temp_mapping = config['glm'].get('temp_celsius_mapping', {})
    pain_threshold = config.get('behavior', {}).get('vas_pain_threshold', 100.0)
    
    # Get events paths for VAS extraction
    events_paths = get_events_paths(inventory)
    log(f"  Found {len(events_paths)} events files")

    # Note: EEG-rejected trials are already filtered in source events files
    # by split_events_to_runs.py, so no need to load drop log here
    drop_log = pd.DataFrame()
    
    # Create output directory
    subject_output_dir = output_dir / "nps_scores" / subject
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results = []
    scoring_metadata = {
        'subject': subject,
        'n_temperatures': len(temp_labels),
        'temperatures': {}
    }
    
    # Process each temperature
    for temp in temp_labels:
        log(f"  {temp}:")
        
        try:
            # Get temperature in Celsius
            temp_celsius = temp_mapping.get(temp, np.nan)
            
            # Find harmonized beta map
            beta_path = nps_ready_dir / subject / f"beta_{temp}_onNPSgrid.nii.gz"
            
            if not beta_path.exists():
                log(f"    ✗ Beta map not found: {beta_path.name}", "WARNING")
                continue
            
            # Load beta map
            beta_img = nib.load(str(beta_path))
            log(f"    Beta shape: {beta_img.shape}")
            
            # Compute signature response
            sig_result = compute_signature_response(beta_img, weights_img, mask)
            
            br_score = sig_result['br_score']
            log(f"    BR score: {br_score:.6f}")
            
            if not sig_result['br_is_finite']:
                log(f"    ✗ BR score is not finite!", "ERROR")
                continue
            
            # Extract VAS ratings
            vas_stats = extract_vas_ratings(events_paths, temp, temp_celsius, drop_log)
            
            if vas_stats['n_trials'] > 0:
                log(f"    VAS: {vas_stats['mean_vas']:.2f} ± {vas_stats['std_vas']:.2f} (n={vas_stats['n_trials']})")
            else:
                log(f"    ⚠ No VAS ratings found", "WARNING")
            
            # Store result
            result_row = {
                'subject': subject,
                'temp_label': temp,
                'temp_celsius': temp_celsius,
                'br_score': br_score,
                'mean_vas': vas_stats['mean_vas'],
                'std_vas': vas_stats['std_vas'],
                'median_vas': vas_stats['median_vas'],
                'n_trials': vas_stats['n_trials']
            }
            results.append(result_row)
            
            # Store metadata
            scoring_metadata['temperatures'][temp] = {
                'temp_celsius': temp_celsius,
                'br_score': br_score,
                'vas_stats': vas_stats,
                'beta_stats': {
                    'mean': sig_result['beta_mean'],
                    'std': sig_result['beta_std'],
                    'min': sig_result['beta_min'],
                    'max': sig_result['beta_max']
                },
                'n_voxels': sig_result['n_voxels'],
                'n_beta_nonzero': sig_result['n_beta_nonzero']
            }
            
            log(f"    ✓ Success")
            
        except Exception as e:
            log(f"    ✗ Failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            continue
    
    if len(results) == 0:
        log(f"  ✗ No results for {subject}", "ERROR")
        return False
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by temperature
    results_df = results_df.sort_values('temp_celsius')
    
    # Check monotonicity
    log(f"  Validating results:")
    
    # Check for finite values
    if results_df['br_score'].isna().any():
        log(f"    ⚠ Contains NaN BR scores", "WARNING")
    
    # Check temperature-BR correlation
    valid_rows = results_df.dropna(subset=['temp_celsius', 'br_score'])
    if len(valid_rows) >= 3:
        corr_temp_br, p_temp_br = spearmanr(valid_rows['temp_celsius'], valid_rows['br_score'])
        log(f"    Temperature-BR correlation: r={corr_temp_br:.3f}, p={p_temp_br:.4f}")
        
        if corr_temp_br > 0:
            log(f"    ✓ Positive temperature-BR trend")
        else:
            log(f"    ⚠ Negative or flat temperature-BR trend", "WARNING")
        
        scoring_metadata['qc'] = {
            'temp_br_correlation': float(corr_temp_br),
            'temp_br_pvalue': float(p_temp_br)
        }
    else:
        log(f"    ⚠ Too few valid temperatures for correlation", "WARNING")
        scoring_metadata['qc'] = {'temp_br_correlation': None}
    
    # Check VAS-BR correlation
    valid_vas_rows = results_df.dropna(subset=['mean_vas', 'br_score'])
    if len(valid_vas_rows) >= 3:
        corr_vas_br, p_vas_br = spearmanr(valid_vas_rows['mean_vas'], valid_vas_rows['br_score'])
        log(f"    VAS-BR correlation: r={corr_vas_br:.3f}, p={p_vas_br:.4f}")
        
        if corr_vas_br > 0:
            log(f"    ✓ Positive VAS-BR trend")
        else:
            log(f"    ⚠ Negative or flat VAS-BR trend", "WARNING")
        
        scoring_metadata['qc']['vas_br_correlation'] = float(corr_vas_br)
        scoring_metadata['qc']['vas_br_pvalue'] = float(p_vas_br)
    
    # Save results
    results_path = subject_output_dir / "level_br.tsv"
    results_df.to_csv(results_path, sep='\t', index=False, float_format='%.6f')
    log(f"  Saved: {results_path.name}")
    
    # Save metadata
    metadata_path = subject_output_dir / "scoring_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(scoring_metadata, f, indent=2)
    log(f"  Saved: {metadata_path.name}")
    
    # QC report
    qc_data = {
        'subject': subject,
        'n_temperatures': len(results),
        'br_mean': float(results_df['br_score'].mean()),
        'br_std': float(results_df['br_score'].std()),
        'br_min': float(results_df['br_score'].min()),
        'br_max': float(results_df['br_score'].max()),
        'all_finite': bool(results_df['br_score'].notna().all()),
        'pain_threshold': float(pain_threshold),
        **scoring_metadata.get('qc', {})
    }
    
    qc_path = qc_dir / f"{subject}_nps_scores_qc.tsv"
    pd.DataFrame([qc_data]).to_csv(qc_path, sep='\t', index=False, float_format='%.6f')
    log(f"  Saved QC: {qc_path.name}")
    
    return True


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Compute NPS signature responses per temperature',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects
  python 07_score_nps_conditions.py
  
  # Process specific subject
  python 07_score_nps_conditions.py --subject sub-0001
        """
    )
    
    parser.add_argument('--config', default='00_config.yaml',
                       help='Path to configuration file (default: 00_config.yaml)')
    parser.add_argument('--subject', default=None,
                       help='Process specific subject (default: all from config)')
    parser.add_argument('--nps-ready-dir', default='outputs/nps_ready',
                       help='Directory with harmonized betas (default: outputs/nps_ready)')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--qc-dir', default='qc',
                       help='QC directory (default: qc)')
    parser.add_argument('--nps-weights', default=None,
                       help='Path to NPS weights (default: from config)')
    
    args = parser.parse_args()
    
    # Load configuration
    log("=" * 70)
    log("COMPUTE NPS SIGNATURE RESPONSES")
    log("=" * 70)
    
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
        weights_img, mask = load_nps_weights(nps_weights_path)
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
    nps_ready_dir = Path(args.nps_ready_dir)
    output_dir = Path(args.output_dir)
    qc_dir = Path(args.qc_dir)
    work_dir = Path('work')
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
        
        # Load inventory (for events paths)
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
                nps_ready_dir,
                output_dir,
                qc_dir,
                weights_img,
                mask
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
        log(f"NPS scores in: {output_dir}/nps_scores/")
        log(f"QC reports in: {qc_dir}/")
        return 0
    else:
        log("✗ Some subjects failed processing", "WARNING")
        return 1


if __name__ == '__main__':
    sys.exit(main())
