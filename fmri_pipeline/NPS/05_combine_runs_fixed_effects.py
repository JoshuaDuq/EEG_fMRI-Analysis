#!/usr/bin/env python3
"""
05_combine_runs_fixed_effects.py - Combine run-wise betas using fixed effects.

Purpose:
    Combine run-wise beta maps into subject-level estimates using fixed-effects
    (precision-weighted) averaging. Handles balanced incomplete designs where
    not all temperatures appear in every run.

Inputs:
    - work/firstlevel/sub-<ID>/run-0<r>_beta_temp*.nii.gz: Run-wise beta maps
    - work/index/sub-<ID>_files.json: File inventory
    - 00_config.yaml: Configuration file

Outputs:
    - outputs/firstlevel/sub-<ID>/beta_temp44p3.nii.gz: Subject-level beta maps
    - outputs/firstlevel/sub-<ID>/beta_temp44p3_variance.nii.gz: Variance maps
    - outputs/firstlevel/sub-<ID>/beta_temp44p3_n_runs.nii.gz: Number of runs per voxel
    - outputs/firstlevel/sub-<ID>/combination_summary.json: Combination metadata

Acceptance Criteria:
    - 6 subject-level beta maps (one per temperature)
    - No NaNs in brain regions
    - Consistent grid/affine across all maps
    - Proper weighting by run variance

Exit codes:
    0 - All subjects processed successfully
    1 - Processing failures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from scipy import stats

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


def find_beta_maps(work_dir: Path, subject: str, temp_label: str, n_runs: int) -> List[Path]:
    """
    Find all beta maps for a given temperature across runs.
    
    Parameters
    ----------
    work_dir : Path
        Working directory
    subject : str
        Subject ID
    temp_label : str
        Temperature label
    n_runs : int
        Total number of runs
    
    Returns
    -------
    list of Path
        Paths to beta maps (may be incomplete if temp not in all runs)
    """
    subject_dir = work_dir / "firstlevel" / subject
    beta_paths = []
    
    for run_num in range(1, n_runs + 1):
        beta_path = subject_dir / f"run-{run_num:02d}_beta_{temp_label}.nii.gz"
        if beta_path.exists():
            beta_paths.append(beta_path)
    
    return beta_paths


def estimate_variance(beta_img: nib.Nifti1Image, mask: np.ndarray = None) -> np.ndarray:
    """
    Estimate variance map from beta map.
    
    Uses spatial variance in local neighborhoods as proxy for estimation variance.
    This is a heuristic when true variance is not available from GLM.
    
    Parameters
    ----------
    beta_img : Nifti1Image
        Beta map
    mask : np.ndarray, optional
        Brain mask
    
    Returns
    -------
    np.ndarray
        Estimated variance map
    """
    data = beta_img.get_fdata()
    
    # Use spatial variance in 3x3x3 neighborhood as proxy
    from scipy.ndimage import generic_filter
    
    if mask is not None:
        # Only compute in masked regions
        data_masked = data.copy()
        data_masked[~mask] = 0
    else:
        data_masked = data
    
    # Compute local variance
    var_map = generic_filter(data_masked, np.var, size=3, mode='constant')
    
    # Regularize: add small constant to avoid division by zero
    var_map = np.maximum(var_map, 1e-6)
    
    return var_map


def combine_betas_fixed_effects(beta_paths: List[Path],
                                variance_method: str = 'uniform') -> Tuple[nib.Nifti1Image, nib.Nifti1Image, nib.Nifti1Image]:
    """
    Combine beta maps using fixed-effects (precision-weighted) averaging.
    
    Parameters
    ----------
    beta_paths : list of Path
        Paths to beta maps to combine
    variance_method : str
        'uniform' = equal weights (simple mean)
        'spatial' = weight by inverse spatial variance
    
    Returns
    -------
    tuple of (Nifti1Image, Nifti1Image, Nifti1Image)
        (combined_beta, combined_variance, n_runs_map)
    """
    if len(beta_paths) == 0:
        raise ValueError("No beta maps to combine")
    
    # Load all beta maps
    beta_imgs = [nib.load(str(p)) for p in beta_paths]
    
    # Validate consistent grid
    ref_affine = beta_imgs[0].affine
    ref_shape = beta_imgs[0].shape
    
    for i, img in enumerate(beta_imgs[1:], 1):
        if not np.allclose(img.affine, ref_affine):
            raise ValueError(f"Affine mismatch in beta map {i+1}")
        if img.shape != ref_shape:
            raise ValueError(f"Shape mismatch in beta map {i+1}")
    
    # Stack beta data
    beta_data = np.stack([img.get_fdata() for img in beta_imgs], axis=-1)
    n_runs = beta_data.shape[-1]
    
    # Create brain mask (any non-zero voxel)
    mask = np.any(beta_data != 0, axis=-1)
    
    if variance_method == 'uniform':
        # Simple mean (equal weights)
        combined_data = np.mean(beta_data, axis=-1)
        
        # Variance of mean: var(mean) = mean(var) / n
        # Using sample variance across runs as estimate
        combined_var = np.var(beta_data, axis=-1, ddof=1) / n_runs
        
    elif variance_method == 'spatial':
        # Estimate variance for each run using spatial smoothness
        var_maps = np.stack([estimate_variance(img, mask) for img in beta_imgs], axis=-1)
        
        # Precision (inverse variance)
        precision = 1.0 / var_maps
        
        # Precision-weighted mean
        combined_data = np.sum(beta_data * precision, axis=-1) / np.sum(precision, axis=-1)
        
        # Combined variance: 1 / sum(precision)
        combined_var = 1.0 / np.sum(precision, axis=-1)
    
    else:
        raise ValueError(f"Unknown variance method: {variance_method}")
    
    # Count number of runs contributing to each voxel
    n_runs_map = np.sum(beta_data != 0, axis=-1).astype(np.float32)
    
    # Set non-brain voxels to zero
    combined_data[~mask] = 0
    combined_var[~mask] = 0
    n_runs_map[~mask] = 0
    
    # Replace any remaining NaNs or Infs with zero
    combined_data = np.nan_to_num(combined_data, nan=0.0, posinf=0.0, neginf=0.0)
    combined_var = np.nan_to_num(combined_var, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create output images
    combined_beta = nib.Nifti1Image(combined_data, ref_affine, beta_imgs[0].header)
    combined_variance = nib.Nifti1Image(combined_var, ref_affine, beta_imgs[0].header)
    n_runs_img = nib.Nifti1Image(n_runs_map, ref_affine, beta_imgs[0].header)
    
    return combined_beta, combined_variance, n_runs_img


def validate_output(beta_img: nib.Nifti1Image) -> Dict:
    """
    Validate combined beta map.
    
    Parameters
    ----------
    beta_img : Nifti1Image
        Combined beta map
    
    Returns
    -------
    dict
        Validation results
    """
    data = beta_img.get_fdata()
    
    # Mask: non-zero voxels
    mask = data != 0
    brain_data = data[mask]
    
    validation = {
        'shape': list(data.shape),
        'n_voxels_total': int(np.prod(data.shape)),
        'n_voxels_brain': int(np.sum(mask)),
        'n_nans': int(np.sum(np.isnan(data))),
        'n_infs': int(np.sum(np.isinf(data))),
        'valid': True,
        'warnings': []
    }
    
    if len(brain_data) > 0:
        validation['mean'] = float(brain_data.mean())
        validation['std'] = float(brain_data.std())
        validation['min'] = float(brain_data.min())
        validation['max'] = float(brain_data.max())
        validation['median'] = float(np.median(brain_data))
    else:
        validation['warnings'].append("No brain voxels found")
        validation['valid'] = False
        return validation
    
    # Check for issues
    if validation['n_nans'] > 0:
        validation['warnings'].append(f"Contains {validation['n_nans']} NaN voxels")
        validation['valid'] = False
    
    if validation['n_infs'] > 0:
        validation['warnings'].append(f"Contains {validation['n_infs']} Inf voxels")
        validation['valid'] = False
    
    if np.abs(validation['mean']) > 10:
        validation['warnings'].append(f"Suspiciously large mean: {validation['mean']:.2f}")
    
    if validation['std'] > 20:
        validation['warnings'].append(f"Suspiciously large std: {validation['std']:.2f}")
    
    return validation


def process_subject(config: Dict,
                   inventory: Dict,
                   work_dir: Path,
                   output_dir: Path,
                   variance_method: str = 'uniform') -> Tuple[int, int]:
    """
    Process single subject: combine run-wise betas.
    
    Parameters
    ----------
    config : dict
        Configuration
    inventory : dict
        File inventory
    work_dir : Path
        Working directory
    output_dir : Path
        Output directory
    variance_method : str
        Variance estimation method
    
    Returns
    -------
    tuple of (int, int)
        (n_success, n_total)
    """
    subject = inventory['subject']
    log(f"Processing {subject}")
    
    temp_labels = config['glm']['temp_labels']
    n_runs = len(inventory['runs'])
    
    log(f"  Temperature conditions: {len(temp_labels)}")
    log(f"  Total runs: {n_runs}")
    
    # Create output directory
    subject_output_dir = output_dir / "firstlevel" / subject
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    n_success = 0
    combination_summary = {
        'subject': subject,
        'n_runs': n_runs,
        'variance_method': variance_method,
        'temperatures': {}
    }
    
    # Track affine for consistency check
    ref_affine = None
    ref_shape = None
    
    for temp in temp_labels:
        log(f"  {temp}:")
        
        try:
            # Find beta maps for this temperature
            beta_paths = find_beta_maps(work_dir, subject, temp, n_runs)
            
            if len(beta_paths) == 0:
                log(f"    ✗ No beta maps found", "WARNING")
                combination_summary['temperatures'][temp] = {
                    'n_runs_found': 0,
                    'success': False,
                    'error': 'No beta maps found'
                }
                continue
            
            log(f"    Found {len(beta_paths)}/{n_runs} runs")
            
            # Combine using fixed effects
            combined_beta, combined_var, n_runs_img = combine_betas_fixed_effects(
                beta_paths,
                variance_method=variance_method
            )
            
            # Validate
            validation = validate_output(combined_beta)
            
            # Check affine consistency across temperatures
            if ref_affine is None:
                ref_affine = combined_beta.affine
                ref_shape = combined_beta.shape
            else:
                if not np.allclose(combined_beta.affine, ref_affine):
                    log(f"    ⚠ Affine differs from first temperature", "WARNING")
                if combined_beta.shape != ref_shape:
                    log(f"    ⚠ Shape differs from first temperature", "WARNING")
            
            # Report validation
            if validation['warnings']:
                for warning in validation['warnings']:
                    log(f"    ⚠ {warning}", "WARNING")
            
            if not validation['valid']:
                log(f"    ✗ Validation failed", "ERROR")
                combination_summary['temperatures'][temp] = {
                    'n_runs_found': len(beta_paths),
                    'validation': validation,
                    'success': False
                }
                continue
            
            # Report statistics
            log(f"    Mean: {validation['mean']:.4f}")
            log(f"    Std: {validation['std']:.4f}")
            log(f"    Range: [{validation['min']:.4f}, {validation['max']:.4f}]")
            log(f"    Brain voxels: {validation['n_voxels_brain']}")
            
            # Save outputs
            beta_path = subject_output_dir / f"beta_{temp}.nii.gz"
            var_path = subject_output_dir / f"beta_{temp}_variance.nii.gz"
            nruns_path = subject_output_dir / f"beta_{temp}_n_runs.nii.gz"
            
            nib.save(combined_beta, str(beta_path))
            nib.save(combined_var, str(var_path))
            nib.save(n_runs_img, str(nruns_path))
            
            log(f"    ✓ Saved: {beta_path.name}")
            
            # Update summary
            combination_summary['temperatures'][temp] = {
                'n_runs_found': len(beta_paths),
                'validation': validation,
                'success': True,
                'output_path': str(beta_path),
                'variance_path': str(var_path),
                'n_runs_path': str(nruns_path)
            }
            
            n_success += 1
            
        except Exception as e:
            log(f"    ✗ Failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            combination_summary['temperatures'][temp] = {
                'n_runs_found': len(beta_paths) if 'beta_paths' in locals() else 0,
                'success': False,
                'error': str(e)
            }
            continue
    
    # Save combination summary
    summary_path = subject_output_dir / "combination_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(combination_summary, f, indent=2)
    log(f"  Saved summary: {summary_path.name}")
    
    return n_success, len(temp_labels)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Combine run-wise betas using fixed effects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects (uniform weighting)
  python 05_combine_runs_fixed_effects.py
  
  # Process specific subject
  python 05_combine_runs_fixed_effects.py --subject sub-0001
  
  # Use spatial variance weighting (slower, more sophisticated)
  python 05_combine_runs_fixed_effects.py --subject sub-0001 --variance-method spatial
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
    parser.add_argument('--variance-method', default='uniform',
                       choices=['uniform', 'spatial'],
                       help='Variance weighting method (default: uniform)')
    
    args = parser.parse_args()
    
    # Load configuration
    log("=" * 70)
    log("COMBINE RUNS - FIXED EFFECTS")
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
    
    log(f"Variance method: {args.variance_method}")
    
    # Setup directories
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
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
                output_dir,
                args.variance_method
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
        log(f"  Temperatures processed: {n_success}/{n_total}")
        
        if n_success == n_total:
            log(f"  ✓ All temperatures successful")
        else:
            log(f"  ✗ {n_total - n_success} temperature(s) failed", "WARNING")
            all_success = False
    
    # Final summary
    log("")
    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    
    if all_success:
        log("✓ All subjects processed successfully")
        log(f"Subject-level betas ready in: {output_dir}/firstlevel/")
        return 0
    else:
        log("✗ Some subjects failed processing", "WARNING")
        return 1


if __name__ == '__main__':
    sys.exit(main())
