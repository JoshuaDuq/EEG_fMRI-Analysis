#!/usr/bin/env python3
"""
06_harmonize_to_nps_grid.py - Resample beta maps to NPS weights grid.

Purpose:
    Guarantee voxelwise compatibility between subject beta maps and NPS weights
    by resampling betas to the exact grid of the NPS weights. This ensures
    accurate dot-product scoring.

Inputs:
    - outputs/firstlevel/sub-<ID>/beta_temp*.nii.gz: Subject-level beta maps
    - NPS weights file (from config): weights_NSF_grouppred_cvpcr.nii.gz
    - 00_config.yaml: Configuration file

Outputs:
    - outputs/nps_ready/sub-<ID>/beta_temp44p3_onNPSgrid.nii.gz: Resampled betas
    - outputs/nps_ready/sub-<ID>/harmonization_metadata.json: Grid info
    - qc/sub-<ID>_nps_grid_check.tsv: QC metrics

Acceptance Criteria:
    - Exact shape match to NPS weights
    - Exact affine match to NPS weights
    - Overlap >98% of non-zero NPS voxels
    - No NaNs in resampled data

Exit codes:
    0 - All subjects processed successfully
    1 - Processing failures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img

from config_loader import load_config


def log(msg: str, level: str = "INFO"):
    """Print log message with level prefix."""
    print(f"[{level}] {msg}", flush=True)


def load_nps_weights(weights_path: Path) -> nib.Nifti1Image:
    """
    Load NPS weights file.
    
    Parameters
    ----------
    weights_path : Path
        Path to NPS weights NIfTI
    
    Returns
    -------
    Nifti1Image
        NPS weights image
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"NPS weights not found: {weights_path}")
    
    weights_img = nib.load(str(weights_path))
    log(f"Loaded NPS weights: {weights_path.name}")
    
    # Report grid properties
    shape = weights_img.shape
    voxel_sizes = weights_img.header.get_zooms()[:3]
    voxel_volume = np.prod(voxel_sizes)
    
    log(f"  Shape: {shape}")
    log(f"  Voxel sizes: {voxel_sizes[0]:.2f} × {voxel_sizes[1]:.2f} × {voxel_sizes[2]:.2f} mm")
    log(f"  Voxel volume: {voxel_volume:.2f} mm³")
    
    # Count non-zero voxels
    weights_data = weights_img.get_fdata()
    n_nonzero = np.sum(weights_data != 0)
    log(f"  Non-zero voxels: {n_nonzero:,}")
    
    return weights_img


def resample_beta_to_nps_grid(beta_img: nib.Nifti1Image,
                               nps_img: nib.Nifti1Image,
                               interpolation: str = 'linear') -> nib.Nifti1Image:
    """
    Resample beta map to NPS weights grid.
    
    Parameters
    ----------
    beta_img : Nifti1Image
        Beta map to resample
    nps_img : Nifti1Image
        Target NPS weights (defines grid)
    interpolation : str
        Interpolation method ('linear', 'nearest', 'continuous')
    
    Returns
    -------
    Nifti1Image
        Resampled beta map on NPS grid (3D, no singleton dimensions)
    """
    # Resample using nilearn
    resampled_beta = resample_to_img(
        beta_img,
        nps_img,
        interpolation=interpolation,
        copy=True,
        force_resample=True
    )
    
    # Ensure 3D output (squeeze any singleton dimensions)
    data = resampled_beta.get_fdata()
    data_squeezed = np.squeeze(data)
    
    # Validate that we got 3D
    if data_squeezed.ndim != 3:
        raise ValueError(
            f"Expected 3D data after resampling, got {data_squeezed.ndim}D. "
            f"Original shape: {data.shape}, squeezed: {data_squeezed.shape}"
        )
    
    # Check for and replace NaNs/Infs (can occur at edges during resampling)
    n_nans = np.sum(np.isnan(data_squeezed))
    n_infs = np.sum(np.isinf(data_squeezed))
    
    if n_nans > 0 or n_infs > 0:
        log(f"      Resampling introduced {n_nans} NaNs and {n_infs} Infs - replacing with 0", "WARNING")
        data_squeezed = np.nan_to_num(data_squeezed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create proper 3D image
    resampled_beta_3d = nib.Nifti1Image(
        data_squeezed.astype(np.float32),
        resampled_beta.affine,
        resampled_beta.header
    )
    
    return resampled_beta_3d


def validate_grid_match(beta_img: nib.Nifti1Image,
                       nps_img: nib.Nifti1Image) -> Dict:
    """
    Validate that beta map exactly matches NPS grid.
    
    Parameters
    ----------
    beta_img : Nifti1Image
        Resampled beta map
    nps_img : Nifti1Image
        NPS weights
    
    Returns
    -------
    dict
        Validation results
    """
    validation = {
        'shape_match': False,
        'affine_match': False,
        'voxel_size_match': False,
        'overlap_pct': 0.0,
        'n_nans': 0,
        'n_infs': 0,
        'valid': False,
        'warnings': []
    }
    
    # Check shape (both should be 3D)
    beta_shape = beta_img.shape
    nps_shape = nps_img.shape
    validation['beta_shape'] = list(beta_shape)
    validation['nps_shape'] = list(nps_shape)
    
    # Verify both are 3D
    if len(beta_shape) != 3:
        validation['warnings'].append(
            f"Beta map is not 3D: shape={beta_shape}"
        )
    if len(nps_shape) != 3:
        validation['warnings'].append(
            f"NPS weights not 3D: shape={nps_shape}"
        )
    
    # Check exact match
    if beta_shape == nps_shape and len(beta_shape) == 3:
        validation['shape_match'] = True
    else:
        validation['warnings'].append(
            f"Shape mismatch: beta={beta_shape}, nps={nps_shape}"
        )
    
    # Check affine
    beta_affine = beta_img.affine
    nps_affine = nps_img.affine
    
    if np.allclose(beta_affine, nps_affine, atol=1e-4):
        validation['affine_match'] = True
    else:
        validation['warnings'].append("Affine mismatch")
        validation['affine_max_diff'] = float(np.max(np.abs(beta_affine - nps_affine)))
    
    # Check voxel sizes
    beta_voxels = beta_img.header.get_zooms()[:3]
    nps_voxels = nps_img.header.get_zooms()[:3]
    validation['beta_voxel_sizes'] = [float(v) for v in beta_voxels]
    validation['nps_voxel_sizes'] = [float(v) for v in nps_voxels]
    
    if np.allclose(beta_voxels, nps_voxels, atol=0.01):
        validation['voxel_size_match'] = True
    else:
        validation['warnings'].append("Voxel size mismatch")
    
    # Check overlap with NPS non-zero voxels
    beta_data = beta_img.get_fdata()
    nps_data = nps_img.get_fdata()
    
    # Squeeze out any singleton dimensions (e.g., 4D -> 3D)
    beta_data = np.squeeze(beta_data)
    nps_data = np.squeeze(nps_data)
    
    # NPS mask (non-zero voxels)
    nps_mask = nps_data != 0
    n_nps_voxels = np.sum(nps_mask)
    
    # Beta mask (non-zero voxels)
    beta_mask = beta_data != 0
    
    # Overlap
    overlap_mask = nps_mask & beta_mask
    n_overlap = np.sum(overlap_mask)
    
    if n_nps_voxels > 0:
        overlap_pct = (n_overlap / n_nps_voxels) * 100
        validation['overlap_pct'] = float(overlap_pct)
        validation['n_nps_voxels'] = int(n_nps_voxels)
        validation['n_beta_voxels'] = int(np.sum(beta_mask))
        validation['n_overlap_voxels'] = int(n_overlap)
    else:
        validation['warnings'].append("NPS has no non-zero voxels")
    
    # Check for NaNs and Infs
    validation['n_nans'] = int(np.sum(np.isnan(beta_data)))
    validation['n_infs'] = int(np.sum(np.isinf(beta_data)))
    
    if validation['n_nans'] > 0:
        validation['warnings'].append(f"Contains {validation['n_nans']} NaN voxels")
    
    if validation['n_infs'] > 0:
        validation['warnings'].append(f"Contains {validation['n_infs']} Inf voxels")
    
    # Overall validation
    if (validation['shape_match'] and 
        validation['affine_match'] and 
        validation['overlap_pct'] >= 98.0 and
        validation['n_nans'] == 0 and
        validation['n_infs'] == 0):
        validation['valid'] = True
    
    return validation


def process_subject(config: Dict,
                   subject: str,
                   input_dir: Path,
                   output_dir: Path,
                   qc_dir: Path,
                   nps_weights: nib.Nifti1Image) -> Tuple[int, int]:
    """
    Process single subject: resample all beta maps to NPS grid.
    
    Parameters
    ----------
    config : dict
        Configuration
    subject : str
        Subject ID
    input_dir : Path
        Input directory (outputs/firstlevel)
    output_dir : Path
        Output directory (outputs/nps_ready)
    qc_dir : Path
        QC directory
    nps_weights : Nifti1Image
        NPS weights image (target grid)
    
    Returns
    -------
    tuple of (int, int)
        (n_success, n_total)
    """
    log(f"Processing {subject}")
    
    temp_labels = config['glm']['temp_labels']
    
    # Create output directory
    subject_output_dir = output_dir / "nps_ready" / subject
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track results
    n_success = 0
    qc_records = []
    
    harmonization_metadata = {
        'subject': subject,
        'nps_grid': {
            'shape': list(nps_weights.shape),
            'voxel_sizes': [float(v) for v in nps_weights.header.get_zooms()[:3]],
            'voxel_volume_mm3': float(np.prod(nps_weights.header.get_zooms()[:3])),
            'affine': nps_weights.affine.tolist()
        },
        'temperatures': {}
    }
    
    for temp in temp_labels:
        log(f"  {temp}:")
        
        try:
            # Find input beta map
            beta_path = input_dir / "firstlevel" / subject / f"beta_{temp}.nii.gz"
            
            if not beta_path.exists():
                log(f"    ✗ Beta map not found: {beta_path.name}", "WARNING")
                continue
            
            # Load beta map
            beta_img = nib.load(str(beta_path))
            log(f"    Original: {beta_img.shape}")
            
            # Resample to NPS grid
            resampled_beta = resample_beta_to_nps_grid(beta_img, nps_weights)
            log(f"    Resampled: {resampled_beta.shape}")
            
            # Validate
            validation = validate_grid_match(resampled_beta, nps_weights)
            
            # Report validation
            if validation['warnings']:
                for warning in validation['warnings']:
                    log(f"    ⚠ {warning}", "WARNING")
            
            log(f"    Shape match: {'✓' if validation['shape_match'] else '✗'}")
            log(f"    Affine match: {'✓' if validation['affine_match'] else '✗'}")
            log(f"    Overlap: {validation['overlap_pct']:.2f}%")
            
            if not validation['valid']:
                log(f"    ✗ Validation failed", "ERROR")
                harmonization_metadata['temperatures'][temp] = {
                    'validation': validation,
                    'success': False
                }
                continue
            
            # Save resampled beta
            output_path = subject_output_dir / f"beta_{temp}_onNPSgrid.nii.gz"
            nib.save(resampled_beta, str(output_path))
            log(f"    ✓ Saved: {output_path.name}")
            
            # Update metadata
            harmonization_metadata['temperatures'][temp] = {
                'input_path': str(beta_path),
                'output_path': str(output_path),
                'validation': validation,
                'success': True
            }
            
            # QC record
            qc_records.append({
                'subject': subject,
                'temperature': temp,
                'original_shape': str(beta_img.shape),
                'resampled_shape': str(resampled_beta.shape),
                'nps_shape': str(nps_weights.shape),
                'shape_match': validation['shape_match'],
                'affine_match': validation['affine_match'],
                'voxel_size_match': validation['voxel_size_match'],
                'overlap_pct': validation['overlap_pct'],
                'n_nps_voxels': validation.get('n_nps_voxels', 0),
                'n_beta_voxels': validation.get('n_beta_voxels', 0),
                'n_overlap_voxels': validation.get('n_overlap_voxels', 0),
                'n_nans': validation['n_nans'],
                'n_infs': validation['n_infs'],
                'valid': validation['valid']
            })
            
            n_success += 1
            
        except Exception as e:
            log(f"    ✗ Failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            harmonization_metadata['temperatures'][temp] = {
                'success': False,
                'error': str(e)
            }
            continue
    
    # Save harmonization metadata
    metadata_path = subject_output_dir / "harmonization_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(harmonization_metadata, f, indent=2)
    log(f"  Saved metadata: {metadata_path.name}")
    
    # Save QC report
    if qc_records:
        qc_df = pd.DataFrame(qc_records)
        qc_path = qc_dir / f"{subject}_nps_grid_check.tsv"
        qc_df.to_csv(qc_path, sep='\t', index=False)
        log(f"  Saved QC report: {qc_path.name}")
    
    return n_success, len(temp_labels)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Resample beta maps to NPS weights grid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects
  python 06_harmonize_to_nps_grid.py
  
  # Process specific subject
  python 06_harmonize_to_nps_grid.py --subject sub-0001
  
  # Custom NPS weights path
  python 06_harmonize_to_nps_grid.py --nps-weights path/to/weights.nii.gz
        """
    )
    
    parser.add_argument('--config', default='00_config.yaml',
                       help='Path to configuration file (default: 00_config.yaml)')
    parser.add_argument('--subject', default=None,
                       help='Process specific subject (default: all from config)')
    parser.add_argument('--input-dir', default='outputs',
                       help='Input directory (default: outputs)')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--qc-dir', default='qc',
                       help='QC directory (default: qc)')
    parser.add_argument('--nps-weights', default=None,
                       help='Path to NPS weights (default: from config)')
    
    args = parser.parse_args()
    
    # Load configuration
    log("=" * 70)
    log("HARMONIZE TO NPS GRID")
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
        # Try different config locations
        if 'nps' in config and 'weights_path' in config['nps']:
            nps_weights_path = Path(config['nps']['weights_path'])
        elif 'resources' in config and 'nps_weights_path' in config['resources']:
            nps_weights_path = Path(config['resources']['nps_weights_path'])
        else:
            log("NPS weights path not found in config", "ERROR")
            log("Please specify with --nps-weights or add to config", "ERROR")
            return 1
    
    log(f"NPS weights: {nps_weights_path}")
    
    # Load NPS weights
    try:
        nps_weights = load_nps_weights(nps_weights_path)
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
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    qc_dir = Path(args.qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    all_success = True
    
    # Process each subject
    for subject in subjects:
        log("")
        log("=" * 70)
        log(f"SUBJECT: {subject}")
        log("=" * 70)
        
        try:
            n_success, n_total = process_subject(
                config,
                subject,
                input_dir,
                output_dir,
                qc_dir,
                nps_weights
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
        log(f"NPS-ready betas in: {output_dir}/nps_ready/")
        log(f"QC reports in: {qc_dir}/")
        return 0
    else:
        log("✗ Some subjects failed processing", "WARNING")
        return 1


if __name__ == '__main__':
    sys.exit(main())
