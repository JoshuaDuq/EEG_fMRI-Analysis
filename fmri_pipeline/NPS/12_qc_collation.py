#!/usr/bin/env python3
"""
12_qc_collation.py - Consolidate QC information from all pipeline steps.

Purpose:
    Generate comprehensive quality control report consolidating artifacts
    from steps 01-06. Checks motion, grid alignment, data integrity, and versions.

Outputs:
    - outputs/qc/summary_qc.tsv: Comprehensive QC metrics per subject
    - outputs/qc/ENV.yaml: Software environment and package versions
    - outputs/qc/config_hash.json: Config parameter checksums
"""

import argparse
import hashlib
import json
import sys
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import pandas as pd
import numpy as np
import nibabel as nib
import yaml

try:
    import nilearn
    import scipy
    import sklearn
    import matplotlib
except ImportError as e:
    warnings.warn(f"Failed to import package: {e}")

from config_loader import load_config


def log(msg: str, level: str = "INFO"):
    """Print log message."""
    print(f"[{level}] {msg}", flush=True)


def compute_fd_from_confounds(confounds_tsv_path: Path) -> Dict:
    """Compute framewise displacement statistics from confounds TSV."""
    if not confounds_tsv_path.exists():
        return {'error': f'File not found'}
    
    try:
        confounds = pd.read_csv(confounds_tsv_path, sep='\t')
        
        fd_col = None
        for col in ['framewise_displacement', 'fd', 'FD']:
            if col in confounds.columns:
                fd_col = col
                break
        
        if fd_col is None:
            return {'error': 'FD column not found'}
        
        fd_values = confounds[fd_col].values
        fd_values = fd_values[~np.isnan(fd_values)]
        
        if len(fd_values) == 0:
            return {'error': 'All FD values are NaN'}
        
        return {
            'mean_fd': float(np.mean(fd_values)),
            'median_fd': float(np.median(fd_values)),
            'max_fd': float(np.max(fd_values)),
            'std_fd': float(np.std(fd_values)),
            'n_volumes': len(fd_values) + 1
        }
        
    except Exception as e:
        return {'error': str(e)}


def get_outliers_during_heat_fraction(work_dir: Path, subject: str,
                                     run_num: int, inventory: Dict,
                                     tr: Optional[float]) -> Optional[float]:
    """Compute fraction of outliers during heat stimulation."""
    try:
        confounds_path = work_dir / "firstlevel" / subject / f"run-{run_num:02d}_confounds_24hmp_outliers.tsv"
        if not confounds_path.exists():
            return None

        confounds = pd.read_csv(confounds_path, sep='\t')
        outlier_cols = [col for col in confounds.columns if col.startswith('motion_outlier')]

        if not outlier_cols:
            return 0.0

        outlier_mask = confounds[outlier_cols].sum(axis=1) > 0
        n_outliers_total = outlier_mask.sum()

        if n_outliers_total == 0:
            return 0.0

        run_key = f"run-{run_num:02d}"
        if run_key not in inventory['runs']:
            return None

        events_path = Path(inventory['runs'][run_key]['files']['events']['path'])
        if not events_path.exists():
            return None

        events = pd.read_csv(events_path, sep='\t')
        heat_events = events[events['trial_type'].str.contains('temp', case=False, na=False)]

        if len(heat_events) == 0:
            return None

        if tr is None or tr <= 0:
            log("    ⚠ TR unavailable; skipping heat outlier fraction", "WARNING")
            return None

        n_volumes = len(confounds)
        heat_mask = np.zeros(n_volumes, dtype=bool)

        for _, event in heat_events.iterrows():
            start_vol = max(0, int(event['onset'] / tr))
            end_vol = min(n_volumes, int((event['onset'] + event['duration']) / tr))
            heat_mask[start_vol:end_vol] = True

        n_outliers_during_heat = (outlier_mask & heat_mask).sum()
        return float(n_outliers_during_heat / n_outliers_total)

    except Exception as e:
        log(f"    ⚠ Failed to compute heat outliers: {e}", "WARNING")
        return None


def summarize_motion_per_subject(config: Dict, inventory: Dict, work_dir: Path,
                                 tr: Optional[float]) -> Dict:
    """Summarize motion metrics across all runs."""
    subject = inventory['subject']
    
    all_fd_means = []
    all_outlier_counts = []
    all_outlier_fracs = []
    all_heat_outlier_fracs = []
    
    for run_key, run_data in inventory['runs'].items():
        run_num = run_data['run_number']
        
        # Load confounds summary
        confounds_summary_path = work_dir / "firstlevel" / subject / f"run-{run_num:02d}_confounds_summary.json"
        if confounds_summary_path.exists():
            with open(confounds_summary_path, 'r') as f:
                summary = json.load(f)
            n_outliers = summary.get('motion_outliers_found', 0)
            n_volumes = summary.get('output_rows', 0)
        else:
            n_outliers = 0
            n_volumes = 0
        
        # Get FD stats
        confounds_raw_path = Path(run_data['files']['confounds']['path'])
        fd_stats = compute_fd_from_confounds(confounds_raw_path)
        
        # Get heat outlier fraction
        heat_outlier_frac = get_outliers_during_heat_fraction(work_dir, subject, run_num, inventory, tr)
        
        if fd_stats.get('mean_fd') is not None:
            all_fd_means.append(fd_stats['mean_fd'])
        
        all_outlier_counts.append(n_outliers)
        
        if n_volumes > 0:
            all_outlier_fracs.append(n_outliers / n_volumes)
        
        if heat_outlier_frac is not None:
            all_heat_outlier_fracs.append(heat_outlier_frac)
    
    return {
        'mean_fd_across_runs': float(np.mean(all_fd_means)) if all_fd_means else None,
        'median_fd_across_runs': float(np.median(all_fd_means)) if all_fd_means else None,
        'max_fd_across_runs': float(np.max([fd_stats.get('max_fd', 0) for run_data in inventory['runs'].values() if (fd_stats := compute_fd_from_confounds(Path(run_data['files']['confounds']['path']))).get('max_fd')])) if any((fd_stats := compute_fd_from_confounds(Path(run_data['files']['confounds']['path']))).get('max_fd') for run_data in inventory['runs'].values()) else None,
        'total_outliers': sum(all_outlier_counts),
        'mean_outlier_fraction': float(np.mean(all_outlier_fracs)) if all_outlier_fracs else None,
        'mean_heat_outlier_fraction': float(np.mean(all_heat_outlier_fracs)) if all_heat_outlier_fracs else None
    }


def _resolve_nps_weights_path(config: Dict) -> Optional[Path]:
    """Resolve NPS weights path following pipeline conventions."""
    if 'nps' in config and isinstance(config['nps'], dict) and config['nps'].get('weights_path'):
        path = Path(config['nps']['weights_path'])
        return path
    if 'resources' in config and isinstance(config['resources'], dict) and config['resources'].get('nps_weights_path'):
        path = Path(config['resources']['nps_weights_path'])
        return path
    return None


def validate_nps_grid_match(harmonization_metadata_path: Path, nps_weights_path: Path) -> Dict:
    """Validate resampled betas match NPS weights grid."""
    validation = {
        'exact_match': False,
        'n_nonzero_nps_voxels': 0,
        'temperatures_validated': []
    }

    if not harmonization_metadata_path.exists() or not nps_weights_path.exists():
        validation['error'] = 'Files not found'
        return validation

    try:
        with open(harmonization_metadata_path, 'r') as f:
            metadata = json.load(f)

        nps_img = nib.load(str(nps_weights_path))
        nps_data = nps_img.get_fdata()

        validation['n_nonzero_nps_voxels'] = int(np.sum(nps_data != 0))

        all_match = True
        for temp, temp_data in metadata.get('temperatures', {}).items():
            if not temp_data.get('success', False):
                all_match = False
                continue

            temp_validation = temp_data.get('validation', {})
            if not temp_validation.get('valid', False):
                all_match = False

            validation['temperatures_validated'].append(temp)

        validation['exact_match'] = all_match and len(validation['temperatures_validated']) > 0

    except Exception as e:
        validation['error'] = str(e)

    return validation


def check_nans_infs_in_betas(nps_ready_dir: Path, subject: str, temp_labels: List[str]) -> Dict:
    """Check for NaNs/Infs in resampled beta maps."""
    total_nans = 0
    total_infs = 0
    
    subject_dir = nps_ready_dir / subject
    
    for temp in temp_labels:
        beta_path = subject_dir / f"beta_{temp}_onNPSgrid.nii.gz"
        
        if beta_path.exists():
            try:
                beta_img = nib.load(str(beta_path))
                beta_data = beta_img.get_fdata()
                
                total_nans += int(np.sum(np.isnan(beta_data)))
                total_infs += int(np.sum(np.isinf(beta_data)))
                
            except Exception:
                pass
    
    return {
        'total_nans': total_nans,
        'total_infs': total_infs,
        'has_issues': (total_nans > 0 or total_infs > 0)
    }


def summarize_glm_diagnostics(work_dir: Path, subject: str, n_runs: int) -> Dict:
    """Summarize GLM fit diagnostics."""
    all_dofs = []
    all_r2s = []
    
    for run_num in range(1, n_runs + 1):
        diag_path = work_dir / "firstlevel" / subject / f"run-{run_num:02d}_modeldiag.json"
        
        if diag_path.exists():
            try:
                with open(diag_path, 'r') as f:
                    diag = json.load(f)
                
                if diag.get('dof') is not None:
                    all_dofs.append(diag['dof'])
                if diag.get('mean_r_squared') is not None:
                    all_r2s.append(diag['mean_r_squared'])
                    
            except Exception:
                pass
    
    return {
        'mean_dof': float(np.mean(all_dofs)) if all_dofs else None,
        'min_dof': int(np.min(all_dofs)) if all_dofs else None,
        'mean_r_squared': float(np.mean(all_r2s)) if all_r2s else None
    }


def get_package_versions() -> Dict[str, str]:
    """Get versions of all relevant packages."""
    versions = {}
    
    for pkg_name in ['numpy', 'scipy', 'pandas', 'nibabel', 'nilearn', 'sklearn', 'matplotlib', 'yaml']:
        try:
            pkg = __import__(pkg_name)
            versions[pkg_name] = pkg.__version__
        except (ImportError, AttributeError):
            versions[pkg_name] = 'not installed'
    
    return versions


def get_system_info() -> Dict[str, str]:
    """Get system information."""
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'system': platform.system()
    }


def compute_config_hash(config: Dict) -> Dict[str, str]:
    """Compute checksums for critical config parameters."""
    hashes = {}
    
    for section in ['glm', 'confounds', 'nps', 'acquisition_params']:
        if section in config:
            section_str = json.dumps(config[section], sort_keys=True)
            section_hash = hashlib.sha256(section_str.encode()).hexdigest()[:16]
            hashes[section] = section_hash
    
    return hashes


def collate_qc_for_subject(config: Dict, subject: str, work_dir: Path, outputs_dir: Path) -> Dict:
    """Collate all QC information for a subject."""
    log(f"Collating QC for {subject}")
    
    qc_summary = {
        'subject': subject,
        'critical_flags': [],
        'warnings': []
    }
    
    # Load inventory
    inventory_path = work_dir / "index" / f"{subject}_files.json"
    if not inventory_path.exists():
        qc_summary['critical_flags'].append('File inventory missing')
        return qc_summary
    
    with open(inventory_path, 'r') as f:
        inventory = json.load(f)
    
    # Motion QC
    log(f"  Summarizing motion...")
    tr = config.get('glm', {}).get('tr')
    qc_summary['motion'] = summarize_motion_per_subject(config, inventory, work_dir, tr)
    
    mean_fd = qc_summary['motion'].get('mean_fd_across_runs')
    if mean_fd is not None:
        fd_warn_thresh = config.get('qc', {}).get('motion_thresholds', {}).get('fd_mean_warn', 0.3)
        if mean_fd > fd_warn_thresh:
            qc_summary['warnings'].append(f'Mean FD ({mean_fd:.3f}) exceeds threshold ({fd_warn_thresh})')
    
    # GLM Diagnostics
    log(f"  Summarizing GLM diagnostics...")
    qc_summary['glm'] = summarize_glm_diagnostics(work_dir, subject, len(inventory['runs']))
    
    mean_dof = qc_summary['glm'].get('mean_dof')
    if mean_dof is not None and mean_dof < 50:
        qc_summary['warnings'].append(f'Low DOF ({mean_dof:.0f})')
    
    # NPS Grid Validation
    log(f"  Validating NPS grid...")
    harmonization_path = outputs_dir / "nps_ready" / subject / "harmonization_metadata.json"
    nps_weights_path = _resolve_nps_weights_path(config)
    if nps_weights_path is None:
        qc_summary['critical_flags'].append('NPS weights path missing in config')
        grid_validation = {'exact_match': False}
    else:
        grid_validation = validate_nps_grid_match(harmonization_path, nps_weights_path)
    qc_summary['nps_grid'] = {
        'exact_match': grid_validation.get('exact_match', False),
        'n_nonzero_nps_voxels': grid_validation.get('n_nonzero_nps_voxels', 0),
        'n_temperatures_validated': len(grid_validation.get('temperatures_validated', []))
    }
    
    if not grid_validation.get('exact_match', False):
        qc_summary['critical_flags'].append('NPS grid mismatch')
    
    # Check NaNs/Infs
    log(f"  Checking data integrity...")
    nps_ready_dir = outputs_dir / "nps_ready"
    temp_labels = config['glm']['temp_labels']
    
    nan_check = check_nans_infs_in_betas(nps_ready_dir, subject, temp_labels)
    qc_summary['data_integrity'] = nan_check
    
    if nan_check['total_nans'] > 0:
        qc_summary['critical_flags'].append(f'{nan_check["total_nans"]} NaN voxels in betas')
    if nan_check['total_infs'] > 0:
        qc_summary['critical_flags'].append(f'{nan_check["total_infs"]} Inf voxels in betas')
    
    # Status
    qc_summary['status'] = 'PASS' if len(qc_summary['critical_flags']) == 0 else 'FAIL'
    
    log(f"  Status: {qc_summary['status']}")
    for flag in qc_summary['critical_flags']:
        log(f"    ✗ {flag}", "ERROR")
    for warning in qc_summary['warnings']:
        log(f"    ⚠ {warning}", "WARNING")
    
    return qc_summary


def create_summary_tsv(qc_summaries: List[Dict], output_path: Path):
    """Create summary QC table."""
    rows = []
    
    for qc in qc_summaries:
        rows.append({
            'subject': qc['subject'],
            'status': qc['status'],
            'n_critical_flags': len(qc['critical_flags']),
            'n_warnings': len(qc['warnings']),
            'mean_fd': qc['motion'].get('mean_fd_across_runs'),
            'max_fd': qc['motion'].get('max_fd_across_runs'),
            'total_outliers': qc['motion'].get('total_outliers'),
            'mean_outlier_fraction': qc['motion'].get('mean_outlier_fraction'),
            'mean_heat_outlier_fraction': qc['motion'].get('mean_heat_outlier_fraction'),
            'mean_dof': qc['glm'].get('mean_dof'),
            'mean_r_squared': qc['glm'].get('mean_r_squared'),
            'nps_grid_exact_match': qc['nps_grid'].get('exact_match'),
            'n_nps_voxels_used': qc['nps_grid'].get('n_nonzero_nps_voxels'),
            'total_nans': qc['data_integrity'].get('total_nans'),
            'total_infs': qc['data_integrity'].get('total_infs'),
            'critical_flags': '; '.join(qc['critical_flags']) if qc['critical_flags'] else '',
            'warnings': '; '.join(qc['warnings']) if qc['warnings'] else ''
        })
    
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
    log(f"Saved: {output_path}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Consolidate QC from all pipeline steps')
    parser.add_argument('--config', default='00_config.yaml', help='Config file')
    parser.add_argument('--subject', default=None, help='Specific subject')
    parser.add_argument('--work-dir', default='work', help='Working directory')
    parser.add_argument('--outputs-dir', default='outputs', help='Outputs directory')
    
    args = parser.parse_args()
    
    log("=" * 70)
    log("QC COLLATION")
    log("=" * 70)
    
    # Load config
    try:
        config = load_config(args.config)
        log(f"Loaded config: {args.config}")
    except Exception as e:
        log(f"Failed to load config: {e}", "ERROR")
        return 1
    
    # Determine subjects
    subjects = [args.subject] if args.subject else config['subjects']
    log(f"Processing {len(subjects)} subject(s)")
    
    work_dir = Path(args.work_dir)
    outputs_dir = Path(args.outputs_dir)
    qc_dir = outputs_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    # Collate QC for each subject
    qc_summaries = []
    all_pass = True
    
    for subject in subjects:
        log("")
        log(f"SUBJECT: {subject}")
        log("=" * 70)
        
        try:
            qc_summary = collate_qc_for_subject(config, subject, work_dir, outputs_dir)
            qc_summaries.append(qc_summary)
            
            if qc_summary['status'] != 'PASS':
                all_pass = False
                
        except Exception as e:
            log(f"Failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            all_pass = False
    
    # Save summary TSV
    if qc_summaries:
        log("")
        log("Saving QC summary...")
        summary_path = qc_dir / "summary_qc.tsv"
        create_summary_tsv(qc_summaries, summary_path)
    
    # Save environment info
    log("Saving environment info...")
    env_path = qc_dir / "ENV.yaml"
    env_info = {'system': get_system_info(), 'packages': get_package_versions()}
    with open(env_path, 'w') as f:
        yaml.dump(env_info, f, default_flow_style=False, sort_keys=False)
    log(f"Saved: {env_path}")
    
    # Save config hash
    log("Saving config hashes...")
    hash_path = qc_dir / "config_hash.json"
    hash_info = {
        'config_version': config.get('metadata', {}).get('config_version', 'unknown'),
        'pipeline_version': config.get('metadata', {}).get('pipeline_version', 'unknown'),
        'parameter_hashes': compute_config_hash(config)
    }
    with open(hash_path, 'w') as f:
        json.dump(hash_info, f, indent=2)
    log(f"Saved: {hash_path}")
    
    # Final summary
    log("")
    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"Subjects processed: {len(qc_summaries)}")
    
    n_pass = sum(1 for qc in qc_summaries if qc['status'] == 'PASS')
    n_fail = len(qc_summaries) - n_pass
    
    log(f"PASS: {n_pass}, FAIL: {n_fail}")
    
    if all_pass:
        log("✓ All QC checks passed")
        return 0
    else:
        log("✗ Some subjects have critical issues", "WARNING")
        log(f"Review: {qc_dir}/summary_qc.tsv")
        return 1


if __name__ == '__main__':
    sys.exit(main())
