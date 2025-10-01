#!/usr/bin/env python3
"""
09_subject_metrics.py - Compute within-subject NPS metrics.

Purpose:
    Compute dose-response and discrimination metrics for each subject that
    mirror the analyses in Wager et al. (2013) and related NPS papers.

Inputs:
    - outputs/nps_scores/sub-<ID>/level_br.tsv: Condition-level BR scores
    - outputs/nps_scores/sub-<ID>/trial_br.tsv: Trial-level BR scores (optional)
    - 00_config.yaml: Configuration file

Outputs:
    - outputs/nps_scores/sub-<ID>/subject_metrics.tsv: All metrics
    - outputs/nps_scores/sub-<ID>/subject_metrics.json: Detailed results
    - qc/subject_metrics_summary.tsv: Across-subjects summary

Metrics Computed:
    Dose-Response (from level_br.tsv):
        - slope_BR_temp: Linear slope BR ~ temperature (°C)
        - r_BR_temp: Pearson correlation BR vs temperature
        - p_BR_temp: P-value for temperature correlation
        - r_BR_VAS: Pearson correlation BR vs VAS ratings
        - p_BR_VAS: P-value for VAS correlation
    
    Discrimination (from trial_br.tsv, if available):
        - auc_pain: ROC AUC for pain vs no-pain classification
        - auc_ci_lower: 95% CI lower bound (bootstrap)
        - auc_ci_upper: 95% CI upper bound (bootstrap)
        - forced_choice_acc: Warm vs pain forced-choice accuracy
        - forced_choice_n_pairs: Number of pairs tested

Exit codes:
    0 - All subjects processed successfully
    1 - Processing failures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

from config_loader import load_config


def log(msg: str, level: str = "INFO"):
    """Print log message with level prefix."""
    print(f"[{level}] {msg}", flush=True)


def compute_dose_response_metrics(level_df: pd.DataFrame) -> Dict:
    """
    Compute dose-response metrics from condition-level data.
    
    Parameters
    ----------
    level_df : pd.DataFrame
        Condition-level BR scores (level_br.tsv)
    
    Returns
    -------
    dict
        Dose-response metrics
    """
    metrics = {
        'n_levels': len(level_df),
        'slope_BR_temp': np.nan,
        'intercept_BR_temp': np.nan,
        'r_BR_temp': np.nan,
        'p_BR_temp': np.nan,
        'r_BR_VAS': np.nan,
        'p_BR_VAS': np.nan,
        'warnings': []
    }
    
    # Filter valid data
    valid_temp = level_df.dropna(subset=['temp_celsius', 'br_score'])
    valid_vas = level_df.dropna(subset=['mean_vas', 'br_score'])
    
    if len(valid_temp) < 3:
        metrics['warnings'].append(f"Insufficient temperature data (n={len(valid_temp)})")
        return metrics
    
    # Linear regression: BR ~ temperature
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_temp['temp_celsius'],
            valid_temp['br_score']
        )
        
        metrics['slope_BR_temp'] = float(slope)
        metrics['intercept_BR_temp'] = float(intercept)
        metrics['r_BR_temp'] = float(r_value)
        metrics['p_BR_temp'] = float(p_value)
        
    except Exception as e:
        metrics['warnings'].append(f"Failed to compute BR~temp regression: {e}")
    
    # Pearson correlation: BR vs VAS
    if len(valid_vas) >= 3:
        try:
            r_vas, p_vas = stats.pearsonr(
                valid_vas['mean_vas'],
                valid_vas['br_score']
            )
            
            metrics['r_BR_VAS'] = float(r_vas)
            metrics['p_BR_VAS'] = float(p_vas)
            
        except Exception as e:
            metrics['warnings'].append(f"Failed to compute BR~VAS correlation: {e}")
    else:
        metrics['warnings'].append(f"Insufficient VAS data (n={len(valid_vas)})")
    
    return metrics


def bootstrap_auc(y_true: np.ndarray, y_scores: np.ndarray, 
                 n_bootstraps: int = 1000, ci: float = 0.95,
                 random_state: int = 42) -> Tuple[float, float, float]:
    """
    Compute AUC with bootstrap confidence interval.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : np.ndarray
        Predicted scores
    n_bootstraps : int
        Number of bootstrap samples
    ci : float
        Confidence interval (default: 0.95)
    random_state : int
        Random seed
    
    Returns
    -------
    tuple of (float, float, float)
        (auc, ci_lower, ci_upper)
    """
    # Compute original AUC
    auc = roc_auc_score(y_true, y_scores)
    
    # Bootstrap
    rng = np.random.RandomState(random_state)
    bootstrapped_aucs = []
    
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = rng.randint(0, len(y_true), len(y_true))
        
        # Skip if bootstrap sample doesn't have both classes
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        try:
            boot_auc = roc_auc_score(y_true[indices], y_scores[indices])
            bootstrapped_aucs.append(boot_auc)
        except Exception:
            continue
    
    # Compute CI
    if len(bootstrapped_aucs) > 0:
        alpha = (1 - ci) / 2
        ci_lower = np.percentile(bootstrapped_aucs, alpha * 100)
        ci_upper = np.percentile(bootstrapped_aucs, (1 - alpha) * 100)
    else:
        ci_lower = np.nan
        ci_upper = np.nan
    
    return auc, ci_lower, ci_upper


def compute_forced_choice_accuracy(trial_df: pd.DataFrame,
                                   warm_threshold: float = 46.0,
                                   pain_threshold: float = 47.0) -> Tuple[float, int]:
    """
    Compute forced-choice accuracy: warm vs pain trials.
    
    For each warm-pain pair, check if BR(pain) > BR(warm).
    
    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial-level data
    warm_threshold : float
        Temperature threshold for "warm" (below this)
    pain_threshold : float
        Temperature threshold for "pain" (at or above this)
    
    Returns
    -------
    tuple of (float, int)
        (accuracy, n_pairs)
    """
    # Define warm and pain trials by temperature
    warm_trials = trial_df[trial_df['temp_celsius'] < warm_threshold].copy()
    pain_trials = trial_df[trial_df['temp_celsius'] >= pain_threshold].copy()
    
    if len(warm_trials) == 0 or len(pain_trials) == 0:
        return np.nan, 0
    
    # Remove NaN scores
    warm_trials = warm_trials.dropna(subset=['br_score'])
    pain_trials = pain_trials.dropna(subset=['br_score'])
    
    if len(warm_trials) == 0 or len(pain_trials) == 0:
        return np.nan, 0
    
    # Compare all pairs
    n_correct = 0
    n_total = 0
    
    for _, warm_trial in warm_trials.iterrows():
        for _, pain_trial in pain_trials.iterrows():
            if pain_trial['br_score'] > warm_trial['br_score']:
                n_correct += 1
            n_total += 1
    
    accuracy = n_correct / n_total if n_total > 0 else np.nan
    
    return accuracy, n_total


def compute_discrimination_metrics(trial_df: pd.DataFrame,
                                   pain_threshold: float = 100.0) -> Dict:
    """
    Compute discrimination metrics from trial-level data.
    
    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial-level BR scores (trial_br.tsv)
    pain_threshold : float
        VAS threshold for pain vs no-pain (default: 100)
    
    Returns
    -------
    dict
        Discrimination metrics
    """
    metrics = {
        'n_trials': len(trial_df),
        'auc_pain': np.nan,
        'auc_ci_lower': np.nan,
        'auc_ci_upper': np.nan,
        'forced_choice_acc': np.nan,
        'forced_choice_n_pairs': 0,
        'warnings': []
    }
    
    # Filter valid data
    valid_df = trial_df.dropna(subset=['br_score'])
    
    if len(valid_df) < 10:
        metrics['warnings'].append(f"Insufficient trial data (n={len(valid_df)})")
        return metrics
    
    # ROC AUC for pain classification
    # Define pain based on VAS if available, otherwise use pain_binary column
    if 'pain_binary' in valid_df.columns:
        pain_labels = valid_df['pain_binary'].values
    elif 'vas_rating' in valid_df.columns:
        pain_labels = (valid_df['vas_rating'] > pain_threshold).astype(int).values
    else:
        metrics['warnings'].append("No pain labels available (pain_binary or vas_rating)")
        pain_labels = None
    
    if pain_labels is not None:
        # Check if we have both classes
        if len(np.unique(pain_labels)) == 2:
            try:
                br_scores = valid_df['br_score'].values
                auc, ci_lower, ci_upper = bootstrap_auc(pain_labels, br_scores)
                
                metrics['auc_pain'] = float(auc)
                metrics['auc_ci_lower'] = float(ci_lower)
                metrics['auc_ci_upper'] = float(ci_upper)
                
            except Exception as e:
                metrics['warnings'].append(f"Failed to compute AUC: {e}")
        else:
            metrics['warnings'].append("Only one class present in pain labels")
    
    # Forced-choice accuracy
    if 'temp_celsius' in valid_df.columns:
        try:
            fc_acc, fc_n_pairs = compute_forced_choice_accuracy(valid_df)
            
            metrics['forced_choice_acc'] = float(fc_acc) if not np.isnan(fc_acc) else np.nan
            metrics['forced_choice_n_pairs'] = int(fc_n_pairs)
            
        except Exception as e:
            metrics['warnings'].append(f"Failed to compute forced-choice: {e}")
    else:
        metrics['warnings'].append("No temperature column for forced-choice")
    
    return metrics


def process_subject(subject: str,
                   scores_dir: Path,
                   output_dir: Path) -> Dict:
    """
    Compute all metrics for a single subject.
    
    Parameters
    ----------
    subject : str
        Subject ID
    scores_dir : Path
        Directory with NPS scores
    output_dir : Path
        Output directory (same as scores_dir)
    
    Returns
    -------
    dict
        All metrics
    """
    log(f"Processing {subject}")
    
    subject_dir = scores_dir / subject
    
    # Initialize results
    results = {
        'subject': subject,
        'has_level_data': False,
        'has_trial_data': False
    }
    
    # Load condition-level data
    level_path = subject_dir / "level_br.tsv"
    
    if not level_path.exists():
        log(f"  ✗ level_br.tsv not found", "ERROR")
        return None
    
    try:
        level_df = pd.read_csv(level_path, sep='\t')
        results['has_level_data'] = True
        log(f"  Loaded level_br.tsv: {len(level_df)} conditions")
    except Exception as e:
        log(f"  ✗ Failed to load level_br.tsv: {e}", "ERROR")
        return None
    
    # Compute dose-response metrics
    log(f"  Computing dose-response metrics...")
    dose_metrics = compute_dose_response_metrics(level_df)
    results.update(dose_metrics)
    
    # Report dose-response
    log(f"    Slope BR~°C: {dose_metrics['slope_BR_temp']:.6f}")
    log(f"    r(BR, °C): {dose_metrics['r_BR_temp']:.3f}, p={dose_metrics['p_BR_temp']:.4f}")
    
    if not np.isnan(dose_metrics['r_BR_VAS']):
        log(f"    r(BR, VAS): {dose_metrics['r_BR_VAS']:.3f}, p={dose_metrics['p_BR_VAS']:.4f}")
    
    # Load trial-level data (if exists)
    trial_path = subject_dir / "trial_br.tsv"
    
    if trial_path.exists():
        try:
            trial_df = pd.read_csv(trial_path, sep='\t')
            results['has_trial_data'] = True
            log(f"  Loaded trial_br.tsv: {len(trial_df)} trials")
            
            # Compute discrimination metrics
            log(f"  Computing discrimination metrics...")
            discrim_metrics = compute_discrimination_metrics(trial_df)
            results.update(discrim_metrics)
            
            # Report discrimination
            if not np.isnan(discrim_metrics['auc_pain']):
                log(f"    AUC: {discrim_metrics['auc_pain']:.3f} "
                    f"[{discrim_metrics['auc_ci_lower']:.3f}, {discrim_metrics['auc_ci_upper']:.3f}]")
            
            if not np.isnan(discrim_metrics['forced_choice_acc']):
                log(f"    Forced-choice: {discrim_metrics['forced_choice_acc']:.1%} "
                    f"({discrim_metrics['forced_choice_n_pairs']} pairs)")
            
        except Exception as e:
            log(f"  ⚠ Could not process trial data: {e}", "WARNING")
            results['has_trial_data'] = False
    else:
        log(f"  No trial_br.tsv (optional)")
    
    # Check for warnings
    all_warnings = []
    if 'warnings' in dose_metrics and dose_metrics['warnings']:
        all_warnings.extend(dose_metrics['warnings'])
    if results['has_trial_data'] and 'warnings' in discrim_metrics and discrim_metrics['warnings']:
        all_warnings.extend(discrim_metrics['warnings'])
    
    if all_warnings:
        for warning in all_warnings:
            log(f"    ⚠ {warning}", "WARNING")
        results['notes'] = '; '.join(all_warnings)
    else:
        results['notes'] = ''
    
    # Save subject-level metrics
    metrics_tsv_path = subject_dir / "subject_metrics.tsv"
    metrics_row = {
        'subject': subject,
        'slope_BR_temp': results.get('slope_BR_temp', np.nan),
        'r_BR_temp': results.get('r_BR_temp', np.nan),
        'p_BR_temp': results.get('p_BR_temp', np.nan),
        'r_BR_VAS': results.get('r_BR_VAS', np.nan),
        'p_BR_VAS': results.get('p_BR_VAS', np.nan),
        'auc_pain': results.get('auc_pain', np.nan),
        'auc_ci_lower': results.get('auc_ci_lower', np.nan),
        'auc_ci_upper': results.get('auc_ci_upper', np.nan),
        'forced_choice_acc': results.get('forced_choice_acc', np.nan),
        'forced_choice_n_pairs': results.get('forced_choice_n_pairs', 0),
        'n_levels': results.get('n_levels', 0),
        'n_trials': results.get('n_trials', 0),
        'notes': results.get('notes', '')
    }
    
    pd.DataFrame([metrics_row]).to_csv(metrics_tsv_path, sep='\t', index=False, float_format='%.6f')
    log(f"  Saved: {metrics_tsv_path.name}")
    
    # Save detailed JSON
    metrics_json_path = subject_dir / "subject_metrics.json"
    with open(metrics_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"  Saved: {metrics_json_path.name}")
    
    log(f"  ✓ Success")
    
    return results


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Compute within-subject NPS metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects
  python 09_subject_metrics.py
  
  # Process specific subject
  python 09_subject_metrics.py --subject sub-0001

Metrics Computed:
  Dose-Response:
    - Linear slope BR ~ temperature
    - Pearson r(BR, temperature)
    - Pearson r(BR, VAS)
  
  Discrimination (if trial data available):
    - ROC AUC with 95% CI
    - Forced-choice accuracy
        """
    )
    
    parser.add_argument('--config', default='00_config.yaml',
                       help='Path to configuration file (default: 00_config.yaml)')
    parser.add_argument('--subject', default=None,
                       help='Process specific subject (default: all from config)')
    parser.add_argument('--scores-dir', default='outputs/nps_scores',
                       help='Directory with NPS scores (default: outputs/nps_scores)')
    parser.add_argument('--qc-dir', default='qc',
                       help='QC directory (default: qc)')
    
    args = parser.parse_args()
    
    # Load configuration
    log("=" * 70)
    log("COMPUTE WITHIN-SUBJECT NPS METRICS")
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
    
    log("")
    
    # Setup directories
    scores_dir = Path(args.scores_dir)
    qc_dir = Path(args.qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    if not scores_dir.exists():
        log(f"Scores directory not found: {scores_dir}", "ERROR")
        log("Run 07_score_nps_conditions.py first", "ERROR")
        return 1
    
    # Process subjects
    all_results = []
    all_success = True
    
    for subject in subjects:
        log("")
        log("=" * 70)
        log(f"SUBJECT: {subject}")
        log("=" * 70)
        
        try:
            results = process_subject(subject, scores_dir, scores_dir)
            
            if results is None:
                all_success = False
                continue
            
            all_results.append(results)
            
        except Exception as e:
            log(f"Processing failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            all_success = False
            continue
    
    # Create summary across subjects
    if len(all_results) > 0:
        log("")
        log("=" * 70)
        log("CREATING SUMMARY")
        log("=" * 70)
        
        summary_rows = []
        for result in all_results:
            summary_row = {
                'subject': result['subject'],
                'slope_BR_temp': result.get('slope_BR_temp', np.nan),
                'r_BR_temp': result.get('r_BR_temp', np.nan),
                'p_BR_temp': result.get('p_BR_temp', np.nan),
                'r_BR_VAS': result.get('r_BR_VAS', np.nan),
                'p_BR_VAS': result.get('p_BR_VAS', np.nan),
                'auc_pain': result.get('auc_pain', np.nan),
                'forced_choice_acc': result.get('forced_choice_acc', np.nan),
                'n_levels': result.get('n_levels', 0),
                'n_trials': result.get('n_trials', 0),
                'has_trial_data': result.get('has_trial_data', False)
            }
            summary_rows.append(summary_row)
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Save summary
        summary_path = qc_dir / "subject_metrics_summary.tsv"
        summary_df.to_csv(summary_path, sep='\t', index=False, float_format='%.6f')
        log(f"Saved summary: {summary_path}")
        
        # Report group statistics
        log("")
        log("Group-level statistics:")
        
        valid_slope = summary_df['slope_BR_temp'].dropna()
        if len(valid_slope) > 0:
            log(f"  Slope BR~°C: {valid_slope.mean():.6f} ± {valid_slope.std():.6f}")
        
        valid_r_temp = summary_df['r_BR_temp'].dropna()
        if len(valid_r_temp) > 0:
            log(f"  r(BR, °C): {valid_r_temp.mean():.3f} ± {valid_r_temp.std():.3f}")
        
        valid_r_vas = summary_df['r_BR_VAS'].dropna()
        if len(valid_r_vas) > 0:
            log(f"  r(BR, VAS): {valid_r_vas.mean():.3f} ± {valid_r_vas.std():.3f}")
        
        valid_auc = summary_df['auc_pain'].dropna()
        if len(valid_auc) > 0:
            log(f"  AUC: {valid_auc.mean():.3f} ± {valid_auc.std():.3f}")
    
    # Final summary
    log("")
    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    
    if all_success:
        log("✓ All subjects processed successfully")
        log(f"Subject metrics in: {scores_dir}/sub-*/subject_metrics.tsv")
        log(f"Summary in: {qc_dir}/subject_metrics_summary.tsv")
        return 0
    else:
        log("✗ Some subjects failed processing", "WARNING")
        return 1


if __name__ == '__main__':
    sys.exit(main())
