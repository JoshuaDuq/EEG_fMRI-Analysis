#!/usr/bin/env python3
"""
10_group_stats.py - Group-level statistical inference for NPS metrics.

Purpose:
    Perform group-level one-sample tests on within-subject NPS metrics.
    Tests whether the group shows significant dose-response relationships
    and discrimination ability.

Inputs:
    - outputs/nps_scores/sub-*/subject_metrics.tsv: Subject-level metrics
    - qc/subject_metrics_summary.tsv: Combined summary (optional)
    - 00_config.yaml: Configuration file

Outputs:
    - outputs/group/group_stats.tsv: Group statistics with CIs and p-values
    - outputs/group/group_diagnostics.json: Sample info and exclusions
    - outputs/group/group_stats_verbose.json: Detailed results

Statistical Tests:
    - One-sample t-test: slope > 0 (right-tailed)
    - One-sample t-test: Fisher-z(r_BR_VAS) > 0 (right-tailed)
    - One-sample t-test: AUC - 0.5 > 0 (right-tailed)
    - One-sample t-test: forced_choice_acc vs 0.5
    - Bootstrap bias-corrected 95% CIs
    - FDR correction across primary endpoints

Exit codes:
    0 - Processing completed successfully
    1 - Processing failures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def log(msg: str, level: str = "INFO"):
    """Print log message with level prefix."""
    print(f"[{level}] {msg}", flush=True)


def fisher_z_transform(r: np.ndarray) -> np.ndarray:
    """
    Fisher z-transformation for correlation coefficients.
    
    Parameters
    ----------
    r : np.ndarray
        Correlation coefficients
    
    Returns
    -------
    np.ndarray
        Fisher z-transformed values
    """
    # Clip to valid range to avoid numerical issues
    r_clipped = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))


def bias_corrected_bootstrap_ci(data: np.ndarray,
                                n_bootstrap: int = 10000,
                                ci: float = 0.95,
                                random_state: int = 42) -> Tuple[float, float, float]:
    """
    Compute bias-corrected and accelerated (BCa) bootstrap confidence interval.
    
    Parameters
    ----------
    data : np.ndarray
        Sample data
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence level
    random_state : int
        Random seed
    
    Returns
    -------
    tuple of (float, float, float)
        (mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    
    # Original statistic
    theta_hat = np.mean(data)
    
    # Bootstrap
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Bias correction factor
    z0 = stats.norm.ppf(np.mean(bootstrap_means < theta_hat))
    
    # Acceleration factor (jackknife)
    jackknife_means = []
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_means.append(np.mean(jackknife_sample))
    
    jackknife_means = np.array(jackknife_means)
    jackknife_mean = np.mean(jackknife_means)
    
    numerator = np.sum((jackknife_mean - jackknife_means) ** 3)
    denominator = 6 * (np.sum((jackknife_mean - jackknife_means) ** 2) ** 1.5)
    
    if denominator != 0:
        a = numerator / denominator
    else:
        a = 0
    
    # Adjusted percentiles
    alpha = 1 - ci
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1_alpha = stats.norm.ppf(1 - alpha / 2)
    
    p_lower = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    p_upper = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))
    
    # Clip to valid range
    p_lower = np.clip(p_lower, 0.001, 0.999)
    p_upper = np.clip(p_upper, 0.001, 0.999)
    
    ci_lower = np.percentile(bootstrap_means, p_lower * 100)
    ci_upper = np.percentile(bootstrap_means, p_upper * 100)
    
    return theta_hat, ci_lower, ci_upper


def one_sample_t_test(data: np.ndarray,
                     popmean: float = 0,
                     alternative: str = 'greater') -> Tuple[float, float, float]:
    """
    One-sample t-test.
    
    Parameters
    ----------
    data : np.ndarray
        Sample data
    popmean : float
        Null hypothesis population mean
    alternative : str
        'two-sided', 'greater', or 'less'
    
    Returns
    -------
    tuple of (float, float, float)
        (t_statistic, p_value, df)
    """
    result = stats.ttest_1samp(data, popmean, alternative=alternative)
    return result.statistic, result.pvalue, len(data) - 1


def load_subject_metrics(scores_dir: Path, subjects: List[str]) -> pd.DataFrame:
    """
    Load subject metrics from individual files.
    
    Parameters
    ----------
    scores_dir : Path
        Directory with NPS scores
    subjects : list of str
        Subject IDs
    
    Returns
    -------
    pd.DataFrame
        Combined subject metrics
    """
    metrics_list = []
    
    for subject in subjects:
        metrics_path = scores_dir / subject / "subject_metrics.tsv"
        
        if not metrics_path.exists():
            log(f"  ⚠ Metrics not found for {subject}", "WARNING")
            continue
        
        try:
            df = pd.read_csv(metrics_path, sep='\t')
            metrics_list.append(df)
        except Exception as e:
            log(f"  ⚠ Failed to load metrics for {subject}: {e}", "WARNING")
            continue
    
    if len(metrics_list) == 0:
        raise ValueError("No subject metrics could be loaded")
    
    combined_df = pd.concat(metrics_list, ignore_index=True)
    
    return combined_df


def compute_group_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute group-level statistics for all metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Subject metrics
    
    Returns
    -------
    dict
        Group statistics
    """
    results = {
        'n_subjects_total': len(df),
        'metrics': {}
    }
    
    log("")
    log("=" * 70)
    log("COMPUTING GROUP STATISTICS")
    log("=" * 70)
    
    # Metric: slope_BR_temp
    log("")
    log("1. Slope BR ~ Temperature:")
    
    slope_data = df['slope_BR_temp'].dropna().values
    n_slope = len(slope_data)
    
    if n_slope >= 3:
        mean_slope, ci_lower, ci_upper = bias_corrected_bootstrap_ci(slope_data)
        t_stat, p_val, df_val = one_sample_t_test(slope_data, popmean=0, alternative='greater')
        
        results['metrics']['slope_BR_temp'] = {
            'n': int(n_slope),
            'mean': float(mean_slope),
            'std': float(np.std(slope_data, ddof=1)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'df': float(df_val),
            'test': 'one-sample t-test (H1: slope > 0)'
        }
        
        log(f"  n = {n_slope}")
        log(f"  Mean: {mean_slope:.6f} [95% CI: {ci_lower:.6f}, {ci_upper:.6f}]")
        log(f"  t({df_val}) = {t_stat:.3f}, p = {p_val:.4f}")
    else:
        log(f"  ⚠ Insufficient data (n={n_slope})", "WARNING")
        results['metrics']['slope_BR_temp'] = {'n': int(n_slope), 'error': 'insufficient_data'}
    
    # Metric: r_BR_temp
    log("")
    log("2. Correlation r(BR, Temperature):")
    
    r_temp_data = df['r_BR_temp'].dropna().values
    n_r_temp = len(r_temp_data)
    
    if n_r_temp >= 3:
        mean_r, ci_lower, ci_upper = bias_corrected_bootstrap_ci(r_temp_data)
        
        # Fisher z-transform for test
        z_temp = fisher_z_transform(r_temp_data)
        t_stat, p_val, df_val = one_sample_t_test(z_temp, popmean=0, alternative='greater')
        
        results['metrics']['r_BR_temp'] = {
            'n': int(n_r_temp),
            'mean': float(mean_r),
            'std': float(np.std(r_temp_data, ddof=1)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'df': float(df_val),
            'test': 'one-sample t-test on Fisher-z(r) (H1: z > 0)'
        }
        
        log(f"  n = {n_r_temp}")
        log(f"  Mean r: {mean_r:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
        log(f"  t({df_val}) = {t_stat:.3f}, p = {p_val:.4f} (Fisher z-transformed)")
    else:
        log(f"  ⚠ Insufficient data (n={n_r_temp})", "WARNING")
        results['metrics']['r_BR_temp'] = {'n': int(n_r_temp), 'error': 'insufficient_data'}
    
    # Metric: r_BR_VAS
    log("")
    log("3. Correlation r(BR, VAS):")
    
    r_vas_data = df['r_BR_VAS'].dropna().values
    n_r_vas = len(r_vas_data)
    
    if n_r_vas >= 3:
        mean_r, ci_lower, ci_upper = bias_corrected_bootstrap_ci(r_vas_data)
        
        # Fisher z-transform for test
        z_vas = fisher_z_transform(r_vas_data)
        t_stat, p_val, df_val = one_sample_t_test(z_vas, popmean=0, alternative='greater')
        
        results['metrics']['r_BR_VAS'] = {
            'n': int(n_r_vas),
            'mean': float(mean_r),
            'std': float(np.std(r_vas_data, ddof=1)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'df': float(df_val),
            'test': 'one-sample t-test on Fisher-z(r) (H1: z > 0)'
        }
        
        log(f"  n = {n_r_vas}")
        log(f"  Mean r: {mean_r:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
        log(f"  t({df_val}) = {t_stat:.3f}, p = {p_val:.4f} (Fisher z-transformed)")
    else:
        log(f"  ⚠ Insufficient data (n={n_r_vas})", "WARNING")
        results['metrics']['r_BR_VAS'] = {'n': int(n_r_vas), 'error': 'insufficient_data'}
    
    # Metric: auc_pain
    log("")
    log("4. ROC AUC (pain classification):")
    
    auc_data = df['auc_pain'].dropna().values
    n_auc = len(auc_data)
    
    if n_auc >= 3:
        # Subtract 0.5 for test (H1: AUC > 0.5)
        auc_centered = auc_data - 0.5
        
        mean_auc, ci_lower_centered, ci_upper_centered = bias_corrected_bootstrap_ci(auc_centered)
        
        # Shift CIs back to original scale
        mean_auc_orig = mean_auc + 0.5
        ci_lower = ci_lower_centered + 0.5
        ci_upper = ci_upper_centered + 0.5
        
        t_stat, p_val, df_val = one_sample_t_test(auc_centered, popmean=0, alternative='greater')
        
        results['metrics']['auc_pain'] = {
            'n': int(n_auc),
            'mean': float(mean_auc_orig),
            'std': float(np.std(auc_data, ddof=1)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'df': float(df_val),
            'test': 'one-sample t-test (H1: AUC - 0.5 > 0)'
        }
        
        log(f"  n = {n_auc}")
        log(f"  Mean: {mean_auc_orig:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
        log(f"  t({df_val}) = {t_stat:.3f}, p = {p_val:.4f} (vs 0.5)")
    else:
        log(f"  ⚠ Insufficient data (n={n_auc})", "WARNING")
        results['metrics']['auc_pain'] = {'n': int(n_auc), 'error': 'insufficient_data'}
    
    # Metric: forced_choice_acc
    log("")
    log("5. Forced-choice accuracy:")
    
    fc_data = df['forced_choice_acc'].dropna().values
    n_fc = len(fc_data)
    
    if n_fc >= 3:
        # Subtract 0.5 for test (H1: accuracy > 0.5)
        fc_centered = fc_data - 0.5
        
        mean_fc, ci_lower_centered, ci_upper_centered = bias_corrected_bootstrap_ci(fc_centered)
        
        # Shift CIs back to original scale
        mean_fc_orig = mean_fc + 0.5
        ci_lower = ci_lower_centered + 0.5
        ci_upper = ci_upper_centered + 0.5
        
        t_stat, p_val, df_val = one_sample_t_test(fc_centered, popmean=0, alternative='greater')
        
        results['metrics']['forced_choice_acc'] = {
            'n': int(n_fc),
            'mean': float(mean_fc_orig),
            'std': float(np.std(fc_data, ddof=1)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'df': float(df_val),
            'test': 'one-sample t-test (H1: accuracy > 0.5)'
        }
        
        log(f"  n = {n_fc}")
        log(f"  Mean: {mean_fc_orig:.3f} ({mean_fc_orig*100:.1f}%) [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
        log(f"  t({df_val}) = {t_stat:.3f}, p = {p_val:.4f} (vs 0.5)")
    else:
        log(f"  ⚠ Insufficient data (n={n_fc})", "WARNING")
        results['metrics']['forced_choice_acc'] = {'n': int(n_fc), 'error': 'insufficient_data'}
    
    # FDR correction on primary endpoints
    log("")
    log("=" * 70)
    log("MULTIPLE COMPARISON CORRECTION")
    log("=" * 70)
    
    primary_metrics = ['slope_BR_temp', 'r_BR_VAS']
    p_values = []
    metric_names = []
    
    for metric in primary_metrics:
        if metric in results['metrics'] and 'p_value' in results['metrics'][metric]:
            p_values.append(results['metrics'][metric]['p_value'])
            metric_names.append(metric)
    
    if len(p_values) > 0:
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        log("Primary endpoints (FDR correction):")
        for i, metric in enumerate(metric_names):
            results['metrics'][metric]['p_value_fdr'] = float(p_adjusted[i])
            results['metrics'][metric]['fdr_significant'] = bool(reject[i])
            
            log(f"  {metric}: p={p_values[i]:.4f}, p_FDR={p_adjusted[i]:.4f}, "
                f"significant={'Yes' if reject[i] else 'No'}")
    else:
        log("  ⚠ No valid p-values for FDR correction", "WARNING")
    
    return results


def create_summary_table(results: Dict) -> pd.DataFrame:
    """
    Create summary table of group statistics.
    
    Parameters
    ----------
    results : dict
        Group statistics results
    
    Returns
    -------
    pd.DataFrame
        Summary table
    """
    rows = []
    
    for metric_name, metric_data in results['metrics'].items():
        if 'error' in metric_data:
            continue
        
        row = {
            'metric': metric_name,
            'n': metric_data['n'],
            'mean': metric_data['mean'],
            'std': metric_data['std'],
            'ci_lower': metric_data['ci_lower'],
            'ci_upper': metric_data['ci_upper'],
            't_statistic': metric_data['t_statistic'],
            'df': metric_data['df'],
            'p_value': metric_data['p_value'],
            'p_value_fdr': metric_data.get('p_value_fdr', np.nan),
            'fdr_significant': metric_data.get('fdr_significant', False)
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    return df


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Group-level statistical inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run group analysis
  python 10_group_stats.py

Tests Performed:
  1. Slope BR ~ temperature > 0 (one-sample t-test)
  2. Fisher-z(r_BR_temp) > 0 (one-sample t-test)
  3. Fisher-z(r_BR_VAS) > 0 (one-sample t-test)
  4. AUC - 0.5 > 0 (one-sample t-test)
  5. Forced-choice accuracy > 0.5 (one-sample t-test)
  
FDR correction applied to primary endpoints: slope, r_BR_VAS
        """
    )
    
    parser.add_argument('--config', default='00_config.yaml',
                       help='Path to configuration file (default: 00_config.yaml)')
    parser.add_argument('--scores-dir', default='outputs/nps_scores',
                       help='Directory with NPS scores (default: outputs/nps_scores)')
    parser.add_argument('--output-dir', default='outputs/group',
                       help='Output directory (default: outputs/group)')
    
    args = parser.parse_args()
    
    # Load configuration
    log("=" * 70)
    log("GROUP-LEVEL STATISTICAL INFERENCE")
    log("=" * 70)
    
    try:
        from config_loader import load_config
        config = load_config(args.config)
        log(f"Loaded config: {args.config}")
        subjects = config['subjects']
    except Exception as e:
        log(f"Failed to load config: {e}", "ERROR")
        return 1
    
    log(f"Subjects in config: {len(subjects)}")
    log("")
    
    # Setup directories
    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not scores_dir.exists():
        log(f"Scores directory not found: {scores_dir}", "ERROR")
        log("Run 09_subject_metrics.py first", "ERROR")
        return 1
    
    # Load subject metrics
    log("Loading subject metrics...")
    
    try:
        df = load_subject_metrics(scores_dir, subjects)
        log(f"Loaded metrics for {len(df)} subjects")
    except Exception as e:
        log(f"Failed to load metrics: {e}", "ERROR")
        return 1
    
    # Check for exclusions
    missing_subjects = set(subjects) - set(df['subject'].values)
    if missing_subjects:
        log(f"⚠ Missing metrics for {len(missing_subjects)} subject(s): {missing_subjects}", "WARNING")
    
    # Compute group statistics
    try:
        results = compute_group_statistics(df)
    except Exception as e:
        log(f"Failed to compute statistics: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create summary table
    log("")
    log("=" * 70)
    log("SAVING RESULTS")
    log("=" * 70)
    
    summary_df = create_summary_table(results)
    
    # Save summary table
    summary_path = output_dir / "group_stats.tsv"
    summary_df.to_csv(summary_path, sep='\t', index=False, float_format='%.6f')
    log(f"Saved: {summary_path}")
    
    # Save diagnostics
    diagnostics = {
        'n_subjects_total': len(subjects),
        'n_subjects_analyzed': len(df),
        'missing_subjects': list(missing_subjects),
        'subjects_analyzed': df['subject'].tolist()
    }
    
    diagnostics_path = output_dir / "group_diagnostics.json"
    with open(diagnostics_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    log(f"Saved: {diagnostics_path}")
    
    # Save verbose results
    verbose_path = output_dir / "group_stats_verbose.json"
    with open(verbose_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"Saved: {verbose_path}")
    
    # Final summary
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"✓ Group analysis complete")
    log(f"  Subjects analyzed: {len(df)}/{len(subjects)}")
    log(f"  Results saved in: {output_dir}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
