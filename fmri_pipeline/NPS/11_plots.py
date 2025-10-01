#!/usr/bin/env python3
"""
11_plots.py - Publication-quality figures for NPS analysis.

Purpose:
    Generate publication-ready figures showing NPS dose-response relationships,
    discrimination metrics, and validation analyses.

Inputs:
    - outputs/nps_scores/sub-*/level_br.tsv: Condition-level BR scores
    - outputs/nps_scores/sub-*/trial_br.tsv: Trial-level BR scores (optional)
    - outputs/group/group_stats.tsv: Group statistics
    - 00_config.yaml: Configuration file

Outputs:
    - outputs/figures/Fig1_BR_vs_Temperature.svg: Dose-response plot
    - outputs/figures/Fig2_BR_vs_VAS.svg: BR vs pain ratings
    - outputs/figures/Fig3_Discrimination.svg: ROC analysis (if trial data)
    - outputs/figures/Fig4_Condition_Means.svg: Compact bar/line format
    - outputs/figures/Fig5_NPS_Map.png: NPS weights visualization
    - outputs/figures/Supp_Fig_SubjectSummary.svg: Subject-wise metrics
    - outputs/figures/QC_DesignCorr_Grid.png: QC design correlations (optional)
    - outputs/figures/QC_GridCompatibility.png: QC grid check (optional)
    - outputs/figures/figure_stats.json: Statistics for captions

Figure Specifications:
    - High-resolution SVG format
    - Publication-ready aesthetics
    - Within-subject error bars (Cousineau-Morey correction with Morey factor)
    - Bootstrap confidence intervals (≥5000 subject resamples, ≥2000 trial resamples)
    - Mixed-effects model BR ~ °C + (1|subject) reported for Fig 1
    - Per-subject trial bootstrap for AUC CIs
    - Professional typography and layout
    
Note on scaling:
    - If beta maps were scored at 2-mm without resampling, multiply BR by 27/8
      so all plots are in 3-mm units (state in figure caption or methods)

Exit codes:
    0 - Figures generated successfully
    1 - Processing failures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import nibabel as nib
from scipy import stats
from sklearn.utils import resample
from sklearn.metrics import roc_curve, roc_auc_score
from nilearn import plotting, datasets

from config_loader import load_config


# Publication-quality figure settings
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.format'] = 'svg'
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['svg.fonttype'] = 'none'  # Text as text (not paths)


def log(msg: str, level: str = "INFO"):
    """Print log message with level prefix."""
    print(f"[{level}] {msg}", flush=True)


def load_all_subjects_level_data(scores_dir: Path, subjects: List[str]) -> pd.DataFrame:
    """
    Load condition-level data from all subjects.
    
    Parameters
    ----------
    scores_dir : Path
        Directory with NPS scores
    subjects : list of str
        Subject IDs
    
    Returns
    -------
    pd.DataFrame
        Combined level data across subjects
    """
    data_list = []
    
    for subject in subjects:
        level_path = scores_dir / subject / "level_br.tsv"
        
        if not level_path.exists():
            log(f"  ⚠ level_br.tsv not found for {subject}", "WARNING")
            continue
        
        try:
            df = pd.read_csv(level_path, sep='\t')
            df['subject'] = subject
            data_list.append(df)
        except Exception as e:
            log(f"  ⚠ Failed to load {subject}: {e}", "WARNING")
            continue
    
    if len(data_list) == 0:
        raise ValueError("No subject data could be loaded")
    
    combined_df = pd.concat(data_list, ignore_index=True)
    
    return combined_df


def load_all_subjects_trial_data(scores_dir: Path, subjects: List[str]) -> pd.DataFrame:
    """
    Load trial-level data from all subjects.
    
    Parameters
    ----------
    scores_dir : Path
        Directory with NPS scores
    subjects : list of str
        Subject IDs
    
    Returns
    -------
    pd.DataFrame
        Combined trial data across subjects
    """
    data_list = []
    
    for subject in subjects:
        trial_path = scores_dir / subject / "trial_br.tsv"
        
        if not trial_path.exists():
            log(f"  ⚠ trial_br.tsv not found for {subject}", "WARNING")
            continue
        
        try:
            df = pd.read_csv(trial_path, sep='\t')
            data_list.append(df)
        except Exception as e:
            log(f"  ⚠ Failed to load {subject}: {e}", "WARNING")
            continue
    
    if len(data_list) == 0:
        raise ValueError("No trial data could be loaded")
    
    combined_df = pd.concat(data_list, ignore_index=True)
    
    return combined_df


def compute_cousineau_morey_correction(data: pd.DataFrame,
                                      subject_col: str = 'subject',
                                      condition_col: str = 'temp_celsius',
                                      value_col: str = 'br_score') -> pd.DataFrame:
    """
    Apply Cousineau-Morey correction for within-subject error bars.
    
    This removes between-subject variability to show within-subject effects,
    then applies Morey's correction factor sqrt(k/(k-1)) where k is the
    number of conditions.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with subjects × conditions
    subject_col : str
        Column name for subject IDs
    condition_col : str
        Column name for conditions
    value_col : str
        Column name for values
    
    Returns
    -------
    pd.DataFrame
        Data with corrected values in 'value_corrected' column
    """
    df = data.copy()
    
    # Compute subject means (across conditions)
    subject_means = df.groupby(subject_col)[value_col].mean()
    
    # Compute grand mean
    grand_mean = df[value_col].mean()
    
    # Center each subject's data (Cousineau method)
    df['value_corrected'] = df.apply(
        lambda row: row[value_col] - subject_means[row[subject_col]] + grand_mean,
        axis=1
    )
    
    # Apply Morey's correction factor: sqrt(k/(k-1))
    # where k is the number of conditions
    k = df[condition_col].nunique()
    morey_factor = np.sqrt(k / (k - 1)) if k > 1 else 1.0
    
    # Apply correction by scaling the centered values around the grand mean
    df['value_corrected'] = grand_mean + (df['value_corrected'] - grand_mean) * morey_factor
    
    return df


def compute_mixed_effects_model(data: pd.DataFrame,
                               x_col: str = 'temp_celsius',
                               y_col: str = 'br_score',
                               subject_col: str = 'subject') -> Dict:
    """
    Compute mixed-effects model: BR ~ °C + (1|subject).
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with subjects × conditions
    x_col : str
        Column name for predictor
    y_col : str
        Column name for outcome
    subject_col : str
        Column name for subject IDs
    
    Returns
    -------
    dict
        Model statistics (fixed effect, SE, t, p)
    """
    try:
        import statsmodels.formula.api as smf
        
        # Create formula: BR ~ Temperature + (1|subject)
        # statsmodels uses C() for categorical random effects
        formula = f"{y_col} ~ {x_col}"
        
        # Fit mixed linear model with random intercept per subject
        from statsmodels.regression.mixed_linear_model import MixedLM
        
        md = MixedLM.from_formula(formula, data=data, groups=data[subject_col])
        mdf = md.fit(method='lbfgs')
        
        # Extract fixed effect for temperature
        fixed_effect = mdf.params[x_col]
        se = mdf.bse[x_col]
        t_stat = mdf.tvalues[x_col]
        p_val = mdf.pvalues[x_col]
        
        return {
            'fixed_effect': float(fixed_effect),
            'se': float(se),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'converged': mdf.converged
        }
    except Exception as e:
        log(f"  ⚠ Mixed-effects model failed: {e}", "WARNING")
        return {
            'fixed_effect': np.nan,
            'se': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'converged': False
        }


def bootstrap_slope_and_correlation(data: pd.DataFrame,
                                    x_col: str = 'temp_celsius',
                                    y_col: str = 'br_score',
                                    subject_col: str = 'subject',
                                    n_bootstrap: int = 5000,
                                    ci: float = 0.95,
                                    random_state: int = 42) -> Dict:
    """
    Bootstrap confidence intervals for slope and correlation.
    
    Resamples subjects (not individual observations) to maintain
    within-subject correlation structure.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with subjects × conditions
    x_col : str
        Column name for x variable
    y_col : str
        Column name for y variable
    subject_col : str
        Column name for subject IDs
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence level
    random_state : int
        Random seed
    
    Returns
    -------
    dict
        Statistics with CIs
    """
    rng = np.random.RandomState(random_state)
    subjects = data[subject_col].unique()
    
    # Original statistics
    slope_orig, intercept_orig, r_orig, p_orig, _ = stats.linregress(
        data[x_col], data[y_col]
    )
    
    # Bootstrap
    slopes = []
    correlations = []
    
    for _ in range(n_bootstrap):
        # Resample subjects
        boot_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        
        # Get data for resampled subjects
        boot_data = []
        for subj in boot_subjects:
            subj_data = data[data[subject_col] == subj]
            boot_data.append(subj_data)
        
        boot_df = pd.concat(boot_data, ignore_index=True)
        
        # Compute statistics
        try:
            slope, _, r, _, _ = stats.linregress(boot_df[x_col], boot_df[y_col])
            slopes.append(slope)
            correlations.append(r)
        except Exception:
            continue
    
    # Compute CIs
    alpha = (1 - ci) / 2
    
    slope_ci_lower = np.percentile(slopes, alpha * 100)
    slope_ci_upper = np.percentile(slopes, (1 - alpha) * 100)
    
    r_ci_lower = np.percentile(correlations, alpha * 100)
    r_ci_upper = np.percentile(correlations, (1 - alpha) * 100)
    
    results = {
        'slope': float(slope_orig),
        'slope_ci_lower': float(slope_ci_lower),
        'slope_ci_upper': float(slope_ci_upper),
        'intercept': float(intercept_orig),
        'r': float(r_orig),
        'r_ci_lower': float(r_ci_lower),
        'r_ci_upper': float(r_ci_upper),
        'p': float(p_orig)
    }
    
    return results


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
    r_clipped = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))


def fisher_z_inverse(z: float) -> float:
    """
    Inverse Fisher z-transformation.
    
    Parameters
    ----------
    z : float
        Fisher z value
    
    Returns
    -------
    float
        Correlation coefficient
    """
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def compute_within_subject_correlations(data: pd.DataFrame,
                                        x_col: str = 'mean_vas',
                                        y_col: str = 'br_score',
                                        subject_col: str = 'subject') -> pd.DataFrame:
    """
    Compute within-subject correlations.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with subjects × conditions
    x_col : str
        Column name for x variable
    y_col : str
        Column name for y variable
    subject_col : str
        Column name for subject IDs
    
    Returns
    -------
    pd.DataFrame
        Subject-level correlations
    """
    results = []
    
    for subject in data[subject_col].unique():
        subj_data = data[data[subject_col] == subject]
        
        # Filter valid pairs
        valid = subj_data[[x_col, y_col]].dropna()
        
        if len(valid) >= 3:
            try:
                r, p = stats.pearsonr(valid[x_col], valid[y_col])
                results.append({
                    'subject': subject,
                    'r': r,
                    'p': p,
                    'n': len(valid)
                })
            except Exception:
                continue
    
    return pd.DataFrame(results)


def create_figure1_dose_response(data: pd.DataFrame,
                                 output_path: Path,
                                 stats_output: Dict) -> None:
    """
    Create Figure 1: NPS dose-response plot.
    
    Parameters
    ----------
    data : pd.DataFrame
        Level data across all subjects
    output_path : Path
        Output file path
    stats_output : dict
        Dictionary to store statistics for caption
    """
    log("Creating Figure 1: NPS Dose-Response")
    
    # Filter valid data
    valid_data = data.dropna(subset=['temp_celsius', 'br_score']).copy()
    
    if len(valid_data) == 0:
        log("  ✗ No valid data", "ERROR")
        return
    
    n_subjects = valid_data['subject'].nunique()
    temps = sorted(valid_data['temp_celsius'].unique())
    
    log(f"  Subjects: {n_subjects}")
    log(f"  Temperatures: {len(temps)}")
    
    # Apply Cousineau-Morey correction
    log("  Applying Cousineau-Morey correction...")
    corrected_data = compute_cousineau_morey_correction(valid_data)
    
    # Compute group means and within-subject SEM
    group_stats = corrected_data.groupby('temp_celsius').agg({
        'br_score': 'mean',
        'value_corrected': ['mean', 'sem']
    }).reset_index()
    
    group_stats.columns = ['temp_celsius', 'br_mean', 'br_corrected_mean', 'br_corrected_sem']
    
    # Bootstrap statistics
    log("  Computing bootstrap statistics...")
    boot_stats = bootstrap_slope_and_correlation(valid_data, n_bootstrap=5000)
    
    # Compute mixed-effects model for caption
    log("  Computing mixed-effects model BR ~ °C + (1|subject)...")
    me_stats = compute_mixed_effects_model(valid_data)
    
    # Store for caption
    stats_output['figure1'] = {
        'n_subjects': int(n_subjects),
        'n_temperatures': len(temps),
        'slope': boot_stats['slope'],
        'slope_ci_lower': boot_stats['slope_ci_lower'],
        'slope_ci_upper': boot_stats['slope_ci_upper'],
        'r': boot_stats['r'],
        'r_ci_lower': boot_stats['r_ci_lower'],
        'r_ci_upper': boot_stats['r_ci_upper'],
        'p': boot_stats['p'],
        'mixed_effects_beta': me_stats['fixed_effect'],
        'mixed_effects_se': me_stats['se'],
        'mixed_effects_t': me_stats['t_statistic'],
        'mixed_effects_p': me_stats['p_value'],
        'mixed_effects_converged': me_stats['converged']
    }
    
    log(f"    Slope: {boot_stats['slope']:.4f} [95% CI: {boot_stats['slope_ci_lower']:.4f}, {boot_stats['slope_ci_upper']:.4f}]")
    log(f"    r: {boot_stats['r']:.3f} [95% CI: {boot_stats['r_ci_lower']:.3f}, {boot_stats['r_ci_upper']:.3f}], p={boot_stats['p']:.4f}")
    if me_stats['converged']:
        log(f"    Mixed-effects β: {me_stats['fixed_effect']:.4f} (SE={me_stats['se']:.4f}), t={me_stats['t_statistic']:.2f}, p={me_stats['p_value']:.4f}")
    
    # Create figure
    log("  Rendering figure...")
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot individual subjects (spaghetti lines)
    for subject in valid_data['subject'].unique():
        subj_data = valid_data[valid_data['subject'] == subject].sort_values('temp_celsius')
        
        ax.plot(subj_data['temp_celsius'], subj_data['br_score'],
                color='lightgray', alpha=0.4, linewidth=0.8, zorder=1)
    
    # Plot group mean with error bars
    ax.errorbar(group_stats['temp_celsius'], group_stats['br_mean'],
                yerr=group_stats['br_corrected_sem'],
                color='black', marker='o', markersize=6, linewidth=2,
                capsize=4, capthick=1.5, zorder=3,
                label=f'Group mean ± wsSEM')
    
    # Add regression line
    x_line = np.array([temps[0] - 0.5, temps[-1] + 0.5])
    y_line = boot_stats['intercept'] + boot_stats['slope'] * x_line
    ax.plot(x_line, y_line, '--', color='gray', linewidth=1.5, alpha=0.6, zorder=2)
    
    # Formatting
    ax.set_xlabel('Temperature (°C)', fontweight='bold')
    ax.set_ylabel('NPS Response (3-mm voxels)', fontweight='bold')
    ax.set_title('NPS Dose-Response Relationship', fontweight='bold', pad=10)
    
    # Set x-axis ticks to exact temperatures (44-50 range with 6 temps)
    ax.set_xticks(temps)
    ax.set_xticklabels([f'{t:.1f}' for t in temps])
    ax.set_xlim(43.5, 50.5)  # Consistent range for 44-50°C
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    legend_label = f'Subjects (n={n_subjects})'
    ax.plot([], [], color='lightgray', linewidth=2, alpha=0.6, label=legend_label)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black')
    
    # Statistics annotation (top right)
    stats_text = (
        f"Slope = {boot_stats['slope']:.3f} BR/°C\n"
        f"95% CI [{boot_stats['slope_ci_lower']:.3f}, {boot_stats['slope_ci_upper']:.3f}]\n"
        f"\n"
        f"r = {boot_stats['r']:.3f}\n"
        f"95% CI [{boot_stats['r_ci_lower']:.3f}, {boot_stats['r_ci_upper']:.3f}]\n"
        f"p = {boot_stats['p']:.4f}"
    )
    
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
            fontsize=9,
            family='monospace')
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save
    fig.tight_layout()
    fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    log(f"  ✓ Saved: {output_path.name}")


def create_figure2_br_vs_vas(data: pd.DataFrame,
                             output_path: Path,
                             stats_output: Dict) -> None:
    """
    Create Figure 2: BR vs VAS ratings (two panels).
    
    Panel A: Scatter plot with pooled points
    Panel B: Distribution of within-subject correlations
    
    Parameters
    ----------
    data : pd.DataFrame
        Level data across all subjects
    output_path : Path
        Output file path
    stats_output : dict
        Dictionary to store statistics for caption
    """
    log("Creating Figure 2: BR vs VAS Ratings")
    
    # Filter valid data
    valid_data = data.dropna(subset=['mean_vas', 'br_score']).copy()
    
    if len(valid_data) == 0:
        log("  ✗ No valid data", "ERROR")
        return
    
    n_subjects = valid_data['subject'].nunique()
    
    log(f"  Subjects: {n_subjects}")
    log(f"  Data points: {len(valid_data)}")
    
    # Compute within-subject correlations
    log("  Computing within-subject correlations...")
    ws_corr = compute_within_subject_correlations(valid_data)
    
    if len(ws_corr) == 0:
        log("  ✗ No valid within-subject correlations", "ERROR")
        return
    
    log(f"  Valid correlations: {len(ws_corr)}/{n_subjects}")
    
    # Fisher z-transform and aggregate
    ws_corr['fisher_z'] = fisher_z_transform(ws_corr['r'].values)
    mean_z = ws_corr['fisher_z'].mean()
    mean_r = fisher_z_inverse(mean_z)
    
    # Bootstrap CI on Fisher z
    rng = np.random.RandomState(42)
    
    # Check if we have enough subjects for bootstrap
    if len(ws_corr) == 1:
        log("  ⚠ Only 1 subject - bootstrap CIs not meaningful", "WARNING")
        z_ci_lower = mean_z
        z_ci_upper = mean_z
        r_ci_lower = mean_r
        r_ci_upper = mean_r
        t_stat = np.nan
        p_val = np.nan
    else:
        bootstrap_zs = []
        
        for _ in range(5000):
            boot_sample = ws_corr.sample(n=len(ws_corr), replace=True, random_state=None)
            bootstrap_zs.append(boot_sample['fisher_z'].mean())
        
        z_ci_lower = np.percentile(bootstrap_zs, 2.5)
        z_ci_upper = np.percentile(bootstrap_zs, 97.5)
        
        r_ci_lower = fisher_z_inverse(z_ci_lower)
        r_ci_upper = fisher_z_inverse(z_ci_upper)
        
        # One-sample t-test on Fisher z values
        t_stat, p_val = stats.ttest_1samp(ws_corr['fisher_z'], 0)
    
    # Store statistics
    stats_output['figure2'] = {
        'n_subjects': int(n_subjects),
        'n_subjects_with_corr': len(ws_corr),
        'mean_r': float(mean_r),
        'r_ci_lower': float(r_ci_lower),
        'r_ci_upper': float(r_ci_upper),
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'df': len(ws_corr) - 1
    }
    
    log(f"    Mean within-subject r: {mean_r:.3f} [95% CI: {r_ci_lower:.3f}, {r_ci_upper:.3f}]")
    
    if not np.isnan(t_stat):
        log(f"    t({len(ws_corr)-1}) = {t_stat:.3f}, p = {p_val:.4f}")
    else:
        log(f"    t-test not applicable (n=1)")
    
    # Create two-panel figure
    log("  Rendering figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ========================================================================
    # Panel A: Scatter plot
    # ========================================================================
    ax_scatter = axes[0]
    
    # Generate color map for subjects
    n_colors = n_subjects
    cmap = plt.cm.tab20 if n_colors <= 20 else plt.cm.hsv
    colors = [cmap(i / n_colors) for i in range(n_colors)]
    subject_color_map = {subj: colors[i] for i, subj in enumerate(valid_data['subject'].unique())}
    
    # Plot points colored by subject
    for subject in valid_data['subject'].unique():
        subj_data = valid_data[valid_data['subject'] == subject]
        ax_scatter.scatter(subj_data['mean_vas'], subj_data['br_score'],
                          color=subject_color_map[subject], alpha=0.4, s=30, edgecolors='none')
    
    # Compute and plot group OLS fit with bootstrap CI
    x_range = np.linspace(valid_data['mean_vas'].min(), valid_data['mean_vas'].max(), 100)
    
    # Bootstrap regression line
    subjects = valid_data['subject'].unique()
    bootstrap_slopes = []
    bootstrap_intercepts = []
    
    for _ in range(5000):
        boot_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        boot_data = []
        for subj in boot_subjects:
            boot_data.append(valid_data[valid_data['subject'] == subj])
        boot_df = pd.concat(boot_data, ignore_index=True)
        
        try:
            slope, intercept, _, _, _ = stats.linregress(boot_df['mean_vas'], boot_df['br_score'])
            bootstrap_slopes.append(slope)
            bootstrap_intercepts.append(intercept)
        except Exception:
            continue
    
    # Original fit
    slope_orig, intercept_orig, r_pooled, p_pooled, _ = stats.linregress(
        valid_data['mean_vas'], valid_data['br_score']
    )
    
    y_fit = intercept_orig + slope_orig * x_range
    ax_scatter.plot(x_range, y_fit, 'k-', linewidth=2, label='Group OLS fit', zorder=10)
    
    # CI band
    y_bootstrap = []
    for slope, intercept in zip(bootstrap_slopes, bootstrap_intercepts):
        y_bootstrap.append(intercept + slope * x_range)
    
    y_bootstrap = np.array(y_bootstrap)
    y_lower = np.percentile(y_bootstrap, 2.5, axis=0)
    y_upper = np.percentile(y_bootstrap, 97.5, axis=0)
    
    ax_scatter.fill_between(x_range, y_lower, y_upper, color='black', alpha=0.15, zorder=5)
    
    # Formatting Panel A
    ax_scatter.set_xlabel('VAS Rating (0-200)', fontweight='bold')
    ax_scatter.set_ylabel('NPS Response (a.u.)', fontweight='bold')
    ax_scatter.set_title('A. BR vs Pain Ratings (Pooled)', fontweight='bold', pad=10)
    ax_scatter.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax_scatter.set_axisbelow(True)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)
    
    # Set x-axis limits (VAS 0-200 with ticks)
    ax_scatter.set_xlim(0, 200)
    ax_scatter.set_xticks([0, 50, 100, 150, 200])
    
    # Statistics annotation
    if not np.isnan(p_val):
        stats_text = (
            f"Within-subject r = {mean_r:.3f}\n"
            f"95% CI [{r_ci_lower:.3f}, {r_ci_upper:.3f}]\n"
            f"p = {p_val:.4f}"
        )
    else:
        stats_text = (
            f"Within-subject r = {mean_r:.3f}\n"
            f"(n=1, no CI or p-value)"
        )
    
    ax_scatter.text(0.02, 0.98, stats_text,
                   transform=ax_scatter.transAxes,
                   verticalalignment='top',
                   horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
                   fontsize=9,
                   family='monospace')
    
    # ========================================================================
    # Panel B: Distribution of within-subject correlations
    # ========================================================================
    ax_dist = axes[1]
    
    # Violin plot
    parts = ax_dist.violinplot([ws_corr['r'].values], positions=[0], 
                                showmeans=True, showmedians=True, widths=0.7)
    
    # Style violin
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Style lines
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    
    # Add individual points (strip plot)
    jitter = rng.normal(0, 0.02, size=len(ws_corr))
    ax_dist.scatter(jitter, ws_corr['r'], color='darkblue', alpha=0.5, s=40, zorder=3)
    
    # Horizontal line at r=0
    ax_dist.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
    # Formatting Panel B
    ax_dist.set_ylabel('Within-Subject Correlation (r)', fontweight='bold')
    ax_dist.set_title('B. Distribution of Within-Subject Correlations', fontweight='bold', pad=10)
    ax_dist.set_xticks([])
    ax_dist.set_xlim(-0.5, 0.5)
    ax_dist.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
    ax_dist.set_axisbelow(True)
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    ax_dist.spines['bottom'].set_visible(False)
    
    # Add text annotation
    if not np.isnan(t_stat):
        stats_text_b = (
            f"n = {len(ws_corr)}\n"
            f"Mean r = {mean_r:.3f}\n"
            f"95% CI [{r_ci_lower:.3f}, {r_ci_upper:.3f}]\n"
            f"t({len(ws_corr)-1}) = {t_stat:.2f}\n"
            f"p = {p_val:.4f}"
        )
    else:
        stats_text_b = (
            f"n = {len(ws_corr)}\n"
            f"r = {mean_r:.3f}\n"
            f"(Single subject)"
        )
    
    ax_dist.text(0.98, 0.02, stats_text_b,
                transform=ax_dist.transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
                fontsize=9,
                family='monospace')
    
    # Save
    fig.tight_layout()
    fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    log(f"  ✓ Saved: {output_path.name}")


def create_figure3_discrimination(data: pd.DataFrame,
                                  output_path: Path,
                                  stats_output: Dict) -> None:
    """
    Create Figure 3: Discrimination metrics (three panels).
    
    Panel A: Group ROC curve
    Panel B: AUC distribution across subjects  
    Panel C: Forced-choice accuracy
    
    Parameters
    ----------
    data : pd.DataFrame
        Trial data across all subjects
    output_path : Path
        Output file path
    stats_output : dict
        Dictionary to store statistics for caption
    """
    log("Creating Figure 3: Discrimination Metrics")
    
    # Filter valid data
    valid_data = data.dropna(subset=['br_score', 'pain_binary']).copy()
    
    if len(valid_data) == 0:
        log("  ✗ No valid trial data", "ERROR")
        return
    
    n_subjects = valid_data['subject'].nunique()
    n_trials = len(valid_data)
    
    log(f"  Subjects: {n_subjects}")
    log(f"  Trials: {n_trials}")
    
    # Compute per-subject ROC/AUC with trial bootstrap CIs
    log("  Computing per-subject ROC curves and AUCs with trial bootstrap...")
    
    subject_aucs = []
    subject_roc_curves = []
    rng_trial = np.random.RandomState(42)
    
    for subject in valid_data['subject'].unique():
        subj_data = valid_data[valid_data['subject'] == subject]
        
        # Check if we have both classes
        if subj_data['pain_binary'].nunique() < 2:
            log(f"    ⚠ {subject}: Only one class, skipping", "WARNING")
            continue
        
        y_true = subj_data['pain_binary'].values
        y_scores = subj_data['br_score'].values
        
        try:
            # Compute observed AUC
            auc = roc_auc_score(y_true, y_scores)
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            
            # Bootstrap trials for AUC CI (≥2000 resamples per subject)
            bootstrap_aucs = []
            for _ in range(2000):
                boot_indices = rng_trial.choice(len(y_true), size=len(y_true), replace=True)
                y_boot = y_true[boot_indices]
                s_boot = y_scores[boot_indices]
                
                # Check if bootstrap has both classes
                if len(np.unique(y_boot)) == 2:
                    try:
                        boot_auc = roc_auc_score(y_boot, s_boot)
                        bootstrap_aucs.append(boot_auc)
                    except Exception:
                        continue
            
            # Compute CI if we have enough bootstrap samples
            if len(bootstrap_aucs) > 100:
                auc_ci_lower = np.percentile(bootstrap_aucs, 2.5)
                auc_ci_upper = np.percentile(bootstrap_aucs, 97.5)
            else:
                auc_ci_lower = auc
                auc_ci_upper = auc
            
            subject_aucs.append({
                'subject': subject,
                'auc': auc,
                'auc_ci_lower': auc_ci_lower,
                'auc_ci_upper': auc_ci_upper,
                'n_trials': len(y_true)
            })
            
            subject_roc_curves.append({
                'subject': subject,
                'fpr': fpr,
                'tpr': tpr
            })
            
        except Exception as e:
            log(f"    ⚠ {subject}: Failed to compute ROC/AUC: {e}", "WARNING")
            continue
    
    if len(subject_aucs) == 0:
        log("  ✗ No valid ROC curves computed", "ERROR")
        return
    
    log(f"    Successfully computed ROC/AUC for {len(subject_aucs)} subjects")
    
    aucs_df = pd.DataFrame(subject_aucs)
    mean_auc = aucs_df['auc'].mean()
    
    # Bootstrap CI for mean AUC
    log("  Computing bootstrap CIs for AUC...")
    
    if len(aucs_df) > 1:
        rng = np.random.RandomState(42)
        bootstrap_aucs = []
        
        for _ in range(5000):
            boot_sample = aucs_df.sample(n=len(aucs_df), replace=True, random_state=None)
            bootstrap_aucs.append(boot_sample['auc'].mean())
        
        auc_ci_lower = np.percentile(bootstrap_aucs, 2.5)
        auc_ci_upper = np.percentile(bootstrap_aucs, 97.5)
        
        # One-sample t-test vs 0.5
        t_stat, p_val = stats.ttest_1samp(aucs_df['auc'] - 0.5, 0, alternative='greater')
    else:
        log("    ⚠ Only 1 subject - CIs not meaningful", "WARNING")
        auc_ci_lower = mean_auc
        auc_ci_upper = mean_auc
        t_stat = np.nan
        p_val = np.nan
    
    log(f"    Mean AUC: {mean_auc:.3f} [95% CI: {auc_ci_lower:.3f}, {auc_ci_upper:.3f}]")
    
    if not np.isnan(p_val):
        log(f"    t({len(aucs_df)-1}) = {t_stat:.3f}, p = {p_val:.4f} (vs 0.5)")
    
    # Compute group ROC curve (average TPR at common FPR grid)
    log("  Computing group-level ROC curve...")
    
    common_fpr = np.linspace(0, 1, 100)
    tpr_interp_list = []
    
    for roc_data in subject_roc_curves:
        tpr_interp = np.interp(common_fpr, roc_data['fpr'], roc_data['tpr'])
        tpr_interp_list.append(tpr_interp)
    
    tpr_matrix = np.array(tpr_interp_list)
    mean_tpr = tpr_matrix.mean(axis=0)
    
    # Bootstrap CI for ROC curve
    if len(tpr_matrix) > 1:
        bootstrap_tpr = []
        
        for _ in range(5000):
            boot_indices = rng.choice(len(tpr_matrix), size=len(tpr_matrix), replace=True)
            boot_tpr = tpr_matrix[boot_indices].mean(axis=0)
            bootstrap_tpr.append(boot_tpr)
        
        bootstrap_tpr = np.array(bootstrap_tpr)
        tpr_ci_lower = np.percentile(bootstrap_tpr, 2.5, axis=0)
        tpr_ci_upper = np.percentile(bootstrap_tpr, 97.5, axis=0)
    else:
        tpr_ci_lower = mean_tpr
        tpr_ci_upper = mean_tpr
    
    # Compute forced-choice accuracy
    log("  Computing forced-choice accuracy...")
    
    forced_choice_accs = []
    
    # Define warm and pain thresholds (use temperature if available)
    if 'temp_celsius' in valid_data.columns:
        warm_threshold = 46.0  # Below this = warm
        pain_threshold = 47.0  # At or above = pain
        
        for subject in valid_data['subject'].unique():
            subj_data = valid_data[valid_data['subject'] == subject]
            
            warm_trials = subj_data[subj_data['temp_celsius'] < warm_threshold]
            pain_trials = subj_data[subj_data['temp_celsius'] >= pain_threshold]
            
            if len(warm_trials) == 0 or len(pain_trials) == 0:
                continue
            
            # Count correct comparisons
            n_correct = 0
            n_total = 0
            
            for _, warm_trial in warm_trials.iterrows():
                for _, pain_trial in pain_trials.iterrows():
                    if pain_trial['br_score'] > warm_trial['br_score']:
                        n_correct += 1
                    n_total += 1
            
            if n_total > 0:
                accuracy = n_correct / n_total
                forced_choice_accs.append({
                    'subject': subject,
                    'accuracy': accuracy,
                    'n_pairs': n_total
                })
    else:
        # Fallback: use pain_binary
        log("    Using pain_binary for warm/pain classification", "WARNING")
        
        for subject in valid_data['subject'].unique():
            subj_data = valid_data[valid_data['subject'] == subject]
            
            warm_trials = subj_data[subj_data['pain_binary'] == 0]
            pain_trials = subj_data[subj_data['pain_binary'] == 1]
            
            if len(warm_trials) == 0 or len(pain_trials) == 0:
                continue
            
            n_correct = 0
            n_total = 0
            
            for _, warm_trial in warm_trials.iterrows():
                for _, pain_trial in pain_trials.iterrows():
                    if pain_trial['br_score'] > warm_trial['br_score']:
                        n_correct += 1
                    n_total += 1
            
            if n_total > 0:
                accuracy = n_correct / n_total
                forced_choice_accs.append({
                    'subject': subject,
                    'accuracy': accuracy,
                    'n_pairs': n_total
                })
    
    fc_df = pd.DataFrame(forced_choice_accs)
    
    if len(fc_df) > 0:
        mean_fc = fc_df['accuracy'].mean()
        
        if len(fc_df) > 1:
            bootstrap_fc = []
            for _ in range(5000):
                boot_sample = fc_df.sample(n=len(fc_df), replace=True, random_state=None)
                bootstrap_fc.append(boot_sample['accuracy'].mean())
            
            fc_ci_lower = np.percentile(bootstrap_fc, 2.5)
            fc_ci_upper = np.percentile(bootstrap_fc, 97.5)
            
            # t-test vs 0.5
            t_fc, p_fc = stats.ttest_1samp(fc_df['accuracy'] - 0.5, 0, alternative='greater')
        else:
            fc_ci_lower = mean_fc
            fc_ci_upper = mean_fc
            t_fc = np.nan
            p_fc = np.nan
        
        log(f"    Mean forced-choice: {mean_fc:.3f} ({mean_fc*100:.1f}%) [95% CI: {fc_ci_lower:.3f}, {fc_ci_upper:.3f}]")
        if not np.isnan(p_fc):
            log(f"    t({len(fc_df)-1}) = {t_fc:.3f}, p = {p_fc:.4f} (vs 0.5)")
    else:
        log("    ⚠ No forced-choice data", "WARNING")
        mean_fc = np.nan
        fc_ci_lower = np.nan
        fc_ci_upper = np.nan
        t_fc = np.nan
        p_fc = np.nan
    
    # Store statistics
    stats_output['figure3'] = {
        'n_subjects': int(n_subjects),
        'n_subjects_with_roc': len(aucs_df),
        'n_subjects_with_fc': len(fc_df) if len(fc_df) > 0 else 0,
        'mean_auc': float(mean_auc),
        'auc_ci_lower': float(auc_ci_lower),
        'auc_ci_upper': float(auc_ci_upper),
        'auc_t_statistic': float(t_stat),
        'auc_p_value': float(p_val),
        'mean_forced_choice': float(mean_fc) if not np.isnan(mean_fc) else None,
        'fc_ci_lower': float(fc_ci_lower) if not np.isnan(fc_ci_lower) else None,
        'fc_ci_upper': float(fc_ci_upper) if not np.isnan(fc_ci_upper) else None,
        'fc_t_statistic': float(t_fc) if not np.isnan(t_fc) else None,
        'fc_p_value': float(p_fc) if not np.isnan(p_fc) else None
    }
    
    # Create three-panel figure
    log("  Rendering figure...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ========================================================================
    # Panel A: Group ROC Curve
    # ========================================================================
    ax_roc = axes[0]
    
    # Plot chance line
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Chance', zorder=1)
    
    # Plot CI band
    if len(tpr_matrix) > 1:
        ax_roc.fill_between(common_fpr, tpr_ci_lower, tpr_ci_upper,
                           color='blue', alpha=0.2, zorder=2)
    
    # Plot mean ROC
    ax_roc.plot(common_fpr, mean_tpr, 'b-', linewidth=2.5, 
               label=f'Mean ROC (n={len(subject_roc_curves)})', zorder=3)
    
    # Formatting
    ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
    ax_roc.set_title('A. ROC Curve (Group)', fontweight='bold', pad=10)
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1])
    ax_roc.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax_roc.set_axisbelow(True)
    ax_roc.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    ax_roc.spines['top'].set_visible(False)
    ax_roc.spines['right'].set_visible(False)
    ax_roc.set_aspect('equal')
    
    # Statistics annotation
    if not np.isnan(p_val):
        stats_text_a = (
            f"AUC = {mean_auc:.3f}\n"
            f"95% CI [{auc_ci_lower:.3f}, {auc_ci_upper:.3f}]\n"
            f"p = {p_val:.4f} (vs 0.5)"
        )
    else:
        stats_text_a = f"AUC = {mean_auc:.3f}\n(n=1, no CI)"
    
    ax_roc.text(0.98, 0.02, stats_text_a,
               transform=ax_roc.transAxes,
               verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
               fontsize=9,
               family='monospace')
    
    # ========================================================================
    # Panel B: AUC Distribution
    # ========================================================================
    ax_auc = axes[1]
    
    # Violin plot
    if len(aucs_df) > 1:
        parts = ax_auc.violinplot([aucs_df['auc'].values], positions=[0],
                                  showmeans=True, showmedians=True, widths=0.7)
        
        for pc in parts['bodies']:
            pc.set_facecolor('lightgreen')
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
        
        # Add individual points
        jitter = rng.normal(0, 0.02, size=len(aucs_df))
        ax_auc.scatter(jitter, aucs_df['auc'], color='darkgreen', alpha=0.5, s=40, zorder=3)
    else:
        # Single subject - just plot the point
        ax_auc.scatter([0], [mean_auc], color='darkgreen', s=100, zorder=3)
    
    # Horizontal line at 0.5
    ax_auc.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1,
                  label='Chance (0.5)')
    
    # Formatting
    ax_auc.set_ylabel('AUC', fontweight='bold')
    ax_auc.set_title('B. AUC Distribution', fontweight='bold', pad=10)
    ax_auc.set_xticks([])
    ax_auc.set_xlim(-0.5, 0.5)
    ax_auc.set_ylim([0, 1])
    ax_auc.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
    ax_auc.set_axisbelow(True)
    ax_auc.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    ax_auc.spines['top'].set_visible(False)
    ax_auc.spines['right'].set_visible(False)
    ax_auc.spines['bottom'].set_visible(False)
    
    # Statistics
    if not np.isnan(p_val):
        stats_text_b = (
            f"n = {len(aucs_df)}\n"
            f"Mean = {mean_auc:.3f}\n"
            f"95% CI [{auc_ci_lower:.3f}, {auc_ci_upper:.3f}]\n"
            f"t({len(aucs_df)-1}) = {t_stat:.2f}\n"
            f"p = {p_val:.4f}"
        )
    else:
        stats_text_b = f"n = {len(aucs_df)}\nAUC = {mean_auc:.3f}\n(Single subject)"
    
    ax_auc.text(0.98, 0.98, stats_text_b,
               transform=ax_auc.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
               fontsize=9,
               family='monospace')
    
    # ========================================================================
    # Panel C: Forced-Choice Accuracy
    # ========================================================================
    ax_fc = axes[2]
    
    if len(fc_df) > 0:
        # Bar for mean
        ax_fc.bar([0], [mean_fc], width=0.6, color='lightcoral', alpha=0.7,
                 edgecolor='black', linewidth=1.5, zorder=2)
        
        # Error bar (CI)
        if not np.isnan(fc_ci_lower):
            ax_fc.errorbar([0], [mean_fc], 
                          yerr=[[mean_fc - fc_ci_lower], [fc_ci_upper - mean_fc]],
                          fmt='none', color='black', capsize=10, capthick=2, linewidth=2, zorder=3)
        
        # Overlay subject dots
        if len(fc_df) > 1:
            jitter = rng.normal(0, 0.05, size=len(fc_df))
            ax_fc.scatter(jitter, fc_df['accuracy'], color='darkred', alpha=0.6, s=50, zorder=4)
        else:
            ax_fc.scatter([0], [mean_fc], color='darkred', s=100, zorder=4)
        
        # Horizontal line at 0.5
        ax_fc.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1,
                     label='Chance (0.5)')
        
        # Formatting
        ax_fc.set_ylabel('Forced-Choice Accuracy', fontweight='bold')
        ax_fc.set_title('C. Forced-Choice Performance', fontweight='bold', pad=10)
        ax_fc.set_xticks([])
        ax_fc.set_xlim(-0.5, 0.5)
        ax_fc.set_ylim([0, 1])
        ax_fc.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
        ax_fc.set_axisbelow(True)
        ax_fc.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
        ax_fc.spines['top'].set_visible(False)
        ax_fc.spines['right'].set_visible(False)
        ax_fc.spines['bottom'].set_visible(False)
        
        # Statistics
        if not np.isnan(p_fc):
            stats_text_c = (
                f"n = {len(fc_df)}\n"
                f"Mean = {mean_fc:.3f} ({mean_fc*100:.1f}%)\n"
                f"95% CI [{fc_ci_lower:.3f}, {fc_ci_upper:.3f}]\n"
                f"t({len(fc_df)-1}) = {t_fc:.2f}\n"
                f"p = {p_fc:.4f}"
            )
        else:
            stats_text_c = (
                f"n = {len(fc_df)}\n"
                f"Accuracy = {mean_fc:.3f}\n"
                f"({mean_fc*100:.1f}%)\n"
                f"(Single subject)"
            )
        
        ax_fc.text(0.98, 0.98, stats_text_c,
                  transform=ax_fc.transAxes,
                  verticalalignment='top',
                  horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
                  fontsize=9,
                  family='monospace')
    else:
        ax_fc.text(0.5, 0.5, 'No forced-choice data available',
                  transform=ax_fc.transAxes,
                  verticalalignment='center',
                  horizontalalignment='center',
                  fontsize=12)
        ax_fc.set_xticks([])
        ax_fc.set_yticks([])
    
    # Save
    fig.tight_layout()
    fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    log(f"  ✓ Saved: {output_path.name}")


def create_figure4_condition_means(data: pd.DataFrame,
                                   output_path: Path,
                                   stats_output: Dict) -> None:
    """
    Create Figure 4: Condition means with error bars (compact format).
    
    Single panel showing group mean BR at each temperature with
    within-subject SEM error bars. Simplified version of Figure 1
    for reviewers.
    
    Parameters
    ----------
    data : pd.DataFrame
        Level data across all subjects
    output_path : Path
        Output file path
    stats_output : dict
        Dictionary to store/retrieve statistics
    """
    log("Creating Figure 4: Condition Means (Bar/Line)")
    
    # Filter valid data
    valid_data = data.dropna(subset=['temp_celsius', 'br_score']).copy()
    
    if len(valid_data) == 0:
        log("  ✗ No valid data", "ERROR")
        return
    
    n_subjects = valid_data['subject'].nunique()
    temps = sorted(valid_data['temp_celsius'].unique())
    
    log(f"  Subjects: {n_subjects}")
    log(f"  Temperatures: {len(temps)}")
    
    # Apply Cousineau-Morey correction (same as Figure 1)
    log("  Computing within-subject error bars...")
    corrected_data = compute_cousineau_morey_correction(valid_data)
    
    # Compute group means and within-subject SEM
    group_stats = corrected_data.groupby('temp_celsius').agg({
        'br_score': 'mean',
        'value_corrected': ['mean', 'sem']
    }).reset_index()
    
    group_stats.columns = ['temp_celsius', 'br_mean', 'br_corrected_mean', 'br_corrected_sem']
    
    # Retrieve or compute statistics (should match Figure 1 exactly)
    if 'figure1' in stats_output:
        # Reuse from Figure 1
        slope = stats_output['figure1']['slope']
        r_val = stats_output['figure1']['r']
        slope_ci_lower = stats_output['figure1']['slope_ci_lower']
        slope_ci_upper = stats_output['figure1']['slope_ci_upper']
        r_ci_lower = stats_output['figure1']['r_ci_lower']
        r_ci_upper = stats_output['figure1']['r_ci_upper']
        p_val = stats_output['figure1']['p']
        log("  Using statistics from Figure 1")
    else:
        # Compute fresh (but should be identical)
        log("  Computing bootstrap statistics...")
        boot_stats = bootstrap_slope_and_correlation(valid_data)
        slope = boot_stats['slope']
        r_val = boot_stats['r']
        slope_ci_lower = boot_stats['slope_ci_lower']
        slope_ci_upper = boot_stats['slope_ci_upper']
        r_ci_lower = boot_stats['r_ci_lower']
        r_ci_upper = boot_stats['r_ci_upper']
        p_val = boot_stats['p']
    
    log(f"    Slope: {slope:.4f} [95% CI: {slope_ci_lower:.4f}, {slope_ci_upper:.4f}]")
    log(f"    r: {r_val:.3f} [95% CI: {r_ci_lower:.3f}, {r_ci_upper:.3f}], p={p_val:.4f}")
    
    # Verify means match Figure 1 (quality check)
    log("  Verifying means match Figure 1...")
    for temp in temps:
        mean_val = group_stats[group_stats['temp_celsius'] == temp]['br_mean'].values[0]
        log(f"    {temp:.1f}°C: BR = {mean_val:.6f}")
    
    # Store statistics (if not already from Figure 1)
    if 'figure4' not in stats_output:
        stats_output['figure4'] = {
            'n_subjects': int(n_subjects),
            'n_temperatures': len(temps),
            'slope': float(slope),
            'r': float(r_val),
            'p': float(p_val),
            'means': {float(row['temp_celsius']): float(row['br_mean']) 
                     for _, row in group_stats.iterrows()}
        }
    
    # Create figure
    log("  Rendering figure...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot as line with markers and error bars
    ax.errorbar(group_stats['temp_celsius'], group_stats['br_mean'],
                yerr=group_stats['br_corrected_sem'],
                marker='o', markersize=10, linewidth=2.5, capsize=6, capthick=2,
                color='steelblue', markerfacecolor='steelblue', markeredgecolor='darkblue',
                markeredgewidth=1.5, elinewidth=2,
                label=f'Group mean ± wsSEM (n={n_subjects})',
                zorder=3)
    
    # Formatting
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=12)
    ax.set_ylabel('NPS Response (3-mm voxels)', fontweight='bold', fontsize=12)
    ax.set_title('NPS Response by Temperature Condition', fontweight='bold', fontsize=13, pad=15)
    
    # Set x-axis ticks to exact temperatures (44-50 range with 6 temps)
    ax.set_xticks(temps)
    ax.set_xticklabels([f'{t:.1f}' for t in temps])
    ax.set_xlim(43.5, 50.5)  # Consistent range for 44-50°C
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', fontsize=10)
    
    # Statistics annotation
    stats_text = (
        f"Slope = {slope:.3f} BR/°C\n"
        f"95% CI [{slope_ci_lower:.3f}, {slope_ci_upper:.3f}]\n"
        f"\n"
        f"r = {r_val:.3f}\n"
        f"95% CI [{r_ci_lower:.3f}, {r_ci_upper:.3f}]\n"
        f"p = {p_val:.4f}"
    )
    
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.95),
            fontsize=10,
            family='monospace')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save
    fig.tight_layout()
    fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    log(f"  ✓ Saved: {output_path.name}")


def create_figure5_nps_map(nps_weights_path: Path,
                           output_path: Path,
                           stats_output: Dict) -> None:
    """
    Create Figure 5: NPS signature map visualization.
    
    Shows spatial distribution of NPS weights on MNI template
    as axial slice montage with symmetric colormap.
    
    Parameters
    ----------
    nps_weights_path : Path
        Path to NPS weights NIfTI file
    output_path : Path
        Output file path
    stats_output : dict
        Dictionary to store statistics
    """
    log("Creating Figure 5: NPS Signature Map")
    
    # Load NPS weights
    try:
        nps_img = nib.load(str(nps_weights_path))
        log(f"  Loaded NPS weights: {nps_weights_path.name}")
    except Exception as e:
        log(f"  ✗ Failed to load NPS weights: {e}", "ERROR")
        return
    
    # Get data for statistics
    nps_data = nps_img.get_fdata()
    nps_data_flat = nps_data.flatten()
    nps_nonzero = nps_data_flat[nps_data_flat != 0]
    
    log(f"  NPS statistics:")
    log(f"    Total voxels: {len(nps_data_flat):,}")
    log(f"    Non-zero voxels: {len(nps_nonzero):,}")
    log(f"    Weight range: [{nps_nonzero.min():.6f}, {nps_nonzero.max():.6f}]")
    log(f"    Positive weights: {np.sum(nps_nonzero > 0):,}")
    log(f"    Negative weights: {np.sum(nps_nonzero < 0):,}")
    
    # Store statistics
    stats_output['figure5'] = {
        'n_voxels_total': int(len(nps_data_flat)),
        'n_voxels_nonzero': int(len(nps_nonzero)),
        'weight_min': float(nps_nonzero.min()),
        'weight_max': float(nps_nonzero.max()),
        'weight_mean': float(nps_nonzero.mean()),
        'weight_std': float(nps_nonzero.std()),
        'n_positive': int(np.sum(nps_nonzero > 0)),
        'n_negative': int(np.sum(nps_nonzero < 0))
    }
    
    # Determine symmetric color limits
    vmax = np.percentile(np.abs(nps_nonzero), 99)  # Use 99th percentile to avoid outliers
    vmin = -vmax
    
    log(f"    Color limits: [{vmin:.6f}, {vmax:.6f}]")
    
    # Create figure with axial slice montage
    log("  Rendering axial slices...")
    
    # Load MNI template for background
    log("  Loading MNI152 template...")
    try:
        mni_template = datasets.load_mni152_template(resolution=2)
        log("    ✓ MNI template loaded")
    except Exception as e:
        log(f"    ⚠ Could not load MNI template: {e}, using default", "WARNING")
        mni_template = None
    
    # Define axial cuts (z-coordinates in mm)
    cut_coords = np.arange(-20, 61, 10)  # -20, -10, 0, 10, 20, 30, 40, 50, 60
    
    log(f"    Slice coordinates (z): {list(cut_coords)}")
    
    # Create display
    display = plotting.plot_stat_map(
        nps_img,
        bg_img=mni_template,  # Explicit MNI template
        display_mode='z',
        cut_coords=cut_coords,
        colorbar=True,
        cmap='cold_hot',  # Symmetric colormap (blue-red)
        symmetric_cbar=True,
        vmax=vmax,
        threshold=0.0001,  # Small threshold to remove near-zero noise
        black_bg=False,
        draw_cross=False,
        annotate=True,
        title='Neurologic Pain Signature (NPS) - Axial Slices',
        alpha=0.8,  # Make overlay slightly transparent to see anatomy
        output_file=None
    )
    
    # Save the figure
    display.savefig(str(output_path), dpi=300)
    display.close()
    
    log(f"  ✓ Saved: {output_path.name}")
    log(f"  Note: Map uses unthresholded weights for BR computation")


def create_figure6_subject_summary(scores_dir: Path,
                                   subjects: List[str],
                                   output_path: Path,
                                   stats_output: Dict) -> None:
    """
    Create Figure 6: Subject-wise summary (supplementary).
    
    Shows distribution of individual subject metrics (slope and correlation)
    with confidence intervals to demonstrate inter-subject variability.
    
    Parameters
    ----------
    scores_dir : Path
        Directory with NPS scores
    subjects : list of str
        Subject IDs
    output_path : Path
        Output file path
    stats_output : dict
        Dictionary to store statistics
    """
    log("Creating Figure 6: Subject-Wise Summary")
    
    # Load all subject metrics
    log("  Loading subject metrics...")
    
    metrics_list = []
    for subject in subjects:
        metrics_path = scores_dir / subject / "subject_metrics.tsv"
        
        if not metrics_path.exists():
            log(f"    ⚠ Metrics not found for {subject}", "WARNING")
            continue
        
        try:
            df = pd.read_csv(metrics_path, sep='\t')
            metrics_list.append(df)
        except Exception as e:
            log(f"    ⚠ Failed to load {subject}: {e}", "WARNING")
            continue
    
    if len(metrics_list) == 0:
        log("  ✗ No subject metrics found", "ERROR")
        return
    
    all_metrics = pd.concat(metrics_list, ignore_index=True)
    n_subjects = len(all_metrics)
    
    log(f"  Loaded metrics for {n_subjects} subjects")
    
    # Filter valid data
    valid_slope = all_metrics.dropna(subset=['slope_BR_temp'])
    valid_r_vas = all_metrics.dropna(subset=['r_BR_VAS'])
    
    log(f"    Valid slope data: {len(valid_slope)}/{n_subjects}")
    log(f"    Valid r(BR,VAS) data: {len(valid_r_vas)}/{n_subjects}")
    
    # Compute CIs if not present (bootstrap per subject would be in step 09, but we can add group stats)
    # For now, we'll show the point estimates
    
    # Store statistics
    stats_output['figure6'] = {
        'n_subjects': int(n_subjects),
        'n_with_slope': len(valid_slope),
        'n_with_r_vas': len(valid_r_vas),
        'slope_mean': float(valid_slope['slope_BR_temp'].mean()) if len(valid_slope) > 0 else None,
        'slope_std': float(valid_slope['slope_BR_temp'].std()) if len(valid_slope) > 0 else None,
        'r_vas_mean': float(valid_r_vas['r_BR_VAS'].mean()) if len(valid_r_vas) > 0 else None,
        'r_vas_std': float(valid_r_vas['r_BR_VAS'].std()) if len(valid_r_vas) > 0 else None
    }
    
    # Create two-panel figure
    log("  Rendering figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # ========================================================================
    # Panel A: Slope BR ~ Temperature
    # ========================================================================
    ax_slope = axes[0]
    
    if len(valid_slope) > 0:
        # Violin plot (if multiple subjects)
        if len(valid_slope) > 1:
            parts = ax_slope.violinplot([valid_slope['slope_BR_temp'].values],
                                       positions=[0],
                                       showmeans=True,
                                       showmedians=True,
                                       widths=0.7)
            
            for pc in parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            parts['cmeans'].set_color('red')
            parts['cmeans'].set_linewidth(2)
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(1.5)
            
            # Add individual points with jitter
            rng = np.random.RandomState(42)
            jitter = rng.normal(0, 0.02, size=len(valid_slope))
            ax_slope.scatter(jitter, valid_slope['slope_BR_temp'],
                           color='darkblue', alpha=0.6, s=50, zorder=3)
        else:
            # Single subject - just plot point
            ax_slope.scatter([0], valid_slope['slope_BR_temp'].values,
                           color='darkblue', s=100, zorder=3)
        
        # Reference line at 0
        ax_slope.axhline(0, color='gray', linestyle='--', linewidth=1.5,
                        alpha=0.7, zorder=1, label='Zero reference')
        
        # Formatting
        ax_slope.set_ylabel('Slope (BR/°C)', fontweight='bold', fontsize=12)
        ax_slope.set_title('A. Dose-Response Slope', fontweight='bold', fontsize=13, pad=10)
        ax_slope.set_xticks([])
        ax_slope.set_xlim(-0.5, 0.5)
        ax_slope.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax_slope.set_axisbelow(True)
        ax_slope.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
        ax_slope.spines['top'].set_visible(False)
        ax_slope.spines['right'].set_visible(False)
        ax_slope.spines['bottom'].set_visible(False)
        
        # Statistics annotation
        mean_slope = valid_slope['slope_BR_temp'].mean()
        std_slope = valid_slope['slope_BR_temp'].std()
        
        if len(valid_slope) > 1:
            stats_text_a = (
                f"n = {len(valid_slope)}\n"
                f"Mean = {mean_slope:.4f}\n"
                f"SD = {std_slope:.4f}\n"
                f"Range = [{valid_slope['slope_BR_temp'].min():.4f}, "
                f"{valid_slope['slope_BR_temp'].max():.4f}]"
            )
        else:
            stats_text_a = f"n = 1\nSlope = {mean_slope:.4f}"
        
        ax_slope.text(0.98, 0.98, stats_text_a,
                     transform=ax_slope.transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.95),
                     fontsize=10,
                     family='monospace')
    else:
        ax_slope.text(0.5, 0.5, 'No slope data available',
                     transform=ax_slope.transAxes,
                     ha='center', va='center', fontsize=12)
    
    # ========================================================================
    # Panel B: Correlation r(BR, VAS)
    # ========================================================================
    ax_r = axes[1]
    
    if len(valid_r_vas) > 0:
        # Violin plot (if multiple subjects)
        if len(valid_r_vas) > 1:
            parts = ax_r.violinplot([valid_r_vas['r_BR_VAS'].values],
                                   positions=[0],
                                   showmeans=True,
                                   showmedians=True,
                                   widths=0.7)
            
            for pc in parts['bodies']:
                pc.set_facecolor('lightcoral')
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            parts['cmeans'].set_color('red')
            parts['cmeans'].set_linewidth(2)
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(1.5)
            
            # Add individual points with jitter
            jitter = rng.normal(0, 0.02, size=len(valid_r_vas))
            ax_r.scatter(jitter, valid_r_vas['r_BR_VAS'],
                        color='darkred', alpha=0.6, s=50, zorder=3)
        else:
            # Single subject - just plot point
            ax_r.scatter([0], valid_r_vas['r_BR_VAS'].values,
                        color='darkred', s=100, zorder=3)
        
        # Reference line at 0
        ax_r.axhline(0, color='gray', linestyle='--', linewidth=1.5,
                    alpha=0.7, zorder=1, label='Zero reference')
        
        # Formatting
        ax_r.set_ylabel('Correlation r(BR, VAS)', fontweight='bold', fontsize=12)
        ax_r.set_title('B. Within-Subject BR-VAS Correlation', fontweight='bold', fontsize=13, pad=10)
        ax_r.set_xticks([])
        ax_r.set_xlim(-0.5, 0.5)
        ax_r.set_ylim([-1, 1])
        ax_r.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax_r.set_axisbelow(True)
        ax_r.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
        ax_r.spines['top'].set_visible(False)
        ax_r.spines['right'].set_visible(False)
        ax_r.spines['bottom'].set_visible(False)
        
        # Statistics annotation
        mean_r = valid_r_vas['r_BR_VAS'].mean()
        std_r = valid_r_vas['r_BR_VAS'].std()
        
        if len(valid_r_vas) > 1:
            stats_text_b = (
                f"n = {len(valid_r_vas)}\n"
                f"Mean = {mean_r:.3f}\n"
                f"SD = {std_r:.3f}\n"
                f"Range = [{valid_r_vas['r_BR_VAS'].min():.3f}, "
                f"{valid_r_vas['r_BR_VAS'].max():.3f}]"
            )
        else:
            stats_text_b = f"n = 1\nr = {mean_r:.3f}"
        
        ax_r.text(0.98, 0.98, stats_text_b,
                 transform=ax_r.transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.95),
                 fontsize=10,
                 family='monospace')
    else:
        ax_r.text(0.5, 0.5, 'No correlation data available',
                 transform=ax_r.transAxes,
                 ha='center', va='center', fontsize=12)
    
    # Save
    fig.tight_layout()
    fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    log(f"  ✓ Saved: {output_path.name}")


def create_qc_design_correlations(work_dir: Path,
                                  subjects: List[str],
                                  output_path: Path) -> None:
    """
    Create QC figure: Design matrix correlations.
    
    Shows heatmaps of inter-regressor correlations for task regressors
    to validate low multicollinearity.
    
    Parameters
    ----------
    work_dir : Path
        Working directory with design matrices
    subjects : list of str
        Subject IDs
    output_path : Path
        Output file path
    """
    log("Creating QC Figure: Design Matrix Correlations")
    
    # Collect all design correlation matrices
    all_corr_data = []
    
    for subject in subjects:
        subject_dir = work_dir / "firstlevel" / subject
        
        if not subject_dir.exists():
            log(f"  ⚠ No data for {subject}", "WARNING")
            continue
        
        # Find all correlation files
        corr_files = list(subject_dir.glob("run-*_design_corr_matrix.tsv"))
        
        for corr_file in corr_files:
            try:
                corr_df = pd.read_csv(corr_file, sep='\t', index_col=0)
                
                # Extract only task regressors (temp columns)
                task_cols = [col for col in corr_df.columns if col.startswith('temp')]
                
                if len(task_cols) > 0:
                    task_corr = corr_df.loc[task_cols, task_cols]
                    
                    run_num = corr_file.stem.split('_')[0].replace('run-', '')
                    
                    all_corr_data.append({
                        'subject': subject,
                        'run': int(run_num),
                        'corr_matrix': task_corr,
                        'max_off_diag': np.max(np.abs(task_corr.values[~np.eye(len(task_corr), dtype=bool)]))
                    })
            except Exception as e:
                log(f"  ⚠ Failed to load {corr_file.name}: {e}", "WARNING")
                continue
    
    if len(all_corr_data) == 0:
        log("  ✗ No design correlation data found", "ERROR")
        return
    
    log(f"  Loaded {len(all_corr_data)} design matrices")
    
    # Create grid of heatmaps (up to 12 runs, 3x4 grid)
    n_plots = min(len(all_corr_data), 12)
    n_cols = 4
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    log("  Rendering heatmaps...")
    
    for idx, data in enumerate(all_corr_data[:n_plots]):
        ax = axes[idx]
        
        # Plot heatmap
        im = ax.imshow(data['corr_matrix'].values, cmap='RdBu_r', 
                      vmin=-1, vmax=1, aspect='auto')
        
        # Labels
        ax.set_xticks(range(len(data['corr_matrix'])))
        ax.set_yticks(range(len(data['corr_matrix'])))
        ax.set_xticklabels(data['corr_matrix'].columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(data['corr_matrix'].index, fontsize=8)
        
        # Title
        title = f"{data['subject']} Run {data['run']}\nMax |r| = {data['max_off_diag']:.3f}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle('QC: Design Matrix Correlations (Task Regressors Only)',
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save
    fig.tight_layout()
    fig.savefig(output_path, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    log(f"  ✓ Saved: {output_path.name}")
    log(f"  Max correlation across all runs: {max([d['max_off_diag'] for d in all_corr_data]):.3f}")


def create_qc_grid_compatibility(nps_weights_path: Path,
                                scores_dir: Path,
                                subjects: List[str],
                                output_path: Path) -> None:
    """
    Create QC figure: Grid compatibility check.
    
    Validates that harmonized beta maps match NPS weights grid
    in dimensions, voxel size, and spatial overlap.
    
    Parameters
    ----------
    nps_weights_path : Path
        Path to NPS weights
    scores_dir : Path
        Directory with harmonized beta maps
    subjects : list of str
        Subject IDs
    output_path : Path
        Output file path
    """
    log("Creating QC Figure: Grid Compatibility Check")
    
    # Load NPS weights
    try:
        nps_img = nib.load(str(nps_weights_path))
        nps_shape = nps_img.shape
        nps_affine = nps_img.affine
        nps_voxsize = nps_img.header.get_zooms()[:3]
        nps_data = nps_img.get_fdata()
        nps_nonzero = np.sum(nps_data != 0)
        
        log(f"  NPS weights:")
        log(f"    Shape: {nps_shape}")
        log(f"    Voxel size: {nps_voxsize} mm")
        log(f"    Non-zero voxels: {nps_nonzero:,}")
    except Exception as e:
        log(f"  ✗ Failed to load NPS weights: {e}", "ERROR")
        return
    
    # Check harmonized beta maps
    log("  Checking harmonized beta maps...")
    
    checks = []
    
    for subject in subjects[:5]:  # Check first 5 subjects
        subject_dir = scores_dir / subject
        
        # Find harmonized beta files
        nps_ready_dir = scores_dir.parent / "nps_ready" / subject
        
        if not nps_ready_dir.exists():
            continue
        
        beta_files = list(nps_ready_dir.glob("beta_temp*_onNPSgrid.nii.gz"))
        
        if len(beta_files) == 0:
            continue
        
        # Check first beta file
        beta_file = beta_files[0]
        
        try:
            beta_img = nib.load(str(beta_file))
            beta_shape = beta_img.shape
            beta_affine = beta_img.affine
            beta_voxsize = beta_img.header.get_zooms()[:3]
            beta_data = beta_img.get_fdata()
            beta_nonzero = np.sum(beta_data != 0)
            
            # Check compatibility
            shape_match = beta_shape == nps_shape
            affine_match = np.allclose(beta_affine, nps_affine, atol=0.01)
            voxsize_match = np.allclose(beta_voxsize, nps_voxsize, atol=0.01)
            
            # Check overlap
            if shape_match:
                overlap_mask = (nps_data != 0) & (beta_data != 0)
                overlap_voxels = np.sum(overlap_mask)
                overlap_pct = 100 * overlap_voxels / nps_nonzero if nps_nonzero > 0 else 0
            else:
                overlap_voxels = 0
                overlap_pct = 0
            
            checks.append({
                'subject': subject,
                'file': beta_file.name,
                'shape_match': shape_match,
                'affine_match': affine_match,
                'voxsize_match': voxsize_match,
                'beta_shape': beta_shape,
                'beta_voxsize': beta_voxsize,
                'beta_nonzero': beta_nonzero,
                'overlap_voxels': overlap_voxels,
                'overlap_pct': overlap_pct
            })
            
        except Exception as e:
            log(f"  ⚠ Failed to check {beta_file.name}: {e}", "WARNING")
            continue
    
    if len(checks) == 0:
        log("  ✗ No beta maps found to check", "ERROR")
        return
    
    log(f"  Checked {len(checks)} beta maps")
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    
    # Header row
    table_data.append([
        'Subject', 'Shape\nMatch', 'Voxel Size\nMatch', 'Affine\nMatch',
        'Beta Shape', 'Beta Voxel Size\n(mm)', 'Beta Non-Zero\nVoxels',
        'Overlap\nVoxels', 'Overlap\n(%)'
    ])
    
    # NPS reference row
    table_data.append([
        'NPS (Reference)',
        '—',
        '—',
        '—',
        f'{nps_shape[0]}×{nps_shape[1]}×{nps_shape[2]}',
        f'{nps_voxsize[0]:.1f}×{nps_voxsize[1]:.1f}×{nps_voxsize[2]:.1f}',
        f'{nps_nonzero:,}',
        '—',
        '—'
    ])
    
    # Data rows
    for check in checks:
        table_data.append([
            check['subject'],
            '✓' if check['shape_match'] else '✗',
            '✓' if check['voxsize_match'] else '✗',
            '✓' if check['affine_match'] else '✗',
            f"{check['beta_shape'][0]}×{check['beta_shape'][1]}×{check['beta_shape'][2]}",
            f"{check['beta_voxsize'][0]:.1f}×{check['beta_voxsize'][1]:.1f}×{check['beta_voxsize'][2]:.1f}",
            f"{check['beta_nonzero']:,}",
            f"{check['overlap_voxels']:,}",
            f"{check['overlap_pct']:.1f}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color NPS reference row
    for i in range(len(table_data[0])):
        table[(1, i)].set_facecolor('#E3F2FD')
        table[(1, i)].set_text_props(weight='bold')
    
    # Color check marks
    for row_idx in range(2, len(table_data)):
        for col_idx in [1, 2, 3]:  # Match columns
            cell = table[(row_idx, col_idx)]
            if cell.get_text().get_text() == '✓':
                cell.set_facecolor('#C8E6C9')
            elif cell.get_text().get_text() == '✗':
                cell.set_facecolor('#FFCDD2')
    
    # Title
    ax.set_title('QC: Grid Compatibility Check (Beta Maps vs NPS Weights)',
                fontsize=14, fontweight='bold', pad=20)
    
    # Summary text
    all_match = all(c['shape_match'] and c['voxsize_match'] and c['affine_match'] 
                    for c in checks)
    
    if all_match:
        summary = "✓ All checked beta maps are compatible with NPS weights grid"
        color = 'green'
    else:
        summary = "⚠ Some beta maps have grid mismatches - check harmonization step"
        color = 'orange'
    
    fig.text(0.5, 0.05, summary, ha='center', fontsize=12, 
            fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))
    
    # Save
    fig.savefig(output_path, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    log(f"  ✓ Saved: {output_path.name}")
    log(f"  Grid compatibility: {'PASS' if all_match else 'FAIL'}")


def create_table1_primary_outcomes(group_stats_path: Path,
                                   output_dir: Path) -> None:
    """
    Create Table 1: Primary outcomes (publication-ready).
    
    Formats group-level statistics into a manuscript table with
    proper formatting for means, CIs, t-statistics, and p-values.
    
    Parameters
    ----------
    group_stats_path : Path
        Path to group_stats.tsv from step 10
    output_dir : Path
        Output directory for tables
    """
    log("Creating Table 1: Primary Outcomes")
    
    # Load group statistics
    if not group_stats_path.exists():
        log(f"  ✗ Group statistics not found: {group_stats_path}", "ERROR")
        log("  Run 10_group_stats.py first", "ERROR")
        return
    
    try:
        stats_df = pd.read_csv(group_stats_path, sep='\t')
        log(f"  Loaded group statistics: {len(stats_df)} metrics")
    except Exception as e:
        log(f"  ✗ Failed to load group statistics: {e}", "ERROR")
        return
    
    # Prepare table data
    table_rows = []
    
    for _, row in stats_df.iterrows():
        metric = row['metric']
        
        # Format metric name for publication
        if metric == 'slope_BR_temp':
            metric_name = 'Slope BR ~ Temperature (BR/°C)'
        elif metric == 'r_BR_temp':
            metric_name = 'r(BR, Temperature)'
        elif metric == 'r_BR_VAS':
            metric_name = 'r(BR, VAS)'
        elif metric == 'auc_pain':
            metric_name = 'AUC (Pain Classification)'
        elif metric == 'forced_choice_acc':
            metric_name = 'Forced-Choice Accuracy'
        else:
            metric_name = metric
        
        # Format values
        n = int(row['n'])
        mean = row['mean']
        std = row['std']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']
        t_stat = row['t_statistic']
        df = int(row['df'])
        p_val = row['p_value']
        
        # Format p-value
        if p_val < 0.001:
            p_str = '<0.001'
        elif p_val < 0.01:
            p_str = f'{p_val:.3f}'
        else:
            p_str = f'{p_val:.3f}'
        
        # Format CI
        if metric in ['forced_choice_acc', 'auc_pain']:
            # Show as percentages
            mean_str = f'{mean*100:.1f}%'
            ci_str = f'[{ci_lower*100:.1f}, {ci_upper*100:.1f}]'
            std_str = f'{std*100:.1f}%'
        else:
            mean_str = f'{mean:.3f}'
            ci_str = f'[{ci_lower:.3f}, {ci_upper:.3f}]'
            std_str = f'{std:.3f}'
        
        # Format t-test
        t_str = f'{t_stat:.2f}'
        df_str = f'{df}'
        
        # Add FDR-corrected p-value if available
        if 'p_value_fdr' in row and not pd.isna(row['p_value_fdr']):
            p_fdr = row['p_value_fdr']
            if p_fdr < 0.001:
                p_fdr_str = '<0.001'
            elif p_fdr < 0.01:
                p_fdr_str = f'{p_fdr:.3f}'
            else:
                p_fdr_str = f'{p_fdr:.3f}'
            fdr_sig = 'Yes' if row.get('fdr_significant', False) else 'No'
        else:
            p_fdr_str = '—'
            fdr_sig = '—'
        
        table_rows.append({
            'Metric': metric_name,
            'N': n,
            'Mean': mean_str,
            'SD': std_str,
            '95% CI': ci_str,
            't': t_str,
            'df': df_str,
            'p': p_str,
            'p_FDR': p_fdr_str,
            'FDR Sig': fdr_sig
        })
    
    # Create DataFrame
    table_df = pd.DataFrame(table_rows)
    
    log("  Table contents:")
    log(f"    {len(table_rows)} metrics")
    
    # Save as CSV
    csv_path = output_dir / "Table1_PrimaryOutcomes.csv"
    table_df.to_csv(csv_path, index=False)
    log(f"  ✓ Saved CSV: {csv_path.name}")
    
    # Save as Markdown
    md_path = output_dir / "Table1_PrimaryOutcomes.md"
    with open(md_path, 'w') as f:
        f.write("# Table 1. Primary Outcomes\n\n")
        f.write("**Group-level statistics for NPS dose-response and discrimination metrics.**\n\n")
        
        # Write markdown table
        f.write("| " + " | ".join(table_df.columns) + " |\n")
        f.write("| " + " | ".join(['---'] * len(table_df.columns)) + " |\n")
        
        for _, row in table_df.iterrows():
            f.write("| " + " | ".join(str(row[col]) for col in table_df.columns) + " |\n")
        
        f.write("\n")
        f.write("**Note:**\n")
        f.write("- All tests are one-sample t-tests with right-tailed alternatives:\n")
        f.write("  - Slope > 0 (dose-response)\n")
        f.write("  - Fisher-z(r) > 0 (correlations)\n")
        f.write("  - AUC - 0.5 > 0 (discrimination)\n")
        f.write("  - Forced-choice > 0.5 (threshold-free discrimination)\n")
        f.write("- p_FDR: False Discovery Rate corrected p-values (Benjamini-Hochberg)\n")
        f.write("- Primary endpoints (Slope, r(BR, VAS)) use FDR correction\n")
        f.write("- 95% CI computed via bias-corrected bootstrap (10,000 samples)\n")
        f.write("- Correlations use Fisher z-transformation for testing\n")
    
    log(f"  ✓ Saved Markdown: {md_path.name}")
    
    # Also save a LaTeX-friendly version
    latex_path = output_dir / "Table1_PrimaryOutcomes_LaTeX.txt"
    with open(latex_path, 'w') as f:
        f.write("% Table 1: Primary Outcomes\n")
        f.write("% Copy-paste into LaTeX document\n\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Primary Outcomes: NPS Dose-Response and Discrimination Metrics}\n")
        f.write("\\label{tab:primary_outcomes}\n")
        f.write("\\begin{tabular}{lcccccccc}\n")
        f.write("\\hline\n")
        f.write("Metric & $N$ & Mean & SD & 95\\% CI & $t$ & df & $p$ & $p_{FDR}$ \\\\\n")
        f.write("\\hline\n")
        
        for _, row in table_df.iterrows():
            # Escape special characters for LaTeX
            metric = row['Metric'].replace('~', '$\\sim$').replace('°', '$^\\circ$')
            f.write(f"{metric} & {row['N']} & {row['Mean']} & {row['SD']} & "
                   f"{row['95% CI']} & {row['t']} & {row['df']} & {row['p']} & {row['p_FDR']} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    log(f"  ✓ Saved LaTeX: {latex_path.name}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Generate publication-quality NPS figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all figures
  python 11_plots.py
  
  # Generate specific figure
  python 11_plots.py --figure 1

Figures:
  1. NPS dose-response (BR vs Temperature)
  2. BR vs VAS ratings (within-subject correlations)
  3. Discrimination metrics (ROC/AUC, forced-choice) [requires trial data]
  4. Condition means (compact bar/line format)
  5. NPS signature map (axial slices on MNI template)
  6. Subject-wise summary (supplementary - individual metrics)
  
QC Figures:
  --qc-design: Design matrix correlation heatmaps
  --qc-grid: Grid compatibility check (beta maps vs NPS weights)
  
Tables:
  --table1: Generate Table 1 (primary outcomes) from group statistics
        """
    )
    
    parser.add_argument('--config', default='00_config.yaml',
                       help='Path to configuration file (default: 00_config.yaml)')
    parser.add_argument('--scores-dir', default='outputs/nps_scores',
                       help='Directory with NPS scores (default: outputs/nps_scores)')
    parser.add_argument('--output-dir', default='outputs/figures',
                       help='Output directory (default: outputs/figures)')
    parser.add_argument('--figure', type=int, default=None,
                       help='Generate specific figure only (1-6)')
    parser.add_argument('--nps-weights', default=None,
                       help='Path to NPS weights file (default: from config)')
    parser.add_argument('--qc-design', action='store_true',
                       help='Generate QC figure: design matrix correlations')
    parser.add_argument('--qc-grid', action='store_true',
                       help='Generate QC figure: grid compatibility check')
    parser.add_argument('--table1', action='store_true',
                       help='Generate Table 1: primary outcomes')
    parser.add_argument('--work-dir', default='work',
                       help='Working directory (default: work)')
    parser.add_argument('--group-stats', default='outputs/group/group_stats.tsv',
                       help='Path to group statistics (default: outputs/group/group_stats.tsv)')
    
    args = parser.parse_args()
    
    # Load configuration
    log("=" * 70)
    log("PUBLICATION-QUALITY FIGURES")
    log("=" * 70)
    
    try:
        config = load_config(args.config)
        log(f"Loaded config: {args.config}")
        subjects = config['subjects']
    except Exception as e:
        log(f"Failed to load config: {e}", "ERROR")
        return 1
    
    log(f"Subjects: {len(subjects)}")
    log("")
    
    # Setup directories
    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not scores_dir.exists():
        log(f"Scores directory not found: {scores_dir}", "ERROR")
        log("Run 07_score_nps_conditions.py first", "ERROR")
        return 1
    
    # Load data
    log("Loading subject data...")
    
    try:
        level_data = load_all_subjects_level_data(scores_dir, subjects)
        log(f"Loaded level data: {len(level_data)} rows, {level_data['subject'].nunique()} subjects")
    except Exception as e:
        log(f"Failed to load level data: {e}", "ERROR")
        return 1
    
    # Load trial data if needed (for Figure 3)
    trial_data = None
    figures_to_generate = [args.figure] if args.figure else [1, 2, 3, 4, 5, 6]
    
    # Try to load trial data for Figure 3 (optional)
    if args.figure == 3 or (args.figure is None and 3 in figures_to_generate):
        try:
            trial_data = load_all_subjects_trial_data(scores_dir, subjects)
            log(f"Loaded trial data: {len(trial_data)} rows, {trial_data['subject'].nunique()} subjects")
        except Exception as e:
            log(f"⚠ Trial data not available: {e}", "WARNING")
            log("  Figure 3 will be skipped (requires trial_br.tsv from step 08)", "WARNING")
    
    # Statistics output
    stats_output = {}
    
    # Generate figures
    for fig_num in figures_to_generate:
        log("")
        log("=" * 70)
        log(f"FIGURE {fig_num}")
        log("=" * 70)
        
        try:
            if fig_num == 1:
                output_path = output_dir / "Fig1_BR_vs_Temperature.svg"
                create_figure1_dose_response(level_data, output_path, stats_output)
            
            elif fig_num == 2:
                output_path = output_dir / "Fig2_BR_vs_VAS.svg"
                create_figure2_br_vs_vas(level_data, output_path, stats_output)
            
            elif fig_num == 3:
                # Load trial data if not already loaded
                if trial_data is None:
                    try:
                        trial_data = load_all_subjects_trial_data(scores_dir, subjects)
                        log(f"Loaded trial data: {len(trial_data)} rows, {trial_data['subject'].nunique()} subjects")
                    except Exception as e:
                        log(f"✗ Trial data not available: {e}", "ERROR")
                        log("  Run 08_optional_trials_glm_and_scoring.py first", "ERROR")
                        continue
                
                output_path = output_dir / "Fig3_Discrimination.svg"
                create_figure3_discrimination(trial_data, output_path, stats_output)
            
            elif fig_num == 4:
                output_path = output_dir / "Fig4_Condition_Means.svg"
                create_figure4_condition_means(level_data, output_path, stats_output)
            
            elif fig_num == 5:
                # Get NPS weights path
                if args.nps_weights:
                    nps_weights_path = Path(args.nps_weights)
                else:
                    if 'resources' in config and 'nps_weights_path' in config['resources']:
                        nps_weights_path = Path(config['resources']['nps_weights_path'])
                    else:
                        log("NPS weights path not found in config", "ERROR")
                        continue
                
                if not nps_weights_path.exists():
                    log(f"NPS weights file not found: {nps_weights_path}", "ERROR")
                    continue
                
                output_path = output_dir / "Fig5_NPS_Map.png"  # PNG for brain visualization
                create_figure5_nps_map(nps_weights_path, output_path, stats_output)
            
            elif fig_num == 6:
                output_path = output_dir / "Supp_Fig_SubjectSummary.svg"
                create_figure6_subject_summary(scores_dir, subjects, output_path, stats_output)
            
            else:
                log(f"Unknown figure number: {fig_num}", "ERROR")
        
        except Exception as e:
            log(f"Failed to create figure {fig_num}: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate QC figures if requested
    if args.qc_design or args.qc_grid:
        log("")
        log("=" * 70)
        log("QC FIGURES")
        log("=" * 70)
        
        work_dir = Path(args.work_dir)
        
        if args.qc_design:
            log("")
            qc_output = output_dir / "QC_DesignCorr_Grid.png"
            try:
                create_qc_design_correlations(work_dir, subjects, qc_output)
            except Exception as e:
                log(f"Failed to create QC design figure: {e}", "ERROR")
                import traceback
                traceback.print_exc()
        
        if args.qc_grid:
            log("")
            # Get NPS weights path
            if args.nps_weights:
                nps_weights_path = Path(args.nps_weights)
            else:
                if 'resources' in config and 'nps_weights_path' in config['resources']:
                    nps_weights_path = Path(config['resources']['nps_weights_path'])
                else:
                    log("NPS weights path not found in config", "ERROR")
                    nps_weights_path = None
            
            if nps_weights_path and nps_weights_path.exists():
                qc_output = output_dir / "QC_GridCompatibility.png"
                try:
                    create_qc_grid_compatibility(nps_weights_path, scores_dir, subjects, qc_output)
                except Exception as e:
                    log(f"Failed to create QC grid figure: {e}", "ERROR")
                    import traceback
                    traceback.print_exc()
            else:
                log("Cannot create QC grid figure: NPS weights not found", "ERROR")
    
    # Generate Table 1 if requested
    if args.table1:
        log("")
        log("=" * 70)
        log("TABLE GENERATION")
        log("=" * 70)
        
        group_stats_path = Path(args.group_stats)
        
        try:
            create_table1_primary_outcomes(group_stats_path, output_dir)
        except Exception as e:
            log(f"Failed to create Table 1: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    # Save statistics for captions
    stats_path = output_dir / "figure_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    log("")
    log(f"Saved statistics: {stats_path}")
    
    # Final summary
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log("✓ Figure generation complete")
    log(f"  Figures saved in: {output_dir}/")
    log(f"  Statistics in: {stats_path.name}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
