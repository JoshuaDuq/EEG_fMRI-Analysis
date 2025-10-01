#!/usr/bin/env python3
"""
Configuration loader for fMRI pipeline.

Provides utilities to load and validate 00_config.yaml across all pipeline scripts.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import sys


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to config file. If None, looks for 00_config.yaml in:
        1. Current directory
        2. Script's parent directory
        3. fmri_pipeline directory
    
    Returns
    -------
    dict
        Configuration dictionary
    
    Raises
    ------
    FileNotFoundError
        If config file cannot be found
    """
    if config_path is None:
        # Search common locations
        search_paths = [
            Path.cwd() / "00_config.yaml",
            Path(__file__).parent / "00_config.yaml",
            Path(__file__).parent.parent / "fmri_pipeline" / "00_config.yaml"
        ]
        
        for path in search_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(
                "Could not find 00_config.yaml in standard locations. "
                "Please specify config_path explicitly."
            )
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths to absolute (relative to config file location)
    config = _resolve_paths(config, config_path.parent)
    
    return config


def _resolve_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """
    Resolve relative paths in config to absolute paths.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    base_dir : Path
        Base directory for resolving relative paths
    
    Returns
    -------
    dict
        Config with resolved paths
    """
    # Core directory paths
    for key in ['bids_root', 'fmriprep_root', 'derivatives_root']:
        if key in config:
            config[key] = str((base_dir / config[key]).resolve())
    
    # Resource paths
    if 'resources' in config:
        for key, value in config['resources'].items():
            if key.endswith('_path') and isinstance(value, str):
                config['resources'][key] = str((base_dir / value).resolve())
    
    # Output directories
    if 'outputs' in config:
        for key, value in config['outputs'].items():
            if key.endswith('_dir') and isinstance(value, str):
                config['outputs'][key] = str((base_dir / value).resolve())
    
    # Logging directory
    if 'logging' in config and 'log_dir' in config['logging']:
        config['logging']['log_dir'] = str((base_dir / config['logging']['log_dir']).resolve())
    
    return config


def validate_config(config: Dict[str, Any], check_files: bool = True) -> None:
    """
    Validate configuration for common issues.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    check_files : bool, default=True
        Whether to check that referenced files/directories exist
    
    Raises
    ------
    AssertionError
        If validation fails
    """
    # Check required top-level keys
    required_keys = ['bids_root', 'fmriprep_root', 'subjects', 'task', 'runs', 'glm', 'confounds']
    for key in required_keys:
        assert key in config, f"Missing required config key: {key}"
    
    # Check GLM settings
    assert 'tr' in config['glm'], "Missing GLM TR"
    assert 'temp_labels' in config['glm'], "Missing temperature labels"
    assert 'nuisance_events' in config['glm'], "Missing nuisance events"
    
    # Check TR consistency
    if 'acquisition_params' in config and 'tr' in config['acquisition_params']:
        assert abs(config['glm']['tr'] - config['acquisition_params']['tr']) < 0.01, \
            f"TR mismatch: GLM={config['glm']['tr']}, acquisition_params={config['acquisition_params']['tr']}"
    
    # Check confound settings
    assert 'motion_24_params' in config['confounds'], "Missing 24-parameter motion model"
    assert len(config['confounds']['motion_24_params']) == 24, \
        f"Expected 24 motion params, got {len(config['confounds']['motion_24_params'])}"
    
    # Check file existence
    if check_files:
        bids_root = Path(config['bids_root'])
        assert bids_root.exists(), f"BIDS root not found: {bids_root}"
        
        fmriprep_root = Path(config['fmriprep_root'])
        assert fmriprep_root.exists(), f"fMRIPrep root not found: {fmriprep_root}"
        
        # Check subjects exist in BIDS
        for subj in config['subjects']:
            subj_dir = bids_root / subj
            assert subj_dir.exists(), f"Subject directory not found: {subj_dir}"
    
    print("✓ Configuration validation passed")


def get_subject_files(config: Dict[str, Any], subject: str, run: int, 
                     file_type: str = 'bold') -> Path:
    """
    Build standardized file paths for a subject/run.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject ID (e.g., 'sub-0001')
    run : int
        Run number (1-6)
    file_type : str
        Type of file to retrieve:
        - 'bold': Preprocessed BOLD
        - 'mask': Brain mask
        - 'confounds': Confounds TSV
        - 'events': Events TSV
        - 'anat': Anatomical T1w
    
    Returns
    -------
    Path
        Path to requested file
    """
    task = config['task']
    space = config['space']
    res = config.get('resolution', '2')  # Default to 2mm
    
    if file_type == 'bold':
        root = Path(config['fmriprep_root'])
        # Try both run-{run} and run-{run:02d} formats for compatibility
        candidate = root / subject / "func" / \
            f"{subject}_task-{task}_run-{run}_space-{space}_res-{res}_desc-preproc_bold.nii.gz"
        if candidate.exists():
            return candidate
        return root / subject / "func" / \
            f"{subject}_task-{task}_run-{run:02d}_space-{space}_res-{res}_desc-preproc_bold.nii.gz"
    
    elif file_type == 'mask':
        root = Path(config['fmriprep_root'])
        candidate = root / subject / "func" / \
            f"{subject}_task-{task}_run-{run}_space-{space}_res-{res}_desc-brain_mask.nii.gz"
        if candidate.exists():
            return candidate
        return root / subject / "func" / \
            f"{subject}_task-{task}_run-{run:02d}_space-{space}_res-{res}_desc-brain_mask.nii.gz"
    
    elif file_type == 'confounds':
        root = Path(config['fmriprep_root'])
        candidate = root / subject / "func" / \
            f"{subject}_task-{task}_run-{run}_desc-confounds_timeseries.tsv"
        if candidate.exists():
            return candidate
        return root / subject / "func" / \
            f"{subject}_task-{task}_run-{run:02d}_desc-confounds_timeseries.tsv"
    
    elif file_type == 'events':
        root = Path(config['bids_root'])
        # Try zero-padded first (BIDS standard), then single-digit
        candidate = root / subject / "func" / \
            f"{subject}_task-{task}_run-{run:02d}_events.tsv"
        if candidate.exists():
            return candidate
        return root / subject / "func" / \
            f"{subject}_task-{task}_run-{run}_events.tsv"
    
    elif file_type == 'anat':
        root = Path(config['fmriprep_root'])
        acq = config.get('anat_acquisition', 'mprageipat2')
        return root / subject / "anat" / \
            f"{subject}_acq-{acq}_space-{space}_res-{res}_desc-preproc_T1w.nii.gz"
    
    else:
        raise ValueError(f"Unknown file_type: {file_type}")


def get_confound_columns(config: Dict[str, Any], 
                        include_motion: bool = True,
                        include_compcor: bool = None,
                        include_physio: bool = False) -> list:
    """
    Get list of confound column names based on config.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    include_motion : bool, default=True
        Include 24-parameter motion model
    include_compcor : bool, optional
        Include CompCor components. If None, uses config setting.
    include_physio : bool, default=False
        Include physiological regressors (GSR, WM, CSF)
    
    Returns
    -------
    list
        Column names to extract from confounds file
    """
    columns = []
    
    # Motion parameters
    if include_motion:
        columns.extend(config['confounds']['motion_24_params'])
    
    # CompCor
    if include_compcor is None:
        include_compcor = config['confounds'].get('compcor', {}).get('enabled', False)
    
    if include_compcor:
        n_comp = config['confounds']['compcor'].get('n_components', 5)
        method = config['confounds']['compcor'].get('method', 'acompcor')
        
        if method in ['acompcor', 'both']:
            columns.extend([f"a_comp_cor_{i:02d}" for i in range(n_comp)])
        if method in ['tcompcor', 'both']:
            columns.extend([f"t_comp_cor_{i:02d}" for i in range(n_comp)])
    
    # Physiological
    if include_physio:
        physio_cols = config['confounds'].get('physiological', [])
        columns.extend(physio_cols)
    
    return columns


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print human-readable summary of configuration."""
    print("=" * 70)
    print("FMRI PIPELINE CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print(f"\n📁 Paths:")
    print(f"  BIDS root:      {config['bids_root']}")
    print(f"  fMRIPrep root:  {config['fmriprep_root']}")
    print(f"  Derivatives:    {config.get('derivatives_root', 'N/A')}")
    
    print(f"\n👥 Subjects: {', '.join(config['subjects'])}")
    print(f"📋 Task: {config['task']}")
    print(f"🔢 Runs: {', '.join(map(str, config['runs']))}")
    
    print(f"\n🧠 Acquisition:")
    print(f"  TR: {config['glm']['tr']} sec")
    print(f"  Space: {config['space']}")
    
    print(f"\n📊 GLM Settings:")
    print(f"  HRF: {config['glm']['hrf']['model']}")
    print(f"  High-pass: {config['glm']['high_pass_sec']} sec")
    print(f"  Stimulus duration: {config['glm']['stim_dur_sec']} sec")
    print(f"  Temperature conditions: {len(config['glm']['temp_labels'])}")
    print(f"  Nuisance events: {', '.join(config['glm']['nuisance_events'])}")
    
    print(f"\n🎯 Confounds:")
    print(f"  Motion: 24-parameter model")
    if config['confounds'].get('compcor', {}).get('enabled'):
        print(f"  CompCor: {config['confounds']['compcor']['n_components']} components")
    
    print(f"\n💾 Outputs:")
    print(f"  GLM dir: {config.get('outputs', {}).get('glm_dir', 'N/A')}")
    print(f"  Figure formats: {', '.join(config.get('outputs', {}).get('figure_formats', ['png']))}")
    
    print("=" * 70)


if __name__ == "__main__":
    """Test config loading and validation."""
    try:
        config = load_config()
        print_config_summary(config)
        validate_config(config, check_files=True)
        print("\n✓ Config successfully loaded and validated")
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
