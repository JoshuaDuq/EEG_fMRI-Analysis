"""Configuration loader for EEG pipeline.

Provides utilities to load and access centralized configuration from YAML files.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


class _NestedDict:
    """Wrapper for nested dictionary access via attributes."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        
    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
        
        if key not in self._data:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
        
        value = self._data[key]
        if isinstance(value, dict):
            return _NestedDict(value)
        return value
        
    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)
        
    def __contains__(self, key: str) -> bool:
        return key in self._data
        
    def keys(self):
        return self._data.keys()
        
    def items(self):
        return self._data.items()
        
    def values(self):
        return self._data.values()


class EEGConfig:
    """Configuration manager for EEG pipeline.
    
    Loads configuration from YAML file and provides convenient attribute access.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. If None, looks for eeg_config.yaml
                        in the same directory as this module.
        """
        self._data: Dict[str, Any] = {}
        self._project_root: Optional[Path] = None
        
        if config_path is None:
            # Look for config in same directory as this file
            config_dir = Path(__file__).parent
            config_path = config_dir / "eeg_config.yaml"
            
        # Ensure path is a Path object
        config_path = Path(config_path)
        
        self.config_path = Path(config_path)
        self.load()
        
    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise ConfigError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML config: {e}") from e
        except OSError as e:
            raise ConfigError(f"Failed to load config file: {e}") from e
            
        # Resolve project root and update paths
        self._resolve_paths()
        
        # Apply environment settings and setup matplotlib
        self.apply_thread_limits()
        self.setup_matplotlib()
    
    def __getattr__(self, key: str) -> Any:
        """Delegate attribute access to the nested config.

        For convenience and robustness, return an empty namespace for common
        optional sections that may be omitted from the YAML (e.g., 'visualization').
        This avoids AttributeError in code paths that probe for these sections
        before falling back to defaults.
        """
        if key.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

        if key not in self._data:
            if key in {"visualization", "output", "logging"}:
                return _NestedDict({})
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

        value = self._data[key]
        if isinstance(value, dict):
            return _NestedDict(value)
        return value
            
    def _resolve_paths(self) -> None:
        """Resolve relative paths to absolute paths."""
        # Find project root (parent of eeg_pipeline directory)
        pipeline_dir = Path(__file__).parent
        self._project_root = pipeline_dir.parent
        
        # Update path configurations to absolute paths
        if "paths" in self._data:
            paths = self._data["paths"]
            for key, value in paths.items():
                if isinstance(value, str) and key != "project_root":
                    if value.startswith("eeg_pipeline/"):
                        paths[key] = str(self._project_root / value)
                    elif not Path(value).is_absolute():
                        paths[key] = str(self._project_root / value)
                        
        # Update sourcecoords_file path in features section
        if "features" in self._data and "sourcecoords_file" in self._data["features"]:
            sourcecoords = self._data["features"]["sourcecoords_file"]
            if not Path(sourcecoords).is_absolute():
                self._data["features"]["sourcecoords_file"] = str(self._project_root / sourcecoords)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'project.task', 'decoding.models.elasticnet')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration section by key."""
        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration section exists."""
        return key in self._data
        
    @property
    def project_root(self) -> Path:
        """Get resolved project root path."""
        if self._project_root is None:
            raise ConfigError("Project root not resolved")
        return self._project_root
        
    @property
    def bids_root(self) -> Path:
        """Get BIDS root path."""
        return Path(self.get("paths.bids_root"))
        
    @property 
    def deriv_root(self) -> Path:
        """Get derivatives root path."""
        return Path(self.get("paths.deriv_root"))
        
    @property
    def subjects(self) -> list[str]:
        """Get list of subjects to process."""
        return self.get("project.subjects", [])
        
    @property
    def task(self) -> str:
        """Get BIDS task name."""
        return self.get("project.task", "thermalactive")
        
    @property
    def frequency_bands(self) -> Dict[str, tuple[float, float]]:
        """Get frequency bands as dict with tuple values."""
        bands = self.get("frequency_bands", {})
        return {name: tuple(freqs) for name, freqs in bands.items()}
        
    def apply_thread_limits(self) -> None:
        """Apply thread limit environment variables."""
        limits = self.get("environment.thread_limits", {})
        for var, value in limits.items():
            os.environ.setdefault(var, str(value))
            
    def setup_matplotlib(self) -> None:
        """Setup matplotlib with headless backend and style."""
        import matplotlib
        matplotlib.use("Agg")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Journal-quality plot style
        sns.set_theme(context="paper", style="white", font_scale=1.05)
        plt.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "0.85",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "savefig.dpi": self.get("visualization.dpi", 300),
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
        })
        
    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with legacy CONFIG variable.
        
        This helps maintain backward compatibility with existing code that expects
        the CONFIG dictionary structure.
        """
        return {
            # Paths
            "paths": {
                "results_subdir": self.get("decoding.paths.results_subdir", "decoding"),
                "plots_subdir": self.get("decoding.paths.plots_subdir", "plots"),
                "indices": self.get("decoding.paths.indices", {}),
                "best_params": self.get("decoding.paths.best_params", {}),
                "predictions": self.get("decoding.paths.predictions", {}),
                "per_subject_metrics": self.get("decoding.paths.per_subject_metrics", {}),
                "summaries": self.get("decoding.paths.summaries", {}),
            },
            # CV
            "cv": {
                "inner_splits": self.get("decoding.cv.inner_splits", 5),
            },
            # Models
            "models": self.get("decoding.models", {}),
            # Analysis
            "analysis": self.get("decoding.analysis", {}),
            # Flags
            "flags": self.get("decoding.flags", {}),
            # Viz
            "viz": {
                "montage": self.get("visualization.montage", "standard_1005"),
                "coef_agg": self.get("visualization.coef_agg", "abs"),
            },
        }


def load_config(config_path: Optional[Union[str, Path]] = None) -> EEGConfig:
    """Load EEG pipeline configuration.
    
    Args:
        config_path: Path to YAML config file. If None, uses default location.
        
    Returns:
        Loaded configuration object
    """
    return EEGConfig(config_path)


def get_legacy_constants(config: EEGConfig) -> Dict[str, Any]:
    """Extract legacy constants for backward compatibility.
    
    Args:
        config: Loaded configuration object
        
    Returns:
        Dictionary of constants that can be used to replace module-level variables
    """
    return {
        # Project paths
        "PROJECT_ROOT": config.project_root,
        "BIDS_ROOT": config.bids_root,
        "DERIV_ROOT": config.deriv_root,
        
        # Basic settings
        "SUBJECTS": config.subjects,
        "TASK": config.task,
        
        # Frequency bands
        # Prefer new per-script bands, fall back to legacy top-level if present
        "FEATURES_FREQ_BANDS": (config.get("time_frequency_analysis.bands") or config.frequency_bands),
        # Default now includes theta to ensure analyses and plots cover it
        "POWER_BANDS_TO_USE": config.get("power.bands_to_use", ["theta", "alpha", "beta", "gamma"]),
        
        # Event columns
        "PSYCH_TEMP_COLUMNS": config.get("event_columns.temperature", []),
        "RATING_COLUMNS": config.get("event_columns.rating", []),
        "PAIN_BINARY_COLUMNS": config.get("event_columns.pain_binary", []),
        
        # Analysis windows
        "PLATEAU_WINDOW": tuple(config.get("time_frequency_analysis.plateau_window", [3.0, 10.0])),
        "BASELINE": tuple(config.get("time_frequency_analysis.baseline_window", [-2.0, 0.0])),
        
        # Visualization
        "FIG_DPI": config.get("output.fig_dpi", 300),
        "SAVE_FORMATS": tuple(config.get("output.save_formats", ["png"])),
        "BAND_COLORS": config.get("visualization.band_colors", {}),
        
        # Advanced visualization parameters
        "TFR_SPECTRO_FONTSIZE": config.get("visualization.advanced.tfr_spectro.fontsize", 8),
        "TFR_SPECTRO_DPI": config.get("visualization.advanced.tfr_spectro.dpi", 1200),
        "TFR_SPECTRO_EXTENSION": config.get("visualization.advanced.tfr_spectro.extension", "svg"),
        "TFR_SPECTRO_BBOX_INCHES": config.get("visualization.advanced.tfr_spectro.bbox_inches", "tight"),
        "TFR_SPECTRO_TRANSPARENT": config.get("visualization.advanced.tfr_spectro.transparent", True),
        
        "TFR_TOPOMAP_FONTSIZE": config.get("visualization.advanced.tfr_topomap.fontsize", 10),
        "TFR_TOPOMAP_DPI": config.get("visualization.advanced.tfr_topomap.dpi", 800),
        "TFR_TOPOMAP_EXTENSION": config.get("visualization.advanced.tfr_topomap.extension", "png"),
        "TFR_TOPOMAP_BBOX_INCHES": config.get("visualization.advanced.tfr_topomap.bbox_inches", "tight"),
        "TFR_TOPOMAP_TRANSPARENT": config.get("visualization.advanced.tfr_topomap.transparent", True),
        
        # Random seed and reproducibility
        "RANDOM_STATE": config.get("random.seed", 42),
        
        # Statistics defaults
        "USE_SPEARMAN_DEFAULT": config.get("statistics.use_spearman_default", True),
        "PARTIAL_COVARS_DEFAULT": config.get("statistics.partial_covars_default"),
        "BOOTSTRAP_DEFAULT": config.get("random.bootstrap_default", 0),
        "N_PERM_DEFAULT": config.get("statistics.n_perm_default", 0),
        "DO_GROUP_DEFAULT": config.get("statistics.do_group_default", False),
        "GROUP_ONLY_DEFAULT": config.get("statistics.group_only_default", False),
        "BUILD_REPORTS_DEFAULT": config.get("statistics.build_reports_default", False),
        
        # Log file names  
        "LOG_FILE_NAME": config.get("logging.file_names.behavior_analysis", "04_behavior_feature_analysis.log"),
        
        # ERP analysis
        "ERP_PICKS": config.get("foundational_analysis.erp.picks", "eeg"),
        "PAIN_COLUMNS": config.get("event_columns.pain_binary", []),
        "TEMPERATURE_COLUMNS": config.get("event_columns.temperature", []),
        
        # Time-frequency analysis  
        "DEFAULT_TEMPERATURE_STRATEGY": config.get("time_frequency_analysis.temperature_strategy", "pooled"),
        "DEFAULT_PLATEAU_TMIN": config.get("time_frequency_analysis.plateau_window", [3.0,10.0])[0],
        "DEFAULT_PLATEAU_TMAX": config.get("time_frequency_analysis.plateau_window", [3.0,10.0])[1],
        
        # Raw-to-BIDS
        "DEFAULT_MONTAGE": config.get("raw_to_bids.default_montage", "easycap-M1"),
        "DEFAULT_LINE_FREQ": config.get("raw_to_bids.default_line_freq", 60.0),
        
        # TFR settings
        "CUSTOM_TFR_FREQS": np.arange(*config.get("tfr.custom_freqs", [1, 101, 1])),
        "CUSTOM_TFR_DECIM": config.get("time_frequency_analysis.tfr.decim", 4),
        
        # Feature extraction
        "PLATEAU_START": config.get("time_frequency_analysis.plateau_window", [3.0, 10.0])[0],
        "PLATEAU_END": config.get("time_frequency_analysis.plateau_window", [3.0, 10.0])[1],
        "TARGET_COLUMNS": config.get("event_columns.rating", []),
        "POWER_BANDS": config.get("power.bands_to_use", ["theta", "alpha", "beta", "gamma"]),
    }
