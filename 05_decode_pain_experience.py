from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

# Limit native BLAS thread pools to avoid oversubscription when using joblib/outer CV
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import matplotlib
# Use a non-interactive backend to avoid Tkinter/main loop issues in headless contexts
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import mne
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import r2_score, make_scorer, explained_variance_score, roc_auc_score
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr, spearmanr, probplot, ConstantInputWarning, kendalltau, rankdata, norm
import logging
import warnings
import threading
import scipy
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
import json
import hashlib
import random as pyrandom
import seaborn as sns
from joblib import Parallel, delayed

# -----------------------------------------------------------------------------
# Project config and helpers
# -----------------------------------------------------------------------------
try:
    from eeg_pipeline.config import (
        deriv_root as DERIV_ROOT,
        subjects as SUBJECTS,
        task as TASK,
        random_state as RANDOM_STATE,
    )
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DERIV_ROOT = PROJECT_ROOT / "eeg_pipeline" / "bids_output" / "derivatives"
    SUBJECTS = ["001"]
    TASK = "thermalactive"
    RANDOM_STATE = 42

# Reuse helpers from feature engineering if available; otherwise load via importlib
_HAVE_FE_HELPERS = False
try:
    # Attempt dynamic import by file path because the filename starts with digits
    import importlib.util as _ilus
    _fe_path = (Path(__file__).resolve().parent / "03_feature_engineering.py").resolve()
    if _fe_path.exists():
        _spec = _ilus.spec_from_file_location("_fe_helpers", str(_fe_path))
        if _spec and _spec.loader:  # type: ignore[attr-defined]
            _fe_mod = _ilus.module_from_spec(_spec)  # type: ignore
            _spec.loader.exec_module(_fe_mod)  # type: ignore[attr-defined]
            _find_clean_epochs_path = getattr(_fe_mod, "_find_clean_epochs_path")
            _load_events_df = getattr(_fe_mod, "_load_events_df")
            _align_events_to_epochs = getattr(_fe_mod, "_align_events_to_epochs")
            _pick_target_column = getattr(_fe_mod, "_pick_target_column")
            _HAVE_FE_HELPERS = True
except Exception:
    _HAVE_FE_HELPERS = False


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("decode_pain")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)

# Optional file handler will be attached per run
_FILE_LOG_HANDLER: Optional[logging.Handler] = None

# Reduce sklearn convergence log noise during extensive inner CV fits
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)


# Best-params logging behavior (can be overridden via CLI)
BEST_PARAMS_MODE: str = os.environ.get("BEST_PARAMS_MODE", "truncate")  # one of: append|truncate|run_scoped
RUN_ID: Optional[str] = None

# Track which best-params base paths have been logged to avoid repeated messages
_BEST_PARAMS_LOGGED: set = set()


# -----------------------------------------------------------------------------
# Top-level configuration (centralized parameters, grids, and output paths)
# -----------------------------------------------------------------------------
CONFIG = {
    "paths": {
        "results_subdir": "decoding",
        "plots_subdir": "plots",
        "indices": {
            "elasticnet_loso": "elasticnet_loso_indices.tsv",
            "elasticnet_within": "elasticnet_within_kfold_indices.tsv",
            "rf_loso": "rf_loso_indices.tsv",
            "rf_within": "rf_within_kfold_indices.tsv",
            "riemann_loso": "riemann_loso_indices.tsv",
            "riemann_band_template": "riemann_loso_indices_{label}.tsv",
            "temperature_only": "temperature_only_loso_indices.tsv",
            "baseline_global": "baseline_global_loso_indices.tsv",
            "diagnostic_subject_test_mean": "diagnostic_subject_test_mean_loso_indices.tsv",
        },
        "best_params": {
            "elasticnet_loso": "best_params_elasticnet.jsonl",
            "elasticnet_within": "best_params_elasticnet_within.jsonl",
            "rf_loso": "best_params_rf.jsonl",
            "rf_within": "best_params_rf_within.jsonl",
            "temperature_only": "best_params_temperature_only.jsonl",
            "riemann_loso": "best_params_riemann.jsonl",
            "riemann_band_template": "best_params_riemann_{label}.jsonl",
        },
        "predictions": {
            "elasticnet_loso": "elasticnet_loso_predictions.tsv",
            "elasticnet_within": "elasticnet_within_kfold_predictions.tsv",
            "rf_loso": "rf_loso_predictions.tsv",
            "rf_within": "rf_within_kfold_predictions.tsv",
            "baseline_global": "baseline_global_loso_predictions.tsv",
            "diagnostic_subject_test_mean": "diagnostic_subject_test_mean_loso_predictions.tsv",
            "temperature_only": "temperature_only_loso_predictions.tsv",
            "riemann_loso": "riemann_loso_predictions.tsv",
            "riemann_band_template": "riemann_loso_predictions_{label}.tsv",
        },
        "per_subject_metrics": {
            "elasticnet_loso": "elasticnet_per_subject_metrics.tsv",
            "elasticnet_within": "elasticnet_within_kfold_per_subject_metrics.tsv",
            "rf_loso": "rf_per_subject_metrics.tsv",
            "rf_within": "rf_within_kfold_per_subject_metrics.tsv",
            "baseline_global": "baseline_global_per_subject_metrics.tsv",
            "diagnostic_subject_test_mean": "diagnostic_subject_test_per_subject_metrics.tsv",
            "riemann_loso": "riemann_per_subject_metrics.tsv",
            "temperature_only": "temperature_only_per_subject_metrics.tsv",
        },
        "summaries": {
            "bootstrap": "summary_bootstrap.json",
            "incremental": "summary_incremental.json",
            "permutation_refit_null_rs": "permutation_refit_null_rs.txt",
            "permutation_refit_summary": "permutation_refit_summary.json",
            "summary": "summary.json",
            "all_metrics_wide": "all_metrics_wide.tsv",
            "riemann_bands": "summary_riemann_bands.json",
            "riemann_sliding_window": "riemann_sliding_window.json",
        },
    },
    "cv": {
        "inner_splits": 5,
    },
    "models": {
        "elasticnet": {
            "max_iter": 200000,
            "tol": 1e-3,
            "selection": "random",
            "grid": {
                "alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100],
                "l1_ratio": [0.2, 0.5, 0.8],
            },
        },
        "random_forest": {
            "n_estimators": 500,
            "estimator_n_jobs": -1,
            "bootstrap": True,
            "grid": {
                "max_depth": [None, 8, 16, 32],
                "max_features": ["sqrt", 0.2, 0.5, 1.0],
                "min_samples_leaf": [1, 5, 10],
            },
        },
        "temperature_only": {
            "ridge_alpha_grid": [0.0, 1e-3, 1e-2, 1e-1, 1, 10],
        },
    },
    "analysis": {
        "n_perm_quick": 1000,
        "n_perm_refit": 500,
        "rf_perm_importance_repeats": 20,
        "bootstrap_n": 1000,
        "riemann": {
            "plateau_window": (3.0, 10.5),
            "bands": [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)],
            "sliding_window": {"window_len": 0.75, "step": 0.25},
        },
    },
    "flags": {
        "run_within_subject_kfold": True,
        "run_riemann": True,
        "run_shap": True,
    },
    "viz": {
        "montage": "standard_1005",  # can be a standard montage name, "bids_auto", or "bids:<path-to-electrodes.tsv>"
        "coef_agg": "abs",           # one of: "abs" (mean |coef|) or "signed" (mean signed coef)
    },
}


# -----------------------------------------------------------------------------
# Pipeline Factory Functions
# -----------------------------------------------------------------------------

def _create_base_preprocessing_pipeline(include_scaling: bool = True) -> Pipeline:
    """Create standardized preprocessing pipeline to avoid code duplication.
    
    Args:
        include_scaling: Whether to include StandardScaler step
    
    Returns:
        Pipeline with imputation, variance threshold, and optionally scaling
    """
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
    ]
    if include_scaling:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)

def _create_elasticnet_pipeline(seed: int = 42) -> Pipeline:
    """Create ElasticNet pipeline with TransformedTargetRegressor."""
    base_steps = _create_base_preprocessing_pipeline(include_scaling=True).steps
    base_steps.append((
        "regressor", 
        TransformedTargetRegressor(
            regressor=ElasticNet(random_state=seed, max_iter=200000, tol=1e-3, selection="random"),
            transformer=PowerTransformer(method="yeo-johnson", standardize=True)
        )
    ))
    return Pipeline(base_steps)

def _create_rf_pipeline(n_estimators: int = 500, n_jobs: int = -1, seed: int = 42) -> Pipeline:
    """Create RandomForest pipeline (no scaling needed)."""
    base_steps = _create_base_preprocessing_pipeline(include_scaling=False).steps
    base_steps.append((
        "rf", 
        RandomForestRegressor(
            n_estimators=n_estimators, 
            n_jobs=n_jobs, 
            random_state=seed, 
            bootstrap=True
        )
    ))
    return Pipeline(base_steps)

def _create_logistic_pipeline(n_jobs: int = -1, seed: int = 42) -> Pipeline:
    """Create LogisticRegression pipeline with scaling."""
    base_steps = _create_base_preprocessing_pipeline(include_scaling=True).steps
    base_steps.append((
        "logreg", 
        LogisticRegression(
            solver="saga", 
            max_iter=10000, 
            n_jobs=n_jobs, 
            random_state=seed, 
            multi_class="multinomial"
        )
    ))
    return Pipeline(base_steps)

# Input Validation
# -----------------------------------------------------------------------------

def _validate_inputs(func):
    """Input validation decorator for critical functions."""
    def wrapper(*args, **kwargs):
        # Basic validation
        if 'subjects' in kwargs:
            subjects = kwargs['subjects']
            if subjects is not None and not isinstance(subjects, (list, tuple)):
                raise ValueError("subjects must be a list, tuple, or None")
            if subjects is not None and len(subjects) == 0:
                raise ValueError("subjects list cannot be empty")
        
        if 'n_jobs' in kwargs:
            n_jobs = kwargs['n_jobs']
            if not isinstance(n_jobs, int) or (n_jobs < -1 or n_jobs == 0):
                raise ValueError("n_jobs must be -1 or a positive integer")
        
        if 'seed' in kwargs:
            seed = kwargs['seed']
            if not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer")
        
        # Check for required paths
        if 'deriv_root' in kwargs:
            deriv_root = kwargs['deriv_root']
            if deriv_root is not None and not Path(deriv_root).exists():
                raise FileNotFoundError(f"deriv_root path does not exist: {deriv_root}")
                
        return func(*args, **kwargs)
    return wrapper

# Utilities
# -----------------------------------------------------------------------------

_handler_lock = threading.Lock()
_FILE_LOG_HANDLER: Optional[logging.FileHandler] = None

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _create_run_manifest(results_dir: Path, cli_args: dict, config: dict, run_id: Optional[str] = None) -> None:
    """Create a comprehensive run manifest with CLI args, config, and environment info."""
    import platform
    import os
    import subprocess
    
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "cli_args": cli_args,
        "resolved_config": config,
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        },
        "thread_limits": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
        }
    }
    
    # Try to get git commit hash if available
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], 
                                         stderr=subprocess.DEVNULL, 
                                         cwd=Path(__file__).parent,
                                         text=True).strip()
        manifest["git_commit"] = git_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        manifest["git_commit"] = None
    
    # Package versions
    import sklearn as _sk
    import mne as _mne
    manifest["package_versions"] = {"sklearn": _sk.__version__, "mne": _mne.__version__}
    try:
        import pyriemann as _pr
        manifest["package_versions"]["pyriemann"] = _pr.__version__
    except ImportError:
        manifest["package_versions"]["pyriemann"] = None
    
    _ensure_dir(results_dir)
    with open(results_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _setup_file_logging(results_dir: Path, run_id: Optional[str] = None) -> Path:
    """Attach a per-run FileHandler under results_dir/logs and return its path.

    If a previous file handler was attached in this process, it will be removed
    to avoid duplicate writes when main() is called multiple times.
    Thread-safe implementation.
    """
    global _FILE_LOG_HANDLER
    with _handler_lock:
        # Ensure logs directory exists
        log_dir = results_dir / "logs"
        _ensure_dir(log_dir)
        # Timestamped filename; optionally include run_id
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{run_id}" if run_id else ""
        log_path = log_dir / f"decode_pain_{ts}{suffix}.log"
        # Remove any existing file handler for a fresh run
        try:
            if _FILE_LOG_HANDLER is not None:
                logger.removeHandler(_FILE_LOG_HANDLER)
        except (AttributeError, ValueError):
            pass
        # Attach a new file handler
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        _FILE_LOG_HANDLER = fh
        return log_path


# Prepare a best-params JSONL path according to mode
def _prepare_best_params_path(base_path: Path, mode: str, run_id: Optional[str]) -> Path:
    try:
        _ensure_dir(base_path.parent)
        if mode == "run_scoped":
            rid = run_id or time.strftime("%Y%m%d_%H%M%S")
            out_path = base_path.with_name(f"{base_path.stem}_{rid}{base_path.suffix}")
        elif mode == "truncate":
            # create/truncate the file for this run
            with open(base_path, "w", encoding="utf-8"):
                pass
            out_path = base_path
        else:
            # append mode: use the base path as-is
            out_path = base_path

        # Log resolved path and mode once per base_path for traceability
        try:
            if base_path not in _BEST_PARAMS_LOGGED:
                logger.info(f"Best-params mode='{mode}'; resolved path: {out_path}")
                _BEST_PARAMS_LOGGED.add(base_path)
        except Exception:
            pass

        return out_path
    except Exception:
        return base_path


# -----------------------------------------------------------------------------
# Interpretability helpers (Elastic Net and Random Forest)
# -----------------------------------------------------------------------------

def _parse_pow_feature(feat: str) -> Optional[Tuple[str, str]]:
    """Return (band, channel) if feature is a power feature 'pow_<band>_<ch>'."""
    if not isinstance(feat, str):
        return None
    if not feat.startswith("pow_"):
        return None
    # Remove 'pow_' prefix and split from the right once to allow band names with underscores
    rest = feat[len("pow_"):]
    if "_" not in rest:
        return None
    try:
        band, ch = rest.rsplit("_", 1)
    except ValueError:
        return None
    if band == "" or ch == "":
        return None
    return band, ch


def _read_best_params_jsonl(path: Path) -> dict:
    """Return {fold: params_dict} from a JSONL file."""
    best = {}
    if path is None or (not Path(path).exists()):
        return best
    try:
        with open(path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                fold = rec.get("fold")
                params = rec.get("best_params", {})
                if fold is None:
                    # older records might not include fold; just append
                    continue
                best[int(fold)] = params
    except Exception:
        return best
    return best


def compute_enet_coefs_per_fold(X: pd.DataFrame, y: pd.Series, groups: np.ndarray, best_params_map: dict, seed: int) -> np.ndarray:
    """Fit ElasticNet per LOSO fold using logged best params; return coef matrix (n_folds, n_features).

    Pipeline: impute -> var -> scale -> regressor(TransformedTargetRegressor[ElasticNet + PowerTransformer])
    Note: Feature selection removed to prevent data leakage.
    """
    logo = LeaveOneGroupOut()
    coefs = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        params = best_params_map.get(fold, {})
        pipe = _create_elasticnet_pipeline(seed=seed)
        # Apply best parameters if available
        if isinstance(params, dict) and len(params) > 0:
            try:
                pipe.set_params(**params)
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Fold {fold}: failed to set params {params}: {e}")
        try:
            pipe.fit(X.iloc[train_idx, :], y.iloc[train_idx])
            # Coefficients from the final ElasticNet inside TTR
            en_final = pipe.named_steps["regressor"].regressor_
            coef = getattr(en_final, "coef_", None)
            if coef is None:
                raise RuntimeError("Final ElasticNet has no coef_ after fit.")
            coef = np.asarray(coef, dtype=float)

            # Only VarianceThreshold mask (no feature selection)
            var_mask = np.asarray(pipe.named_steps["var"].get_support(indices=False), dtype=bool)
            
            # Diagnostics: how many features kept after variance threshold
            total_after_var = int(var_mask.sum())
            pow_after_var = int(np.sum([str(X.columns[i]).startswith("pow_") for i in np.where(var_mask)[0]]))
            logger.info(
                f"Fold {fold}: Var-kept={total_after_var} (power features={pow_after_var})"
            )

            # Expand back to original feature space (zeros for removed features)
            full_coef = np.zeros(X.shape[1], dtype=float)
            if coef.shape[0] == int(var_mask.sum()):
                full_coef[var_mask] = coef
            else:
                logger.warning(
                    f"Fold {fold}: coef length ({coef.shape[0]}) != var-kept features ({int(var_mask.sum())}); setting to NaN"
                )
                full_coef[:] = np.nan
            coefs.append(full_coef)
        except (ValueError, RuntimeError, MemoryError) as e:
            logger.warning(f"ElasticNet fold {fold} failed to fit for coef extraction: {e}")
            coefs.append(np.full(X.shape[1], np.nan, dtype=float))
    return np.asarray(coefs)


def _find_bids_electrodes_tsv(bids_root: Path, subjects: Optional[List[str]] = None) -> Optional[Path]:
    """Locate a BIDS electrodes.tsv file under bids_root/sub-XXX/eeg/.

    If subjects is provided, search those first; otherwise scan all subjects.
    Returns the first matching path if found, else None.
    """
    try:
        # Prioritize provided subjects
        if subjects not in (None, [], ["all"]):
            for s in subjects:
                sub = f"sub-{s}"
                cand = list((bids_root / sub / "eeg").glob("*_electrodes.tsv"))
                if len(cand) > 0:
                    return cand[0]
        # Fallback: any subject
        for sub_dir in sorted((bids_root).glob("sub-*/eeg")):
            cand = list(sub_dir.glob("*_electrodes.tsv"))
            if len(cand) > 0:
                return cand[0]
    except Exception:
        return None
    return None


def make_montage_from_bids_electrodes(electrodes_tsv: Path) -> mne.channels.DigMontage:
    """Create an MNE DigMontage from a BIDS electrodes.tsv file.

    Expects columns including name (or label) and x, y, z coordinates. Coordinates are
    assumed to be in meters (BIDS spec); if values look like millimeters (>2 abs), they
    are converted to meters.
    """
    try:
        df = pd.read_csv(electrodes_tsv, sep="\t")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise RuntimeError(f"Failed to read electrodes TSV at {electrodes_tsv}: {e}")

    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("label") or cols.get("electrode")
    x_col = cols.get("x")
    y_col = cols.get("y")
    z_col = cols.get("z")
    if name_col is None or x_col is None or y_col is None or z_col is None:
        raise RuntimeError("electrodes.tsv must contain columns for name/label and x,y,z coordinates")

    df = df[[name_col, x_col, y_col, z_col]].dropna()
    df = df.rename(columns={name_col: "name", x_col: "x", y_col: "y", z_col: "z"})

    # Ensure numeric
    for c in ("x", "y", "z"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x", "y", "z"]).copy()

    # Convert to meters if looks like mm
    mx = float(np.nanmax(np.abs(df[["x", "y", "z"]].to_numpy()))) if len(df) else 0.0
    scale = 0.001 if mx > 2.0 else 1.0
    if scale != 1.0:
        logger.info("Electrode coordinates appear to be in mm; converting to meters.")
    df[["x", "y", "z"]] = df[["x", "y", "z"]] * scale

    ch_pos = {str(row["name"]): np.array([row["x"], row["y"], row["z"]], dtype=float) for _, row in df.iterrows()}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    return montage


def resolve_montage(montage_opt: Optional[str], deriv_root: Path, subjects: Optional[List[str]]) -> mne.channels.DigMontage:
    """Resolve montage selection.

    montage_opt can be:
    - a standard montage name (e.g., "standard_1020", "standard_1005")
    - "bids_auto" to attempt locating the first electrodes.tsv under bids_root
    - "bids:<path>" where <path> points to an electrodes.tsv
    """
    if montage_opt is None:
        montage_opt = "standard_1005"
    if montage_opt.startswith("bids:"):
        path_str = montage_opt.split(":", 1)[1]
        tsv_path = Path(path_str)
        return make_montage_from_bids_electrodes(tsv_path)
    if montage_opt == "bids_auto":
        bids_root = deriv_root.parent  # parent of derivatives should be bids root
        tsv_path = _find_bids_electrodes_tsv(bids_root, subjects)
        if tsv_path is None:
            logger.warning("bids_auto montage requested but no electrodes.tsv found; falling back to standard_1005")
            return mne.channels.make_standard_montage("standard_1005")
        return make_montage_from_bids_electrodes(tsv_path)
    # Otherwise assume a standard montage name
    try:
        return mne.channels.make_standard_montage(montage_opt)
    except (ValueError, FileNotFoundError, RuntimeError):
        logger.warning(f"Unknown montage '{montage_opt}'; using standard_1005")
        return mne.channels.make_standard_montage("standard_1005")


def plot_enet_band_topomaps(coef_matrix: np.ndarray, feature_names: List[str], plots_dir: Path,
                            montage: Optional[mne.channels.DigMontage] = None,
                            aggregate: str = "abs") -> None:
    """Aggregate coefficients across folds and plot per-band scalp maps.

    Only uses power features of the form 'pow_<band>_<ch>'. Saves one PNG per band.
    aggregate: 'abs' -> mean |coef| across folds; 'signed' -> mean signed coef across folds.
    """
    if coef_matrix.size == 0:
        return
    # Infer available bands from feature names to avoid hard-coding
    bands_set = set()
    for feat in feature_names:
        parsed = _parse_pow_feature(feat)
        if parsed is not None:
            b, _ = parsed
            bands_set.add(b)
    bands = sorted(list(bands_set))
    # Map features to (band, channel)
    band_ch_to_idx = {}
    for idx, feat in enumerate(feature_names):
        parsed = _parse_pow_feature(feat)
        if parsed is None:
            continue
        b, ch = parsed
        if b not in bands:
            continue
        band_ch_to_idx.setdefault(b, {}).setdefault(ch, []).append(idx)

    # Aggregate coefficients across folds
    if aggregate == "signed":
        coef_agg = np.nanmean(coef_matrix, axis=0)
    else:
        coef_agg = np.nanmean(np.abs(coef_matrix), axis=0)

    montage = montage or mne.channels.make_standard_montage("standard_1005")
    for b in bands:
        ch_names = []
        values = []
        ch_map = band_ch_to_idx.get(b, {})
        for ch, idxs in ch_map.items():
            if ch in montage.ch_names:
                ch_names.append(ch)
                values.append(float(np.nanmean(coef_agg[idxs])))
        if not ch_names:
            logger.warning(f"No channels found for band {b} to plot topomap.")
            continue
        info = mne.create_info(ch_names, sfreq=1000.0, ch_types="eeg")
        info.set_montage(montage)
        fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.0), dpi=150)
        try:
            vals = np.asarray(values)
            # Skip plotting if coefficients are all-zero or not finite
            if (not np.isfinite(vals).any()) or np.allclose(np.nan_to_num(vals, nan=0.0), 0.0, atol=1e-12):
                logger.warning(f"Skipping band {b} topomap: all-zero or NaN coefficients.")
                continue
            if aggregate == "signed":
                vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else None
                mne.viz.plot_topomap(vals, info, axes=ax, show=False, contours=6, cmap="RdBu_r", vlim=(-vmax, vmax) if vmax else None)
                ax.set_title(f"ElasticNet | {b}: mean signed coef")
            else:
                mne.viz.plot_topomap(vals, info, axes=ax, show=False, contours=6, cmap="Reds")
                ax.set_title(f"ElasticNet | {b}: mean |coef|")
            plt.tight_layout()
            suffix = "signed" if aggregate == "signed" else "abs"
            plt.savefig(plots_dir / f"elasticnet_coef_topomap_{b}_{suffix}.png")
        finally:
            plt.close(fig)

def plot_permutation_refit_null_hist(null_rs: np.ndarray, obs_r: float, save_path: Path) -> None:
    """Plot histogram of refit-based permutation null (null_rs) with observed r line."""
    plt.figure(figsize=(6, 4), dpi=150)
    plt.hist(null_rs, bins=30, alpha=0.7, color="#bdb2ff", edgecolor="white")
    plt.axvline(obs_r, color="red", linestyle="--", label=f"observed r={obs_r:.3f}")
    plt.xlabel("Null pooled r (refit, within-subject shuffles)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_within_vs_loso_combined(
    per_left: pd.DataFrame,
    per_right: pd.DataFrame,
    metric: str,
    save_path: Path,
    label_left: str = "LOSO",
    label_right: str = "WithinKFold",
) -> None:
    """Combined figure: paired scatter and per-subject dumbbell for a metric.

    Expects both DataFrames to have columns: 'group' and the specified metric.
    """
    df_l = per_left[["group", metric]].rename(columns={metric: f"{label_left}_{metric}"})
    df_r = per_right[["group", metric]].rename(columns={metric: f"{label_right}_{metric}"})
    d = df_l.merge(df_r, on="group", how="inner").dropna()
    if d.empty:
        logger.warning(f"Combined within-vs-LOSO skipped: no overlapping subjects for metric={metric}")
        return
    x = d[f"{label_left}_{metric}"].to_numpy()
    y = d[f"{label_right}_{metric}"].to_numpy()
    # Order subjects by LOSO metric for dumbbell plot
    order = np.argsort(x)
    d_ord = d.iloc[order].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, gridspec_kw={"wspace": 0.3})

    # Left: paired scatter
    ax0 = axes[0]
    ax0.scatter(x, y, alpha=0.7, edgecolors="none", color="#3a86ff")
    lims = [float(np.nanmin([x.min(), y.min()])), float(np.nanmax([x.max(), y.max()]))]
    if np.isfinite(lims).all():
        ax0.plot(lims, lims, linestyle="--", color="gray", label="y=x")
        ax0.set_xlim(lims)
        ax0.set_ylim(lims)
    ax0.set_xlabel(f"{label_left} {metric}")
    ax0.set_ylabel(f"{label_right} {metric}")
    ax0.set_title(f"Paired: {metric}")
    ax0.grid(True, alpha=0.3)

    # Right: dumbbell (per subject)
    ax1 = axes[1]
    y_pos = np.arange(len(d_ord))
    x_left = d_ord[f"{label_left}_{metric}"].to_numpy()
    x_right = d_ord[f"{label_right}_{metric}"].to_numpy()
    ax1.hlines(y=y_pos, xmin=x_left, xmax=x_right, color="#b0b0b0", alpha=0.8)
    ax1.scatter(x_left, y_pos, color="#2b9348", label=label_left)
    ax1.scatter(x_right, y_pos, color="#f94144", label=label_right)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(d_ord["group"].astype(str).tolist())
    ax1.invert_yaxis()
    ax1.set_xlabel(metric)
    ax1.set_title("Per-subject dumbbell")
    ax1.legend(loc="lower right")
    ax1.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_rf_permutation_importance_per_fold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    best_params_map: dict,
    seed: int,
    n_repeats: Optional[int] = None,
) -> np.ndarray:
    """Compute permutation importance on each LOSO test fold; return matrix (n_folds, n_features)."""
    logo = LeaveOneGroupOut()
    imps = []
    # Allow CONFIG to control repeats by default
    if n_repeats is None:
        try:
            n_repeats = int(CONFIG["analysis"]["rf_perm_importance_repeats"])  # type: ignore[index]
        except (KeyError, ValueError, TypeError):
            n_repeats = 20
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        rf_params = best_params_map.get(fold, {})
        rf = RandomForestRegressor(
            n_estimators=rf_params.get("rf__n_estimators", 500),
            max_depth=rf_params.get("rf__max_depth", None),
            max_features=rf_params.get("rf__max_features", "sqrt"),
            min_samples_leaf=rf_params.get("rf__min_samples_leaf", 1),
            random_state=seed,
            n_jobs=1,
            bootstrap=True,
        )
        pipe = _create_base_preprocessing_pipeline(include_scaling=False)
        pipe.steps.append(("rf", rf))
        try:
            pipe.fit(X.iloc[train_idx, :], y.iloc[train_idx])
            res = permutation_importance(
                pipe,
                X.iloc[test_idx, :],
                y.iloc[test_idx],
                n_repeats=n_repeats,
                random_state=seed,
                n_jobs=1,
                scoring="r2",
            )
            imps.append(res.importances_mean)
        except (ValueError, MemoryError, RuntimeError) as e:
            logger.warning(f"RF permutation importance failed on fold {fold}: {e}")
            imps.append(np.full(X.shape[1], np.nan, dtype=float))
    return np.asarray(imps)


def compute_rf_block_permutation_importance_per_fold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    best_params_map: dict,
    seed: int,
    n_repeats: Optional[int] = None,
) -> np.ndarray:
    """Block-aware permutation importance on each LOSO test fold; return (n_folds, n_features).

    Within each test fold, permute each feature *within subject* blocks to preserve per-subject structure.
    Importance is mean ΔR² over repeats without refitting the model.
    """
    logo = LeaveOneGroupOut()
    imps: List[np.ndarray] = []
    # Allow CONFIG to control repeats by default
    if n_repeats is None:
        try:
            n_repeats = int(CONFIG["analysis"]["rf_perm_importance_repeats"])  # type: ignore[index]
        except (KeyError, ValueError, TypeError):
            n_repeats = 20
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        rf_params = best_params_map.get(fold, {})
        rf = RandomForestRegressor(
            n_estimators=rf_params.get("rf__n_estimators", 500),
            max_depth=rf_params.get("rf__max_depth", None),
            max_features=rf_params.get("rf__max_features", "sqrt"),
            min_samples_leaf=rf_params.get("rf__min_samples_leaf", 1),
            random_state=seed,
            n_jobs=1,
            bootstrap=True,
        )
        pipe = _create_base_preprocessing_pipeline(include_scaling=False)
        pipe.steps.append(("rf", rf))
        try:
            X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            g_test = np.asarray(groups)[test_idx]
            pipe.fit(X_train, y_train)
            y_pred_base = pipe.predict(X_test)
            base_r2 = r2_score(y_test.to_numpy(), np.asarray(y_pred_base))
            rng = np.random.default_rng(seed + fold)
            n_features = X.shape[1]
            deltas_mean = np.zeros(n_features, dtype=float)
            skipped_feats: List[str] = []
            for j in range(n_features):
                # Fast skip: if feature j is constant within all subject blocks in the test fold,
                # permuting within blocks is a no-op. Mark importance as 0 and continue.
                all_const = True
                for s in np.unique(g_test):
                    idxs = np.where(g_test == s)[0]
                    if len(idxs) > 1:
                        vals = X_test.iloc[idxs, j].to_numpy()
                        try:
                            nunq = pd.Series(vals).nunique(dropna=False)
                        except (ValueError, MemoryError):
                            nunq = 2  # treat as non-constant on error
                        if int(nunq) > 1:
                            all_const = False
                            break
                if all_const:
                    deltas_mean[j] = 0.0
                    try:
                        skipped_feats.append(str(X.columns[j]))
                    except Exception:
                        pass
                    continue
                deltas = np.zeros(int(n_repeats), dtype=float)
                # Convert to numpy for efficient operations
                X_test_np = X_test.to_numpy()
                y_test_np = y_test.to_numpy()
                for r in range(int(n_repeats)):
                    # Work directly with numpy arrays to avoid DataFrame copying
                    X_test_perm = X_test_np.copy()
                    # permute feature j within each subject block in the test fold
                    for s in np.unique(g_test):
                        idxs = np.where(g_test == s)[0]
                        if len(idxs) > 1:
                            X_test_perm[idxs, j] = rng.permutation(X_test_perm[idxs, j])
                    # Convert back to DataFrame only for prediction
                    X_perm_df = pd.DataFrame(X_test_perm, columns=X_test.columns, index=X_test.index)
                    y_pred_perm = pipe.predict(X_perm_df)
                    r2_perm = r2_score(y_test_np, np.asarray(y_pred_perm))
                    deltas[r] = base_r2 - r2_perm
                    deltas[r] = deltas[r] if np.isfinite(deltas[r]) else 0.0
                    # Explicit cleanup
                    del X_perm_df, y_pred_perm
                deltas_mean[j] = float(np.mean(deltas))
                # Cleanup deltas array
                del deltas
            # Summarize skipped constant-within-block features for this fold (limit preview)
            if len(skipped_feats) > 0:
                try:
                    prev = skipped_feats[:10]
                    logger.info(f"RF block perm fold {fold}: skipped {len(skipped_feats)}/{n_features} constant-within-block features (showing up to 10): {prev}")
                except Exception:
                    logger.info(f"RF block perm fold {fold}: skipped {len(skipped_feats)}/{n_features} constant-within-block features")
            imps.append(deltas_mean)
        except (ValueError, MemoryError, RuntimeError) as e:
            logger.warning(f"RF block permutation importance failed on fold {fold}: {e}")
            imps.append(np.full(X.shape[1], np.nan, dtype=float))
    return np.asarray(imps)

def plot_rf_perm_importance_bar(mean_imps: np.ndarray, feature_names: List[str], save_path: Path, top_n: int = 20) -> None:
    order = np.argsort(mean_imps)[::-1]
    order = order[:min(top_n, len(order))]
    feats = [feature_names[i] for i in order]
    vals = [float(mean_imps[i]) for i in order]
    plt.figure(figsize=(8, 5), dpi=150)
    sns.barplot(x=vals, y=feats, orient="h", color="#457b9d")
    plt.xlabel("Permutation importance (mean ΔR²)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def subject_id_decodability_auc_plot(X: pd.DataFrame, groups: np.ndarray, save_path: Path, 
                                    results_dir: Optional[Path] = None, seed: int = 42) -> Optional[float]:
    """Sanity check: classify subject ID from features using StratifiedKFold and LogisticRegression.

    Returns pooled macro AUC if successful and saves a simple bar plot to `save_path`.
    Also stores AUC table and per-subject AUCs/confusion matrices if results_dir provided.
    Requires >=2 subjects and >=2 trials per subject to run.
    """
    subjects = pd.Series(groups)
    classes = subjects.unique()
    counts = subjects.value_counts()
    min_n = int(counts.min()) if len(counts) > 0 else 0
    if min_n < 2 or len(classes) < 2:
        logger.warning("Subject-ID decodability check skipped: need >=2 classes and >=2 trials per subject.")
        return None
    n_splits = min(5, max(2, min_n))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lr = LogisticRegression(random_state=seed, max_iter=500, multi_class="ovr")
    proba_list, y_list, pred_list = [], [], []
    all_predictions = []
    
    # Store per-fold results for detailed analysis
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = groups[train_idx], groups[test_idx]
        try:
            lr.fit(X_train, y_train)
            proba = lr.predict_proba(X_test)
            y_pred = lr.predict(X_test)
            
            proba_list.append(proba)
            y_list.append(y_test)
            pred_list.append(y_pred)
            
            # Store fold-level predictions for analysis
            for i, (true_subj, pred_subj, prob_vec) in enumerate(zip(y_test, y_pred, proba)):
                all_predictions.append({
                    "fold": fold_idx,
                    "true_subject": str(true_subj),
                    "pred_subject": str(pred_subj),
                    "correct": true_subj == pred_subj,
                    **{f"prob_{cls}": prob_vec[j] for j, cls in enumerate(lr.classes_)}
                })
                
        except Exception as e:
            logger.warning(f"Subject-ID CV fold {fold_idx} failed: {e}")

    if len(proba_list) == 0:
        logger.warning("Subject-ID decodability: no predictions produced.")
        return None
    Y = np.concatenate(y_list, axis=0)
    P = np.concatenate(proba_list, axis=0)
    Y_pred = np.concatenate(pred_list, axis=0)
    
    try:
        auc_macro = roc_auc_score(Y, P, multi_class="ovr", average="macro")
    except Exception:
        logger.warning("AUC computation failed for subject-ID decodability.")
        return None

    # Compute per-subject AUCs and confusion matrix
    if results_dir is not None:
        try:
            # Per-subject AUCs (one-vs-rest)
            per_subject_aucs = []
            unique_subjects = np.unique(Y)
            
            for subj in unique_subjects:
                # Binary classification: current subject vs all others
                y_binary = (Y == subj).astype(int)
                if len(np.unique(y_binary)) == 2:  # Need both classes present
                    # Get probability for current subject
                    subj_idx = np.where(lr.classes_ == subj)[0]
                    if len(subj_idx) > 0:
                        p_subj = P[:, subj_idx[0]]
                        try:
                            auc_subj = roc_auc_score(y_binary, p_subj)
                            per_subject_aucs.append({
                                "subject_id": str(subj),
                                "auc": float(auc_subj),
                                "n_trials": int(np.sum(Y == subj))
                            })
                        except Exception as e:
                            logger.warning(f"Failed to compute AUC for subject {subj}: {e}")
            
            # Save per-subject AUCs
            if per_subject_aucs:
                per_subj_df = pd.DataFrame(per_subject_aucs)
                _ensure_dir(results_dir)
                per_subj_df.to_csv(results_dir / "subject_id_per_subject_aucs.tsv", sep="\t", index=False)
                logger.info(f"Saved per-subject ID decodability AUCs: {len(per_subject_aucs)} subjects")
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(Y, Y_pred, labels=unique_subjects)
            cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in unique_subjects], 
                               columns=[f"pred_{s}" for s in unique_subjects])
            cm_df.to_csv(results_dir / "subject_id_confusion_matrix.tsv", sep="\t")
            
            # Overall AUC table
            auc_table = {
                "macro_auc": float(auc_macro),
                "n_subjects": len(unique_subjects),
                "n_trials_total": len(Y),
                "n_cv_splits": n_splits,
                "accuracy": float(np.mean(Y == Y_pred)),
                "per_subject_auc_mean": float(np.mean([s["auc"] for s in per_subject_aucs])) if per_subject_aucs else np.nan,
                "per_subject_auc_std": float(np.std([s["auc"] for s in per_subject_aucs])) if per_subject_aucs else np.nan
            }
            
            with open(results_dir / "subject_id_auc_table.json", "w", encoding="utf-8") as f:
                json.dump(auc_table, f, indent=2)
            
            # Detailed predictions
            if all_predictions:
                pred_df = pd.DataFrame(all_predictions)
                pred_df.to_csv(results_dir / "subject_id_detailed_predictions.tsv", sep="\t", index=False)
                
            logger.info(f"Subject-ID decodability: macro AUC={auc_macro:.3f}, accuracy={auc_table['accuracy']:.3f}")
            
        except Exception as e:
            logger.warning(f"Failed to save detailed subject-ID decodability results: {e}")

    # Bar plot
    fig, ax = plt.subplots(1, 1, figsize=(3, 4), dpi=150)
    ax.bar([0], [auc_macro], color="#457b9d")
    plt.ylim(0.0, 1.0)
    plt.xticks([0], ["Subject-ID"])
    plt.ylabel("Macro AUC (OvR)")
    plt.title(f"Subject-ID decodability: AUC={auc_macro:.3f}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return float(auc_macro)


def _bootstrap_pooled_metrics_by_subject(pred_df: pd.DataFrame, n_boot: Optional[int] = None, seed: int = 42) -> dict:
    """Bootstrap across subjects for pooled Pearson r and R^2.

    pred_df must contain columns: 'subject_id', 'y_true', 'y_pred'.
    """
    rng = np.random.default_rng(seed)
    # Allow CONFIG to control bootstrap iterations by default
    if n_boot is None:
        try:
            n_boot = int(CONFIG["analysis"]["bootstrap_n"])  # type: ignore[index]
        except (KeyError, ValueError, TypeError):
            n_boot = 1000
    # Point estimates
    r_point, _ = _safe_pearsonr(pred_df["y_true"].values, pred_df["y_pred"].values)
    r2_point = r2_score(pred_df["y_true"].values, pred_df["y_pred"].values)
    subs = pred_df["subject_id"].astype(str).unique()
    if len(subs) < 2:
        return {"r_point": float(r_point), "r2_point": float(r2_point), "r_ci": [np.nan, np.nan], "r2_ci": [np.nan, np.nan], "n_bootstrap": 0}
    r_vals = []
    r2_vals = []
    for _ in range(n_boot):
        boot_subs = rng.choice(subs, size=len(subs), replace=True)
        # Build bootstrap sample by concatenating sampled subject blocks (with multiplicity)
        boot_blocks = [pred_df[pred_df["subject_id"].astype(str) == s] for s in boot_subs]
        boot_df = pd.concat(boot_blocks, axis=0, ignore_index=True)
        y_t = boot_df["y_true"].values
        y_p = boot_df["y_pred"].values
        r, _ = _safe_pearsonr(y_t, y_p)
        r2 = r2_score(y_t, y_p) if len(y_t) > 1 else np.nan
        r_vals.append(float(r))
        r2_vals.append(float(r2))
    # Guard against all-NaN inputs which would raise warnings in nanpercentile
    _r_finite = np.asarray([v for v in r_vals if np.isfinite(v)], dtype=float)
    _r2_finite = np.asarray([v for v in r2_vals if np.isfinite(v)], dtype=float)

    # Percentile CIs as baseline
    if _r_finite.size > 0:
        r_ci_pct = [float(np.percentile(_r_finite, 2.5)), float(np.percentile(_r_finite, 97.5))]
    else:
        r_ci_pct = [np.nan, np.nan]
    if _r2_finite.size > 0:
        r2_ci_pct = [float(np.percentile(_r2_finite, 2.5)), float(np.percentile(_r2_finite, 97.5))]
    else:
        r2_ci_pct = [np.nan, np.nan]

    # BCa helper
    def _bca_ci(thetas: np.ndarray, theta0: float, jk_vals: np.ndarray, alpha_low=0.025, alpha_high=0.975) -> Tuple[float, float]:
        thetas = np.asarray(thetas, float)
        thetas = thetas[np.isfinite(thetas)]
        jk_vals = np.asarray(jk_vals, float)
        jk_vals = jk_vals[np.isfinite(jk_vals)]
        if thetas.size < 10 or jk_vals.size < 3 or not np.isfinite(theta0):
            return float("nan"), float("nan")
        # acceleration via jackknife
        tdot = float(np.mean(jk_vals))
        num = np.sum((tdot - jk_vals) ** 3)
        den = 6.0 * (np.sum((tdot - jk_vals) ** 2) ** 1.5 + 1e-12)
        a = float(num / den) if np.isfinite(num) and np.isfinite(den) and den != 0 else 0.0
        # bias-correction using bootstrap distribution
        z0 = float(norm.ppf((np.sum(thetas < theta0) + 1e-12) / (len(thetas) + 2e-12)))
        def _adj(alpha: float) -> float:
            zalpha = float(norm.ppf(alpha))
            adj = z0 + (z0 + zalpha) / max(1e-12, (1 - a * (z0 + zalpha)))
            return float(norm.cdf(adj))
        a1 = _adj(alpha_low)
        a2 = _adj(alpha_high)
        q_low = 100 * np.clip(a1, 0.0, 1.0)
        q_high = 100 * np.clip(a2, 0.0, 1.0)
        lo, hi = np.percentile(thetas, [q_low, q_high]).tolist()
        return float(lo), float(hi)

    # Jackknife across subjects for r and r2
    r_jk, r2_jk = [], []
    for s in subs:
        d_jk = pred_df[pred_df["subject_id"].astype(str) != str(s)]
        if len(d_jk) < 2:
            continue
        rj, _ = _safe_pearsonr(d_jk["y_true"].values, d_jk["y_pred"].values)
        r2j = r2_score(d_jk["y_true"].values, d_jk["y_pred"].values) if len(d_jk) > 1 else np.nan
        r_jk.append(float(rj))
        r2_jk.append(float(r2j))
    r_jk = np.asarray(r_jk, float)
    r2_jk = np.asarray(r2_jk, float)

    # Compute BCa CIs; fallback to percentile if BCa invalid
    r_ci_bca = _bca_ci(_r_finite, float(r_point), r_jk)
    r2_ci_bca = _bca_ci(_r2_finite, float(r2_point), r2_jk)
    r_ci = [float(r_ci_bca[0]) if np.isfinite(r_ci_bca[0]) else r_ci_pct[0],
            float(r_ci_bca[1]) if np.isfinite(r_ci_bca[1]) else r_ci_pct[1]]
    r2_ci = [float(r2_ci_bca[0]) if np.isfinite(r2_ci_bca[0]) else r2_ci_pct[0],
             float(r2_ci_bca[1]) if np.isfinite(r2_ci_bca[1]) else r2_ci_pct[1]]
    return {
        "r_point": float(r_point),
        "r2_point": float(r2_point),
        "r_ci": r_ci,
        "r2_ci": r2_ci,
        "n_bootstrap": int(n_boot),
        "n_subjects": int(len(subs)),
    }


def _per_subject_pearson_and_spearman(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject Pearson r and Spearman rho from trial-level predictions."""
    rows = []
    for sid, d in pred_df.groupby("subject_id"):
        if len(d) < 2:
            continue
        r, _ = _safe_pearsonr(d["y_true"].values, d["y_pred"].values)
        try:
            rho, _ = spearmanr(d["y_true"].values, d["y_pred"].values)
        except (ValueError, RuntimeError):
            rho = np.nan
        rows.append({"subject_id": sid, "pearson_r": float(r), "spearman_rho": float(rho)})
    return pd.DataFrame(rows)


def plot_spearman_vs_pearson(models_stats: List[Tuple[pd.DataFrame, str, str]], save_path: Path) -> None:
    """Scatter pearson vs spearman per subject for one or more models.

    models_stats: list of (df, label, color), where df has columns 'pearson_r' and 'spearman_rho'.
    """
    plt.figure(figsize=(5, 5), dpi=150)
    for df, label, color in models_stats:
        if df.empty:
            continue
        plt.scatter(df["pearson_r"], df["spearman_rho"], alpha=0.7, label=label, color=color, edgecolors="none")
    lims = [-1.0, 1.0]
    plt.plot(lims, lims, linestyle="--", color="gray", label="y=x")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Pearson r")
    plt.ylabel("Spearman rho")
    plt.title("Rank robustness: Spearman vs Pearson")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def build_all_metrics_wide(results_dir: Path, save_path: Path) -> None:
    """Aggregate per-subject metrics from all available TSVs into a wide table."""
    files = list(results_dir.glob("*per_subject_metrics.tsv"))
    if len(files) == 0:
        logger.warning("No per-subject metrics found to build wide TSV.")
        return
    merged = None
    for fp in files:
        name = fp.name.replace("_per_subject_metrics.tsv", "")
        # Skip diagnostic subject-test baseline if any
        if "subject_test" in name:
            continue
        try:
            df = pd.read_csv(fp, sep="\t")
        except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError) as e:
            logger.warning(f"Failed reading {fp}: {e}")
            continue
        subj_col = "group" if "group" in df.columns else ("subject_id" if "subject_id" in df.columns else None)
        if subj_col is None:
            continue
        df = df.rename(columns={subj_col: "subject_id"})
        # Keep standard metric columns if present
        keep = [c for c in ["pearson_r", "r2", "explained_variance", "n_trials"] if c in df.columns]
        df = df[["subject_id"] + keep].copy()
        # Suffix columns with model name
        suffix = name
        df = df.rename(columns={c: f"{c}__{suffix}" for c in keep})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="subject_id", how="outer")
    if merged is not None:
        merged.sort_values("subject_id").to_csv(save_path, sep="\t", index=False)

def plot_kendall_tau_heatmap(ranks: np.ndarray, save_path: Path) -> None:
    """Plot fold-by-fold Kendall's tau matrix from rank matrix (n_folds, n_features)."""
    n_folds = ranks.shape[0]
    K = np.zeros((n_folds, n_folds), dtype=float)
    for i in range(n_folds):
        for j in range(n_folds):
            if i == j:
                K[i, j] = 1.0
            else:
                tau, _ = kendalltau(ranks[i], ranks[j])
                K[i, j] = float(tau) if np.isfinite(tau) else np.nan
    plt.figure(figsize=(5.5, 4.5), dpi=150)
    sns.heatmap(K, vmin=-1, vmax=1, cmap="coolwarm", annot=False, square=True, cbar_kws={"label": "Kendall τ"})
    plt.title("Feature ranking stability across folds (Kendall τ)")
    plt.xlabel("Fold")
    plt.ylabel("Fold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_topk_rank_heatmap(ranks: np.ndarray, feature_names: List[str], save_path: Path, top_k: int = 30) -> None:
    """Plot heatmap of per-fold ranks for top-K features by mean importance (lower rank is better)."""
    mean_rank = np.nanmean(ranks, axis=0)
    order = np.argsort(mean_rank)[:min(top_k, len(mean_rank))]
    feats = [feature_names[i] for i in order]
    data = ranks[:, order]
    plt.figure(figsize=(10, 6), dpi=150)
    sns.heatmap(data.T, cmap="viridis", cbar_kws={"label": "Rank (1=best)"}, yticklabels=feats)
    plt.xlabel("Fold")
    plt.ylabel("Feature")
    plt.title("Top-K feature ranks per fold (RF permutation importance)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _aggregate_best_rf_params(best_params_map: dict) -> dict:
    """Aggregate best RF params across folds by taking the most frequent value per hyperparameter."""
    if not best_params_map:
        return {}
    from collections import Counter
    keys = set()
    for d in best_params_map.values():
        keys.update([k for k in d.keys() if k.startswith("rf__")])
    agg = {}
    for k in keys:
        vals = [d.get(k) for d in best_params_map.values() if k in d]
        if not vals:
            continue
        try:
            agg[k] = Counter(vals).most_common(1)[0][0]
        except Exception:
            agg[k] = vals[0]
    return agg


def run_shap_rf_global(X: pd.DataFrame, y: pd.Series, feature_names: List[str], best_params_map: dict, plots_dir: Path, seed: int) -> None:
    """Optional: Fit a global RF and run SHAP.

    - Aggregates RF params across folds
    - Optionally subsamples trials before SHAP (CONFIG['analysis']['shap'])
    - Saves SHAP values to disk (.npz) and plots summary + dependence
    
    METHODOLOGICAL NOTE: This function trains a single "global" RandomForest model on ALL data
    to generate SHAP values for interpretability. This global model is different from the ensemble
    of LOSO models whose performance was actually evaluated. The feature importance profile and
    SHAP values from this global model may not perfectly represent the drivers of prediction for
    any specific left-out subject. For more rigorous interpretation, SHAP values should ideally
    be calculated within each LOSO test fold and then aggregated (computationally expensive).
    """
    try:
        import shap  # type: ignore
    except Exception:
        logger.info("SHAP not installed; skipping SHAP analyses.")
        return

    rf_params = _aggregate_best_rf_params(best_params_map)
    rf = RandomForestRegressor(
        n_estimators=rf_params.get("rf__n_estimators", 500),
        max_depth=rf_params.get("rf__max_depth", None),
        max_features=rf_params.get("rf__max_features", "sqrt"),
        min_samples_leaf=rf_params.get("rf__min_samples_leaf", 1),
        random_state=seed,
        n_jobs=-1,
        bootstrap=True,
    )
    logger.info("Training global RandomForest model for SHAP interpretation...")
    logger.warning("METHODOLOGICAL CAVEAT: SHAP values derived from global model (all data) "
                   "may not perfectly reflect behavior of individual LOSO cross-validation models.")
    
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    rf.fit(X_imp, y.values)

    # Optional subsampling configuration
    shap_cfg = {}
    try:
        shap_cfg = CONFIG.get("analysis", {}).get("shap", {}) if isinstance(CONFIG.get("analysis"), dict) else {}
    except Exception:
        shap_cfg = {}
    subsample_n = None
    subsample_frac = None
    try:
        subsample_n = int(shap_cfg.get("subsample_n")) if shap_cfg.get("subsample_n") is not None else None
    except Exception:
        subsample_n = None
    try:
        subsample_frac = float(shap_cfg.get("subsample_frac")) if shap_cfg.get("subsample_frac") is not None else None
    except Exception:
        subsample_frac = None

    n_total = X_imp.shape[0]
    idx_use = np.arange(n_total)
    if subsample_n is not None or (subsample_frac is not None and subsample_frac > 0 and subsample_frac < 1):
        rng = np.random.default_rng(seed + 777)
        if subsample_n is None and subsample_frac is not None:
            subsample_n = max(1, int(round(subsample_frac * n_total)))
        subsample_n = min(n_total, max(1, int(subsample_n or n_total)))
        idx_use = np.sort(rng.choice(n_total, size=subsample_n, replace=False))

    # Select data to explain
    X_explain = X_imp[idx_use, :]
    y_explain = y.values[idx_use]

    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_explain)
        # Summary beeswarm
        plt.figure(figsize=(8, 6), dpi=150)
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False, plot_type="dot")
        plt.tight_layout()
        plt.savefig(plots_dir / "rf_shap_summary.png")
        plt.close()
        # Dependence for top features
        importances = getattr(rf, "feature_importances_", None)
        if importances is None:
            return
        # Full order for alignment downstream; and top-k for dependence plots
        order_full = np.argsort(importances)[::-1]
        order = order_full[: min(5, len(importances))]
        for idx in order:
            plt.figure(figsize=(6, 4), dpi=150)
            shap.dependence_plot(idx, shap_values, X_explain, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(plots_dir / f"rf_shap_dependence_{feature_names[idx].replace('/', '_')}.png")
            plt.close()

        # Save SHAP values to disk for reuse
        try:
            # Handle regression (np.ndarray) vs classification (list of arrays)
            if isinstance(shap_values, list):
                # Not expected for regression, but handle generically: stack first element
                shap_arr = np.asarray(shap_values[0]) if len(shap_values) > 0 else np.empty((len(idx_use), X_imp.shape[1]))
            else:
                shap_arr = np.asarray(shap_values)
            fname = "rf_shap_values" + (f"_subsample{subsample_n}.npz" if len(idx_use) < n_total else ".npz")
            np.savez_compressed(
                plots_dir / fname,
                shap_values=shap_arr,
                X=X_explain,
                y=y_explain,
                feature_names=np.asarray(feature_names),
                sample_indices=idx_use,
                rf_importances=np.asarray(importances),
                rf_importance_order_full=order_full,
                rf_importance_order_full_names=np.asarray([feature_names[i] for i in order_full]),
                dependence_order_topk=order,
                dependence_order_topk_names=np.asarray([feature_names[i] for i in order]),
            )
        except Exception as e_save:
            logger.warning(f"Failed to save SHAP values: {e_save}")
    except Exception as e:
        logger.warning(f"SHAP plotting failed: {e}")


def _collect_subject_ids_with_features(deriv_root: Path) -> List[str]:
    """Find subjects that have the required feature and target TSVs."""
    subs = []
    for sub_dir in sorted((deriv_root).glob("sub-*/eeg/features")):
        eeg_feat = sub_dir / "features_eeg_direct.tsv"
        y_tsv = sub_dir / "target_vas_ratings.tsv"
        if eeg_feat.exists() and y_tsv.exists():
            sub = sub_dir.parts[-3]  # sub-XXX
            subs.append(sub.replace("sub-", ""))
    return subs


def load_tabular_features_and_targets(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, List[str], pd.DataFrame]:
    """Load all subjects' direct EEG features and targets.

    Returns
    -------
    X_all : DataFrame (n_trials_total, n_features)
    y_all : Series (n_trials_total,)
    groups : np.ndarray of subject labels (length == n_trials_total)
    feature_names : list of str
    """
    if subjects is None or subjects == ["all"]:
        subjects = _collect_subject_ids_with_features(deriv_root)
        logger.info(f"Detected {len(subjects)} subjects with features.")
    else:
        logger.info(f"Using subjects from config/args: {subjects}")

    X_list: List[pd.DataFrame] = []
    y_list: List[pd.Series] = []
    g_list: List[str] = []
    trial_ids: List[int] = []
    subj_ids: List[str] = []
    col_template: Optional[List[str]] = None

    n_found = 0
    for s in subjects:
        sub = f"sub-{s}"
        feat_dir = deriv_root / sub / "eeg" / "features"
        X_path = feat_dir / "features_eeg_direct.tsv"
        y_path = feat_dir / "target_vas_ratings.tsv"
        if not (X_path.exists() and y_path.exists()):
            logger.warning(f"Missing features/targets for {sub}; skipping.")
            continue
        try:
            X = pd.read_csv(X_path, sep="\t")
            y_df = pd.read_csv(y_path, sep="\t")
        except Exception as e:
            logger.warning(f"Failed reading TSV for {sub}: {e}")
            continue

        # Ensure numeric target (first column assumed to be rating as saved by 03_feature_engineering)
        tgt_col = y_df.columns[0]
        y = pd.to_numeric(y_df[tgt_col], errors="coerce")
        # Align lengths
        n = min(len(X), len(y))
        if n == 0:
            logger.warning(f"No trials after alignment for {sub}; skipping.")
            continue
        # Enforce consistent column order across subjects
        if col_template is None:
            col_template = list(X.columns)
        else:
            if list(X.columns) != col_template:
                common = [c for c in col_template if c in X.columns]
                if len(common) == 0:
                    logger.error(f"No overlapping feature columns for {sub}. Aborting.")
                    raise RuntimeError("Inconsistent feature columns across subjects.")
                if len(common) < len(col_template):
                    logger.warning(
                        f"Column mismatch for {sub}. Using intersection of {len(common)} features."
                    )
                X = X.loc[:, common]
                col_template = common  # shrink template to intersection

        X = X.iloc[:n, :]
        y = y.iloc[:n]

        X_list.append(X)
        y_list.append(y)
        g_list.extend([sub] * n)
        trial_ids.extend(list(range(n)))
        subj_ids.extend([sub] * n)
        n_found += 1

    if n_found == 0:
        raise RuntimeError("No subjects with both features and targets were found. Run 03_feature_engineering.py first.")

    # Final alignment to the (possibly shrunken) column template to prevent NaN padding
    if col_template is None:
        raise RuntimeError("No feature columns detected.")
    X_list = [Xi.loc[:, col_template].copy() for Xi in X_list]

    X_all = pd.concat(X_list, axis=0, ignore_index=True)
    y_all = pd.concat(y_list, axis=0, ignore_index=True)
    groups = np.array(g_list)
    feature_names = list(X_all.columns)

    # Drop rows with NaNs in targets only; keep NaNs in X (imputed in pipelines)
    mask_valid = ~y_all.isna()
    n_dropped = int((~mask_valid).sum())
    if n_dropped > 0:
        logger.warning(f"Dropping {n_dropped} trials with NaN targets.")
    X_all = X_all.loc[mask_valid].reset_index(drop=True)
    y_all = y_all.loc[mask_valid].reset_index(drop=True)
    groups = groups[mask_valid.values]
    subj_ids = np.asarray(subj_ids)[mask_valid.values].tolist()
    trial_ids = np.asarray(trial_ids)[mask_valid.values].tolist()

    meta = pd.DataFrame({
        "subject_id": subj_ids,
        "trial_id": trial_ids,
    })

    logger.info(
        f"Aggregated features: n_trials={len(X_all)}, n_features={X_all.shape[1]}, n_subjects={n_found}"
    )
    return X_all, y_all, groups, feature_names, meta


# -----------------------------------------------------------------------------
# Metrics and plotting helpers
# -----------------------------------------------------------------------------

def _pearsonr_abs(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Numerically stable absolute Pearson correlation."""
    r, _ = _safe_pearsonr(y_true, y_pred)
    return float(abs(r)) if np.isfinite(r) else 0.0


def _make_pearsonr_abs_scorer():
    return make_scorer(lambda yt, yp: _pearsonr_abs(np.asarray(yt), np.asarray(yp)), greater_is_better=True)


def _make_pearsonr_scorer():
    """Signed Pearson r scorer to avoid sign-flipped models."""
    return make_scorer(lambda yt, yp: _safe_pearsonr(np.asarray(yt), np.asarray(yp))[0], greater_is_better=True)


def _create_stratified_cv_by_binned_targets(y: np.ndarray, n_splits: int = 5, n_bins: int = 5, random_state: int = 42):
    """Create StratifiedKFold using binned continuous targets for more stable inner CV.
    
    Args:
        y: Continuous target values
        n_splits: Number of CV splits
        n_bins: Number of bins for stratification (default: quintiles)
        random_state: Random seed
        
    Returns:
        StratifiedKFold object with binned targets
    """
    # Bin targets into quantiles for stratification
    try:
        y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        # Handle case where qcut fails (e.g., too few unique values)
        if pd.isna(y_binned).all():
            y_binned = np.zeros_like(y, dtype=int)
    except (ValueError, TypeError):
        # Fallback to simple binning if qcut fails
        y_binned = np.digitize(y, bins=np.linspace(np.min(y), np.max(y), n_bins+1)[1:-1])
    
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), y_binned


def _create_block_aware_cv_for_within_subject(blocks: np.ndarray, n_splits: int = 5, random_state: int = 42):
    """Create block-aware CV using GroupKFold to avoid temporal/block leakage in within-subject analysis.
    
    Uses GroupKFold with block IDs to ensure trials from the same block don't appear in both 
    train and test sets, reducing inflation from temporal autocorrelation or block effects.
    
    Args:
        blocks: Block/run identifiers for each trial
        n_splits: Number of CV splits (may be reduced if insufficient unique blocks)
        random_state: Random seed
        
    Returns:
        Tuple of (cv_splitter, effective_n_splits) where cv_splitter is GroupKFold or fallback
    """
    unique_blocks = np.unique(blocks[~pd.isna(blocks)])
    n_unique_blocks = len(unique_blocks)
    
    # Need at least 2 blocks for GroupKFold to work
    if n_unique_blocks < 2:
        logger.warning(f"Block-aware CV: insufficient unique blocks ({n_unique_blocks}), falling back to StratifiedKFold")
        # Fall back to stratified CV - better than regular KFold
        return None, 0
    
    # Adjust n_splits if we have fewer blocks than requested splits
    effective_n_splits = min(n_splits, n_unique_blocks)
    if effective_n_splits < n_splits:
        logger.info(f"Block-aware CV: reduced splits from {n_splits} to {effective_n_splits} due to {n_unique_blocks} unique blocks")
    
    # GroupKFold doesn't have random_state, but we can shuffle blocks beforehand if needed
    cv = GroupKFold(n_splits=effective_n_splits)
    return cv, effective_n_splits


def _log_cv_adjacency_info(indices: np.ndarray, fold_name: str = "") -> None:
    """Log information about temporal adjacency in CV splits to help interpret results.
    
    Args:
        indices: Array of trial indices in the split
        fold_name: Name/description of the fold for logging
    """
    if len(indices) < 2:
        return
        
    # Check for consecutive indices (potential temporal adjacency)
    sorted_idx = np.sort(indices)
    consecutive_pairs = np.sum(np.diff(sorted_idx) == 1)
    total_pairs = len(sorted_idx) - 1
    
    if total_pairs > 0:
        adjacency_ratio = consecutive_pairs / total_pairs
        if adjacency_ratio > 0.5:
            logger.info(f"{fold_name}: High temporal adjacency detected ({consecutive_pairs}/{total_pairs} consecutive pairs, {adjacency_ratio:.2f}). Consider block-aware CV if trials have temporal/sequence effects.")
        elif consecutive_pairs > 0:
            logger.debug(f"{fold_name}: Some temporal adjacency ({consecutive_pairs}/{total_pairs} consecutive pairs, {adjacency_ratio:.2f})")
    
    # Log index range for block structure awareness
    idx_range = sorted_idx[-1] - sorted_idx[0] + 1
    density = len(indices) / idx_range if idx_range > 0 else 1.0
    if density < 0.5:
        logger.debug(f"{fold_name}: Sparse index distribution (density={density:.2f}), may indicate block structure")


def scatter_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, title: str = "") -> None:
    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, ls="--", color="red", label="y=x")
    plt.xlabel("Actual pain rating")
    plt.ylabel("Predicted pain rating")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _pick_temperature_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "stimulus_temp",
        "temperature",
        "stimulus_temperature",
        "temp",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # heuristic fallback: any column containing 'temp'
    for c in df.columns:
        if "temp" in c.lower():
            return c
    return None


def _aggregate_temperature_and_trial(meta: pd.DataFrame, deriv_root: Path, task: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of temperature and trial_number aligned to `meta` rows.

    Attempts to read per-subject events TSVs located under the BIDS root
    (one level up from derivatives): `bids_root/sub-XXX/eeg/sub-XXX_task-{task}_events.tsv`.
    If unavailable or lengths mismatch, fills with NaNs and a simple 1..N trial index.
    """
    return _aggregate_temperature_trial_and_block(meta, deriv_root, task)[:2]


def _aggregate_temperature_trial_and_block(meta: pd.DataFrame, deriv_root: Path, task: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays of temperature, trial_number, and block_id aligned to `meta` rows.

    Attempts to read per-subject events TSVs located under the BIDS root
    (one level up from derivatives): `bids_root/sub-XXX/eeg/sub-XXX_task-{task}_events.tsv`.
    If unavailable or lengths mismatch, fills with NaNs and a simple 1..N trial index.
    Block information is extracted from run, block, or session columns in events TSV.
    """
    # BIDS root is one level up from derivatives
    bids_root = Path(deriv_root).parent
    temps_out = np.full(len(meta), np.nan, dtype=float)
    trials_out = np.full(len(meta), np.nan, dtype=float)
    blocks_out = np.full(len(meta), np.nan, dtype=float)

    for sub in sorted(meta["subject_id"].unique()):
        try:
            sub_str = str(sub)
            if not sub_str.startswith("sub-"):
                sub_id = sub_str
                sub_label = f"sub-{sub_id}"
            else:
                sub_label = sub_str
                sub_id = sub_label.replace("sub-", "")
            ev_path = bids_root / sub_label / "eeg" / f"{sub_label}_task-{task}_events.tsv"
            if not ev_path.exists():
                logger.warning(f"Events TSV not found for {sub_label}: {ev_path}")
                # fill trial numbers sequentially where meta matches this subject
                idx_meta = meta.index[meta["subject_id"] == sub_label].to_numpy()
                trials_out[idx_meta] = (meta.loc[idx_meta, "trial_id"].to_numpy().astype(float) + 1.0)
                continue
            ev = pd.read_csv(ev_path, sep="\t")
            tcol = _pick_temperature_column(ev)
            if tcol is None:
                logger.warning(f"Temperature column not found in events for {sub_label}; columns={list(ev.columns)[:6]}...")
            
            # Find block/run column
            block_col = None
            block_candidates = ["run", "block", "block_id", "session", "run_id"]
            for bcol in block_candidates:
                if bcol in ev.columns:
                    block_col = bcol
                    break
            
            # Align to number of trials used for this subject in features/meta
            idx_meta = meta.index[meta["subject_id"] == sub_label].to_numpy()
            n_sub = len(idx_meta)
            if len(ev) < n_sub:
                logger.warning(f"Events rows ({len(ev)}) < meta trials ({n_sub}) for {sub_label}; padding NaNs.")
            use_n = min(n_sub, len(ev))
            
            # Temperature
            if tcol is not None:
                temps = pd.to_numeric(ev[tcol], errors="coerce").to_numpy()
                temps_out[idx_meta[:use_n]] = temps[:use_n]
            
            # Trial number
            if "trial_number" in ev.columns:
                trials = pd.to_numeric(ev["trial_number"], errors="coerce").to_numpy()
                trials_out[idx_meta[:use_n]] = trials[:use_n]
            else:
                # fallback: meta trial_id is 0..N-1
                trials_out[idx_meta] = (meta.loc[idx_meta, "trial_id"].to_numpy().astype(float) + 1.0)
            
            # Block/run information
            if block_col is not None:
                blocks = pd.to_numeric(ev[block_col], errors="coerce").to_numpy()
                blocks_out[idx_meta[:use_n]] = blocks[:use_n]
                logger.debug(f"Found block info in column '{block_col}' for {sub_label}")
            else:
                # No block info available - use subject ID as block identifier
                blocks_out[idx_meta] = float(hash(sub_label) % 10000)  # Convert to numeric
        except Exception as e:
            logger.warning(f"Failed to read/align events for {sub}: {e}")
            idx_meta = meta.index[meta["subject_id"] == sub].to_numpy()
            trials_out[idx_meta] = (meta.loc[idx_meta, "trial_id"].to_numpy().astype(float) + 1.0)

    return temps_out, trials_out, blocks_out


def _partial_corr_xy_given_z(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Backwards-compatible alias for partial correlation r(x,y|z).

    Accepts 1D or 2D z; delegates to the canonical _partial_corr_xy_given_Z.
    Returns np.nan on failure.
    """
    try:
        Z = np.asarray(z, float)
        if Z.ndim == 1:
            Z = Z[:, None]
        return float(_partial_corr_xy_given_Z(x, y, Z))
    except Exception:
        return float("nan")


def _partial_corr_xy_given_Z(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """Partial correlation r(x,y|Z) with multi-covariate control via residualization.

    Z can be 1D or 2D; an intercept is added internally. Returns np.nan on failure.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[mask], y[mask], Z[mask]
    if len(x) < max(3, Z.shape[1] + 2):
        return float("nan")
    
    # Check for multicollinearity in Z
    if Z.shape[1] > 1:
        corr_Z = np.corrcoef(Z.T)
        if np.any(np.abs(corr_Z - np.eye(Z.shape[1])) > 0.95):
            logger.warning("High multicollinearity detected in covariates for partial correlation")
    
    XZ = np.c_[np.ones(len(Z)), Z]
    try:
        # Check condition number - much stricter threshold
        cond_num = np.linalg.cond(XZ)
        if cond_num > 1e8:  # Much stricter threshold  
            return float("nan")  # Silent failure to reduce log spam
            
        # Check rank
        if np.linalg.matrix_rank(XZ, rcond=1e-10) < XZ.shape[1]:
            return float("nan")
            
        bx, residuals_x, rank_x, _ = np.linalg.lstsq(XZ, x, rcond=1e-10)
        by, residuals_y, rank_y, _ = np.linalg.lstsq(XZ, y, rcond=1e-10)
        
        if rank_x < XZ.shape[1] or rank_y < XZ.shape[1]:
            return float("nan")
            
        x_res = x - XZ @ bx
        y_res = y - XZ @ by
        
        # Check residual variance with relative tolerance
        var_x_res = np.var(x_res, ddof=1) if len(x_res) > 1 else 0.0
        var_y_res = np.var(y_res, ddof=1) if len(y_res) > 1 else 0.0
        tol = 1e-12 * max(np.var(x), np.var(y), 1.0)
        if var_x_res < tol or var_y_res < tol:
            return float("nan")
            
        r, _ = _safe_pearsonr(x_res, y_res)
        return float(r)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")
    except Exception:
        return float("nan")


def _cluster_bootstrap_subjects(df: pd.DataFrame, subject_col: str, n_boot: int, seed: int, func) -> Tuple[float, Tuple[float, float]]:
    """Generic subject-level cluster bootstrap for a scalar metric.

    func must accept a DataFrame and return a scalar metric.
    Returns (point_estimate, (ci_low, ci_high)).
    """
    rng = np.random.default_rng(seed)
    # point estimate
    theta0 = float(func(df))
    subs = df[subject_col].unique().tolist()
    # bootstrap
    thetas = np.zeros(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        parts = [df[df[subject_col] == s] for s in pick]
        bs = pd.concat(parts, axis=0, ignore_index=True)
        try:
            thetas[b] = float(func(bs))
        except Exception:
            thetas[b] = np.nan
    thetas = thetas[np.isfinite(thetas)]
    if len(thetas) == 0:
        return theta0, (float("nan"), float("nan"))
    # Percentile CI
    lo_p, hi_p = np.percentile(thetas, [2.5, 97.5]).tolist()
    # BCa CI via subject-level jackknife
    try:
        jk_vals = []
        for s in subs:
            d_jk = df[df[subject_col] != s]
            v = float(func(d_jk))
            if np.isfinite(v):
                jk_vals.append(v)
        jk_vals = np.asarray(jk_vals, float)
        if len(jk_vals) >= 3 and np.isfinite(theta0):
            tdot = float(np.mean(jk_vals))
            num = np.sum((tdot - jk_vals) ** 3)
            den = 6.0 * (np.sum((tdot - jk_vals) ** 2) ** 1.5 + 1e-12)
            a = float(num / den) if np.isfinite(num) and np.isfinite(den) and den != 0 else 0.0
            # bias-correction
            z0 = float(norm.ppf((np.sum(thetas < theta0) + 1e-12) / (len(thetas) + 2e-12))) if np.isfinite(theta0) else 0.0
            def _bca_quant(alpha: float) -> float:
                zalpha = float(norm.ppf(alpha))
                adj = z0 + (z0 + zalpha) / max(1e-12, (1 - a * (z0 + zalpha)))
                return float(norm.cdf(adj))
            a1 = _bca_quant(0.025)
            a2 = _bca_quant(0.975)
            lo_bca, hi_bca = np.percentile(thetas, [100 * a1, 100 * a2]).tolist()
        else:
            lo_bca, hi_bca = lo_p, hi_p
    except Exception:
        lo_bca, hi_bca = lo_p, hi_p
    # Return BCa by default; callers that need percentile can compute separately
    return theta0, (float(lo_bca), float(hi_bca))


def _mae(a: np.ndarray) -> float:
    return float(np.mean(np.abs(a))) if len(a) else float("nan")


def _rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def plot_per_subject_violin(per_subj: pd.DataFrame, save_path: Path, title: str = "RF per-subject performance") -> None:
    # Expect columns: 'pearson_r' and 'r2'
    df = per_subj.copy()
    df = df.dropna(subset=["pearson_r", "r2"])
    # Fisher-z mean and 95% CI for r
    r_vals = df["pearson_r"].values
    r_vals = r_vals[np.isfinite(r_vals)]
    if len(r_vals) > 0:
        z = np.arctanh(np.clip(r_vals, -0.999, 0.999))
        z_mean = float(np.mean(z))
        z_se = float(np.std(z, ddof=1) / max(len(z) ** 0.5, 1.0)) if len(z) > 1 else 0.0
        z_ci = (z_mean - 1.96 * z_se, z_mean + 1.96 * z_se)
        r_mean = float(np.tanh(z_mean))
        r_ci_low, r_ci_high = float(np.tanh(z_ci[0])), float(np.tanh(z_ci[1]))
    else:
        r_mean, r_ci_low, r_ci_high = np.nan, np.nan, np.nan

    plt.figure(figsize=(8, 5), dpi=150)
    plt.suptitle(title)
    ax1 = plt.subplot(1, 2, 1)
    sns.violinplot(y=df["pearson_r"], color="#8ecae6", ax=ax1)
    sns.boxplot(y=df["pearson_r"], width=0.2, boxprops=dict(alpha=0.5), ax=ax1)
    if np.isfinite(r_mean):
        ax1.axhline(r_mean, color="red", linestyle="--", label=f"mean z→r={r_mean:.2f}")
        ax1.fill_between([-.5, .5], r_ci_low, r_ci_high, color="red", alpha=0.15, label="95% CI")
    ax1.set_ylabel("Pearson r")
    ax1.set_xticks([])
    ax1.legend(loc="lower right")

    ax2 = plt.subplot(1, 2, 2)
    sns.violinplot(y=df["r2"], color="#a8dadc", ax=ax2)
    sns.boxplot(y=df["r2"], width=0.2, boxprops=dict(alpha=0.5), ax=ax2)
    ax2.set_ylabel("R²")
    ax2.set_xticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def plot_permutation_null_hist(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                               save_path: Path, n_perm: Optional[int] = None, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    if n_perm is None:
        try:
            n_perm = int(CONFIG["analysis"]["n_perm_quick"])  # type: ignore[index]
        except Exception:
            n_perm = 1000
    obs_r, _ = _safe_pearsonr(y_true, y_pred)
    null_rs = np.zeros(n_perm, dtype=float)
    groups = np.asarray(groups)
    for i in range(n_perm):
        y_shuf = y_true.copy() if isinstance(y_true, np.ndarray) else y_true.to_numpy().copy()
        # within-subject shuffle
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) > 1:
                y_shuf[idx] = rng.permutation(y_shuf[idx])
        r_i, _ = _safe_pearsonr(y_shuf, y_pred)
        null_rs[i] = r_i if np.isfinite(r_i) else 0.0
    # two-sided p-value by |r|
    p_val = float((np.sum(np.abs(null_rs) >= abs(obs_r)) + 1) / (n_perm + 1))
    
    # Bootstrap CI for observed correlation using subject-level resampling
    obs_r_ci = [np.nan, np.nan]
    try:
        # Create prediction DataFrame for bootstrap function
        pred_df_bootstrap = pd.DataFrame({
            "subject_id": groups,
            "y_true": y_true,
            "y_pred": y_pred
        })
        bootstrap_res = _bootstrap_pooled_metrics_by_subject(pred_df_bootstrap, seed=seed + 999)
        obs_r_ci = bootstrap_res["r_ci"]
        logger.info(f"Bootstrap 95% CI for observed r: [{obs_r_ci[0]:.3f}, {obs_r_ci[1]:.3f}]")
    except Exception as e:
        logger.warning(f"Failed to compute bootstrap CI for observed r: {e}")
    
    plt.figure(figsize=(6, 4), dpi=150)
    plt.hist(null_rs, bins=30, alpha=0.7, color="#bdb2ff", edgecolor="white")
    plt.axvline(obs_r, color="red", linestyle="--", 
                label=f"observed r={obs_r:.3f}\n95% CI: [{obs_r_ci[0]:.3f}, {obs_r_ci[1]:.3f}]\np={p_val:.3g}")
    plt.xlabel("Null pooled r (within-subject shuffles)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return p_val


def plot_learning_curve_rf(X: pd.DataFrame, y: pd.Series, groups: np.ndarray, results_dir: Path,
                           save_path: Path, seed: int = 42,
                           fractions: Optional[List[float]] = None,
                           best_params_path: Optional[Path] = None) -> None:
    if fractions is None:
        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    rng = np.random.default_rng(seed)
    logo = LeaveOneGroupOut()

    _rf_best_path = best_params_path or (results_dir / CONFIG["paths"]["best_params"]["rf_loso"])
    best_params_map = _read_best_params_jsonl(_rf_best_path)

    r_list = []
    r2_list = []
    sizes = []
    for frac in fractions:
        y_true_all: List[float] = []
        y_pred_all: List[float] = []
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
            # Sample training subset per subject to preserve distribution
            train_groups = groups[train_idx]
            take_idx = []
            for g in np.unique(train_groups):
                idx_g = train_idx[train_groups == g]
                k = max(1, int(np.ceil(len(idx_g) * frac)))
                take_idx.extend(rng.choice(idx_g, size=k, replace=False).tolist())
            take_idx = np.array(sorted(set(take_idx)))

            # Build RF with best params for this fold if available
            rf_params = best_params_map.get(fold, {})
            rf = RandomForestRegressor(n_estimators=rf_params.get("rf__n_estimators", 500),
                                       max_depth=rf_params.get("rf__max_depth", None),
                                       max_features=rf_params.get("rf__max_features", "sqrt"),
                                       min_samples_leaf=rf_params.get("rf__min_samples_leaf", 1),
                                       random_state=seed, n_jobs=-1, bootstrap=True)
            pipe = _create_base_preprocessing_pipeline(include_scaling=False)
            pipe.steps.append(("rf", rf))
            pipe.fit(X.iloc[take_idx], y.iloc[take_idx])
            y_pred = pipe.predict(X.iloc[test_idx])
            y_true_all.extend(y.iloc[test_idx].tolist())
            y_pred_all.extend(y_pred.tolist())

        r, _ = _safe_pearsonr(np.asarray(y_true_all), np.asarray(y_pred_all))
        r2 = r2_score(np.asarray(y_true_all), np.asarray(y_pred_all))
        r_list.append(r)
        r2_list.append(r2)
        sizes.append(frac)

    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(sizes, r_list, marker="o", label="Pooled r")
    plt.plot(sizes, r2_list, marker="s", label="Pooled R²")
    plt.xlabel("Training fraction per fold")
    plt.ylabel("Score")
    plt.ylim(min(min(r_list), min(r2_list), -0.1), 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("RF Learning Curve")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_learning_curve_en(X: pd.DataFrame, y: pd.Series, groups: np.ndarray, results_dir: Path,
                           save_path: Path, seed: int = 42,
                           fractions: Optional[List[float]] = None,
                           best_params_path: Optional[Path] = None) -> None:
    """Plot ElasticNet learning curve for parity with RF."""
    if fractions is None:
        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    rng = np.random.default_rng(seed)
    logo = LeaveOneGroupOut()

    _en_best_path = best_params_path or (results_dir / CONFIG["paths"]["best_params"]["elasticnet_loso"])
    best_params_map = _read_best_params_jsonl(_en_best_path)

    r_list = []
    r2_list = []
    sizes = []
    for frac in fractions:
        y_true_all: List[float] = []
        y_pred_all: List[float] = []
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
            # Sample training subset per subject to preserve distribution
            train_groups = groups[train_idx]
            take_idx = []
            for g in np.unique(train_groups):
                idx_g = train_idx[train_groups == g]
                k = max(1, int(np.ceil(len(idx_g) * frac)))
                take_idx.extend(rng.choice(idx_g, size=k, replace=False).tolist())
            take_idx = np.array(sorted(set(take_idx)))

            # Build ElasticNet with best params for this fold if available
            en_params = best_params_map.get(fold, {})
            en_cfg = CONFIG["models"]["elasticnet"]
            en = ElasticNet(
                alpha=en_params.get("regressor__regressor__alpha", 1.0),
                l1_ratio=en_params.get("regressor__regressor__l1_ratio", 0.5),
                max_iter=en_cfg["max_iter"],
                tol=en_cfg["tol"],
                selection=en_cfg["selection"],
                random_state=seed
            )
            # Create pipeline with scaling and feature selection like the main pipeline
            pipe = _create_base_preprocessing_pipeline(include_scaling=True)
            pipe.steps.append(("feature_selection", SelectFromModel(en, threshold="median")))
            pipe.steps.append(("regressor", TransformedTargetRegressor(regressor=en, transformer=PowerTransformer(method="yeo-johnson", standardize=True))))
            
            pipe.fit(X.iloc[take_idx], y.iloc[take_idx])
            y_pred = pipe.predict(X.iloc[test_idx])
            y_true_all.extend(y.iloc[test_idx].tolist())
            y_pred_all.extend(y_pred.tolist())

        r, _ = _safe_pearsonr(np.asarray(y_true_all), np.asarray(y_pred_all))
        r2 = r2_score(np.asarray(y_true_all), np.asarray(y_pred_all))
        r_list.append(r)
        r2_list.append(r2)
        sizes.append(frac)

    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(sizes, r_list, marker="o", label="Pooled r")
    plt.plot(sizes, r2_list, marker="s", label="Pooled R²")
    plt.xlabel("Training fraction per fold")
    plt.ylabel("Score")
    plt.ylim(min(min(r_list), min(r2_list), -0.1), 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("ElasticNet Learning Curve")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_residuals_and_qq(y_true: np.ndarray, y_pred: np.ndarray, scatter_path: Path, qq_path: Path) -> None:
    resid = y_true - y_pred
    # residuals vs true rating
    plt.figure(figsize=(6, 4), dpi=150)
    plt.scatter(y_true, resid, alpha=0.5, edgecolors="none")
    plt.axhline(0.0, color="red", linestyle="--")
    rho, p = spearmanr(y_true, np.abs(resid))
    plt.title(f"Residuals vs true (Spearman |resid|~true: rho={rho:.2f}, p={p:.3g})")
    plt.xlabel("True rating")
    plt.ylabel("Residual (true - pred)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    # QQ plot of residuals
    plt.figure(figsize=(5, 5), dpi=150)
    (osm, osr), (slope, intercept, r) = probplot(resid, dist="norm", fit=True)
    plt.scatter(osm, osr, alpha=0.6, edgecolors="none")
    plt.plot(osm, slope * np.asarray(osm) + intercept, color="red", linestyle="--", label=f"fit r={r:.2f}")
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Ordered residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(qq_path)
    plt.close()


def plot_bland_altman(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    mean_vals = 0.5 * (y_true + y_pred)
    diff = y_true - y_pred
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_low, loa_high = md - 1.96 * sd, md + 1.96 * sd
    plt.figure(figsize=(6, 4), dpi=150)
    plt.scatter(mean_vals, diff, alpha=0.5, edgecolors="none")
    plt.axhline(md, color="red", linestyle="--", label=f"mean diff={md:.2f}")
    plt.axhline(loa_low, color="gray", linestyle=":", label=f"LoA={loa_low:.2f}")
    plt.axhline(loa_high, color="gray", linestyle=":", label=f"LoA={loa_high:.2f}")
    plt.xlabel("Mean of true and predicted")
    plt.ylabel("Difference (true - pred)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute quantified calibration metrics for regression."""
    # Remove NaN values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 2:
        return {"slope": np.nan, "intercept": np.nan, "r_calibration": np.nan, "n_samples": 0}
    
    # Simple OLS: y_true = slope * y_pred + intercept
    slope, intercept, r_cal, p_val, std_err = scipy.stats.linregress(y_pred_clean, y_true_clean)
    
    return {
        "slope": float(slope),
        "intercept": float(intercept), 
        "r_calibration": float(r_cal),
        "p_value": float(p_val),
        "std_err": float(std_err),
        "n_samples": int(len(y_true_clean)),
        "note": "Perfect calibration: slope=1, intercept=0"
    }


def plot_calibration_curve(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, n_bins: int = 10) -> None:
    # Bin by true ratings into quantiles
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y_true, quantiles)
    # ensure unique bin edges
    bins = np.unique(bins)
    if len(bins) < 3:
        # fallback to equal-width
        bins = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    inds = np.digitize(y_true, bins=bins, right=True)
    xs, ys = [], []
    for b in range(1, len(bins)):
        mask = inds == b
        if np.sum(mask) == 0:
            continue
        xs.append(float(np.mean(y_true[mask])))
        ys.append(float(np.mean(y_pred[mask])))
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    plt.figure(figsize=(5, 5), dpi=150)
    plt.scatter(xs, ys, color="#219ebc", label="Bin means")
    # linear fit
    if len(xs) >= 2:
        m, c = np.polyfit(xs, ys, 1)
        xfit = np.linspace(xs.min(), xs.max(), 100)
        plt.plot(xfit, m * xfit + c, color="orange", label=f"Fit: y={m:.2f}x+{c:.2f}")
    # LOESS (LOWESS) smoothing over raw (true, pred), with simple CV for span if statsmodels is available
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess  # type: ignore
        # Candidate spans and 5-fold CV on raw pairs
        rng = np.random.default_rng(42)
        spans = [0.2, 0.3, 0.4, 0.5, 0.6]
        if len(y_true) >= 10:
            idx = np.arange(len(y_true))
            rng.shuffle(idx)
            folds = np.array_split(idx, 5)
            best_span = spans[0]
            best_mse = np.inf
            for frac in spans:
                mse_total = 0.0
                for k in range(5):
                    te = folds[k]
                    tr = np.setdiff1d(idx, te, assume_unique=False)
                    x_tr = y_true[tr]
                    y_tr = y_pred[tr]
                    # LOWESS returns fitted values at x_tr; interpolate for x_te
                    order = np.argsort(x_tr)
                    sm = _lowess(y_tr[order], x_tr[order], frac=frac, return_sorted=True)
                    x_sm, y_sm = sm[:, 0], sm[:, 1]
                    y_hat_te = np.interp(y_true[te], x_sm, y_sm, left=y_sm[0], right=y_sm[-1])
                    mse_total += float(np.mean((y_pred[te] - y_hat_te) ** 2))
                if mse_total < best_mse:
                    best_mse = mse_total
                    best_span = frac
        else:
            best_span = 0.4
        # Final LOWESS on full data
        order_all = np.argsort(y_true)
        sm_all = _lowess(y_pred[order_all], y_true[order_all], frac=best_span, return_sorted=True)
        plt.plot(sm_all[:, 0], sm_all[:, 1], color="#6a994e", linewidth=2, label=f"LOESS (span={best_span:.2f})")
    except Exception as _e:
        # statsmodels not installed or LOWESS failed; skip smoothing silently but keep plot
        pass
    lims = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
    plt.plot(lims, lims, linestyle="--", color="gray", label="y=x")
    plt.xlabel("Mean true (bin)")
    plt.ylabel("Mean predicted (bin)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_residuals_vs_covariate(y_true: np.ndarray, y_pred: np.ndarray, covariate: np.ndarray,
                                cov_name: str, save_path: Path) -> None:
    """Scatter residuals vs a covariate with Spearman correlations.

    Reports rho and p for resid~cov and |resid|~cov.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cov = np.asarray(covariate)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(cov)
    if np.sum(mask) < 3:
        logger.warning(f"Insufficient data for residuals vs {cov_name} plot")
        return
    resid = y_true[mask] - y_pred[mask]
    cov_m = cov[mask]
    rho_resid, p_resid = spearmanr(cov_m, resid)
    rho_abs, p_abs = spearmanr(cov_m, np.abs(resid))
    plt.figure(figsize=(6, 4), dpi=150)
    plt.scatter(cov_m, resid, alpha=0.5, edgecolors="none")
    plt.axhline(0.0, color="red", linestyle="--")
    plt.title(f"Residuals vs {cov_name} (Spearman resid: rho={rho_resid:.2f}, p={p_resid:.3g}; |resid|: rho={rho_abs:.2f}, p={p_abs:.3g})")
    plt.xlabel(cov_name)
    plt.ylabel("Residual (true - pred)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_paired_metric_scatter(
    per_left: pd.DataFrame,
    per_right: pd.DataFrame,
    metric: str,
    save_path: Path,
    label_left: str = "LOSO",
    label_right: str = "WithinKFold",
) -> None:
    """Scatter x=metric(left) vs y=metric(right) paired by subject.

    Expects both DataFrames to have columns: 'group' and the specified metric.
    """
    df_l = per_left[["group", metric]].rename(columns={metric: f"{label_left}_{metric}"})
    df_r = per_right[["group", metric]].rename(columns={metric: f"{label_right}_{metric}"})
    d = df_l.merge(df_r, on="group", how="inner").dropna()
    if d.empty:
        logger.warning(f"Paired scatter skipped: no overlapping subjects for metric={metric}")
        return
    x = d[f"{label_left}_{metric}"].to_numpy()
    y = d[f"{label_right}_{metric}"].to_numpy()
    plt.figure(figsize=(5, 5), dpi=150)
    plt.scatter(x, y, alpha=0.7, edgecolors="none", color="#3a86ff")
    # y=x
    lims = [float(np.nanmin([x.min(), y.min()])), float(np.nanmax([x.max(), y.max()]))]
    if np.isfinite(lims).all():
        plt.plot(lims, lims, linestyle="--", color="gray", label="y=x")
        plt.xlim(lims)
        plt.ylim(lims)
    plt.xlabel(f"{label_left} {metric}")
    plt.ylabel(f"{label_right} {metric}")
    plt.title(f"Paired comparison: {metric}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _fisher_z_mean(r_values: List[float]) -> float:
    r_values = [r for r in r_values if np.isfinite(r) and -0.999 < r < 0.999]
    if len(r_values) == 0:
        return np.nan
    z = np.arctanh(r_values)
    z_mean = np.mean(z)
    return float(np.tanh(z_mean))


def _safe_pearsonr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Numerically stable Pearson correlation with comprehensive checks."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    # Basic shape and size checks
    if len(a) != len(b) or len(a) < 2:
        return np.nan, np.nan
    
    # Check for NaN/inf values
    a_valid = np.isfinite(a)
    b_valid = np.isfinite(b)
    both_valid = a_valid & b_valid
    
    if np.sum(both_valid) < 2:
        return np.nan, np.nan
    
    # Use only valid values
    a_clean = a[both_valid]
    b_clean = b[both_valid]
    
    # Check for zero variance with numerical tolerance
    var_a = np.var(a_clean, ddof=1) if len(a_clean) > 1 else 0.0
    var_b = np.var(b_clean, ddof=1) if len(b_clean) > 1 else 0.0
    
    # Use relative tolerance for variance check
    tol = 1e-12 * max(np.abs(np.mean(a_clean)), np.abs(np.mean(b_clean)), 1.0)
    if var_a < tol or var_b < tol:
        return np.nan, np.nan
    
    try:
        r, p = pearsonr(a_clean, b_clean)
        # Additional stability check on result
        if not np.isfinite(r) or not np.isfinite(p):
            return np.nan, np.nan
        # Clamp correlation to valid range due to numerical precision
        r = np.clip(r, -1.0, 1.0)
    except (ValueError, RuntimeError, FloatingPointError):
        r, p = np.nan, np.nan
    
    return float(r), float(p)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Tuple[dict, pd.DataFrame]:
    # Apply finite mask to handle NaN predictions before computing metrics
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    groups = np.asarray(groups)[mask]
    
    # pooled
    r, p = _safe_pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    pooled = {"pearson_r": float(r), "p_value": float(p), "r2": float(r2), "explained_variance": float(evs)}

    # per-subject
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups})
    rows = []
    r_list = []
    for g, d in df.groupby("group"):
        if len(d) < 2:
            continue
        # Apply finite mask per subject as well
        y_true_subj = d["y_true"].values
        y_pred_subj = d["y_pred"].values
        mask_subj = np.isfinite(y_true_subj) & np.isfinite(y_pred_subj)
        y_true_subj = y_true_subj[mask_subj]
        y_pred_subj = y_pred_subj[mask_subj]
        
        if len(y_true_subj) < 2:
            continue
            
        rg, pg = _safe_pearsonr(y_true_subj, y_pred_subj)
        r2g = r2_score(y_true_subj, y_pred_subj)
        evsg = explained_variance_score(y_true_subj, y_pred_subj)
        rows.append({"group": g, "pearson_r": float(rg), "p_value": float(pg), "r2": float(r2g), "explained_variance": float(evsg), "n_trials": int(len(y_true_subj))})
        r_list.append(float(rg))
    per_subject = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    avg_r_fisher_z = _fisher_z_mean(r_list)
    pooled["avg_subject_r_fisher_z"] = float(avg_r_fisher_z)
    return pooled, per_subject


def _nested_loso_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    pipe: Pipeline,
    param_grid: dict,
    inner_cv_splits: int,
    n_jobs: int,
    seed: int,
    best_params_log_path: Optional[Path] = None,
    model_name: str = "",
    outer_n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """Thin wrapper that delegates to the unified LOSO implementation using array inputs."""
    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    return _nested_loso_predictions_unified(
        X=X_arr,
        y=y_arr,
        groups=groups,
        pipe=pipe,
        param_grid=param_grid,
        inner_cv_splits=inner_cv_splits,
        n_jobs=n_jobs,
        seed=seed,
        best_params_log_path=best_params_log_path,
        model_name=model_name,
        outer_n_jobs=outer_n_jobs,
    )


def _loso_predictions_with_fixed_params(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    pipe_template: Pipeline,
    best_params_map: dict,
    seed: int = 42,
    outer_n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """LOSO predictions using a fixed pipeline with per-fold best params (no inner CV).

    best_params_map: dict mapping fold index (1-based) -> flat param dict (e.g., {'rf__max_depth': None, ...}).
    Returns pooled out-of-fold predictions with deterministic fold ordering.
    """
    logo = LeaveOneGroupOut()
    folds = [(fold, train_idx, test_idx) for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1)]

    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
        np.random.seed(seed + fold)
        pyrandom.seed(seed + fold)
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipe = clone(pipe_template)
        params = best_params_map.get(fold, {}) if isinstance(best_params_map, dict) else {}
        try:
            if isinstance(params, dict) and len(params) > 0:
                pipe.set_params(**params)
        except Exception:
            # Ignore if params fail to set
            pass
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            logger.warning(f"Fixed-params fold {fold}: fit failed: {e}")
            y_pred = np.full(len(test_idx), np.nan, dtype=float)
            return {
                "fold": fold,
                "y_true": y_test.to_numpy(),
                "y_pred": np.asarray(y_pred),
                "groups": groups[test_idx].tolist(),
                "test_idx": test_idx.tolist(),
            }
        try:
            y_pred = pipe.predict(X_test)
        except Exception as e:
            logger.warning(f"Fixed-params fold {fold}: predict failed: {e}")
            y_pred = np.full(len(test_idx), np.nan, dtype=float)
        return {
            "fold": fold,
            "y_true": y_test.to_numpy(),
            "y_pred": np.asarray(y_pred),
            "groups": groups[test_idx].tolist(),
            "test_idx": test_idx.tolist(),
        }

    if outer_n_jobs and outer_n_jobs > 1:
        results = Parallel(n_jobs=outer_n_jobs, prefer="threads")(delayed(_run_fold)(f, tr, te) for (f, tr, te) in folds)
    else:
        results = [_run_fold(f, tr, te) for (f, tr, te) in folds]

    # Aggregate in deterministic order
    results = sorted(results, key=lambda r: r["fold"])  # type: ignore
    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    groups_ordered: List[str] = []
    test_indices_order: List[int] = []
    fold_ids: List[int] = []
    for rec in results:
        y_true_all.extend(rec["y_true"].tolist())
        y_pred_all.extend(rec["y_pred"].tolist())
        groups_ordered.extend(rec["groups"])  # type: ignore
        test_indices_order.extend(rec["test_idx"])  # type: ignore
        fold_ids.extend([rec["fold"]] * len(rec["test_idx"]))  # type: ignore

    return np.asarray(y_true_all), np.asarray(y_pred_all), groups_ordered, test_indices_order, fold_ids


# -----------------------------------------------------------------------------
# Within-subject KFold CV predictions (tabular features)
# -----------------------------------------------------------------------------

def _within_subject_kfold_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    pipe: Pipeline,
    param_grid: dict,
    inner_cv_splits: int,
    n_jobs: int,
    seed: int,
    best_params_log_path: Optional[Path] = None,
    model_name: str = "",
    outer_n_jobs: int = 1,
    deriv_root: Optional[Path] = None,
    task: str = TASK,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """Perform within-subject KFold CV across trials for each subject.
    
    Now uses block-aware CV (GroupKFold) when block information is available to avoid
    temporal autocorrelation and block effects that can inflate performance estimates.

    Returns concatenated arrays in deterministic fold order, with global test indices.
    """
    # Use multi-metric scoring with signed Pearson r as primary to avoid sign-flipped models
    scoring = {'r': _make_pearsonr_scorer(), 'neg_mse': 'neg_mean_squared_error'}
    refit_metric = 'r'
    
    # Extract block information for block-aware CV if deriv_root is provided
    block_ids = None
    if deriv_root is not None:
        try:
            # Create minimal meta DataFrame for block extraction
            meta = pd.DataFrame({
                'subject_id': groups,
                'trial_id': np.arange(len(groups))
            })
            _, _, block_ids = _aggregate_temperature_trial_and_block(meta, deriv_root, task)
            logger.info(f"Block-aware within-subject CV: extracted block info for {len(np.unique(block_ids[~pd.isna(block_ids)]))} unique blocks")
        except Exception as e:
            logger.warning(f"Block extraction failed for within-subject CV: {e}. Falling back to trial-level splits.")
            block_ids = None

    # Build outer folds: for each subject, use block-aware CV when possible
    folds: List[Tuple[int, np.ndarray, np.ndarray, str]] = []
    fold_counter = 0
    unique_subs = [str(s) for s in np.unique(groups)]
    for s in unique_subs:
        idx_s = np.where(groups == s)[0]
        n_samp = len(idx_s)
        if n_samp < 2:
            logger.warning(f"Subject {s}: <2 trials, skipping within-subject KFold")
            continue
        n_splits = min(inner_cv_splits, n_samp)
        if n_splits < 2:
            n_splits = 2  # minimal valid CV
            if n_samp < 2:
                continue
        
        # Try block-aware CV first
        use_block_cv = False
        if block_ids is not None:
            blocks_s = block_ids[idx_s]
            block_cv, effective_splits = _create_block_aware_cv_for_within_subject(
                blocks_s, n_splits=n_splits, random_state=seed
            )
            if block_cv is not None:
                use_block_cv = True
                logger.info(f"Subject {s}: using block-aware CV ({effective_splits} splits)")
                for tr_local, te_local in block_cv.split(idx_s, groups=blocks_s):
                    fold_counter += 1
                    train_idx = idx_s[tr_local]
                    test_idx = idx_s[te_local]
                    folds.append((fold_counter, train_idx, test_idx, s))
        
        if not use_block_cv:
            # Fall back to trial-level KFold
            stable = int(hashlib.sha1(str(s).encode()).hexdigest(), 16) % 1_000_000
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + stable)
            logger.info(f"Subject {s}: using trial-level KFold ({n_splits} splits)")
            for tr_local, te_local in kf.split(idx_s):
                fold_counter += 1
                train_idx = idx_s[tr_local]
                test_idx = idx_s[te_local]
                folds.append((fold_counter, train_idx, test_idx, s))

    logger.info(
        f"Executing {len(folds)} within-subject KFold folds over {len(unique_subs)} subjects; "
        f"outer_n_jobs={outer_n_jobs}; inner GridSearchCV n_jobs={1 if (outer_n_jobs and outer_n_jobs != 1) else n_jobs}"
    )

    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray, subj: str):
        # Seed per-fold for reproducibility
        np.random.seed(seed + fold)
        pyrandom.seed(seed + fold)

        _t_fold = time.time()
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner CV on training trials of the same subject - prefer block-aware CV
        n_train = len(train_idx)
        best = None
        best_params_rec = None
        if n_train >= 3:
            n_splits_inner = int(np.clip(inner_cv_splits, 2, n_train))
            
            # Try block-aware inner CV first if block info available
            inner_cv_used = "trial-level"
            cv_splits = None
            if block_ids is not None:
                blocks_train = block_ids[train_idx]
                block_inner_cv, effective_inner_splits = _create_block_aware_cv_for_within_subject(
                    blocks_train, n_splits=n_splits_inner, random_state=seed + fold
                )
                if block_inner_cv is not None:
                    inner_cv_used = "block-aware"                    
                    cv_splits = list(block_inner_cv.split(np.arange(n_train), groups=blocks_train))
                    logger.info(f"Within fold {fold} ({subj}): using block-aware inner CV ({effective_inner_splits} splits)")
            
            # Fall back to stratified CV if block-aware failed
            if cv_splits is None:
                try:
                    inner_cv, y_train_binned = _create_stratified_cv_by_binned_targets(
                        y_train, n_splits=n_splits_inner, random_state=seed + fold
                    )
                    cv_splits = list(inner_cv.split(X_train, y_train_binned))
                    logger.info(f"Within fold {fold} ({subj}): using StratifiedKFold with binned targets (n_bins={len(np.unique(y_train_binned))})")
                except Exception as e:
                    logger.warning(f"Within fold {fold} ({subj}): stratified CV failed ({e}), falling back to KFold")
                    inner_cv = KFold(n_splits=n_splits_inner, shuffle=True, random_state=seed + fold)
                    cv_splits = list(inner_cv.split(X_train))
                    inner_cv_used = "KFold fallback"
            
            # Log adjacency info for first few splits to check temporal effects
            for i, (tr_idx, te_idx) in enumerate(cv_splits[:2]):
                _log_cv_adjacency_info(te_idx, f"Within fold {fold} ({subj}) inner split {i+1} test ({inner_cv_used})")

            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                cv=cv_splits,
                scoring=scoring,
                n_jobs=(1 if outer_n_jobs and outer_n_jobs != 1 else n_jobs),
                refit=refit_metric,
                error_score=np.nan,
                pre_dispatch="2*n_jobs",
            )
            try:
                _t0 = time.time()
                grid.fit(X_train, y_train)
                _dt = time.time() - _t0
                best = grid.best_estimator_
                
                # Extract best parameters for both metrics
                cv_results = pd.DataFrame(grid.cv_results_)
                best_by_r_idx = cv_results['rank_test_r'].idxmin()
                best_by_mse_idx = cv_results['rank_test_neg_mse'].idxmin()
                
                best_params_r = cv_results.loc[best_by_r_idx, 'params']
                best_params_mse = cv_results.loc[best_by_mse_idx, 'params']
                
                logger.info(f"Within fold {fold} ({subj}): best params by r = {best_params_r}")
                logger.info(f"Within fold {fold} ({subj}): best params by neg_mse = {best_params_mse}")
                logger.info(f"Within fold {fold} ({subj}): inner CV (n_splits={n_splits_inner}) took {_dt:.1f}s")
                
                best_params_rec = {
                    "model": (model_name or None),
                    "fold": int(fold),
                    "subject": subj,
                    "best_params_by_r": best_params_r,
                    "best_params_by_neg_mse": best_params_mse,
                    "best_score_r": float(cv_results.loc[best_by_r_idx, 'mean_test_r']),
                    "best_score_neg_mse": float(cv_results.loc[best_by_mse_idx, 'mean_test_neg_mse']),
                    "best_params": grid.best_params_,  # Keep backward compatibility
                }
            except Exception as e:
                logger.warning(f"Within fold {fold} ({subj}): GridSearchCV failed: {e}; falling back to default params.")
                best = None
        else:
            logger.info(
                f"Within fold {fold} ({subj}): insufficient train trials for inner CV; fitting default pipeline params."
            )
            best = clone(pipe)
            _t0 = time.time()
            try:
                best.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Within fold {fold} ({subj}): default fit failed: {e}")
                y_pred_nan = np.full(len(test_idx), np.nan, dtype=float)
                return {
                    "fold": fold,
                    "subject": subj,
                    "y_true": y_test.to_numpy(),
                    "y_pred": y_pred_nan,
                    "groups": groups[test_idx].tolist(),
                    "test_idx": test_idx.tolist(),
                    "best_params_rec": None,
                }
            logger.info(f"Within fold {fold} ({subj}): default fit took {time.time() - _t0:.1f}s")

        if best is None:
            best = clone(pipe)
            _t0 = time.time()
            try:
                best.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Within fold {fold} ({subj}): fallback default fit failed after GridSearch error: {e}")
                y_pred_nan = np.full(len(test_idx), np.nan, dtype=float)
                return {
                    "fold": fold,
                    "subject": subj,
                    "y_true": y_test.to_numpy(),
                    "y_pred": y_pred_nan,
                    "groups": groups[test_idx].tolist(),
                    "test_idx": test_idx.tolist(),
                    "best_params_rec": None,
                }
            logger.info(f"Within fold {fold} ({subj}): fallback default fit took {time.time() - _t0:.1f}s")

        _t0 = time.time()
        try:
            y_pred = best.predict(X_test)
        except Exception as e:
            logger.warning(f"Within fold {fold} ({subj}): prediction failed: {e}")
            y_pred = np.full(len(test_idx), np.nan, dtype=float)
        _t_pred = time.time() - _t0
        logger.info(
            f"Within fold {fold} ({subj}): predict on {len(test_idx)} trials took {_t_pred:.1f}s; total fold {time.time() - _t_fold:.1f}s"
        )
        return {
            "fold": fold,
            "subject": subj,
            "y_true": y_test.to_numpy(),
            "y_pred": np.asarray(y_pred),
            "groups": groups[test_idx].tolist(),
            "test_idx": test_idx.tolist(),
            "best_params_rec": best_params_rec,
        }

    # Execute folds (optionally in parallel)
    if outer_n_jobs and outer_n_jobs != 1 and len(folds) > 1:
        results = Parallel(n_jobs=outer_n_jobs, prefer="threads")(delayed(_run_fold)(f, tr, te, s) for (f, tr, te, s) in folds)
    else:
        results = [_run_fold(f, tr, te, s) for (f, tr, te, s) in folds]

    # Aggregate
    results = sorted(results, key=lambda r: r["fold"])  # deterministic order
    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    groups_ordered: List[str] = []
    test_indices_order: List[int] = []
    fold_ids: List[int] = []
    best_param_records: List[dict] = []
    for rec in results:
        y_true_all.extend(rec["y_true"].tolist())
        y_pred_all.extend(rec["y_pred"].tolist())
        groups_ordered.extend(rec["groups"])
        test_indices_order.extend(rec["test_idx"])
        fold_ids.extend([rec["fold"]] * len(rec["test_idx"]))
        if rec["best_params_rec"] is not None:
            best_param_records.append(rec["best_params_rec"])

    # Safe logging after parallel completes
    if best_params_log_path is not None and len(best_param_records) > 0:
        try:
            _ensure_dir(best_params_log_path.parent)
            with open(best_params_log_path, "a", encoding="utf-8") as f:
                for r in best_param_records:
                    f.write(json.dumps(r) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log best params (within-subject): {e}")

    return np.asarray(y_true_all), np.asarray(y_pred_all), groups_ordered, test_indices_order, fold_ids


def _loso_baseline_predictions(
    y: pd.Series,
    groups: np.ndarray,
    mode: str = "global",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """Compute naïve LOSO baselines.

    mode="global": predict the training-set mean on each test fold.
    Note: subject_test mode removed due to data leakage (uses test labels).
    """
    if mode not in ["global"]:
        raise ValueError(f"Invalid baseline mode '{mode}'. Only 'global' is supported to prevent data leakage.")
    
    logo = LeaveOneGroupOut()
    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    groups_ordered: List[str] = []
    test_indices_order: List[int] = []
    fold_ids: List[int] = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(np.zeros(len(y)), y, groups=groups), start=1):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_test = groups[test_idx]
        if mode == "global":
            mu = float(y_train.mean())
            y_pred = np.full_like(y_test.values, fill_value=mu, dtype=float)
        else:
            raise ValueError("Unknown baseline mode")

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        groups_ordered.extend(g_test.tolist())
        test_indices_order.extend(test_idx.tolist())
        fold_ids.extend([fold] * len(test_idx))

    return np.asarray(y_true_all), np.asarray(y_pred_all), groups_ordered, test_indices_order, fold_ids


def _nested_loso_predictions_unified(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pipe: Pipeline,
    param_grid: dict,
    inner_cv_splits: int,
    n_jobs: int,
    seed: int,
    best_params_log_path: Optional[Path] = None,
    model_name: str = "",
    outer_n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """Unified nested LOSO CV implementation operating on NumPy arrays.

    Returns pooled out-of-fold predictions with deterministic fold ordering and
    optionally logs best params per outer fold.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    logo = LeaveOneGroupOut()
    # Use multi-metric scoring with signed Pearson r as primary to avoid sign-flipped models
    scoring = {'r': _make_pearsonr_scorer(), 'neg_mse': 'neg_mean_squared_error'}
    refit_metric = 'r'

    # Prepare outer folds (use dummy X in split to avoid copying large arrays)
    folds = [
        (fold, train_idx, test_idx)
        for fold, (train_idx, test_idx) in enumerate(
            logo.split(np.zeros(len(y)), y, groups=groups), start=1
        )
    ]
    logger.info(
        f"Executing {len(folds)} LOSO folds with outer_n_jobs={outer_n_jobs}; inner GridSearchCV n_jobs="
        f"{1 if (outer_n_jobs and outer_n_jobs != 1) else n_jobs}"
    )

    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
        # Seed per-fold for reproducibility
        np.random.seed(seed + fold)
        pyrandom.seed(seed + fold)

        test_subs = np.unique(groups[test_idx]).tolist()
        logger.info(f"LOSO fold {fold}: held-out {test_subs}")
        _t_fold = time.time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = groups[train_idx]

        # Inner CV across training groups; if only one, skip inner CV
        n_unique = len(np.unique(train_groups))
        best = None
        best_params_rec = None
        if n_unique >= 2:
            n_splits = min(inner_cv_splits, n_unique)
            inner_cv = GroupKFold(n_splits=n_splits)
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=(1 if outer_n_jobs and outer_n_jobs != 1 else n_jobs),
                refit=refit_metric,
                error_score=np.nan,
                pre_dispatch="2*n_jobs",
            )
            try:
                _t0 = time.time()
                grid.fit(X_train, y_train, **{"groups": train_groups})
                _dt = time.time() - _t0
                best = grid.best_estimator_
                
                # Extract best parameters for both metrics
                cv_results = pd.DataFrame(grid.cv_results_)
                best_by_r_idx = cv_results['rank_test_r'].idxmin()
                best_by_mse_idx = cv_results['rank_test_neg_mse'].idxmin()
                
                best_params_r = cv_results.loc[best_by_r_idx, 'params']
                best_params_mse = cv_results.loc[best_by_mse_idx, 'params']
                
                logger.info(f"Fold {fold}: best params by r = {best_params_r}")
                logger.info(f"Fold {fold}: best params by neg_mse = {best_params_mse}")
                logger.info(f"Fold {fold}: inner CV (n_splits={n_splits}) grid-search took {_dt:.1f}s")
                
                best_params_rec = {
                    "model": (model_name or None),
                    "fold": int(fold),
                    "heldout_subjects": test_subs,
                    "best_params_by_r": best_params_r,
                    "best_params_by_neg_mse": best_params_mse,
                    "best_score_r": float(cv_results.loc[best_by_r_idx, 'mean_test_r']),
                    "best_score_neg_mse": float(cv_results.loc[best_by_mse_idx, 'mean_test_neg_mse']),
                    "best_params": grid.best_params_,  # Keep backward compatibility
                }
            except Exception as e:
                logger.warning(
                    f"Fold {fold}: GridSearchCV failed: {e}; falling back to default pipeline params."
                )
                best = None
        else:
            logger.info(
                f"Only one training group available in fold {fold}; skipping inner CV and fitting default pipeline params."
            )
            best = clone(pipe)
            _t0 = time.time()
            try:
                best.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Fold {fold}: default fit failed: {e}")
                y_pred_nan = np.full(len(test_idx), np.nan, dtype=float)
                return {
                    "fold": fold,
                    "y_true": y_test.tolist(),
                    "y_pred": y_pred_nan.tolist(),
                    "groups": groups[test_idx].tolist(),
                    "test_idx": test_idx.tolist(),
                    "best_params_rec": None,
                }
            logger.info(f"Fold {fold}: default fit took {time.time() - _t0:.1f}s")

        # If inner CV failed entirely, try default fit now
        if best is None:
            best = clone(pipe)
            _t0 = time.time()
            try:
                best.fit(X_train, y_train)
            except Exception as e:
                logger.warning(
                    f"Fold {fold}: fallback default fit failed after GridSearch error: {e}"
                )
                y_pred_nan = np.full(len(test_idx), np.nan, dtype=float)
                return {
                    "fold": fold,
                    "y_true": y_test.tolist(),
                    "y_pred": y_pred_nan.tolist(),
                    "groups": groups[test_idx].tolist(),
                    "test_idx": test_idx.tolist(),
                    "best_params_rec": None,
                }
            logger.info(f"Fold {fold}: fallback default fit took {time.time() - _t0:.1f}s")

        _t0 = time.time()
        try:
            y_pred = best.predict(X_test)
        except Exception as e:
            logger.warning(f"Fold {fold}: prediction failed: {e}")
            y_pred = np.full(len(test_idx), np.nan, dtype=float)
        _t_pred = time.time() - _t0
        logger.info(
            f"Fold {fold}: predict on {len(test_idx)} trials took {_t_pred:.1f}s; total fold {time.time() - _t_fold:.1f}s"
        )
        return {
            "fold": fold,
            "y_true": y_test.tolist(),
            "y_pred": np.asarray(y_pred).tolist(),
            "groups": groups[test_idx].tolist(),
            "test_idx": test_idx.tolist(),
            "best_params_rec": best_params_rec,
        }

    # Execute outer folds (optional parallel)
    if outer_n_jobs and outer_n_jobs != 1 and len(folds) > 1:
        results = Parallel(n_jobs=outer_n_jobs, prefer="threads")(  # use threads consistently
            delayed(_run_fold)(fold, tr, te) for (fold, tr, te) in folds
        )
    else:
        results = [_run_fold(fold, tr, te) for (fold, tr, te) in folds]

    # Aggregate in deterministic order
    results = sorted(results, key=lambda r: r["fold"])  # ensure deterministic order
    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    groups_ordered: List[str] = []
    test_indices_order: List[int] = []
    fold_ids: List[int] = []
    best_param_records: List[dict] = []
    for rec in results:
        y_true_all.extend(rec["y_true"]) 
        y_pred_all.extend(rec["y_pred"]) 
        groups_ordered.extend(rec["groups"]) 
        test_indices_order.extend(rec["test_idx"]) 
        fold_ids.extend([rec["fold"]] * len(rec["test_idx"]))
        if rec["best_params_rec"] is not None:
            best_param_records.append(rec["best_params_rec"])

    # Safe logging after parallel completes
    if best_params_log_path is not None and len(best_param_records) > 0:
        try:
            _ensure_dir(best_params_log_path.parent)
            with open(best_params_log_path, "a", encoding="utf-8") as f:
                for r in best_param_records:
                    f.write(json.dumps(r) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log best params: {e}")

    return np.asarray(y_true_all), np.asarray(y_pred_all), groups_ordered, test_indices_order, fold_ids


def _nested_loso_predictions_array(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pipe: Pipeline,
    param_grid: dict,
    inner_cv_splits: int,
    n_jobs: int,
    seed: int,
    best_params_log_path: Optional[Path] = None,
    model_name: str = "",
    outer_n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """Thin wrapper calling the unified LOSO implementation with array inputs."""
    return _nested_loso_predictions_unified(
        X=X,
        y=y,
        groups=groups,
        pipe=pipe,
        param_grid=param_grid,
        inner_cv_splits=inner_cv_splits,
        n_jobs=n_jobs,
        seed=seed,
        best_params_log_path=best_params_log_path,
        model_name=model_name,
        outer_n_jobs=outer_n_jobs,
    )


# -----------------------------------------------------------------------------
# Model 2: Riemannian covariance-based regression with LOSO
# -----------------------------------------------------------------------------

def _check_pyriemann() -> bool:
    try:
        import pyriemann  # noqa: F401
        from pyriemann.estimation import Covariances  # noqa: F401
        from pyriemann.tangentspace import TangentSpace  # noqa: F401
    except Exception:
        return False
    return True


def load_epochs_targets_groups(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = TASK,
) -> Tuple[List[Tuple[str, "mne.Epochs", pd.Series]], List[str]]:
    """Load cleaned epochs and aligned targets for each subject.

    Returns a list of tuples (sub_id, epochs, y) for subjects successfully loaded,
    and an empty channel list placeholder (legacy return kept for compatibility).
    Channel selection must be done fold-wise downstream to avoid leakage.
    """
    if not _HAVE_FE_HELPERS:
        # Provide minimal local fallbacks if helpers cannot be imported
        from mne_bids import BIDSPath  # type: ignore
        import pandas as _pd
        import numpy as _np

        def __fallback_find_clean_epochs_path(subj: str, task: str):
            bp = BIDSPath(
                subject=subj,
                task=task,
                datatype="eeg",
                processing="clean",
                suffix="epo",
                extension=".fif",
                root=deriv_root,
                check=False,
            )
            p = bp.fpath
            if p and Path(p).exists():
                return p
            p2 = deriv_root / f"sub-{subj}" / "eeg" / f"sub-{subj}_task-{task}_proc-clean_epo.fif"
            if p2.exists():
                return p2
            subj_dir = deriv_root / f"sub-{subj}"
            for c in sorted(subj_dir.rglob(f"sub-{subj}_task-{task}*epo.fif")):
                return c
            return None

        def __fallback_load_events_df(subj: str, task: str):
            ebp = BIDSPath(
                subject=subj,
                task=task,
                datatype="eeg",
                suffix="events",
                extension=".tsv",
                root=Path(deriv_root).parents[1],  # bids_root
                check=False,
            )
            p = ebp.fpath
            if p is None:
                p = Path(deriv_root).parents[1] / f"sub-{subj}" / "eeg" / f"sub-{subj}_task-{task}_events.tsv"
            if p.exists():
                try:
                    return _pd.read_csv(p, sep="\t")
                except Exception:
                    return None
            return None

        def __fallback_align_events_to_epochs(events_df, epochs):
            if events_df is None:
                return None
            sel = getattr(epochs, "selection", None)
            if sel is not None and len(sel) == len(epochs):
                try:
                    if len(events_df) > int(_np.max(sel)):
                        return events_df.iloc[sel].reset_index(drop=True)
                except Exception:
                    pass
            if "sample" in events_df.columns and isinstance(getattr(epochs, "events", None), _np.ndarray):
                try:
                    samples = epochs.events[:, 0]
                    out = events_df.set_index("sample").reindex(samples)
                    if len(out) == len(epochs) and not out.isna().all(axis=1).any():
                        return out.reset_index()
                except Exception:
                    pass
            n = min(len(events_df), len(epochs))
            if n == 0:
                return None
            return events_df.iloc[:n].reset_index(drop=True)

        def __fallback_pick_target_column(df):
            candidates = [
                "vas_final_coded_rating",
                "vas_final_rating",
                "vas_rating",
                "pain_intensity",
                "pain_rating",
                "rating",
                "pain_binary_coded",
                "pain_binary",
                "pain",
            ]
            for c in candidates:
                if c in df.columns:
                    return c
            for c in df.columns:
                cl = c.lower()
                if ("vas" in cl or "rating" in cl) and df[c].dtype != "O":
                    return c
            return None

        # Select fallbacks
        find_clean_epochs_path = __fallback_find_clean_epochs_path
        load_events_df_local = __fallback_load_events_df
        align_events_to_epochs_local = __fallback_align_events_to_epochs
        pick_target_column_local = __fallback_pick_target_column
    else:
        # Use helpers imported from 03_feature_engineering.py at module scope
        find_clean_epochs_path = _find_clean_epochs_path  # type: ignore[name-defined]
        load_events_df_local = _load_events_df  # type: ignore[name-defined]
        align_events_to_epochs_local = _align_events_to_epochs  # type: ignore[name-defined]
        pick_target_column_local = _pick_target_column  # type: ignore[name-defined]

    import mne  # local import to limit overhead when only running RF model

    if subjects is None or subjects == ["all"]:
        # prefer those with features to ensure targets exist
        subjects = _collect_subject_ids_with_features(deriv_root)
        logger.info(f"Detected {len(subjects)} subjects with features for epochs loading.")

    out: List[Tuple[str, mne.Epochs, pd.Series]] = []
    ch_sets: List[set] = []

    for s in subjects:
        sub = f"sub-{s}"
        epo_path = find_clean_epochs_path(s, task)
        if epo_path is None or not Path(epo_path).exists():
            logger.warning(f"Clean epochs not found for {sub}; skipping.")
            continue
        try:
            # Preload data so that subsequent channel picking/reordering is allowed by MNE
            epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
        except Exception as e:
            logger.warning(f"Failed to read epochs for {sub}: {e}")
            continue

        # Ensure a montage is set so that interpolation has proper sensor geometry
        try:
            try:
                epochs.set_montage(mne.channels.make_standard_montage("standard_1005"))
            except Exception:
                epochs.set_montage(mne.channels.make_standard_montage("standard_1020"))
        except Exception:
            pass

        # Interpolate only pre-marked bad channels (no cross-subject channel addition here)
        try:
            if len(epochs.info.get("bads", [])) > 0:
                epochs.interpolate_bads(reset_bads=True)
        except Exception as e:
            logger.warning(f"{sub}: interpolation failed: {e}")

        events_df = load_events_df_local(s, task)
        aligned = align_events_to_epochs_local(events_df, epochs)
        if aligned is None or len(aligned) == 0:
            logger.warning(f"No aligned events/targets for {sub}; skipping.")
            continue
        tgt_col = pick_target_column_local(aligned)
        if tgt_col is None:
            logger.warning(f"No suitable target column for {sub}; skipping.")
            continue
        y = pd.to_numeric(aligned[tgt_col], errors="coerce")
        # Align lengths with epochs
        n = min(len(epochs), len(y))
        if n == 0:
            logger.warning(f"No trials after alignment for {sub}; skipping.")
            continue
        if len(epochs) != n:
            epochs = epochs[:n]
        y = y.iloc[:n]

        out.append((sub, epochs, y))
        ch_sets.append(set([ch for ch in epochs.info["ch_names"] if epochs.get_channel_types(picks=[ch])[0] == "eeg"]))

    if not out:
        raise RuntimeError("No epochs + targets could be loaded for any subject.")

    # Do not compute a global intersection to avoid leakage in downstream CV.
    # Keep legacy return signature but provide an empty list.
    return out, []


def _align_epochs_to_pivot_chs(epochs: "mne.Epochs", pivot_chs: List[str]) -> "mne.Epochs":
    """Align an Epochs object to a given pivot EEG channel list.

    This adds any missing channels as zeros, marks them bad, interpolates using the current
    montage, and finally picks/reorders to exactly `pivot_chs`.
    """
    import mne  # local import
    # Current EEG channel names
    eeg_chs_now = [ch for ch in epochs.info["ch_names"] if epochs.get_channel_types(picks=[ch])[0] == "eeg"]
    missing = [ch for ch in pivot_chs if ch not in eeg_chs_now]
    
    # Only copy if we need to modify the epochs
    if len(missing) > 0:
        ep = epochs.copy()
        try:
            n_ep = len(ep)
            n_times = len(ep.times)
            data_missing = np.zeros((n_ep, len(missing), n_times), dtype=np.float64)
            info_missing = mne.create_info(
                ch_names=missing,
                sfreq=ep.info["sfreq"],
                ch_types="eeg",
                verbose=False,
            )
            # Avoid copying events/event_id unless necessary
            events = getattr(ep, "events", None)
            event_id = getattr(ep, "event_id", None)
            epochs_missing = mne.EpochsArray(
                data_missing,
                info_missing,
                events=events if events is not None else None,
                event_id=event_id if event_id is not None else None,
                tmin=ep.tmin,
                verbose=False,
            )
            ep = mne.concatenate_epochs([ep, epochs_missing], verbose=False)
            
            # Mark missing channels as bad and interpolate
            ep.info['bads'] = list(set(ep.info.get('bads', [])) | set(missing))
            try:
                ep.interpolate_bads(reset_bads=True)
            except Exception:
                pass
        except Exception:
            pass
    else:
        ep = epochs  # Use reference if no missing channels
    
    # Finally pick and reorder to the pivot channel list (dropping any extras)
    try:
        # Check if we need to copy for picking
        if set(ep.info["ch_names"]) != set(pivot_chs) or ep.info["ch_names"] != pivot_chs:
            ep = ep.copy().pick(pivot_chs)
    except Exception:
        # As a fallback, intersect then reorder
        present = [ch for ch in pivot_chs if ch in ep.info["ch_names"]]
        if ep.info["ch_names"] != present:
            ep = ep.copy().pick(present)
            try:
                if ep.info["ch_names"] != present:
                    ep.reorder_channels(present)
            except Exception:
                pass
    return ep


@_validate_inputs
def loso_riemann_regression(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = TASK,
    results_dir: Path = None,
    n_jobs: int = -1,
    seed: int = 42,
    outer_n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, dict, pd.DataFrame]:
    if not _check_pyriemann():
        raise ImportError(
            "pyriemann is not installed. Install with `pip install pyriemann` to run Model 2."
        )

    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    # Load epochs and targets (no global channel selection)
    tuples, _ = load_epochs_targets_groups(deriv_root, subjects=subjects, task=task)

    # Build trial index, y and groups without fixing channels
    trial_records: List[Tuple[str, int]] = []  # (subject_id, trial_idx)
    y_all_list: List[float] = []
    groups_list: List[str] = []
    subj_to_epochs: Dict[str, "mne.Epochs"] = {}
    subj_to_y: Dict[str, pd.Series] = {}
    for sub, epochs, y in tuples:
        n = min(len(epochs), len(y))
        if n == 0:
            continue
        subj_to_epochs[sub] = epochs
        subj_to_y[sub] = pd.to_numeric(y.iloc[:n], errors="coerce")
        for ti in range(n):
            trial_records.append((sub, ti))
            y_all_list.append(float(subj_to_y[sub].iloc[ti]))
            groups_list.append(sub)

    if len(trial_records) == 0:
        raise RuntimeError("No trial data available.")

    y_all_arr = np.asarray(y_all_list)
    groups_arr = np.asarray(groups_list)

    # Guard against too few subjects
    if len(np.unique(groups_arr)) < 2:
        raise RuntimeError("Need at least 2 subjects for LOSO.")

    # Define Riemann pipeline and param grid
    pipe = Pipeline(steps=[
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("ridge", Ridge()),
    ])
    param_grid = {
        "cov__estimator": ["oas", "lwf"],
        "ridge__alpha": [1e-3, 1e-2, 1e-1, 1, 10],
    }

    logo = LeaveOneGroupOut()
    folds = list(enumerate(logo.split(np.arange(len(trial_records)), groups=groups_arr), start=1))

    # Outer loop (optionally parallel)
    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
        np.random.seed(seed + fold)
        pyrandom.seed(seed + fold)

        # Find intersection of EEG channels across all training subjects in this fold
        train_subs_seq = [trial_records[i][0] for i in train_idx]
        train_subjects = list({s for s in train_subs_seq if s is not None})
        
        # Get EEG channel names for each training subject
        train_subject_eeg_chs = {}
        for s in train_subjects:
            train_subject_eeg_chs[s] = [
                ch for ch in subj_to_epochs[s].info["ch_names"]
                if subj_to_epochs[s].get_channel_types(picks=[ch])[0] == "eeg"
            ]
        
        # Find intersection of channels across all training subjects
        if len(train_subjects) == 1:
            common_chs = train_subject_eeg_chs[train_subjects[0]]
        else:
            common_chs = list(set.intersection(*[set(train_subject_eeg_chs[s]) for s in train_subjects]))
        
        if not common_chs:
            logger.warning(f"Fold {fold}: No common EEG channels across training subjects. Skipping fold.")
            return {
                "fold": fold,
                "y_true": y_all_arr[test_idx].tolist(),
                "y_pred": np.full(len(test_idx), np.nan, dtype=float).tolist(),
                "groups": groups_arr[test_idx].tolist(),
                "test_idx": test_idx.tolist(),
                "best_params_rec": None,
            }
        
        # Sort channels for consistency
        common_chs = sorted(common_chs)
        logger.info(f"Fold {fold}: Using {len(common_chs)} common EEG channels across {len(train_subjects)} training subjects.")
        
        # Project all subjects (train and test) onto the EXACT common channel space
        # Critical: Enforce identical channel dimensionality for TangentSpace consistency
        subjects_in_fold = list({trial_records[i][0] for i in np.concatenate([train_idx, test_idx])})
        aligned_epochs: Dict[str, "mne.Epochs"] = {}
        for s in subjects_in_fold:
            # Check if subject has ALL required channels from training intersection
            subject_chs = subj_to_epochs[s].info["ch_names"]
            missing_chs = [ch for ch in common_chs if ch not in subject_chs]
            
            if missing_chs:
                logger.error(f"Fold {fold}: Subject {s} missing {len(missing_chs)}/{len(common_chs)} required channels: {missing_chs[:5]}{'...' if len(missing_chs) > 5 else ''}")
                logger.error(f"Fold {fold}: Cannot maintain tangent-space dimensionality consistency. Skipping fold.")
                return {
                    "fold": fold,
                    "y_true": y_all_arr[test_idx].tolist(),
                    "y_pred": np.full(len(test_idx), np.nan, dtype=float).tolist(),
                    "groups": groups_arr[test_idx].tolist(),
                    "test_idx": test_idx.tolist(),
                    "best_params_rec": None,
                }
            
            # All channels present - safe to pick exact training intersection
            try:
                aligned_epochs[s] = subj_to_epochs[s].copy().pick(common_chs)
            except Exception as e:
                logger.error(f"Fold {fold}: Failed to pick channels for subject {s} despite availability check: {e}")
                return {
                    "fold": fold,
                    "y_true": y_all_arr[test_idx].tolist(),
                    "y_pred": np.full(len(test_idx), np.nan, dtype=float).tolist(),
                    "groups": groups_arr[test_idx].tolist(),
                    "test_idx": test_idx.tolist(),
                    "best_params_rec": None,
                }

        # Assemble X_train, X_test keeping (n_trials, n_ch, n_times)
        # Drop NaN targets only
        y_train = y_all_arr[train_idx]
        y_test = y_all_arr[test_idx]
        train_sel = np.isfinite(y_train)
        test_sel = np.isfinite(y_test)
        train_idx_f = train_idx[train_sel]
        test_idx_f = test_idx[test_sel]
        y_train = y_train[train_sel]
        y_test = y_test[test_sel]

        def _extract_block(indices: np.ndarray) -> np.ndarray:
            X_list: List[np.ndarray] = []
            for i in indices:
                sub_i, ti = trial_records[int(i)]
                # MNE version compatibility: reject_by_annotation parameter removed in newer versions
                try:
                    X_i = aligned_epochs[sub_i].get_data(picks="eeg", reject_by_annotation=None)[ti]
                except TypeError:
                    X_i = aligned_epochs[sub_i].get_data(picks="eeg")[ti]
                X_list.append(X_i)
            return np.stack(X_list, axis=0)

        X_train = _extract_block(train_idx_f)
        X_test = _extract_block(test_idx_f)

        train_groups = groups_arr[train_idx_f]

        # Inner CV on training groups
        n_unique = len(np.unique(train_groups))
        best = None
        best_params_rec = None
        # Use multi-metric scoring with signed Pearson r as primary to avoid sign-flipped models
        scoring = {'r': _make_pearsonr_scorer(), 'neg_mse': 'neg_mean_squared_error'}
        refit_metric = 'r'
        if n_unique >= 2:
            n_splits_inner = int(np.clip(int(CONFIG["cv"]["inner_splits"]), 2, n_unique))
            inner_cv = GroupKFold(n_splits=n_splits_inner)
            try:
                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=inner_cv,
                    n_jobs=n_jobs,
                    refit=refit_metric,
                )
                _t0 = time.time()
                gs.fit(X_train, y_train, groups=train_groups)
                best = gs.best_estimator_
                
                # Extract best parameters for both metrics
                cv_results = pd.DataFrame(gs.cv_results_)
                best_by_r_idx = cv_results['rank_test_r'].idxmin()
                best_by_mse_idx = cv_results['rank_test_neg_mse'].idxmin()
                
                best_params_r = cv_results.loc[best_by_r_idx, 'params']
                best_params_mse = cv_results.loc[best_by_mse_idx, 'params']
                
                best_params_rec = {
                    "fold": fold,
                    "model": "Riemann",
                    "best_params_by_r": best_params_r,
                    "best_params_by_neg_mse": best_params_mse,
                    "best_score_r": float(cv_results.loc[best_by_r_idx, 'mean_test_r']),
                    "best_score_neg_mse": float(cv_results.loc[best_by_mse_idx, 'mean_test_neg_mse']),
                    "best_params": gs.best_params_,  # Keep backward compatibility
                }
                logger.info(f"Fold {fold}: best params by r = {best_params_r}")
                logger.info(f"Fold {fold}: best params by neg_mse = {best_params_mse}")
            except Exception as e:
                logger.warning(f"Fold {fold}: GridSearchCV failed: {e}; falling back to default pipeline params.")
                best = None
        else:
            best = clone(pipe)
            try:
                best.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Fold {fold}: default fit failed: {e}")
                return {
                    "fold": fold,
                    "y_true": y_test.tolist(),
                    "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                    "groups": groups_arr[test_idx_f].tolist(),
                    "test_idx": test_idx_f.tolist(),
                    "best_params_rec": None,
                }

        if best is None:
            best = clone(pipe)
            try:
                best.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Fold {fold}: fallback default fit failed after GridSearch error: {e}")
                return {
                    "fold": fold,
                    "y_true": y_test.tolist(),
                    "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                    "groups": groups_arr[test_idx_f].tolist(),
                    "test_idx": test_idx_f.tolist(),
                    "best_params_rec": None,
                }

        try:
            y_pred = best.predict(X_test)
        except Exception as e:
            logger.warning(f"Fold {fold}: prediction failed: {e}")
            y_pred = np.full(len(y_test), np.nan, dtype=float)

        return {
            "fold": fold,
            "y_true": y_test.tolist(),
            "y_pred": np.asarray(y_pred).tolist(),
            "groups": groups_arr[test_idx_f].tolist(),
            "test_idx": test_idx_f.tolist(),
            "best_params_rec": best_params_rec,
        }

    if outer_n_jobs and outer_n_jobs != 1 and len(folds) > 1:
        results = Parallel(n_jobs=outer_n_jobs, prefer="threads")(delayed(_run_fold)(fold, tr, te) for (fold, (tr, te)) in folds)
    else:
        results = [_run_fold(fold, tr, te) for (fold, (tr, te)) in folds]

    # Aggregate results
    results = sorted(results, key=lambda r: r["fold"])  # ensure deterministic order
    y_true, y_pred, groups_ordered, test_indices_order, fold_ids = [], [], [], [], []
    best_param_records: List[dict] = []
    for rec in results:
        y_true.extend(rec["y_true"])  # type: ignore
        y_pred.extend(rec["y_pred"])  # type: ignore
        groups_ordered.extend(rec["groups"])  # type: ignore
        test_indices_order.extend(rec["test_idx"])  # type: ignore
        fold_ids.extend([rec["fold"]] * len(rec["test_idx"]))  # type: ignore
        if rec.get("best_params_rec") is not None:
            best_param_records.append(rec["best_params_rec"])  # type: ignore

    # Log best params JSONL if requested
    if results_dir is not None and len(best_param_records) > 0:
        try:
            _ensure_dir(results_dir)
            path = results_dir / CONFIG["paths"]["best_params"]["riemann_loso"]
            _ensure_dir(path.parent)
            with open(path, "a", encoding="utf-8") as f:
                for r in best_param_records:
                    f.write(json.dumps(r) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log best params: {e}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pooled, per_subj = compute_metrics(y_true, y_pred, np.asarray(groups_ordered))
    logger.info(
        f"Model 2 (Riemann) pooled: r={pooled['pearson_r']:.3f}, R2={pooled['r2']:.3f}, EVS={pooled['explained_variance']:.3f}, "
        f"avg_r_Fz={pooled['avg_subject_r_fisher_z']:.3f}"
    )

    if results_dir is not None:
        _ensure_dir(results_dir)
        pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "group": np.asarray(groups_ordered),
            "fold": fold_ids,
            "trial_index": test_indices_order,
        }).to_csv(results_dir / CONFIG["paths"]["predictions"]["riemann_loso"], sep="\t", index=False)

        per_subj.to_csv(results_dir / CONFIG["paths"]["per_subject_metrics"]["riemann_loso"], sep="\t", index=False)

        # Save Riemann LOSO test indices explicitly
        try:
            pd.DataFrame({
                "group": np.asarray(groups_ordered),
                "fold": fold_ids,
                "trial_index": test_indices_order,
            }).to_csv(results_dir / CONFIG["paths"]["indices"]["riemann_loso"], sep="\t", index=False)
        except Exception as e:
            logger.warning(f"Failed to save Riemann LOSO indices: {e}")

    return y_true, y_pred, pooled, per_subj


# -----------------------------------------------------------------------------
# Riemannian covariance-based analyses: visualizations, bands, sliding windows
# -----------------------------------------------------------------------------

def riemann_visualize_cov_bins(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = TASK,
    plots_dir: Path = None,
    plateau_window: Optional[Tuple[float, float]] = (3.0, 10.5),
) -> None:
    """Visualize mean covariance difference (high pain - low pain) and node strength topomap.

    Bins are defined by pooled tertiles across all included trials: low ≤ 33rd pct, high ≥ 66th pct.
    Covariance means use Riemannian (geometric) mean per bin per subject, then group mean (riemann) across subjects.
    
    NOTE: For visualization purposes, the common channel set is determined globally across all subjects,
    which differs from the strict per-fold channel selection used in the modeling pipeline. This represents
    a minor form of data leakage acceptable for exploratory visualization but not for inference.
    """
    if plots_dir is None:
        return
    if not _check_pyriemann():
        logger.warning("pyriemann not installed; skipping Riemann visualizations.")
        return
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.utils.mean import mean_riemann
    except Exception:
        logger.warning("pyriemann unavailable for visualization; skipping.")
        return

    tuples, _ = load_epochs_targets_groups(deriv_root, subjects=subjects, task=task)

    # VISUALIZATION-ONLY: Determine a global common EEG channel set across ALL loaded subjects
    # NOTE: This represents minor data leakage (using global info) but is acceptable for visualization.
    # The modeling pipeline uses strict per-fold channel selection to avoid this leakage.
    eeg_sets = []
    for _sub, _epochs, _y in tuples:
        try:
            eeg_chs = [ch for ch in _epochs.info["ch_names"] if _epochs.get_channel_types(picks=[ch])[0] == "eeg"]
            eeg_sets.append(set(eeg_chs))
        except Exception:
            continue
    if len(eeg_sets) == 0:
        logger.warning("No subjects found for Riemann visualization.")
        return
    local_common_chs = sorted(list(set.intersection(*eeg_sets))) if len(eeg_sets) > 1 else sorted(list(eeg_sets[0]))
    if len(local_common_chs) == 0:
        logger.warning("No common EEG channels across subjects; skipping visualization.")
        return
    # Establish canonical order from the first subject's EEG channel order filtered to local_common_chs
    try:
        first_ep = tuples[0][1]
        first_eeg_order = [ch for ch in first_ep.info["ch_names"] if first_ep.get_channel_types(picks=[ch])[0] == "eeg"]
        canonical_chs: Optional[List[str]] = [ch for ch in first_eeg_order if ch in set(local_common_chs)]
    except Exception:
        canonical_chs = local_common_chs

    # Collect per-subject arrays and targets
    subj_data = []  # (sub, X[n_trials,n_ch,n_times], y[n_trials], info)
    info_for_plot = None
    for sub, epochs, y in tuples:
        epochs_use = epochs.copy().pick(canonical_chs)
        # Enforce the canonical channel order across all subjects
        try:
            epochs_use.reorder_channels(canonical_chs)
        except Exception:
            # If reorder fails (name mismatch), fall back to picking again then continue
            epochs_use = epochs_use.pick(canonical_chs)
        # Strict sanity check: skip this subject if channel count does not match canonical list
        if canonical_chs is not None and len(epochs_use.info["ch_names"]) != len(canonical_chs):
            logger.warning(f"Skipping {sub}: channel count {len(epochs_use.info['ch_names'])} != canonical {len(canonical_chs)}")
            continue
        if plateau_window is not None:
            tmin, tmax = plateau_window
            try:
                epochs_use.crop(tmin=tmin, tmax=tmax)
            except Exception:
                pass
        X = epochs_use.get_data(picks="eeg")
        n = min(len(X), len(y))
        if n < 2:
            continue
        yv = pd.to_numeric(y.iloc[:n], errors="coerce").to_numpy()
        subj_data.append((sub, X[:n], yv, epochs_use.info))
        if info_for_plot is None:
            # Use first subject's info (already in canonical order) for topomap positions
            info_for_plot = epochs_use.info

    if not subj_data:
        logger.warning("No subject data available for Riemann visualization.")
        return

    y_all = np.concatenate([v[2] for v in subj_data])
    y_all = y_all[np.isfinite(y_all)]
    if len(y_all) < 4:
        logger.warning("Too few valid ratings for binning; skipping visualization.")
        return
    q_low = float(np.percentile(y_all, 33.3))
    q_high = float(np.percentile(y_all, 66.7))

    cov_means_low = []
    cov_means_high = []
    for sub, X, yv, _info in subj_data:
        mask_low = np.isfinite(yv) & (yv <= q_low)
        mask_high = np.isfinite(yv) & (yv >= q_high)
        if mask_low.sum() >= 2:
            cov_low = Covariances(estimator="oas").transform(X[mask_low])
            C_low = mean_riemann(cov_low)
            cov_means_low.append(C_low)
        if mask_high.sum() >= 2:
            cov_high = Covariances(estimator="oas").transform(X[mask_high])
            C_high = mean_riemann(cov_high)
            cov_means_high.append(C_high)

    if len(cov_means_low) == 0 or len(cov_means_high) == 0:
        logger.warning("Insufficient trials in bins for visualization.")
        return

    M_low = mean_riemann(np.stack(cov_means_low))
    M_high = mean_riemann(np.stack(cov_means_high))
    D = M_high - M_low

    # Use the canonical channel order (exact names and order) for plotting and labels
    ch_names = canonical_chs
    # Defensive check before building Info
    if D.shape[0] != len(ch_names):
        # Add diagnostic dump of first 5 channel names to speed debugging mismatches
        ch_preview = ch_names[:5] if ch_names else []
        logger.warning(
            f"Skipping Riemann topomap: data channels (D:{D.shape[0]}) != ch_names ({len(ch_names)}). "
            f"First 5 ch_names: {ch_preview}"
        )
        return
    # Always rebuild a clean Info to avoid any hidden extra channels
    info_for_plot = mne.create_info(ch_names, sfreq=1000.0, ch_types="eeg")
    try:
        # Prefer a richer montage to cover more 64-ch systems; fallback to 10-20
        try:
            info_for_plot.set_montage(mne.channels.make_standard_montage("standard_1005"))
        except Exception:
            info_for_plot.set_montage(mne.channels.make_standard_montage("standard_1020"))
    except Exception:
        pass

    # Heatmap of difference matrix
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5), dpi=150)
    try:
        sns.heatmap(
            D, ax=ax, cmap="RdBu_r", center=0.0, square=True,
            xticklabels=ch_names, yticklabels=ch_names, cbar_kws={"label": "Δ covariance (high - low)"}
        )
        ax.set_title("Mean covariance difference (high - low pain)\n(Global channel set for visualization)")
        plt.tight_layout()
        plt.savefig(plots_dir / "riemann_cov_diff_matrix.png")
    finally:
        plt.close(fig)

    # Node strength topomap (sum |off-diagonal| per node)
    A = D.copy()
    np.fill_diagonal(A, 0.0)
    node_strength = np.sum(np.abs(A), axis=1)
    # Info already rebuilt above to match exactly the vector length and order

    fig2, ax2 = plt.subplots(1, 1, figsize=(4.5, 4.2), dpi=150)
    try:
        mne.viz.plot_topomap(node_strength, info_for_plot, axes=ax2, show=False, contours=6, cmap="Reds")
        ax2.set_title("Node strength |Δcov| (high - low)\n(Global channel set for visualization)")
        plt.tight_layout()
        plt.savefig(plots_dir / "riemann_node_strength_topomap.png")
    finally:
        plt.close(fig2)


def run_riemann_band_limited_decoding(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = TASK,
    results_dir: Path = None,
    plots_dir: Path = None,
    bands: Optional[List[Tuple[float, float]]] = None,
    n_jobs: int = -1,
    seed: int = 42,
    outer_n_jobs: int = 1,
) -> Optional[dict]:
    """Run LOSO Riemann regression per frequency band; return and save summary dict."""
    if not _check_pyriemann():
        logger.warning("pyriemann not installed; skipping band-limited Riemann decoding.")
        return None
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
    except Exception:
        logger.warning("pyriemann unavailable; skipping band-limited decoding.")
        return None

    if bands is None:
        bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]

    tuples, _ = load_epochs_targets_groups(deriv_root, subjects=subjects, task=task)

    summary = {}
    r_vals = []
    labels = []

    # Prepare global trial index to drive LOSO folds (bands handled inside folds)
    trial_records: List[Tuple[str, int]] = []
    y_all_list: List[float] = []
    groups_list: List[str] = []
    subj_to_epochs: Dict[str, "mne.Epochs"] = {}
    subj_to_y: Dict[str, pd.Series] = {}
    for sub, epochs, y in tuples:
        n = min(len(epochs), len(y))
        if n == 0:
            continue
        subj_to_epochs[sub] = epochs
        subj_to_y[sub] = pd.to_numeric(y.iloc[:n], errors="coerce")
        for ti in range(n):
            trial_records.append((sub, ti))
            y_all_list.append(float(subj_to_y[sub].iloc[ti]))
            groups_list.append(sub)
    if len(trial_records) == 0:
        logger.warning("No trials available for band-limited decoding.")
        return None

    y_all_arr = np.asarray(y_all_list)
    groups_arr = np.asarray(groups_list)
    logo = LeaveOneGroupOut()
    folds = list(enumerate(logo.split(np.arange(len(trial_records)), groups=groups_arr), start=1))

    for (l_freq, h_freq) in bands:
        label = f"{int(l_freq)}-{int(h_freq)}Hz"
        labels.append(label)

        pipe = Pipeline(steps=[
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("ridge", Ridge()),
        ])
        param_grid = {
            "cov__estimator": ["oas", "lwf"],
            "ridge__alpha": [1e-3, 1e-2, 1e-1, 1, 10],
        }

        def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
            np.random.seed(seed + fold)
            pyrandom.seed(seed + fold)
            # Find intersection of EEG channels across all training subjects in this fold
            train_subs_seq = [trial_records[i][0] for i in train_idx]
            train_subjects = list({s for s in train_subs_seq if s is not None})
            
            # Get EEG channel names for each training subject
            train_subject_eeg_chs = {}
            for s in train_subjects:
                train_subject_eeg_chs[s] = [
                    ch for ch in subj_to_epochs[s].info["ch_names"]
                    if subj_to_epochs[s].get_channel_types(picks=[ch])[0] == "eeg"
                ]
            
            # Find intersection of channels across all training subjects
            if len(train_subjects) == 1:
                common_chs = train_subject_eeg_chs[train_subjects[0]]
            else:
                common_chs = list(set.intersection(*[set(train_subject_eeg_chs[s]) for s in train_subjects]))
            
            if not common_chs:
                logger.warning(f"Band {label} fold {fold}: No common EEG channels across training subjects. Skipping fold.")
                return {
                    "fold": fold,
                    "y_true": y_test.tolist(),
                    "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                    "groups": groups_arr[test_idx_f].tolist(),
                    "test_idx": test_idx_f.tolist(),
                    "best_params_rec": None,
                }
            
            # Sort channels for consistency
            common_chs = sorted(common_chs)
            logger.info(f"Band {label} fold {fold}: Using {len(common_chs)} common EEG channels across {len(train_subjects)} training subjects.")
            
            # Separate train/test subjects to optimize memory usage
            test_subjects = list({trial_records[i][0] for i in test_idx})
            all_subjects = list(set(train_subjects + test_subjects))
            
            # Separate caches for train/test to reduce memory footprint
            train_data_cache: Dict[str, np.ndarray] = {}
            test_data_cache: Dict[str, np.ndarray] = {}
            
            for s in all_subjects:
                # Enforce EXACT common channel set for tangent-space consistency
                subject_chs = subj_to_epochs[s].info["ch_names"]
                missing_chs = [ch for ch in common_chs if ch not in subject_chs]
                
                if missing_chs:
                    logger.error(f"Band {label} fold {fold}: Subject {s} missing {len(missing_chs)}/{len(common_chs)} required channels: {missing_chs[:5]}{'...' if len(missing_chs) > 5 else ''}")
                    logger.error(f"Band {label} fold {fold}: Cannot maintain tangent-space dimensionality. Skipping fold.")
                    return {
                        "fold": fold,
                        "y_true": y_test.tolist(),
                        "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                        "groups": groups_arr[test_idx_f].tolist(),
                        "test_idx": test_idx_f.tolist(),
                        "best_params_rec": None,
                    }
                
                # All channels present - safe to pick exact training intersection
                try:
                    ep = subj_to_epochs[s].copy().pick(common_chs)
                except Exception as e:
                    logger.error(f"Band {label} fold {fold}: Failed to pick channels for subject {s} despite availability check: {e}")
                    return {
                        "fold": fold,
                        "y_true": y_test.tolist(),
                        "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                        "groups": groups_arr[test_idx_f].tolist(),
                        "test_idx": test_idx_f.tolist(),
                        "best_params_rec": None,
                    }
                # Log bandpass filtering details for reproducibility
                logger.info(f"Band {label} fold {fold}: applying {l_freq}-{h_freq}Hz filter to subject {s} (sfreq={ep.info['sfreq']:.1f}Hz)")
                try:
                    ep = ep.copy()
                    ep.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)
                    # Log realized passband after filtering
                    logger.info(f"Band {label} fold {fold}: subject {s} filtered to {l_freq}-{h_freq}Hz passband")
                except Exception as e:
                    logger.warning(f"Band {label} fold {fold}: filtering failed for subject {s}: {e}")
                    
                # Pre-extract all data for this subject once, convert to float32 for memory efficiency
                try:
                    data = ep.get_data(picks="eeg", reject_by_annotation=None)
                except TypeError:
                    data = ep.get_data(picks="eeg")
                
                # Convert to float32 before covariance computation to reduce memory usage
                data = data.astype(np.float32)
                
                # Store in appropriate cache based on train/test membership
                if s in train_subjects:
                    train_data_cache[s] = data
                if s in test_subjects:
                    test_data_cache[s] = data

            # Drop NaN targets only
            y_train = y_all_arr[train_idx]
            y_test = y_all_arr[test_idx]
            train_sel = np.isfinite(y_train)
            test_sel = np.isfinite(y_test)
            train_idx_f = train_idx[train_sel]
            test_idx_f = test_idx[test_sel]
            y_train = y_train[train_sel]
            y_test = y_test[test_sel]

            def _extract_block(indices: np.ndarray) -> np.ndarray:
                X_list: List[np.ndarray] = []
                for i in indices:
                    sub_i, ti = trial_records[int(i)]
                    # Use appropriate cache based on whether this is for training or testing
                    if i in train_idx_f:
                        X_i = train_data_cache[sub_i][ti]
                    else:
                        X_i = test_data_cache[sub_i][ti]
                    X_list.append(X_i)
                return np.stack(X_list, axis=0)

            X_train = _extract_block(train_idx_f)
            X_test = _extract_block(test_idx_f)
            train_groups = groups_arr[train_idx_f]

            # Inner CV
            n_unique = len(np.unique(train_groups))
            best = None
            best_params_rec = None
            # Use multi-metric scoring with signed Pearson r as primary to avoid sign-flipped models
            scoring = {'r': _make_pearsonr_scorer(), 'neg_mse': 'neg_mean_squared_error'}
            refit_metric = 'r'
            if n_unique >= 2:
                n_splits_inner = int(np.clip(int(CONFIG["cv"]["inner_splits"]), 2, n_unique))
                inner_cv = GroupKFold(n_splits=n_splits_inner)
                try:
                    gs = GridSearchCV(
                        estimator=pipe,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=inner_cv,
                        n_jobs=n_jobs,
                        refit=refit_metric,
                    )
                    gs.fit(X_train, y_train, groups=train_groups)
                    best = gs.best_estimator_
                    
                    # Extract best parameters for both metrics
                    cv_results = pd.DataFrame(gs.cv_results_)
                    best_by_r_idx = cv_results['rank_test_r'].idxmin()
                    best_by_mse_idx = cv_results['rank_test_neg_mse'].idxmin()
                    
                    best_params_r = cv_results.loc[best_by_r_idx, 'params']
                    best_params_mse = cv_results.loc[best_by_mse_idx, 'params']
                    
                    best_params_rec = {
                        "fold": fold,
                        "model": f"Riemann_{label}",
                        "best_params_by_r": best_params_r,
                        "best_params_by_neg_mse": best_params_mse,
                        "best_score_r": float(cv_results.loc[best_by_r_idx, 'mean_test_r']),
                        "best_score_neg_mse": float(cv_results.loc[best_by_mse_idx, 'mean_test_neg_mse']),
                        "best_params": gs.best_params_,  # Keep backward compatibility
                    }
                except Exception as e:
                    logger.warning(f"Band {label} fold {fold}: GridSearchCV failed: {e}; using defaults.")
                    best = None
            else:
                best = clone(pipe)
                try:
                    best.fit(X_train, y_train)
                except Exception as e:
                    logger.warning(f"Band {label} fold {fold}: default fit failed: {e}")
                    return {
                        "fold": fold,
                        "y_true": y_test.tolist(),
                        "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                        "groups": groups_arr[test_idx_f].tolist(),
                        "test_idx": test_idx_f.tolist(),
                        "best_params_rec": None,
                    }

            if best is None:
                best = clone(pipe)
                try:
                    best.fit(X_train, y_train)
                except Exception as e:
                    logger.warning(f"Band {label} fold {fold}: fallback default fit failed: {e}")
                    return {
                        "fold": fold,
                        "y_true": y_test.tolist(),
                        "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                        "groups": groups_arr[test_idx_f].tolist(),
                        "test_idx": test_idx_f.tolist(),
                        "best_params_rec": None,
                    }

            try:
                y_pred = best.predict(X_test)
            except Exception as e:
                logger.warning(f"Band {label} fold {fold}: prediction failed: {e}")
                y_pred = np.full(len(y_test), np.nan, dtype=float)

            return {
                "fold": fold,
                "y_true": y_test.tolist(),
                "y_pred": np.asarray(y_pred).tolist(),
                "groups": groups_arr[test_idx_f].tolist(),
                "test_idx": test_idx_f.tolist(),
                "best_params_rec": best_params_rec,
            }

        if outer_n_jobs and outer_n_jobs != 1 and len(folds) > 1:
            results = Parallel(n_jobs=outer_n_jobs, prefer="threads")(delayed(_run_fold)(fold, tr, te) for (fold, (tr, te)) in folds)
        else:
            results = [_run_fold(fold, tr, te) for (fold, (tr, te)) in folds]

        # Aggregate and compute metrics
        results = sorted(results, key=lambda r: r["fold"])  # type: ignore
        y_true_b, y_pred_b, groups_ordered, test_indices_order, fold_ids = [], [], [], [], []
        best_param_records: List[dict] = []
        for rec in results:
            y_true_b.extend(rec["y_true"])  # type: ignore
            y_pred_b.extend(rec["y_pred"])  # type: ignore
            groups_ordered.extend(rec["groups"])  # type: ignore
            test_indices_order.extend(rec["test_idx"])  # type: ignore
            fold_ids.extend([rec["fold"]] * len(rec["test_idx"]))  # type: ignore
            if rec.get("best_params_rec") is not None:
                best_param_records.append(rec["best_params_rec"])  # type: ignore

        y_true_b = np.asarray(y_true_b)
        y_pred_b = np.asarray(y_pred_b)
        pooled, per_subj = compute_metrics(y_true_b, y_pred_b, np.asarray(groups_ordered))
        summary[label] = pooled
        r_vals.append(pooled.get("pearson_r", np.nan))

        # Save per-band predictions
        if results_dir is not None:
            dfp = pd.DataFrame({
                "y_true": y_true_b,
                "y_pred": y_pred_b,
                "group": np.asarray(groups_ordered),
                "fold": fold_ids,
                "trial_index": test_indices_order,
                "band": label,
            })
            dfp.to_csv(results_dir / CONFIG["paths"]["predictions"]["riemann_band_template"].format(label=label), sep="\t", index=False)
            # Save band-specific Riemann LOSO test indices explicitly
            try:
                pd.DataFrame({
                    "group": np.asarray(groups_ordered),
                    "fold": fold_ids,
                    "trial_index": test_indices_order,
                }).to_csv(results_dir / CONFIG["paths"]["indices"]["riemann_band_template"].format(label=label), sep="\t", index=False)
            except Exception as e:
                logger.warning(f"Failed to save Riemann indices for band {label}: {e}")
            # Log best params per band
            if len(best_param_records) > 0:
                try:
                    path = results_dir / CONFIG["paths"]["best_params"]["riemann_band_template"].format(label=label)
                    _ensure_dir(path.parent)
                    with open(path, "a", encoding="utf-8") as f:
                        for r in best_param_records:
                            f.write(json.dumps(r) + "\n")
                except Exception as e:
                    logger.warning(f"Failed to log best params for band {label}: {e}")

    # Save summary JSON and bar plot
    if results_dir is not None:
        with open(results_dir / CONFIG["paths"]["summaries"]["riemann_bands"], "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if plots_dir is not None and len(labels) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0), dpi=150)
        ax.bar(range(len(labels)), r_vals, color="#4cc9f0")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Pooled Pearson r")
        ax.set_title("Riemann band-limited decoding")
        plt.tight_layout()
        plt.savefig(plots_dir / "riemann_bands_performance.png")
        plt.close(fig)

    return summary


def run_riemann_sliding_window(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = TASK,
    results_dir: Path = None,
    plots_dir: Path = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    window_len: float = 0.75,
    step: float = 0.25,
    n_jobs: int = -1,
    seed: int = 42,
    outer_n_jobs: int = 1,
) -> Optional[pd.DataFrame]:
    """Temporal sliding-window LOSO Riemann regression over plateau.

    Returns a DataFrame with columns [t_center, r, R2, EVS]. Saves JSON and line plot if dirs provided.
    """
    if not _check_pyriemann():
        logger.warning("pyriemann not installed; skipping sliding-window Riemann analysis.")
        return None
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
    except Exception:
        logger.warning("pyriemann unavailable; skipping sliding-window analysis.")
        return None

    tuples, _ = load_epochs_targets_groups(deriv_root, subjects=subjects, task=task)

    tmin_pl, tmax_pl = plateau_window
    starts = []
    t = tmin_pl
    while t + window_len <= tmax_pl + 1e-6:
        starts.append(t)
        t += step

    records = []

    # Build global trial index
    trial_records: List[Tuple[str, int]] = []
    y_all_list: List[float] = []
    groups_list: List[str] = []
    subj_to_epochs: Dict[str, "mne.Epochs"] = {}
    subj_to_y: Dict[str, pd.Series] = {}
    for sub, epochs, y in tuples:
        n = min(len(epochs), len(y))
        if n == 0:
            continue
        subj_to_epochs[sub] = epochs
        subj_to_y[sub] = pd.to_numeric(y.iloc[:n], errors="coerce")
        for ti in range(n):
            trial_records.append((sub, ti))
            y_all_list.append(float(subj_to_y[sub].iloc[ti]))
            groups_list.append(sub)
    if len(trial_records) == 0:
        return None

    y_all_arr = np.asarray(y_all_list)
    groups_arr = np.asarray(groups_list)
    logo = LeaveOneGroupOut()
    folds = list(enumerate(logo.split(np.arange(len(trial_records)), groups=groups_arr), start=1))

    for t0 in starts:
        pipe = Pipeline(steps=[
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("ridge", Ridge()),
        ])
        param_grid = {
            "cov__estimator": ["oas", "lwf"],
            "ridge__alpha": [1e-2, 1e-1, 1, 10],
        }

        def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
            np.random.seed(seed + fold)
            pyrandom.seed(seed + fold)
            # Find intersection of EEG channels across all training subjects in this fold
            train_subs_seq = [trial_records[i][0] for i in train_idx]
            train_subjects = list({s for s in train_subs_seq if s is not None})
            
            # Get EEG channel names for each training subject
            train_subject_eeg_chs = {}
            for s in train_subjects:
                train_subject_eeg_chs[s] = [
                    ch for ch in subj_to_epochs[s].info["ch_names"]
                    if subj_to_epochs[s].get_channel_types(picks=[ch])[0] == "eeg"
                ]
            
            # Find intersection of channels across all training subjects
            if len(train_subjects) == 1:
                common_chs = train_subject_eeg_chs[train_subjects[0]]
            else:
                common_chs = list(set.intersection(*[set(train_subject_eeg_chs[s]) for s in train_subjects]))
            
            if not common_chs:
                logger.warning(f"Sliding t0={t0:.2f}s fold {fold}: No common EEG channels across training subjects. Skipping fold.")
                return {
                    "y_true": y_test.tolist(),
                    "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                    "groups": groups_arr[test_idx_f].tolist(),
                }
            
            # Sort channels for consistency
            common_chs = sorted(common_chs)
            logger.info(f"Sliding t0={t0:.2f}s fold {fold}: Using {len(common_chs)} common EEG channels across {len(train_subjects)} training subjects.")
            
            # Separate train/test subjects to optimize memory usage
            test_subjects = list({trial_records[i][0] for i in test_idx})
            all_subjects = list(set(train_subjects + test_subjects))
            
            # Separate caches for train/test to reduce memory footprint
            train_data_cache: Dict[str, np.ndarray] = {}
            test_data_cache: Dict[str, np.ndarray] = {}
            
            for s in all_subjects:
                # Enforce EXACT common channel set for tangent-space consistency
                subject_chs = subj_to_epochs[s].info["ch_names"]
                missing_chs = [ch for ch in common_chs if ch not in subject_chs]
                
                if missing_chs:
                    logger.error(f"Sliding t0={t0:.2f}s fold {fold}: Subject {s} missing {len(missing_chs)}/{len(common_chs)} required channels: {missing_chs[:5]}{'...' if len(missing_chs) > 5 else ''}")
                    logger.error(f"Sliding t0={t0:.2f}s fold {fold}: Cannot maintain tangent-space dimensionality. Skipping fold.")
                    return {
                        "y_true": y_test.tolist(),
                        "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                        "groups": groups_arr[test_idx_f].tolist(),
                    }
                
                # All channels present - safe to pick exact training intersection
                try:
                    ep = subj_to_epochs[s].copy().pick(common_chs)
                except Exception as e:
                    logger.error(f"Sliding t0={t0:.2f}s fold {fold}: Failed to pick channels for subject {s} despite availability check: {e}")
                    return {
                        "y_true": y_test.tolist(),
                        "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                        "groups": groups_arr[test_idx_f].tolist(),
                    }
                # Log sliding window details for reproducibility
                logger.info(f"Sliding t0={t0:.2f}s fold {fold}: cropping subject {s} to [{t0:.2f}, {t0 + window_len:.2f}]s (sfreq={ep.info['sfreq']:.1f}Hz)")
                try:
                    ep = ep.copy().crop(tmin=t0, tmax=t0 + window_len)
                    logger.info(f"Sliding t0={t0:.2f}s fold {fold}: subject {s} cropped successfully")
                except Exception as e:
                    logger.warning(f"Sliding t0={t0:.2f}s fold {fold}: cropping failed for subject {s}: {e}")
                    
                # Pre-extract all data for this subject once, convert to float32 for memory efficiency
                try:
                    data = ep.get_data(picks="eeg", reject_by_annotation=None)
                except TypeError:
                    data = ep.get_data(picks="eeg")
                
                # Convert to float32 before covariance computation to reduce memory usage
                data = data.astype(np.float32)
                
                # Store in appropriate cache based on train/test membership
                if s in train_subjects:
                    train_data_cache[s] = data
                if s in test_subjects:
                    test_data_cache[s] = data

            # Drop NaN targets only
            y_train = y_all_arr[train_idx]
            y_test = y_all_arr[test_idx]
            train_sel = np.isfinite(y_train)
            test_sel = np.isfinite(y_test)
            train_idx_f = train_idx[train_sel]
            test_idx_f = test_idx[test_sel]
            y_train = y_train[train_sel]
            y_test = y_test[test_sel]

            def _extract_block(indices: np.ndarray) -> np.ndarray:
                X_list: List[np.ndarray] = []
                for i in indices:
                    sub_i, ti = trial_records[int(i)]
                    # Use appropriate cache based on whether this is for training or testing
                    if i in train_idx_f:
                        X_i = train_data_cache[sub_i][ti]
                    else:
                        X_i = test_data_cache[sub_i][ti]
                    X_list.append(X_i)
                return np.stack(X_list, axis=0)

            X_train = _extract_block(train_idx_f)
            X_test = _extract_block(test_idx_f)
            train_groups = groups_arr[train_idx_f]

            # Inner CV
            n_unique = len(np.unique(train_groups))
            best = None
            # Use multi-metric scoring with signed Pearson r as primary to avoid sign-flipped models
            scoring = {'r': _make_pearsonr_scorer(), 'neg_mse': 'neg_mean_squared_error'}
            refit_metric = 'r'
            if n_unique >= 2:
                n_splits_inner = int(np.clip(int(CONFIG["cv"]["inner_splits"]), 2, n_unique))
                inner_cv = GroupKFold(n_splits=n_splits_inner)
                try:
                    gs = GridSearchCV(
                        estimator=pipe,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=inner_cv,
                        n_jobs=n_jobs,
                        refit=refit_metric,
                    )
                    gs.fit(X_train, y_train, groups=train_groups)
                    best = gs.best_estimator_
                    
                    # Extract best parameters for both metrics (for sliding window)
                    try:
                        cv_results = pd.DataFrame(gs.cv_results_)
                        best_by_r_idx = cv_results['rank_test_r'].idxmin()
                        best_by_mse_idx = cv_results['rank_test_neg_mse'].idxmin()
                        
                        best_params_r = cv_results.loc[best_by_r_idx, 'params']
                        best_params_mse = cv_results.loc[best_by_mse_idx, 'params']
                        
                        logger.debug(f"Sliding t0={t0} fold {fold}: best params by r = {best_params_r}")
                        logger.debug(f"Sliding t0={t0} fold {fold}: best params by neg_mse = {best_params_mse}")
                    except Exception as param_e:
                        logger.warning(f"Failed to extract dual metrics for sliding t0={t0} fold {fold}: {param_e}")
                        
                except Exception as e:
                    logger.warning(f"Sliding t0={t0} fold {fold}: GridSearchCV failed: {e}; using defaults.")
                    best = None
            else:
                best = clone(pipe)
                try:
                    best.fit(X_train, y_train)
                except Exception as e:
                    logger.warning(f"Sliding t0={t0} fold {fold}: default fit failed: {e}")
                    return {
                        "y_true": y_test.tolist(),
                        "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                        "groups": groups_arr[test_idx_f].tolist(),
                    }

            if best is None:
                best = clone(pipe)
                try:
                    best.fit(X_train, y_train)
                except Exception:
                    return {
                        "y_true": y_test.tolist(),
                        "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                        "groups": groups_arr[test_idx_f].tolist(),
                    }

            try:
                y_pred = best.predict(X_test)
            except Exception:
                y_pred = np.full(len(y_test), np.nan, dtype=float)

            return {"y_true": y_test.tolist(), "y_pred": np.asarray(y_pred).tolist(), "groups": groups_arr[test_idx_f].tolist()}

        if outer_n_jobs and outer_n_jobs != 1 and len(folds) > 1:
            results = Parallel(n_jobs=outer_n_jobs, prefer="threads")(delayed(_run_fold)(fold, tr, te) for (fold, (tr, te)) in folds)
        else:
            results = [_run_fold(fold, tr, te) for (fold, (tr, te)) in folds]

        # Aggregate
        y_true_sw, y_pred_sw, groups_ordered = [], [], []
        for rec in results:
            y_true_sw.extend(rec["y_true"])  # type: ignore
            y_pred_sw.extend(rec["y_pred"])  # type: ignore
            groups_ordered.extend(rec["groups"])  # type: ignore

        y_true_sw = np.asarray(y_true_sw)
        y_pred_sw = np.asarray(y_pred_sw)
        pooled, _per_subj = compute_metrics(y_true_sw, y_pred_sw, np.asarray(groups_ordered))
        records.append({
            "t_center": float(t0 + window_len / 2.0),
            "pearson_r": pooled.get("pearson_r", np.nan),
            "r2": pooled.get("r2", np.nan),
            "explained_variance": pooled.get("explained_variance", np.nan),
        })

    if len(records) == 0:
        return None

    df = pd.DataFrame.from_records(records).sort_values("t_center").reset_index(drop=True)
    if results_dir is not None:
        with open(results_dir / CONFIG["paths"]["summaries"]["riemann_sliding_window"], "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="list"), f, indent=2)
    if plots_dir is not None:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0), dpi=150)
        ax.plot(df["t_center"], df["pearson_r"], marker="o", color="#ef476f")
        ax.set_xlabel("Time (s) relative to stimulus")
        ax.set_ylabel("Pooled Pearson r")
        ax.set_title("Riemann sliding-window decoding over plateau")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "riemann_sliding_window_r_vs_time.png")
        plt.close(fig)

    return df

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@_validate_inputs
def main(subjects: Optional[List[str]] = None, task: str = TASK, n_jobs: int = -1, seed: int = RANDOM_STATE, outer_n_jobs: int = 1) -> None:
    deriv_root = Path(DERIV_ROOT)
    results_dir = deriv_root / CONFIG["paths"]["results_subdir"]
    plots_dir = results_dir / CONFIG["paths"]["plots_subdir"]
    _ensure_dir(plots_dir)

    # Initialize per-run file logging under results_dir/logs
    log_path = _setup_file_logging(results_dir, RUN_ID)
    logger.info(f"File logging initialized: {log_path}")

    # Create run manifest with CLI args, config, and environment info
    try:
        cli_args_dict = {
            "subjects": subjects,
            "task": task,
            "n_jobs": n_jobs,
            "seed": seed,
            "outer_n_jobs": outer_n_jobs,
            "best_params_mode": BEST_PARAMS_MODE,
            "run_id": RUN_ID
        }
        _create_run_manifest(results_dir, cli_args_dict, CONFIG, RUN_ID)
        logger.info(f"Created run manifest at {results_dir / 'run_manifest.json'}")
    except Exception as e:
        logger.warning(f"Failed to create run manifest: {e}")

    # Reproducibility and env logging
    import sklearn as _sk
    import mne as _mne
    info = {"sklearn": _sk.__version__, "mne": _mne.__version__}
    try:
        import pyriemann as _pr
        info["pyriemann"] = _pr.__version__
    except Exception:
        info["pyriemann"] = None
    logger.info(f"Package versions: {json.dumps(info)}")

    # Run header
    _ts = time.strftime("%Y-%m-%d %H:%M:%S")
    _subs = "all" if subjects is None else ",".join([str(s) for s in subjects])
    logger.info(
        f"Run header | ts={_ts} task={task} seed={seed} subjects={_subs} "
        f"best_params_mode={BEST_PARAMS_MODE} run_id={RUN_ID or ''} results_dir={results_dir} plots_dir={plots_dir}"
    )

    # Get CLI-configurable seed and n_jobs if present in env variables set by CLI wrapper
    # (These will be explicitly set via argparse in __main__.)

    # 1) Tabular features aggregation
    X_all, y_all, groups, feat_names, meta = load_tabular_features_and_targets(deriv_root, subjects=subjects)
    # Guard against too few subjects for LOSO
    if len(np.unique(groups)) < 2:
        raise RuntimeError("Need at least 2 subjects for LOSO.")

    # Define models and grids
    # Model 0: Elastic Net with feature selection (SelectFromModel) and target transform (TTR)
    en_cfg = CONFIG["models"]["elasticnet"]
    enet_pipe = _create_elasticnet_pipeline(seed=seed)
    # Update with config parameters
    enet_pipe.named_steps["regressor"].regressor.max_iter = en_cfg["max_iter"]
    enet_pipe.named_steps["regressor"].regressor.tol = en_cfg["tol"]
    enet_pipe.named_steps["regressor"].regressor.selection = en_cfg["selection"]
    enet_grid = {
        # Final ElasticNet inside TransformedTargetRegressor
        "regressor__regressor__alpha": en_cfg["grid"]["alpha"],
        "regressor__regressor__l1_ratio": en_cfg["grid"]["l1_ratio"],
    }

    # Model 1: Random Forest (no scaling)
    rf_cfg = CONFIG["models"]["random_forest"]
    rf_pipe = _create_rf_pipeline(n_estimators=rf_cfg["n_estimators"], n_jobs=rf_cfg["estimator_n_jobs"], seed=seed)
    rf_pipe.named_steps["rf"].bootstrap = rf_cfg["bootstrap"]
    
    # RF n_jobs safety check to avoid nested parallelism issues
    rf_n_jobs = rf_pipe.named_steps["rf"].n_jobs
    if rf_n_jobs != 1 and (outer_n_jobs != 1 or CONFIG["analysis"].get("rf_perm_importance_repeats", 1) > 1):
        logger.warning(f"RF n_jobs={rf_n_jobs} with outer_n_jobs={outer_n_jobs} may cause nested parallelism issues. Consider setting RF n_jobs=1.")
        if rf_n_jobs == -1:
            logger.warning("Forcing RF n_jobs=1 to avoid nested parallelism during permutation importance and outer CV.")
            rf_pipe.named_steps["rf"].n_jobs = 1
    
    rf_grid = {
        "rf__max_depth": rf_cfg["grid"]["max_depth"],
        "rf__max_features": rf_cfg["grid"]["max_features"],
        "rf__min_samples_leaf": rf_cfg["grid"]["min_samples_leaf"],
    }

    inner_splits = int(CONFIG["cv"]["inner_splits"])

    # Elastic Net nested LOSO
    # Paths for best-params logs (prepared per mode)
    best_params_en_path = _prepare_best_params_path(
        results_dir / CONFIG["paths"]["best_params"]["elasticnet_loso"], BEST_PARAMS_MODE, RUN_ID
    )
    best_params_rf_path = _prepare_best_params_path(
        results_dir / CONFIG["paths"]["best_params"]["rf_loso"], BEST_PARAMS_MODE, RUN_ID
    )

    y_true_en, y_pred_en, groups_ordered_en, test_indices_en, fold_ids_en = _nested_loso_predictions(
        X=X_all, y=y_all, groups=groups,
        pipe=enet_pipe, param_grid=enet_grid,
        inner_cv_splits=inner_splits, n_jobs=n_jobs, seed=seed,
        best_params_log_path=best_params_en_path, model_name="ElasticNet",
        outer_n_jobs=outer_n_jobs,
    )
    pooled_en, per_subj_en = compute_metrics(y_true_en, y_pred_en, np.asarray(groups_ordered_en))
    logger.info(
        f"ElasticNet pooled: r={pooled_en['pearson_r']:.3f}, R2={pooled_en['r2']:.3f}, EVS={pooled_en['explained_variance']:.3f}, "
        f"avg_r_Fz={pooled_en['avg_subject_r_fisher_z']:.3f}"
    )

    # Save predictions
    _ensure_dir(results_dir)
    pred_en = pd.DataFrame({
        "subject_id": groups_ordered_en,
        "trial_id": meta.loc[test_indices_en, "trial_id"].values,
        "y_true": y_true_en,
        "y_pred": y_pred_en,
        "fold": fold_ids_en,
        "model": "ElasticNet",
    })
    _ensure_dir((results_dir / CONFIG["paths"]["predictions"]["elasticnet_loso"]).parent)
    pred_en.to_csv(results_dir / CONFIG["paths"]["predictions"]["elasticnet_loso"], sep="\t", index=False)
    _ensure_dir((results_dir / CONFIG["paths"]["per_subject_metrics"]["elasticnet_loso"]).parent)
    per_subj_en.to_csv(results_dir / CONFIG["paths"]["per_subject_metrics"]["elasticnet_loso"], sep="\t", index=False)
    # Save ElasticNet LOSO test indices explicitly
    try:
        idx_en = pd.DataFrame({
            "subject_id": groups_ordered_en,
            "trial_id": meta.loc[test_indices_en, "trial_id"].values,
            "fold": fold_ids_en,
        })
        _ensure_dir((results_dir / CONFIG["paths"]["indices"]["elasticnet_loso"]).parent)
        idx_en.to_csv(results_dir / CONFIG["paths"]["indices"]["elasticnet_loso"], sep="\t", index=False)
    except Exception as e:
        logger.warning(f"Failed to save ElasticNet LOSO indices: {e}")
    scatter_actual_vs_predicted(y_true_en, y_pred_en, plots_dir / "elasticnet_loso_actual_vs_predicted.png",
                                title=f"ElasticNet LOSO: r={pooled_en['pearson_r']:.2f}, R2={pooled_en['r2']:.2f}")

    # ElasticNet within-subject KFold (trial-wise within each subject)
    try:
        if not CONFIG["flags"]["run_within_subject_kfold"]:
            raise RuntimeError("Within-subject KFold disabled via CONFIG['flags']['run_within_subject_kfold']")
        best_params_en_within_path = _prepare_best_params_path(
            results_dir / CONFIG["paths"]["best_params"]["elasticnet_within"], BEST_PARAMS_MODE, RUN_ID
        )
        y_true_wen, y_pred_wen, groups_ordered_wen, test_indices_wen, fold_ids_wen = _within_subject_kfold_predictions(
            X=X_all, y=y_all, groups=groups,
            pipe=enet_pipe, param_grid=enet_grid,
            inner_cv_splits=inner_splits, n_jobs=n_jobs, seed=seed,
            best_params_log_path=best_params_en_within_path, model_name="ElasticNet",
            outer_n_jobs=outer_n_jobs, deriv_root=deriv_root, task=task,
        )
        pooled_wen, per_subj_wen = compute_metrics(y_true_wen, y_pred_wen, np.asarray(groups_ordered_wen))
        logger.info(
            f"ElasticNet Within-Subject KFold pooled: r={pooled_wen['pearson_r']:.3f}, R2={pooled_wen['r2']:.3f}, EVS={pooled_wen['explained_variance']:.3f}, "
            f"avg_r_Fz={pooled_wen['avg_subject_r_fisher_z']:.3f}"
        )

        pred_wen = pd.DataFrame({
            "subject_id": groups_ordered_wen,
            "trial_id": meta.loc[test_indices_wen, "trial_id"].values,
            "y_true": y_true_wen,
            "y_pred": y_pred_wen,
            "fold": fold_ids_wen,
            "model": "ElasticNet_WithinKFold",
        })
        _ensure_dir((results_dir / CONFIG["paths"]["predictions"]["elasticnet_within"]).parent)
        pred_wen.to_csv(results_dir / CONFIG["paths"]["predictions"]["elasticnet_within"], sep="\t", index=False)
        _ensure_dir((results_dir / CONFIG["paths"]["per_subject_metrics"]["elasticnet_within"]).parent)
        per_subj_wen.to_csv(results_dir / CONFIG["paths"]["per_subject_metrics"]["elasticnet_within"], sep="\t", index=False)
        scatter_actual_vs_predicted(
            y_true_wen, y_pred_wen, plots_dir / "elasticnet_within_kfold_actual_vs_predicted.png",
            title=f"ElasticNet Within-Subject KFold: r={pooled_wen['pearson_r']:.2f}, R2={pooled_wen['r2']:.2f}"
        )

        # Save ElasticNet within-subject KFold test indices explicitly
        try:
            idx_wen = pd.DataFrame({
                "subject_id": groups_ordered_wen,
                "trial_id": meta.loc[test_indices_wen, "trial_id"].values,
                "fold": fold_ids_wen,
            })
            _ensure_dir((results_dir / CONFIG["paths"]["indices"]["elasticnet_within"]).parent)
            idx_wen.to_csv(results_dir / CONFIG["paths"]["indices"]["elasticnet_within"], sep="\t", index=False)
        except Exception as e:
            logger.warning(f"Failed to save ElasticNet within-subject indices: {e}")

        # Paired comparisons vs LOSO (per-subject)
        try:
            plot_paired_metric_scatter(
                per_left=per_subj_en, per_right=per_subj_wen, metric="pearson_r",
                save_path=plots_dir / "paired_elasticnet_loso_vs_withinkfold_pearson_r.png",
                label_left="LOSO", label_right="WithinKFold",
            )
        except Exception as e:
            logger.warning(f"Paired plot (pearson_r) failed: {e}")
        try:
            plot_paired_metric_scatter(
                per_left=per_subj_en, per_right=per_subj_wen, metric="r2",
                save_path=plots_dir / "paired_elasticnet_loso_vs_withinkfold_r2.png",
                label_left="LOSO", label_right="WithinKFold",
            )
        except Exception as e:
            logger.warning(f"Paired plot (r2) failed: {e}")
        try:
            plot_paired_metric_scatter(
                per_left=per_subj_en, per_right=per_subj_wen, metric="explained_variance",
                save_path=plots_dir / "paired_elasticnet_loso_vs_withinkfold_evs.png",
                label_left="LOSO", label_right="WithinKFold",
            )
        except Exception as e:
            logger.warning(f"Paired plot (explained_variance) failed: {e}")
    except Exception as e:
        logger.warning(f"Within-subject KFold ElasticNet decoding failed: {e}")

    # Random Forest nested LOSO
    y_true_rf, y_pred_rf, groups_ordered_rf, test_indices_rf, fold_ids_rf = _nested_loso_predictions(
        X=X_all, y=y_all, groups=groups,
        pipe=rf_pipe, param_grid=rf_grid,
        inner_cv_splits=inner_splits, n_jobs=n_jobs, seed=seed,
        best_params_log_path=best_params_rf_path, model_name="RandomForest",
        outer_n_jobs=outer_n_jobs,
    )
    pooled_rf, per_subj_rf = compute_metrics(y_true_rf, y_pred_rf, np.asarray(groups_ordered_rf))
    logger.info(
        f"RandomForest pooled: r={pooled_rf['pearson_r']:.3f}, R2={pooled_rf['r2']:.3f}, EVS={pooled_rf['explained_variance']:.3f}, "
        f"avg_r_Fz={pooled_rf['avg_subject_r_fisher_z']:.3f}"
    )

    pred_rf = pd.DataFrame({
        "subject_id": groups_ordered_rf,
        "trial_id": meta.loc[test_indices_rf, "trial_id"].values,
        "y_true": y_true_rf,
        "y_pred": y_pred_rf,
        "fold": fold_ids_rf,
        "model": "RandomForest",
    })
    _ensure_dir((results_dir / CONFIG["paths"]["predictions"]["rf_loso"]).parent)
    pred_rf.to_csv(results_dir / CONFIG["paths"]["predictions"]["rf_loso"], sep="\t", index=False)
    _ensure_dir((results_dir / CONFIG["paths"]["per_subject_metrics"]["rf_loso"]).parent)
    per_subj_rf.to_csv(results_dir / CONFIG["paths"]["per_subject_metrics"]["rf_loso"], sep="\t", index=False)
    scatter_actual_vs_predicted(y_true_rf, y_pred_rf, plots_dir / "rf_loso_actual_vs_predicted.png",
                                title=f"RF LOSO: r={pooled_rf['pearson_r']:.2f}, R2={pooled_rf['r2']:.2f}")

    # Save RF LOSO test indices explicitly
    try:
        idx_rf = pd.DataFrame({
            "subject_id": groups_ordered_rf,
            "trial_id": meta.loc[test_indices_rf, "trial_id"].values,
            "fold": fold_ids_rf,
        })
        _ensure_dir((results_dir / CONFIG["paths"]["indices"]["rf_loso"]).parent)
        idx_rf.to_csv(results_dir / CONFIG["paths"]["indices"]["rf_loso"], sep="\t", index=False)
    except Exception as e:
        logger.warning(f"Failed to save RF LOSO indices: {e}")

    # Random Forest within-subject KFold (trial-wise within each subject)
    try:
        if not CONFIG["flags"]["run_within_subject_kfold"]:
            raise RuntimeError("Within-subject KFold disabled via CONFIG['flags']['run_within_subject_kfold']")
        best_params_rf_within_path = _prepare_best_params_path(
            results_dir / CONFIG["paths"]["best_params"]["rf_within"], BEST_PARAMS_MODE, RUN_ID
        )
        y_true_wrf, y_pred_wrf, groups_ordered_wrf, test_indices_wrf, fold_ids_wrf = _within_subject_kfold_predictions(
            X=X_all, y=y_all, groups=groups,
            pipe=rf_pipe, param_grid=rf_grid,
            inner_cv_splits=inner_splits, n_jobs=n_jobs, seed=seed,
            best_params_log_path=best_params_rf_within_path, model_name="RandomForest",
            outer_n_jobs=outer_n_jobs, deriv_root=deriv_root, task=task,
        )
        pooled_wrf, per_subj_wrf = compute_metrics(y_true_wrf, y_pred_wrf, np.asarray(groups_ordered_wrf))
        logger.info(
            f"RF Within-Subject KFold pooled: r={pooled_wrf['pearson_r']:.3f}, R2={pooled_wrf['r2']:.3f}, EVS={pooled_wrf['explained_variance']:.3f}, "
            f"avg_r_Fz={pooled_wrf['avg_subject_r_fisher_z']:.3f}"
        )

        pred_wrf = pd.DataFrame({
            "subject_id": groups_ordered_wrf,
            "trial_id": meta.loc[test_indices_wrf, "trial_id"].values,
            "y_true": y_true_wrf,
            "y_pred": y_pred_wrf,
            "fold": fold_ids_wrf,
            "model": "RandomForest_WithinKFold",
        })
        _ensure_dir((results_dir / CONFIG["paths"]["predictions"]["rf_within"]).parent)
        pred_wrf.to_csv(results_dir / CONFIG["paths"]["predictions"]["rf_within"], sep="\t", index=False)
        _ensure_dir((results_dir / CONFIG["paths"]["per_subject_metrics"]["rf_within"]).parent)
        per_subj_wrf.to_csv(results_dir / CONFIG["paths"]["per_subject_metrics"]["rf_within"], sep="\t", index=False)
        scatter_actual_vs_predicted(
            y_true_wrf, y_pred_wrf, plots_dir / "rf_within_kfold_actual_vs_predicted.png",
            title=f"RF Within-Subject KFold: r={pooled_wrf['pearson_r']:.2f}, R2={pooled_wrf['r2']:.2f}"
        )

        # Save RF within-subject KFold test indices explicitly
        try:
            idx_wrf = pd.DataFrame({
                "subject_id": groups_ordered_wrf,
                "trial_id": meta.loc[test_indices_wrf, "trial_id"].values,
                "fold": fold_ids_wrf,
            })
            _ensure_dir((results_dir / CONFIG["paths"]["indices"]["rf_within"]).parent)
            idx_wrf.to_csv(results_dir / CONFIG["paths"]["indices"]["rf_within"], sep="\t", index=False)
        except Exception as e:
            logger.warning(f"Failed to save RF within-subject indices: {e}")

        # Paired comparisons vs LOSO (per-subject)
        try:
            plot_paired_metric_scatter(
                per_left=per_subj_rf, per_right=per_subj_wrf, metric="pearson_r",
                save_path=plots_dir / "paired_rf_loso_vs_withinkfold_pearson_r.png",
                label_left="LOSO", label_right="WithinKFold",
            )
        except Exception as e:
            logger.warning(f"Paired plot (pearson_r) failed: {e}")
        # Combined paired scatter + dumbbell figure (pearson_r)
        try:
            plot_within_vs_loso_combined(
                per_left=per_subj_rf, per_right=per_subj_wrf, metric="pearson_r",
                save_path=plots_dir / "combined_rf_loso_vs_withinkfold_pearson_r.png",
                label_left="LOSO", label_right="WithinKFold",
            )
        except Exception as e:
            logger.warning(f"Combined paired+dumbbell (pearson_r) failed: {e}")
        try:
            plot_paired_metric_scatter(
                per_left=per_subj_rf, per_right=per_subj_wrf, metric="r2",
                save_path=plots_dir / "paired_rf_loso_vs_withinkfold_r2.png",
                label_left="LOSO", label_right="WithinKFold",
            )
        except Exception as e:
            logger.warning(f"Paired plot (r2) failed: {e}")
        try:
            plot_paired_metric_scatter(
                per_left=per_subj_rf, per_right=per_subj_wrf, metric="explained_variance",
                save_path=plots_dir / "paired_rf_loso_vs_withinkfold_evs.png",
                label_left="LOSO", label_right="WithinKFold",
            )
        except Exception as e:
            logger.warning(f"Paired plot (explained_variance) failed: {e}")
    except Exception as e:
        logger.warning(f"Within-subject KFold RF decoding failed: {e}")

    # Diagnostics & robustness plots for RF and ElasticNet
    try:
        plot_per_subject_violin(per_subj_rf, plots_dir / "per_subject_metrics_violin.png")
    except Exception as e:
        logger.warning(f"per-subject violin plot failed: {e}")
    try:
        plot_learning_curve_rf(X_all, y_all, groups, results_dir, plots_dir / "learning_curve_rf.png", seed=seed, best_params_path=best_params_rf_path)
    except Exception as e:
        logger.warning(f"RF learning curve plotting failed: {e}")
    try:
        plot_learning_curve_en(X_all, y_all, groups, results_dir, plots_dir / "learning_curve_elasticnet.png", seed=seed, best_params_path=best_params_en_path)
    except Exception as e:
        logger.warning(f"ElasticNet learning curve plotting failed: {e}")

    # Determine best EEG model by pooled Pearson r
    best_model_name = None
    best_pooled = None
    y_true_best = None
    y_pred_best = None
    groups_ordered_best = None
    best_pipe_template = None
    best_params_path = None
    try:
        candidates = []
        if 'pooled_en' in locals():
            candidates.append(("ElasticNet", pooled_en.get("pearson_r", np.nan), y_true_en, y_pred_en, groups_ordered_en, enet_pipe, _prepare_best_params_path(
                results_dir / CONFIG["paths"]["best_params"]["elasticnet_loso"], BEST_PARAMS_MODE, RUN_ID
            )))
        if 'pooled_rf' in locals():
            candidates.append(("RandomForest", pooled_rf.get("pearson_r", np.nan), y_true_rf, y_pred_rf, groups_ordered_rf, rf_pipe, _prepare_best_params_path(
                results_dir / CONFIG["paths"]["best_params"]["rf_loso"], BEST_PARAMS_MODE, RUN_ID
            )))
        if len(candidates) > 0:
            best_model_name, _, y_true_best, y_pred_best, groups_ordered_best, best_pipe_template, best_params_path = max(
                candidates, key=lambda t: (t[1] if np.isfinite(t[1]) else -np.inf)
            )
            best_pooled = pooled_en if best_model_name == "ElasticNet" else pooled_rf
    except Exception as e:
        logger.warning(f"Failed to select best model for permutation tests: {e}")

    # Quick (non-refit) permutation null using best model predictions
    try:
        if y_true_best is not None and y_pred_best is not None and groups_ordered_best is not None:
            p_perm = plot_permutation_null_hist(np.asarray(y_true_best), np.asarray(y_pred_best), np.asarray(groups_ordered_best),
                                                plots_dir / "permutation_null_hist.png", n_perm=int(CONFIG["analysis"]["n_perm_quick"]), seed=seed)
            logger.info(f"Permutation test (no-refit) p-value (pooled r, {best_model_name}): {p_perm:.4g}")
    except Exception as e:
        logger.warning(f"permutation null plotting failed: {e}")

    # Refit-based permutation null (within-subject shuffles, LOSO refit using fixed best-model params)
    try:
        if best_params_path is not None and best_pipe_template is not None and best_pooled is not None:
            best_params_map = _read_best_params_jsonl(best_params_path)
            n_perm_refit = int(CONFIG["analysis"]["n_perm_refit"])
            perm_jobs = int(CONFIG["analysis"].get("perm_refit_n_jobs", 1)) if isinstance(CONFIG.get("analysis"), dict) else 1
            def _one_perm(i: int) -> float:
                rng_i = np.random.default_rng(seed + 12345 + i)
                y_perm = y_all.to_numpy().copy()
                for g in np.unique(groups):
                    idx = np.where(groups == g)[0]
                    if len(idx) > 1:
                        y_perm[idx] = rng_i.permutation(y_perm[idx])
                y_perm_series = pd.Series(y_perm)
                y_true_p, y_pred_p, groups_p, _, _ = _loso_predictions_with_fixed_params(
                    X_all, y_perm_series, groups, best_pipe_template, best_params_map, seed=seed + 1000 + i, outer_n_jobs=outer_n_jobs,
                )
                r_p, _ = _safe_pearsonr(y_true_p, y_pred_p)
                return float(r_p) if np.isfinite(r_p) else 0.0

            if perm_jobs > 1:
                null_list = Parallel(n_jobs=perm_jobs, prefer="processes")(delayed(_one_perm)(i) for i in range(n_perm_refit))
                null_rs = np.asarray(null_list, dtype=float)
            else:
                null_rs = np.zeros(n_perm_refit, dtype=float)
                rng = np.random.default_rng(seed + 12345)
                for i in range(n_perm_refit):
                    y_perm = y_all.to_numpy().copy()
                    for g in np.unique(groups):
                        idx = np.where(groups == g)[0]
                        if len(idx) > 1:
                            y_perm[idx] = rng.permutation(y_perm[idx])
                    y_perm_series = pd.Series(y_perm)
                    y_true_p, y_pred_p, groups_p, _, _ = _loso_predictions_with_fixed_params(
                        X_all, y_perm_series, groups, best_pipe_template, best_params_map, seed=seed + 1000 + i, outer_n_jobs=outer_n_jobs,
                    )
                    r_p, _ = _safe_pearsonr(y_true_p, y_pred_p)
                    null_rs[i] = float(r_p) if np.isfinite(r_p) else 0.0

            obs_r = float(best_pooled.get("pearson_r", np.nan))
            # Two-sided p-value for |r|
            p_refit_two_sided = float((np.sum(np.abs(null_rs) >= abs(obs_r)) + 1) / (len(null_rs) + 1))
            # One-sided p-values (assuming positive correlation expected)
            p_refit_one_sided_pos = float((np.sum(null_rs >= obs_r) + 1) / (len(null_rs) + 1))
            p_refit_one_sided_neg = float((np.sum(null_rs <= obs_r) + 1) / (len(null_rs) + 1))
            
            _ensure_dir((results_dir / CONFIG["paths"]["summaries"]["permutation_refit_null_rs"]).parent)
            np.savetxt(results_dir / CONFIG["paths"]["summaries"]["permutation_refit_null_rs"], null_rs, fmt="%.6f")
            plot_permutation_refit_null_hist(null_rs, obs_r, plots_dir / "permutation_refit_null_hist.png")
            
            # Enhanced null distribution statistics
            null_stats = {
                "mean": float(np.mean(null_rs)),
                "std": float(np.std(null_rs, ddof=1)) if n_perm_refit > 1 else 0.0,
                "median": float(np.median(null_rs)),
                "q25": float(np.percentile(null_rs, 25)),
                "q75": float(np.percentile(null_rs, 75)),
                "min": float(np.min(null_rs)),
                "max": float(np.max(null_rs)),
                "skewness": float(scipy.stats.skew(null_rs)) if len(null_rs) > 2 else 0.0,
                "kurtosis": float(scipy.stats.kurtosis(null_rs)) if len(null_rs) > 2 else 0.0,
            }
            
            _ensure_dir((results_dir / CONFIG["paths"]["summaries"]["permutation_refit_summary"]).parent)
            with open(results_dir / CONFIG["paths"]["summaries"]["permutation_refit_summary"], "w", encoding="utf-8") as f:
                json.dump({
                    "model": best_model_name,
                    "observed_r": obs_r,
                    "n_perm": int(n_perm_refit),
                    "p_two_sided_abs_r": p_refit_two_sided,
                    "p_one_sided_positive": p_refit_one_sided_pos,
                    "p_one_sided_negative": p_refit_one_sided_neg,
                    "null_distribution_stats": null_stats,
                }, f, indent=2)
            logger.info(f"Saved refit-based permutation null for {best_model_name} (p_two_sided={p_refit_two_sided:.4g}, p_one_sided_pos={p_refit_one_sided_pos:.4g})")
    except Exception as e:
        logger.warning(f"Refit-based permutation null failed: {e}")
    try:
        plot_residuals_and_qq(y_true_rf, y_pred_rf, plots_dir / "residuals_scatter.png", plots_dir / "qq_plot.png")
    except Exception as e:
        logger.warning(f"residual/QQ plotting failed: {e}")
    try:
        plot_bland_altman(y_true_rf, y_pred_rf, plots_dir / "bland_altman.png")
    except Exception as e:
        logger.warning(f"Bland–Altman plotting failed: {e}")
    try:
        plot_calibration_curve(y_true_rf, y_pred_rf, plots_dir / "calibration_curve.png")
        # Compute and save calibration metrics
        cal_metrics = compute_calibration_metrics(y_true_rf, y_pred_rf)
        _ensure_dir((results_dir / "calibration_metrics.json").parent)
        with open(results_dir / "calibration_metrics.json", "w", encoding="utf-8") as f:
            json.dump(cal_metrics, f, indent=2)
        logger.info(f"Calibration metrics: slope={cal_metrics['slope']:.3f}, intercept={cal_metrics['intercept']:.3f}")
    except Exception as e:
        logger.warning(f"calibration curve plotting/metrics failed: {e}")

    # Interpretability analyses: ElasticNet coefs, RF permutation importance, stability, SHAP
    if 'pooled_en' in locals() or 'pooled_rf' in locals():
        # Read best params per fold
        best_params_en_map = _read_best_params_jsonl(best_params_en_path)
        best_params_rf_map = _read_best_params_jsonl(best_params_rf_path)

        # Elastic Net: per-fold coefficients and band-aggregated topomaps
        if 'pooled_en' in locals():
            try:
                en_coefs = compute_enet_coefs_per_fold(
                    X=X_all, y=y_all, groups=np.asarray(groups), best_params_map=best_params_en_map, seed=seed
                )
                try:
                    _montage = resolve_montage(CONFIG.get("viz", {}).get("montage", "standard_1005"), deriv_root, subjects)
                except (FileNotFoundError, ValueError, RuntimeError) as e:
                    logger.warning(f"Montage resolution failed ({e}); using standard_1005")
                    _montage = mne.channels.make_standard_montage("standard_1005")
                _agg = str(CONFIG.get("viz", {}).get("coef_agg", "abs")).lower()
                if _agg not in ("abs", "signed"):
                    _agg = "abs"
                plot_enet_band_topomaps(en_coefs, feat_names, plots_dir, montage=_montage, aggregate=_agg)
            except (ValueError, RuntimeError, MemoryError) as e:
                logger.error(f"ElasticNet interpretability failed: {e}")
            except Exception as e:
                logger.warning(f"ElasticNet interpretability failed with unexpected error: {e}")

        # Random Forest: permutation importance across folds
        if 'pooled_rf' in locals():
            try:
                rf_imps = compute_rf_block_permutation_importance_per_fold(
                    X=X_all, y=y_all, groups=np.asarray(groups), best_params_map=best_params_rf_map, seed=seed, n_repeats=int(CONFIG["analysis"]["rf_perm_importance_repeats"])
                )
                mean_imps = np.nanmean(rf_imps, axis=0)
                plot_rf_perm_importance_bar(mean_imps, feat_names, plots_dir / "rf_block_permutation_importance_top20.png", top_n=20)

                # Feature ranking stability (Kendall's tau) and top-K rank heatmap
                try:
                    # Rank features per fold (1=best). Use negative importances so larger importance -> smaller rank value
                    ranks = np.vstack([
                        rankdata(-row, method="average") if np.all(np.isfinite(row)) else rankdata(-np.nan_to_num(row), method="average")
                        for row in rf_imps
                    ])
                    plot_kendall_tau_heatmap(ranks, plots_dir / "rf_kendall_tau_rank_stability.png")
                    plot_topk_rank_heatmap(ranks, feat_names, plots_dir / "rf_topk_feature_ranks_heatmap.png", top_k=30)
                except (ValueError, IndexError) as e:
                    logger.error(f"RF ranking stability plots failed: {e}")
                except Exception as e:
                    logger.warning(f"RF ranking stability plots failed with unexpected error: {e}")
            except (MemoryError, RuntimeError) as e:
                logger.error(f"RF permutation importance failed: {e}")
            except Exception as e:
                logger.warning(f"RF permutation importance failed with unexpected error: {e}")

        # Optional SHAP analysis for a global RF model
        if 'pooled_rf' in locals() and CONFIG["flags"]["run_shap"]:
            try:
                run_shap_rf_global(X_all, y_all, feat_names, best_params_rf_map, plots_dir, seed)
            except ImportError:
                logger.info("SHAP not installed; skipping SHAP analysis")
            except (MemoryError, RuntimeError) as e:
                logger.error(f"RF SHAP analysis failed: {e}")
            except Exception as e:
                logger.warning(f"RF SHAP analysis failed with unexpected error: {e}")
    else:
        logger.info("Skipping interpretability analyses: no models available")

    # Incremental validity and confound-control analysis (temperature, trial index, block)
    try:
        # Load temperature, trial numbers, and block info aligned to meta
        temps_all, trials_all, blocks_all = _aggregate_temperature_trial_and_block(meta, deriv_root, task)
        # Attach to RF predictions using the test indices order
        pred_rf["temperature"] = temps_all[np.asarray(test_indices_rf)]
        pred_rf["trial_number"] = trials_all[np.asarray(test_indices_rf)]
        pred_rf["block_id"] = blocks_all[np.asarray(test_indices_rf)]
        # Residuals vs Temperature / Trial number diagnostics
        try:
            plot_residuals_vs_covariate(
                y_true_rf, y_pred_rf,
                covariate=temps_all[np.asarray(test_indices_rf)],
                cov_name="Temperature",
                save_path=plots_dir / "residuals_vs_temperature.png",
            )
        except Exception as e:
            logger.warning(f"Residuals vs Temperature plot failed: {e}")
        try:
            plot_residuals_vs_covariate(
                y_true_rf, y_pred_rf,
                covariate=trials_all[np.asarray(test_indices_rf)],
                cov_name="Trial number",
                save_path=plots_dir / "residuals_vs_trial_number.png",
            )
        except Exception as e:
            logger.warning(f"Residuals vs Trial number plot failed: {e}")

        # Temperature-only LOSO regression (standardized preprocessing + Ridge)
        try:
            X_temp = pd.DataFrame({"temperature": temps_all})
            temp_pipe = _create_base_preprocessing_pipeline(include_scaling=True)
            temp_pipe.steps.append(("ridge", Ridge(alpha=1.0)))
            temp_grid = {"ridge__alpha": CONFIG["models"]["temperature_only"]["ridge_alpha_grid"]}
            y_true_t, y_pred_t, groups_t, test_idx_t, fold_ids_t = _nested_loso_predictions(
                X=X_temp, y=y_all, groups=groups,
                pipe=temp_pipe, param_grid=temp_grid,
                inner_cv_splits=int(CONFIG["cv"]["inner_splits"]), n_jobs=n_jobs, seed=seed,
                best_params_log_path=results_dir / CONFIG["paths"]["best_params"]["temperature_only"],
                model_name="TempOnly",
                outer_n_jobs=outer_n_jobs,
            )
            pooled_t, per_subj_t = compute_metrics(y_true_t, y_pred_t, np.asarray(groups_t))
            pred_t = pd.DataFrame({
                "subject_id": groups_t,
                "trial_id": meta.loc[test_idx_t, "trial_id"].values,
                "y_true": y_true_t,
                "y_pred": y_pred_t,
                "fold": fold_ids_t,
                "model": "TemperatureOnly",
                "temperature": temps_all[np.asarray(test_idx_t)],
            })
            _ensure_dir((results_dir / CONFIG["paths"]["predictions"]["temperature_only"]).parent)
            pred_t.to_csv(results_dir / CONFIG["paths"]["predictions"]["temperature_only"], sep="\t", index=False)
            _ensure_dir((results_dir / CONFIG["paths"]["per_subject_metrics"]["temperature_only"]).parent)
            per_subj_t.to_csv(results_dir / CONFIG["paths"]["per_subject_metrics"]["temperature_only"], sep="\t", index=False)
            # Save Temperature-only LOSO test indices explicitly
            try:
                idx_t = pd.DataFrame({
                    "subject_id": groups_t,
                    "trial_id": meta.loc[test_idx_t, "trial_id"].values,
                    "fold": fold_ids_t,
                })
                _ensure_dir((results_dir / CONFIG["paths"]["indices"]["temperature_only"]).parent)
                idx_t.to_csv(results_dir / CONFIG["paths"]["indices"]["temperature_only"], sep="\t", index=False)
            except Exception as e:
                logger.warning(f"Failed to save Temperature-only LOSO indices: {e}")
        except Exception as e:
            logger.warning(f"Temperature-only LOSO failed: {e}")
            pooled_t = {"pearson_r": float("nan"), "r2": float("nan")}
            pred_t = None

        # Partial correlation r(y_true, y_pred_rf | temperature)
        r_partial = _partial_corr_xy_given_z(
            x=pred_rf["y_pred"].to_numpy(),
            y=pred_rf["y_true"].to_numpy(),
            z=pred_rf["temperature"].to_numpy(),
        )
        r2_partial = float(r_partial ** 2) if np.isfinite(r_partial) else float("nan")
        # Multi-covariate partial r controlling for temperature, trial number, block_id, and subject mean rating
        try:
            Z_multi = np.c_[
                pred_rf["temperature"].to_numpy(),
                pred_rf["trial_number"].to_numpy(),
                pred_rf["block_id"].to_numpy(),
                pred_rf.groupby("subject_id")["y_true"].transform("mean").to_numpy(),
            ]
            r_partial_multi = _partial_corr_xy_given_Z(
                x=pred_rf["y_pred"].to_numpy(),
                y=pred_rf["y_true"].to_numpy(),
                Z=Z_multi,
            )
            r2_partial_multi = float(r_partial_multi ** 2) if np.isfinite(r_partial_multi) else float("nan")
        except Exception:
            r_partial_multi, r2_partial_multi = float("nan"), float("nan")

        # Delta r and Delta R2 vs temperature-only
        r_rf = float(pooled_rf.get("pearson_r", np.nan))
        r_t = float(pooled_t.get("pearson_r", np.nan)) if isinstance(pooled_t, dict) else np.nan
        delta_r = float(r_rf - r_t) if np.isfinite(r_rf) and np.isfinite(r_t) else float("nan")
        r2_t = float(pooled_t.get("r2", np.nan)) if isinstance(pooled_t, dict) else np.nan
        r2_rf_full = float(pooled_rf.get("r2", np.nan))
        delta_r2_full_minus_temp = float(r2_rf_full - r2_t) if np.isfinite(r2_rf_full) and np.isfinite(r2_t) else float("nan")
        delta_r2 = r2_partial  # incremental variance accounted for beyond temperature (partial r^2)
        r2_combined_est = float(np.clip((r2_t if np.isfinite(r2_t) else 0.0) + (r2_partial if np.isfinite(r2_partial) else 0.0), 0.0, 1.0))

        # Bootstrap CIs (subject-level cluster bootstrap)
        def _metric_delta_r(d: pd.DataFrame) -> float:
            # compute r for RF and temp-only on available rows
            r_rf_i, _ = _safe_pearsonr(d["y_true"].to_numpy(), d["y_pred_rf"].to_numpy())
            r_t_i, _ = _safe_pearsonr(d["y_true"].to_numpy(), d["y_pred_temp"].to_numpy())
            return float(r_rf_i - r_t_i)

        def _metric_partial_r(d: pd.DataFrame) -> float:
            return float(_partial_corr_xy_given_z(d["y_pred_rf"].to_numpy(), d["y_true"].to_numpy(), d["temperature"].to_numpy()))

        def _metric_partial_r_multi(d: pd.DataFrame) -> float:
            try:
                Zm = np.c_[
                    d["temperature"].to_numpy(),
                    d["trial_number"].to_numpy(),
                    d["block_id"].to_numpy(),
                    d.groupby("subject_id")["y_true"].transform("mean").to_numpy(),
                ]
                return float(_partial_corr_xy_given_Z(d["y_pred_rf"].to_numpy(), d["y_true"].to_numpy(), Zm))
            except Exception:
                return float("nan")

        # Merge RF and Temp-only predictions for the same trials if available
        df_boot = pred_rf.copy()
        df_boot = df_boot.rename(columns={"y_pred": "y_pred_rf"})
        if pred_t is not None:
            key_cols = ["subject_id", "trial_id"]
            df_boot = df_boot.merge(pred_t[key_cols + ["y_pred", "temperature"]].rename(columns={"y_pred": "y_pred_temp"}),
                                    on=key_cols, how="left")
        else:
            df_boot["y_pred_temp"] = temps_all[np.asarray(test_indices_rf)]  # raw temperature as fallback

        # Ensure column present
        if "temperature" not in df_boot.columns:
            df_boot["temperature"] = temps_all[np.asarray(test_indices_rf)]

        # Compute CIs
        delta_r_est, delta_r_ci = _cluster_bootstrap_subjects(df_boot, "subject_id", n_boot=int(CONFIG["analysis"]["bootstrap_n"]), seed=seed, func=_metric_delta_r)
        r_partial_est, r_partial_ci = _cluster_bootstrap_subjects(df_boot, "subject_id", n_boot=int(CONFIG["analysis"]["bootstrap_n"]), seed=seed + 1, func=_metric_partial_r)
        r_partial_multi_est, r_partial_multi_ci = _cluster_bootstrap_subjects(df_boot, "subject_id", n_boot=int(CONFIG["analysis"]["bootstrap_n"]), seed=seed + 2, func=_metric_partial_r_multi)

        # Plots
        try:
            # Incremental bar: R2_temp vs R2_temp+EEG
            plt.figure(figsize=(5, 4), dpi=150)
            bars = ["TempOnly R²", "Temp+EEG R²"]
            vals = [r2_t if np.isfinite(r2_t) else 0.0, r2_combined_est]
            _df_iv = pd.DataFrame({"bar": bars, "val": vals})
            sns.barplot(data=_df_iv, x="bar", y="val", hue="bar", palette=["#bde0fe", "#80ed99"], legend=False)
            plt.ylim(0, 1)
            plt.title("Incremental validity (ΔR² ≈ partial r²)")
            plt.tight_layout()
            plt.savefig(plots_dir / "incremental_bar.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Incremental bar plot failed: {e}")

        try:
            # Partial correlation residual scatter
            d = df_boot.dropna(subset=["y_true", "y_pred_rf", "temperature"]).copy()
            Z = np.c_[np.ones(len(d)), d["temperature"].to_numpy()]
            # residualize
            bx, *_ = np.linalg.lstsq(Z, d["y_pred_rf"].to_numpy(), rcond=None)
            by, *_ = np.linalg.lstsq(Z, d["y_true"].to_numpy(), rcond=None)
            x_res = d["y_pred_rf"].to_numpy() - Z @ bx
            y_res = d["y_true"].to_numpy() - Z @ by
            r_p, _ = _safe_pearsonr(x_res, y_res)
            plt.figure(figsize=(5, 4), dpi=150)
            plt.scatter(x_res, y_res, alpha=0.5, edgecolors="none")
            if len(x_res) >= 2:
                m, c = np.polyfit(x_res, y_res, 1)
                xs = np.linspace(np.nanmin(x_res), np.nanmax(x_res), 100)
                plt.plot(xs, m * xs + c, color="orange", label=f"fit r={r_p:.2f}")
                plt.legend()
            plt.xlabel("RF prediction residual (| temp)")
            plt.ylabel("Rating residual (| temp)")
            plt.title("Partial correlation: pred vs rating | temperature")
            plt.tight_layout()
            plt.savefig(plots_dir / "partial_corr_pred_rating_given_temp.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Partial correlation plot failed: {e}")

        try:
            # Error by temperature level (use unique temps or quantile bins)
            d = df_boot.copy()
            d["abs_err"] = np.abs(d["y_true"] - d["y_pred_rf"])
            d["sq_err"] = (d["y_true"] - d["y_pred_rf"]) ** 2
            # bin temperatures into unique rounded or quantile bins if too many levels
            temp_vals = d["temperature"].to_numpy()
            n_unique = np.unique(temp_vals[~np.isnan(temp_vals)]).size
            if n_unique <= 8:
                d["temp_bin"] = d["temperature"].round(1).astype(str)
            else:
                bins = np.unique(np.quantile(temp_vals[~np.isnan(temp_vals)], np.linspace(0, 1, 6)))
                if len(bins) < 3:
                    bins = np.linspace(np.nanmin(temp_vals), np.nanmax(temp_vals), 6)
                d["temp_bin"] = pd.cut(d["temperature"], bins=bins, include_lowest=True).astype(str)

            # Compute MAE/RMSE with subject bootstrap CIs per bin
            stats = []
            for tb, dfb in d.groupby("temp_bin"):
                def _mae_bin(xdf):
                    return _mae((xdf["y_true"] - xdf["y_pred_rf"]).to_numpy())
                def _rmse_bin(xdf):
                    return _rmse((xdf["y_true"] - xdf["y_pred_rf"]).to_numpy())
                mae_est, mae_ci = _cluster_bootstrap_subjects(dfb, "subject_id", n_boot=int(CONFIG["analysis"]["bootstrap_n"]), seed=seed + 2, func=_mae_bin)
                rmse_est, rmse_ci = _cluster_bootstrap_subjects(dfb, "subject_id", n_boot=int(CONFIG["analysis"]["bootstrap_n"]), seed=seed + 3, func=_rmse_bin)
                stats.append({"temp_bin": tb, "mae": mae_est, "mae_ci_low": mae_ci[0], "mae_ci_high": mae_ci[1],
                              "rmse": rmse_est, "rmse_ci_low": rmse_ci[0], "rmse_ci_high": rmse_ci[1]})
            stats_df = pd.DataFrame(stats)
            stats_df = stats_df.sort_values("temp_bin")
            plt.figure(figsize=(7, 4), dpi=150)
            # Calculate error bars, ensuring non-negative values
            mae_err_low = np.maximum(0, stats_df["mae"] - stats_df["mae_ci_low"])
            mae_err_high = np.maximum(0, stats_df["mae_ci_high"] - stats_df["mae"])
            rmse_err_low = np.maximum(0, stats_df["rmse"] - stats_df["rmse_ci_low"])
            rmse_err_high = np.maximum(0, stats_df["rmse_ci_high"] - stats_df["rmse"])
            
            plt.errorbar(range(len(stats_df)), stats_df["mae"],
                         yerr=[mae_err_low, mae_err_high],
                         fmt="o-", label="MAE")
            plt.errorbar(range(len(stats_df)), stats_df["rmse"],
                         yerr=[rmse_err_low, rmse_err_high],
                         fmt="s-", label="RMSE")
            plt.xticks(range(len(stats_df)), stats_df["temp_bin"].tolist(), rotation=45)
            plt.ylabel("Error")
            plt.xlabel("Temperature bin")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "error_by_temperature.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Error-by-temperature plot failed: {e}")

        try:
            # Error drift over trial index: bin trial_number into quintiles
            d = df_boot.copy()
            d["abs_err"] = np.abs(d["y_true"] - d["y_pred_rf"])
            tn = d["trial_number"].to_numpy()
            bins = np.unique(np.quantile(tn[~np.isnan(tn)], np.linspace(0, 1, 6))) if np.isfinite(tn).any() else np.array([0, 1])
            if len(bins) < 3:
                bins = np.linspace(np.nanmin(tn), np.nanmax(tn), 6)
            d["trial_bin"] = pd.cut(d["trial_number"], bins=bins, include_lowest=True).astype(str)
            drift = []
            for tb, dfb in d.groupby("trial_bin"):
                def _mae_bin(xdf):
                    return _mae((xdf["y_true"] - xdf["y_pred_rf"]).to_numpy())
                mae_est, mae_ci = _cluster_bootstrap_subjects(dfb, "subject_id", n_boot=int(CONFIG["analysis"]["bootstrap_n"]), seed=seed + 4, func=_mae_bin)
                drift.append({"trial_bin": tb, "mae": mae_est, "mae_ci_low": mae_ci[0], "mae_ci_high": mae_ci[1]})
            drift_df = pd.DataFrame(drift).sort_values("trial_bin")
            plt.figure(figsize=(7, 4), dpi=150)
            # Calculate error bars, ensuring non-negative values
            mae_err_low = np.maximum(0, drift_df["mae"] - drift_df["mae_ci_low"])
            mae_err_high = np.maximum(0, drift_df["mae_ci_high"] - drift_df["mae"])
            
            plt.errorbar(range(len(drift_df)), drift_df["mae"],
                         yerr=[mae_err_low, mae_err_high],
                         fmt="o-")
            plt.xticks(range(len(drift_df)), drift_df["trial_bin"].tolist(), rotation=45)
            plt.ylabel("MAE")
            plt.xlabel("Trial index bin")
            plt.title("Error drift over trial index")
            plt.tight_layout()
            plt.savefig(plots_dir / "error_by_trial_idx.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Error-drift plot failed: {e}")

        # Save summary for incremental validity
        inc_summary = {
            "RandomForest": {"pearson_r": r_rf, "r2": float(pooled_rf.get("r2", np.nan))},
            "TemperatureOnly": {"pearson_r": r_t, "r2": r2_t},
            "delta_r": {"estimate": delta_r, "ci95": [float(delta_r_ci[0]), float(delta_r_ci[1])]},
            "partial_r_given_temperature": {"estimate": float(r_partial_est), "ci95": [float(r_partial_ci[0]), float(r_partial_ci[1])]},
            "partial_r_given_temp_trial_subjectmean": {"estimate": float(r_partial_multi_est), "ci95": [float(r_partial_multi_ci[0]), float(r_partial_multi_ci[1])]},
            "delta_r2_incremental": {"estimate": float(r2_partial), "note": "≈ partial r^2 (unique variance beyond temperature)"},
            "delta_r2_incremental_multi": {"estimate": float(r2_partial_multi), "note": "≈ partial r^2 (unique variance beyond temp, trial, subj mean)"},
            "delta_r2_full_minus_temp": float(delta_r2_full_minus_temp),
            "combined_r2_estimated": float(r2_combined_est),
        }
        _ensure_dir((results_dir / CONFIG["paths"]["summaries"]["incremental"]).parent)
        with open(results_dir / CONFIG["paths"]["summaries"]["incremental"], "w", encoding="utf-8") as f:
            json.dump(inc_summary, f, indent=2)
        logger.info(f"Saved incremental validity summary at {results_dir / CONFIG['paths']['summaries']['incremental']}")
    except Exception as e:
        logger.warning(f"Incremental validity analysis failed: {e}")

    # 3) Baselines: global mean and diagnostic-only subject-test mean
    y_true_bg, y_pred_bg, groups_bg, test_idx_bg, fold_bg = _loso_baseline_predictions(y_all, groups, mode="global")
    pooled_bg, per_subj_bg = compute_metrics(y_true_bg, y_pred_bg, np.asarray(groups_bg))
    logger.info(f"Baseline (global mean) pooled: r={pooled_bg['pearson_r']:.3f}, R2={pooled_bg['r2']:.3f}")
    pred_bg = pd.DataFrame({
        "subject_id": groups_bg,
        "trial_id": meta.loc[test_idx_bg, "trial_id"].values,
        "y_true": y_true_bg,
        "y_pred": y_pred_bg,
        "fold": fold_bg,
        "model": "BaselineGlobal",
    })
    _ensure_dir((results_dir / CONFIG["paths"]["predictions"]["baseline_global"]).parent)
    pred_bg.to_csv(results_dir / CONFIG["paths"]["predictions"]["baseline_global"], sep="\t", index=False)
    _ensure_dir((results_dir / CONFIG["paths"]["per_subject_metrics"]["baseline_global"]).parent)
    per_subj_bg.to_csv(results_dir / CONFIG["paths"]["per_subject_metrics"]["baseline_global"], sep="\t", index=False)
    # Save Baseline Global LOSO test indices
    try:
        idx_bg = pd.DataFrame({
            "subject_id": groups_bg,
            "trial_id": meta.loc[test_idx_bg, "trial_id"].values,
            "fold": fold_bg,
        })
        _ensure_dir((results_dir / CONFIG["paths"]["indices"]["baseline_global"]).parent)
        idx_bg.to_csv(results_dir / CONFIG["paths"]["indices"]["baseline_global"], sep="\t", index=False)
    except Exception as e:
        logger.warning(f"Failed to save Baseline Global LOSO indices: {e}")

    # Note: Removed diagnostic subject-test baseline to prevent data leakage
    # This baseline used test labels to predict themselves, which is invalid
    logger.info("Diagnostic subject-test baseline removed to prevent data leakage.")

    # 4) Advanced: covariance-based decoding (optional if pyriemann available)
    pooled_riem = None
    if _check_pyriemann() and CONFIG["flags"]["run_riemann"]:
        try:
            y_true_r, y_pred_r, pooled_riem, per_subj_riem = loso_riemann_regression(
                deriv_root=deriv_root,
                subjects=subjects,
                task=task,
                results_dir=results_dir,
                n_jobs=n_jobs,
                seed=seed,
                outer_n_jobs=outer_n_jobs,
            )
        except Exception as e:
            logger.warning(f"Riemann model encountered an issue and was skipped: {e}")
            pooled_riem = None
    else:
        logger.warning("pyriemann not installed; skipping Model 2. Install with `pip install pyriemann`.")

    # Riemannian covariance-based insights (if available)
    if _check_pyriemann():
        try:
            riemann_visualize_cov_bins(deriv_root=deriv_root, subjects=subjects, task=task, plots_dir=plots_dir,
                                       plateau_window=tuple(CONFIG["analysis"]["riemann"]["plateau_window"]))
        except Exception as e:
            logger.warning(f"Riemann visualization failed: {e}")
        try:
            run_riemann_band_limited_decoding(
                deriv_root=deriv_root,
                subjects=subjects,
                task=task,
                results_dir=results_dir,
                plots_dir=plots_dir,
                bands=CONFIG["analysis"]["riemann"]["bands"],
                n_jobs=n_jobs,
                seed=seed,
                outer_n_jobs=outer_n_jobs,
            )
        except Exception as e:
            logger.warning(f"Riemann band-limited decoding failed: {e}")
        try:
            run_riemann_sliding_window(
                deriv_root=deriv_root,
                subjects=subjects,
                task=task,
                results_dir=results_dir,
                plots_dir=plots_dir,
                plateau_window=tuple(CONFIG["analysis"]["riemann"]["plateau_window"]),
                window_len=float(CONFIG["analysis"]["riemann"]["sliding_window"]["window_len"]),
                step=float(CONFIG["analysis"]["riemann"]["sliding_window"]["step"]),
                n_jobs=n_jobs,
                seed=seed,
                outer_n_jobs=outer_n_jobs,
            )
        except Exception as e:
            logger.warning(f"Riemann sliding-window analysis failed: {e}")

    # Subject-ID decodability sanity check
    try:
        subject_id_decodability_auc_plot(X_all, groups, plots_dir / "subject_id_auc.png", 
                                        results_dir=results_dir, seed=seed)
    except Exception as e:
        logger.warning(f"Subject-ID decodability check failed: {e}")

    # Bootstrap CIs for pooled metrics (across subjects)
    try:
        bootstrap_results = {}
        # Collect available prediction DataFrames
        model_preds = []
        if 'pred_en' in locals():
            model_preds.append(("ElasticNet", pred_en))
        if 'pred_wen' in locals():
            model_preds.append(("ElasticNetWithinKFold", pred_wen))
        if 'pred_rf' in locals():
            model_preds.append(("RandomForest", pred_rf))
        if 'pred_wrf' in locals():
            model_preds.append(("RandomForestWithinKFold", pred_wrf))
        if 'pred_bg' in locals():
            model_preds.append(("BaselineGlobal", pred_bg))
        for name, df_pred in model_preds:
            res = _bootstrap_pooled_metrics_by_subject(df_pred[["subject_id", "y_true", "y_pred"]].copy(), seed=seed)
            bootstrap_results[name] = res
        _ensure_dir(results_dir)
        with open(results_dir / "summary_bootstrap.json", "w", encoding="utf-8") as f:
            json.dump(bootstrap_results, f, indent=2)
        logger.info(f"Saved bootstrap CIs at {results_dir / 'summary_bootstrap.json'}")
    except Exception as e:
        logger.warning(f"Bootstrap CI computation failed: {e}")

    # Rank-based robustness: Spearman's rho vs Pearson's r
    try:
        models_stats = []
        if 'pred_en' in locals():
            models_stats.append((_per_subject_pearson_and_spearman(pred_en), "ElasticNet", "#1f77b4"))
        if 'pred_rf' in locals():
            models_stats.append((_per_subject_pearson_and_spearman(pred_rf), "RandomForest", "#ff7f0e"))
        if 'pred_wen' in locals():
            models_stats.append((_per_subject_pearson_and_spearman(pred_wen), "EN-WithinKFold", "#2ca02c"))
        if 'pred_wrf' in locals():
            models_stats.append((_per_subject_pearson_and_spearman(pred_wrf), "RF-WithinKFold", "#d62728"))
        plot_spearman_vs_pearson(models_stats, plots_dir / "spearman_vs_pearson.png")
    except Exception as e:
        logger.warning(f"Spearman vs Pearson plotting failed: {e}")

    # Results dashboard TSV (wide)
    try:
        _ensure_dir((results_dir / CONFIG["paths"]["summaries"]["all_metrics_wide"]).parent)
        build_all_metrics_wide(results_dir, results_dir / CONFIG["paths"]["summaries"]["all_metrics_wide"])
    except Exception as e:
        logger.warning(f"Building all_metrics_wide.tsv failed: {e}")

    # Summary JSON
    summary = {
        "BaselineGlobal": pooled_bg,
        "ElasticNet": pooled_en,
        "ElasticNetWithinKFold": (pooled_wen if 'pooled_wen' in locals() else None),
        "RandomForest": pooled_rf,
        "RandomForestWithinKFold": (pooled_wrf if 'pooled_wrf' in locals() else None),
        "Riemann": pooled_riem,
        "n_trials": int(len(X_all)),
        "n_subjects": int(len(np.unique(groups))),
        "n_features": int(X_all.shape[1]),
        "versions": info,
    }
    summary_path = results_dir / CONFIG["paths"]["summaries"]["summary"]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary at {summary_path}")
    logger.info(
        "Artifacts: "
        f"pred_en={results_dir / CONFIG['paths']['predictions']['elasticnet_loso']} | "
        f"pred_rf={results_dir / CONFIG['paths']['predictions']['rf_loso']} | "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Decode subjective pain from EEG features with LOSO cross-validation.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject IDs (e.g., 001 002) or 'all'")
    parser.add_argument("--task", default=TASK, help="Task label (default from config)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for inner CV (GridSearchCV)")
    parser.add_argument("--outer_n_jobs", type=int, default=1, help="Parallel jobs for outer LOSO folds (processes)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed")
    # Best-params JSONL handling
    parser.add_argument("--best_params_mode", choices=["append", "truncate", "run_scoped"], default="truncate",
                        help="How to handle best-params JSONL: append to existing, truncate on run start, or write to run-scoped file.")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run identifier used when best_params_mode=run_scoped.")
    # Heavy computation controls
    parser.add_argument("--n_perm_quick", type=int, default=int(CONFIG["analysis"]["n_perm_quick"]), help="Quick (no-refit) permutation draws")
    parser.add_argument("--n_perm_refit", type=int, default=int(CONFIG["analysis"]["n_perm_refit"]), help="Refit-based permutation draws")
    parser.add_argument("--perm_refit_n_jobs", type=int, default=1, help="Parallel jobs for refit-based permutation loop")
    parser.add_argument("--rf_perm_repeats", type=int, default=int(CONFIG["analysis"]["rf_perm_importance_repeats"]), help="Repeats for RF permutation importance")
    parser.add_argument("--bootstrap_n", type=int, default=int(CONFIG["analysis"]["bootstrap_n"]), help="Bootstrap resamples for pooled metrics")
    parser.add_argument("--inner_splits", type=int, default=int(CONFIG["cv"]["inner_splits"]), help="Inner CV splits for nested LOSO/GridSearchCV")
    # Feature toggles
    parser.add_argument("--no-within", action="store_true", help="Disable within-subject KFold analyses")
    parser.add_argument("--no-riemann", action="store_true", help="Disable Riemann analyses")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP analysis")
    parser.add_argument("--montage", type=str, default=CONFIG["viz"]["montage"],
                        help="Montage selection: standard montage name (e.g., 'standard_1020'), 'bids_auto', or 'bids:<path-to-electrodes.tsv>'")
    parser.add_argument("--coef_agg", choices=["abs", "signed"], default=CONFIG["viz"]["coef_agg"],
                        help="Aggregation for ElasticNet coefficient topomaps: 'abs' (mean |coef|) or 'signed' (mean signed coef)")

    args = parser.parse_args()

    # seeds
    RANDOM_STATE = int(args.seed)
    np.random.seed(RANDOM_STATE)
    pyrandom.seed(RANDOM_STATE)

    # Apply CLI configs
    BEST_PARAMS_MODE = args.best_params_mode
    RUN_ID = args.run_id
    CONFIG["analysis"]["n_perm_quick"] = int(args.n_perm_quick)
    CONFIG["analysis"]["n_perm_refit"] = int(args.n_perm_refit)
    CONFIG["analysis"]["rf_perm_importance_repeats"] = int(args.rf_perm_repeats)
    CONFIG["analysis"]["perm_refit_n_jobs"] = int(args.perm_refit_n_jobs)
    CONFIG["analysis"]["bootstrap_n"] = int(args.bootstrap_n)
    CONFIG["cv"]["inner_splits"] = int(args.inner_splits)
    # Visualization/interpretability options
    CONFIG["viz"]["montage"] = args.montage
    CONFIG["viz"]["coef_agg"] = args.coef_agg
    if args.no_within:
        CONFIG["flags"]["run_within_subject_kfold"] = False
    if args.no_riemann:
        CONFIG["flags"]["run_riemann"] = False
    if args.no_shap:
        CONFIG["flags"]["run_shap"] = False

    subs = None if args.subjects in (None, [], ["all"]) else args.subjects
    main(subjects=subs, task=args.task, n_jobs=args.n_jobs, seed=RANDOM_STATE, outer_n_jobs=int(args.outer_n_jobs))
