#!/usr/bin/env python
"""
Train machine learning models that predict fMRI Neurologic Pain Signature (NPS) beta response scores
from EEG oscillatory band power features (alpha, beta, gamma). The script aligns outputs from the EEG
and fMRI pipelines, performs nested cross-validation, and saves predictions, metrics, and fitted models.
"""

import argparse
import importlib
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUPPORTED_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
DEFAULT_BANDS = ("alpha", "beta", "gamma")


PACKAGE_VERSION_TARGETS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "torch": "torch",
}

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]


@dataclass
class SubjectDataset:
    subject: str
    data: pd.DataFrame
    feature_columns: List[str]
    dropped_trials: List[Dict[str, float]]


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("eeg_to_nps")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "train.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict fMRI NPS beta responses from EEG oscillatory features."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store outputs. Defaults to machine_learning/outputs/eeg_to_nps_<timestamp>.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject identifiers (with or without sub- prefix). Default uses all available.",
    )
    parser.add_argument(
        "--bands",
        nargs="*",
        default=list(DEFAULT_BANDS),
        help="EEG frequency bands to include (subset of %s)." % (SUPPORTED_BANDS,),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["elasticnet", "random_forest"],
        help="Model names to evaluate (elasticnet, random_forest).",
    )
    parser.add_argument(
        "--include-temperature",
        action="store_true",
        help="Include trial temperature as an additional predictor.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for grid searches (-1 uses all cores).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits and models.",
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=None,
        help="Random seed for label permutations (defaults to --random-state when omitted).",
    )

    parser.add_argument(
        "--permutation-per-model",
        type=int,
        default=0,
        help="Number of label permutations to run per model for R^2 significance testing (0 disables).",
    )
    return parser.parse_args()


def discover_subjects(eeg_deriv_root: Path, fmri_outputs_root: Path) -> List[str]:
    eeg_subjects = {
        path.parent.parent.parent.name
        for path in eeg_deriv_root.glob("sub-*/eeg/features/features_eeg_direct.tsv")
    }
    nps_scores_dir = fmri_outputs_root / "nps_scores"
    fmri_subjects = {
        path.parent.name for path in nps_scores_dir.glob("sub-*/trial_br.tsv")
    }
    return sorted(eeg_subjects & fmri_subjects)


def sanitize_for_json(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    """Serialize payload to JSON with NaN/Inf made JSON-friendly."""
    path.write_text(json.dumps(sanitize_for_json(payload), indent=2), encoding="utf-8")


def _stringify_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _stringify_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_stringify_for_json(v) for v in value]
    return value


def _collect_package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for module_name, display_name in PACKAGE_VERSION_TARGETS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            versions[display_name] = None
            continue
        version = getattr(module, "__version__", None)
        if version is None and hasattr(module, "version"):
            attr = getattr(module, "version")
            try:
                version = attr() if callable(attr) else attr
            except Exception:
                version = None
        versions[display_name] = str(version) if version is not None else None
    return versions


def _collect_thread_limits() -> Dict[str, Optional[str]]:
    return {var: os.environ.get(var) for var in THREAD_ENV_VARS}


def _gather_git_metadata(repo_root: Path) -> Dict[str, Optional[str]]:
    metadata: Dict[str, Optional[str]] = {
        "commit": None,
        "branch": None,
        "dirty": None,
    }

    def _run_git(args: Sequence[str]) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        output = completed.stdout.strip()
        return output or None

    metadata["commit"] = _run_git(["rev-parse", "HEAD"])
    metadata["branch"] = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    metadata["dirty"] = "true" if status else ("false" if status == "" else None)
    return metadata


def create_run_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    repo_root: Path,
    *,
    additional: Optional[Dict[str, Any]] = None,
) -> Path:
    cli_arguments = {k: _stringify_for_json(v) for k, v in vars(args).items()}
    manifest = {
        "created_at": datetime.now().isoformat(),
        "script": str(Path(__file__).resolve()),
        "working_directory": str(Path.cwd()),
        "cli_arguments": cli_arguments,
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
            "thread_env": _collect_thread_limits(),
            "random_state": getattr(args, "random_state", None),
            "permutation_seed": getattr(args, "permutation_seed", None),
        },
        "git": _gather_git_metadata(repo_root),
        "packages": _collect_package_versions(),
    }
    if additional:
        manifest["analysis"] = _stringify_for_json(additional)

    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def identify_missing_events(events: pd.DataFrame, ratings: Sequence[float]) -> List[int]:
    missing: List[int] = []
    ev_values = events["vas_final_coded_rating"].tolist()
    ratings_list = list(ratings)
    if len(ev_values) < len(ratings_list):
        raise ValueError(
            f"Events have fewer rows ({len(ev_values)}) than EEG features ({len(ratings_list)})."
        )

    rating_idx = 0
    for event_idx, event_rating in enumerate(ev_values):
        if rating_idx >= len(ratings_list):
            missing.extend(range(event_idx, len(ev_values)))
            break
        if math.isclose(event_rating, ratings_list[rating_idx], rel_tol=1e-5, abs_tol=1e-4):
            rating_idx += 1
        else:
            missing.append(event_idx)

    if rating_idx != len(ratings_list):
        raise ValueError(
            "Could not perfectly align events to EEG features (matched "
            f"{rating_idx} of {len(ratings_list)} ratings)."
        )

    expected_missing = len(ev_values) - len(ratings_list)
    if len(missing) != expected_missing:
        raise ValueError(
            f"Alignment discrepancy: expected {expected_missing} missing events, found {len(missing)}."
        )
    return missing


def select_direct_power_columns(columns: Sequence[str], bands: Sequence[str]) -> List[str]:
    prefixes = tuple(f"pow_{band}_" for band in bands)
    return sorted([col for col in columns if col.startswith(prefixes)])


def select_roi_power_columns(columns: Sequence[str], bands: Sequence[str]) -> List[str]:
    keep: List[str] = []
    for col in columns:
        for band in bands:
            prefix = f"{band}_power_"
            if col.startswith(prefix):
                keep.append(col)
                break
    return sorted(keep)


def load_subject_dataset(
    subject: str,
    eeg_deriv_root: Path,
    fmri_outputs_root: Path,
    bands: Sequence[str],
    logger: logging.Logger,
) -> SubjectDataset:
    subject_dir = eeg_deriv_root / subject / "eeg"
    features_dir = subject_dir / "features"
    features_path = features_dir / "features_eeg_direct.tsv"
    roi_path = subject_dir / f"{subject}_task-thermalactive_features_frame.tsv"
    target_path = features_dir / "target_vas_ratings.tsv"
    events_path = eeg_deriv_root.parent / subject / "eeg" / f"{subject}_task-thermalactive_events.tsv"
    trial_br_path = fmri_outputs_root / "nps_scores" / subject / "trial_br.tsv"

    for path in [features_path, roi_path, target_path, events_path, trial_br_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file for {subject}: {path}")

    direct = pd.read_csv(features_path, sep="\t")
    direct_cols = select_direct_power_columns(direct.columns, bands)
    if not direct_cols:
        sample_cols = list(direct.columns[:10])
        raise ValueError("No direct EEG power columns for bands %s in %s. Sample columns: %s" % (bands, features_path, sample_cols))
    direct = direct.loc[:, direct_cols].reset_index(drop=True)

    roi_df = pd.read_csv(roi_path, sep="\t")
    roi_cols = select_roi_power_columns(roi_df.columns, bands)
    roi = roi_df.loc[:, roi_cols].reset_index(drop=True) if roi_cols else pd.DataFrame(index=direct.index)

    feature_df = pd.concat([direct, roi], axis=1)

    for band in bands:
        band_cols = [col for col in direct_cols if col.startswith(f"pow_{band}_")]
        if band_cols:
            feature_df[f"pow_{band}_mean"] = direct[band_cols].mean(axis=1)
            feature_df[f"pow_{band}_std"] = direct[band_cols].std(axis=1)

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    target_df = pd.read_csv(target_path, sep="\t")
    target_values = pd.to_numeric(target_df.iloc[:, 0], errors="coerce")
    events_df = pd.read_csv(events_path, sep="\t")

    missing_indices = identify_missing_events(events_df, target_values.tolist())
    dropped_trials: List[Dict[str, float]] = []
    if missing_indices:
        dropped_view = events_df.iloc[missing_indices][["run", "trial_number", "stimulus_temp", "vas_final_coded_rating"]]
        dropped_trials = [
            {
                "run": int(row["run"]),
                "trial_number": int(row["trial_number"]),
                "temp_celsius": float(row["stimulus_temp"]),
                "vas_rating": float(row["vas_final_coded_rating"]),
            }
            for _, row in dropped_view.iterrows()
        ]
        events_aligned = events_df.drop(index=missing_indices).reset_index(drop=True)
    else:
        events_aligned = events_df.reset_index(drop=True)

    trial_br_df = pd.read_csv(trial_br_path, sep="\t")
    if missing_indices:
        # Try to filter out dropped trials from trial_br
        # Note: If trials were filtered at source (split_events_to_runs.py), they won't exist in trial_br
        mask = np.ones(len(trial_br_df), dtype=bool)
        for dropped in dropped_trials:
            idx = trial_br_df[
                (trial_br_df["run"] == dropped["run"]) &
                (trial_br_df["trial_idx_run"] == dropped["trial_number"] - 1)
            ].index
            if idx.empty:
                # Trial already filtered at source - this is expected
                logger.info(f"  Dropped trial (run={dropped['run']}, trial={dropped['trial_number']}) "
                           f"already filtered from fMRI outputs")
            else:
                mask[idx] = False
        trial_br_aligned = trial_br_df.loc[mask].reset_index(drop=True)
    else:
        trial_br_aligned = trial_br_df.reset_index(drop=True)

    if not len(feature_df) == len(events_aligned) == len(trial_br_aligned):
        raise ValueError(
            f"Alignment mismatch for {subject}: features {len(feature_df)}, events {len(events_aligned)}, "
            f"trial_br {len(trial_br_aligned)}."
        )

    metadata = pd.DataFrame(
        {
            "subject": subject,
            "run": events_aligned["run"].astype(int),
            "trial_idx_run": events_aligned["trial_number"].astype(int) - 1,
            "trial_idx_global": trial_br_aligned["trial_idx_global"].astype(int),
            "temp_celsius": trial_br_aligned["temp_celsius"].astype(float),
            "vas_rating": trial_br_aligned["vas_rating"].astype(float),
            "pain_binary": trial_br_aligned["pain_binary"].astype(int),
            "br_score": trial_br_aligned["br_score"].astype(float),
        }
    )

    data = pd.concat([metadata, feature_df.reset_index(drop=True)], axis=1)
    feature_columns = list(feature_df.columns)

    logger.info(
        "Loaded %d aligned trials for %s (dropped %d trials).",
        len(data),
        subject,
        len(dropped_trials),
    )

    return SubjectDataset(subject=subject, data=data, feature_columns=feature_columns, dropped_trials=dropped_trials)


def make_outer_splitter(groups: np.ndarray, random_state: int):
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        desc = f"LeaveOneGroupOut (n_groups={len(unique_groups)})"
        return LeaveOneGroupOut(), groups, desc
    n_splits = min(5, len(groups))
    if n_splits < 2:
        raise ValueError("Not enough samples to perform cross-validation.")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    desc = f"KFold(n_splits={n_splits}, shuffle=True, random_state={random_state}) on pooled trials"
    return splitter, None, desc


def make_inner_cv(run_groups: np.ndarray, random_state: int):
    unique_runs = np.unique(run_groups)
    if len(unique_runs) >= 3:
        n_splits = min(5, len(unique_runs))
        desc = f"GroupKFold(n_splits={n_splits}) on run labels"
        return GroupKFold(n_splits=n_splits), run_groups, desc
    n_samples = len(run_groups)
    n_splits = min(5, n_samples)
    if n_splits < 2:
        n_splits = 2
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    desc = f"KFold(n_splits={n_splits}, shuffle=True, random_state={random_state}) on pooled trials"
    return splitter, None, desc


def fold_group_label(groups: Optional[np.ndarray], test_idx: np.ndarray) -> str:
    if groups is None:
        return "pooled"
    values = np.unique(groups[test_idx])
    return ",".join(str(v) for v in values)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }
    try:
        metrics["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
    except Exception:
        metrics["pearson_r"] = float("nan")
    try:
        metrics["spearman_r"] = float(spearmanr(y_true, y_pred)[0])
    except Exception:
        metrics["spearman_r"] = float("nan")

    try:
        slope, intercept, r_value, p_value, std_err = linregress(y_pred, y_true)
        metrics.update(
            {
                "calibration_slope": float(slope),
                "calibration_intercept": float(intercept),
                "calibration_r": float(r_value),
                "calibration_p": float(p_value),
                "calibration_std_err": float(std_err),
            }
        )
    except Exception:
        metrics.update(
            {
                "calibration_slope": float("nan"),
                "calibration_intercept": float("nan"),
                "calibration_r": float("nan"),
                "calibration_p": float("nan"),
                "calibration_std_err": float("nan"),
            }
        )

    return metrics


def compute_group_metrics(pred_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Compute regression metrics for grouped predictions."""
    if not group_cols:
        raise ValueError('group_cols must contain at least one column name')
    rows: List[Dict[str, float]] = []
    for keys, grp in pred_df.groupby(list(group_cols), dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        metrics = compute_metrics(grp['br_true'].to_numpy(), grp['br_pred'].to_numpy())
        row = {col: key_tuple[idx] for idx, col in enumerate(group_cols)}
        row['n_trials'] = int(len(grp))
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)



def build_prediction_frame(data: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray,
                           model_name: str, fold_assignments: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Return a tidy dataframe combining metadata with true/predicted scores."""
    frame = data[['subject', 'run', 'trial_idx_run', 'trial_idx_global', 'temp_celsius', 'vas_rating', 'pain_binary']].copy()
    frame['br_true'] = y_true.to_numpy()
    frame['br_pred'] = y_pred
    frame['model'] = model_name
    if fold_assignments is not None:
        frame['cv_fold'] = fold_assignments
    return frame


def permutation_test_r2(
    *,
    model_name: str,
    builder,
    param_grid: Dict[str, Sequence],
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Sequence[str],
    meta: pd.DataFrame,
    outer_groups: np.ndarray,
    run_groups: np.ndarray,
    n_permutations: int,
    true_r2: float,
    random_state: Optional[int],
    n_jobs: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Estimate R^2 significance via label permutation with full nested CV retraining."""

    if n_permutations <= 0:
        raise ValueError("n_permutations must be a positive integer")

    if not isinstance(y, pd.Series):
        y_series = pd.Series(np.asarray(y), name="target")
    else:
        y_series = y.copy()

    rng = np.random.default_rng(random_state)
    null_scores = np.zeros(n_permutations, dtype=float)
    seed_display = random_state if random_state is not None else "random"
    logger.info(
        "Running permutation test with %d permutations for model %s (nested CV retraining, seed=%s)",
        n_permutations,
        model_name,
        seed_display,
    )

    log_check_interval = max(1, n_permutations // 10)

    for idx in range(n_permutations):
        permuted = pd.Series(
            rng.permutation(y_series.to_numpy()),
            index=y_series.index,
            name=y_series.name,
        )
        perm_result = nested_cv_evaluate(
            model_name=model_name,
            builder=builder,
            param_grid=param_grid,
            X=X,
            y=permuted,
            feature_names=feature_names,
            meta=meta,
            outer_groups=outer_groups,
            run_groups=run_groups,
            random_state=random_state,
            n_jobs=n_jobs,
            logger=logger,
            log_progress=False,
        )
        null_scores[idx] = perm_result["summary_metrics"]["r2"]
        if (idx + 1) % log_check_interval == 0 or (idx + 1) == n_permutations:
            logger.info(
                "Permutation progress: %d/%d (null mean R2=%.4f)",
                idx + 1,
                n_permutations,
                float(np.mean(null_scores[: idx + 1])),
            )

    p_value = (np.sum(null_scores >= true_r2) + 1) / (n_permutations + 1)
    summary = {
        "true_r2": float(true_r2),
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
        "null_quantiles": {
            "05": float(np.quantile(null_scores, 0.05)),
            "50": float(np.quantile(null_scores, 0.5)),
            "95": float(np.quantile(null_scores, 0.95)),
        },
        "n_permutations": int(n_permutations),
        "random_state": random_state,
    }
    return summary, null_scores




def compute_temperature_baseline_cv(
    temp: pd.Series,
    target: pd.Series,
    outer_groups: np.ndarray,
    random_state: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], str]:
    """Compute cross-validated temperature-only baseline metrics using the outer CV strategy."""

    if temp.isna().all():
        raise ValueError("Temperature series contains only NaN values; cannot compute baseline")

    baseline_X = temp.to_numpy().reshape(-1, 1)
    baseline_y = target.to_numpy()

    splitter, groups_used, desc = make_outer_splitter(outer_groups, random_state)
    predictions = np.zeros_like(baseline_y, dtype=float)

    for train_idx, test_idx in splitter.split(baseline_X, baseline_y, groups=groups_used):
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("reg", LinearRegression()),
            ]
        )
        model.fit(baseline_X[train_idx], baseline_y[train_idx])
        predictions[test_idx] = model.predict(baseline_X[test_idx])

    metrics = compute_metrics(baseline_y, predictions)
    logger.info(
        "Temperature-only baseline via %s | R2=%.3f | MAE=%.3f | RMSE=%.3f",
        desc,
        metrics["r2"],
        metrics["mae"],
        metrics["rmse"],
    )
    return metrics, desc



def nested_cv_evaluate(
    model_name: str,
    builder,
    param_grid: Dict[str, Sequence],
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Sequence[str],
    meta: pd.DataFrame,
    outer_groups: np.ndarray,
    run_groups: np.ndarray,
    random_state: int,
    n_jobs: int,
    logger: logging.Logger,
    log_progress: bool = True,
):
    predictions = np.zeros(len(y), dtype=float)
    fold_assignments = np.full(len(y), -1, dtype=int)
    fold_details: List[Dict[str, object]] = []

    outer_splitter, outer_groups_used, outer_desc = make_outer_splitter(outer_groups, random_state)
    if log_progress:
        logger.info("Model %s | outer CV strategy: %s", model_name, outer_desc)
    inner_desc_record: Optional[str] = None

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_splitter.split(X, y, groups=outer_groups_used)
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv, inner_groups, inner_desc = make_inner_cv(run_groups[train_idx], random_state)
        if inner_desc_record is None:
            inner_desc_record = inner_desc
            if log_progress:
                logger.info("Model %s | inner CV strategy: %s", model_name, inner_desc_record)

        estimator = builder(random_state, n_jobs)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="r2",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
        )
        if inner_groups is not None:
            search.fit(X_train, y_train, groups=inner_groups)
        else:
            search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        y_pred = best_estimator.predict(X_test)

        predictions[test_idx] = y_pred
        fold_assignments[test_idx] = fold_idx

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        try:
            pear = pearsonr(y_test, y_pred)[0]
        except Exception:
            pear = float("nan")
        try:
            spear = spearmanr(y_test, y_pred)[0]
        except Exception:
            spear = float("nan")

        temp_counts = (
            meta.iloc[test_idx]["temp_celsius"].value_counts().sort_index().to_dict()
            if "temp_celsius" in meta.columns
            else {}
        )

        fold_info = {
            "name": model_name,
            "fold": fold_idx,
            "outer_group": fold_group_label(outer_groups_used, test_idx),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "test_r2": float(r2),
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "test_explained_variance": float(evs),
            "test_pearson_r": float(pear),
            "test_spearman_r": float(spear),
            "best_params": search.best_params_,
            "test_temp_counts": temp_counts,
        }
        fold_details.append(fold_info)
        if log_progress:
            logger.info(
                "Model %s | fold %d | group %s | R2=%.3f | Pearson=%.3f",
                model_name,
                fold_idx,
                fold_info["outer_group"],
                r2,
                pear,
            )
            if temp_counts:
                logger.info(
                    "Model %s | fold %d temperature counts: %s",
                    model_name,
                    fold_idx,
                    temp_counts,
                )

    metrics = compute_metrics(y.to_numpy(), predictions)
    if log_progress:
        logger.info(
            "Model %s | overall R2=%.3f | Pearson=%.3f | Spearman=%.3f | MAE=%.3f",
            model_name,
            metrics["r2"],
            metrics.get("pearson_r", float("nan")),
            metrics.get("spearman_r", float("nan")),
            metrics["mae"],
        )

    return {
        "name": model_name,
        "predictions": predictions,
        "fold_assignments": fold_assignments,
        "fold_details": fold_details,
        "summary_metrics": metrics,
        "feature_names": list(feature_names),
        "outer_cv_desc": outer_desc,
        "inner_cv_desc": inner_desc_record or "Not determined",
    }





def fit_final_estimator(
    builder,
    param_grid: Dict[str, Sequence],
    X: pd.DataFrame,
    y: pd.Series,
    run_groups: np.ndarray,
    random_state: int,
    n_jobs: int,
):
    estimator = builder(random_state, n_jobs)
    unique_runs = np.unique(run_groups)
    if len(unique_runs) >= 3:
        cv = GroupKFold(n_splits=min(5, len(unique_runs)))
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="r2",
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(X, y, groups=run_groups)
        cv_desc = f"GroupKFold(n_splits={min(5, len(unique_runs))}) on run labels"
    else:
        n_splits = min(5, len(X))
        if n_splits < 2:
            n_splits = 2
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="r2",
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(X, y)
        cv_desc = f"KFold(n_splits={n_splits}, shuffle=True, random_state={random_state}) on pooled trials"
    return search.best_estimator_, search.best_params_, float(search.best_score_), cv_desc


def extract_feature_importance(model: Pipeline, feature_names: Sequence[str]) -> Optional[pd.DataFrame]:
    if "reg" in model.named_steps:
        reg = model.named_steps["reg"]
        coef = getattr(reg, "coef_", None)
        if coef is None:
            return None
        coef = np.asarray(coef)
        scaler = model.named_steps.get("scaler")
        if scaler is not None and hasattr(scaler, "scale_"):
            # Undo standard-scaling to aid interpretability. Centering effects may still shift the intercept.
            scale = np.asarray(scaler.scale_)
            scale[scale == 0] = 1.0
            coef = coef / scale
        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": coef,
                "importance_abs": np.abs(coef),
            }
        )
        return df.sort_values("importance_abs", ascending=False)
    if "rf" in model.named_steps:
        rf = model.named_steps["rf"]
        importances = np.asarray(rf.feature_importances_)
        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
                "importance_abs": np.abs(importances),
            }
        )
        return df.sort_values("importance_abs", ascending=False)
    return None


def build_elasticnet(random_state: int, _: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", ElasticNet(max_iter=5000, random_state=random_state)),
        ]
    )


ELASTICNET_PARAM_GRID = {
    "reg__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    "reg__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}


def build_random_forest(random_state: int, estimator_jobs: int) -> Pipeline:
    # Force n_jobs=1 to avoid nested parallelism with GridSearchCV
    # GridSearchCV handles outer parallelization; RF should run single-threaded per fold
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=600,
                    random_state=random_state,
                    n_jobs=1,  # Always 1 to avoid nested parallelism
                    bootstrap=True,
                ),
            ),
        ]
    )


RANDOM_FOREST_PARAM_GRID = {
    "rf__max_depth": [None, 12, 20],
    "rf__max_features": ["sqrt", "log2", 0.3],
    "rf__min_samples_leaf": [1, 2, 4],
}


MODEL_REGISTRY = {
    "elasticnet": (build_elasticnet, ELASTICNET_PARAM_GRID),
    "random_forest": (build_random_forest, RANDOM_FOREST_PARAM_GRID),
}


def main() -> None:
    args = parse_args()

    bands = tuple(dict.fromkeys(band.lower() for band in args.bands))
    for band in bands:
        if band not in SUPPORTED_BANDS:
            raise ValueError(f"Unsupported band '{band}'. Supported bands: {SUPPORTED_BANDS}.")
    if not bands:
        raise ValueError("No bands specified.")

    repo_root = Path(__file__).resolve().parents[1]
    default_output = Path(__file__).resolve().parent / "outputs" / f"eeg_to_nps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("EEG -> NPS training pipeline started.")
    logger.info("Using bands: %s", ", ".join(bands))

    eeg_deriv_root = repo_root / "eeg_pipeline" / "bids_output" / "derivatives"
    fmri_outputs_root = repo_root / "fmri_pipeline" / "NPS" / "outputs"

    if not eeg_deriv_root.exists():
        logger.error("EEG derivatives directory not found: %s", eeg_deriv_root)
        sys.exit(1)
    if not fmri_outputs_root.exists():
        logger.error("fMRI outputs directory not found: %s", fmri_outputs_root)
        sys.exit(1)

    available_subjects = discover_subjects(eeg_deriv_root, fmri_outputs_root)
    if not available_subjects:
        logger.error("No subjects with both EEG features and fMRI NPS scores were found.")
        sys.exit(1)
    logger.info("Available subjects with aligned data: %s", ", ".join(available_subjects))

    if args.subjects:
        requested = [s if s.startswith("sub-") else f"sub-{s}" for s in args.subjects]
        invalid = sorted(set(requested) - set(available_subjects))
        if invalid:
            logger.error("Requested subjects missing required data: %s", ", ".join(invalid))
            sys.exit(1)
        subjects = sorted(requested)
    else:
        subjects = available_subjects

    subject_results: List[SubjectDataset] = []
    feature_template: Optional[List[str]] = None
    drops_summary: Dict[str, List[Dict[str, float]]] = {}

    for subject in subjects:
        result = load_subject_dataset(subject, eeg_deriv_root, fmri_outputs_root, bands, logger)
        subject_results.append(result)
        if feature_template is None:
            feature_template = result.feature_columns
        elif feature_template != result.feature_columns:
            logger.error("Feature column mismatch detected for %s.", subject)
            sys.exit(1)
        if result.dropped_trials:
            drops_summary[subject] = result.dropped_trials

    if feature_template is None:
        logger.error("No feature columns detected.")
        sys.exit(1)

    data = pd.concat([res.data for res in subject_results], ignore_index=True)
    feature_columns = list(feature_template)
    if args.include_temperature and "temp_celsius" not in feature_columns:
        feature_columns.append("temp_celsius")

    X = data[feature_columns].copy()
    y = data["br_score"].copy()

    # Validate target values
    if y.isna().any():
        n_na = y.isna().sum()
        logger.error("Target variable (br_score) contains %d NaN values; cannot proceed.", n_na)
        sys.exit(1)
    if np.isinf(y).any():
        n_inf = np.isinf(y).sum()
        logger.error("Target variable (br_score) contains %d infinite values; cannot proceed.", n_inf)
        sys.exit(1)
    
    # Log target statistics for quality assurance
    logger.info("Target (br_score) statistics: min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
                y.min(), y.max(), y.mean(), y.std())

    if len(subjects) > 1:
        outer_groups = data["subject"].to_numpy()
        outer_group_level = "subject"
    else:
        outer_groups = data["run"].to_numpy()
        outer_group_level = "run"
    run_groups = data["run"].to_numpy()

    logger.info("Assembled dataset: %d trials, %d subjects, %d features.", len(data), len(subjects), len(feature_columns))

    feature_ratio = len(feature_columns) / max(len(data), 1)
    logger.info("Feature-to-sample ratio: %.2f (%d features / %d trials)", feature_ratio, len(feature_columns), len(data))
    if feature_ratio > (1.0 / 3.0):
        logger.warning("High feature-to-sample ratio may cause overfitting despite regularisation.")

    if "temp_celsius" in data.columns:
        temp_counts = data["temp_celsius"].value_counts().sort_index()
        logger.info("Temperature distribution (counts per condition):\n%s", temp_counts.to_string())
    else:
        temp_counts = pd.Series(dtype=int)

    logger.info("NPS br_score betas reflect the delayed (~5-7 s) hemodynamic response; EEG features are interpreted with this lag in mind.")

    plateau_window = None
    eeg_config_path = repo_root / "eeg_pipeline" / "eeg_config.yaml"
    if eeg_config_path.exists():
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(eeg_config_path.read_text())
            plateau_window = cfg.get("time_frequency_analysis", {}).get("plateau_window")
            if plateau_window is not None:
                logger.info("EEG plateau window from config: %s seconds", plateau_window)
        except Exception as exc:  # pragma: no cover - best-effort logging only
            logger.warning("Could not read EEG plateau window from %s: %s", eeg_config_path, exc)
    else:
        logger.warning("EEG config file not found at %s; plateau window not verified.", eeg_config_path)

    temperature_baseline_metrics: Optional[Dict[str, float]] = None
    temperature_baseline_desc: Optional[str] = None
    if args.include_temperature and "temp_celsius" in data.columns:
        temperature_baseline_metrics, temperature_baseline_desc = compute_temperature_baseline_cv(
            temp=data["temp_celsius"],
            target=y,
            outer_groups=outer_groups,
            random_state=args.random_state,
            logger=logger,
        )

    model_names = [name.lower() for name in args.models]
    for name in model_names:
        if name not in MODEL_REGISTRY:
            logger.error("Unknown model '%s'. Available models: %s", name, ", ".join(MODEL_REGISTRY.keys()))
            sys.exit(1)

    model_results: List[Dict[str, object]] = []
    summary_model_entries: List[Dict[str, object]] = []
    outer_cv_strategy_record: Optional[str] = None
    inner_cv_strategy_record: Optional[str] = None

    for name in model_names:
        builder, param_grid = MODEL_REGISTRY[name]
        grid_size = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
        if grid_size * 5 > len(X):
            logger.warning(
                "Model %s grid (%d combinations) is large relative to available trials (%d); consider simplifying the search space or switching to randomized search.",
                name,
                grid_size,
                len(X),
            )

        result = nested_cv_evaluate(
            model_name=name,
            builder=builder,
            param_grid=param_grid,
            X=X,
            y=y,
            feature_names=feature_columns,
            meta=data,
            outer_groups=outer_groups,
            run_groups=run_groups,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            logger=logger,
        )
        model_results.append(result)

        if outer_cv_strategy_record is None:
            outer_cv_strategy_record = result.get("outer_cv_desc")
        if inner_cv_strategy_record is None:
            inner_cv_strategy_record = result.get("inner_cv_desc")

        pred_df = build_prediction_frame(
            data=data,
            y_true=y,
            y_pred=result["predictions"],
            model_name=name,
            fold_assignments=result["fold_assignments"],
        )
        pred_path = output_dir / f"predictions_{name}.tsv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, sep="\t", index=False)

        subj_metrics = compute_group_metrics(pred_df, ["subject"])
        subj_metrics_path = output_dir / f"per_subject_metrics_{name}.tsv"
        subj_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        subj_metrics.to_csv(subj_metrics_path, sep="\t", index=False)

        temp_metrics = compute_group_metrics(pred_df, ["temp_celsius"])
        temp_metrics_path = output_dir / f"per_temperature_metrics_{name}.tsv"
        temp_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temp_metrics.to_csv(temp_metrics_path, sep="\t", index=False)

        fold_path = None
        fold_df = pd.DataFrame(result["fold_details"])
        if not fold_df.empty:
            fold_df["best_params"] = fold_df["best_params"].apply(lambda d: json.dumps(d))
            if "test_temp_counts" in fold_df.columns:
                fold_df["test_temp_counts"] = fold_df["test_temp_counts"].apply(lambda d: json.dumps(d))
            fold_path = output_dir / f"cv_folds_{name}.tsv"
            fold_path.parent.mkdir(parents=True, exist_ok=True)
            fold_df.to_csv(fold_path, sep="\t", index=False)

        metrics_path = output_dir / f"metrics_{name}.json"
        write_json(metrics_path, result["summary_metrics"])

        best_params_path = output_dir / f"best_params_{name}.json"
        write_json(best_params_path, [fold["best_params"] for fold in result["fold_details"]])

        r2_values = [fold["test_r2"] for fold in result["fold_details"]]
        entry: Dict[str, Any] = {
            "name": name,
            "metrics": result["summary_metrics"],
            "fold_mean_r2": float(np.mean(r2_values)) if r2_values else None,
            "fold_std_r2": float(np.std(r2_values)) if r2_values else None,
            "prediction_file": pred_path.name,
            "per_subject_metrics_file": subj_metrics_path.name,
            "per_temperature_metrics_file": temp_metrics_path.name,
            "fold_details_file": fold_path.name if fold_path else None,
            "metrics_file": metrics_path.name,
            "best_params_file": best_params_path.name,
            "outer_cv": result.get("outer_cv_desc"),
            "inner_cv": result.get("inner_cv_desc"),
        }

        if args.permutation_per_model > 0:
            perm_summary, perm_null = permutation_test_r2(
                model_name=name,
                builder=builder,
                param_grid=param_grid,
                X=X,
                y=y,
                feature_names=feature_columns,
                meta=data,
                outer_groups=outer_groups,
                run_groups=run_groups,
                n_permutations=args.permutation_per_model,
                true_r2=result["summary_metrics"]["r2"],
                random_state=
                    args.permutation_seed if args.permutation_seed is not None else args.random_state,
                n_jobs=args.n_jobs,
                logger=logger,
            )
            perm_json_path = output_dir / f"permutation_{name}.json"
            write_json(perm_json_path, perm_summary)
            null_path = output_dir / f"permutation_{name}_null.npy"
            np.save(null_path, perm_null)
            entry["permutation_test"] = {
                "p_value": perm_summary["p_value"],
                "true_r2": perm_summary["true_r2"],
                "null_mean": perm_summary["null_mean"],
                "null_std": perm_summary["null_std"],
                "null_quantiles": perm_summary["null_quantiles"],
                "result_file": perm_json_path.name,
                "null_distribution_file": null_path.name,
            }

        summary_model_entries.append(entry)

    if not model_results:
        logger.error("No models were evaluated; aborting.")
        sys.exit(1)

    best_result = max(model_results, key=lambda res: res["summary_metrics"]["r2"])
    best_model_name = best_result["name"]
    builder, param_grid = MODEL_REGISTRY[best_model_name]
    final_estimator, final_best_params, final_cv_score, final_cv_desc = fit_final_estimator(
        builder=builder,
        param_grid=param_grid,
        X=X,
        y=y,
        run_groups=run_groups,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    logger.info("Best model %s refit using %s", best_model_name, final_cv_desc)
    final_predictions = final_estimator.predict(X)
    final_metrics = compute_metrics(y.to_numpy(), final_predictions)

    final_pred_df = build_prediction_frame(
        data=data,
        y_true=y,
        y_pred=final_predictions,
        model_name=best_model_name,
    )
    final_pred_path = output_dir / f"final_model_predictions_{best_model_name}.tsv"
    final_pred_path.parent.mkdir(parents=True, exist_ok=True)
    final_pred_df.to_csv(final_pred_path, sep="\t", index=False)

    final_subj_metrics = compute_group_metrics(final_pred_df, ["subject"])
    final_subj_metrics_path = output_dir / f"final_per_subject_metrics_{best_model_name}.tsv"
    final_subj_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_subj_metrics.to_csv(final_subj_metrics_path, sep="\t", index=False)

    final_temp_metrics = compute_group_metrics(final_pred_df, ["temp_celsius"])
    final_temp_metrics_path = output_dir / f"final_per_temperature_metrics_{best_model_name}.tsv"
    final_temp_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_temp_metrics.to_csv(final_temp_metrics_path, sep="\t", index=False)

    model_payload = {
        "model": final_estimator,
        "feature_names": feature_columns,
        "bands": list(bands),
        "subjects": subjects,
        "trained_on": {
            "n_trials": int(len(data)),
            "outer_group_level": outer_group_level,
        },
    }
    joblib.dump(model_payload, output_dir / f"final_model_{best_model_name}.joblib")

    importance_path: Optional[Path] = None
    importance_df = extract_feature_importance(final_estimator, feature_columns)
    if importance_df is not None:
        importance_path = output_dir / f"feature_importance_{best_model_name}.tsv"
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(importance_path, sep="\t", index=False)

    best_entry = next((entry for entry in summary_model_entries if entry["name"] == best_model_name), None)
    if best_entry is None:
        logger.warning("Best model entry missing from summary list; reusing aggregate metrics.")
        best_entry = {
            "name": best_model_name,
            "metrics": best_result["summary_metrics"],
        }

    temperature_distribution = temp_counts.to_dict() if isinstance(temp_counts, pd.Series) else {}
    temporal_alignment_note = ("EEG features derived from the stimulation plateau window (see eeg_pipeline/eeg_config.yaml) and fMRI betas estimated with a canonical HRF at stimulus onset (~5-7 s peak).")
    nonzero_coefficients = None
    reg_step = final_estimator.named_steps.get('reg') if isinstance(final_estimator, Pipeline) else None
    if reg_step is not None and hasattr(reg_step, 'coef_'):
        coef = np.asarray(reg_step.coef_)
        nonzero_coefficients = int(np.count_nonzero(coef))
        logger.info("Final ElasticNet non-zero coefficients: %d / %d", nonzero_coefficients, coef.size)

    temperature_baseline_entry = None
    if temperature_baseline_metrics is not None:
        temperature_baseline_entry = {
            "metrics": temperature_baseline_metrics,
            "cv_strategy": temperature_baseline_desc,
        }

    summary = {
        "bands": list(bands),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "n_trials": int(len(data)),
        "feature_count": len(feature_columns),
        "feature_to_sample_ratio": feature_ratio,
        "temperature_distribution": temperature_distribution,
        "eeg_plateau_window": plateau_window,
        "temperature_only_r2": temperature_baseline_metrics["r2"] if temperature_baseline_metrics else None,
        "temperature_only_baseline": temperature_baseline_entry,
        "outer_cv_level": outer_group_level,
        "outer_cv_strategy": outer_cv_strategy_record or "Not determined",
        "inner_cv_strategy": inner_cv_strategy_record or "Not determined",
        "include_temperature": bool(args.include_temperature),
        "models": summary_model_entries,
        "best_model": {
            "name": best_model_name,
            "cv_best_score": final_cv_score,
            "final_best_params": final_best_params,
            "final_metrics": final_metrics,
            "prediction_file": final_pred_path.name,
            "per_subject_metrics_file": final_subj_metrics_path.name,
            "per_temperature_metrics_file": final_temp_metrics_path.name,
            "model_artifact_file": f"final_model_{best_model_name}.joblib",
            "feature_importance_file": importance_path.name if importance_path else None,
            "refit_cv_strategy": final_cv_desc,
            "nonzero_coefficients": nonzero_coefficients,
            "permutation_test": best_entry.get("permutation_test") if isinstance(best_entry, dict) else None,
        },
        "notes": [
            temporal_alignment_note,
        ],
    }
    if drops_summary:
        summary["dropped_trials"] = drops_summary

    write_json(output_dir / "summary.json", summary)
    logger.info("Best model: %s | CV R2=%.3f | Final R2=%.3f", best_model_name, final_cv_score, final_metrics["r2"])

    manifest_additional = {
        "dataset": {
            "n_trials": int(len(data)),
            "n_subjects": len(subjects),
            "bands": list(bands),
            "feature_count": len(feature_columns),
            "include_temperature": bool(args.include_temperature),
            "outer_groups": outer_group_level,
        },
        "temperature_baseline": temperature_baseline_entry,
        "models": summary_model_entries,
    }
    create_run_manifest(output_dir, args, repo_root, additional=manifest_additional)

    logger.info("All outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
