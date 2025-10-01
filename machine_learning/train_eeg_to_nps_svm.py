#!/usr/bin/env python
"""Train an RBF-kernel Support Vector Machine to predict fMRI Neurologic Pain Signature scores from EEG features."""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for candidate in (str(REPO_ROOT), str(THIS_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from machine_learning.train_eeg_to_nps import (
    DEFAULT_BANDS,
    SUPPORTED_BANDS,
    build_prediction_frame,
    compute_group_metrics,
    compute_metrics,
    compute_temperature_baseline_cv,
    discover_subjects,
    fit_final_estimator,
    load_subject_dataset,
    nested_cv_evaluate,
    permutation_test_r2,
    setup_logging,
    write_json,
)

MODEL_NAME = "svm_rbf"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict NPS beta responses from EEG oscillatory power with an RBF-kernel SVR."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs. Defaults to machine_learning/outputs/eeg_to_nps_svm_<timestamp>.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject identifiers (with or without sub- prefix). Defaults to all available.",
    )
    parser.add_argument(
        "--bands",
        nargs="*",
        default=list(DEFAULT_BANDS),
        help="EEG frequency bands to include (subset of %s)." % (SUPPORTED_BANDS,),
    )
    parser.add_argument(
        "--include-temperature",
        action="store_true",
        help="Include stimulus temperature as an additional predictor.",
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
        help="Random seed for cross-validation reproducibility.",
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=None,
        help="Random seed for label permutations (defaults to --random-state when omitted).",
    )
    parser.add_argument(
        "--permutation-count",
        type=int,
        default=0,
        help="Number of label permutations for R^2 significance testing (0 disables).",
    )
    parser.add_argument(
        "--c-grid",
        dest="c_grid",
        type=float,
        nargs="*",
        default=[0.1, 1.0, 10.0, 100.0],
        help="Grid of C values to explore.",
    )
    parser.add_argument(
        "--gamma-grid",
        dest="gamma_grid",
        nargs="*",
        default=["scale", "auto", "0.01", "0.1", "1.0"],
        help="Grid of gamma values to explore (numeric or 'scale'/'auto').",
    )
    parser.add_argument(
        "--epsilon-grid",
        dest="epsilon_grid",
        type=float,
        nargs="*",
        default=[0.01, 0.1, 0.5],
        help="Grid of epsilon values for the SVR loss.",
    )
    parser.add_argument(
        "--svm-cache-size",
        type=float,
        default=500.0,
        help="Cache size in MB for the SVR solver.",
    )
    return parser.parse_args()


def make_svm_builder(cache_size: float):
    def builder(random_state: int, _n_jobs: int) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("svm", SVR(kernel="rbf", cache_size=cache_size)),
            ]
        )

    return builder

def main() -> None:
    args = parse_args()

    bands = tuple(dict.fromkeys(band.lower() for band in args.bands))
    if not bands:
        raise ValueError("No bands specified.")
    for band in bands:
        if band not in SUPPORTED_BANDS:
            raise ValueError(f"Unsupported band '{band}'. Supported bands: {SUPPORTED_BANDS}.")

    c_grid = list(dict.fromkeys(float(val) for val in args.c_grid))
    if not c_grid:
        raise ValueError("C grid must include at least one value.")
    if any(val <= 0 for val in c_grid):
        raise ValueError("C grid values must be positive.")

    epsilon_grid = list(dict.fromkeys(float(val) for val in args.epsilon_grid))
    if not epsilon_grid:
        raise ValueError("Epsilon grid must include at least one value.")
    if any(val < 0 for val in epsilon_grid):
        raise ValueError("Epsilon grid values must be non-negative.")

    gamma_grid = normalize_gamma_grid(args.gamma_grid)

    repo_root = Path(__file__).resolve().parents[1]
    default_output = THIS_DIR / "outputs" / f"eeg_to_nps_svm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("EEG -> NPS SVM training pipeline started.")
    logger.info("Using bands: %s", ", ".join(bands))
    logger.info("Hyperparameter grid | C: %s | gamma: %s | epsilon: %s", c_grid, gamma_grid, epsilon_grid)
    logger.info("SVR cache size: %.1f MB", args.svm_cache_size)

    eeg_deriv_root = repo_root / "eeg_pipeline" / "bids_output" / "derivatives"
    fmri_outputs_root = repo_root / "fmri_pipeline" / "NPS" / "outputs"

    if not eeg_deriv_root.exists():
        logger.error("EEG derivatives directory not found: %s", eeg_deriv_root)
        raise SystemExit(1)
    if not fmri_outputs_root.exists():
        logger.error("fMRI outputs directory not found: %s", fmri_outputs_root)
        raise SystemExit(1)

    available_subjects = discover_subjects(eeg_deriv_root, fmri_outputs_root)
    if not available_subjects:
        logger.error("No subjects with both EEG features and fMRI NPS scores were found.")
        raise SystemExit(1)
    logger.info("Available subjects with aligned data: %s", ", ".join(available_subjects))

    if args.subjects:
        requested = [s if s.startswith("sub-") else f"sub-{s}" for s in args.subjects]
        invalid = sorted(set(requested) - set(available_subjects))
        if invalid:
            logger.error("Requested subjects missing required data: %s", ", ".join(invalid))
            raise SystemExit(1)
        subjects = sorted(requested)
    else:
        subjects = available_subjects

    subject_results = []
    feature_template: List[str] = []
    drops_summary: Dict[str, List[Dict[str, float]]] = {}

    for subject in subjects:
        result = load_subject_dataset(subject, eeg_deriv_root, fmri_outputs_root, bands, logger)
        subject_results.append(result)
        if not feature_template:
            feature_template = result.feature_columns
        elif feature_template != result.feature_columns:
            logger.error("Feature column mismatch detected for %s.", subject)
            raise SystemExit(1)
        if result.dropped_trials:
            drops_summary[subject] = result.dropped_trials

    if not feature_template:
        logger.error("No feature columns detected.")
        raise SystemExit(1)

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
        raise SystemExit(1)
    if np.isinf(y).any():
        n_inf = np.isinf(y).sum()
        logger.error("Target variable (br_score) contains %d infinite values; cannot proceed.", n_inf)
        raise SystemExit(1)
    
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

    logger.info(
        "Assembled dataset: %d trials, %d subjects, %d features.",
        len(data),
        len(subjects),
        len(feature_columns),
    )

    feature_ratio = len(feature_columns) / max(len(data), 1)
    logger.info(
        "Feature-to-sample ratio: %.2f (%d features / %d trials)",
        feature_ratio,
        len(feature_columns),
        len(data),
    )
    if feature_ratio > (1.0 / 3.0):
        logger.warning("High feature-to-sample ratio may cause overfitting despite regularisation.")

    if "temp_celsius" in data.columns:
        temp_counts = data["temp_celsius"].value_counts().sort_index()
        logger.info("Temperature distribution (counts per condition):\n%s", temp_counts.to_string())
    else:
        temp_counts = pd.Series(dtype=int)

    logger.info(
        "NPS br_score betas reflect the delayed (~5-7 s) hemodynamic response; EEG features are interpreted with this lag in mind."
    )

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

    svm_builder = make_svm_builder(cache_size=args.svm_cache_size)
    param_grid = {
        "svm__C": c_grid,
        "svm__gamma": gamma_grid,
        "svm__epsilon": epsilon_grid,
    }

    grid_size = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    if grid_size * 5 > len(X):
        logger.warning(
            "SVM grid (%d combinations) is large relative to available trials (%d); consider trimming the search space.",
            grid_size,
            len(X),
        )

    result = nested_cv_evaluate(
        model_name=MODEL_NAME,
        builder=svm_builder,
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

    pred_df = build_prediction_frame(
        data=data,
        y_true=y,
        y_pred=result["predictions"],
        model_name=MODEL_NAME,
        fold_assignments=result["fold_assignments"],
    )
    pred_path = output_dir / f"predictions_{MODEL_NAME}.tsv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, sep="\t", index=False)

    subj_metrics = compute_group_metrics(pred_df, ["subject"])
    subj_metrics_path = output_dir / f"per_subject_metrics_{MODEL_NAME}.tsv"
    subj_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    subj_metrics.to_csv(subj_metrics_path, sep="\t", index=False)
    temp_metrics = compute_group_metrics(pred_df, ["temp_celsius"])
    temp_metrics_path = output_dir / f"per_temperature_metrics_{MODEL_NAME}.tsv"
    temp_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temp_metrics.to_csv(temp_metrics_path, sep="\t", index=False)

    fold_path = None
    fold_df = pd.DataFrame(result["fold_details"])
    if not fold_df.empty:
        fold_df["best_params"] = fold_df["best_params"].apply(lambda d: json.dumps(d))
        if "test_temp_counts" in fold_df.columns:
            fold_df["test_temp_counts"] = fold_df["test_temp_counts"].apply(lambda d: json.dumps(d))
        fold_path = output_dir / f"cv_folds_{MODEL_NAME}.tsv"
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(fold_path, sep="\t", index=False)

    metrics_path = output_dir / f"metrics_{MODEL_NAME}.json"
    write_json(metrics_path, result["summary_metrics"])

    best_params_path = output_dir / f"best_params_{MODEL_NAME}.json"
    write_json(best_params_path, [fold["best_params"] for fold in result["fold_details"]])

    r2_values = [fold["test_r2"] for fold in result["fold_details"]]
    svm_entry: Dict[str, Any] = {
        "name": MODEL_NAME,
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
        "param_grid": param_grid,
    }

    if args.permutation_count > 0:
        perm_summary, perm_null = permutation_test_r2(
            model_name=MODEL_NAME,
            builder=svm_builder,
            param_grid=param_grid,
            X=X,
            y=y,
            feature_names=feature_columns,
            meta=data,
            outer_groups=outer_groups,
            run_groups=run_groups,
            n_permutations=args.permutation_count,
            true_r2=result["summary_metrics"]["r2"],
            random_state=
                args.permutation_seed if args.permutation_seed is not None else args.random_state,
            n_jobs=args.n_jobs,
            logger=logger,
        )
        perm_json_path = output_dir / f"permutation_{MODEL_NAME}.json"
        write_json(perm_json_path, perm_summary)
        null_path = output_dir / f"permutation_{MODEL_NAME}_null.npy"
        np.save(null_path, perm_null)
        svm_entry["permutation_test"] = {
            "p_value": perm_summary["p_value"],
            "true_r2": perm_summary["true_r2"],
            "null_mean": perm_summary["null_mean"],
            "null_std": perm_summary["null_std"],
            "null_quantiles": perm_summary["null_quantiles"],
            "result_file": perm_json_path.name,
            "null_distribution_file": null_path.name,
        }

    final_estimator, final_best_params, final_cv_score, final_cv_desc = fit_final_estimator(
        builder=svm_builder,
        param_grid=param_grid,
        X=X,
        y=y,
        run_groups=run_groups,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    logger.info("SVM refit using %s", final_cv_desc)
    final_predictions = final_estimator.predict(X)
    final_metrics = compute_metrics(y.to_numpy(), final_predictions)

    final_pred_df = build_prediction_frame(
        data=data,
        y_true=y,
        y_pred=final_predictions,
        model_name=MODEL_NAME,
    )
    final_pred_path = output_dir / f"final_model_predictions_{MODEL_NAME}.tsv"
    final_pred_path.parent.mkdir(parents=True, exist_ok=True)
    final_pred_df.to_csv(final_pred_path, sep="\t", index=False)

    final_subj_metrics = compute_group_metrics(final_pred_df, ["subject"])
    final_subj_metrics_path = output_dir / f"final_per_subject_metrics_{MODEL_NAME}.tsv"
    final_subj_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_subj_metrics.to_csv(final_subj_metrics_path, sep="\t", index=False)

    final_temp_metrics = compute_group_metrics(final_pred_df, ["temp_celsius"])
    final_temp_metrics_path = output_dir / f"final_per_temperature_metrics_{MODEL_NAME}.tsv"
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
    joblib.dump(model_payload, output_dir / f"final_model_{MODEL_NAME}.joblib")

    temperature_distribution = temp_counts.to_dict() if isinstance(temp_counts, pd.Series) else {}
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
        "temperature_only_baseline": {
            "metrics": temperature_baseline_metrics,
            "cv_strategy": temperature_baseline_desc,
        }
        if temperature_baseline_metrics
        else None,
        "outer_cv_level": outer_group_level,
        "outer_cv_strategy": result.get("outer_cv_desc"),
        "inner_cv_strategy": result.get("inner_cv_desc"),
        "include_temperature": bool(args.include_temperature),
        "models": [svm_entry],
        "best_model": {
            "name": MODEL_NAME,
            "cv_best_score": final_cv_score,
            "final_best_params": final_best_params,
            "final_metrics": final_metrics,
            "prediction_file": final_pred_path.name,
            "per_subject_metrics_file": final_subj_metrics_path.name,
            "per_temperature_metrics_file": final_temp_metrics_path.name,
            "model_artifact_file": f"final_model_{MODEL_NAME}.joblib",
            "refit_cv_strategy": final_cv_desc,
            "permutation_test": svm_entry.get("permutation_test"),
        },
        "notes": [
            "EEG features derive from the stimulation plateau window and are aligned to delayed hemodynamic responses.",
        ],
    }
    if drops_summary:
        summary["dropped_trials"] = drops_summary

    write_json(output_dir / "summary.json", summary)
    logger.info("SVM final R2=%.3f", final_metrics["r2"])
    logger.info("All outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
