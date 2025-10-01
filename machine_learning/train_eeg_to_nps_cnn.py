#!/usr/bin/env python
"""Train a 1D CNN to predict fMRI Neurologic Pain Signature scores from EEG features."""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - dependency guard
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - early failure path
    raise ImportError(
        "PyTorch is required for train_eeg_to_nps_cnn.py. Install torch before running this script."
    ) from exc

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for candidate in (str(REPO_ROOT), str(THIS_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from machine_learning.train_eeg_to_nps import (  # noqa: E402
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

MODEL_NAME = "cnn_1d"


class TorchCNNRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible 1D CNN regressor implemented with PyTorch."""

    def __init__(
        self,
        conv_channels: int = 32,
        kernel_size: int = 5,
        hidden_units: int = 128,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        max_epochs: int = 200,
        patience: int = 20,
        grad_clip: Optional[float] = None,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        if self.device == "cpu":
            return torch.device("cpu")
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but not available.")
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, n_features: int, kernel_size: int) -> nn.Module:
        class _Regressor(nn.Module):
            def __init__(self, features: int, channels: int, ks: int, hidden: int, drop: float) -> None:
                super().__init__()
                self.features = features
                self.conv = nn.Sequential(
                    nn.Conv1d(1, channels, ks, padding=ks // 2),
                    nn.ReLU(),
                    nn.Dropout(drop),
                )
                self.regressor = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(channels * features, hidden),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(hidden, 1),
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                x = self.conv(inputs)
                return self.regressor(x).squeeze(-1)

        return _Regressor(
            features=n_features,
            channels=self.conv_channels,
            ks=kernel_size,
            hidden=self.hidden_units,
            drop=self.dropout,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchCNNRegressor":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("Expected 2D array for X.")
        n_samples, n_features = X_arr.shape
        if n_samples < 2:
            raise ValueError("At least two samples are required to train the CNN regressor.")
        if self.conv_channels <= 0:
            raise ValueError("conv_channels must be positive.")
        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be positive.")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer.")
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1).")
        effective_kernel = min(int(self.kernel_size), n_features)
        if effective_kernel % 2 == 0:
            effective_kernel = max(1, effective_kernel - 1)
        if effective_kernel < 1:
            raise ValueError("Effective kernel size must be at least 1.")

        rng = np.random.default_rng(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        device = self._resolve_device()

        if n_samples > 8 and self.patience > 0:
            val_fraction = min(0.2, max(1.0 / n_samples, 0.1))
            train_X, val_X, train_y, val_y = train_test_split(
                X_arr,
                y_arr,
                test_size=val_fraction,
                random_state=self.random_state,
            )
            val_dataset = TensorDataset(
                torch.from_numpy(val_X.astype(np.float32)),
                torch.from_numpy(val_y.astype(np.float32)),
            )
            val_loader = DataLoader(val_dataset, batch_size=min(self.batch_size, len(val_dataset)))
        else:
            train_X, train_y = X_arr, y_arr
            val_loader = None

        train_dataset = TensorDataset(
            torch.from_numpy(train_X.astype(np.float32)),
            torch.from_numpy(train_y.astype(np.float32)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
        )

        model = self._build_model(n_features=n_features, kernel_size=effective_kernel).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        history: List[Dict[str, float]] = []
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            model.train()
            epoch_losses: List[float] = []
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device).unsqueeze(1)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            if val_loader is not None:
                model.eval()
                val_losses: List[float] = []
                with torch.no_grad():
                    for val_X_batch, val_y_batch in val_loader:
                        val_X_batch = val_X_batch.to(device).unsqueeze(1)
                        val_y_batch = val_y_batch.to(device)
                        val_pred = model(val_X_batch)
                        val_loss = criterion(val_pred, val_y_batch)
                        val_losses.append(float(val_loss.detach().cpu().item()))
                monitored_loss = float(np.mean(val_losses)) if val_losses else train_loss
            else:
                monitored_loss = train_loss

            history.append({"epoch": epoch + 1, "train_loss": train_loss, "monitor_loss": monitored_loss})
            improved = monitored_loss < (best_loss - 1e-6)
            if improved:
                best_loss = monitored_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose:
                print(f"Epoch {epoch + 1:03d} | train_loss={train_loss:.4f} | monitor={monitored_loss:.4f}")

            if self.patience > 0 and epochs_without_improvement >= self.patience:
                break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model = model.to(torch.device("cpu"))
        model.eval()

        self.model_ = model
        self.history_ = history
        self.best_loss_ = best_loss
        self.input_features_ = n_features
        self.device_ = torch.device("cpu")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model has not been fitted yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[1] != getattr(self, "input_features_", X_arr.shape[1]):
            raise ValueError("Input feature dimension does not match the fitted data.")
        dataset = TensorDataset(torch.from_numpy(X_arr.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=min(len(dataset), 512))
        preds: List[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for (batch_X,) in loader:
                outputs = self.model_(batch_X.unsqueeze(1))
                preds.append(outputs.numpy())
        return np.concatenate(preds, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict NPS beta responses from EEG oscillatory power with a 1D CNN."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs. Defaults to machine_learning/outputs/eeg_to_nps_cnn_<timestamp>.",
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
        default=1,
        help="Parallel workers for grid searches (values <=0 fallback to 1).",
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
        "--conv-channels-grid",
        type=int,
        nargs="*",
        default=[16, 32],
        help="Grid of convolution channel counts to explore.",
    )
    parser.add_argument(
        "--kernel-size-grid",
        type=int,
        nargs="*",
        default=[3, 5],
        help="Grid of convolution kernel sizes (positive integers).",
    )
    parser.add_argument(
        "--hidden-dim-grid",
        type=int,
        nargs="*",
        default=[128],
        help="Grid of fully connected hidden unit sizes.",
    )
    parser.add_argument(
        "--dropout-grid",
        type=float,
        nargs="*",
        default=[0.1, 0.3],
        help="Grid of dropout probabilities (between 0 and 1).",
    )
    parser.add_argument(
        "--learning-rate-grid",
        type=float,
        nargs="*",
        default=[1e-3],
        help="Grid of learning rates.",
    )
    parser.add_argument(
        "--weight-decay-grid",
        type=float,
        nargs="*",
        default=[0.0, 1e-4],
        help="Grid of weight decay (L2 regularisation) values.",
    )
    parser.add_argument(
        "--batch-size-grid",
        type=int,
        nargs="*",
        default=[32],
        help="Grid of batch sizes.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Maximum training epochs per fit call.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Gradient clipping norm (disabled if not provided or <= 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device to use for training (auto selects CUDA if available).",
    )
    parser.add_argument(
        "--verbose-cnn",
        action="store_true",
        help="Print per-epoch training progress for the CNN estimator.",
    )
    return parser.parse_args()


def normalize_grid(values: Sequence[Union[int, float]], *, value_type: str) -> List[Union[int, float]]:
    unique_values = []
    for val in values:
        cast_val = float(val)
        if value_type == "int":
            cast_val = int(round(cast_val))
        if cast_val not in unique_values:
            unique_values.append(cast_val)
    if not unique_values:
        raise ValueError("Provided grid must contain at least one value.")
    return unique_values


def make_cnn_builder(device: str, verbose: bool) -> callable:
    def builder(random_state: int, _: int) -> Pipeline:
        regressor = TorchCNNRegressor(random_state=random_state, device=device, verbose=verbose)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("cnn", regressor),
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

    conv_grid = [int(v) for v in normalize_grid(args.conv_channels_grid, value_type="int") if int(v) > 0]
    if not conv_grid:
        raise ValueError("conv_channels grid must contain positive integers.")
    kernel_grid = [int(v) for v in normalize_grid(args.kernel_size_grid, value_type="int") if int(v) > 0]
    if not kernel_grid:
        raise ValueError("kernel_size grid must contain positive integers.")
    hidden_grid = [int(v) for v in normalize_grid(args.hidden_dim_grid, value_type="int") if int(v) > 0]
    if not hidden_grid:
        raise ValueError("hidden dimension grid must contain positive integers.")
    dropout_grid = [float(v) for v in normalize_grid(args.dropout_grid, value_type="float") if 0 <= float(v) < 1]
    if not dropout_grid:
        raise ValueError("dropout grid must contain probabilities in [0, 1).")
    lr_grid = [float(v) for v in normalize_grid(args.learning_rate_grid, value_type="float") if float(v) > 0]
    if not lr_grid:
        raise ValueError("learning-rate grid must contain positive floats.")
    weight_decay_grid = [float(v) for v in normalize_grid(args.weight_decay_grid, value_type="float") if float(v) >= 0]
    if not weight_decay_grid:
        raise ValueError("weight-decay grid must contain non-negative floats.")
    batch_grid = [int(v) for v in normalize_grid(args.batch_size_grid, value_type="int") if int(v) > 0]
    if not batch_grid:
        raise ValueError("batch-size grid must contain positive integers.")

    repo_root = Path(__file__).resolve().parents[1]
    default_output = THIS_DIR / "outputs" / f"eeg_to_nps_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("EEG -> NPS CNN training pipeline started.")
    logger.info("Using bands: %s", ", ".join(bands))
    logger.info(
        "Hyperparameter grid | channels: %s | kernel: %s | hidden: %s | dropout: %s | lr: %s | weight_decay: %s | batch: %s",
        conv_grid,
        kernel_grid,
        hidden_grid,
        dropout_grid,
        lr_grid,
        weight_decay_grid,
        batch_grid,
    )
    logger.info("Max epochs: %d | Patience: %d | Grad clip: %s", args.max_epochs, args.patience, args.grad_clip)

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
    feature_template: Optional[List[str]] = None
    drops_summary: Dict[str, List[Dict[str, float]]] = {}

    for subject in subjects:
        result = load_subject_dataset(subject, eeg_deriv_root, fmri_outputs_root, bands, logger)
        subject_results.append(result)
        if feature_template is None:
            feature_template = result.feature_columns
        elif feature_template != result.feature_columns:
            logger.error("Feature column mismatch detected for %s.", subject)
            raise SystemExit(1)
        if result.dropped_trials:
            drops_summary[subject] = result.dropped_trials

    if feature_template is None:
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

    cnn_builder = make_cnn_builder(device=args.device, verbose=args.verbose_cnn)

    param_grid = {
        "cnn__conv_channels": conv_grid,
        "cnn__kernel_size": kernel_grid,
        "cnn__hidden_units": hidden_grid,
        "cnn__dropout": dropout_grid,
        "cnn__lr": lr_grid,
        "cnn__weight_decay": weight_decay_grid,
        "cnn__batch_size": batch_grid,
        "cnn__max_epochs": [int(args.max_epochs)],
        "cnn__patience": [int(args.patience)],
    }
    if args.grad_clip is not None and args.grad_clip > 0:
        param_grid["cnn__grad_clip"] = [float(args.grad_clip)]

    grid_size = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    if grid_size * 5 > len(X):
        logger.warning(
            "CNN grid (%d combinations) is large relative to available trials (%d); consider trimming the search space.",
            grid_size,
            len(X),
        )

    effective_n_jobs = args.n_jobs if args.n_jobs and args.n_jobs > 0 else 1
    if effective_n_jobs != args.n_jobs:
        logger.warning("CNN pipeline forcing n_jobs=%d to ensure deterministic Torch training.", effective_n_jobs)

    result = nested_cv_evaluate(
        model_name=MODEL_NAME,
        builder=cnn_builder,
        param_grid=param_grid,
        X=X,
        y=y,
        feature_names=feature_columns,
        meta=data,
        outer_groups=outer_groups,
        run_groups=run_groups,
        random_state=args.random_state,
        n_jobs=effective_n_jobs,
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
    cnn_entry: Dict[str, Any] = {
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
            builder=cnn_builder,
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
            n_jobs=effective_n_jobs,
            logger=logger,
        )
        perm_json_path = output_dir / f"permutation_{MODEL_NAME}.json"
        write_json(perm_json_path, perm_summary)
        null_path = output_dir / f"permutation_{MODEL_NAME}_null.npy"
        np.save(null_path, perm_null)
        cnn_entry["permutation_test"] = {
            "p_value": perm_summary["p_value"],
            "true_r2": perm_summary["true_r2"],
            "null_mean": perm_summary["null_mean"],
            "null_std": perm_summary["null_std"],
            "null_quantiles": perm_summary["null_quantiles"],
            "result_file": perm_json_path.name,
            "null_distribution_file": null_path.name,
        }

    final_estimator, final_best_params, final_cv_score, final_cv_desc = fit_final_estimator(
        builder=cnn_builder,
        param_grid=param_grid,
        X=X,
        y=y,
        run_groups=run_groups,
        random_state=args.random_state,
        n_jobs=effective_n_jobs,
    )
    logger.info("CNN refit using %s", final_cv_desc)
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
        "models": [cnn_entry],
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
            "permutation_test": cnn_entry.get("permutation_test"),
        },
        "notes": [
            "EEG features derive from the stimulation plateau window and are aligned to delayed hemodynamic responses.",
        ],
    }
    if drops_summary:
        summary["dropped_trials"] = drops_summary

    write_json(output_dir / "summary.json", summary)
    logger.info("CNN final R2=%.3f", final_metrics["r2"])
    logger.info("All outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
