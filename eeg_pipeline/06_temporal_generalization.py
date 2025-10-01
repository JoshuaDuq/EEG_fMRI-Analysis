from __future__ import annotations

"""
Temporal Generalization (Time x Time) decoding/regression.

Per-subject analysis that trains a model at each time point and evaluates
at every other time point to reveal the temporal stability of pain-related
representations. Supports binary classification (pain/non-pain) and
regression (ratings). Saves heatmaps and diagonal time courses.

Outputs
- derivatives/sub-<ID>/eeg/plots/06_temporal_generalization/*.png (and configured formats)
- derivatives/sub-<ID>/eeg/stats/time_generalization_*.tsv (scores grid)
- group aggregation (optional): derivatives/group/eeg/plots|stats/06_temporal_generalization/
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from mne.decoding import GeneralizingEstimator, SlidingEstimator

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold, GridSearchCV
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import BrokenProcessPool
from scipy.stats import t as _student_t
from scipy.signal import hilbert as _hilbert
from threadpoolctl import threadpool_limits
try:
    from sklearn.model_selection import StratifiedGroupKFold as _StratifiedGroupKFold  # type: ignore
except Exception:  # pragma: no cover
    _StratifiedGroupKFold = None  # type: ignore

# Centralized config and helpers
from utils.config_loader import load_config, get_legacy_constants
from utils.logging_utils import get_subject_logger, get_group_logger
from utils.io_utils import (
    _find_clean_epochs_path as _find_clean_epochs_path,
    _load_events_df as _load_events_df,
    _align_events_to_epochs as _align_events_to_epochs,
    _pick_target_column as _pick_target_column,
)
from utils.roi_utils import build_rois_from_info as _build_rois


config = load_config()
_C = get_legacy_constants(config)
# Constrain BLAS backends to a single thread to avoid oversubscription during joblib parallelism
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

DERIV_ROOT: Path = _C["DERIV_ROOT"]
TASK: str = _C["TASK"]
RATING_COLUMNS: List[str] = _C["RATING_COLUMNS"]
PAIN_BINARY_COLUMNS: List[str] = _C["PAIN_BINARY_COLUMNS"]
ERP_PICKS = _C.get("ERP_PICKS", "eeg")
SAVE_FORMATS = tuple(config.get("output.save_formats", ["png"]))
FIG_DPI = int(config.get("output.fig_dpi", 300))

DIAG_CLUSTER_P = float(config.get("temporal_generalization.diag_cluster_p", 0.05))
MIN_BAND_BASELINE_SEC = float(config.get("time_frequency_analysis.min_baseline_sec", 0.1))
BAND_BASELINE_EPS = float(config.get("time_frequency_analysis.baseline_epsilon", 1e-12))

# FDR correction (fallback BH)
try:
    from statsmodels.stats.multitest import fdrcorrection as _fdrcorrection
except Exception:  # pragma: no cover
    _fdrcorrection = None

def _fdr_bh(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    if _fdrcorrection is not None:
        _, q = _fdrcorrection(p, alpha=alpha)
        return q
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    q = np.minimum(q, 1.0)
    out = np.empty_like(p)
    out[order] = q
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plots_dir_subject(subject: str) -> Path:
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "plots" / "06_temporal_generalization"


def _stats_dir_subject(subject: str) -> Path:
    return DERIV_ROOT / f"sub-{subject}" / "eeg" / "stats"


def _plots_dir_group() -> Path:
    return DERIV_ROOT / "group" / "eeg" / "plots" / "06_temporal_generalization"


def _stats_dir_group() -> Path:
    return DERIV_ROOT / "group" / "eeg" / "stats"




def _load_scores_from_stats(stats_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load the averaged score matrix and time axes from a saved stats TSV."""
    if not stats_path.exists():
        return None
    try:
        df = pd.read_csv(stats_path, sep="\t")
    except Exception:
        return None
    required = {"t_train", "t_test", "score"}
    if not required.issubset(df.columns):
        return None
    t_train = np.array(sorted(pd.unique(df["t_train"])), dtype=float)
    t_test = np.array(sorted(pd.unique(df["t_test"])), dtype=float)
    if t_train.size == 0 or t_test.size == 0:
        return None
    idx = pd.MultiIndex.from_product([t_train, t_test], names=["t_train", "t_test"])
    scores_series = (
        df.set_index(["t_train", "t_test"])
        .sort_index()
        .reindex(idx)
        .get("score")
    )
    if scores_series is None or scores_series.isnull().any():
        return None
    scores = scores_series.to_numpy(dtype=float).reshape(t_train.size, t_test.size)
    return scores, t_train, t_test


def _load_diag_from_disk(diag_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
    """Load diagonal scores and metadata from a saved diag TSV."""
    if not diag_path.exists():
        return None
    try:
        df = pd.read_csv(diag_path, sep="\t")
    except Exception:
        return None
    required = {"time", "score"}
    if not required.issubset(df.columns):
        return None
    diag = df["score"].to_numpy(dtype=float)
    times = df["time"].to_numpy(dtype=float)
    meta = {}
    for key in ("folds_total", "folds_used", "folds_skipped"):
        if key in df.columns:
            try:
                meta[key] = float(df.iloc[0][key])
            except Exception:
                pass
    return diag, times, meta


def _load_diag_folds(path: Path) -> Optional[np.ndarray]:
    """Load per-fold diagonal arrays from NPZ if available."""
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return None
    if "diag_folds" not in data:
        return None
    return np.asarray(data["diag_folds"], dtype=float)


def _save_fig(fig: plt.Figure, base: Path) -> None:
    base = base.with_suffix("")
    # Footer annotations for transparency
    try:
        fig.text(0.01, 0.01, "Temporal Generalization | Nested CV | Only diagonal tested (inference)", fontsize=8, alpha=0.75)
    except Exception:
        pass
    for ext in SAVE_FORMATS:
        try:
            fig.savefig(base.with_suffix(f".{ext}"), dpi=FIG_DPI, bbox_inches="tight")
        except Exception:
            pass
    plt.close(fig)


def _pick_target(events: pd.DataFrame, target_pref: str) -> Tuple[str, str]:
    """Return (target_col, mode) where mode in {"classification","regression"}."""
    if target_pref == "rating":
        for c in RATING_COLUMNS:
            if c in events.columns:
                return c, "regression"
    if target_pref == "pain":
        for c in PAIN_BINARY_COLUMNS:
            if c in events.columns:
                return c, "classification"
    # auto: prefer rating, then pain
    for c in RATING_COLUMNS:
        if c in events.columns:
            return c, "regression"
    for c in PAIN_BINARY_COLUMNS:
        if c in events.columns:
            return c, "classification"
    raise ValueError("No suitable target column found (ratings or pain binary).")


def _build_cv(y: np.ndarray, events_aligned: pd.DataFrame, mode: str, n_splits: int = 5, seed: int = 42):
    def _safe_stratified_splits(y_arr: np.ndarray, requested: int) -> int:
        y_arr = np.asarray(y_arr)
        classes, counts = np.unique(y_arr, return_counts=True)
        if classes.size < 2:
            return 2
        min_count = int(counts.min())
        return max(2, min(requested, min_count))

    def _fix_groups(g: np.ndarray) -> np.ndarray:
        g = np.asarray(g).copy()
        # Treat unknown runs (-1) as one shared group to avoid exploding singletons
        g[g == -1] = 9999
        return g
    # Prefer GroupKFold on runs when at least 2 valid run labels exist
    if "run" in events_aligned.columns and events_aligned["run"].notna().any():
        groups = pd.to_numeric(events_aligned["run"], errors="coerce").fillna(-1).astype(int).to_numpy()
        valid = groups != -1
        uniq_valid = np.unique(groups[valid])
        if uniq_valid.size >= 2:
            # If too many singleton groups, drop group constraint (over-constrains splits on tiny datasets)
            gfix = _fix_groups(groups)
            g_valid = gfix[valid]
            _, counts = np.unique(g_valid, return_counts=True)
            ratio_singletons = (counts == 1).sum() / float(len(counts)) if len(counts) > 0 else 0.0
            if ratio_singletons <= 0.30:
                n_splits = min(n_splits, uniq_valid.size)
                if mode == "classification" and _StratifiedGroupKFold is not None:
                    return _StratifiedGroupKFold(n_splits=n_splits), gfix
                return GroupKFold(n_splits=n_splits), gfix
    # Fallback: stratify for classification; shuffled KFold for regression
    if mode == "classification":
        n_sf = _safe_stratified_splits(y, n_splits)
        return StratifiedKFold(n_splits=n_sf, shuffle=True, random_state=int(seed)), None
    return KFold(n_splits=min(n_splits, max(2, len(y) // 5)), shuffle=True, random_state=int(seed)), None


def _build_inner_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    mode: str,
    seed: int,
    max_splits: int = 3,
):
    """Construct deterministic inner-CV splits with consistent strategy.

    - Uses StratifiedGroupKFold for classification when available and groups are valid.
    - Falls back to GroupKFold (classification/regression) when groups are valid but stratified is unavailable.
    - Otherwise uses StratifiedKFold (classification) or KFold (regression).
    - Filters out single-class inner folds for classification; if none remain, falls back to
      StratifiedKFold without grouping to ensure valid hyperparameter selection.
    Returns (splits, desc) where desc is a human-readable description of strategy used.
    """
    n_splits = int(max_splits)
    desc = ""
    orig_count = 0
    if groups is not None:
        groups = np.asarray(groups)
        valid = groups != -1
        uniq = np.unique(groups[valid]) if np.any(valid) else np.array([])
        if uniq.size >= 2:
            n_splits = min(n_splits, uniq.size)
            # Treat unknown group labels as one shared group to avoid exploding singletons
            gfix = np.where(valid, groups, 9999)
            # Optionally drop group constraint if too many singleton groups
            g_valid = gfix[valid]
            _, counts = np.unique(g_valid, return_counts=True)
            ratio_singletons = (counts == 1).sum() / float(len(counts)) if len(counts) > 0 else 0.0
            use_groups = ratio_singletons <= 0.30
            if mode == "classification" and _StratifiedGroupKFold is not None:
                if use_groups:
                    inner = _StratifiedGroupKFold(n_splits=n_splits)
                    splits = list(inner.split(X, y, gfix))
                    desc = f"StratifiedGroupKFold(n={n_splits}, groups=run)"
                else:
                    inner = None
                    splits = None
                    desc = "StratifiedKFold (fallback, many singleton groups)"
            else:
                if use_groups:
                    inner = GroupKFold(n_splits=n_splits)
                    splits = list(inner.split(X, y, gfix))
                    desc = f"GroupKFold(n={n_splits}, groups=run)"
                else:
                    inner = None
                    splits = None
                    desc = "KFold (fallback, many singleton groups)"
            orig_count = len(splits)
        else:
            splits = None
    else:
        splits = None

    if splits is None:
        if mode == "classification":
            # Limit by minority class count to avoid degenerate folds
            counts = np.bincount(np.asarray(y, dtype=int))
            min_count = int(counts.min()) if counts.size > 1 else 2
            n_splits = max(2, min(n_splits, min_count))
            inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            splits = list(inner.split(X, y))
            desc = f"StratifiedKFold(n={n_splits}, shuffle=True, seed={int(seed)})"
        else:
            inner = KFold(n_splits=max(2, n_splits), shuffle=True, random_state=int(seed))
            splits = list(inner.split(X))
            desc = f"KFold(n={max(2, n_splits)}, shuffle=True, seed={int(seed)})"
        orig_count = len(splits)

    # For classification, drop any inner splits that are single-class in train or test
    if mode == "classification":
        clean = []
        y_arr = np.asarray(y)
        for tr_i, te_i in splits:
            if np.unique(y_arr[tr_i]).size < 2 or np.unique(y_arr[te_i]).size < 2:
                continue
            clean.append((tr_i, te_i))
        if not clean:
            # Fallback: ignore groups and try stratified KFold
            n_sf = min(3, max(2, np.bincount(y_arr).min() if np.unique(y_arr).size > 1 else 2))
            inner = StratifiedKFold(n_splits=n_sf, shuffle=True, random_state=int(seed))
            clean = list(inner.split(X, y))
            desc = f"StratifiedKFold(n={n_sf}, shuffle=True, seed={int(seed)}) [fallback, no-groups]"
            orig_count = len(clean)
        else:
            dropped = orig_count - len(clean)
            if dropped > 0:
                desc += f" [filtered {dropped}/{orig_count} single-class folds]"
        splits = clean

    return splits, desc


def _make_estimator(mode: str, random_state: int = 42) -> Pipeline:
    if mode == "classification":
        return Pipeline([
            ("scale", StandardScaler(with_mean=True, copy=True)),
            ("clf", LogisticRegression(solver="saga", max_iter=10000, class_weight="balanced", n_jobs=1, random_state=int(random_state))),
        ])
    else:
        return Pipeline([
            ("scale", StandardScaler(with_mean=True, copy=True)),
            ("reg", Ridge(alpha=1.0, random_state=42)),
        ])


def _binarize_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    vals = np.unique(y)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return y
    # Already 0/1
    if set(vals.tolist()) == {0, 1}:
        return y.astype(int)
    # 1/2 -> 0/1
    if set(vals.tolist()) == {1, 2}:
        return (y.astype(float) - 1).astype(int)
    # -1/1 -> 0/1
    if set(vals.tolist()) == {-1, 1}:
        return ((y.astype(float) + 1) / 2).astype(int)
    # Generic 2-class numeric mapping: min->0, max->1
    if vals.size == 2:
        lo, hi = float(vals.min()), float(vals.max())
        yb = np.full_like(y, fill_value=-1, dtype=int)
        yb[np.isclose(y.astype(float), lo)] = 0
        yb[np.isclose(y.astype(float), hi)] = 1
        if np.any(yb == -1):
            raise ValueError("Unmappable labels encountered during binarization.")
        return yb
    raise ValueError("Classification target has >2 unique values; cannot binarize reliably.")


def _regression_scoring(estimator, X, y):
    """Pearson correlation between predictions and y (signed)."""
    try:
        y_pred = estimator.predict(X)
        y = np.asarray(y, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        finite_mask = np.isfinite(y) & np.isfinite(y_pred)
        if finite_mask.sum() < 2:
            return 0.0
        if not np.all(finite_mask):
            y = y[finite_mask]
            y_pred = y_pred[finite_mask]
        y_centered = y - y.mean()
        y_pred_centered = y_pred - y_pred.mean()
        denom = np.linalg.norm(y_centered) * np.linalg.norm(y_pred_centered)
        if denom <= 1e-12:
            return 0.0
        r = float(np.dot(y_centered, y_pred_centered) / denom)
        if not np.isfinite(r):
            return 0.0
        return r
    except Exception as exc:
        raise RuntimeError(f"Regression scoring failed: {exc}") from exc


def _process_outer_fold(
    fi: int,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    mode: str,
    scoring,
    param_grid: dict,
    hyper_name: str,
    seed: int,
    time_jobs: int,
):
    result = {
        "index": int(fi),
        "score_matrix": None,
        "weight": float(len(te_idx)),
        "diag": None,
        "hyper_median": None,
        "hyper_count": 0,
        "hyper_time": None,
        "inner_desc": None,
        "skip_reason": None,
        "error": None,
    }

    if mode == "classification":
        if np.unique(y[tr_idx]).size < 2 or np.unique(y[te_idx]).size < 2:
            result["skip_reason"] = "single-class"
            return result

    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    X_te = X[te_idx]
    y_te = y[te_idx]
    groups_arr = np.asarray(groups) if groups is not None else None
    groups_tr = (groups_arr[tr_idx] if groups_arr is not None else None)

    inner_cv_splits, inner_desc = _build_inner_cv_splits(
        X_tr,
        y_tr,
        groups_tr,
        mode=mode,
        seed=seed,
        max_splits=3,
    )
    result["inner_desc"] = inner_desc

    base = _make_estimator(mode, random_state=seed)
    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=inner_cv_splits,
        scoring=scoring,
        n_jobs=1,
        refit=True,
        error_score='raise',
    )
    gen = GeneralizingEstimator(search, scoring=scoring, n_jobs=int(max(1, time_jobs)))

    try:
        with threadpool_limits(1):
            gen.fit(X_tr, y_tr)
        with threadpool_limits(1):
            sc_mat = gen.score(X_te, y_te)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    sc_mat = np.atleast_2d(sc_mat)
    result["score_matrix"] = sc_mat
    result["diag"] = np.diag(sc_mat)

    try:
        vals_all = [
            getattr(est, "best_params_", {}).get(hyper_name, np.nan)
            for est in getattr(gen, "estimators_", [])
        ]
        vals = [float(v) for v in vals_all if isinstance(v, (int, float)) and np.isfinite(v)]
        if vals:
            result["hyper_median"] = float(np.median(vals))
            result["hyper_count"] = int(len(vals))
            result["hyper_time"] = [
                float(v) if isinstance(v, (int, float)) and np.isfinite(v) else np.nan
                for v in vals_all
            ]
    except Exception:
        pass

    return result


def _run_permutation_worker(
    pi: int,
    child_seed,
    outer_splits,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    mode: str,
    scoring,
    param_grid: dict,
    hyper_name: str,
    seed: int,
    time_jobs: int,
    obs_fold_diags: Dict[int, np.ndarray],
    obs_fold_weights: Dict[int, float],
) -> dict:
    result = {
        "index": int(pi),
        "diag_null": None,
        "diag_obs": None,
        "error": None,
        "skipped": False,
    }

    rng = np.random.default_rng(child_seed)
    y_perm = np.asarray(y).copy()
    groups_arr = np.asarray(groups) if groups is not None else None

    if groups_arr is not None:
        for gg in np.unique(groups_arr):
            idx = np.where(groups_arr == gg)[0]
            if idx.size > 1:
                rng.shuffle(y_perm[idx])
    else:
        rng.shuffle(y_perm)

    fold_diags: List[np.ndarray] = []
    fold_weights: List[float] = []
    kept_folds: List[int] = []

    for fi, (tr, te) in enumerate(outer_splits):
        tr_idx = np.asarray(tr)
        te_idx = np.asarray(te)
        res = _process_outer_fold(
            fi,
            tr_idx,
            te_idx,
            X,
            y_perm,
            groups_arr,
            mode,
            scoring,
            param_grid,
            hyper_name,
            seed,
            time_jobs,
        )
        if res.get("skip_reason") == "single-class":
            continue
        err_msg = res.get("error")
        if err_msg:
            result["error"] = err_msg
            return result
        diag_fold = res.get("diag")
        if diag_fold is None:
            continue
        diag_arr = np.asarray(diag_fold)
        fold_diags.append(diag_arr)
        fold_weights.append(float(len(te_idx)))
        kept_folds.append(fi)

    if not fold_diags:
        result["skipped"] = True
        return result

    lengths = {len(d) for d in fold_diags}
    if len(lengths) != 1:
        result["error"] = f"Inconsistent permuted diagonal lengths across folds: {sorted(lengths)}"
        return result

    A = np.stack(fold_diags, axis=0)
    w = np.asarray(fold_weights, dtype=float)
    w_sum = np.nansum(w)
    if w_sum <= 0:
        diag_null = np.nanmean(A, axis=0)
    else:
        diag_null = np.nansum(A * w[:, None], axis=0) / w_sum
    result["diag_null"] = diag_null

    if kept_folds and obs_fold_diags:
        obs_kept = [obs_fold_diags[i] for i in kept_folds if i in obs_fold_diags]
        if obs_kept:
            lengths_obs = {len(d) for d in obs_kept}
            if len(lengths_obs) != 1:
                result["error"] = f"Inconsistent observed diagonal lengths across kept folds: {sorted(lengths_obs)}"
                return result
            AO = np.stack(obs_kept, axis=0)
            wobs = np.asarray([obs_fold_weights.get(i, float(len(outer_splits[i][1]))) for i in kept_folds], dtype=float)
            if np.nansum(wobs) <= 0:
                diag_obs_perm = np.nanmean(AO, axis=0)
            else:
                diag_obs_perm = np.nansum(AO * wobs[:, None], axis=0) / np.nansum(wobs)
            result["diag_obs"] = diag_obs_perm

    return result




def temporal_generalization_subject(
    subject: str,
    task: str = TASK,
    target: str = "auto",  # one of: auto|rating|pain
    roi: Optional[str] = None,
    picks: str | List[str] = ERP_PICKS,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    decim: int = 1,
    n_splits: int = 5,
    permutations: int = 0,
    seed: int = 42,
    save_perm_samples: bool = False,
    max_perm_save: Optional[int] = None,
    inner_jobs: int = 1,
    time_jobs: int = 1,
    outer_jobs: int = 1,
    perm_jobs: int = 1,
    two_sided_classification: bool = False,

) -> Optional[Path]:
    logger = get_subject_logger("temporal_generalization", subject, "06_temporal_generalization.log")
    plots_dir = _plots_dir_subject(subject)
    stats_dir = _stats_dir_subject(subject)
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    time_jobs = max(1, int(time_jobs))
    outer_jobs = max(1, int(outer_jobs))
    perm_jobs = max(1, int(perm_jobs))
    if permutations == 0 and perm_jobs != 1:
        logger.info("No permutations requested; forcing perm_jobs=1.")
        perm_jobs = 1

    heavy_axes = sum(val > 1 for val in (time_jobs, outer_jobs, perm_jobs))
    if heavy_axes > 1:
        if permutations > 0 and perm_jobs > 1:
            if time_jobs > 1:
                logger.info("Permutations requested; setting time_jobs=1 to avoid nested parallelism.")
                time_jobs = 1
            if outer_jobs > 1:
                logger.info("Permutations requested; setting outer_jobs=1 to avoid nested parallelism.")
                outer_jobs = 1
        elif time_jobs > 1:
            if perm_jobs > 1:
                logger.info("Focusing parallelism on time axis; setting perm_jobs=1.")
                perm_jobs = 1
            if outer_jobs > 1:
                logger.info("Focusing parallelism on time axis; setting outer_jobs=1.")
                outer_jobs = 1
        elif outer_jobs > 1 and perm_jobs > 1:
            if permutations > 0:
                logger.info("Preferring permutation workers; setting outer_jobs=1.")
                outer_jobs = 1
            else:
                logger.info("No permutations pending; setting perm_jobs=1 to keep outer CV parallelism.")
                perm_jobs = 1

    if int(inner_jobs) != 1:
        logger.info("GridSearchCV is forced to n_jobs=1; ignoring inner_jobs=%s", inner_jobs)

    epo_path = _find_clean_epochs_path(subject, task)
    if epo_path is None or not Path(epo_path).exists():
        logger.error(f"No cleaned epochs found for sub-{subject}, task-{task}")
        return None

    try:
        epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
    except Exception as e:
        logger.error(f"Failed loading epochs: {e}")
        return None

    # ROI selection if requested
    if roi is not None:
        roi_map = _build_rois(epochs.info)
        if roi in roi_map:
            epochs.pick_channels(roi_map[roi])
            logger.info(f"Using ROI '{roi}' with {len(epochs.ch_names)} channels")
        else:
            logger.warning(f"ROI '{roi}' not found; using all EEG channels")

    # Crop and resample for speed
    if tmin is not None or tmax is not None:
        try:
            epochs.crop(tmin=tmin, tmax=tmax)
        except Exception as e:
            logger.error(f"Failed to crop epochs: {e}")
            return None
    if int(decim) > 1:
        try:
            new_sfreq = float(epochs.info["sfreq"]) / int(decim)
            epochs.resample(new_sfreq, npad="auto")
            logger.info(f"Resampled epochs to {new_sfreq:.2f} Hz (factor {int(decim)})")
        except Exception as e:
            logger.error(f"Failed to resample epochs: {e}")
            return None

    # Align behavioral events
    events = _load_events_df(subject, task)
    events_aligned = _align_events_to_epochs(events, epochs)
    if events_aligned is None:
        logger.error("Could not align events to epochs")
        return None

    # Target selection
    try:
        target_col, mode = _pick_target(events_aligned, target)
    except ValueError as e:
        logger.error(str(e))
        return None

    y = pd.to_numeric(events_aligned[target_col], errors="coerce")
    mask = ~y.isna()
    epochs = epochs[mask.to_numpy()]  # drop NaN target trials
    y = y.loc[mask].to_numpy()
    # Always carry a masked, reindexed events_aligned to avoid desynchrony later
    events_aligned_masked = events_aligned.loc[mask].reset_index(drop=True)

    if mode == "classification":
        # Ensure binary 0/1 robustly
        try:
            y = _binarize_labels(y.astype(float))
        except Exception as e:
            logger.error(f"Binarization failed: {e}")
            return None
        if np.unique(y).size < 2:
            logger.error("Classification target has a single class after cleaning")
            return None
        scoring = "roc_auc"
        vmin, vmax = 0.5, 1.0
        cmap = "viridis"
    else:
        if np.isnan(y).all() or np.std(y[~np.isnan(y)]) <= 1e-12:
            logger.error("Regression target has insufficient variance")
            return None
        scoring = _regression_scoring
        vlim = 0.6
        vmin, vmax = -vlim, vlim
        cmap = "RdBu_r"

    roi_label = roi or "all"
    base_tag = f"{mode}_{roi_label}_{target_col}"
    stats_path = stats_dir / f"time_generalization_{base_tag}.tsv"
    diag_path = stats_dir / f"time_generalization_diag_{base_tag}.tsv"
    diag_perm_path = stats_dir / f"time_generalization_diag_{base_tag}_perm.tsv"
    diag_folds_path = stats_dir / f"time_generalization_diag_folds_{base_tag}.npz"
    hyper_timecourse_path = stats_dir / f"time_generalization_hyperparams_timecourse_{base_tag}.tsv"
    hyper_summary_path = stats_dir / f"time_generalization_hyperparams_{base_tag}.tsv"
    perm_samples_path = stats_dir / f"time_generalization_diag_{base_tag}_perm_samples.npz"
    clusters_path = stats_dir / f"time_generalization_diag_{base_tag}_clusters.tsv"
    plot_path = plots_dir / f"time_generalization_{base_tag}"
    plot_outputs = [plot_path.with_suffix(f".{ext}") for ext in SAVE_FORMATS]
    stats_json_path = stats_path.with_suffix(".json")
    diag_json_path = diag_path.with_suffix(".json")

    perm_requested = permutations > 0
    plots_missing = any(not p.exists() for p in plot_outputs)
    stats_exists = stats_path.exists()
    diag_exists = diag_path.exists()
    diag_perm_exists = diag_perm_path.exists()
    perm_samples_exists = perm_samples_path.exists()
    diag_folds_exists = diag_folds_path.exists()
    perm_outputs_missing = perm_requested and (not diag_perm_exists or not perm_samples_exists)
    perm_should_run = perm_requested and (permutations_only or perm_outputs_missing)
    need_core_fit = (not stats_exists) or (not diag_exists) or (perm_should_run and not diag_folds_exists)

    cached_scores = cached_diag = cached_diag_meta = None
    if not need_core_fit:
        cached_scores = _load_scores_from_stats(stats_path)
        cached_diag = _load_diag_from_disk(diag_path)
        if cached_scores is None or cached_diag is None:
            logger.warning("Cached outputs incomplete or unreadable; recomputing core analysis.")
            need_core_fit = True
        else:
            cached_diag_meta = cached_diag[2]

    if permutations_only and not perm_requested:
        logger.info("Permutations-only mode requested but permutations=0; nothing to do.")
        if need_core_fit:
            logger.info("Core outputs missing; falling back to full analysis.")
        else:
            return stats_path


    # Build data matrix (n_epochs, n_channels, n_times)
    try:
        X = epochs.get_data(picks=picks)
    except Exception:
        X = epochs.get_data()
    X = np.ascontiguousarray(X)

    # Outer CV and nested inner-CV for hyperparameter tuning per subject
    if len(y) < 2:
        logger.error("Need ≥2 samples after masking to run temporal generalization")
        return None

    cv, groups = _build_cv(y, events_aligned_masked, mode=mode, n_splits=n_splits, seed=seed)
    try:
        outer_splits = list(cv.split(X, y, groups))
    except Exception:
        outer_splits = []
    if not outer_splits:
        logger.error("Could not create outer CV splits for temporal generalization")
        return None
    # Log outer CV summary
    try:
        outer_name = cv.__class__.__name__
        outer_n = getattr(cv, "n_splits", len(outer_splits))
        if groups is not None:
            g = np.asarray(groups)
            valid = g != -1
            uniq = np.unique(g[valid]) if np.any(valid) else np.array([])
            logger.info(f"Outer CV: {outer_name}(n={outer_n}); groups=run, unique_groups={uniq.size}")
            outer_desc = f"{outer_name}(n={outer_n}); groups=run; unique_groups={uniq.size}"
        else:
            logger.info(f"Outer CV: {outer_name}(n={outer_n}); no groups")
            outer_desc = f"{outer_name}(n={outer_n}); no groups"
    except Exception:
        outer_desc = "unknown"

    # Provenance (subject-level)
    prov = {
        "subject": subject,
        "task": task,
        "mode": mode,
        "target": target_col,
        "roi": roi or "all",
        "picks": (picks if isinstance(picks, str) else "eeg"),
        "tmin": (float(tmin) if tmin is not None else None),
        "tmax": (float(tmax) if tmax is not None else None),
        "decim": int(decim),
        "outer_cv": outer_desc,
        "n_splits": int(n_splits),
        "permutations": int(permutations),
        "time_jobs": int(time_jobs),
        "outer_jobs": int(outer_jobs),
        "perm_jobs": int(perm_jobs),
        "two_sided_classification": bool(two_sided_classification),
        "versions": {
            "mne": getattr(mne, "__version__", None),
            "sklearn": __import__("sklearn").__version__,
        },
        "heatmap_visual_centering": ("AUC-0.5 for classification; TSV stores raw AUC" if mode == "classification" else "raw r"),
    }

    # For classification, try to rebuild outer splits to ensure both classes appear in every train/test
    def _splits_have_both_classes(splits):
        for tr_i, te_i in splits:
            if np.unique(y[tr_i]).size < 2 or np.unique(y[te_i]).size < 2:
                return False
        return True
    if mode == "classification" and not _splits_have_both_classes(outer_splits):
        rebuilt = False
        req_splits = getattr(cv, "n_splits", n_splits)
        # Try StratifiedGroupKFold if groups available
        if groups is not None and _StratifiedGroupKFold is not None:
            try:
                cv2 = _StratifiedGroupKFold(n_splits=min(req_splits, max(2, len(np.unique(groups)))))
                splits2 = list(cv2.split(X, y, groups))
                if _splits_have_both_classes(splits2):
                    outer_splits = splits2
                    logger.info("Rebuilt outer CV with StratifiedGroupKFold to ensure class presence in all folds")
                    rebuilt = True
            except Exception:
                pass
        # Fallback: StratifiedKFold without groups, try multiple seeds
        if not rebuilt:
            for s_try in [seed, seed + 7, seed + 13, seed + 29]:
                try:
                    counts = np.bincount(y.astype(int))
                    min_count = int(counts.min()) if counts.size > 1 else 2
                    n_sf = max(2, min(req_splits, min_count))
                    cv3 = StratifiedKFold(n_splits=n_sf, shuffle=True, random_state=int(s_try))
                    splits3 = list(cv3.split(X, y))
                    if _splits_have_both_classes(splits3):
                        outer_splits = splits3
                        groups = None
                        logger.info(f"Rebuilt outer CV with StratifiedKFold(n={n_sf}, seed={int(s_try)}) to ensure class presence in all folds (groups dropped)")
                        rebuilt = True
                        break
                except Exception:
                    continue
        if not rebuilt:
            logger.warning("Could not rebuild outer CV to satisfy class presence per fold; some folds may be skipped.")

    logger.info("Nested hyperparameter tuning per train-time via inner CV at each train time point.")

    # Parameter grid shared across folds
    if mode == "classification":
        param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
        hyper_name = "clf__C"
    else:
        param_grid = {"reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        hyper_name = "reg__alpha"
    prov["param_grid"] = param_grid

        # Manual outer-CV to allow inner GridSearchCV within GeneralizingEstimator
    fold_scores: List[np.ndarray] = []
    fold_weights: List[float] = []
    fold_weight_map: Dict[int, float] = {}
    fold_hyper_values: List[float] = []  # median best param across train-times per fold
    fold_hyper_counts: List[int] = []    # number of train-times contributing
    hyper_time_per_fold: List[List[float]] = []
    diag_per_fold: List[np.ndarray] = []
    diag_per_fold_map: Dict[int, np.ndarray] = {}
    inner_descs: List[str] = []
    total_outer_folds = len(outer_splits)
    skipped_outer_folds = 0
    folds_total_meta = int(total_outer_folds)
    folds_used_meta = 0
    folds_skipped_meta = 0

    scores_mean: Optional[np.ndarray] = None
    train_time_grid: Optional[np.ndarray] = None
    test_time_grid: Optional[np.ndarray] = None
    diag: Optional[np.ndarray] = None
    diag_times: Optional[np.ndarray] = None

    times = np.round(epochs.times, 6)

    if need_core_fit:
        logger.info("Running core temporal generalization analysis (model fitting).")
        groups_array = np.asarray(groups) if groups is not None else None

        if outer_jobs == 1:
            fold_outputs = [
                _process_outer_fold(
                    fi,
                    np.asarray(tr),
                    np.asarray(te),
                    X,
                    y,
                    groups_array,
                    mode,
                    scoring,
                    param_grid,
                    hyper_name,
                    seed,
                    time_jobs,
                )
                for fi, (tr, te) in enumerate(outer_splits)
            ]
        else:
            fold_outputs = Parallel(
                n_jobs=int(outer_jobs),
                backend="loky",
                prefer="processes",
                pre_dispatch="n_jobs",
            )(
                delayed(_process_outer_fold)(
                    fi,
                    np.asarray(tr),
                    np.asarray(te),
                    X,
                    y,
                    groups_array,
                    mode,
                    scoring,
                    param_grid,
                    hyper_name,
                    seed,
                    time_jobs,
                )
                for fi, (tr, te) in enumerate(outer_splits)
            )

        fold_outputs_sorted = sorted(fold_outputs, key=lambda r: r["index"])
        for res in fold_outputs_sorted:
            fi = res["index"]
            if res.get("skip_reason") == "single-class":
                logger.warning(f"Fold {fi+1}: skipped due to single-class in train/test with grouped splits")
                skipped_outer_folds += 1
                continue
            inner_desc = res.get("inner_desc")
            if inner_desc:
                inner_descs.append(inner_desc)
                logger.info(f"Outer fold {fi+1}: Inner CV -> {inner_desc}")
            error_msg = res.get("error")
            if error_msg:
                logger.warning(f"Fold {fi+1}: nested CV failed with error: {error_msg}")
                continue
            sc_mat = res.get("score_matrix")
            if sc_mat is None:
                continue
            sc_mat = np.atleast_2d(sc_mat)
            fold_scores.append(sc_mat)
            weight_val = res.get("weight")
            if weight_val is None:
                weight_val = float(len(outer_splits[fi][1]))
            weight_float = float(weight_val)
            fold_weights.append(weight_float)
            fold_weight_map[fi] = weight_float
            diag_entry = res.get("diag")
            if diag_entry is not None:
                diag_arr = np.asarray(diag_entry)
                diag_per_fold.append(diag_arr)
                diag_per_fold_map[fi] = diag_arr
            hyper_med = res.get("hyper_median")
            if hyper_med is not None:
                fold_hyper_values.append(float(hyper_med))
                fold_hyper_counts.append(int(res.get("hyper_count", 0)))
                hyper_list = res.get("hyper_time")
                if hyper_list is not None:
                    hyper_time_per_fold.append(list(hyper_list))

        if not fold_scores:
            logger.error("All folds failed during nested temporal generalization")
            return None
        shapes = {tuple(s.shape) for s in fold_scores}
        if len(shapes) != 1:
            logger.error(
                f"Inconsistent fold score shapes across outer folds: {sorted(shapes)}. Check time-axis alignment and estimator outputs."
            )
            return None
        scores = np.stack(fold_scores, axis=0)
        w = np.asarray(fold_weights, dtype=float)
        w_sum = np.nansum(w)
        if w_sum <= 0:
            scores_mean = np.nanmean(scores, axis=0)
        else:
            scores_mean = np.nansum(scores * w[:, None, None], axis=0) / w_sum

        n_tr, n_te = scores_mean.shape
        train_time_grid = times[:n_tr]
        test_time_grid = times[:n_te]

        df = pd.DataFrame({
            "t_train": np.repeat(train_time_grid, n_te),
            "t_test": np.tile(test_time_grid, n_tr),
            "score": scores_mean.flatten(),
            "mode": mode,
            "target": target_col,
            "roi": roi or "all",
        })
        _ensure_dir(stats_path.parent)
        df.to_csv(stats_path, sep="\t", index=False)
        stats_exists = True

        try:
            prov["inner_cv_descriptions"] = inner_descs
            import json
            with open(stats_json_path, "w", encoding="utf-8") as f:
                json.dump(prov, f, indent=2)
        except Exception:
            pass

        diag = np.diag(scores_mean)
        diag_times = train_time_grid[: len(diag)]
        folds_used_meta = int(len(fold_scores))
        folds_skipped_meta = int(skipped_outer_folds)
        folds_total_meta = int(total_outer_folds)

        diag_df = pd.DataFrame({
            "time": diag_times,
            "score": diag,
            "mode": mode,
            "target": target_col,
            "roi": roi or "all",
            "folds_total": folds_total_meta,
            "folds_used": folds_used_meta,
            "folds_skipped": folds_skipped_meta,
        })
        diag_df.to_csv(diag_path, sep="\t", index=False)
        diag_exists = True

        try:
            import json
            diag_prov = {
                "subject": subject,
                "mode": mode,
                "target": target_col,
                "roi": roi or "all",
                "folds_total": folds_total_meta,
                "folds_used": folds_used_meta,
                "folds_skipped": folds_skipped_meta,
                "baseline": (0.5 if mode == "classification" else 0.0),
            }
            with open(diag_json_path, "w", encoding="utf-8") as f:
                json.dump(diag_prov, f, indent=2)
        except Exception:
            pass

        try:
            if diag_per_fold:
                diag_fold_mat = np.stack(diag_per_fold, axis=0)
                np.savez(diag_folds_path, diag_folds=diag_fold_mat, time=diag_times)
                diag_folds_exists = True
            if hyper_time_per_fold and len(hyper_time_per_fold[0]) == len(diag_times):
                rows = []
                for i, vals in enumerate(hyper_time_per_fold, 1):
                    for t, v in zip(diag_times, vals):
                        rows.append({"fold": i, "t_train": t, "hyper": v})
                pd.DataFrame(rows).to_csv(
                    hyper_timecourse_path,
                    sep="\t", index=False)
        except Exception:
            pass
    else:
        logger.info("Core temporal generalization outputs already exist; reusing cached files.")
        fold_weight_map = {fi: float(len(te)) for fi, (_, te) in enumerate(outer_splits)}
        if cached_scores is not None:
            scores_mean, train_time_grid, test_time_grid = cached_scores
        if cached_diag is not None:
            diag, diag_times, meta = cached_diag
            if meta:
                folds_total_meta = int(meta.get("folds_total", total_outer_folds))
                folds_used_meta = int(meta.get("folds_used", folds_total_meta))
                folds_skipped_meta = int(meta.get("folds_skipped", max(0, folds_total_meta - folds_used_meta)))
                skipped_outer_folds = int(folds_skipped_meta)
        if diag is None or diag_times is None:
            logger.error("Diagonal results missing from cached outputs; rerun analysis with --permutations-only disabled.")
            return None
        diag_fold_arr = _load_diag_folds(diag_folds_path) if diag_folds_exists else None
        if diag_fold_arr is not None:
            diag_per_fold_map = {i: np.asarray(diag_fold_arr[i]) for i in range(diag_fold_arr.shape[0])}

    if perm_should_run and outer_splits:
        save_perm_samples_flag = perm_requested
        rng_global = np.random.default_rng(int(seed))
        obs_fold_diags = {idx: np.asarray(val).copy() for idx, val in diag_per_fold_map.items()}
        obs_fold_weights = {
            idx: float(fold_weight_map.get(idx, float(len(outer_splits[idx][1]))))
            for idx in obs_fold_diags
        }
        if obs_fold_diags:
            lengths = {len(arr) for arr in obs_fold_diags.values()}
            if len(lengths) != 1:
                logger.error(
                    f"Inconsistent observed diagonal lengths across folds: {sorted(lengths)}. Aborting permutations."
                )
                return None
            ordered_idx = sorted(obs_fold_diags)
            obs_stack = np.stack([obs_fold_diags[i] for i in ordered_idx], axis=0)
            w_obs = np.asarray([obs_fold_weights[i] for i in ordered_idx], dtype=float)
            if np.nansum(w_obs) <= 0:
                diag_obs = np.nanmean(obs_stack, axis=0)
            else:
                diag_obs = np.nansum(obs_stack * w_obs[:, None], axis=0) / np.nansum(w_obs)
        else:
            if diag is None:
                logger.error("Cannot compute permutations without diagonal scores.")
                return None
            diag_obs = diag.copy()

        null_all: List[np.ndarray] = []
        obs_perm_all: List[np.ndarray] = []
        perm_count = permutations
        seed_sequence = np.random.SeedSequence(int(seed))
        child_sequences = seed_sequence.spawn(perm_count)

        perm_args = [
            (
                pi,
                child_sequences[pi],
                outer_splits,
                X,
                y,
                groups_array,
                mode,
                scoring,
                param_grid,
                hyper_name,
                seed,
                time_jobs,
                obs_fold_diags,
                obs_fold_weights,
            )
            for pi in range(perm_count)
        ]

        if perm_jobs == 1:
            perm_outputs = [
                _run_permutation_worker(*call_args)
                for call_args in perm_args
            ]
        else:
            try:
                perm_outputs = Parallel(
                    n_jobs=int(perm_jobs),
                    backend="loky",
                    prefer="processes",
                    pre_dispatch="n_jobs",
                )(
                    delayed(_run_permutation_worker)(*call_args)
                    for call_args in perm_args
                )
            except BrokenProcessPool as exc:
                cause_bits = " ".join(
                    str(bit)
                    for bit in (exc, getattr(exc, "__cause__", None), getattr(exc, "__context__", None))
                    if bit
                )
                if "'numpy.ufunc' object has no attribute '__module__'" in cause_bits:
                    logger.warning(
                        "Permutation workers crashed due to numpy serialization issue; retrying with thread-based parallelism."
                    )
                    perm_outputs = Parallel(
                        n_jobs=int(perm_jobs),
                        backend="threading",
                        prefer="threads",
                        pre_dispatch="n_jobs",
                    )(
                        delayed(_run_permutation_worker)(*call_args)
                        for call_args in perm_args
                    )
                else:
                    raise

        perm_outputs_sorted = sorted(perm_outputs, key=lambda r: r["index"])
        for res in perm_outputs_sorted:
            idx = res.get("index", 0)
            error_msg = res.get("error")
            if error_msg:
                logger.error(f"Permutation {idx + 1}: {error_msg}. Aborting permutations.")
                return None
            if res.get("skipped"):
                continue
            diag_null = res.get("diag_null")
            if diag_null is not None:
                null_all.append(np.asarray(diag_null))
            diag_obs_perm = res.get("diag_obs")
            if diag_obs_perm is not None:
                obs_perm_all.append(np.asarray(diag_obs_perm))
        if null_all:
            null_mat = np.vstack(null_all)
            obs_perm_mat = np.vstack(obs_perm_all) if obs_perm_all else None
            if obs_perm_mat is None and null_mat.shape[1] != len(diag_obs):
                logger.error("Permutation null/observed diagonal length mismatch; aborting.")
                return None
            if mode == "regression":
                null_mat_z = _fisher_z(null_mat)
                null_mean_z = np.nanmean(null_mat_z, axis=0)
                null_sd_z = np.nanstd(null_mat_z, axis=0, ddof=1)
                null_mean_r = np.nanmean(null_mat, axis=0)
                null_sd_r = np.nanstd(null_mat, axis=0, ddof=1)
            else:
                null_mean = np.nanmean(null_mat, axis=0)
                null_sd = np.nanstd(null_mat, axis=0, ddof=1)
            if save_perm_samples_flag:
                try:
                    save_idx = None
                    if (
                        isinstance(max_perm_save, int)
                        and max_perm_save > 0
                        and null_mat.shape[0] > max_perm_save
                    ):
                        save_idx = rng_global.choice(
                            null_mat.shape[0], size=int(max_perm_save), replace=False
                        )
                    if save_idx is not None:
                        null_to_save = null_mat[save_idx]
                    else:
                        null_to_save = null_mat
                    perm_npz = perm_samples_path
                    t_len = int(null_mat.shape[1])
                    np.savez(perm_npz, null_diag=null_to_save, time=diag_times[:t_len])
                    perm_samples_exists = True
                    logger.info(
                        f"Saved subject permutation samples: {perm_npz} ({null_to_save.shape[0]}x{null_to_save.shape[1]})"
                    )
                except Exception as e:
                    logger.warning(f"Failed saving subject permutation samples: {e}")
            if mode == "regression":
                obs_for_p = np.abs(_fisher_z(obs_perm_mat if obs_perm_mat is not None else diag_obs))
                null_for_p = np.abs(_fisher_z(null_mat))
            else:
                if bool(two_sided_classification):
                    base = 0.5
                    obs_raw = obs_perm_mat if obs_perm_mat is not None else diag_obs
                    obs_for_p = np.abs(obs_raw - base)
                    null_for_p = np.abs(null_mat - base)
                else:
                    obs_for_p = obs_perm_mat if obs_perm_mat is not None else diag_obs
                    null_for_p = null_mat
            if obs_perm_mat is not None:
                p_emp = np.array([
                    (1.0 + np.sum(null_for_p[:, i] >= obs_for_p[:, i])) / (null_for_p.shape[0] + 1.0)
                    for i in range(null_for_p.shape[1])
                ])
            else:
                p_emp = np.array([
                    (1.0 + np.sum(null_for_p[:, i] >= obs_for_p[i])) / (null_for_p.shape[0] + 1.0)
                    for i in range(len(obs_for_p))
                ])
            q_emp = _fdr_bh(p_emp, alpha=0.05)
            diag_times_eff = diag_times[:len(diag_obs)]
            cluster_two_sided = True if mode == "regression" else bool(two_sided_classification)
            baseline_val = 0.0 if mode == "regression" else 0.5
            cluster_records, cluster_labels = _compute_diag_clusters(
                diag_obs,
                null_mat,
                diag_times_eff,
                mode,
                baseline_val,
                cluster_two_sided,
                DIAG_CLUSTER_P,
            )
            cluster_labels = np.asarray(cluster_labels, dtype=int)
            if cluster_labels.shape[0] != len(diag_obs):
                cluster_labels = np.full(len(diag_obs), -1, dtype=int)
            cluster_lookup = {rec["cluster_id"]: rec["p_value"] for rec in cluster_records}
            cluster_p_vals = np.array([
                cluster_lookup.get(int(cid), np.nan) if cid >= 0 else np.nan
                for cid in cluster_labels
            ], dtype=float)
            cluster_sig = cluster_p_vals < 0.05
            cluster_note = f"cluster-mass (alpha={DIAG_CLUSTER_P:.3f})"
            if mode == "regression":
                diag_perm_df = pd.DataFrame({
                    "time": diag_times_eff,
                    "score": diag_obs,
                    "mode": mode,
                    "target": target_col,
                    "roi": roi or "all",
                    "p_emp": p_emp,
                    "q_fdr": q_emp,
                    "fdr_method": "BH (pointwise)",
                    "tail": "two-sided",
                    "null_mean_z": null_mean_z[:len(diag_obs)],
                    "null_sd_z": null_sd_z[:len(diag_obs)],
                    "null_mean_r": null_mean_r[:len(diag_obs)],
                    "null_sd_r": null_sd_r[:len(diag_obs)],
                    "n_perm": null_mat.shape[0],
                    "method": "subject-permutation",
                    "cluster_id": cluster_labels,
                    "cluster_p": cluster_p_vals,
                    "cluster_sig": cluster_sig,
                    "cluster_method": cluster_note,
                    "cluster_alpha": float(DIAG_CLUSTER_P),
                })
            else:
                diag_perm_df = pd.DataFrame({
                    "time": diag_times_eff,
                    "score": diag_obs,
                    "mode": mode,
                    "target": target_col,
                    "roi": roi or "all",
                    "p_emp": p_emp,
                    "q_fdr": q_emp,
                    "fdr_method": "BH (pointwise)",
                    "tail": ("two-sided" if bool(two_sided_classification) else "one-sided"),
                    "null_mean": null_mean[:len(diag_obs)],
                    "null_sd": null_sd[:len(diag_obs)],
                    "n_perm": null_mat.shape[0],
                    "method": "subject-permutation",
                    "cluster_id": cluster_labels,
                    "cluster_p": cluster_p_vals,
                    "cluster_sig": cluster_sig,
                    "cluster_method": cluster_note,
                    "cluster_alpha": float(DIAG_CLUSTER_P),
                })
            diag_perm_df["cluster_sig"] = diag_perm_df["cluster_sig"].astype(bool)
            if cluster_records:
                try:
                    pd.DataFrame(cluster_records).to_csv(clusters_path, sep="\t", index=False)
                    logger.info(f"Saved diagonal clusters: {clusters_path}")
                except Exception as e:
                    logger.warning(f"Failed to save diagonal cluster summary: {e}")
            diag_perm_df.to_csv(diag_perm_path, sep="\t", index=False)
            diag_perm_exists = True
            logger.info(f"Saved diagonal permutations: {diag_perm_path}")
    else:
        if perm_requested and not perm_should_run:
            logger.info("Permutation outputs present; skipping permutation resampling.")
# Save per-fold tuned hyperparameter summary (subject-level)
    try:
        if fold_hyper_values:
            hyper_df = pd.DataFrame({
                "fold": np.arange(1, len(fold_hyper_values) + 1, dtype=int),
                hyper_name: fold_hyper_values,
                "n_train_times_used": (fold_hyper_counts if len(fold_hyper_counts) == len(fold_hyper_values) else [np.nan] * len(fold_hyper_values)),
            })
            hyper_df["mode"] = mode
            hyper_df["target"] = target_col
            hyper_df["roi"] = roi or "all"
            hyper_df["subject"] = subject
            hyper_df["median"] = float(np.median(fold_hyper_values))
            hyper_df["mean"] = float(np.mean(fold_hyper_values))
            hyper_path = hyper_summary_path
            hyper_df.to_csv(hyper_path, sep="\t", index=False)
            logger.info(f"Saved nested-CV hyperparams: {hyper_path}")
    except Exception:
        pass

        # Plot heatmap and diagonal time course
    plot_needed = need_core_fit or plots_missing
    if plot_needed:
        if (
            scores_mean is not None
            and train_time_grid is not None
            and test_time_grid is not None
            and diag is not None
            and diag_times is not None
        ):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ax = axes[0]
            if mode == "classification":
                plot_matrix = scores_mean - 0.5
                plot_cmap = "RdBu_r"
                plot_vmin, plot_vmax = -0.2, 0.2
                cbar_label = "AUC - 0.5"
            else:
                plot_matrix = scores_mean
                plot_cmap = cmap
                plot_vmin, plot_vmax = vmin, vmax
                cbar_label = "r(pred, y)"
            extent = [test_time_grid[0], test_time_grid[-1], train_time_grid[0], train_time_grid[-1]]
            im = ax.imshow(
                plot_matrix,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=plot_cmap,
                vmin=plot_vmin,
                vmax=plot_vmax,
            )
            ax.set_xlabel("Test time (s)")
            ax.set_ylabel("Train time (s)")
            ax.set_title(f"Temporal Generalization - sub-{subject}\\n{mode} on '{target_col}' (ROI: {roi or 'all'})")
            ax.axline((train_time_grid[0], train_time_grid[0]), (test_time_grid[-1], train_time_grid[-1]), color="k", linestyle="--", linewidth=0.8, alpha=0.6)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)

            axes[1].plot(diag_times, diag, color="#1f77b4", lw=2)
            axes[1].axhline(0.5 if mode == "classification" else 0.0, color="k", ls=":", alpha=0.6)
            axes[1].axvline(0.0, color="k", ls="--", alpha=0.6)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Diagonal score")
            axes[1].set_title("Diagonal (train=test) decoding")
            plt.tight_layout()

            _ensure_dir(plot_path.parent)
            _save_fig(fig, plot_path)
            plots_missing = False
            logger.info(f"Saved TGM plot: {plot_path.with_suffix('.png')}")
        else:
            logger.warning("Skipping plot generation because cached scores or diagonal data are unavailable.")
    elif not need_core_fit:
        logger.info("TGM plot already exists; skipping plot generation.")

    if stats_exists:
        logger.info(f"Saved TGM stats: {stats_path}")
    return stats_path


    # Build data matrix (n_epochs, n_channels, n_times)
    try:
        X = epochs.get_data(picks=picks)
    except Exception:
        X = epochs.get_data()
    X = np.ascontiguousarray(X)

    # Outer CV and nested inner-CV for hyperparameter tuning per subject
    if len(y) < 2:
        logger.error("Need ≥2 samples after masking to run temporal generalization")
        return None

    cv, groups = _build_cv(y, events_aligned_masked, mode=mode, n_splits=n_splits, seed=seed)
    try:
        outer_splits = list(cv.split(X, y, groups))
    except Exception:
        outer_splits = []
    if not outer_splits:
        logger.error("Could not create outer CV splits for temporal generalization")
        return None
    # Log outer CV summary
    try:
        outer_name = cv.__class__.__name__
        outer_n = getattr(cv, "n_splits", len(outer_splits))
        if groups is not None:
            g = np.asarray(groups)
            valid = g != -1
            uniq = np.unique(g[valid]) if np.any(valid) else np.array([])
            logger.info(f"Outer CV: {outer_name}(n={outer_n}); groups=run, unique_groups={uniq.size}")
            outer_desc = f"{outer_name}(n={outer_n}); groups=run; unique_groups={uniq.size}"
        else:
            logger.info(f"Outer CV: {outer_name}(n={outer_n}); no groups")
            outer_desc = f"{outer_name}(n={outer_n}); no groups"
    except Exception:
        outer_desc = "unknown"

    # Provenance (subject-level)
    prov = {
        "subject": subject,
        "task": task,
        "mode": mode,
        "target": target_col,
        "roi": roi or "all",
        "picks": (picks if isinstance(picks, str) else "eeg"),
        "tmin": (float(tmin) if tmin is not None else None),
        "tmax": (float(tmax) if tmax is not None else None),
        "decim": int(decim),
        "outer_cv": outer_desc,
        "n_splits": int(n_splits),
        "permutations": int(permutations),
        "time_jobs": int(time_jobs),
        "outer_jobs": int(outer_jobs),
        "perm_jobs": int(perm_jobs),
        "two_sided_classification": bool(two_sided_classification),
        "versions": {
            "mne": getattr(mne, "__version__", None),
            "sklearn": __import__("sklearn").__version__,
        },
        "heatmap_visual_centering": ("AUC-0.5 for classification; TSV stores raw AUC" if mode == "classification" else "raw r"),
    }

    # For classification, try to rebuild outer splits to ensure both classes appear in every train/test
    def _splits_have_both_classes(splits):
        for tr_i, te_i in splits:
            if np.unique(y[tr_i]).size < 2 or np.unique(y[te_i]).size < 2:
                return False
        return True
    if mode == "classification" and not _splits_have_both_classes(outer_splits):
        rebuilt = False
        req_splits = getattr(cv, "n_splits", n_splits)
        # Try StratifiedGroupKFold if groups available
        if groups is not None and _StratifiedGroupKFold is not None:
            try:
                cv2 = _StratifiedGroupKFold(n_splits=min(req_splits, max(2, len(np.unique(groups)))))
                splits2 = list(cv2.split(X, y, groups))
                if _splits_have_both_classes(splits2):
                    outer_splits = splits2
                    logger.info("Rebuilt outer CV with StratifiedGroupKFold to ensure class presence in all folds")
                    rebuilt = True
            except Exception:
                pass
        # Fallback: StratifiedKFold without groups, try multiple seeds
        if not rebuilt:
            for s_try in [seed, seed + 7, seed + 13, seed + 29]:
                try:
                    counts = np.bincount(y.astype(int))
                    min_count = int(counts.min()) if counts.size > 1 else 2
                    n_sf = max(2, min(req_splits, min_count))
                    cv3 = StratifiedKFold(n_splits=n_sf, shuffle=True, random_state=int(s_try))
                    splits3 = list(cv3.split(X, y))
                    if _splits_have_both_classes(splits3):
                        outer_splits = splits3
                        groups = None
                        logger.info(f"Rebuilt outer CV with StratifiedKFold(n={n_sf}, seed={int(s_try)}) to ensure class presence in all folds (groups dropped)")
                        rebuilt = True
                        break
                except Exception:
                    continue
        if not rebuilt:
            logger.warning("Could not rebuild outer CV to satisfy class presence per fold; some folds may be skipped.")

    logger.info("Nested hyperparameter tuning per train-time via inner CV at each train time point.")

    # Parameter grid shared across folds
    if mode == "classification":
        param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
        hyper_name = "clf__C"
    else:
        param_grid = {"reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        hyper_name = "reg__alpha"
    prov["param_grid"] = param_grid

        # Manual outer-CV to allow inner GridSearchCV within GeneralizingEstimator
    fold_scores: List[np.ndarray] = []
    fold_weights: List[float] = []
    fold_weight_map: Dict[int, float] = {}
    fold_hyper_values: List[float] = []  # median best param across train-times per fold
    fold_hyper_counts: List[int] = []    # number of train-times contributing
    hyper_time_per_fold: List[List[float]] = []
    diag_per_fold: List[np.ndarray] = []
    diag_per_fold_map: Dict[int, np.ndarray] = {}
    inner_descs: List[str] = []
    total_outer_folds = len(outer_splits)
    skipped_outer_folds = 0
    folds_total_meta = int(total_outer_folds)
    folds_used_meta = 0
    folds_skipped_meta = 0

    scores_mean: Optional[np.ndarray] = None
    train_time_grid: Optional[np.ndarray] = None
    test_time_grid: Optional[np.ndarray] = None
    diag: Optional[np.ndarray] = None
    diag_times: Optional[np.ndarray] = None

    times = np.round(epochs.times, 6)

    if need_core_fit:
        logger.info("Running core temporal generalization analysis (model fitting).")
        groups_array = np.asarray(groups) if groups is not None else None

        if outer_jobs == 1:
            fold_outputs = [
                _process_outer_fold(
                    fi,
                    np.asarray(tr),
                    np.asarray(te),
                    X,
                    y,
                    groups_array,
                    mode,
                    scoring,
                    param_grid,
                    hyper_name,
                    seed,
                    time_jobs,
                )
                for fi, (tr, te) in enumerate(outer_splits)
            ]
        else:
            fold_outputs = Parallel(
                n_jobs=int(outer_jobs),
                backend="loky",
                prefer="processes",
                pre_dispatch="n_jobs",
            )(
                delayed(_process_outer_fold)(
                    fi,
                    np.asarray(tr),
                    np.asarray(te),
                    X,
                    y,
                    groups_array,
                    mode,
                    scoring,
                    param_grid,
                    hyper_name,
                    seed,
                    time_jobs,
                )
                for fi, (tr, te) in enumerate(outer_splits)
            )

        fold_outputs_sorted = sorted(fold_outputs, key=lambda r: r["index"])
        for res in fold_outputs_sorted:
            fi = res["index"]
            if res.get("skip_reason") == "single-class":
                logger.warning(f"Fold {fi+1}: skipped due to single-class in train/test with grouped splits")
                skipped_outer_folds += 1
                continue
            inner_desc = res.get("inner_desc")
            if inner_desc:
                inner_descs.append(inner_desc)
                logger.info(f"Outer fold {fi+1}: Inner CV -> {inner_desc}")
            error_msg = res.get("error")
            if error_msg:
                logger.warning(f"Fold {fi+1}: nested CV failed with error: {error_msg}")
                continue
            sc_mat = res.get("score_matrix")
            if sc_mat is None:
                continue
            sc_mat = np.atleast_2d(sc_mat)
            fold_scores.append(sc_mat)
            weight_val = res.get("weight")
            if weight_val is None:
                weight_val = float(len(outer_splits[fi][1]))
            weight_float = float(weight_val)
            fold_weights.append(weight_float)
            fold_weight_map[fi] = weight_float
            diag_entry = res.get("diag")
            if diag_entry is not None:
                diag_arr = np.asarray(diag_entry)
                diag_per_fold.append(diag_arr)
                diag_per_fold_map[fi] = diag_arr
            hyper_med = res.get("hyper_median")
            if hyper_med is not None:
                fold_hyper_values.append(float(hyper_med))
                fold_hyper_counts.append(int(res.get("hyper_count", 0)))
                hyper_list = res.get("hyper_time")
                if hyper_list is not None:
                    hyper_time_per_fold.append(list(hyper_list))

        if not fold_scores:
            logger.error("All folds failed during nested temporal generalization")
            return None
        shapes = {tuple(s.shape) for s in fold_scores}
        if len(shapes) != 1:
            logger.error(
                f"Inconsistent fold score shapes across outer folds: {sorted(shapes)}. Check time-axis alignment and estimator outputs."
            )
            return None
        scores = np.stack(fold_scores, axis=0)
        w = np.asarray(fold_weights, dtype=float)
        w_sum = np.nansum(w)
        if w_sum <= 0:
            scores_mean = np.nanmean(scores, axis=0)
        else:
            scores_mean = np.nansum(scores * w[:, None, None], axis=0) / w_sum

        n_tr, n_te = scores_mean.shape
        train_time_grid = times[:n_tr]
        test_time_grid = times[:n_te]

        df = pd.DataFrame({
            "t_train": np.repeat(train_time_grid, n_te),
            "t_test": np.tile(test_time_grid, n_tr),
            "score": scores_mean.flatten(),
            "mode": mode,
            "target": target_col,
            "roi": roi or "all",
        })
        _ensure_dir(stats_path.parent)
        df.to_csv(stats_path, sep="\t", index=False)
        stats_exists = True

        try:
            prov["inner_cv_descriptions"] = inner_descs
            import json
            with open(stats_json_path, "w", encoding="utf-8") as f:
                json.dump(prov, f, indent=2)
        except Exception:
            pass

        diag = np.diag(scores_mean)
        diag_times = train_time_grid[: len(diag)]
        folds_used_meta = int(len(fold_scores))
        folds_skipped_meta = int(skipped_outer_folds)
        folds_total_meta = int(total_outer_folds)

        diag_df = pd.DataFrame({
            "time": diag_times,
            "score": diag,
            "mode": mode,
            "target": target_col,
            "roi": roi or "all",
            "folds_total": folds_total_meta,
            "folds_used": folds_used_meta,
            "folds_skipped": folds_skipped_meta,
        })
        diag_df.to_csv(diag_path, sep="\t", index=False)
        diag_exists = True

        try:
            import json
            diag_prov = {
                "subject": subject,
                "mode": mode,
                "target": target_col,
                "roi": roi or "all",
                "folds_total": folds_total_meta,
                "folds_used": folds_used_meta,
                "folds_skipped": folds_skipped_meta,
                "baseline": (0.5 if mode == "classification" else 0.0),
            }
            with open(diag_json_path, "w", encoding="utf-8") as f:
                json.dump(diag_prov, f, indent=2)
        except Exception:
            pass

        try:
            if diag_per_fold:
                diag_fold_mat = np.stack(diag_per_fold, axis=0)
                np.savez(diag_folds_path, diag_folds=diag_fold_mat, time=diag_times)
                diag_folds_exists = True
            if hyper_time_per_fold and len(hyper_time_per_fold[0]) == len(diag_times):
                rows = []
                for i, vals in enumerate(hyper_time_per_fold, 1):
                    for t, v in zip(diag_times, vals):
                        rows.append({"fold": i, "t_train": t, "hyper": v})
                pd.DataFrame(rows).to_csv(
                    hyper_timecourse_path,
                    sep="\t", index=False)
        except Exception:
            pass
    else:
        logger.info("Core temporal generalization outputs already exist; reusing cached files.")
        fold_weight_map = {fi: float(len(te)) for fi, (_, te) in enumerate(outer_splits)}
        if cached_scores is not None:
            scores_mean, train_time_grid, test_time_grid = cached_scores
        if cached_diag is not None:
            diag, diag_times, meta = cached_diag
            if meta:
                folds_total_meta = int(meta.get("folds_total", total_outer_folds))
                folds_used_meta = int(meta.get("folds_used", folds_total_meta))
                folds_skipped_meta = int(meta.get("folds_skipped", max(0, folds_total_meta - folds_used_meta)))
                skipped_outer_folds = int(folds_skipped_meta)
        if diag is None or diag_times is None:
            logger.error("Diagonal results missing from cached outputs; rerun analysis with --permutations-only disabled.")
            return None
        diag_fold_arr = _load_diag_folds(diag_folds_path) if diag_folds_exists else None
        if diag_fold_arr is not None:
            diag_per_fold_map = {i: np.asarray(diag_fold_arr[i]) for i in range(diag_fold_arr.shape[0])}

    if perm_should_run and outer_splits:
        save_perm_samples_flag = perm_requested
        rng_global = np.random.default_rng(int(seed))
        obs_fold_diags = {idx: np.asarray(val).copy() for idx, val in diag_per_fold_map.items()}
        obs_fold_weights = {
            idx: float(fold_weight_map.get(idx, float(len(outer_splits[idx][1]))))
            for idx in obs_fold_diags
        }
        if obs_fold_diags:
            lengths = {len(arr) for arr in obs_fold_diags.values()}
            if len(lengths) != 1:
                logger.error(
                    f"Inconsistent observed diagonal lengths across folds: {sorted(lengths)}. Aborting permutations."
                )
                return None
            ordered_idx = sorted(obs_fold_diags)
            obs_stack = np.stack([obs_fold_diags[i] for i in ordered_idx], axis=0)
            w_obs = np.asarray([obs_fold_weights[i] for i in ordered_idx], dtype=float)
            if np.nansum(w_obs) <= 0:
                diag_obs = np.nanmean(obs_stack, axis=0)
            else:
                diag_obs = np.nansum(obs_stack * w_obs[:, None], axis=0) / np.nansum(w_obs)
        else:
            if diag is None:
                logger.error("Cannot compute permutations without diagonal scores.")
                return None
            diag_obs = diag.copy()

        null_all: List[np.ndarray] = []
        obs_perm_all: List[np.ndarray] = []
        perm_count = permutations
        seed_sequence = np.random.SeedSequence(int(seed))
        child_sequences = seed_sequence.spawn(perm_count)

        perm_args = [
            (
                pi,
                child_sequences[pi],
                outer_splits,
                X,
                y,
                groups_array,
                mode,
                scoring,
                param_grid,
                hyper_name,
                seed,
                time_jobs,
                obs_fold_diags,
                obs_fold_weights,
            )
            for pi in range(perm_count)
        ]

        if perm_jobs == 1:
            perm_outputs = [
                _run_permutation_worker(*call_args)
                for call_args in perm_args
            ]
        else:
            try:
                perm_outputs = Parallel(
                    n_jobs=int(perm_jobs),
                    backend="loky",
                    prefer="processes",
                    pre_dispatch="n_jobs",
                )(
                    delayed(_run_permutation_worker)(*call_args)
                    for call_args in perm_args
                )
            except BrokenProcessPool as exc:
                cause_bits = " ".join(
                    str(bit)
                    for bit in (exc, getattr(exc, "__cause__", None), getattr(exc, "__context__", None))
                    if bit
                )
                if "'numpy.ufunc' object has no attribute '__module__'" in cause_bits:
                    logger.warning(
                        "Permutation workers crashed due to numpy serialization issue; retrying with thread-based parallelism."
                    )
                    perm_outputs = Parallel(
                        n_jobs=int(perm_jobs),
                        backend="threading",
                        prefer="threads",
                        pre_dispatch="n_jobs",
                    )(
                        delayed(_run_permutation_worker)(*call_args)
                        for call_args in perm_args
                    )
                else:
                    raise

        perm_outputs_sorted = sorted(perm_outputs, key=lambda r: r["index"])
        for res in perm_outputs_sorted:
            idx = res.get("index", 0)
            error_msg = res.get("error")
            if error_msg:
                logger.error(f"Permutation {idx + 1}: {error_msg}. Aborting permutations.")
                return None
            if res.get("skipped"):
                continue
            diag_null = res.get("diag_null")
            if diag_null is not None:
                null_all.append(np.asarray(diag_null))
            diag_obs_perm = res.get("diag_obs")
            if diag_obs_perm is not None:
                obs_perm_all.append(np.asarray(diag_obs_perm))
        if null_all:
            null_mat = np.vstack(null_all)
            obs_perm_mat = np.vstack(obs_perm_all) if obs_perm_all else None
            if obs_perm_mat is None and null_mat.shape[1] != len(diag_obs):
                logger.error("Permutation null/observed diagonal length mismatch; aborting.")
                return None
            if mode == "regression":
                null_mat_z = _fisher_z(null_mat)
                null_mean_z = np.nanmean(null_mat_z, axis=0)
                null_sd_z = np.nanstd(null_mat_z, axis=0, ddof=1)
                null_mean_r = np.nanmean(null_mat, axis=0)
                null_sd_r = np.nanstd(null_mat, axis=0, ddof=1)
            else:
                null_mean = np.nanmean(null_mat, axis=0)
                null_sd = np.nanstd(null_mat, axis=0, ddof=1)
            if save_perm_samples_flag:
                try:
                    save_idx = None
                    if (
                        isinstance(max_perm_save, int)
                        and max_perm_save > 0
                        and null_mat.shape[0] > max_perm_save
                    ):
                        save_idx = rng_global.choice(
                            null_mat.shape[0], size=int(max_perm_save), replace=False
                        )
                    if save_idx is not None:
                        null_to_save = null_mat[save_idx]
                    else:
                        null_to_save = null_mat
                    perm_npz = perm_samples_path
                    t_len = int(null_mat.shape[1])
                    np.savez(perm_npz, null_diag=null_to_save, time=diag_times[:t_len])
                    perm_samples_exists = True
                    logger.info(
                        f"Saved subject permutation samples: {perm_npz} ({null_to_save.shape[0]}x{null_to_save.shape[1]})"
                    )
                except Exception as e:
                    logger.warning(f"Failed saving subject permutation samples: {e}")
            if mode == "regression":
                obs_for_p = np.abs(_fisher_z(obs_perm_mat if obs_perm_mat is not None else diag_obs))
                null_for_p = np.abs(_fisher_z(null_mat))
            else:
                if bool(two_sided_classification):
                    base = 0.5
                    obs_raw = obs_perm_mat if obs_perm_mat is not None else diag_obs
                    obs_for_p = np.abs(obs_raw - base)
                    null_for_p = np.abs(null_mat - base)
                else:
                    obs_for_p = obs_perm_mat if obs_perm_mat is not None else diag_obs
                    null_for_p = null_mat
            if obs_perm_mat is not None:
                p_emp = np.array([
                    (1.0 + np.sum(null_for_p[:, i] >= obs_for_p[:, i])) / (null_for_p.shape[0] + 1.0)
                    for i in range(null_for_p.shape[1])
                ])
            else:
                p_emp = np.array([
                    (1.0 + np.sum(null_for_p[:, i] >= obs_for_p[i])) / (null_for_p.shape[0] + 1.0)
                    for i in range(len(obs_for_p))
                ])
            q_emp = _fdr_bh(p_emp, alpha=0.05)
            diag_times_eff = diag_times[:len(diag_obs)]
            cluster_two_sided = True if mode == "regression" else bool(two_sided_classification)
            baseline_val = 0.0 if mode == "regression" else 0.5
            cluster_records, cluster_labels = _compute_diag_clusters(
                diag_obs,
                null_mat,
                diag_times_eff,
                mode,
                baseline_val,
                cluster_two_sided,
                DIAG_CLUSTER_P,
            )
            cluster_labels = np.asarray(cluster_labels, dtype=int)
            if cluster_labels.shape[0] != len(diag_obs):
                cluster_labels = np.full(len(diag_obs), -1, dtype=int)
            cluster_lookup = {rec["cluster_id"]: rec["p_value"] for rec in cluster_records}
            cluster_p_vals = np.array([
                cluster_lookup.get(int(cid), np.nan) if cid >= 0 else np.nan
                for cid in cluster_labels
            ], dtype=float)
            cluster_sig = cluster_p_vals < 0.05
            cluster_note = f"cluster-mass (alpha={DIAG_CLUSTER_P:.3f})"
            if mode == "regression":
                diag_perm_df = pd.DataFrame({
                    "time": diag_times_eff,
                    "score": diag_obs,
                    "mode": mode,
                    "target": target_col,
                    "roi": roi or "all",
                    "p_emp": p_emp,
                    "q_fdr": q_emp,
                    "fdr_method": "BH (pointwise)",
                    "tail": "two-sided",
                    "null_mean_z": null_mean_z[:len(diag_obs)],
                    "null_sd_z": null_sd_z[:len(diag_obs)],
                    "null_mean_r": null_mean_r[:len(diag_obs)],
                    "null_sd_r": null_sd_r[:len(diag_obs)],
                    "n_perm": null_mat.shape[0],
                    "method": "subject-permutation",
                    "cluster_id": cluster_labels,
                    "cluster_p": cluster_p_vals,
                    "cluster_sig": cluster_sig,
                    "cluster_method": cluster_note,
                    "cluster_alpha": float(DIAG_CLUSTER_P),
                })
            else:
                diag_perm_df = pd.DataFrame({
                    "time": diag_times_eff,
                    "score": diag_obs,
                    "mode": mode,
                    "target": target_col,
                    "roi": roi or "all",
                    "p_emp": p_emp,
                    "q_fdr": q_emp,
                    "fdr_method": "BH (pointwise)",
                    "tail": ("two-sided" if bool(two_sided_classification) else "one-sided"),
                    "null_mean": null_mean[:len(diag_obs)],
                    "null_sd": null_sd[:len(diag_obs)],
                    "n_perm": null_mat.shape[0],
                    "method": "subject-permutation",
                    "cluster_id": cluster_labels,
                    "cluster_p": cluster_p_vals,
                    "cluster_sig": cluster_sig,
                    "cluster_method": cluster_note,
                    "cluster_alpha": float(DIAG_CLUSTER_P),
                })
            diag_perm_df["cluster_sig"] = diag_perm_df["cluster_sig"].astype(bool)
            if cluster_records:
                try:
                    pd.DataFrame(cluster_records).to_csv(clusters_path, sep="\t", index=False)
                    logger.info(f"Saved diagonal clusters: {clusters_path}")
                except Exception as e:
                    logger.warning(f"Failed to save diagonal cluster summary: {e}")
            diag_perm_df.to_csv(diag_perm_path, sep="\t", index=False)
            diag_perm_exists = True
            logger.info(f"Saved diagonal permutations: {diag_perm_path}")
    else:
        if perm_requested and not perm_should_run:
            logger.info("Permutation outputs present; skipping permutation resampling.")
# Save per-fold tuned hyperparameter summary (subject-level)
    try:
        if fold_hyper_values:
            hyper_df = pd.DataFrame({
                "fold": np.arange(1, len(fold_hyper_values) + 1, dtype=int),
                hyper_name: fold_hyper_values,
                "n_train_times_used": (fold_hyper_counts if len(fold_hyper_counts) == len(fold_hyper_values) else [np.nan] * len(fold_hyper_values)),
            })
            hyper_df["mode"] = mode
            hyper_df["target"] = target_col
            hyper_df["roi"] = roi or "all"
            hyper_df["subject"] = subject
            hyper_df["median"] = float(np.median(fold_hyper_values))
            hyper_df["mean"] = float(np.mean(fold_hyper_values))
            hyper_path = hyper_summary_path
            hyper_df.to_csv(hyper_path, sep="\t", index=False)
            logger.info(f"Saved nested-CV hyperparams: {hyper_path}")
    except Exception:
        pass

    # Plot heatmap and diagonal time course
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    if mode == "classification":
        plot_matrix = scores_mean - 0.5
        plot_cmap = "RdBu_r"
        plot_vmin, plot_vmax = -0.2, 0.2
        cbar_label = "AUC - 0.5"
    else:
        plot_matrix = scores_mean
        plot_cmap = cmap
        plot_vmin, plot_vmax = vmin, vmax
        cbar_label = "r(pred, y)"
    im = ax.imshow(
        plot_matrix,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[len(times[:n_te]) - 1], times[0], times[len(times[:n_tr]) - 1]],
        cmap=plot_cmap,
        vmin=plot_vmin,
        vmax=plot_vmax,
    )
    ax.set_xlabel("Test time (s)")
    ax.set_ylabel("Train time (s)")
    ax.set_title(f"Temporal Generalization — sub-{subject}\n{mode} on '{target_col}' (ROI: {roi or 'all'})")
    # Draw diagonal only across displayed extent
    ax.axline((times[0], times[0]), (times[n_te - 1], times[n_tr - 1]), color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    # Diagonal time course
    axes[1].plot(diag_times, diag, color="#1f77b4", lw=2)
    axes[1].axhline(0.5 if mode == "classification" else 0.0, color="k", ls=":", alpha=0.6)
    axes[1].axvline(0.0, color="k", ls="--", alpha=0.6)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Diagonal score")
    axes[1].set_title("Diagonal (train=test) decoding")
    plt.tight_layout()

    _ensure_dir(plot_path.parent)
    _save_fig(fig, plot_path)

    logger.info(f"Saved TGM plot: {plot_path.with_suffix('.png')}")
    logger.info(f"Saved TGM stats: {stats_path}")
    return stats_path


def _fisher_z(x: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(x, -0.999999, 0.999999))


def _fisher_inv(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _compute_diag_clusters(
    diag_obs: np.ndarray,
    null_mat: np.ndarray,
    times: np.ndarray,
    mode: str,
    baseline: float,
    two_sided: bool,
    cluster_alpha: float,
) -> Tuple[List[dict], np.ndarray]:
    """Compute 1D cluster-mass statistics for the temporal diagonal.

    Returns a list of cluster records and an array of cluster labels per time point
    (or -1 when not assigned to any suprathreshold cluster).
    """
    diag_obs = np.asarray(diag_obs, dtype=float)
    null_mat = np.asarray(null_mat, dtype=float)
    times = np.asarray(times, dtype=float)

    if null_mat.ndim != 2 or null_mat.shape[1] != diag_obs.size:
        return [], np.full(diag_obs.shape, -1, dtype=int)

    if mode == "regression":
        obs_stat = np.abs(_fisher_z(diag_obs))
        null_stat = np.abs(_fisher_z(null_mat))
        tail_desc = "two-sided"
    else:
        obs_stat = diag_obs - baseline
        null_stat = null_mat - baseline
        if two_sided:
            obs_stat = np.abs(obs_stat)
            null_stat = np.abs(null_stat)
            tail_desc = "two-sided"
        else:
            obs_stat = np.clip(obs_stat, a_min=0.0, a_max=None)
            null_stat = np.clip(null_stat, a_min=0.0, a_max=None)
            tail_desc = "upper"

    obs_stat = np.where(np.isfinite(obs_stat), obs_stat, 0.0)
    null_stat = np.where(np.isfinite(null_stat), null_stat, np.nan)
    flat_null = null_stat[np.isfinite(null_stat)].ravel()
    if flat_null.size == 0:
        return [], np.full(diag_obs.shape, -1, dtype=int)

    threshold = float(np.nanpercentile(flat_null, 100.0 * (1.0 - cluster_alpha)))
    if not np.isfinite(threshold) or threshold <= 0.0:
        return [], np.full(diag_obs.shape, -1, dtype=int)

    def _clusters_from_stat(stat: np.ndarray) -> List[Tuple[int, int]]:
        mask = stat >= threshold
        spans: List[Tuple[int, int]] = []
        start = None
        for idx, flag in enumerate(mask):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                spans.append((start, idx - 1))
                start = None
        if start is not None:
            spans.append((start, len(stat) - 1))
        return spans

    null_cluster_masses: List[float] = []
    for row in null_stat:
        row = np.where(np.isfinite(row), row, 0.0)
        spans = _clusters_from_stat(row)
        if spans:
            masses = [float(np.sum(row[s:e + 1])) for s, e in spans]
            null_cluster_masses.append(max(masses))
        else:
            null_cluster_masses.append(0.0)
    null_cluster_masses = np.asarray(null_cluster_masses, dtype=float)

    cluster_labels = np.full(diag_obs.shape, -1, dtype=int)
    clusters_records: List[dict] = []
    spans_obs = _clusters_from_stat(obs_stat)
    if spans_obs:
        for cid, (start, end) in enumerate(spans_obs):
            mass = float(np.sum(obs_stat[start:end + 1]))
            p_cluster = (1.0 + np.sum(null_cluster_masses >= mass)) / (null_cluster_masses.size + 1.0)
            cluster_labels[start:end + 1] = cid
            clusters_records.append({
                "cluster_id": int(cid),
                "t_start": float(times[start]),
                "t_end": float(times[end]),
                "n_timepoints": int(end - start + 1),
                "mass": mass,
                "p_value": float(p_cluster),
                "threshold": threshold,
                "tail": tail_desc,
            })

    return clusters_records, cluster_labels


def _compute_band_envelope_logpower(
    epochs: "mne.Epochs",
    l_freq: float,
    h_freq: float,
    baseline_window: Tuple[float, float],
    picks: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute band-limited analytic amplitude log-power with baseline correction.

    Returns (X, times) where X has shape (n_epochs, n_channels_selected, n_times).
    Baseline: 10*log10(power / mean_power_baseline) per epoch×channel.
    """
    ep = epochs.copy()
    if picks is not None:
        try:
            ep.pick_channels(picks)
        except Exception:
            pass
    data = ep.get_data(picks="eeg")  # (n_epochs, n_channels, n_times)
    n_ep, n_ch, n_t = data.shape
    times = ep.times.copy()
    # Filter per channel (vectorized across epochs*channels)
    X2d = data.reshape(n_ep * n_ch, n_t)
    try:
        Xf = mne.filter.filter_data(X2d, sfreq=ep.info["sfreq"], l_freq=l_freq, h_freq=h_freq, verbose=False)
    except Exception:
        Xf = X2d
    # Analytic amplitude
    Xa = np.abs(_hilbert(Xf, axis=-1))
    power = Xa ** 2
    # Baseline indices
    tmin_b, tmax_b = baseline_window
    sfreq = float(ep.info.get("sfreq", epochs.info.get("sfreq", 1.0)))
    if float(tmax_b) > 0.0:
        raise ValueError(f"Baseline window {baseline_window} must end <= 0 s for log-power normalization.")
    bmask = (times >= float(tmin_b)) & (times <= float(tmax_b))
    if not np.any(bmask):
        bmask = times < 0.0
    bmask_idx = np.where(bmask)[0]
    min_samples = max(int(np.ceil(MIN_BAND_BASELINE_SEC * sfreq)), 1)
    if bmask_idx.size < min_samples:
        raise ValueError(
            f"Baseline window {baseline_window} yields {bmask_idx.size} samples; require >= {min_samples} (>= {MIN_BAND_BASELINE_SEC:.3f}s)."
        )
    with np.errstate(divide='ignore', invalid='ignore'):
        bmean = np.nanmean(power[:, bmask_idx], axis=1)
    bmean = np.where(np.isfinite(bmean) & (bmean > 0.0), bmean, BAND_BASELINE_EPS)
    denom = np.maximum(bmean[:, None], BAND_BASELINE_EPS)
    with np.errstate(divide='ignore', invalid='ignore'):
        lp = 10.0 * np.log10(power / denom)
    X = lp.reshape(n_ep, n_ch, n_t)
    return X, times


def group_average_tgm(subjects: List[str], mode_filter: Optional[str] = None
) -> Optional[Path]:
    """Average per-subject TGMs and save group heatmap and TSV.

    Validity safeguard: requires that all included subject TGMs share the same
    mode, target column, and ROI. If multiple configurations are detected across
    subjects, abort with a warning to prevent mixing heterogeneous analyses.
    """
    logger = get_group_logger("temporal_generalization", "06_temporal_generalization.log")
    pivots: List[pd.DataFrame] = []
    times_union: Optional[np.ndarray] = None
    meta_seen = set()
    included_subs = []
    for sub in subjects:
        # Load per-subject TGM matrices (exclude diagonals/permutation TSVs)
        stats_dir = _stats_dir_subject(sub)
        all_cands = sorted(stats_dir.glob("time_generalization_*.tsv"))
        cands = [c for c in all_cands if "time_generalization_diag_" not in c.name and "_perm" not in c.name]
        if not cands:
            continue
        # Inspect candidates to detect ambiguity across target/roi
        cand_meta = []
        for c in cands:
            try:
                head = pd.read_csv(c, sep="\t", nrows=5)
            except Exception:
                continue
            needed = {"mode", "target", "roi", "t_train", "t_test"}
            if not needed.issubset(set(head.columns)):
                continue
            mode_u = list(pd.unique(head["mode"]))
            target_u = list(pd.unique(head["target"]))
            roi_u = list(pd.unique(head["roi"]))
            if len(mode_u) != 1 or len(target_u) != 1 or len(roi_u) != 1:
                continue
            cand_meta.append((mode_u[0], target_u[0], roi_u[0], c))
        if mode_filter is not None:
            cand_meta = [t for t in cand_meta if t[0] == mode_filter]
        if not cand_meta:
            continue
        # Require unambiguous target/roi per subject
        uniq_tr = sorted({(t[1], t[2]) for t in cand_meta})
        if len(uniq_tr) != 1:
            logger.warning(f"Skipping sub-{sub}: multiple TGM configs detected for subject (target/roi combos: {uniq_tr}); pass explicit filters to avoid mixing")
            continue
        # Read the selected file fully
        df_path = cand_meta[0][3]
        df = pd.read_csv(df_path, sep="\t")
        # Validate uniform configuration
        mode_u = list(pd.unique(df["mode"]))
        target_u = list(pd.unique(df["target"]))
        roi_u = list(pd.unique(df["roi"]))
        if len(mode_u) != 1 or len(target_u) != 1 or len(roi_u) != 1:
            logger.warning(f"Skipping sub-{sub}: non-unique metadata in TGM file")
            continue
        meta_seen.add((mode_u[0], target_u[0], roi_u[0]))
        # Round time keys to avoid float-bin mismatches across subjects
        df = df.copy()
        df["t_train"] = np.round(df["t_train"].astype(float), 6)
        df["t_test"] = np.round(df["t_test"].astype(float), 6)
        piv = df.pivot_table(index="t_train", columns="t_test", values="score", aggfunc="mean")
        # Build a shared time vector; pad each subject to it with NaN
        tr_times = piv.index.to_numpy()
        te_times = piv.columns.to_numpy()
        subj_times = np.union1d(tr_times, te_times)
        if times_union is None:
            times_union = subj_times
        else:
            times_union = np.union1d(times_union, subj_times)
        pivots.append(piv)
        logger.info(
            f"Group avg include sub-{sub}: mode={mode_u[0]}, target={target_u[0]}, roi={roi_u[0]}, n_times={len(subj_times)}"
        )
    if not pivots or times_union is None:
        logger.warning("No subject TGMs found for group averaging")
        return None
    if len(meta_seen) > 1:
        logger.warning(f"Heterogeneous TGMs across subjects (mode/target/roi combos: {sorted(meta_seen)}); aborting group average to avoid invalid mixing")
        return None
    # Determine effective mode for plotting/scales
    mode_eff = mode_filter
    if mode_eff is None and len(meta_seen) == 1:
        mode_eff = list(meta_seen)[0][0]

    # Domain conversion: if regression, convert subject matrices to Fisher-z, then average and invert
    # Reindex all subject TGMs to the union time grid and stack
    mats = []
    for piv in pivots:
        aligned = piv.reindex(index=times_union, columns=times_union)
        mats.append(aligned.to_numpy())
    if mode_eff == "regression":
        tgms_z = [_fisher_z(M) for M in mats]
        stack = np.stack(tgms_z, axis=0)
        mean_mat = np.nanmean(stack, axis=0)
        G = _fisher_inv(mean_mat)
    else:
        stack = np.stack(mats, axis=0)
        mean_mat = np.nanmean(stack, axis=0)
        G = mean_mat

    # Log summary of group averaging configuration
    try:
        if len(meta_seen) == 1:
            m, t, r = next(iter(meta_seen))
        else:
            m, t, r = (mode_eff or "classification"), "various", "various"
        logger.info(
            f"Group TGM averaging: mode={m}, target={t}, roi={r}, subjects_used={stack.shape[0]}, time_points={G.shape[0]}"
        )
    except Exception:
        pass

    plots_dir = _plots_dir_group()
    stats_dir = _stats_dir_group()
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)

    # Determine uniform target/roi for provenance (or mark various)
    if len(meta_seen) == 1:
        target_val = list(meta_seen)[0][1]
        roi_val = list(meta_seen)[0][2]
    else:
        target_val = "various"
        roi_val = "various"

    # Save TSV
    n = G.shape[0]
    dfG = pd.DataFrame({
        "t_train": np.repeat(times_union[:n], n),
        "t_test": np.tile(times_union[:n], n),
        "score": G.flatten(),
        "mode": (mode_eff or "classification"),
        "target": target_val,
        "roi": roi_val,
    })
    out_tsv = stats_dir / f"group_time_generalization_{mode_filter or 'any'}.tsv"
    dfG.to_csv(out_tsv, sep="\t", index=False)
    # Group TGM provenance JSON
    try:
        import json
        provG = {
            "mode": (mode_eff or "classification"),
            "target": target_val,
            "roi": roi_val,
            "subjects_used": int(stack.shape[0]),
            "time_points": int(G.shape[0]),
            "note": "Only diagonal tested; heatmap descriptive. Regression averaged in Fisher-z then tanh back.",
        }
        with open(out_tsv.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(provG, f, indent=2)
    except Exception:
        pass

    # Plot
    if mode_eff == "classification":
        # Visualize AUC - 0.5 with a symmetric diverging scale to reveal below-chance structure
        G_vis = G - 0.5
        vmin, vmax, cmap = (-0.2, 0.2, "RdBu_r")
        cbar_label = "AUC - 0.5"
        plot_matrix = G_vis
    else:
        vmin, vmax, cmap = (-0.6, 0.6, "RdBu_r")
        cbar_label = "Score (r)"
        plot_matrix = G
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(
        plot_matrix,
        origin="lower",
        aspect="auto",
        extent=[times_union[0], times_union[len(times_union) - 1], times_union[0], times_union[len(times_union) - 1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.axline((times_union[0], times_union[0]), (times_union[-1], times_union[-1]), color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Test time (s)")
    ax.set_ylabel("Train time (s)")
    ax.set_title(f"Group Temporal Generalization (N={stack.shape[0]})\n{mode_eff or 'classification'} on '{target_val}' (ROI: {roi_val})")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    plt.tight_layout()
    out_fig = plots_dir / f"group_time_generalization_{mode_filter or 'any'}"
    _save_fig(fig, out_fig)
    logger.info(f"Saved group TGM: {out_fig.with_suffix('.png')} | {out_tsv}")
    
    # Group diagonal summary with CI
    if (mode_eff or "classification") == "regression":
        # stack is z-domain; compute CI in z, then back-transform for saving/plotting
        diag_z_vals = np.asarray([np.diag(M) for M in stack])
        mean_z = np.nanmean(diag_z_vals, axis=0)
        sd_z = np.nanstd(diag_z_vals, axis=0, ddof=1)
        n = np.sum(np.isfinite(diag_z_vals), axis=0)
        sem_z = sd_z / np.sqrt(np.maximum(1, n))
        ci_low_z = mean_z - 1.96 * sem_z
        ci_high_z = mean_z + 1.96 * sem_z
        # Back-transform
        mean_diag = _fisher_inv(mean_z)
        ci_low = _fisher_inv(ci_low_z)
        ci_high = _fisher_inv(ci_high_z)
        # Also report r-domain dispersion for reference
        diag_r_vals = np.asarray([np.diag(_fisher_inv(M)) for M in stack])
        sd_diag = np.nanstd(diag_r_vals, axis=0, ddof=1)
        sem_diag = sd_diag / np.sqrt(np.maximum(1, n))
    else:
        diag_vals = np.asarray([np.diag(M) for M in stack])
        mean_diag = np.nanmean(diag_vals, axis=0)
        sd_diag = np.nanstd(diag_vals, axis=0, ddof=1)
        n = np.sum(np.isfinite(diag_vals), axis=0)
        sem_diag = sd_diag / np.sqrt(np.maximum(1, n))
        ci_low = mean_diag - 1.96 * sem_diag
        ci_high = mean_diag + 1.96 * sem_diag

    diag_df = pd.DataFrame({
        "time": times_union[: len(mean_diag)],
        "mean": mean_diag,
        "sd": sd_diag,
        "sem": sem_diag,
        "n": n,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "mode": (mode_eff or "classification"),
        "target": target_val,
        "roi": roi_val,
    })
    out_diag_tsv = stats_dir / f"group_time_generalization_diag_{mode_filter or 'any'}.tsv"
    diag_df.to_csv(out_diag_tsv, sep="\t", index=False)

    # Plot group diagonal
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.0))
    ax2.plot(times_union[: len(mean_diag)], mean_diag, color="#1f77b4", lw=2, label="Group mean")
    ax2.fill_between(times_union[: len(mean_diag)], ci_low, ci_high, color="#1f77b4", alpha=0.2, label="95% CI")
    ax2.axvline(0.0, color="k", ls="--", alpha=0.7)
    ax2.axhline(0.5 if (mode_eff or "classification") == "classification" else 0.0, color="k", ls=":", alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Diagonal score")
    ax2.set_title(f"Group Diagonal — Temporal Generalization (N={stack.shape[0]})")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    plt.tight_layout()
    out_diag_fig = plots_dir / f"group_time_generalization_diag_{mode_filter or 'any'}"
    _save_fig(fig2, out_diag_fig)

    logger.info(f"Saved group diag: {out_diag_fig.with_suffix('.png')} | {out_diag_tsv}")
    return out_tsv


def group_diag_permutation(subjects: List[str], mode_filter: Optional[str] = None, n_samples: int = 5000, seed: int = 42
) -> Optional[Path]:
    """Approximate group-level empirical p-values for diagonal curve.

    Requires per-subject diag permutation TSVs containing null_mean and null_sd.
    Approximates each subject's null as Normal(null_mean, null_sd) per time and
    Monte Carlo samples the group null of the mean across subjects.
    """
    logger = get_group_logger("temporal_generalization", "06_temporal_generalization.log")
    rng = np.random.default_rng(int(seed))
    # Load observed group diag and per-subject null summaries
    obs_group_path = _stats_dir_group() / f"group_time_generalization_diag_{mode_filter or 'any'}.tsv"
    if not obs_group_path.exists():
        logger.warning("Group diagonal TSV not found; run group averaging first")
        return None
    obs_df = pd.read_csv(obs_group_path, sep="\t")
    times_ref = np.round(obs_df["time"].to_numpy().astype(float), 6)
    obs = obs_df["mean"].to_numpy()
    # Determine mode for correct testing if present in file; fallback to arg
    if "mode" in obs_df.columns:
        mode_eff = str(pd.unique(obs_df["mode"])[0])
    else:
        mode_eff = mode_filter or "classification"

    mu_z_list = []     # regression: z-domain (from file or transformed)
    sd_z_list = []
    mu_r_list = []     # classification: r-domain (AUC)
    sd_r_list = []
    actual_z_list: List[np.ndarray] = []
    actual_r_list: List[np.ndarray] = []
    kept = []
    for s in subjects:
        # find perm TSV matching mode; avoid ambiguity across target/roi
        cand_paths = sorted(_stats_dir_subject(s).glob(f"time_generalization_diag_{mode_filter or '*'}_*_perm.tsv"))
        if not cand_paths:
            continue
        cand_meta = []
        for c in cand_paths:
            try:
                head = pd.read_csv(c, sep="\t", nrows=5)
            except Exception:
                continue
            if not {"time", "mode"}.issubset(set(head.columns)):
                continue
            mode_u = list(pd.unique(head["mode"]))
            target_u = list(pd.unique(head.get("target", pd.Series(["unknown"])) ))
            roi_u = list(pd.unique(head.get("roi", pd.Series(["all"])) ))
            if len(mode_u) != 1 or len(target_u) != 1 or len(roi_u) != 1:
                continue
            cand_meta.append((mode_u[0], target_u[0], roi_u[0], c))
        if mode_filter is not None:
            cand_meta = [t for t in cand_meta if t[0] == mode_filter]
        if not cand_meta:
            continue
        uniq_tr = sorted({(t[1], t[2]) for t in cand_meta})
        if len(uniq_tr) != 1:
            logger.warning(f"Skipping sub-{s}: multiple diag permutation configs detected (target/roi combos: {uniq_tr})")
            continue
        df_path = cand_meta[0][3]
        df = pd.read_csv(df_path, sep="\t")
        t = np.round(df["time"].to_numpy().astype(float), 6)
        if len(t) != len(times_ref) or not np.allclose(t, times_ref, atol=1e-6):
            logger.warning(f"Skipping sub-{s}: time axis mismatch for group permutation")
            continue
        samples_path = df_path.with_name(df_path.name.replace("_perm.tsv", "_perm_samples.npz"))
        if samples_path.exists():
            try:
                samples_data = np.load(samples_path)
                arr = np.asarray(samples_data["null_diag"], dtype=float)
                t_samples = np.round(samples_data["time"].astype(float), 6)
                if arr.ndim == 2 and arr.shape[1] == len(times_ref) and np.allclose(t_samples, times_ref, atol=1e-6):
                    if (mode_filter or mode_eff) == "regression":
                        actual_z_list.append(_fisher_z(arr))
                    else:
                        actual_r_list.append(arr)
                    per_subject_P.append(arr.shape[0])
                    kept.append(s)
                    continue
                else:
                    logger.warning(f"sub-{s}: permutation samples time axis mismatch for group permutation; using summary stats instead")
            except Exception as exc:
                logger.warning(f"sub-{s}: failed loading permutation samples ({exc}); using summary stats instead")
        if (mode_filter or mode_eff) == "regression":
            # Require z-domain; if missing, transform r -> z via delta method
            if "null_mean_z" in df.columns and "null_sd_z" in df.columns:
                mu_z_list.append(df["null_mean_z"].to_numpy())
                sd_z_list.append(np.maximum(df["null_sd_z"].to_numpy(), 1e-9))
            elif "null_mean" in df.columns and "null_sd" in df.columns:
                mu_r = df["null_mean"].to_numpy()
                sd_r = np.maximum(df["null_sd"].to_numpy(), 1e-9)
                mu_z = _fisher_z(mu_r)
                # delta: var(z) ≈ var(r) / (1 - mu_r^2)^2
                denom = np.clip(1.0 - np.square(mu_r), 1e-6, None)
                sd_z = sd_r / denom
                mu_z_list.append(mu_z)
                sd_z_list.append(sd_z)
            else:
                logger.warning(f"Subject {s}: missing null summary columns for regression; skipping")
                continue
        else:
            # classification uses r-domain/AUC directly
            if "null_mean" in df.columns and "null_sd" in df.columns:
                mu_r_list.append(df["null_mean"].to_numpy())
                sd_r_list.append(np.maximum(df["null_sd"].to_numpy(), 1e-9))
            else:
                logger.warning(f"Subject {s}: missing null summary columns for classification; skipping")
                continue
        kept.append(s)

    total_reg = len(mu_z_list) + len(actual_z_list)
    total_clf = len(mu_r_list) + len(actual_r_list)
    if ((mode_filter or mode_eff) == "regression" and total_reg == 0) or ((mode_filter or mode_eff) != "regression" and total_clf == 0):
        logger.warning("No subject permutation summaries found; skipping group permutation")
        return None

    if (mode_filter or mode_eff) == "regression":
        mu = np.stack(mu_z_list, axis=0) if mu_z_list else np.empty((0, len(times_ref)), dtype=float)
        sd = np.stack(sd_z_list, axis=0) if sd_z_list else np.empty((0, len(times_ref)), dtype=float)
        domain = "Fisher-z"
        actual_list = actual_z_list
    else:
        mu = np.stack(mu_r_list, axis=0) if mu_r_list else np.empty((0, len(times_ref)), dtype=float)
        sd = np.stack(sd_r_list, axis=0) if sd_r_list else np.empty((0, len(times_ref)), dtype=float)
        domain = "AUC"
        actual_list = actual_r_list

    n_approx = mu.shape[0]
    n_time = mu.shape[1] if n_approx > 0 else (actual_list[0].shape[1] if actual_list else len(times_ref))
    n_actual = len(actual_list)
    n_subj = n_approx + n_actual

    try:
        logger.info(
            f"Group diag perm (approx): kept={n_subj}/{len(subjects)} subjects; domain={domain}; MC_samples={int(n_samples)}; T={n_time}; actual={n_actual}; approx={n_approx}"
        )
    except Exception:
        pass

    mu_arr = mu
    sd_arr = sd
    p_emp = np.ones(n_time, dtype=float)
    n_samples_int = int(n_samples)
    for i in range(n_time):
        draw_blocks = []
        if n_approx > 0:
            approx_draws = rng.normal(loc=mu_arr[:, i], scale=sd_arr[:, i], size=(n_samples_int, n_approx))
            draw_blocks.append(approx_draws)
        if n_actual > 0:
            actual_draws = []
            for arr in actual_list:
                idx = rng.integers(arr.shape[0], size=n_samples_int)
                actual_draws.append(arr[idx, i])
            if actual_draws:
                draw_blocks.append(np.stack(actual_draws, axis=1))
        if draw_blocks:
            subj_mat = np.concatenate(draw_blocks, axis=1) if len(draw_blocks) > 1 else draw_blocks[0]
            finite_mask = np.isfinite(subj_mat)
            counts = finite_mask.sum(axis=1)
            counts = np.maximum(counts, 1)
            sums = np.nan_to_num(subj_mat, nan=0.0, posinf=0.0, neginf=0.0).sum(axis=1)
            group_null = sums / counts
        else:
            group_null = np.zeros(n_samples_int, dtype=float)

        if mode_eff == "regression":
            obs_eff = np.abs(_fisher_z(obs[i]))
            mixes_eff = np.abs(group_null)
            p_emp[i] = (1.0 + np.sum(mixes_eff >= obs_eff)) / (len(group_null) + 1.0)
        else:
            p_emp[i] = (1.0 + np.sum(group_null >= obs[i])) / (len(group_null) + 1.0)

    q_emp = _fdr_bh(p_emp, alpha=0.05)
    out_df = obs_df.copy()
    out_df["p_emp"] = p_emp
    out_df["q_fdr"] = q_emp
    out_df["n_subjects_perm"] = n_subj
    out_df["n_subjects_actual"] = n_actual
    out_df["n_subjects_approx"] = n_approx
    out_path = _stats_dir_group() / f"group_time_generalization_diag_{mode_filter or 'any'}_perm.tsv"
    method_tag = "hybrid" if n_actual > 0 else "approx"
    approx_desc = ("Exact subject samples where available; Normal(mu, sd) otherwise" if n_actual > 0 else "Normal(mu, sd) per subject")
    out_df["method"] = method_tag
    out_df["approximation"] = approx_desc
    out_df.to_csv(out_path, sep="\t", index=False)
    # Provenance
    try:
        import json
        prov = {
            "method": method_tag,
            "domain": ("Fisher-z" if mode_eff == "regression" else "AUC"),
            "n_samples": int(n_samples),
            "n_subjects": int(n_subj),
            "n_subjects_actual": int(n_actual),
            "n_subjects_approx": int(n_approx),
            "approximation": approx_desc,
        }
        with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(prov, f, indent=2)
    except Exception:
        pass

    # Plot with significance dots
    sig_mask = q_emp < 0.05
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(times_ref, obs, color="#1f77b4", lw=2, label="Group mean")
    ax.fill_between(times_ref, out_df["ci95_low"], out_df["ci95_high"], color="#1f77b4", alpha=0.2)
    if np.any(sig_mask):
        ymin = np.nanmin(obs) if (mode_eff == "regression") else 0.5
        ax.scatter(times_ref[sig_mask], np.full(sig_mask.sum(), ymin) - 0.02, s=10, color="crimson", marker="o", label="q < .05")
    ax.axvline(0.0, color="k", ls="--", alpha=0.7)
    ax.axhline(0.5 if (mode_eff or "classification") == "classification" else 0.0, color="k", ls=":", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Diagonal score")
    ax.set_title(f"Group Diagonal — Empirical p-values (N={n_subj})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    out_fig = _plots_dir_group() / f"group_time_generalization_diag_{mode_filter or 'any'}_perm"
    _save_fig(fig, out_fig)
    logger.info(f"Saved group diag permutation: {out_fig.with_suffix('.png')} | {out_path}")
    return out_path


def group_diag_permutation_exact(subjects: List[str], mode_filter: Optional[str] = None, n_samples: int = 10000, seed: int = 42
) -> Optional[Path]:
    """Exact group-level permutation for diagonal using saved subject permutation samples.

    Requires each subject to have saved `*_perm_samples.npz` (null_diag x time) files with
    matching time axes. Monte Carlo samples are drawn by selecting a random null diag per
    subject and averaging across subjects, yielding a group-null distribution per time.
    Regression uses two-sided tests in Fisher-z domain; classification uses one-sided AUC.
    """
    logger = get_group_logger("temporal_generalization", "06_temporal_generalization.log")
    rng = np.random.default_rng(int(seed))

    # Load observed group diag
    obs_group_path = _stats_dir_group() / f"group_time_generalization_diag_{mode_filter or 'any'}.tsv"
    if not obs_group_path.exists():
        logger.warning("Group diagonal TSV not found; run group averaging first")
        return None
    obs_df = pd.read_csv(obs_group_path, sep="\t")
    times_ref = obs_df["time"].to_numpy()
    obs = obs_df["mean"].to_numpy()
    if "mode" in obs_df.columns:
        mode_eff = str(pd.unique(obs_df["mode"])[0])
    else:
        mode_eff = mode_filter or "classification"

    # Load subject permutation samples
    subj_nulls = []  # list of arrays (P_i, T)
    per_subject_P = []
    for s in subjects:
        # Grab a perm_samples npz for any matching mode
        cands = sorted(_stats_dir_subject(s).glob(f"time_generalization_diag_{mode_filter or '*'}_*_perm_samples.npz"))
        if not cands:
            logger.warning(f"sub-{s}: no saved perm_samples; skipping")
            continue
        try:
            d = np.load(cands[0])
            arr = d["null_diag"]
            t = d["time"]
            if arr.ndim != 2:
                continue
            if len(t) != len(times_ref) or not np.allclose(t, times_ref, atol=1e-6):
                logger.warning(f"sub-{s}: time axis mismatch in perm_samples; skipping")
                continue
            subj_nulls.append(arr)
            per_subject_P.append(arr.shape[0])
        except Exception:
            continue
    if not subj_nulls:
        logger.warning("No subject permutation samples found; cannot run exact group permutation")
        return None

    n_time = len(times_ref)
    n_subj = len(subj_nulls)
    # Log exact perm summary
    try:
        medP = int(np.median(per_subject_P)) if per_subject_P else 0
        logger.info(f"Group diag perm (exact): kept={n_subj}/{len(subjects)} subjects; per-subject permutations median={medP}; MC_samples={int(n_samples)}; T={n_time}")
    except Exception:
        pass
    # Build group null via Monte Carlo sampling across subjects
    p_emp = np.ones(n_time, dtype=float)
    for i in range(n_time):
        mixes = np.empty(int(n_samples), dtype=float)
        for k in range(int(n_samples)):
            vals = [subj_nulls[s][rng.integers(0, subj_nulls[s].shape[0]), i] for s in range(n_subj)]
            mixes[k] = float(np.mean(vals))
        if mode_eff == "regression":
            obs_eff = np.abs(_fisher_z(obs[i]))
            mixes_eff = np.abs(_fisher_z(mixes))
            p_emp[i] = (1.0 + np.sum(mixes_eff >= obs_eff)) / (len(mixes_eff) + 1.0)
        else:
            p_emp[i] = (1.0 + np.sum(mixes >= obs[i])) / (len(mixes) + 1.0)
    q_emp = _fdr_bh(p_emp, alpha=0.05)

    out_df = obs_df.copy()
    out_df["p_emp_exact"] = p_emp
    out_df["q_fdr_exact"] = q_emp
    out_df["n_subjects_perm"] = n_subj
    out_path = _stats_dir_group() / f"group_time_generalization_diag_{mode_filter or 'any'}_perm_exact.tsv"
    out_df["method"] = "exact"
    out_df.to_csv(out_path, sep="\t", index=False)
    try:
        import json
        prov = {
            "method": "exact",
            "n_samples": int(n_samples),
            "n_subjects": int(n_subj),
        }
        with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(prov, f, indent=2)
    except Exception:
        pass

    # Plot
    sig_mask = q_emp < 0.05
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(times_ref, obs, color="#1f77b4", lw=2, label="Group mean")
    ax.fill_between(times_ref, out_df.get("ci95_low", obs*0), out_df.get("ci95_high", obs*0), color="#1f77b4", alpha=0.2)
    if np.any(sig_mask):
        ymin = np.nanmin(obs) if (mode_eff == "regression") else 0.5
        ax.scatter(times_ref[sig_mask], np.full(sig_mask.sum(), ymin) - 0.02, s=10, color="crimson", marker="o", label="q < .05 (exact)")
    ax.axvline(0.0, color="k", ls="--", alpha=0.7)
    ax.axhline(0.5 if (mode_eff or "classification") == "classification" else 0.0, color="k", ls=":", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Diagonal score")
    ax.set_title(f"Group Diagonal — Exact Permutation (N={n_subj})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    out_fig = _plots_dir_group() / f"group_time_generalization_diag_{mode_filter or 'any'}_perm_exact"
    _save_fig(fig, out_fig)
    logger.info(f"Saved exact group diag permutation: {out_fig.with_suffix('.png')} | {out_path}")
    return out_path


def _interp_to_union_times(times_src: np.ndarray, values: np.ndarray, times_union: np.ndarray) -> np.ndarray:
    times_src = np.asarray(times_src, float)
    values = np.asarray(values, float)
    times_union = np.asarray(times_union, float)
    out = np.full_like(times_union, np.nan, dtype=float)
    ok = np.isfinite(values)
    if np.sum(ok) < 2:
        return out
    x = times_src[ok]
    y = values[ok]
    lo, hi = np.nanmin(x), np.nanmax(x)
    sel = (times_union >= lo) & (times_union <= hi)
    if np.any(sel):
        out[sel] = np.interp(times_union[sel], x, y)
    return out


def group_roi_diag_cluster_permutation(
    subjects: List[str],
    mode_filter: Optional[str] = None,
    target_filter: Optional[str] = None,
    n_perm: int = 5000,
    cluster_p: float = 0.05,
    seed: int = 42,

) -> Optional[Path]:
    logger = get_group_logger("temporal_generalization", "06_temporal_generalization.log")
    rng = np.random.default_rng(int(seed))

    subj_roi_paths = {}
    roi_set = set()
    mode_eff: Optional[str] = None
    target_eff: Optional[str] = None
    for s in subjects:
        paths = sorted(_stats_dir_subject(s).glob("time_generalization_diag_*.tsv"))
        if not paths:
            continue
        per_roi = {}
        for pth in paths:
            try:
                head = pd.read_csv(pth, sep="\t", nrows=5)
            except Exception:
                continue
            cols = set(head.columns)
            if not {"mode", "target", "roi", "time", "score"}.issubset(cols):
                continue
            mode_u = list(pd.unique(head["mode"]))
            target_u = list(pd.unique(head["target"]))
            roi_u = list(pd.unique(head["roi"]))
            if len(mode_u) != 1 or len(target_u) != 1 or len(roi_u) != 1:
                continue
            mode_v = str(mode_u[0])
            target_v = str(target_u[0])
            roi_v = str(roi_u[0])
            if mode_filter is not None and mode_v != mode_filter:
                continue
            if target_filter is not None and target_v != target_filter:
                continue
            per_roi[roi_v] = pth
            roi_set.add(roi_v)
            if mode_eff is None:
                mode_eff = mode_v
            if target_eff is None:
                target_eff = target_v
        if per_roi:
            subj_roi_paths[s] = per_roi
    if not subj_roi_paths or not roi_set:
        logger.warning("No subject ROI diag files found; ensure subject TGMs have been computed and diag TSVs exist.")
        return None
    rois = sorted(list(roi_set))

    effects_by_roi = {}
    times_union_by_roi = {}
    for roi in rois:
        subj_series = []
        times_union = None
        for s, mapping in subj_roi_paths.items():
            if roi not in mapping:
                continue
            try:
                df = pd.read_csv(mapping[roi], sep="\t")
            except Exception:
                continue
            t = np.round(df["time"].astype(float).to_numpy(), 6)
            y = df["score"].astype(float).to_numpy()
            if (mode_eff or "classification") == "classification":
                val = y - 0.5
            else:
                val = np.arctanh(np.clip(y, -0.999999, 0.999999))
            subj_series.append((t, val))
            times_union = np.union1d(times_union, t) if times_union is not None else t
        if not subj_series:
            continue
        mat = []
        for t, v in subj_series:
            mat.append(_interp_to_union_times(t, v, times_union))
        effects_by_roi[roi] = np.vstack(mat)
        times_union_by_roi[roi] = times_union

    effects_rows = []
    mask_rows = []
    clusters_records = []
    all_times_union = None
    for roi in rois:
        if roi not in effects_by_roi:
            continue
        X = effects_by_roi[roi]
        times = times_union_by_roi[roi]
        n_subj = X.shape[0]
        means = np.nanmean(X, axis=0)
        sds = np.nanstd(X, axis=0, ddof=1)
        counts = np.sum(np.isfinite(X), axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_vals = means / (sds / np.sqrt(np.maximum(counts, 1)))
        df_eff = int(np.nanmin(counts) - 1) if np.nanmin(counts) >= 1 else 0
        if df_eff <= 0:
            t_thr = np.inf
        else:
            t_thr = float(_student_t.ppf(1.0 - cluster_p / 2.0, df=df_eff))
        supra = np.where(np.abs(t_vals) >= t_thr, 1, 0).astype(int)
        def _find_clusters(mask: np.ndarray):
            spans = []
            i = 0
            while i < len(mask):
                if mask[i]:
                    j = i
                    while j + 1 < len(mask) and mask[j + 1]:
                        j += 1
                    spans.append((i, j))
                    i = j + 1
                else:
                    i += 1
            return spans
        obs_spans = _find_clusters(supra)
        obs_masses = [float(np.nansum(np.abs(t_vals[s:e + 1]))) for (s, e) in obs_spans]
        max_masses = np.zeros(int(n_perm), dtype=float)
        for pi in range(int(n_perm)):
            signs = rng.choice([-1.0, 1.0], size=(n_subj, 1))
            Xp = X * signs
            mp = np.nanmean(Xp, axis=0)
            sdp = np.nanstd(Xp, axis=0, ddof=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                tp = mp / (sdp / np.sqrt(np.maximum(counts, 1)))
            mp_sup = np.where(np.abs(tp) >= t_thr, 1, 0).astype(int)
            spans_p = _find_clusters(mp_sup)
            if spans_p:
                masses_p = [float(np.nansum(np.abs(tp[s:e + 1]))) for (s, e) in spans_p]
                max_masses[pi] = np.max(masses_p)
            else:
                max_masses[pi] = 0.0
        sig_mask = np.zeros_like(supra, dtype=bool)
        for (s0, e0), mass in zip(obs_spans, obs_masses):
            p_fwe = (1.0 + np.sum(max_masses >= mass)) / (len(max_masses) + 1.0)
            clusters_records.append({
                "roi": roi,
                "t_start": float(times[s0]),
                "t_end": float(times[e0]),
                "mass": float(mass),
                "p_fwe": float(p_fwe),
                "n_subjects": int(n_subj),
                "df_eff": int(df_eff),
                "t_threshold": float(t_thr),
            })
            if p_fwe <= 0.05:
                sig_mask[s0:e0 + 1] = True
        effects_rows.append((roi, times, means))
        mask_rows.append((roi, times, sig_mask))
        all_times_union = np.union1d(all_times_union, times) if all_times_union is not None else times

    if all_times_union is None:
        logger.warning("No ROI effects to aggregate; aborting.")
        return None
    times_out = all_times_union
    rois_out = [r for r, _, _ in effects_rows]
    E = np.full((len(rois_out), len(times_out)), np.nan, dtype=float)
    M = np.zeros((len(rois_out), len(times_out)), dtype=bool)
    for i, (roi, t, v) in enumerate(effects_rows):
        E[i, :] = _interp_to_union_times(t, v, times_out)
    for i, (roi, t, msk) in enumerate(mask_rows):
        pos = np.isin(times_out, t)
        M[i, pos] = msk

    stats_dir = _stats_dir_group()
    plots_dir = _plots_dir_group()
    _ensure_dir(stats_dir)
    _ensure_dir(plots_dir)
    mode_str = (mode_filter or (mode_eff or "any"))
    target_str = (target_filter or (target_eff or "any"))
    eff_tsv = stats_dir / f"group_roi_diag_effects_{mode_str}_{target_str}.tsv"
    mask_tsv = stats_dir / f"group_roi_diag_mask_{mode_str}_{target_str}.tsv"
    eff_df = pd.DataFrame(E, index=rois_out, columns=[f"t_{t:.6f}" for t in times_out])
    eff_df.insert(0, "roi", rois_out)
    eff_df.to_csv(eff_tsv, sep="\t", index=False)
    mask_df = pd.DataFrame(M.astype(int), index=rois_out, columns=[f"t_{t:.6f}" for t in times_out])
    mask_df.insert(0, "roi", rois_out)
    mask_df.to_csv(mask_tsv, sep="\t", index=False)
    clus_tsv = stats_dir / f"group_roi_diag_clusters_{mode_str}_{target_str}.tsv"
    if clusters_records:
        pd.DataFrame(clusters_records).to_csv(clus_tsv, sep="\t", index=False)
    else:
        pd.DataFrame(columns=["roi", "t_start", "t_end", "mass", "p_fwe", "n_subjects", "df_eff", "t_threshold"]).to_csv(clus_tsv, sep="\t", index=False)
    try:
        import json
        params = {
            "n_perm": int(n_perm),
            "cluster_p": float(cluster_p),
            "tail": "two-sided",
            "times_round_decimals": 6,
            "n_rois": len(rois_out),
            "n_times": int(len(times_out)),
            "mode": mode_str,
            "target": target_str,
            "method": "sign-flip cluster permutation",
            "adjacency": "temporal (1D)",
        }
        with open(stats_dir / f"group_roi_diag_cluster_params_{mode_str}_{target_str}.json", "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
    except Exception:
        pass

    vmax = np.nanmax(np.abs(E))
    vmax = float(vmax if np.isfinite(vmax) and vmax > 0 else 0.2)
    vmin = -vmax
    fig, ax = plt.subplots(figsize=(10.0, max(3.0, 0.35 * len(rois_out))))
    im = ax.imshow(
        E,
        aspect="auto",
        origin="lower",
        extent=[times_out[0], times_out[-1], -0.5, len(rois_out) - 0.5],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_yticks(range(len(rois_out)))
    ax.set_yticklabels(rois_out)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROI")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("AUC-0.5" if (mode_eff or "classification") == "classification" else "z(r)")
    for i, (roi, t, msk) in enumerate(mask_rows):
        if roi not in rois_out:
            continue
        def segs(b):
            spans = []
            j = 0
            while j < len(b):
                if b[j]:
                    k = j
                    while k + 1 < len(b) and b[k + 1]:
                        k += 1
                    spans.append((j, k))
                    j = k + 1
                else:
                    j += 1
            return spans
        spans = segs(msk)
        y = i - 0.48
        for s0, e0 in spans:
            ax.hlines(y, t[s0], t[e0], colors="k", linewidth=3)
    ax.set_title(f"Group ROI diagonal — {mode_str} on '{target_str}'")
    plt.tight_layout()
    fig_path = _plots_dir_group() / f"group_roi_diag_clusters_{mode_str}_{target_str}"
    _save_fig(fig, fig_path)
    logger.info(f"Saved group ROI diag: {fig_path.with_suffix('.png')} | {eff_tsv} | {mask_tsv} | {clus_tsv}")
    return eff_tsv


def _run_subject_bands(sub_id: str, args, bands_cfg: dict, baseline_window: Tuple[float, float]) -> None:
    """Compute per-subject band-limited diagonal decoding for a given ROI across configured bands."""
    logger = get_subject_logger("temporal_generalization", sub_id, "06_temporal_generalization.log")
    plots_dir = _plots_dir_subject(sub_id)
    _ensure_dir(plots_dir)
    epo_path = _find_clean_epochs_path(sub_id, args.task)
    if epo_path is None or not Path(epo_path).exists():
        logger.warning(f"No cleaned epochs found for sub-{sub_id}, task-{args.task}")
        return
    try:
        epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
    except Exception as e:
        logger.warning(f"Failed loading epochs: {e}")
        return
    # ROI channels
    roi_chs = None
    if args.roi is not None:
        try:
            roi_map = _build_rois(epochs.info)
            if args.roi in roi_map:
                roi_chs = roi_map[args.roi]
        except Exception:
            pass
    # Align events
    events = _load_events_df(sub_id, args.task)
    events_aligned = _align_events_to_epochs(events, epochs)
    if events_aligned is None:
        logger.warning("Could not align events to epochs")
        return
    try:
        target_col, mode = _pick_target(events_aligned, args.target)
    except ValueError as e:
        logger.warning(str(e))
        return
    y = pd.to_numeric(events_aligned[target_col], errors="coerce")
    mask = ~y.isna()
    epochs = epochs[mask.to_numpy()]
    y = y.loc[mask].to_numpy()
    if mode == "classification":
        try:
            y = _binarize_labels(y.astype(float))
        except Exception as e:
            logger.warning(f"Binarization failed: {e}")
            return
        if np.unique(y).size < 2:
            logger.warning("Classification target has a single class after cleaning")
            return
        scoring = "roc_auc"
    else:
        if np.isnan(y).all() or np.std(y[~np.isnan(y)]) <= 1e-12:
            logger.warning("Regression target has insufficient variance")
            return
        scoring = _regression_scoring
    # Build outer CV
    cv, groups = _build_cv(y, events_aligned.loc[mask].reset_index(drop=True), mode=mode, n_splits=max(2, int(args.splits)), seed=int(args.seed))
    try:
        outer_splits = list(cv.split(np.zeros(len(y)), y, groups))
    except Exception:
        outer_splits = []
    if not outer_splits:
        logger.warning("Could not create outer CV splits for subject bands")
        return
    groups_arr = np.asarray(groups) if groups is not None else None
    if getattr(args, "inner_jobs", None) not in (None, 1):
        logger.info("GridSearchCV is forced to n_jobs=1; ignoring inner_jobs=%s", args.inner_jobs)

    # Param grid
    if mode == "classification":
        param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
        base_est = Pipeline([("scale", StandardScaler(copy=True)), ("clf", LogisticRegression(solver="saga", max_iter=10000, class_weight="balanced", n_jobs=1, random_state=int(args.seed)))])
    else:
        param_grid = {"reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        base_est = Pipeline([("scale", StandardScaler(copy=True)), ("reg", Ridge(alpha=1.0, random_state=42))])
    # For metadata
    filt_meta = []
    band_results = []
    for band_name, (lo, hi) in bands_cfg.items():
        try:
            # Compute band-limited log-power envelope in ROI
            Xb, times = _compute_band_envelope_logpower(epochs, float(lo), float(hi), baseline_window, picks=roi_chs)
            Xb = np.ascontiguousarray(Xb)
        except Exception as e:
            logger.warning(f"Band {band_name} filter failed: {e}")
            continue
        # Outer folds: diagonal via SlidingEstimator w/ inner GridSearch per fold
        diag_per_fold = []
        weights = []
        for (tr, te) in outer_splits:
            if mode == "classification":
                if np.unique(y[tr]).size < 2 or np.unique(y[te]).size < 2:
                    continue
            X_tr, X_te = Xb[tr], Xb[te]
            y_tr, y_te = y[tr], y[te]
            groups_tr = (groups_arr[tr] if groups_arr is not None else None)
            inner_cv_splits, _ = _build_inner_cv_splits(X_tr, y_tr, groups_tr, mode=mode, seed=int(args.seed), max_splits=3)
            search = GridSearchCV(base_est, param_grid=param_grid, cv=inner_cv_splits, scoring=scoring, n_jobs=1, refit=True)
            slide = SlidingEstimator(search, scoring=scoring, n_jobs=int(max(1, args.time_jobs)))
            try:
                with threadpool_limits(1):
                    slide.fit(X_tr, y_tr)
                with threadpool_limits(1):
                    d = np.atleast_1d(slide.score(X_te, y_te))
                diag_per_fold.append(d)
                weights.append(len(te))
            except Exception:
                continue
        if not diag_per_fold:
            logger.warning(f"No valid folds for band {band_name}")
            continue
        # Aggregate
        lengths = {len(d) for d in diag_per_fold}
        L = min(lengths) if len(lengths) > 1 else list(lengths)[0]
        mat = np.vstack([d[:L] for d in diag_per_fold])
        w = np.asarray(weights, float)[: mat.shape[0]]
        if np.nansum(w) > 0:
            diag_obs = np.nansum(mat * w[:, None], axis=0) / np.nansum(w)
        else:
            diag_obs = np.nanmean(mat, axis=0)
        diag_times = np.round(times[: len(diag_obs)], 6)
        # Save TSV per band
        out_df = pd.DataFrame({
            "time": diag_times,
            "score": diag_obs,
            "mode": mode,
            "target": target_col,
            "roi": args.roi or "all",
            "band": band_name,
        })
        out_path = _stats_dir_subject(sub_id) / f"time_generalization_diag_band_{band_name}_{mode}_{args.roi or 'all'}_{target_col}.tsv"
        _ensure_dir(out_path.parent)
        out_df.to_csv(out_path, sep="\t", index=False)
        filt_meta.append({"band": band_name, "l_freq": float(lo), "h_freq": float(hi), "baseline_window": list(map(float, baseline_window))})
    # Save filter specs
    try:
        import json
        meta_path = _stats_dir_subject(sub_id) / f"time_generalization_band_filter_specs_{args.roi or 'all'}_{mode}_{target_col}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"bands": filt_meta, "method": "FIR + Hilbert amplitude, 10*log10 baseline"}, f, indent=2)
    except Exception:
        pass


def group_roi_band_diag_cluster_permutation(
    subjects: List[str],
    mode_filter: Optional[str] = None,
    target_filter: Optional[str] = None,
    n_perm: int = 5000,
    cluster_p: float = 0.05,
    seed: int = 42,

) -> Optional[Path]:
    """Group-level ROI×time (bands×time) effects and cluster-corrected significance.

    One heatmap per ROI (bands on y, time on x). Saves effect matrices and masks per ROI,
    plus a single metadata JSON with filter specs from config.
    """
    logger = get_group_logger("temporal_generalization", "06_temporal_generalization.log")
    rng = np.random.default_rng(int(seed))
    bands_cfg = config.get("time_frequency_analysis.bands", {}) or {}
    band_names = list(bands_cfg.keys())
    if not band_names:
        logger.warning("No bands configured under time_frequency_analysis.bands in YAML.")
        return None
    # Discover ROI/band files
    roi_set = set()
    mode_eff = mode_filter
    target_eff = target_filter
    per_roi_band_paths = {}
    for s in subjects:
        stats_dir = _stats_dir_subject(s)
        cands = sorted(stats_dir.glob("time_generalization_diag_band_*.tsv"))
        for c in cands:
            try:
                head = pd.read_csv(c, sep="\t", nrows=5)
            except Exception:
                continue
            need = {"mode", "target", "roi", "time", "score", "band"}
            if not need.issubset(set(head.columns)):
                continue
            m = str(pd.unique(head["mode"])[0])
            t = str(pd.unique(head["target"])[0])
            r = str(pd.unique(head["roi"])[0])
            b = str(pd.unique(head["band"])[0])
            if mode_filter is not None and m != mode_filter:
                continue
            if target_filter is not None and t != target_filter:
                continue
            if b not in band_names:
                continue
            per_roi_band_paths.setdefault(r, {}).setdefault(b, {})[s] = c
            roi_set.add(r)
            if mode_eff is None:
                mode_eff = m
            if target_eff is None:
                target_eff = t
    if not roi_set:
        logger.warning("No per-subject band diagonals found. Run with --subject-bands first.")
        return None
    plots_dir = _plots_dir_group()
    stats_dir = _stats_dir_group()
    _ensure_dir(plots_dir)
    _ensure_dir(stats_dir)
    # First pass: compute per-ROI effects and masks and collect global vmax
    roi_effects = {}
    roi_masks = {}
    times_global = None
    global_vmax = 0.0
    for roi in sorted(roi_set):
        band_effects = {}
        band_masks = {}
        band_times = {}
        for band in band_names:
            subs_map = per_roi_band_paths.get(roi, {}).get(band, {})
            if not subs_map:
                continue
            times_union = None
            series = []
            for s, pth in subs_map.items():
                try:
                    df = pd.read_csv(pth, sep="\t")
                    t = np.round(df["time"].astype(float).to_numpy(), 6)
                    y = df["score"].astype(float).to_numpy()
                except Exception:
                    continue
                if (mode_eff or "classification") == "classification":
                    val = y - 0.5
                else:
                    val = np.arctanh(np.clip(y, -0.999999, 0.999999))
                series.append((t, val))
                times_union = np.union1d(times_union, t) if times_union is not None else t
            if not series or times_union is None:
                continue
            X = np.vstack([_interp_to_union_times(t, v, times_union) for (t, v) in series])
            means = np.nanmean(X, axis=0)
            sds = np.nanstd(X, axis=0, ddof=1)
            counts = np.sum(np.isfinite(X), axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                t_vals = means / (sds / np.sqrt(np.maximum(counts, 1)))
            df_eff = int(np.nanmin(counts) - 1) if np.nanmin(counts) >= 1 else 0
            if df_eff <= 0:
                t_thr = np.inf
            else:
                t_thr = float(_student_t.ppf(1.0 - cluster_p / 2.0, df=df_eff))
            supra = np.where(np.abs(t_vals) >= t_thr, 1, 0).astype(int)
            def _find_clusters(mask: np.ndarray):
                spans = []
                i = 0
                while i < len(mask):
                    if mask[i]:
                        j = i
                        while j + 1 < len(mask) and mask[j + 1]:
                            j += 1
                        spans.append((i, j))
                        i = j + 1
                    else:
                        i += 1
                return spans
            obs_spans = _find_clusters(supra)
            obs_masses = [float(np.nansum(np.abs(t_vals[s:e + 1]))) for (s, e) in obs_spans]
            max_masses = np.zeros(int(n_perm), dtype=float)
            n_subj = X.shape[0]
            for pi in range(int(n_perm)):
                signs = rng.choice([-1.0, 1.0], size=(n_subj, 1))
                Xp = X * signs
                mp = np.nanmean(Xp, axis=0)
                sdp = np.nanstd(Xp, axis=0, ddof=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    tp = mp / (sdp / np.sqrt(np.maximum(counts, 1)))
                mp_sup = np.where(np.abs(tp) >= t_thr, 1, 0).astype(int)
                spans_p = _find_clusters(mp_sup)
                if spans_p:
                    masses_p = [float(np.nansum(np.abs(tp[s:e + 1]))) for (s, e) in spans_p]
                    max_masses[pi] = np.max(masses_p)
                else:
                    max_masses[pi] = 0.0
            sig_mask = np.zeros_like(supra, dtype=bool)
            for (s0, e0), mass in zip(obs_spans, obs_masses):
                p_fwe = (1.0 + np.sum(max_masses >= mass)) / (len(max_masses) + 1.0)
                if p_fwe <= 0.05:
                    sig_mask[s0:e0 + 1] = True
            band_effects[band] = means
            band_masks[band] = sig_mask
            band_times[band] = times_union
            global_vmax = max(global_vmax, float(np.nanmax(np.abs(means))))
            times_global = np.union1d(times_global, times_union) if times_global is not None else times_union
        roi_effects[roi] = (band_effects, band_times)
        roi_masks[roi] = band_masks
    if times_global is None:
        logger.warning("No bands effects aggregated.")
        return None
    for roi in sorted(roi_set):
        band_effects, band_times = roi_effects.get(roi, ({}, {}))
        band_masks = roi_masks.get(roi, {})
        E = np.full((len(band_names), len(times_global)), np.nan, dtype=float)
        M = np.zeros((len(band_names), len(times_global)), dtype=bool)
        for i, b in enumerate(band_names):
            if b in band_effects:
                E[i, :] = _interp_to_union_times(band_times[b], band_effects[b], times_global)
                pos = np.isin(times_global, band_times[b])
                M[i, pos] = band_masks[b]
        eff_tsv = stats_dir / f"group_roi_band_effects_{roi}_{mode_eff or 'any'}_{target_eff or 'any'}.tsv"
        mask_tsv = stats_dir / f"group_roi_band_mask_{roi}_{mode_eff or 'any'}_{target_eff or 'any'}.tsv"
        eff_df = pd.DataFrame(E, index=band_names, columns=[f"t_{t:.6f}" for t in times_global])
        eff_df.insert(0, "band", band_names)
        eff_df.to_csv(eff_tsv, sep="\t", index=False)
        mask_df = pd.DataFrame(M.astype(int), index=band_names, columns=[f"t_{t:.6f}" for t in times_global])
        mask_df.insert(0, "band", band_names)
        mask_df.to_csv(mask_tsv, sep="\t", index=False)
        vmax = float(global_vmax if np.isfinite(global_vmax) and global_vmax > 0 else 0.2)
        vmin = -vmax
        fig, ax = plt.subplots(figsize=(10.0, max(2.8, 0.35 * len(band_names))))
        im = ax.imshow(
            E,
            aspect="auto",
            origin="lower",
            extent=[times_global[0], times_global[-1], -0.5, len(band_names) - 0.5],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_yticks(range(len(band_names)))
        ax.set_yticklabels(band_names)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Band")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("AUC-0.5" if (mode_eff or "classification") == "classification" else "z(r)")
        for i, b in enumerate(band_names):
            msk = band_masks.get(b)
            tt = band_times.get(b)
            if msk is None or tt is None:
                continue
            def _segs(bm):
                spans = []
                j = 0
                while j < len(bm):
                    if bm[j]:
                        k = j
                        while k + 1 < len(bm) and bm[k + 1]:
                            k += 1
                        spans.append((j, k))
                        j = k + 1
                    else:
                        j += 1
                return spans
            spans = _segs(msk)
            y = i - 0.48
            for s0, e0 in spans:
                ax.hlines(y, tt[s0], tt[e0], colors="k", linewidth=2)
        ax.set_title(f"ROI {roi} — {mode_eff or 'any'} on '{target_eff or 'any'}'")
        plt.tight_layout()
        fig_path = plots_dir / f"group_roi_band_clusters_{roi}_{mode_eff or 'any'}_{target_eff or 'any'}"
        _save_fig(fig, fig_path)
    # Save shared metadata
    try:
        import json
        meta = {
            "bands": [{"name": k, "l_freq": float(v[0]), "h_freq": float(v[1])} for k, v in bands_cfg.items()],
            "baseline_window": list(map(float, config.get("time_frequency_analysis.baseline_window", [-0.5, 0.0]))),
            "method": "FIR + Hilbert amplitude, 10*log10 baseline",
            "n_perm": int(n_perm),
            "cluster_p": float(cluster_p),
            "tail": "two-sided",
        }
        with open(stats_dir / f"group_roi_band_cluster_params_{mode_eff or 'any'}_{target_eff or 'any'}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    return None


def main():
    p = argparse.ArgumentParser(description="Temporal Generalization (time×time) decoding/regression per subject, with optional group averaging.")
    p.add_argument("--subjects", nargs="*", default=None, help="Subject IDs (no 'sub-' prefix). If omitted, tries to infer from derivatives.")
    p.add_argument("--task", type=str, default=TASK, help="BIDS task label.")
    p.add_argument("--target", type=str, default="auto", choices=["auto", "rating", "pain"], help="Target preference: rating (regression), pain (classification), or auto.")
    p.add_argument("--roi", type=str, default=None, help="Optional ROI name (from roi_utils) to restrict channels.")
    p.add_argument("--tmin", type=float, default=None, help="Epoch crop start (s).")
    p.add_argument("--tmax", type=float, default=None, help="Epoch crop end (s).")
    p.add_argument("--decim", type=int, default=5, help="Decimation factor for epochs time axis (>=1). Default 2.")
    p.add_argument("--splits", type=int, default=5, help="CV splits within subject (GroupKFold/StratifiedKFold/KFold).")
    p.add_argument("--permutations", type=int, default=0, help="Number of label permutations for subject-level diagonal empirical p-values (0 to disable).")
    p.add_argument("--permutations-only", action="store_true", help="Reuse cached core outputs when available and run only the permutation stage.")
    p.add_argument("--save-subject-perm-samples", action="store_true", help="Save per-subject permutation diagonal samples for exact group permutation.")
    p.add_argument("--perm-samples-limit", type=int, default=None, help="Optional cap on number of saved subject permutation samples.")
    p.add_argument("--n-jobs-subjects", type=int, default=1, help="Parallel jobs across subjects (joblib). Use >1 to process multiple subjects concurrently.")
    p.add_argument("--inner-jobs", type=int, default=1, help="Deprecated; inner GridSearchCV runs single-threaded.")
    p.add_argument("--time-jobs", type=int, default=1, help="Parallel jobs across time points inside GeneralizingEstimator/SlidingEstimator.")
    p.add_argument("--outer-jobs", type=int, default=1, help="Parallel jobs across outer CV folds (loky backend).")
    p.add_argument("--perm-jobs", type=int, default=1, help="Parallel jobs across subject-level label permutations for the diagonal null.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for permutations.")
    p.add_argument("--two-sided-classification", action="store_true", help="Use two-sided tests around 0.5 for classification diagonals.")
    p.add_argument("--group-average", action="store_true", help="After per-subject runs, compute a group-average TGM.")
    p.add_argument("--subject-bands", action="store_true", help="Compute per-subject band-limited diagonal decoding within the selected ROI.")
    p.add_argument("--group-roi-bands-clusters", action="store_true", help="Compute ROI×time (bands×time) group effects with cluster-corrected significance.")
    p.add_argument("--group-bands-perm", type=int, default=5000, help="Permutations for ROI band cluster test.")
    p.add_argument("--group-bands-cluster-p", type=float, default=0.05, help="Cluster-forming p-threshold (two-sided).")
    p.add_argument("--group-average-perm", action="store_true", help="Compute group-level permutation p-values for diagonal (requires subject permutations).")
    p.add_argument("--group-average-perm-exact", action="store_true", help="Compute exact group-level permutation using saved subject permutation samples.")
    p.add_argument("--group-perm-samples", type=int, default=5000, help="Monte Carlo samples for approximating group null (diagonal).")
    p.add_argument("--group-roi-clusters", action="store_true", help="Compute ROI×time group effects with cluster-corrected significance.")
    p.add_argument("--group-roi-perm", type=int, default=5000, help="Permutations for ROI cluster test.")
    p.add_argument("--group-roi-cluster-p", type=float, default=0.05, help="Cluster-forming p-threshold (two-sided).")
    p.add_argument("--group-mode", type=str, default=None, choices=["classification", "regression", None], help="Group mode filter (optional).")
    p.add_argument("--group-target", type=str, default=None, help="Group target filter (optional; must match subject diag target).")
    args = p.parse_args()

    # If subjects not given, try to list available sub-*/eeg epochs under derivatives
    subs = args.subjects
    if not subs:
        subs = []
        root = DERIV_ROOT
        if root.exists():
            for sd in sorted(root.glob("sub-*")):
                if not sd.is_dir():
                    continue
                sid = sd.name[4:]
                if _find_clean_epochs_path(sid, args.task) is not None:
                    subs.append(sid)
    if not subs:
        print("No subjects found. Provide via --subjects or create derivatives first.")
        return

    saved_stats: List[Path] = []
    # Avoid thread oversubscription from BLAS when using multiple subject jobs
    if int(args.n_jobs_subjects) != 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    def _run_one(sub_id: str) -> Optional[Path]:
        return temporal_generalization_subject(
            subject=sub_id,
            task=args.task,
            target=args.target,
            roi=args.roi,
            tmin=args.tmin,
            tmax=args.tmax,
            decim=max(1, int(args.decim)),
            n_splits=max(2, int(args.splits)),
            permutations=max(0, int(args.permutations)),
            seed=int(args.seed),
            save_perm_samples=bool(args.save_subject_perm_samples),
            max_perm_save=(int(args.perm_samples_limit) if args.perm_samples_limit is not None else None),
            inner_jobs=int(args.inner_jobs),
            time_jobs=int(args.time_jobs),
            outer_jobs=int(args.outer_jobs),
            perm_jobs=int(args.perm_jobs),
            two_sided_classification=bool(args.two_sided_classification),
            permutations_only=bool(args.permutations_only),
        )
    if int(args.n_jobs_subjects) == 1:
        for s in subs:
            out = _run_one(s)
            if out is not None:
                saved_stats.append(out)
    else:
        results = Parallel(n_jobs=int(args.n_jobs_subjects), backend="loky", verbose=0)(
            delayed(_run_one)(s) for s in subs
        )
        saved_stats = [p for p in results if p is not None]

    # Optional per-subject band-limited diagonals (requires ROI)
    if args.subject_bands:
        if not args.roi:
            print("--subject-bands requires --roi to specify the ROI for channel selection.")
        else:
            bands_cfg = config.get("time_frequency_analysis.bands", {}) or {}
            baseline = tuple(config.get("time_frequency_analysis.baseline_window", [-0.5, 0.0]))
            # Run sequentially per subject to limit memory
            for s in subs:
                try:
                    _run_subject_bands(s, args, bands_cfg, baseline)
                except Exception as e:
                    print(f"Subject {s} band-limited diagonals failed: {e}")

    if args.group_average and saved_stats:
        # Pick mode based on requested target
        mode_f = "classification" if args.target == "pain" else ("regression" if args.target == "rating" else None)
        group_average_tgm(subs, mode_filter=mode_f)
        # Prefer exact if samples exist; else approximate
        if args.group_average_perm or args.group_average_perm_exact:
            exact_path = group_diag_permutation_exact(subs, mode_filter=mode_f, n_samples=int(args.group_perm_samples), seed=int(args.seed))
            if exact_path is None and args.group_average_perm:
                group_diag_permutation(subs, mode_filter=mode_f, n_samples=int(args.group_perm_samples), seed=int(args.seed))

    # ROI×time group with cluster-corrected significance
    if args.group_roi_clusters:
        group_roi_diag_cluster_permutation(
            subs,
            mode_filter=(args.group_mode or ("classification" if args.target == "pain" else ("regression" if args.target == "rating" else None))),
            target_filter=args.group_target,
            n_perm=int(args.group_roi_perm),
            cluster_p=float(args.group_roi_cluster_p),
            seed=int(args.seed),
        )

    # ROI×time (bands×time) cluster analysis
    if args.group_roi_bands_clusters:
        group_roi_band_diag_cluster_permutation(
            subs,
            mode_filter=(args.group_mode or ("classification" if args.target == "pain" else ("regression" if args.target == "rating" else None))),
            target_filter=args.group_target,
            n_perm=int(args.group_bands_perm),
            cluster_p=float(args.group_bands_cluster_p),
            seed=int(args.seed),
        )


if __name__ == "__main__":
    main()






