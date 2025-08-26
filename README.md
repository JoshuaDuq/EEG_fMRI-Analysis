# EEG Analysis Pipeline: CLI Reference and Quickstart

This repository contains a small EEG analysis pipeline built on top of MNE-Python and MNE-BIDS. It provides command-line tools to:

- Convert BrainVision EEG recordings into a BIDS dataset
- Merge behavioral TrialSummary.csv data into BIDS events.tsv
- Run foundational QC and ERP plots
- Run time–frequency (TFR) analyses with pooled and per-temperature outputs

All scripts live in `eeg_pipeline/` and write outputs into a BIDS-style tree under `eeg_pipeline/bids_output/` and its `derivatives/` subfolder.


## Contents
- CLI tools
  - `raw_to_bids.py`
  - `merge_behavior_to_events.py`
  - `01_foundational_analysis.py`
  - `02_time_frequency_analysis.py`
  - `03_feature_engineering.py`
  - `05_decode_pain_experience.py`
- Project layout and data expectations
- Configuration and defaults
- Environment setup
- Typical workflows
- Outputs and file naming
- Troubleshooting


## CLI Tools

### 1) Convert raw EEG to BIDS: `eeg_pipeline/raw_to_bids.py`

Purpose: Convert BrainVision `.vhdr/.eeg/.vmrk` recordings found under `source_data/sub-*/eeg/` into a BIDS dataset under `bids_output/` using MNE-BIDS.

Key behavior:
- Filters annotations to keep only Stim_on markers ("Stim_on" or exactly `"Stim_on/S 1"`).
- Optionally zero-bases those onsets.
- Writes BIDS subject-level folders and sidecars.
- Optionally merges behavior (TrialSummary.csv) into resulting events.tsv (Stim_on rows only).

Usage (Windows PowerShell examples):

```powershell
# Basic conversion using defaults
python eeg_pipeline/raw_to_bids.py

# Convert only specific subjects and overwrite
python eeg_pipeline/raw_to_bids.py --subjects 001 002 --overwrite

# Customize montage and line frequency metadata
python eeg_pipeline/raw_to_bids.py --montage easycap-M1 --line_freq 60

# Also merge behavior into events after conversion
python eeg_pipeline/raw_to_bids.py --merge_behavior

# Zero-base kept annotation onsets (first kept onset becomes 0.0 s)
python eeg_pipeline/raw_to_bids.py --zero_base_onsets
```

Arguments:
- `--source_root` (str): Path to `source_data` root containing `sub-*/eeg/*.vhdr`. Default: project `eeg_pipeline/source_data`.
- `--bids_root` (str): Output BIDS root directory. Default: from `config.py` if present, else `eeg_pipeline/bids_output`.
- `--task` (str): BIDS task label. Default: from `config.py` or `thermalactive`.
- `--subjects` (list[str]): Optional list of subject labels to include, e.g., `001 002`. If omitted, all found are used.
- `--montage` (str): Standard montage name for `mne.channels.make_standard_montage` (e.g., `easycap-M1`). Use empty string `""` to skip.
- `--line_freq` (float): Line noise frequency metadata for sidecar (Hz). Default: from `config.py` (`zapline_fline`) or 60.0.
- `--overwrite` (flag): Overwrite existing BIDS files.
- `--merge_behavior` (flag): Merge Psychopy TrialSummary.csv into Stim_on events after conversion.
- `--zero_base_onsets` (flag): Zero-base kept annotation onsets.

Outputs:
- BIDS dataset under `eeg_pipeline/bids_output/` (unless overridden by `--bids_root`).
- Creates `dataset_description.json` via MNE-BIDS.


### 2) Merge behavior into events: `eeg_pipeline/merge_behavior_to_events.py`

Purpose: Merge behavioral TrialSummary.csv columns into the BIDS `events.tsv` files, aligning rows to Stim_on events per subject.

Usage:

```powershell
# Dry-run (no writes) to see which columns would be merged
python eeg_pipeline/merge_behavior_to_events.py --dry_run

# Perform merge for a custom BIDS root and source_data root
python eeg_pipeline/merge_behavior_to_events.py --bids_root eeg_pipeline/bids_output --source_root eeg_pipeline/source_data --task thermalactive
```

Arguments:
- `--bids_root` (str): BIDS root containing `sub-*/eeg/*_events.tsv`. Default: from `config.py` if available.
- `--source_root` (str): Source root containing `sub-*/PsychoPy_Data/*TrialSummary.csv`. Default: `eeg_pipeline/source_data`.
- `--task` (str): Task label used in events filenames. Default: from `config.py` or `thermalactive`.
- `--dry_run` (flag): Print planned changes without writing.

Notes:
- Only `Stim_on` rows are updated with behavioral columns, preserving non-stim rows.
- Length mismatches are trimmed to the shorter length with a warning.


### 3) Foundational QC and ERP: `eeg_pipeline/01_foundational_analysis.py`

Purpose: Generate quick QC plots and basic ERP analyses for one subject. Expects a cleaned epochs FIF in derivatives.

Usage:

```powershell
python eeg_pipeline/01_foundational_analysis.py --subject 001 --task thermalactive
```

Arguments:
- `--subject, -s` (str): BIDS subject label without the `sub-` prefix (e.g., `001`).
- `--task, -t` (str): BIDS task label. Default: from `config.py` or `thermalactive`.

What it does:
- Loads cleaned epochs from `derivatives/sub-<ID>/eeg/*proc-clean*_epo.fif` (several fallbacks).
- Loads `events.tsv`, aligns to epochs length, and attaches as `epochs.metadata` for native MNE selection.
- Saves counts TSVs (e.g., pain levels, temperature levels) in the subject plots directory.
- QC plots: PSD, sensor layout, drop log, trial images.
- ERP contrasts: pain vs non-pain (`pain_binary_coded`/`pain_binary`/`pain`).
- ERP by temperature: one butterfly per temperature and a multi-level GFP comparison.

Outputs (subject-specific):
- `eeg_pipeline/bids_output/derivatives/sub-001/eeg/plots/`
  - `qc_*.png`, `erp_pain_binary_*.png`, `erp_by_temperature_gfp.png`, `erp_temperature_*.png`
  - `counts_pain.tsv`, `counts_temperature.tsv`


### 4) Time–Frequency Analysis: `eeg_pipeline/02_time_frequency_analysis.py`

Purpose: Compute per-trial TFRs (Morlet), apply baseline correction, and generate pooled and per-temperature visualizations (Cz, ROI TFRs, ROI topomaps, and pain/non-pain contrasts).

Usage:

```powershell
# Pooled across all trials (default)
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --task thermalactive

# Per-temperature only
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --task thermalactive --temperature_strategy per

# Run both pooled and per-temperature in one go
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --task thermalactive --temperature_strategy both

# Customize plateau time window (for topomaps)
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --plateau_tmin 0.5 --plateau_tmax 8.0
```

Arguments:
- `--subject, -s` (str): BIDS subject label without `sub-`.
- `--task, -t` (str): BIDS task label. Default: from `config.py` or `thermalactive`.
- `--plateau_tmin` (float): Start of plateau window in seconds (default `0.5`).
- `--plateau_tmax` (float): End of plateau window in seconds (default `8.0`).
- `--temperature_strategy` (str): One of `pooled`, `per`, or `both` (default `pooled`).

What it does:
- Loads cleaned epochs and attaches `events.tsv` as metadata.
- Computes per-trial TFRs via `mne.time_frequency.tfr_morlet` from 4–100 Hz (log-spaced), `n_cycles = freqs/2`.
- Baseline correction applied with `mode='logratio'` using pre-stim baseline `(None, 0.0)`.
- Pooled outputs:
  - Cz TFR (all trials), pain/non-pain Cz, difference Cz
  - ROI TFRs (all trials) + band-limited (alpha/beta/gamma)
  - ROI topomaps (all trials) + pain/non-pain + difference, per-band
- Per-temperature outputs (when `per` or `both`): same suite under temperature-specific folders.

Outputs (subject-specific):
- Pooled: `eeg_pipeline/bids_output/derivatives/sub-001/eeg/plots/`
- Per-temperature: `eeg_pipeline/bids_output/derivatives/sub-001/eeg/plots/temperature/temp-<label>/`
  - Temperature labels are made filesystem-safe (e.g., `47.3 → 47p3`).

Notes:
- Temperature column is auto-detected among `stimulus_temp`, `stimulus_temperature`, `temp`, `temperature`.
- ROI sets include Frontal, Central, Parietal, Occipital, Temporal, and Sensorimotor, matched via 10–10 name regexes.
- Pain/non-pain contrasts require `pain_binary_coded` in events metadata.


### 5) Feature Engineering: `eeg_pipeline/03_feature_engineering.py`

Purpose: Build ML-ready per-trial feature matrices and targets. Extracts direct EEG power features from Morlet TFRs and, if present, flattens per-trial connectivity matrices.

Usage:

```powershell
# Single subject (uses default task if not provided)
python eeg_pipeline/03_feature_engineering.py --subjects 001 --task thermalactive

# Multiple subjects
python eeg_pipeline/03_feature_engineering.py --subjects 001 002 --task thermalactive
```

Arguments:
- `--subjects` (list[str]): Subject labels without the `sub-` prefix; use space-separated list.
- `--task` (str): BIDS task label. Default from `config.py` or `thermalactive`.

What it does:
- Loads cleaned epochs from `derivatives/sub-<ID>/eeg/*proc-clean*_epo.fif`.
- Loads BIDS `events.tsv` and aligns rows to epochs.
- Computes per-trial Morlet TFRs in-memory (uses config overrides when present).
- Extracts band-limited mean power per EEG channel over a plateau window (default 3.0–10.5 s) for bands: alpha, beta, gamma.
- Optionally loads per-trial functional connectivity arrays (AEC, wPLI) when available as `.npy` files under derivatives and flattens their lower triangles per trial.
- Aligns trial counts across all parts (power, connectivity, targets) by trimming to the minimum length.
- Auto-selects a behavioral target column from events (preference order includes `vas_final_coded_rating`, `vas_final_rating`, `vas_rating`, then fallbacks to binary pain columns).

Outputs (subject-specific) written to `eeg_pipeline/bids_output/derivatives/sub-<ID>/eeg/features/`:
- `features_eeg_direct.tsv` — direct EEG (power) features with descriptive headers like `pow_alpha_Cz`, `pow_beta_F3`, ...
- `features_eeg_direct_columns.tsv` — one-column list mirroring the headers of `features_eeg_direct.tsv` (for traceability).
- `features_connectivity.tsv` — connectivity features (if connectivity arrays were found). Columns use prefixes like `aec_alpha_<roi_i>__<roi_j>` or `wpli_beta_<roi_i>__<roi_j>`.
- `features_all.tsv` — combined matrix of direct EEG power and connectivity (if available).
- `target_vas_ratings.tsv` — one-column TSV with the selected behavioral target.

Connectivity input expectations:
- This repository does not currently compute or save connectivity arrays; if you place files matching patterns like `sub-<ID>_task-<task>_*connectivity_aec_<band>*_all_trials.npy` (and similarly for `wpli` and optional `*_labels.npy`) under `derivatives/sub-<ID>/eeg/`, they will be detected and included.


### 6) Pain Decoding (LOSO CV): `eeg_pipeline/05_decode_pain_experience.py`

Purpose: Perform leave-one-subject-out (LOSO) decoding of pain ratings from per-trial EEG features. Implements nested CV for hyperparameter search and saves predictions, metrics, and plots. Includes three models:
- Elastic Net (with scaling)
- Random Forest
- Optional Riemannian regression (Covariances → TangentSpace → Ridge) using raw epochs (requires `pyriemann`)

Usage:

```powershell
# Decode all subjects discovered under derivatives (recommended)
python eeg_pipeline/05_decode_pain_experience.py --subjects all --n_jobs -1 --seed 42

# Decode a subset of subjects
python eeg_pipeline/05_decode_pain_experience.py --subjects 001 002 003 --n_jobs 4 --seed 123
```

Arguments:
- `--subjects` (list[str] or `all`): Subject labels without the `sub-` prefix, or `all` to include all with features present.
- `--task` (str): BIDS task label. Default from `config.py` or `thermalactive`.
- `--n_jobs` (int): Parallel jobs for inner CV/grid search. Use `-1` for all cores.
- `--seed` (int): Random seed for reproducibility.

What it does:
- Aggregates per-trial tabular features and targets across subjects from `.../derivatives/sub-<ID>/eeg/features/`.
- Runs nested LOSO CV to produce out-of-subject predictions and per-subject metrics.
- Saves pooled scatter plots and TSVs for each model.
- If `pyriemann` is installed, also loads cleaned epochs to run the Riemannian model; epochs are preloaded and channels aligned across subjects.

Outputs (written to `eeg_pipeline/bids_output/derivatives/decoding/`):
- Predictions: `elasticnet_loso_predictions.tsv`, `rf_loso_predictions.tsv`, `riemann_loso_predictions.tsv` (if enabled), plus naïve baselines.
- Per-subject metrics: `elasticnet_per_subject_metrics.tsv`, `rf_per_subject_metrics.tsv`, `riemann_per_subject_metrics.tsv` (if enabled).
- Plots: `plots/elasticnet_loso_actual_vs_predicted.png`, `plots/rf_loso_actual_vs_predicted.png`.
- Summary: `summary.json` with overall metrics (if present).

Notes:
- The script uses a non-interactive matplotlib backend (`Agg`) for headless environments.
- Riemann model requires `pyriemann` (install with `pip install pyriemann`).
- Ensure features/targets exist by running `03_feature_engineering.py` first for the subjects of interest.


## Project Layout and Data Expectations

```
EEG_fMRI_Analysis/
├─ eeg_pipeline/
│  ├─ raw_to_bids.py
│  ├─ merge_behavior_to_events.py
│  ├─ 01_foundational_analysis.py
│  ├─ 02_time_frequency_analysis.py
│  ├─ config.py            # optional; provides defaults (bids_root, deriv_root, task, montage, etc.)
│  ├─ source_data/
│  │  └─ sub-XXX/
│  │     ├─ eeg/          # raw BrainVision files (*.vhdr)
│  │     └─ PsychoPy_Data/   # behavioral CSV (*TrialSummary.csv)
│  └─ bids_output/
│     ├─ sub-XXX/
│     │  └─ eeg/          # raw BIDS outputs
│     └─ derivatives/
│        └─ sub-XXX/
│           └─ eeg/
│              └─ plots/  # figures and counts TSVs
└─ requirements.txt
```


## Configuration and Defaults

You may create `eeg_pipeline/config.py` to centralize defaults. The scripts gracefully fall back to built-ins if the module is missing.

Common config fields used by scripts:
- `bids_root` (str or Path): Path to the BIDS root (default: `eeg_pipeline/bids_output`).
- `deriv_root` (str or Path): Path to the derivatives root (default: `<bids_root>/derivatives`).
- `task` (str): Default BIDS task label (default: `thermalactive`).
- `eeg_template_montage` (str): Montage name for `raw_to_bids.py` (default: `easycap-M1`).
- `zapline_fline` (float): Line frequency metadata for sidecar (default: 60.0 Hz).


## Environment Setup

```powershell
# From repository root (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you use a different Python version, ensure compatibility with MNE and MNE-BIDS used in `requirements.txt`.


## Typical Workflows

### Workflow A: Full pipeline from raw → BIDS → behavior merge → QC → TFR

```powershell
# 1) Convert BrainVision to BIDS; also merge behavior and overwrite if needed
python eeg_pipeline/raw_to_bids.py --merge_behavior --overwrite

# 2) (Optional) If you skipped behavior merge in step 1, do it now
python eeg_pipeline/merge_behavior_to_events.py

# 3) Foundational QC + ERP
python eeg_pipeline/01_foundational_analysis.py --subject 001 --task thermalactive

# 4) Time–frequency (pooled)
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --task thermalactive

# 5) Time–frequency (per-temperature)
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --task thermalactive --temperature_strategy per
```

### Workflow B: Re-run TFR with both pooled and per-temperature

```powershell
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --temperature_strategy both
```

### Workflow C: Feature engineering for all subjects → LOSO pain decoding

```powershell
# 1) Build features for all desired subjects (example list; repeat or extend as needed)
python eeg_pipeline/03_feature_engineering.py --subjects 001 002 003 004 005 --task thermalactive

# 2) Run LOSO decoding across the same set (or use --subjects all if all are prepared)
python eeg_pipeline/05_decode_pain_experience.py --subjects all --n_jobs -1 --seed 42
```


## Outputs and File Naming

- Subject plots live under: `eeg_pipeline/bids_output/derivatives/sub-<ID>/eeg/plots/`
- Per-temperature plots live under: `.../plots/temperature/temp-<label>/`
- Common filenames:
  - `tfr_Cz_*_baseline_logratio.png`
  - `tfr_ROI-<ROI>_*_baseline_logratio.png`
  - `topomap_<band>_*_baseline_logratio.png`
  - `topomap_ROI-<ROI>_<band>_*_baseline_logratio.png`
  - `erp_*` and `counts_*.tsv`

Decoding outputs (under `eeg_pipeline/bids_output/derivatives/decoding/`):
- `elasticnet_loso_predictions.tsv`, `elasticnet_per_subject_metrics.tsv`
- `rf_loso_predictions.tsv`, `rf_per_subject_metrics.tsv`
- `riemann_loso_predictions.tsv`, `riemann_per_subject_metrics.tsv` (if `pyriemann` installed)
- `baseline_global_loso_predictions.tsv` and per-subject metrics for baselines
- `plots/*_loso_actual_vs_predicted.png`
- `summary.json` (if present)


## Troubleshooting

- Missing or misaligned events length: scripts will trim to the min length and warn. Ensure `events.tsv` matches the epochs count.
- No `pain_binary_coded` column: pain/non-pain contrasts will be skipped.
- No temperature column: per-temperature analyses will be skipped. Expected columns include `stimulus_temp`, `stimulus_temperature`, `temp`, or `temperature`.
- Runtime: TFR computation is the slowest step. Consider running pooled only or specific subjects first.
- Montage issues: If your channel names don’t match the template, `raw_to_bids.py` will continue without setting a montage.

Decoding-specific:
- ConstantInputWarning or unrealistically high inner-CV scores: Check for duplicated subjects with identical data causing leakage; deduplicate or ensure duplicates share the same group label so LOSO leaves them out together.
- ElasticNet convergence warnings: Already tuned; if they persist, increase `max_iter` or restrict the alpha grid to stronger regularization.
- PyRiemann missing: Install `pip install pyriemann` or run only ElasticNet/RF.
- Epochs channel picking requires preloading: epochs are read with `preload=True`. If you adapt the code, ensure data are loaded before dropping/reordering channels.


---

If you want additional flags (e.g., temperature binning, custom frequency bands, or parallelization), open an issue or request and we can extend the CLIs accordingly.
