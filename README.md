# EEG_fMRI-Analysis

A production-ready analysis stack for multimodal thermal pain experiments that combines high-density EEG preprocessing, Neurologic Pain Signature (NPS) fMRI modeling, and cross-modal machine learning.

## Table of contents
1. [Key capabilities](#key-capabilities)
2. [Repository structure](#repository-structure)
3. [Environment setup](#environment-setup)
4. [Configuration](#configuration)
5. [EEG workflow](#eeg-workflow)
6. [fMRI workflow](#fmri-workflow)
7. [Cross-modal modeling](#cross-modal-modeling)
8. [Outputs and quality control](#outputs-and-quality-control)
9. [Extending the project](#extending-the-project)

## Key capabilities
- End-to-end EEG ingestion from BrainVision to BIDS with stringent behavioral alignment and logging safeguards.
- Automated fMRI processing to reproduce the Neurologic Pain Signature GLM, including harmonization to the canonical NPS grid.
- Machine-learning bridge that maps EEG-derived features to fMRI NPS beta estimates with nested cross-validation and reproducibility manifests.
- Shared configuration loaders, derivative layout helpers, and logging utilities to keep analyses synchronized across modalities.

## Repository structure
```
.
├── README.md
├── requirements.txt
├── eeg_pipeline/
│   ├── raw_to_bids.py
│   ├── merge_behavior_to_events.py
│   ├── 01_foundational_analysis.py
│   ├── 02_time_frequency_analysis.py
│   ├── 03_feature_engineering.py
│   ├── 04_behavior_feature_analysis.py
│   ├── 05_decode_pain_experience.py
│   ├── 06_temporal_generalization.py
│   ├── config_loader.py
│   ├── eeg_config.yaml
│   └── utility modules (alignment, IO, logging, ROI, TFR)
├── fmri_pipeline/
│   └── NPS/
│       ├── 00_config.yaml
│       ├── config_loader.py
│       ├── 01_discover_inputs.py
│       ├── 02_build_confounds_24HMP_outliers.py
│       ├── 03_build_design_matrices.py
│       ├── 04_fit_first_level_glm.py
│       ├── 05_combine_runs_fixed_effects.py
│       ├── 06_harmonize_to_nps_grid.py
│       ├── 07_score_nps_conditions.py
│       ├── 08_optional_trials_glm_and_scoring.py
│       ├── 09_subject_metrics.py
│       ├── 10_group_stats.py
│       ├── 11_plots.py
│       └── 12_qc_collation.py
└── machine_learning/
    └── train_eeg_to_nps.py
```

## Environment setup
1. **Create a Python ≥3.10 environment.** Conda, `venv`, and Hatch are supported.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Verify neuroimaging toolboxes:** ensure the required FreeSurfer licenses and FSL installations exist if your preprocessing workflow depends on them.
4. **Set thread limits** (optional) via `OMP_NUM_THREADS` and `MKL_NUM_THREADS` for reproducible performance.

## Configuration
- `eeg_pipeline/eeg_config.yaml`: project paths, subject lists, montage selections, ROI definitions, ERP/TFR windows, decoding models, and derivative directories. Adjusting this single file propagates to every EEG script.
- `fmri_pipeline/NPS/00_config.yaml`: BIDS and fMRIPrep directories, TR, slice timing, task runs, nuisance regressors, contrasts, NPS templates, and derivative roots.
- `config_loader.py` modules: provide attribute-style access, environment guards (thread limits, Matplotlib backends), and shared styling for plots and logs.

## EEG workflow
1. **Convert BrainVision recordings to BIDS:**
   ```bash
   python eeg_pipeline/raw_to_bids.py --source_root /path/to/source --bids_root /path/to/bids
   ```
   - Applies channel normalization, montage assignment, metadata validation, and dataset descriptions.
2. **Attach behavioral annotations:**
   ```bash
   python eeg_pipeline/merge_behavior_to_events.py /path/to/bids/sub-*/eeg/sub-*_events.tsv
   ```
   - Enforces strict trial matching using centralized alignment utilities and raises on mismatches.
3. **Run numbered analyses:**
   ```bash
   python eeg_pipeline/01_foundational_analysis.py --group all
   python eeg_pipeline/02_time_frequency_analysis.py --group all
   python eeg_pipeline/03_feature_engineering.py --subject 0001 --subject 0002
   python eeg_pipeline/04_behavior_feature_analysis.py --subject 0001 --subject 0002
   python eeg_pipeline/05_decode_pain_experience.py --subjects 0001 0002 --models elasticnet random_forest
   python eeg_pipeline/06_temporal_generalization.py --subjects 0001 0002 --group-average
   ```
   - Each stage writes plots, TSV/JSON summaries, and logs under `bids_root/derivatives/eeg/` while reusing shared ROI and time–frequency utilities.

## fMRI workflow
1. **Inventory inputs and QC:**
   ```bash
   python fmri_pipeline/NPS/01_discover_inputs.py
   ```
   - Confirms BOLD runs, masks, confounds, and events; records trial counts and temperature balance.
2. **Prepare design matrices and confounds:** run steps `02` and `03` to assemble the 24-parameter motion model and HRF-convolved regressors.
3. **Fit Neurologic Pain Signature GLM:** execute scripts `04`–`06` to fit first-level models, combine runs with fixed effects, and harmonize outputs to the canonical NPS voxel grid.
4. **Score, summarize, and plot:**
   ```bash
   python fmri_pipeline/NPS/07_score_nps_conditions.py
   python fmri_pipeline/NPS/09_subject_metrics.py
   python fmri_pipeline/NPS/10_group_stats.py
   python fmri_pipeline/NPS/11_plots.py
   python fmri_pipeline/NPS/12_qc_collation.py
   ```
   - Generates NPS scores, subject metrics, group comparisons, visualizations, and QC bundles.
5. **Optional trial-level modeling:** `08_optional_trials_glm_and_scoring.py` estimates trial-wise betas when finer temporal resolution is required.

## Cross-modal modeling
```bash
python machine_learning/train_eeg_to_nps.py \
    --models elasticnet random_forest \
    --bands alpha beta gamma \
    --permutation-per-model 100
```
- Discovers subjects with matched EEG features and NPS scores, validates schema consistency, and concatenates aligned trials.
- Performs nested cross-validation with subject/run grouping to control for leakage, logs diagnostics, and exports model weights.
- Emits predictions, metrics, permutation tests, and manifests containing CLI arguments, git metadata, and package versions.

## Outputs and quality control
- **EEG derivatives:** BIDS-compliant metadata, ERP/TFR plots, TSV summaries, and logs stored under `derivatives/sub-*/eeg/` and `derivatives/group/eeg/`.
- **fMRI derivatives:** Harmonized NPS grids, QC TSVs, and figure panels organized by subject and group.
- **Machine-learning artifacts:** Serialized models, feature importances, per-fold metrics, and reproducibility manifests saved in timestamped directories.
- **Logging:** Each script writes console and file logs to support auditing and reruns.

## Extending the project
1. Favor established neuroimaging toolkits (MNE/mne-bids, Nilearn, scikit-learn, PyTorch) rather than custom implementations.
2. Expose new parameters through the YAML configuration files so downstream scripts stay synchronized.
3. Maintain BIDS-compliant derivatives and register additional outputs in QC summaries to preserve traceability.
4. Reuse existing logging, manifest, and cross-validation utilities when adding new modeling scripts to ensure reproducibility.