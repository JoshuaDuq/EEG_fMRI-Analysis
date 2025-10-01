# EEG_fMRI-Analysis

Comprehensive EEG and fMRI processing pipelines for multimodal thermal-pain experiments. The project standardizes how raw BrainVision EEG, fMRIPrep outputs, and behavioral annotations are converted into BIDS-compliant derivatives, Neurologic Pain Signature (NPS) scores, and cross-modal machine-learning models.

---

## 📚 Table of contents
1. [Overview](#overview)
2. [Key capabilities](#key-capabilities)
3. [Repository layout](#repository-layout)
4. [Prerequisites](#prerequisites)
5. [Quick start](#quick-start)
6. [Data organization](#data-organization)
7. [Configuration](#configuration)
8. [EEG workflow](#eeg-workflow)
9. [fMRI workflow](#fmri-workflow)
10. [Cross-modal learning](#cross-modal-learning)
11. [Outputs & quality control](#outputs--quality-control)
12. [Reproducibility tips](#reproducibility-tips)
13. [Extending the pipelines](#extending-the-pipelines)
14. [Support & citation](#support--citation)

---

## Overview
This repository contains synchronized pipelines for high-density EEG preprocessing, NPS-focused fMRI modeling, and supervised learning that links the two modalities. Each pipeline is designed for reproducibility in multi-site, high-throughput settings:

- **Config-driven orchestration.** YAML configuration files capture subject rosters, preprocessing parameters, statistical models, and output locations.
- **Shared utilities.** Common modules manage logging, manifest generation, layout discovery, and environment guards to make reruns deterministic.
- **BIDS-first mindset.** EEG and fMRI derivatives follow BIDS and BIDS-Derivatives conventions, simplifying downstream integration with MNE, Nilearn, and other neuroimaging tools.

---

## Key capabilities
- **EEG ingestion to derivatives.** Automated BrainVision → BIDS conversion, behavior-event alignment, and pipelines for ERP, time–frequency, and decoding analyses powered by MNE/mne-bids-style utilities.
- **NPS-centric fMRI modeling.** Modular scripts reproduce the Neurologic Pain Signature GLM, generate subject and group summaries, and harmonize outputs to the canonical NPS voxel grid.
- **Cross-modal prediction.** Machine-learning bridges use matched EEG features and NPS betas with nested cross-validation, permutation testing, and manifest logging.
- **Audit-ready outputs.** Every script records CLI arguments, git metadata, package versions, and QC summaries alongside numerical results and figures.

---

## Repository layout

| Path | Description |
|------|-------------|
| `eeg_pipeline/` | EEG ingestion, preprocessing, feature extraction, and decoding scripts, all governed by `eeg_config.yaml`. |
| `fmri_pipeline/NPS/` | Neurologic Pain Signature modeling workflow with scripts for design matrix creation, GLM fitting, harmonization, and QC. |
| `machine_learning/` | Cross-modal model training scripts for linear, kernel, and deep models that map EEG features onto NPS outcomes. |
| `requirements.txt` | Minimal Python dependencies for running the pipelines. |
| `README.md` | Project documentation (this file). |

> Tip: Use `python <script_name>.py --help` within each submodule to inspect CLI options and available safeguards.

---

## Prerequisites
- **Python:** Version 3.10 or newer.
- **System dependencies:**
  - fMRI processing relies on [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki), [AFNI](https://afni.nimh.nih.gov/), or similar toolboxes depending on your fMRIPrep configuration. Ensure licenses (e.g., FreeSurfer) are installed before running GLM steps.
  - EEG preprocessing expects standard scientific libraries (NumPy, SciPy) and neuroimaging stacks such as [MNE](https://mne.tools/).
- **Python packages:**
  ```bash
  pip install -r requirements.txt
  ```
- **Hardware:** Pipelines have been executed on HPC clusters and high-memory workstations. For deterministic results, pin thread counts with `OMP_NUM_THREADS` and `MKL_NUM_THREADS` and consider containerizing via Singularity/Apptainer.

---

## Quick start
1. **Clone the repository** and create a virtual environment.
2. **Install Python dependencies** with `pip install -r requirements.txt`.
3. **Populate configuration files** (`eeg_config.yaml`, `00_config.yaml`) with project-specific paths and subject identifiers.
4. **Validate inputs** using discovery scripts (`raw_to_bids.py`, `01_discover_inputs.py`) before launching heavy computations.
5. **Execute numbered scripts sequentially** within each modality (EEG `01–06`, fMRI `01–12`) to build complete derivative sets.
6. **Run cross-modal learning** once EEG features and NPS betas are materialized.

The pipelines are modular—rerun only the components impacted by new data or configuration changes.

---

## Data organization
- **EEG raw data:** BrainVision `.vhdr`, `.vmrk`, `.eeg` files organized per participant prior to conversion.
- **EEG BIDS root:** Output of `raw_to_bids.py`, including `sub-*/ses-*/eeg/` files, derivative folders, and study-level metadata.
- **fMRI inputs:** fMRIPrep derivatives, confounds, and events tables stored under a BIDS derivative layout discoverable by the fMRI scripts.
- **Behavioral annotations:** Tabular files containing stimulus temperatures, pain ratings, and trial markers consumed by both EEG and fMRI pipelines.
- **Derivative storage:** Each script writes results under `derivatives/<modality>/` with timestamped subfolders for machine-learning artifacts.

Adhere to BIDS conventions when adding new modalities or behavioral measures—doing so ensures compatibility with existing loaders and QC summaries.

---

## Configuration
- `eeg_pipeline/eeg_config.yaml`
  - Defines input/output roots, montage templates, filtering parameters, ROI/groupings, ERP/TFR windows, and decoding models.
  - Centralizes all subject lists and behavioral file mappings.
- `fmri_pipeline/NPS/00_config.yaml`
  - Specifies BIDS directories, TR, smoothing kernels, nuisance regressors, canonical NPS templates, and derivative destinations.
- `config_loader.py`
  - Shared helper in both pipelines providing attribute-style access, environment validation (thread limits, Matplotlib backends), and consistent logging styles.

> ✅ Keep configuration files version-controlled; they document every analysis decision and enable exact reruns.

---

## EEG workflow
1. **Convert BrainVision recordings to BIDS**
   ```bash
   python eeg_pipeline/raw_to_bids.py --source_root /path/to/source --bids_root /path/to/bids
   ```
   - Normalizes channel labels, attaches standard montages, and populates BIDS metadata.
2. **Merge behavioral annotations with events**
   ```bash
   python eeg_pipeline/merge_behavior_to_events.py /path/to/bids/sub-*/eeg/sub-*_events.tsv
   ```
   - Performs strict trial matching, logging discrepancies and raising errors on mismatches.
3. **Sequential analyses**
   ```bash
   python eeg_pipeline/01_foundational_analysis.py --group all
   python eeg_pipeline/02_time_frequency_analysis.py --group all
   python eeg_pipeline/03_feature_engineering.py --subject 0001 --subject 0002
   python eeg_pipeline/04_behavior_feature_analysis.py --subject 0001 --subject 0002
   python eeg_pipeline/05_decode_pain_experience.py --subjects 0001 0002 --models elasticnet random_forest
   python eeg_pipeline/06_temporal_generalization.py --subjects 0001 0002 --group-average
   ```
   - Generates ERP/TFR visualizations, feature tables, decoding metrics, and QC plots stored under `derivatives/eeg/`.

Scripts embrace MNE best practices (filtering, referencing, artifact rejection) and reuse shared utility modules for ROI definitions, event alignment, and manifest logging.

---

## fMRI workflow
1. **Discover and validate inputs**
   ```bash
   python fmri_pipeline/NPS/01_discover_inputs.py
   ```
   - Confirms the presence of BOLD runs, masks, confounds, and events; tallies thermal stimulus distributions.
2. **Prepare confounds and design matrices**
   - Run `02_build_confounds_24HMP_outliers.py` and `03_build_design_matrices.py` to assemble motion regressors and HRF-convolved condition regressors.
3. **Fit the Neurologic Pain Signature GLM**
   - Execute scripts `04`–`06` to fit first-level GLMs, combine runs with fixed-effects models, and warp betas to the canonical NPS grid.
4. **Score, summarize, and visualize**
   ```bash
   python fmri_pipeline/NPS/07_score_nps_conditions.py
   python fmri_pipeline/NPS/09_subject_metrics.py
   python fmri_pipeline/NPS/10_group_stats.py
   python fmri_pipeline/NPS/11_plots.py
   python fmri_pipeline/NPS/12_qc_collation.py
   ```
   - Produces subject-level metrics, group contrasts, publication-ready figures, and QC packets.
5. **Optional analyses**
   - `08_optional_trials_glm_and_scoring.py` enables trial-wise betas for fine-grained temporal modeling.

Each step logs runtime provenance, including template hashes and model specifications, supporting rigorous replication.

---

## Cross-modal learning
```bash
python machine_learning/train_eeg_to_nps.py \
    --models elasticnet random_forest \
    --bands alpha beta gamma \
    --permutation-per-model 100
```
- Auto-discovers participants with both EEG features and NPS outputs, validates schema consistency, and merges aligned trials.
- Implements nested cross-validation with subject/run grouping to prevent leakage.
- Stores trained models, feature importances, permutation nulls, and reproducibility manifests (CLI arguments, git hash, package versions) under timestamped directories.
- Additional entry points provide alternative estimators:
  - `train_eeg_to_nps_svm.py` exposes radial-basis SVMs with Bayesian optimization of C/γ priors for nonlinear boundaries.
  - `train_eeg_to_nps_cnn.py` ingests time–frequency tensors and trains lightweight convolutional networks with early stopping and permutation testing.
- Shared helpers standardize manifest logging, seed control, and scoring (RMSE, MAE, permutation p-values) across algorithms so comparisons remain fair.

---

## Outputs & quality control
- **EEG derivatives:** ERP/TFR figures, channel diagnostics, feature TSV/JSON files, and detailed logs under `derivatives/sub-*/eeg/` and `derivatives/group/eeg/`.
- **fMRI derivatives:** Harmonized NPS betas, subject metrics, group statistics, and QC HTML/TSV bundles organized by participant.
- **Machine-learning artifacts:** Serialized estimators, validation metrics, permutation distributions, and manifest files for audit trails.
- **Logging:** Console and file logs are written for every script, capturing environment variables and warnings to streamline troubleshooting.

---

## Reproducibility tips
- Commit configuration changes alongside analysis updates to record parameter provenance.
- Fix random seeds via configuration options to replicate cross-validation splits and permutation tests.
- Limit threads (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) for deterministic numerical results on shared systems.
- Export environment manifests (e.g., `pip freeze`) when distributing results externally.

---

## Extending the pipelines
1. Prefer established neuroimaging libraries (MNE/mne-bids, Nilearn GLM, scikit-learn, PyTorch Lightning) over custom code.
2. Expose new functionality through the YAML configuration files to maintain compatibility across scripts.
3. Register additional derivatives in QC summaries and manifests to preserve traceability.
4. Reuse logging and manifest utilities to ensure new analyses remain audit-ready.

---

## Support & citation
- **Issues & feature requests:** Use the repository issue tracker to report bugs or propose enhancements. Include configuration snippets and log excerpts to accelerate triage.
- **Academic citation:** If this pipeline contributes to your work, cite the associated publication or acknowledge the repository in your methods section. Include references to the Neurologic Pain Signature (Wager et al., 2013) and MNE-Python (Gramfort et al., 2013) as appropriate.

For questions about adapting the pipelines to new paradigms or multimodal datasets, reach out via the project maintainers or submit a discussion thread.
