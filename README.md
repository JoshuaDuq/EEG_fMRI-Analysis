# EEG_fMRI-Analysis

## Purpose
This repository contains a production EEG and fMRI analysis stack for simultaneously recorded thermal pain experiments. The EEG pipeline covers ingestion of BrainVision recordings into BIDS, strict behavioral alignment, event-related potential (ERP) and time–frequency analyses, feature engineering, and supervised decoding. The fMRI workflow reproduces the Neurologic Pain Signature (NPS) general linear model (GLM) pipeline, harmonizes outputs to the NPS grid, and generates quality-control (QC) summaries. Machine-learning bridges connect trial-wise EEG spectral features with fMRI NPS beta responses under rigorous cross-validation and reproducibility logging.【F:eeg_pipeline/raw_to_bids.py†L36-L200】【F:eeg_pipeline/alignment_utils.py†L28-L188】【F:eeg_pipeline/01_foundational_analysis.py†L730-L858】【F:fmri_pipeline/NPS/01_discover_inputs.py†L1-L194】【F:machine_learning/train_eeg_to_nps.py†L1-L335】【F:machine_learning/train_eeg_to_nps.py†L900-L1341】

## Repository tour
- `eeg_pipeline/` – EEG BIDS conversion, centralized configuration, behavioral alignment, numbered analysis scripts, and helper utilities for ROI definitions, logging, and QC-ready derivative management.【F:eeg_pipeline/raw_to_bids.py†L381-L455】【F:eeg_pipeline/eeg_config.yaml†L1-L200】【F:eeg_pipeline/merge_behavior_to_events.py†L1-L200】【F:eeg_pipeline/io_utils.py†L1-L183】【F:eeg_pipeline/logging_utils.py†L1-L85】
- `fmri_pipeline/NPS/` – Stage-wise scripts implementing the NPS GLM, grid harmonization, scoring, QC collation, and plotting, all driven by a single YAML configuration loader.【F:fmri_pipeline/NPS/00_config.yaml†L1-L160】【F:fmri_pipeline/NPS/config_loader.py†L1-L200】【F:fmri_pipeline/NPS/01_discover_inputs.py†L1-L194】【F:fmri_pipeline/NPS/06_harmonize_to_nps_grid.py†L1-L200】
- `machine_learning/` – Cross-modal modeling scripts that merge EEG features and NPS betas, perform nested cross-validation, persist models/metrics, and emit complete run manifests for reproducibility.【F:machine_learning/train_eeg_to_nps.py†L83-L335】【F:machine_learning/train_eeg_to_nps.py†L900-L1341】

## Quick start workflow
1. **Create an environment** with Python ≥3.10 and install the scientific stack used across the scripts:
   ```bash
   pip install mne mne-bids numpy scipy pandas matplotlib seaborn scikit-learn joblib pyyaml statsmodels
   pip install nilearn nibabel
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
   These packages back the EEG preprocessing (MNE/mne-bids), statistical analyses (statsmodels, scipy), Nilearn-based GLMs, and scikit-learn/PyTorch models referenced throughout the pipeline.【F:eeg_pipeline/01_foundational_analysis.py†L1-L120】【F:eeg_pipeline/02_time_frequency_analysis.py†L1-L120】【F:fmri_pipeline/NPS/03_build_design_matrices.py†L1-L80】【F:machine_learning/train_eeg_to_nps.py†L1-L134】
2. **Point the configuration files at your data** before running any scripts. Update `eeg_pipeline/eeg_config.yaml` with subject IDs, paths, ROI definitions, and analysis windows; customize `fmri_pipeline/NPS/00_config.yaml` for BIDS/fMRIPrep roots, task runs, confound columns, and contrasts.【F:eeg_pipeline/eeg_config.yaml†L7-L170】【F:fmri_pipeline/NPS/00_config.yaml†L8-L160】
3. **Convert EEG recordings to BIDS and align behavior:**
   ```bash
   python eeg_pipeline/raw_to_bids.py --source_root /path/to/source_data --bids_root eeg_pipeline/bids_output
   python eeg_pipeline/merge_behavior_to_events.py eeg_pipeline/bids_output/sub-0001/eeg/sub-0001_task-thermalactive_events.tsv
   ```
   The converter applies standard montages, normalizes events/channels, and writes sidecars, while the merge script enforces strict trial alignment using the centralized alignment utilities.【F:eeg_pipeline/raw_to_bids.py†L36-L455】【F:eeg_pipeline/merge_behavior_to_events.py†L33-L200】【F:eeg_pipeline/alignment_utils.py†L28-L188】
4. **Run EEG analyses sequentially** (single subjects, groups, or all cleaned epochs):
   ```bash
   python eeg_pipeline/01_foundational_analysis.py --group all
   python eeg_pipeline/02_time_frequency_analysis.py --group all
   python eeg_pipeline/03_feature_engineering.py --subject 0001 --subject 0002
   python eeg_pipeline/04_behavior_feature_analysis.py --subject 0001 --subject 0002
   python eeg_pipeline/05_decode_pain_experience.py --subjects 0001 0002 --models elasticnet random_forest
   python eeg_pipeline/06_temporal_generalization.py --subjects 0001 0002 --group-average
   ```
   Each script shares configuration, logging, and strict alignment helpers, producing subject-level plots, TSV/JSON exports, and group summaries under `bids_output/derivatives`.【F:eeg_pipeline/01_foundational_analysis.py†L730-L858】【F:eeg_pipeline/02_time_frequency_analysis.py†L1-L160】【F:eeg_pipeline/03_feature_engineering.py†L1-L120】【F:eeg_pipeline/04_behavior_feature_analysis.py†L1-L120】【F:eeg_pipeline/05_decode_pain_experience.py†L1-L160】【F:eeg_pipeline/06_temporal_generalization.py†L1-L120】【F:eeg_pipeline/logging_utils.py†L25-L79】
5. **Execute the fMRI NPS pipeline** once fMRIPrep outputs and events exist:
   ```bash
   python fmri_pipeline/NPS/01_discover_inputs.py
   python fmri_pipeline/NPS/02_build_confounds_24HMP_outliers.py
   python fmri_pipeline/NPS/03_build_design_matrices.py
   python fmri_pipeline/NPS/04_fit_first_level_glm.py
   python fmri_pipeline/NPS/05_combine_runs_fixed_effects.py
   python fmri_pipeline/NPS/06_harmonize_to_nps_grid.py
   python fmri_pipeline/NPS/07_score_nps_conditions.py
   python fmri_pipeline/NPS/08_optional_trials_glm_and_scoring.py  # optional
   python fmri_pipeline/NPS/09_subject_metrics.py
   python fmri_pipeline/NPS/10_group_stats.py
   python fmri_pipeline/NPS/11_plots.py
   python fmri_pipeline/NPS/12_qc_collation.py
   ```
   Scripts draw configuration from `00_config.yaml`, validate inputs, fit first-level models with Nilearn, harmonize to the NPS template, and collate QC tables/figures.【F:fmri_pipeline/NPS/00_config.yaml†L8-L160】【F:fmri_pipeline/NPS/config_loader.py†L14-L159】【F:fmri_pipeline/NPS/01_discover_inputs.py†L1-L194】【F:fmri_pipeline/NPS/04_fit_first_level_glm.py†L1-L160】【F:fmri_pipeline/NPS/06_harmonize_to_nps_grid.py†L1-L200】【F:fmri_pipeline/NPS/07_score_nps_conditions.py†L1-L160】【F:fmri_pipeline/NPS/12_qc_collation.py†L1-L200】
6. **Link EEG features to fMRI NPS scores** using the machine-learning bridge:
   ```bash
   python machine_learning/train_eeg_to_nps.py --models elasticnet random_forest --bands alpha beta gamma --permutation-per-model 100
   ```
   The script aligns EEG derivative features with NPS beta scores, performs nested cross-validation, optionally runs permutation tests, and saves models, predictions, metrics, and manifests in timestamped output folders.【F:machine_learning/train_eeg_to_nps.py†L83-L335】【F:machine_learning/train_eeg_to_nps.py†L900-L1341】

## Configuration and data organization
- **EEG configuration (`eeg_config.yaml`)** defines project paths, subject lists, strict alignment behavior, ERP/TFR parameters, ROI regexes, feature windows, and decoding options. Update baseline/plateau windows, ROI groups, and derivative directories here to propagate across all scripts.【F:eeg_pipeline/eeg_config.yaml†L7-L173】【F:eeg_pipeline/config_loader.py†L200-L320】
- **fMRI configuration (`00_config.yaml`)** captures BIDS and fMRIPrep directories, run structure, acquisition parameters (TR, slice timing), GLM design (HRF model, contrasts, nuisance regressors), and template resources. Adjust temperature labels, motion model, and output roots before launching the fMRI scripts.【F:fmri_pipeline/NPS/00_config.yaml†L8-L160】
- **Central loaders** resolve relative paths, apply thread limits, and standardize matplotlib styling for headless execution, ensuring consistent behavior across notebooks, scripts, and schedulers.【F:eeg_pipeline/config_loader.py†L200-L236】【F:fmri_pipeline/NPS/config_loader.py†L36-L107】
- **Derivative layout**: EEG utilities guarantee BIDS-compliant derivative metadata and locate cleaned epochs/events, while logging helpers write per-subject and group logs into `derivatives/sub-*/eeg/logs` and `derivatives/group/eeg/logs`.【F:eeg_pipeline/io_utils.py†L39-L183】【F:eeg_pipeline/logging_utils.py†L25-L79】

## EEG pipeline details
### Ingestion & behavioral alignment
- `raw_to_bids.py` scans BrainVision files, applies montages, normalizes annotations/channels, writes dataset descriptions, and exposes CLI options for trimming to first MRI volume, filtering trigger prefixes, and zero-basing onsets.【F:eeg_pipeline/raw_to_bids.py†L36-L455】
- `merge_behavior_to_events.py` locates PsychoPy `TrialSummary.csv` files, matches runs, enforces strict trial matching, and appends behavioral columns to BIDS events, raising errors on mismatches to prevent label drift.【F:eeg_pipeline/merge_behavior_to_events.py†L33-L200】【F:eeg_pipeline/alignment_utils.py†L28-L188】

### Analysis stages (`01`–`06`)
1. **Foundational ERP analysis** builds subject and group ERPs, temperature contrasts, and trial count summaries with configurable cropping and logging, supporting `--subject`, `--group`, and `--all-subjects` workflows.【F:eeg_pipeline/01_foundational_analysis.py†L730-L858】
2. **Time–frequency analysis** computes Morlet TFRs, ROI averages, permutation/FDR statistics, and topographies using centralized band definitions, baseline windows, and ROI masks from the YAML config.【F:eeg_pipeline/02_time_frequency_analysis.py†L1-L160】【F:eeg_pipeline/eeg_config.yaml†L59-L173】
3. **Feature engineering** aggregates plateau band power, exports TSV/JSON sidecars, and prepares machine-learning ready tables tied to the strict alignment utilities.【F:eeg_pipeline/03_feature_engineering.py†L1-L120】
4. **Behavioral feature analysis** correlates EEG features with ratings/temperatures and generates diagnostic plots via the shared ROI/time–frequency helpers.【F:eeg_pipeline/04_behavior_feature_analysis.py†L1-L120】
5. **Decode pain experience** evaluates elastic net, random forest, and logistic/ridge models under nested cross-validation, storing metrics, best parameters, and permutation importance in derivatives folders.【F:eeg_pipeline/05_decode_pain_experience.py†L1-L160】
6. **Temporal generalization** trains sliding-window classifiers/regressors to map the dynamics of pain decoding over time, outputting temporal transfer matrices for interpretation.【F:eeg_pipeline/06_temporal_generalization.py†L1-L120】

### Shared utilities
- `config_loader.py` exposes attribute-style access, environment/thread controls, matplotlib theming, and legacy constant exports so existing scripts and notebooks remain compatible.【F:eeg_pipeline/config_loader.py†L200-L320】
- `alignment_utils.py` strictly guards event-to-epoch alignment and behavioral trimming, refusing unsafe heuristics to maintain trial integrity across ERP, spectral, and decoding analyses.【F:eeg_pipeline/alignment_utils.py†L28-L188】
- `io_utils.py` discovers cleaned epochs, loads events with BIDSPath fallbacks, ensures derivatives metadata, and selects behavioral targets for downstream correlation scripts.【F:eeg_pipeline/io_utils.py†L39-L183】
- `logging_utils.py` instantiates subject/group loggers that emit to console and derivative log files, supporting reproducible auditing of each run.【F:eeg_pipeline/logging_utils.py†L25-L79】
- ROI utilities (`roi_utils.py`, `tfr_utils.py`) provide canonical channel naming, ROI masks, and adaptive Morlet cycle computation aligned with the YAML ROI definitions.【F:eeg_pipeline/02_time_frequency_analysis.py†L1-L60】【F:eeg_pipeline/roi_utils.py†L1-L160】【F:eeg_pipeline/tfr_utils.py†L1-L120】

## fMRI NPS pipeline details
1. **Input discovery (`01_discover_inputs.py`)** inventories BOLD, masks, confounds, and events per run, checks trial counts and temperature balance, and writes QC TSV/JSON summaries for downstream stages.【F:fmri_pipeline/NPS/01_discover_inputs.py†L1-L194】
2. **Confound modeling (`02_build_confounds_24HMP_outliers.py`)** assembles the 24-parameter motion model with outlier censoring specified in the config.【F:fmri_pipeline/NPS/00_config.yaml†L104-L160】
3. **Design matrices (`03_build_design_matrices.py`)** construct condition-wise regressors using Nilearn with TR, HRF, and nuisance events drawn from the YAML configuration.【F:fmri_pipeline/NPS/03_build_design_matrices.py†L1-L120】【F:fmri_pipeline/NPS/00_config.yaml†L72-L160】
4. **First-level GLM (`04_fit_first_level_glm.py`)** fits per-run models, outputs beta maps and residuals, and logs fit diagnostics.【F:fmri_pipeline/NPS/04_fit_first_level_glm.py†L1-L160】
5. **Fixed-effects combine (`05_combine_runs_fixed_effects.py`)** aggregates run-level betas into subject-level contrasts for each condition.【F:fmri_pipeline/NPS/05_combine_runs_fixed_effects.py†L1-L160】
6. **Harmonization (`06_harmonize_to_nps_grid.py`)** resamples subject maps to the canonical NPS voxel grid with provenance metadata for reproducibility.【F:fmri_pipeline/NPS/06_harmonize_to_nps_grid.py†L1-L200】
7. **NPS scoring (`07_score_nps_conditions.py`)** computes similarity metrics and trial-level scores, writing TSV outputs compatible with the machine-learning bridge.【F:fmri_pipeline/NPS/07_score_nps_conditions.py†L1-L160】
8. **Optional trial-wise GLM (`08_optional_trials_glm_and_scoring.py`)** provides trial-level beta estimation when finer temporal resolution is required.【F:fmri_pipeline/NPS/08_optional_trials_glm_and_scoring.py†L1-L160】
9. **QC & reporting (`09_subject_metrics.py`, `10_group_stats.py`, `11_plots.py`, `12_qc_collation.py`)** summarize subject metrics, perform second-level stats, generate publication-ready figures, and collate QC artifacts plus environment metadata.【F:fmri_pipeline/NPS/09_subject_metrics.py†L1-L160】【F:fmri_pipeline/NPS/10_group_stats.py†L1-L160】【F:fmri_pipeline/NPS/11_plots.py†L1-L200】【F:fmri_pipeline/NPS/12_qc_collation.py†L1-L200】

## Machine-learning EEG→NPS bridge
- The main script discovers subjects with both EEG features (`features_eeg_direct.tsv`) and NPS scores (`trial_br.tsv`), validates feature consistency, and concatenates aligned trials.【F:machine_learning/train_eeg_to_nps.py†L144-L200】【F:machine_learning/train_eeg_to_nps.py†L900-L1018】
- Nested cross-validation leverages subject-wise or run-wise grouping, logs feature-to-sample ratios, and warns about potential overfitting.【F:machine_learning/train_eeg_to_nps.py†L1002-L1106】
- For each model, the pipeline exports predictions, per-subject/per-temperature metrics, CV fold details, JSON metrics, best parameters, and optional permutation test summaries.【F:machine_learning/train_eeg_to_nps.py†L1108-L1196】
- The best-performing model is refit on all data, saved as a Joblib artifact alongside feature importances, final predictions, and summary JSON capturing dataset sizes, CV strategies, and temperature distributions.【F:machine_learning/train_eeg_to_nps.py†L1204-L1338】
- Every run writes a manifest containing CLI arguments, environment details, git metadata, and package versions to aid reproducibility audits.【F:machine_learning/train_eeg_to_nps.py†L200-L281】【F:machine_learning/train_eeg_to_nps.py†L1323-L1336】

## Outputs, QC, and reproducibility
- EEG derivatives automatically include `dataset_description.json`, structured logs, plots, and TSV summaries under subject and group folders, facilitating downstream QC and manuscript figures.【F:eeg_pipeline/io_utils.py†L159-L183】【F:eeg_pipeline/logging_utils.py†L25-L79】
- fMRI stages emit QC TSVs, harmonized NPS grids, and figure panels while validating input completeness up front to prevent silent failures.【F:fmri_pipeline/NPS/01_discover_inputs.py†L12-L194】【F:fmri_pipeline/NPS/12_qc_collation.py†L1-L200】
- Machine-learning runs snapshot metrics, permutation tests, and manifests so that modeling choices remain transparent and rerunnable.【F:machine_learning/train_eeg_to_nps.py†L1108-L1336】

## Best practices for extending the repo
1. Rely on validated neuroimaging toolkits (MNE/mne-bids, Nilearn, scikit-learn, PyTorch) already used throughout the pipeline instead of custom reimplementations.【F:eeg_pipeline/01_foundational_analysis.py†L1-L120】【F:fmri_pipeline/NPS/04_fit_first_level_glm.py†L1-L160】【F:machine_learning/train_eeg_to_nps.py†L1-L134】
2. Add configuration toggles in the YAML files rather than hard-coding parameters; both loaders expose attribute access and propagate values repository-wide.【F:eeg_pipeline/eeg_config.yaml†L7-L173】【F:eeg_pipeline/config_loader.py†L200-L320】【F:fmri_pipeline/NPS/config_loader.py†L36-L159】
3. Ensure new outputs land in BIDS-compliant derivative folders and register them in logging/QC summaries for traceability.【F:eeg_pipeline/io_utils.py†L39-L183】【F:fmri_pipeline/NPS/12_qc_collation.py†L1-L200】
4. When contributing modeling scripts, reuse the manifest, logging, and nested CV utilities from `train_eeg_to_nps.py` to maintain reproducibility standards.【F:machine_learning/train_eeg_to_nps.py†L83-L335】【F:machine_learning/train_eeg_to_nps.py†L900-L1336】

