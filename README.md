# EEG Analysis Pipeline: Comprehensive Analysis Suite

This repository contains a comprehensive EEG analysis pipeline built on top of MNE-Python and MNE-BIDS for pain experience research. It provides command-line tools for the complete analysis workflow from raw data to advanced machine learning decoding:

- **Data Preparation**: Convert BrainVision EEG recordings into BIDS format and merge behavioral data
- **Quality Control**: Foundational QC plots and event-related potential (ERP) analyses  
- **Time-Frequency Analysis**: Comprehensive spectral power analysis with baseline correction
- **Feature Engineering**: Extract ML-ready features from EEG power and connectivity
- **Behavioral Analysis**: Correlate EEG features with behavioral measures
- **Pain Decoding**: Advanced machine learning models with rigorous cross-validation and statistical validation

All scripts live in `eeg_pipeline/` and write outputs into a BIDS-style tree under `eeg_pipeline/bids_output/` and its `derivatives/` subfolder.


## Contents
- CLI tools
  - `raw_to_bids.py` - Convert raw EEG to BIDS format
  - `merge_behavior_to_events.py` - Merge behavioral data into BIDS events
  - `01_foundational_analysis.py` - Quality control and ERP analysis
  - `02_time_frequency_analysis.py` - Time-frequency analysis and spectral power
  - `03_feature_engineering.py` - Extract ML-ready features from EEG data
  - `04_behavior_feature_analysis.py` - Behavioral correlations with EEG features
  - `05_decode_pain_experience.py` - Advanced pain decoding with ML models
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


### 3) Foundational QC and ERP Analysis: `eeg_pipeline/01_foundational_analysis.py`

Purpose: Comprehensive quality control assessment and event-related potential (ERP) analysis for individual subjects. Performs statistical validation of data quality and computes condition-specific neural responses.

Usage:

```powershell
python eeg_pipeline/01_foundational_analysis.py --subject 001 --task thermalactive
```

Arguments:
- `--subject, -s` (str): BIDS subject label without the `sub-` prefix (e.g., `001`).
- `--task, -t` (str): BIDS task label. Default: from `config.py` or `thermalactive`.

**Detailed Analysis Pipeline:**

**1. Data Loading and Validation:**
- Loads cleaned epochs from `derivatives/sub-<ID>/eeg/*proc-clean*_epo.fif` with multiple fallback patterns
- Loads corresponding BIDS `events.tsv` and performs length alignment with epochs
- Attaches behavioral metadata to epochs object for condition-specific analyses
- Validates data integrity and reports trial counts per condition

**2. Quality Control Analyses:**
- **Power Spectral Density (PSD)**: Computes and plots frequency domain characteristics using Welch's method
- **Sensor Layout Visualization**: Displays electrode montage and spatial configuration
- **Drop Log Analysis**: Quantifies and visualizes trial rejection patterns across conditions
- **Trial Image Plots**: Generates epoch-by-epoch amplitude visualizations for artifact detection

**3. Event-Related Potential (ERP) Analyses:**

**Pain vs Non-Pain Contrasts:**
- Identifies pain coding column (`pain_binary_coded`, `pain_binary`, or `pain`)
- Computes condition-averaged ERPs with standard error estimation
- Performs cluster-based permutation tests for statistical significance (if available)
- Generates butterfly plots showing all channels and topographic maps at peak latencies

**Temperature-Specific ERPs:**
- Auto-detects temperature column (`stimulus_temp`, `stimulus_temperature`, `temp`, `temperature`)
- Computes separate ERPs for each temperature level
- Calculates Global Field Power (GFP) for each temperature condition
- Performs statistical comparison of GFP across temperature levels
- Generates multi-panel butterfly plots and GFP time series

**4. Statistical Summaries:**
- **Trial Count Analysis**: Computes and saves condition-wise trial counts with statistical adequacy assessment
- **Amplitude Statistics**: Peak amplitude extraction and latency analysis for key ERP components
- **Condition Balance**: Evaluates design balance across experimental conditions

Outputs (subject-specific in `eeg_pipeline/bids_output/derivatives/sub-<ID>/eeg/plots/`):

**Quality Control Outputs:**
- `qc_psd.png` - Power spectral density across all channels
- `qc_sensor_layout.png` - Electrode montage visualization
- `qc_drop_log.png` - Trial rejection patterns
- `qc_trial_images.png` - Epoch-by-epoch amplitude visualization

**ERP Analysis Outputs:**
- `erp_pain_binary_butterfly.png` - Pain vs non-pain butterfly plots
- `erp_pain_binary_topomaps.png` - Topographic maps at key latencies
- `erp_by_temperature_gfp.png` - Global field power comparison across temperatures
- `erp_temperature_<temp>.png` - Individual temperature condition ERPs

**Statistical Summaries:**
- `counts_pain.tsv` - Trial counts per pain condition with adequacy metrics
- `counts_temperature.tsv` - Trial counts per temperature with balance assessment
- `erp_peak_amplitudes.tsv` - Peak amplitude and latency measurements (if computed)


### 4) Time-Frequency Analysis: `eeg_pipeline/02_time_frequency_analysis.py`

Purpose: Comprehensive spectral power analysis using Morlet wavelets with rigorous baseline correction and statistical contrasts. Computes time-frequency representations (TFRs) for oscillatory activity analysis across experimental conditions.

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

**Detailed Analysis Pipeline:**

**1. Time-Frequency Decomposition:**
- **Morlet Wavelet Transform**: Uses `mne.time_frequency.tfr_morlet` with logarithmically-spaced frequencies from 4–100 Hz
- **Wavelet Parameters**: `n_cycles = freqs/2` providing optimal time-frequency resolution trade-off
- **Temporal Resolution**: Maintains trial-by-trial decomposition for single-trial analyses
- **Frequency Bands**: Automatic segmentation into theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-100 Hz)

**2. Baseline Correction:**
- **Method**: Log-ratio baseline correction (`mode='logratio'`) for relative power changes
- **Baseline Window**: Pre-stimulus period `(None, 0.0)` seconds
- **Statistical Rationale**: Log-ratio provides interpretable percent change from baseline and normalizes across frequency bands

**3. Spatial Analysis Framework:**

**Single-Channel Analysis (Cz):**
- **Rationale**: Central electrode for maximal sensorimotor representation
- **Condition Contrasts**: Pain vs non-pain, temperature-specific responses
- **Statistical Visualization**: Difference plots with significance masking (if implemented)

**Region-of-Interest (ROI) Analysis:**
- **ROI Definitions**: Anatomically-defined regions using 10-10 electrode name matching:
  - Frontal: `F.*` pattern (executive/cognitive processing)
  - Central: `C.*` pattern (sensorimotor cortex)
  - Parietal: `P.*` pattern (somatosensory integration)
  - Occipital: `O.*` pattern (visual processing)
  - Temporal: `T.*` pattern (temporal lobe functions)
  - Sensorimotor: Combined C3, Cz, C4 (primary sensorimotor)
- **Spatial Averaging**: Within-ROI averaging for noise reduction and anatomical specificity

**4. Topographic Analysis:**
- **Plateau Window Averaging**: Time-averaged power within user-defined plateau window (default 0.5-8.0s)
- **Frequency-Band Specific Maps**: Separate topographies for each frequency band
- **Condition Contrasts**: Pain vs non-pain topographic differences
- **Statistical Mapping**: Electrode-wise contrast visualization

**5. Experimental Condition Analysis:**

**Pooled Analysis:**
- **All-Trial TFRs**: Grand-averaged spectral power across all experimental conditions
- **Pain Contrasts**: Binary pain vs non-pain comparisons with statistical difference maps
- **ROI-Specific Spectral Profiles**: Band-limited power analysis per anatomical region

**Temperature-Stratified Analysis:**
- **Temperature Detection**: Auto-identification of temperature column in metadata
- **Condition-Specific TFRs**: Separate analysis for each temperature level
- **Comparative Visualization**: Multi-panel displays for temperature-dependent effects
- **Filesystem Organization**: Temperature-specific subdirectories with sanitized naming

Outputs (subject-specific):

**Pooled Analysis** (`eeg_pipeline/bids_output/derivatives/sub-<ID>/eeg/plots/`):
- `tfr_Cz_all_trials_baseline_logratio.png` - Central electrode TFR
- `tfr_Cz_pain_vs_nonpain_baseline_logratio.png` - Pain contrast at Cz
- `tfr_Cz_difference_pain_minus_nonpain_baseline_logratio.png` - Statistical difference map
- `tfr_ROI-<ROI>_all_trials_baseline_logratio.png` - ROI-averaged TFRs
- `tfr_ROI-<ROI>_<band>_baseline_logratio.png` - Band-specific ROI analysis
- `topomap_<band>_all_trials_baseline_logratio.png` - Topographic power maps
- `topomap_<band>_pain_vs_nonpain_baseline_logratio.png` - Condition contrast topographies
- `topomap_ROI-<ROI>_<band>_difference_baseline_logratio.png` - ROI-specific topographic contrasts

**Temperature-Stratified Analysis** (`eeg_pipeline/bids_output/derivatives/sub-<ID>/eeg/plots/temperature/temp-<label>/`):
- Same analysis suite replicated for each temperature condition
- Temperature labels sanitized for filesystem compatibility (e.g., `47.3°C → temp-47p3`)

**Technical Specifications:**
- **Frequency Resolution**: Logarithmic spacing optimizes coverage across frequency ranges
- **Temporal Resolution**: Maintains millisecond precision for event-related dynamics
- **Statistical Thresholding**: Implements cluster-based correction for multiple comparisons (where applicable)
- **Memory Optimization**: Per-trial computation with efficient memory management for large datasets


### 5) Feature Engineering: `eeg_pipeline/03_feature_engineering.py`

Purpose: Systematic extraction of machine learning-ready features from EEG time-frequency data and functional connectivity matrices. Implements standardized feature engineering pipeline for pain prediction modeling.

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

**Detailed Feature Engineering Pipeline:**

**1. Data Loading and Preprocessing:**
- Loads cleaned epochs from `derivatives/sub-<ID>/eeg/*proc-clean*_epo.fif` with fallback patterns
- Loads corresponding BIDS `events.tsv` and performs trial-wise alignment
- Validates data integrity and handles missing trials with appropriate warnings
- Implements robust error handling for corrupted or incomplete datasets

**2. Time-Frequency Feature Extraction:**

**Spectral Power Features:**
- **Morlet Wavelet Decomposition**: In-memory computation using `mne.time_frequency.tfr_morlet`
- **Frequency Bands**: 
  - Alpha: 8.0-12.9 Hz (posterior alpha rhythm, attention/arousal)
  - Beta: 13.0-30.0 Hz (sensorimotor rhythms, motor cortex activity)
  - Gamma: 30.1-80.0 Hz (high-frequency activity, pain processing)
- **Temporal Averaging**: Mean power within plateau window (default 3.0-10.5s post-stimulus)
- **Spatial Coverage**: All available EEG channels for comprehensive spatial sampling
- **Feature Naming**: Systematic naming convention `pow_<band>_<channel>` for traceability

**Baseline Correction:**
- **Method**: Log-ratio baseline correction for relative power changes
- **Baseline Window**: Pre-stimulus period for condition-independent normalization
- **Statistical Rationale**: Removes individual differences in absolute power while preserving relative changes

**3. Functional Connectivity Feature Extraction:**

**Connectivity Metrics (Optional):**
- **Amplitude Envelope Correlation (AEC)**: Measures amplitude coupling between regions
- **Weighted Phase Lag Index (wPLI)**: Quantifies phase coupling while minimizing volume conduction
- **Frequency-Band Specific**: Computed separately for each frequency band of interest
- **ROI-Based Analysis**: Region-to-region connectivity matrices for anatomical interpretability

**Matrix Flattening:**
- **Lower Triangle Extraction**: Utilizes symmetry to avoid redundant features
- **Feature Naming**: Systematic convention `<metric>_<band>_<roi_i>__<roi_j>` for pair identification
- **Dimensionality**: N*(N-1)/2 features per band per connectivity metric

**4. Behavioral Target Selection:**

**Hierarchical Target Selection:**
- **Primary Targets**: `vas_final_coded_rating`, `vas_final_rating` (continuous pain ratings)
- **Secondary Targets**: `vas_rating`, `pain_rating` (alternative continuous measures)
- **Fallback Targets**: Binary pain coding variables for classification tasks
- **Validation**: Ensures target variable has sufficient variance and valid range

**5. Data Alignment and Quality Control:**
- **Trial Count Alignment**: Trims all feature matrices to minimum trial count across modalities
- **Missing Data Handling**: Reports and handles missing trials with appropriate strategies
- **Feature Matrix Validation**: Ensures consistent dimensionality and data types
- **Quality Metrics**: Computes and reports feature distribution statistics

**6. Output Generation:**

**Feature Matrix Organization:**
- **Standardized Format**: Tab-separated values (TSV) for cross-platform compatibility
- **Header Documentation**: Descriptive column names for feature interpretation
- **Metadata Preservation**: Links features to original data sources and processing parameters

Outputs (subject-specific) written to `eeg_pipeline/bids_output/derivatives/sub-<ID>/eeg/features/`:

**Direct EEG Features:**
- `features_eeg_direct.tsv` — Power features with headers like `pow_alpha_Cz`, `pow_beta_F3`
- `features_eeg_direct_columns.tsv` — Column metadata for feature traceability

**Connectivity Features (if available):**
- `features_connectivity.tsv` — Flattened connectivity matrices with systematic naming
- Feature names: `aec_alpha_<roi_i>__<roi_j>`, `wpli_beta_<roi_i>__<roi_j>`

**Combined Features:**
- `features_all.tsv` — Concatenated feature matrix combining power and connectivity
- Maintains column ordering for consistent downstream analysis

**Behavioral Targets:**
- `target_vas_ratings.tsv` — Selected behavioral target variable
- Includes metadata about target selection rationale

**Technical Specifications:**
- **Memory Efficiency**: Streaming computation for large datasets
- **Numerical Precision**: Float64 precision for downstream statistical analysis
- **Missing Value Handling**: Explicit NaN handling with documentation
- **Scalability**: Designed for batch processing across multiple subjects

**Connectivity Input Requirements:**
- **File Patterns**: `sub-<ID>_task-<task>_*connectivity_<metric>_<band>*_all_trials.npy`
- **Label Files**: Optional `*_labels.npy` for ROI identification
- **Data Format**: NumPy arrays with trials × ROI × ROI structure
- **Validation**: Automatic detection and validation of connectivity data availability


### 6) Behavioral Feature Analysis: `eeg_pipeline/04_behavior_feature_analysis.py`

Purpose: Comprehensive statistical analysis of brain-behavior relationships through correlation analysis between EEG-derived features and behavioral pain measures. Implements rigorous statistical testing with multiple comparison corrections.

Usage:

```powershell
# Single subject analysis
python eeg_pipeline/04_behavior_feature_analysis.py --subjects 001 --task thermalactive

# Multiple subjects
python eeg_pipeline/04_behavior_feature_analysis.py --subjects 001 002 003 --task thermalactive
```

Arguments:
- `--subjects` (list[str]): Subject labels without the `sub-` prefix.
- `--task` (str): BIDS task label. Default from `config.py` or `thermalactive`.

**Detailed Analysis Pipeline:**

**1. Data Integration and Preprocessing:**
- **Feature Loading**: Imports EEG power and connectivity features from feature engineering outputs
- **Behavioral Target Loading**: Loads corresponding behavioral pain ratings and experimental metadata
- **Data Validation**: Ensures trial-wise alignment between neural features and behavioral measures
- **Quality Control**: Identifies and handles missing data, outliers, and data integrity issues

**2. Correlation Analysis Framework:**

**Pearson Product-Moment Correlations:**
- **Assumptions**: Tests for linearity and normality of feature-behavior relationships
- **Application**: Quantifies linear associations between EEG features and continuous pain ratings
- **Interpretation**: Provides effect size estimates for linear brain-behavior relationships

**Spearman Rank-Order Correlations:**
- **Non-parametric Approach**: Robust to non-linear monotonic relationships and outliers
- **Rank-Based Analysis**: Evaluates ordinal associations independent of distribution assumptions
- **Complementary Information**: Captures relationships missed by parametric approaches

**3. Multiple Comparison Correction:**
- **False Discovery Rate (FDR)**: Benjamini-Hochberg procedure for controlling expected proportion of false discoveries
- **Family-Wise Error Rate (FWER)**: Bonferroni correction for strict Type I error control
- **Statistical Power**: Balances sensitivity and specificity in high-dimensional feature space

**4. Spatial and Spectral Analysis:**

**Channel-Specific Analysis:**
- **Topographic Mapping**: Identifies spatial patterns of brain-behavior correlations
- **Anatomical Interpretation**: Links significant correlations to known pain processing regions
- **Electrode-Wise Statistics**: Provides fine-grained spatial resolution of effects

**Frequency Band Analysis:**
- **Alpha Band (8-13 Hz)**: Attention, arousal, and posterior cortical activity
- **Beta Band (13-30 Hz)**: Sensorimotor processing and motor cortex engagement
- **Gamma Band (30-80 Hz)**: High-frequency processing and pain-specific neural activity
- **Cross-Band Comparisons**: Identifies frequency-specific vs broadband effects

**5. Connectivity-Behavior Relationships (if available):**

**Network-Level Analysis:**
- **Inter-Regional Coupling**: Correlations between connectivity strength and pain perception
- **Network Topology**: Relationship between graph-theoretic measures and behavioral outcomes
- **Functional Integration**: How distributed brain networks relate to subjective pain experience

**6. Statistical Visualization and Reporting:**

**Correlation Matrices:**
- **Heatmap Visualization**: Color-coded correlation strength with significance indicators
- **Hierarchical Clustering**: Groups features by similarity in behavioral relationships
- **Effect Size Visualization**: Distinguishes statistical significance from practical significance

**Summary Statistics:**
- **Significant Correlations**: Lists all statistically significant brain-behavior relationships
- **Effect Size Distribution**: Characterizes the magnitude of observed associations
- **Reproducibility Metrics**: Assesses consistency across subjects and conditions

Outputs (subject-specific) written to `eeg_pipeline/bids_output/derivatives/sub-<ID>/eeg/behavior_analysis/`:

**Correlation Analysis:**
- `power_behavior_correlations.png` - Comprehensive heatmap of EEG power vs behavioral correlations
- `power_behavior_pearson_matrix.tsv` - Numerical correlation matrix (Pearson)
- `power_behavior_spearman_matrix.tsv` - Numerical correlation matrix (Spearman)
- `power_behavior_pvalues.tsv` - Uncorrected p-values for all correlations
- `power_behavior_pvalues_corrected.tsv` - FDR-corrected p-values

**Connectivity Analysis (if available):**
- `connectivity_behavior_correlations.png` - Connectivity-behavior correlation heatmap
- `connectivity_behavior_pearson_matrix.tsv` - Connectivity correlation matrix
- `connectivity_behavior_significant_edges.tsv` - Significant connectivity-behavior relationships

**Statistical Summaries:**
- `correlation_summary.tsv` - Comprehensive summary of significant correlations with effect sizes
- `frequency_band_summary.tsv` - Band-specific correlation statistics
- `spatial_correlation_summary.tsv` - Channel-wise correlation patterns
- `statistical_report.txt` - Detailed statistical analysis report with interpretation

**Technical Specifications:**
- **Statistical Thresholds**: α = 0.05 for significance testing with multiple comparison correction
- **Effect Size Metrics**: Cohen's conventions for correlation magnitude interpretation
- **Missing Data**: Pairwise deletion for correlation computation
- **Numerical Precision**: Float64 precision for statistical computations
- **Reproducibility**: Deterministic analysis pipeline for consistent results across runs


### 7) Advanced Pain Decoding: `eeg_pipeline/05_decode_pain_experience.py`

Purpose: State-of-the-art machine learning pipeline for decoding subjective pain experience from EEG features using rigorous cross-validation and advanced statistical validation methods. Implements multiple algorithms with comprehensive model diagnostics and bias-corrected statistical inference.

**Key Features & Recent Enhancements:**
- **Block-aware permutation importance** for Random Forest with subject-level blocking to respect hierarchical data structure
- **Partial correlation analysis** controlling for temperature, trial number, and subject mean ratings to isolate EEG-specific effects
- **BCa confidence intervals** using subject-level cluster bootstrap for bias-corrected and accelerated statistical inference
- **Residual diagnostics** with covariate analysis to detect systematic biases and model assumptions violations
- **Calibration curve analysis** with LOESS smoothing and cross-validation for probabilistic prediction assessment
- **Within-subject vs LOSO comparison** with paired statistical tests to quantify generalization gaps

Usage:

```powershell
# Decode all subjects discovered under derivatives (recommended)
python eeg_pipeline/05_decode_pain_experience.py --subjects all --n_jobs -1 --seed 42

# Decode a subset of subjects with custom parallelization
python eeg_pipeline/05_decode_pain_experience.py --subjects 001 002 003 --n_jobs 4 --seed 123
```

Arguments:
- `--subjects` (list[str] or `all`): Subject labels without the `sub-` prefix, or `all` to include all with features present.
- `--task` (str): BIDS task label. Default from `config.py` or `thermalactive`.
- `--n_jobs` (int): Parallel jobs for inner CV/grid search. Use `-1` for all cores.
- `--seed` (int): Random seed for reproducibility.
- `--outer_n_jobs` (int): Parallel jobs for outer LOSO folds (default: 1 to avoid nested parallelism).

**Detailed Analysis Pipeline:**

**1. Cross-Validation Framework:**

**Leave-One-Subject-Out (LOSO) Cross-Validation:**
- **Rationale**: Ensures generalization to completely unseen subjects, critical for clinical translation
- **Implementation**: Each subject serves as test set while all others form training set
- **Statistical Independence**: Prevents data leakage between training and testing phases
- **Nested CV**: Inner cross-validation for hyperparameter optimization within each LOSO fold

**Hyperparameter Optimization:**
- **Grid Search**: Exhaustive search over predefined parameter spaces
- **Inner CV Folds**: 5-fold cross-validation within training data for unbiased parameter selection
- **Performance Metric**: Negative mean squared error for regression optimization
- **Logging**: Per-fold optimal parameters saved to JSONL files for reproducibility

**2. Machine Learning Models:**

**Elastic Net Regression:**
- **Algorithm**: Linear model combining L1 (Lasso) and L2 (Ridge) regularization
- **Hyperparameters**: 
  - `alpha`: Overall regularization strength [0.01, 0.1, 1.0, 10.0, 100.0]
  - `l1_ratio`: Balance between L1/L2 penalties [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
- **Preprocessing**: StandardScaler for feature normalization
- **Feature Selection**: Automatic via L1 regularization (sparse solutions)
- **Interpretability**: Linear coefficients provide direct feature importance

**Random Forest Regression:**
- **Algorithm**: Ensemble of decision trees with bootstrap aggregating
- **Hyperparameters**:
  - `n_estimators`: Fixed at 100 trees for computational efficiency
  - `max_depth`: [3, 5, 8, None] controlling tree complexity
  - `max_features`: ['sqrt', 0.2, 0.5, 1.0] for feature sampling
  - `min_samples_leaf`: [1, 5, 10] for overfitting control
- **Parallelization**: `n_jobs=1` to avoid conflicts with outer parallelization
- **Feature Importance**: Block-aware permutation importance respecting subject structure

**Riemannian Regression (Optional):**
- **Algorithm**: Covariance-based classification using Riemannian geometry
- **Preprocessing**: Raw epochs → covariance matrices → tangent space projection
- **Manifold Learning**: Projects symmetric positive definite matrices to Euclidean space
- **Requirements**: PyRiemann package for Riemannian operations
- **Application**: Captures spatial covariance patterns in EEG data

**3. Advanced Statistical Validation:**

**Block-Aware Permutation Importance:**
- **Methodology**: Permutes features within subject blocks to respect hierarchical structure
- **Statistical Validity**: Prevents inflation of importance scores due to subject-specific patterns
- **Implementation**: Skips features constant within all subject blocks of test fold
- **Output**: Top 20 most important features with confidence intervals

**Partial Correlation Analysis:**
- **Incremental Validity**: Quantifies unique EEG contribution beyond confounding variables
- **Covariates Controlled**:
  - Temperature: Stimulus intensity effects
  - Trial number: Temporal drift and habituation
  - Subject mean rating: Individual response bias
- **Statistical Method**: Semi-partial correlation using residualization
- **Metrics**: `partial_r_given_temp_trial_subjectmean` and `delta_r2_incremental_multi`

**Bootstrap Confidence Intervals:**
- **Method**: Bias-corrected and accelerated (BCa) bootstrap
- **Clustering**: Subject-level resampling to respect hierarchical data structure
- **Jackknife Acceleration**: Bias correction using leave-one-subject-out estimates
- **Coverage**: 95% confidence intervals for all correlation metrics
- **Robustness**: Handles non-normal sampling distributions

**4. Model Diagnostics and Validation:**

**Residual Analysis:**
- **Systematic Bias Detection**: Plots residuals vs. temperature and trial number
- **Statistical Testing**: Spearman correlations to detect non-random patterns
- **Assumption Validation**: Checks for homoscedasticity and independence
- **Covariate Effects**: Identifies systematic prediction errors

**Calibration Analysis:**
- **Calibration Curves**: Plots predicted vs. actual pain ratings in binned format
- **LOESS Smoothing**: Locally weighted regression for smooth calibration curves
- **Cross-Validation**: 5-fold CV for optimal smoothing span selection
- **Reliability**: Assesses whether prediction confidence matches actual accuracy

**Within-Subject vs. Cross-Subject Comparison:**
- **Within-Subject CV**: 5-fold cross-validation within each subject's data
- **Generalization Gap**: Quantifies performance difference between within and across subjects
- **Paired Statistical Tests**: Wilcoxon signed-rank tests for performance comparisons
- **Clinical Relevance**: Informs about model's potential for personalized vs. population-level applications

**5. Baseline Models and Controls:**

**Naive Baselines:**
- **Global Mean**: Predicts overall mean pain rating for all trials
- **Subject Mean**: Predicts each subject's mean rating for their trials
- **Temperature Mean**: Predicts mean rating for each temperature condition
- **Statistical Comparison**: Establishes minimum performance thresholds

**6. Statistical Inference and Reporting:**

**Performance Metrics:**
- **Pearson Correlation**: Linear association between predicted and actual ratings
- **Spearman Correlation**: Rank-order association robust to outliers
- **R-squared**: Proportion of variance explained by the model
- **Explained Variance**: Alternative to R² handling negative values
- **Mean Absolute Error**: Average absolute prediction error
- **Root Mean Square Error**: Penalizes large prediction errors more heavily

**Effect Size Interpretation:**
- **Correlation Magnitudes**: Small (r=0.1), medium (r=0.3), large (r=0.5) effects
- **Clinical Significance**: Practical importance beyond statistical significance
- **Confidence Intervals**: Precision estimates for all effect sizes



Outputs (written to `eeg_pipeline/bids_output/derivatives/decoding/`):

**Predictions & Metrics:**
- `elasticnet_loso_predictions.tsv`, `rf_loso_predictions.tsv`, `riemann_loso_predictions.tsv`
- `elasticnet_per_subject_metrics.tsv`, `rf_per_subject_metrics.tsv`, `riemann_per_subject_metrics.tsv`
- `baseline_global_loso_predictions.tsv` - Naive baseline predictions

**Diagnostic Plots:**
- `plots/elasticnet_loso_actual_vs_predicted.png` - Scatter plots with regression lines
- `plots/rf_loso_actual_vs_predicted.png`
- `plots/rf_block_permutation_importance_top20.png` - Top 20 most important features
- `plots/rf_residuals_vs_temperature.png` - Residual analysis vs temperature
- `plots/rf_residuals_vs_trial_number.png` - Residual analysis vs trial number
- `plots/rf_calibration_curve.png` - Calibration analysis with LOESS smoothing
- `plots/rf_within_vs_loso_combined.png` - Within-subject vs LOSO comparison

**Statistical Summaries:**
- `summary.json` - Comprehensive metrics including partial correlations and bootstrap CIs
- `best_params_elasticnet.jsonl` - Hyperparameters per fold for Elastic Net
- `best_params_rf.jsonl` - Hyperparameters per fold for Random Forest

**Key Metrics in Summary:**
- `partial_r_given_temp_trial_subjectmean` - Partial correlation controlling for multiple covariates
- `delta_r2_incremental_multi` - Unique variance beyond temperature, trial, and subject effects
- Bootstrap confidence intervals for all correlation metrics
- Explained variance scores and traditional regression metrics

**Technical Implementation Details:**

**Computational Efficiency:**
- **Memory Management**: Efficient handling of large feature matrices through streaming
- **Parallel Processing**: Inner CV parallelization with `n_jobs` parameter
- **Nested Parallelism Avoidance**: RF uses `n_jobs=1` to prevent thread conflicts
- **Progress Logging**: Detailed logging of fold-wise progress and timing

**Data Quality Assurance:**
- **Missing Data Handling**: Drops trials with NaN targets while retaining feature NaNs for imputation
- **Feature Validation**: Automatic detection and handling of constant features
- **Subject Validation**: Requires ≥2 subjects for LOSO cross-validation
- **Trial Count Reporting**: Logs trial counts per subject and condition

**Reproducibility:**
- **Random Seed Control**: Deterministic results across runs with seed parameter
- **Hyperparameter Logging**: Complete parameter sets saved per fold
- **Statistical Reporting**: Comprehensive metrics with confidence intervals
- **Version Control**: Compatible with standard ML pipeline versioning

**Statistical Robustness:**
- **Multiple Testing**: Appropriate corrections for high-dimensional feature spaces
- **Cross-Validation Stability**: Nested CV prevents overfitting to hyperparameters
- **Bootstrap Validity**: Subject-level clustering maintains statistical independence
- **Assumption Testing**: Diagnostic plots reveal model assumption violations


## Project Layout and Data Expectations

```
EEG_fMRI_Analysis/
├─ eeg_pipeline/
│  ├─ raw_to_bids.py                    # Convert BrainVision to BIDS
│  ├─ merge_behavior_to_events.py       # Merge behavioral data
│  ├─ 01_foundational_analysis.py       # QC and ERP analysis
│  ├─ 02_time_frequency_analysis.py     # Time-frequency analysis
│  ├─ 03_feature_engineering.py         # Extract ML features
│  ├─ 04_behavior_feature_analysis.py   # Behavioral correlations
│  ├─ 05_decode_pain_experience.py      # Advanced pain decoding
│  ├─ config.py                         # Optional configuration
│  ├─ source_data/
│  │  └─ sub-XXX/
│  │     ├─ eeg/                        # Raw BrainVision files (*.vhdr)
│  │     └─ PsychoPy_Data/              # Behavioral CSV (*TrialSummary.csv)
│  └─ bids_output/
│     ├─ sub-XXX/
│     │  └─ eeg/                        # Raw BIDS outputs
│     └─ derivatives/
│        ├─ sub-XXX/
│        │  └─ eeg/
│        │     ├─ plots/                # Subject-specific figures
│        │     ├─ features/             # ML-ready feature matrices
│        │     └─ behavior_analysis/    # Behavioral correlation outputs
│        └─ decoding/                   # Cross-subject decoding results
│           ├─ plots/                   # Decoding diagnostic plots
│           ├─ *_predictions.tsv        # Model predictions
│           ├─ *_per_subject_metrics.tsv # Per-subject performance
│           ├─ best_params_*.jsonl      # Hyperparameters per fold
│           └─ summary.json             # Overall performance metrics
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

### Workflow C: Complete analysis pipeline with behavioral correlations and pain decoding

```powershell
# 1) Build features for all desired subjects
python eeg_pipeline/03_feature_engineering.py --subjects 001 002 003 004 005 --task thermalactive

# 2) Analyze behavioral correlations with EEG features
python eeg_pipeline/04_behavior_feature_analysis.py --subjects 001 002 003 004 005 --task thermalactive

# 3) Run advanced LOSO pain decoding with comprehensive diagnostics
python eeg_pipeline/05_decode_pain_experience.py --subjects all --n_jobs -1 --seed 42
```

### Workflow D: Advanced decoding analysis workflow

```powershell
# Complete pipeline for pain decoding research
# 1) Prepare all subjects through feature engineering
python eeg_pipeline/03_feature_engineering.py --subjects all --task thermalactive

# 2) Run behavioral feature analysis to understand EEG-behavior relationships
python eeg_pipeline/04_behavior_feature_analysis.py --subjects all --task thermalactive

# 3) Execute comprehensive pain decoding with all advanced features:
#    - Block-aware permutation importance
#    - Partial correlation analysis controlling for covariates
#    - BCa confidence intervals with cluster bootstrap
#    - Residual diagnostics and calibration analysis
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

**Feature Engineering outputs** (under `eeg_pipeline/bids_output/derivatives/sub-<ID>/eeg/features/`):
- `features_eeg_direct.tsv` - Power features (alpha, beta, gamma per channel)
- `features_connectivity.tsv` - Connectivity features (AEC, wPLI if available)
- `features_all.tsv` - Combined feature matrix
- `target_vas_ratings.tsv` - Behavioral target values

**Behavioral Analysis outputs** (under `eeg_pipeline/bids_output/derivatives/sub-<ID>/eeg/behavior_analysis/`):
- `power_behavior_correlations.png` - EEG power vs behavior correlation heatmaps
- `connectivity_behavior_correlations.png` - Connectivity vs behavior correlations
- `correlation_summary.tsv` - Statistical summary of significant correlations

**Advanced Decoding outputs** (under `eeg_pipeline/bids_output/derivatives/decoding/`):
- **Predictions**: `elasticnet_loso_predictions.tsv`, `rf_loso_predictions.tsv`, `riemann_loso_predictions.tsv`
- **Metrics**: `elasticnet_per_subject_metrics.tsv`, `rf_per_subject_metrics.tsv`, `riemann_per_subject_metrics.tsv`
- **Baselines**: `baseline_global_loso_predictions.tsv` and corresponding metrics
- **Hyperparameters**: `best_params_elasticnet.jsonl`, `best_params_rf.jsonl` (per-fold optimal parameters)
- **Advanced Plots**:
  - `plots/rf_block_permutation_importance_top20.png` - Subject-aware feature importance
  - `plots/rf_residuals_vs_temperature.png` - Residual analysis vs temperature
  - `plots/rf_residuals_vs_trial_number.png` - Residual analysis vs trial effects
  - `plots/rf_calibration_curve.png` - Calibration analysis with LOESS smoothing
  - `plots/rf_within_vs_loso_combined.png` - Within-subject vs cross-subject comparison
- **Summary**: `summary.json` with comprehensive metrics including partial correlations and bootstrap CIs


## Troubleshooting

- Missing or misaligned events length: scripts will trim to the min length and warn. Ensure `events.tsv` matches the epochs count.
- No `pain_binary_coded` column: pain/non-pain contrasts will be skipped.
- No temperature column: per-temperature analyses will be skipped. Expected columns include `stimulus_temp`, `stimulus_temperature`, `temp`, or `temperature`.
- Runtime: TFR computation is the slowest step. Consider running pooled only or specific subjects first.
- Montage issues: If your channel names don’t match the template, `raw_to_bids.py` will continue without setting a montage.

**Advanced Decoding-specific troubleshooting:**
- **ConstantInputWarning or unrealistically high inner-CV scores**: Check for duplicated subjects with identical data causing leakage; deduplicate or ensure duplicates share the same group label so LOSO leaves them out together.
- **ElasticNet convergence warnings**: Already tuned; if they persist, increase `max_iter` or restrict the alpha grid to stronger regularization.
- **PyRiemann missing**: Install `pip install pyriemann` or run only ElasticNet/RF models.
- **Epochs channel picking requires preloading**: epochs are read with `preload=True`. If you adapt the code, ensure data are loaded before dropping/reordering channels.
- **Random Forest n_jobs conflicts**: RF uses `n_jobs=1` to avoid nested parallelism with outer CV. Use `--n_jobs` for inner CV parallelization instead.
- **Block permutation importance warnings**: Features that are constant within all subject blocks of a test fold are automatically skipped with logging.
- **Bootstrap confidence interval failures**: BCa intervals require sufficient bootstrap samples and may fail with very small datasets; fallback to percentile intervals is automatic.
- **LOESS calibration span selection**: Uses 5-fold CV to select optimal span; may take additional time but provides better calibration curve smoothing.
- **Memory usage with large datasets**: Consider reducing `bootstrap_n` in config or processing subjects in smaller batches for very large feature matrices.


---

If you want additional flags (e.g., temperature binning, custom frequency bands, or parallelization), open an issue or request and we can extend the CLIs accordingly.
