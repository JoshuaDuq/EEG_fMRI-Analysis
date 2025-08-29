# EEG Analysis Pipeline: Comprehensive Analysis Suite

This repository contains a comprehensive EEG analysis pipeline built on top of MNE-Python and MNE-BIDS for pain experience research. It provides command-line tools for the complete analysis workflow from raw data to advanced machine learning decoding:

- **Data Preparation**: Convert BrainVision EEG recordings into BIDS format and merge behavioral data
- **Event-Related Potential Analysis**: Foundational ERP analyses comparing pain vs non-pain conditions  
- **Time-Frequency Analysis**: Comprehensive spectral power analysis with baseline correction
- **Feature Engineering**: Extract ML-ready features from EEG power and connectivity
- **Behavioral Analysis**: Correlate EEG features with behavioral measures using advanced statistical methods
- **Pain Decoding**: State-of-the-art machine learning models with rigorous cross-validation and statistical validation

All scripts live in `eeg_pipeline/` and write outputs into a BIDS-style tree under `eeg_pipeline/bids_output/` and its `derivatives/` subfolder.


## Contents
- CLI tools
  - `raw_to_bids.py` - Convert raw EEG to BIDS format with flexible event filtering
  - `merge_behavior_to_events.py` - Merge behavioral data into BIDS events with per-run support
  - `01_foundational_analysis.py` - Event-related potential (ERP) analysis
  - `02_time_frequency_analysis.py` - Time-frequency analysis and spectral power
  - `03_feature_engineering.py` - Extract ML-ready features from EEG data
  - `04_behavior_feature_analysis.py` - Behavioral correlations with advanced statistics
  - `05_decode_pain_experience.py` - Advanced pain decoding with comprehensive diagnostics
  - `verify_decoding_outputs.py` - Validation tool for decoding pipeline outputs
  - `config.py` - Central configuration file
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
- Filters annotations to keep specified event prefixes (default: "Stim_on", supports multiple prefixes)
- Supports trimming EEG at first MRI volume trigger with `--trim_to_first_volume`
- Optionally zero-bases annotation onsets
- Writes BIDS subject-level folders and sidecars
- Optionally merges behavior (TrialSummary.csv) into resulting events.tsv

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
- `--source_root` (str): Path to `source_data` root containing `sub-*/eeg/*.vhdr`. Default: project `eeg_pipeline/source_data`
- `--bids_root` (str): Output BIDS root directory. Default: from `config.py` if present, else `eeg_pipeline/bids_output`
- `--task` (str): BIDS task label. Default: from `config.py` or `thermalactive`
- `--subjects` (list[str]): Optional list of subject labels to include, e.g., `001 002`. If omitted, all found are used
- `--montage` (str): Standard montage name for `mne.channels.make_standard_montage` (e.g., `easycap-M1`). Use empty string `""` to skip
- `--line_freq` (float): Line noise frequency metadata for sidecar (Hz). Default: from `config.py` (`zapline_fline`) or 60.0
- `--event_prefix` (str): Event annotation prefix to keep (repeatable for multiple prefixes). Default: "Stim_on"
- `--keep_all_annotations` (flag): Keep all annotations instead of filtering by prefix
- `--trim_to_first_volume` (flag): Trim EEG recording at first MRI volume trigger
- `--overwrite` (flag): Overwrite existing BIDS files
- `--merge_behavior` (flag): Merge Psychopy TrialSummary.csv into events after conversion
- `--zero_base_onsets` (flag): Zero-base kept annotation onsets

Outputs:
- BIDS dataset under `eeg_pipeline/bids_output/` (unless overridden by `--bids_root`).
- Creates `dataset_description.json` via MNE-BIDS.


### 2) Merge behavior into events: `eeg_pipeline/merge_behavior_to_events.py`

Purpose: Merge behavioral TrialSummary.csv columns into the BIDS `events.tsv` files with support for per-run data and flexible event selection.

Usage:

```powershell
# Dry-run (no writes) to see which columns would be merged
python eeg_pipeline/merge_behavior_to_events.py --dry_run

# Perform merge for a custom BIDS root and source_data root
python eeg_pipeline/merge_behavior_to_events.py --bids_root eeg_pipeline/bids_output --source_root eeg_pipeline/source_data --task thermalactive
```

Arguments:
- `--bids_root` (str): BIDS root containing `sub-*/eeg/*_events.tsv`. Default: from `config.py` if available
- `--source_root` (str): Source root containing `sub-*/PsychoPy_Data/*TrialSummary.csv`. Default: `eeg_pipeline/source_data`
- `--task` (str): Task label used in events filenames. Default: from `config.py` or `thermalactive`
- `--event_prefix` (str): Event prefix to target for behavioral merges (repeatable). Default: "Stim_on"
- `--event_type` (str): Specific event type to target (repeatable)
- `--dry_run` (flag): Print planned changes without writing

Notes:
- Only rows matching specified event prefixes/types are updated with behavioral columns
- Supports per-run TrialSummary matching (run-specific CSV selection based on BIDS run numbers)
- Length mismatches are trimmed to the shorter length with a warning

**Combined per-subject events (new):**
- After per-run merges, the script writes a combined file per subject at `sub-<ID>/eeg/sub-<ID>_task-<task>_events.tsv`.
- It concatenates all `*_task-<task>_run-*_events.tsv` for the subject, sorted by run (ascending) and by `onset` within each run (stable sort).
- If more than one run is present, the combined file includes a `run` column; if only a single run exists, `run` is omitted.
- Columns are aligned using the union across runs, preserving the order from the first run file; missing values are filled with NA.
- In `--dry_run` mode, combined files are not written (dry-run is for preview only).


### 3) Event-Related Potential Analysis: `eeg_pipeline/01_foundational_analysis.py`

Purpose: Event-related potential (ERP) analysis for individual subjects, computing condition-specific neural responses for pain vs non-pain conditions and temperature-specific analyses.

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

**2. Event-Related Potential (ERP) Analyses:**

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

**ERP Analysis Outputs:**
- `erp_pain_binary_butterfly.png` - Pain vs non-pain butterfly plots
- `erp_pain_binary_topomaps.png` - Topographic maps at key latencies
- `erp_by_temperature_gfp.png` - Global field power comparison across temperatures
- `erp_temperature_<temp>.png` - Individual temperature condition ERPs

**Statistical Summaries:**
- Trial count validation and condition balance assessment (logged during execution)
- Peak amplitude and latency measurements (if computed)


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


### 8) Pipeline Output Verification: `eeg_pipeline/verify_decoding_outputs.py`

Purpose: Validation tool for decoding pipeline outputs. Parses the configuration from the decoding script and verifies that expected output files exist with appropriate metadata.

Usage:

```powershell
# Verify all expected decoding outputs
python eeg_pipeline/verify_decoding_outputs.py
```

**Key Features:**
- **Configuration Parsing**: Extracts CONFIG dictionary from `05_decode_pain_experience.py` without importing
- **File Existence Validation**: Checks presence of all expected output files
- **Metadata Verification**: Validates file sizes, modification times, and JSON/JSONL structure
- **Riemann Band Expansion**: Automatically expands band-specific output patterns
- **Comprehensive Reporting**: Provides detailed status for all expected outputs

**Validation Coverage:**
- Prediction files (ElasticNet, Random Forest, Riemannian)
- Performance metrics per algorithm and per subject
- Hyperparameter logs (JSONL format)
- Statistical summaries and indices
- Diagnostic plots and figures
- Band-specific Riemannian outputs

Outputs:
- Console report of file existence, sizes, and validation status
- Identifies missing files and potential data corruption issues
- Useful for quality assurance and pipeline debugging


## Project Layout and Data Expectations

```
EEG_fMRI_Analysis/
├─ eeg_pipeline/
│  ├─ raw_to_bids.py                    # Convert BrainVision to BIDS with flexible filtering
│  ├─ merge_behavior_to_events.py       # Merge behavioral data with per-run support
│  ├─ 01_foundational_analysis.py       # Event-related potential (ERP) analysis
│  ├─ 02_time_frequency_analysis.py     # Time-frequency analysis and spectral power
│  ├─ 03_feature_engineering.py         # Extract ML-ready features from EEG data
│  ├─ 04_behavior_feature_analysis.py   # Behavioral correlations with advanced statistics
│  ├─ 05_decode_pain_experience.py      # Advanced pain decoding with comprehensive diagnostics
│  ├─ verify_decoding_outputs.py        # Validation tool for decoding pipeline outputs
│  ├─ config.py                         # Central configuration file
│  ├─ coll_lab_eeg_pipeline.py          # Alternative comprehensive pipeline
│  ├─ source_data/
│  │  ├─ Schaefer2018/                  # Atlas files for connectivity analysis
│  │  └─ sub-XXX/
│  │     ├─ eeg/                        # Raw BrainVision files (*.vhdr, *.eeg, *.vmrk)
│  │     └─ PsychoPy_Data/              # Behavioral CSV (*TrialSummary.csv)
│  └─ bids_output/
│     ├─ sub-XXX/
│     │  └─ eeg/                        # Raw BIDS outputs (events.tsv, EEG data)
│     └─ derivatives/
│        ├─ sub-XXX/
│        │  └─ eeg/
│        │     ├─ plots/                # Subject-specific figures
│        │     │  └─ temperature/       # Temperature-specific subdirectories
│        │     ├─ features/             # ML-ready feature matrices
│        │     └─ behavior_analysis/    # Behavioral correlation outputs
│        └─ decoding/                   # Cross-subject decoding results
│           ├─ plots/                   # Decoding diagnostic plots
│           ├─ indices/                 # Cross-validation indices
│           ├─ *_predictions.tsv        # Model predictions (ElasticNet, RF, Riemann)
│           ├─ *_per_subject_metrics.tsv # Per-subject performance metrics
│           ├─ best_params_*.jsonl      # Hyperparameters per fold
│           ├─ baseline_*.tsv           # Baseline model results
│           ├─ run_manifest.json        # Runtime environment and configuration
│           └─ summary.json             # Overall performance metrics with CIs
├─ .venv/                               # Python virtual environment
├─ requirements.txt                     # Python dependencies
└─ README.md                            # This documentation
```


## Configuration and Defaults

You may create `eeg_pipeline/config.py` to centralize defaults. The scripts gracefully fall back to built-ins if the module is missing.

Key configuration sections in `config.py`:

**General Settings:**
- `bids_root` (str): Path to the BIDS root (default: `eeg_pipeline/bids_output`)
- `deriv_root` (str): Path to the derivatives root (default: `<bids_root>/derivatives`)
- `task` (str): Default BIDS task label (default: `thermalactive`)
- `subjects` (list): Default subject list for processing

**Preprocessing Parameters:**
- `l_freq` (float): High-pass filter frequency (default: 1.0 Hz)
- `h_freq` (float): Low-pass filter frequency (default: 100 Hz)
- `raw_resample_sfreq` (int): Resampling frequency (default: 500 Hz)
- `eeg_template_montage` (str): Montage name (default: `easycap-M1`)
- `zapline_fline` (float): Line noise frequency (default: 60.0 Hz)

**Epoching and Analysis:**
- `conditions` (list): Event conditions for epoching
- `epochs_tmin`, `epochs_tmax` (float): Epoch time window
- `baseline` (tuple): Baseline correction window
- `spatial_filter` (str): Artifact correction method (default: `ica`)

**Feature Engineering:**
- `features_freq_bands` (dict): Frequency band definitions (theta, alpha, beta, gamma)
- `features_sourcecoords_file` (str): Path to atlas coordinates for connectivity
- `features_compute_sourcespace_features` (bool): Enable connectivity analysis


## Environment Setup

### Prerequisites
- **Python 3.8-3.11** (3.11 recommended for optimal performance)
- **Windows, macOS, or Linux** (examples show Windows PowerShell)
- **Git** for version control

### Installation Steps

```powershell
# 1. Clone or download the repository
git clone <repository-url>
cd EEG_fMRI_Analysis

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate    # macOS/Linux

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "import mne, mne_bids; print('MNE version:', mne.__version__)"
```

### Key Dependencies
- **MNE-Python ≥1.6**: Core EEG analysis functionality
- **MNE-BIDS ≥0.15**: BIDS format handling
- **scikit-learn ≥1.2**: Machine learning algorithms
- **PyRiemann ≥0.3**: Riemannian geometry for covariance-based analysis
- **statsmodels ≥0.14**: Advanced statistical methods (LOESS, partial correlation)
- **NumPy <2.0**: Constrained for ecosystem compatibility on Python 3.11

**Optional but Recommended:**
- **ICLabel support**: Requires `onnxruntime` or `torch` for automated IC classification
- **CUDA support**: For GPU-accelerated computations (PyTorch backend)

If you encounter dependency conflicts, consider using conda/mamba for environment management.


## Typical Workflows

### Workflow A: Basic pipeline from raw → BIDS → ERP → TFR

```powershell
# 1) Convert BrainVision to BIDS with flexible event filtering
python eeg_pipeline/raw_to_bids.py --merge_behavior --overwrite

# 2) (Optional) Standalone behavior merge with per-run support
python eeg_pipeline/merge_behavior_to_events.py

# 3) Event-related potential analysis
python eeg_pipeline/01_foundational_analysis.py --subject 001 --task thermalactive

# 4) Time–frequency analysis (pooled across conditions)
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --task thermalactive

# 5) Time–frequency analysis (per-temperature stratified)
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

### Workflow D: Advanced decoding analysis with validation

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

# 4) Validate pipeline outputs
python eeg_pipeline/verify_decoding_outputs.py
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
- **Baselines**: `baseline_global_loso_predictions.tsv`, `baseline_subject_loso_predictions.tsv`, `baseline_temperature_loso_predictions.tsv`
- **Hyperparameters**: `best_params_elasticnet.jsonl`, `best_params_rf.jsonl` (per-fold optimal parameters)
- **Cross-Validation**: `indices/loso_train_test_indices.json` - Subject groupings per fold
- **Runtime Info**: `run_manifest.json` - Environment, git commit, package versions
- **Advanced Plots**:
  - `plots/rf_block_permutation_importance_top20.png` - Subject-aware feature importance
  - `plots/rf_residuals_vs_temperature.png` - Residual analysis vs temperature
  - `plots/rf_residuals_vs_trial_number.png` - Residual analysis vs trial effects
  - `plots/rf_calibration_curve.png` - Calibration analysis with LOESS smoothing
  - `plots/rf_within_vs_loso_combined.png` - Within-subject vs cross-subject comparison
  - `plots/elasticnet_loso_actual_vs_predicted.png` - Model prediction scatter plots
- **Summary**: `summary.json` with comprehensive metrics including partial correlations and bootstrap CIs


## Troubleshooting

### Common Data Issues
- **Missing or misaligned events length**: Scripts will trim to the min length and warn. Ensure `events.tsv` matches the epochs count
- **No `pain_binary_coded` column**: Pain/non-pain contrasts will be skipped. Check for alternative column names (`pain_binary`, `pain`)
- **No temperature column**: Per-temperature analyses will be skipped. Expected columns: `stimulus_temp`, `stimulus_temperature`, `temp`, `temperature`
- **Channel name mismatches**: If channel names don't match the montage template, `raw_to_bids.py` will continue without setting coordinates
- **Missing source coordinates**: Connectivity analysis requires Schaefer atlas files in `source_data/Schaefer2018/`

### Performance and Runtime
- **TFR computation slowness**: Time-frequency analysis is the most intensive step. Consider:
  - Running pooled analysis only (`--temperature_strategy pooled`)
  - Processing subjects individually rather than batches
  - Reducing frequency resolution in `config.py`
- **Memory issues**: For large datasets, reduce parallel jobs (`--n_jobs`) or process subjects in smaller batches
- **Long bootstrap times**: Reduce `bootstrap_n` in decoding config for faster confidence intervals

### Installation and Dependencies
- **MNE-BIDS compatibility**: Ensure MNE-Python and MNE-BIDS versions are compatible (see `requirements.txt`)
- **NumPy version conflicts**: Use NumPy <2.0 for Python 3.11 compatibility
- **PyRiemann missing**: Install with `pip install pyriemann` or run only ElasticNet/RF models
- **ICLabel backend issues**: Choose either `onnxruntime` (default) or `torch` for IC classification
- **CUDA availability**: For GPU acceleration, ensure PyTorch CUDA version matches your system

### Pipeline-Specific Issues

**Raw-to-BIDS Conversion:**
- **Event annotation filtering**: Use `--keep_all_annotations` if default "Stim_on" filtering is too restrictive
- **Multiple event prefixes**: Use multiple `--event_prefix` flags for complex event structures
- **MRI volume trimming**: `--trim_to_first_volume` requires "Volume" annotations in the data

**Behavioral Merging:**
- **Per-run CSV matching**: Ensure PsychoPy CSV files follow `run<N>` naming convention
- **Column alignment**: Length mismatches between events and behavioral data are automatically trimmed
- **Event type selection**: Use `--event_prefix` and `--event_type` for targeted merging

**Advanced Decoding:**
- **ConstantInputWarning**: Check for duplicated subjects with identical data causing leakage
- **ElasticNet convergence**: Warnings are normal; increase `max_iter` only if performance degrades
- **Random Forest parallelism**: RF uses `n_jobs=1` to avoid conflicts; use `--n_jobs` for inner CV only
- **Block permutation importance**: Features constant within subject blocks are automatically skipped
- **Bootstrap CI failures**: BCa intervals may fail with small datasets; automatic fallback to percentile
- **LOESS calibration**: Cross-validation for span selection adds computational time but improves accuracy
- **Riemann analysis robustness**: Uses channel intersection across subjects to handle heterogeneous montages

### Validation and Quality Control
- **Output verification**: Run `verify_decoding_outputs.py` to check for missing or corrupted files
- **Missing baseline files**: Some baseline models may be skipped if insufficient subjects or data
- **JSON/JSONL corruption**: Verification script checks file structure and reports parsing errors
- **Plot generation failures**: Check for sufficient data points and valid statistical comparisons

### Getting Help
- **Verbose logging**: Most scripts support detailed console output for debugging
- **Configuration validation**: Use `config_validation = "warn"` in config.py for parameter checking
- **File path debugging**: Scripts provide detailed path resolution logs when files are missing
- **Statistical diagnostics**: Decoding pipeline includes extensive diagnostic plots for model validation


---

## Additional Notes

### Performance Optimization
- **Parallel Processing**: Most scripts support `--n_jobs` for CPU parallelization
- **Memory Management**: Pipeline uses streaming computation and efficient file I/O
- **Caching**: Intermediate results are saved to avoid recomputation
- **GPU Support**: Available for PyTorch-based operations (ICLabel, some ML algorithms)

### Extensibility
- **Custom Frequency Bands**: Modify `features_freq_bands` in `config.py`
- **Additional Algorithms**: Extend decoding script with new ML models
- **Alternative Atlases**: Replace Schaefer atlas with custom parcellations
- **Custom Baselines**: Add new baseline models to the decoding pipeline

### Research Applications
- **Pain Research**: Optimized for thermal pain studies but adaptable to other paradigms
- **Clinical Translation**: LOSO cross-validation ensures generalizability to new subjects
- **Methodological Studies**: Comprehensive diagnostics support method validation research
- **Multi-Modal Integration**: Framework supports extension to EEG-fMRI or other combined modalities

For additional features, customizations, or support, please refer to the documentation or open an issue with specific requirements.
