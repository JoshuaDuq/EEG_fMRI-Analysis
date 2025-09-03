# EEG Analysis Pipeline: Comprehensive Analysis Suite

This repository contains a comprehensive EEG analysis pipeline built on top of MNE-Python and MNE-BIDS for pain experience research. It provides command-line tools for the complete analysis workflow from raw data to advanced machine learning decoding:

- **Data Preparation**: Convert BrainVision EEG recordings into BIDS format and merge behavioral data
- **Event-Related Potential Analysis**: Foundational ERP analyses comparing pain vs non-pain conditions  
- **Time-Frequency Analysis**: Comprehensive spectral power analysis with baseline correction
- **Feature Engineering**: Extract ML-ready features from EEG power and connectivity
- **Behavioral Analysis**: Correlate EEG features with behavioral measures using advanced statistical methods
- **Pain Decoding**: State-of-the-art machine learning models with rigorous cross-validation and statistical validation
- **Group-Level Support**: Core analysis scripts (01–04) accept multiple subjects and aggregate results to produce grand-average statistics and visualizations

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
  - `plot_connectivity_edges_fdr.py` - Plotting connectivity edges with FDR correction
  - `eeg_config.yaml` - Centralized configuration file (YAML format)
  - `config_loader.py` - Configuration loader with fallback support
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
- `--bids_root` (str): Output BIDS root directory. Default: from `eeg_config.yaml` if present, else `eeg_pipeline/bids_output`
- `--task` (str): BIDS task label. Default: from `eeg_config.yaml` or `thermalactive`
- `--subjects` (list[str]): Optional list of subject labels to include, e.g., `001 002`. If omitted, all found are used
- `--montage` (str): Standard montage name for `mne.channels.make_standard_montage` (e.g., `easycap-M1`). Use empty string `""` to skip
- `--line_freq` (float): Line noise frequency metadata for sidecar (Hz). Default: from `eeg_config.yaml` (`preprocessing.line_freq`) or 60.0
- `--event_prefix` (str): Event annotation prefix to keep (repeatable for multiple prefixes). Default: "Trig_therm"
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
- `--bids_root` (str): BIDS root containing `sub-*/eeg/*_events.tsv`. Default: from `eeg_config.yaml` if available
- `--source_root` (str): Source root containing `sub-*/PsychoPy_Data/*TrialSummary.csv`. Default: `eeg_pipeline/source_data`
- `--task` (str): Task label used in events filenames. Default: from `eeg_config.yaml` or `thermalactive`
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
- `--task, -t` (str): BIDS task label. Default: from `eeg_config.yaml` or `thermalactive`.
- `--crop-tmin` (float): Optional epoch crop start time in seconds.
- `--crop-tmax` (float): Optional epoch crop end time in seconds (excluded).

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
- `--task, -t` (str): BIDS task label. Default: from `eeg_config.yaml` or `thermalactive`.
- `--plateau_tmin` (float): Start of plateau window in seconds (default from config).
- `--plateau_tmax` (float): End of plateau window in seconds (default from config).
- `--temperature_strategy` (str): One of `pooled`, `per`, or `both` (default: `pooled`).

**Detailed Analysis Pipeline:**

**1. Time-Frequency Decomposition:**
- **Morlet Wavelet Transform**: Uses `mne.time_frequency.tfr_morlet` with logarithmically-spaced frequencies from 4–100 Hz
- **Wavelet Parameters**: `n_cycles = freqs/2` providing optimal time-frequency resolution trade-off
- **Temporal Resolution**: Maintains trial-by-trial decomposition for single-trial analyses
- **Frequency Bands**: Automatic segmentation into theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-100 Hz)

**2. Baseline Correction:**
- **Method**: Log-ratio baseline correction (`mode='logratio'`; returns 10·log10(power/baseline) in dB) for relative power changes
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
- `--task` (str): BIDS task label. Default from `eeg_config.yaml` or `thermalactive`.

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
- `--task` (str): BIDS task label. Default from `eeg_config.yaml` or `thermalactive`.
- `--pearson` (flag): Use Pearson correlations instead of default Spearman when appropriate.
- `--partial-covars` (list[str]): Event columns to control for in partial correlations (e.g., temperature trial_number).
- `--bootstrap` (int): Number of bootstrap resamples for 95% confidence intervals (default 0, disabled).
- `--n-perm` (int): Number of permutations for permutation p-values (default 0, disabled).
- `--group` (flag): Also aggregate group-level results across subjects.
- `--group-only` (flag): Only run group-level aggregation, skip per-subject analysis.
- `--report` (flag): Build per-subject MNE HTML report.

Group-level runs merge per-subject ROI power correlations for both pain ratings and stimulus temperature, and summarize connectivity ROI statistics across participants with FDR correction.

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
- `--task` (str): BIDS task label. Default from `eeg_config.yaml` or `thermalactive`.
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
│  ├─ raw_to_bids.py                     # Convert BrainVision to BIDS with flexible event filtering
│  ├─ merge_behavior_to_events.py        # Merge behavioral data with per-run support
│  ├─ 01_foundational_analysis.py        # Event-related potential (ERP) analysis
│  ├─ 02_time_frequency_analysis.py      # Time-frequency analysis and spectral power
│  ├─ 03_feature_engineering.py          # Extract ML-ready features from EEG data
│  ├─ 04_behavior_feature_analysis.py    # Behavioral correlations with advanced statistics
│  ├─ 05_decode_pain_experience.py       # Advanced pain decoding with comprehensive diagnostics
│  ├─ verify_decoding_outputs.py         # Validation tool for decoding pipeline outputs
│  ├─ plot_connectivity_edges_fdr.py     # Plotting connectivity edges with FDR correction
│  ├─ eeg_config.yaml                    # Centralized configuration file (YAML format)
│  ├─ config_loader.py                   # Configuration loader with fallback support
│  ├─ coll_lab_eeg_pipeline.py           # Alternative comprehensive pipeline
│  ├─ source_data/
│  │  ├─ Schaefer2018/                   # Atlas files for connectivity analysis
│  │  │  ├─ Schaefer2018_400Parcels_7Networks_order.dscalar.nii
│  │  │  ├─ Schaefer2018_400Parcels_7Networks_order.txt
│  │  │  └─ ...                          # Additional atlas files
│  │  └─ sub-XXX/
│  │     ├─ eeg/                         # Raw BrainVision files (*.vhdr, *.eeg, *.vmrk)
│  │     │  ├─ sub-XXX_task-thermalactive_run-01.vhdr
│  │     │  ├─ sub-XXX_task-thermalactive_run-01.eeg
│  │     │  ├─ sub-XXX_task-thermalactive_run-01.vmrk
│  │     │  └─ ...                       # Additional runs if present
│  │     └─ PsychoPy_Data/               # Behavioral CSV files
│  │        └─ sub-XXX_task-thermalactive_run-01_TrialSummary.csv
│  └─ bids_output/
│     ├─ sub-XXX/
│     │  └─ eeg/                         # Raw BIDS outputs
│     │     ├─ sub-XXX_task-thermalactive_run-01_eeg.vhdr
│     │     ├─ sub-XXX_task-thermalactive_run-01_eeg.eeg
│     │     ├─ sub-XXX_task-thermalactive_run-01_eeg.vmrk
│     │     ├─ sub-XXX_task-thermalactive_run-01_events.tsv
│     │     └─ sub-XXX_task-thermalactive_run-01_electrodes.tsv
│     └─ derivatives/
│        ├─ sub-XXX/
│        │  └─ eeg/
│        │     ├─ plots/                  # Subject-specific figures
│        │     │  ├─ 01_foundational_analysis/
│        │     │  │  ├─ erp_pain_binary_butterfly.png
│        │     │  │  ├─ erp_pain_binary_topomaps.png
│        │     │  │  ├─ erp_by_temperature_gfp.png
│        │     │  │  └─ ...               # Additional ERP plots
│        │     │  ├─ 02_time_frequency_analysis/
│        │     │  │  ├─ tfr_Cz_all_trials_baseline_logratio.png
│        │     │  │  ├─ topomap_alpha_all_trials_baseline_logratio.png
│        │     │  │  ├─ temperature/      # Per-temperature plots
│        │     │  │  └─ logs/             # Analysis logs
│        │     │  ├─ 04_behavior_feature_analysis/
│        │     │  │  ├─ power_behavior_correlations.png
│        │     │  │  ├─ scatter_pow_roi_frontal_alpha_vs_rating.png
│        │     │  │  └─ logs/             # Analysis logs
│        │     │  └─ temperature/         # Temperature-specific subdirectories
│        │     ├─ features/               # ML-ready feature matrices
│        │     │  ├─ features_eeg_direct.tsv
│        │     │  ├─ features_connectivity.tsv
│        │     │  ├─ features_all.tsv
│        │     │  └─ target_vas_ratings.tsv
│        │     ├─ behavior_analysis/      # Behavioral correlation outputs
│        │     │  ├─ corr_stats_pow_roi_vs_rating.tsv
│        │     │  ├─ corr_stats_conn_roi_summary_aec_alpha_vs_rating.tsv
│        │     │  ├─ power_behavior_correlations.png
│        │     │  └─ ...                  # Additional correlation files
│        │     ├─ stats/                  # Statistical outputs
│        │     │  ├─ corr_stats_pow_vs_temp.tsv
│        │     │  └─ psychometric_rating_vs_temp_stats.tsv
│        │     └─ logs/                   # Log files for all analyses
│        │        ├─ 01_foundational_analysis.log
│        │        ├─ 02_time_frequency_analysis.log
│        │        ├─ 03_feature_engineering.log
│        │        └─ 04_behavior_feature_analysis.log
│        └─ decoding/                     # Cross-subject decoding results
│           ├─ plots/                     # Decoding diagnostic plots
│           │  ├─ rf_loso_actual_vs_predicted.png
│           │  ├─ rf_block_permutation_importance_top20.png
│           │  ├─ rf_residuals_vs_temperature.png
│           │  └─ ...                     # Additional diagnostic plots
│           ├─ indices/                   # Cross-validation indices
│           │  └─ loso_train_test_indices.json
│           ├─ predictions/               # Model predictions
│           │  ├─ elasticnet_loso_predictions.tsv
│           │  ├─ rf_loso_predictions.tsv
│           │  └─ ...                     # Additional prediction files
│           ├─ metrics/                   # Performance metrics
│           │  ├─ elasticnet_per_subject_metrics.tsv
│           │  ├─ rf_per_subject_metrics.tsv
│           │  └─ ...                     # Additional metric files
│           ├─ hyperparameters/           # Best parameters per fold
│           │  ├─ best_params_elasticnet.jsonl
│           │  ├─ best_params_rf.jsonl
│           │  └─ ...                     # Additional hyperparameter files
│           ├─ baselines/                 # Baseline model results
│           │  ├─ baseline_global_loso_predictions.tsv
│           │  └─ ...                     # Additional baseline files
│           ├─ summaries/                 # Summary statistics
│           │  ├─ summary.json
│           │  ├─ summary_bootstrap.json
│           │  └─ ...                     # Additional summary files
│           ├─ run_manifest.json          # Execution metadata
│           └─ logs/                      # Decoding pipeline logs
│              └─ decode_pain_YYYYMMDD_HHMMSS.log
├─ .venv/                                # Python virtual environment
│  ├─ Scripts/                           # Windows executables
│  ├─ Lib/                               # Package installations
│  └─ pyvenv.cfg                         # Environment configuration
├─ requirements.txt                      # Python dependencies
├─ commands.txt                          # Common command examples
└─ README.md                             # This documentation
```


## Configuration and Defaults

The EEG pipeline uses a centralized YAML configuration system (`eeg_config.yaml`) that provides a single source of truth for all pipeline parameters. This approach ensures consistency, reproducibility, and easy customization without modifying source code.

### Key Configuration Sections

#### Core Project Settings
- `project.root` (str): Project root directory (automatically resolved to absolute path)
- `project.task` (str): BIDS task label (default: `thermalactive`)
- `project.subjects` (list): Default subject list for processing (default: `["0000"]`)
- `project.bids_root` (str): BIDS output directory path
- `project.source_root` (str): Source data directory path

#### Preprocessing Parameters
- `preprocessing.line_freq` (float): Line noise frequency for notch filtering (Hz)
- `preprocessing.high_pass` (float): High-pass filter cutoff (Hz)
- `preprocessing.low_pass` (float): Low-pass filter cutoff (Hz)
- `preprocessing.resample_sfreq` (int): Resampling frequency (Hz)
- `preprocessing.montage` (str): EEG electrode montage name

#### Frequency Band Definitions
- `frequency_bands` (dict): Frequency band definitions for analysis:
  ```yaml
  theta: [4.0, 8.0]
  alpha: [8.0, 13.0]
  beta: [13.0, 30.0]
  gamma: [30.0, 80.0]
  ```

#### Analysis Windows and Parameters
- `power.baseline_window` (list): Baseline correction window [start, end] in seconds
- `power.plateau_window` (list): Time window for power averaging [start, end] in seconds
- `analysis.erp.picks` (str): Channel type for ERP averaging
- `analysis.erp.pain_color` (str): Color for pain condition in plots
- `analysis.erp.nonpain_color` (str): Color for non-pain condition in plots

#### Visualization Settings
- `visualization.dpi` (int): Figure resolution (default: 300)
- `visualization.save_formats` (list): Output formats (default: ["png", "svg"])
- `visualization.band_colors` (dict): Color mapping for frequency bands
- `visualization.montage` (str): Montage for topographic plotting
- `visualization.advanced` (dict): Specialized plotting parameters

#### Machine Learning Parameters
- `decoding.models.elasticnet.alpha` (list): Regularization strengths for ElasticNet
- `decoding.models.rf.n_estimators` (int): Number of trees in Random Forest
- `decoding.analysis.n_perm_quick` (int): Permutations for quick significance tests
- `decoding.analysis.bootstrap_n` (int): Bootstrap samples for confidence intervals

#### Random Seed and Reproducibility
- `random.seed` (int): Global random seed for reproducibility (default: 42)
- `random.bootstrap_default` (int): Default bootstrap iterations

### How to Modify Configuration

1. **Edit `eeg_config.yaml`** directly with any text editor:
   ```yaml
   # Example customizations
   project:
     subjects: ["001", "002", "003"]
     task: "pain_study"
   
   preprocessing:
     line_freq: 50.0  # For European power grid
     high_pass: 0.1   # More aggressive high-pass
   
   visualization:
     dpi: 600  # Higher resolution figures
   ```

2. **Restart Python session** or reload configuration:
   ```python
   # The scripts automatically reload configuration
   from config_loader import load_config
   config = load_config()
   ```

3. **Verify changes** by running any script - it will use your updated parameters

### Configuration Benefits

- **Centralized**: All parameters in one YAML file
- **Human-readable**: Easy to understand and modify
- **Version-controllable**: Track parameter changes in git
- **Validated**: Automatic type checking and fallback defaults
- **Documented**: Self-documenting with comments and structure
- **Reproducible**: Ensures consistent results across runs

### Legacy Compatibility

The system maintains backward compatibility:
- Scripts fall back to built-in defaults if configuration loading fails
- Environment variables and command-line arguments override config values
- Missing config sections use sensible defaults

### Advanced Configuration Options

For complex setups, you can:
- Use environment variable substitution: `${MY_CUSTOM_PATH}`
- Include external YAML files with anchors and references
- Set up different config files for different analysis scenarios
- Use the config_loader API for programmatic access

### Configuration Validation

The system includes built-in validation:
- Type checking for numeric parameters
- Path validation for file/directory parameters
- Range checking for frequency and time parameters
- Automatic fallback to defaults for invalid values


## Environment Setup

### Prerequisites
- **Python 3.8-3.11** (3.11 recommended for optimal performance)
- **Windows, macOS, or Linux** (examples show Windows PowerShell)
- **Git** for version control
- **Minimum 16 GB RAM** recommended for TFR computations
- **At least 100 GB free disk space** for large datasets and outputs

### Installation Steps

#### Option 1: Virtual Environment (Recommended)

```powershell
# 1. Clone or download the repository
git clone <repository-url>
cd EEG_fMRI_Analysis

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate    # macOS/Linux

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "import mne, mne_bids; print('MNE version:', mne.__version__)"
python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
```

#### Option 2: Conda/Mamba Environment

```bash
# Using conda (or mamba for faster package resolution)
conda create -n eeg_pipeline python=3.11
conda activate eeg_pipeline

# Install core dependencies
conda install -c conda-forge mne mne-bids scikit-learn numpy scipy matplotlib pandas seaborn

# Install remaining dependencies
pip install -r requirements.txt
```

#### Option 3: Docker Container

```bash
# Build Docker image
docker build -t eeg-pipeline .

# Run container with data volume mounted
docker run -it -v /path/to/data:/data eeg-pipeline
```

### Data Preparation

Before running the pipeline:

1. **Organize raw EEG data** in BIDS-like structure:
   ```
   source_data/
   ├── sub-001/
   │   ├── eeg/
   │   │   ├── sub-001_task-thermalactive_run-01.vhdr
   │   │   ├── sub-001_task-thermalactive_run-01.eeg
   │   │   ├── sub-001_task-thermalactive_run-01.vmrk
   │   │   └── ...
   │   └── PsychoPy_Data/
   │       └── sub-001_task-thermalactive_run-01_TrialSummary.csv
   └── sub-002/
       └── ...
   ```

2. **Download atlas files** for connectivity analysis:
   ```bash
   # Download Schaefer2018 atlas (7 networks, 400 parcels)
   # Place in source_data/Schaefer2018/
   ```

3. **Verify montage compatibility**:
   - Check that your EEG system matches `easycap-M1` or `standard_1005` electrode layouts
   - Custom montages can be added via `mne.channels.make_dig_montage()`

### Verification Steps

After installation, verify all components:

```python
# Test core EEG functionality
import mne
import numpy as np

# Test TFR computation
data = np.random.randn(10, 64, 1000)  # (n_epochs, n_channels, n_times)
epochs = mne.EpochsArray(data, mne.create_info(64, 1000, ch_types='eeg'))
tfr = mne.time_frequency.tfr_morlet(epochs, freqs=[10, 20], n_cycles=2)
print("TFR computation: SUCCESS")

# Test ML functionality
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

X = np.random.randn(100, 10)
y = np.random.randn(100)
rf = RandomForestRegressor(n_estimators=10, random_state=42)
scores = cross_val_score(rf, X, y, cv=3)
print(f"Random Forest CV scores: {scores}")

# Test BIDS functionality
import mne_bids
print(f"MNE-BIDS version: {mne_bids.__version__}")

print("All core components verified successfully!")
```

### Optional Dependencies

Install additional packages for enhanced functionality:

```bash
# Riemannian geometry analysis
pip install pyriemann

# GPU acceleration for ML
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Automated IC classification
pip install onnxruntime  # or pip install torch for alternative backend

# Enhanced statistical analysis
pip install statsmodels>=0.14

# Parallel processing improvements
pip install joblib
```

### Environment Configuration

Set environment variables for optimal performance:

```bash
# Windows PowerShell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

# Linux/macOS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

### Troubleshooting Installation

- **ImportError**: Check Python version compatibility (3.8-3.11)
- **MemoryError during pip install**: Use `pip install --no-cache-dir`
- **NumPy compatibility**: Ensure NumPy <2.0 for Python 3.11
- **CUDA issues**: Verify PyTorch CUDA version matches your GPU drivers
- **Permission denied**: Run terminal as administrator or use user installation

### Key Dependencies

**Core Requirements:**
- **MNE-Python ≥1.6**: Core EEG analysis functionality
- **MNE-BIDS ≥0.15**: BIDS format handling
- **PyYAML ≥6.0**: YAML configuration file parsing (required for centralized configuration)
- **scikit-learn ≥1.2**: Machine learning algorithms
- **NumPy <2.0**: Array computations (constrained for Python 3.11 compatibility)
- **SciPy ≥1.7**: Scientific computing
- **Matplotlib ≥3.5**: Plotting and visualization
- **Pandas ≥1.5**: Data manipulation
- **Seaborn ≥0.11**: Statistical visualization

**Optional but Recommended:**
- **PyRiemann ≥0.3**: Riemannian geometry for covariance-based analysis
- **statsmodels ≥0.14**: Advanced statistical methods (LOESS, partial correlation)
- **ICLabel support**: Requires `onnxruntime` or `torch` for automated IC classification
- **CUDA support**: PyTorch with CUDA for GPU-accelerated computations

**Development Dependencies:**
- **joblib ≥1.1**: Parallel processing
- **tqdm**: Progress bars
- **json5**: Enhanced JSON parsing
- **psutil**: System monitoring

If you encounter dependency conflicts, consider using conda/mamba for environment management as it handles complex dependency resolution better than pip.


## Typical Workflows

Below are detailed workflow examples with precise commands, expected outputs, and explanations. All commands assume you are in the `eeg_pipeline/` directory.

### Workflow A: Basic pipeline from raw → BIDS → ERP → TFR

```powershell
# 1) Convert BrainVision to BIDS with flexible event filtering
# Expected: Creates BIDS dataset in bids_output/, merges behavior if --merge_behavior used
python eeg_pipeline/raw_to_bids.py --merge_behavior --overwrite

# 2) (Optional) Standalone behavior merge with per-run support
# Expected: Updates events.tsv files with behavioral columns, creates combined per-subject events.tsv
python eeg_pipeline/merge_behavior_to_events.py

# 3) Event-related potential analysis
# Expected: ERP plots in bids_output/derivatives/sub-001/eeg/plots/01_foundational_analysis/
python eeg_pipeline/01_foundational_analysis.py --subject 001 --task thermalactive

# 4) Time–frequency analysis (pooled across conditions)
# Expected: TFR plots and topomaps in bids_output/derivatives/sub-001/eeg/plots/02_time_frequency_analysis/
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --task thermalactive

# 5) Time–frequency analysis (per-temperature stratified)
# Expected: Additional temperature-specific plots in plots/temperature/ subdirectories
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --task thermalactive --temperature_strategy per
```

**Expected Outputs from Workflow A:**
- `bids_output/sub-001/eeg/sub-001_task-thermalactive_eeg.vhdr` (BIDS EEG data)
- `bids_output/sub-001/eeg/sub-001_task-thermalactive_events.tsv` (BIDS events with merged behavior)
- `derivatives/sub-001/eeg/plots/01_foundational_analysis/erp_pain_binary_butterfly.png`
- `derivatives/sub-001/eeg/plots/02_time_frequency_analysis/tfr_Cz_all_trials_baseline_logratio.png`
- `derivatives/sub-001/eeg/plots/02_time_frequency_analysis/topomap_alpha_all_trials_baseline_logratio.png`

### Workflow B: Re-run TFR with both pooled and per-temperature

```powershell
# Run both pooled and per-temperature analyses in one command
# Expected: All plots from pooled + temperature-specific subdirectories created
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --temperature_strategy both
```

**Expected Additional Outputs:**
- `derivatives/sub-001/eeg/plots/02_time_frequency_analysis/temperature/temp-47p3/tfr_Cz_painful_baseline_logratio.png`
- `derivatives/sub-001/eeg/plots/02_time_frequency_analysis/temperature/temp-47p3/topomap_beta_pain_vs_nonpain_baseline_logratio.png`

### Workflow C: Complete analysis pipeline with behavioral correlations and pain decoding

```powershell
# 1) Build features for all desired subjects
# Expected: Feature matrices saved in derivatives/sub-XXX/eeg/features/
python eeg_pipeline/03_feature_engineering.py --subjects 001 002 003 004 005 --task thermalactive

# 2) Analyze behavioral correlations with EEG features
# Expected: Correlation heatmaps and statistics in derivatives/sub-XXX/eeg/behavior_analysis/
python eeg_pipeline/04_behavior_feature_analysis.py --subjects 001 002 003 004 005 --task thermalactive

# 3) Run advanced LOSO pain decoding with comprehensive diagnostics
# Expected: Decoding results in derivatives/decoding/ with plots and metrics
python eeg_pipeline/05_decode_pain_experience.py --subjects all --n_jobs -1 --seed 42
```

**Expected Outputs from Workflow C:**
- `derivatives/sub-001/eeg/features/features_eeg_direct.tsv` (Power features)
- `derivatives/sub-001/eeg/behavior_analysis/power_behavior_correlations.png` (Correlation heatmap)
- `derivatives/decoding/summary.json` (Decoding performance metrics)
- `derivatives/decoding/plots/rf_loso_actual_vs_predicted.png` (Prediction scatter plot)

### Workflow D: Advanced decoding analysis with validation

```powershell
# Complete pipeline for pain decoding research
# 1) Prepare all subjects through feature engineering
# Expected: All feature matrices created for available subjects
python eeg_pipeline/03_feature_engineering.py --subjects all --task thermalactive

# 2) Run behavioral feature analysis to understand EEG-behavior relationships
# Expected: Full correlation analysis including ROI-level and connectivity analyses
python eeg_pipeline/04_behavior_feature_analysis.py --subjects all --task thermalactive

# 3) Execute comprehensive pain decoding with all advanced features:
#    - Block-aware permutation importance
#    - Partial correlation analysis controlling for covariates
#    - BCa confidence intervals with cluster bootstrap
#    - Residual diagnostics and calibration analysis
# Expected: Complete decoding analysis with all diagnostics enabled
python eeg_pipeline/05_decode_pain_experience.py --subjects all --n_jobs -1 --seed 42

# 4) Validate pipeline outputs
# Expected: Verification report of all generated files and their integrity
python eeg_pipeline/verify_decoding_outputs.py
```

**Expected Outputs from Workflow D:**
- `derivatives/decoding/plots/rf_block_permutation_importance_top20.png` (Feature importance)
- `derivatives/decoding/plots/rf_residuals_vs_temperature.png` (Model diagnostics)
- `derivatives/decoding/summary.json` (Complete performance summary with CIs)
- `derivatives/decoding/run_manifest.json` (Execution metadata)

### Workflow E: Custom analysis with specific parameters

```powershell
# Custom time-frequency analysis with adjusted parameters
python eeg_pipeline/02_time_frequency_analysis.py --subject 001 --plateau_tmin 1.0 --plateau_tmax 8.0 --temperature_strategy pooled

# Behavioral analysis with advanced statistics
python eeg_pipeline/04_behavior_feature_analysis.py --subjects 001 --bootstrap 1000 --n-perm 1000 --partial-covars temperature trial_number

# Decoding with custom settings
python eeg_pipeline/05_decode_pain_experience.py --subjects 001 002 --n_jobs 2 --seed 123
```

**Expected Outputs from Workflow E:**
- `derivatives/sub-001/eeg/plots/02_time_frequency_analysis/topomap_gamma_all_trials_baseline_logratio.png` (Custom plateau window)
- `derivatives/sub-001/eeg/behavior_analysis/corr_stats_pow_roi_vs_rating.tsv` (Advanced statistics with bootstrap CIs)
- `derivatives/decoding/elasticnet_per_subject_metrics.tsv` (Subset analysis metrics)


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
- `corr_stats_pow_roi_vs_rating.tsv` - ROI-averaged power correlations with ratings
- `corr_stats_pow_roi_vs_temp.tsv` - ROI-averaged power correlations with temperature
- `corr_stats_conn_roi_summary_<measure_band>_vs_rating.tsv` - Connectivity ROI summary correlations with ratings
- `corr_stats_conn_roi_summary_<measure_band>_vs_temp.tsv` - Connectivity ROI summary correlations with temperature
- `scatter_pow_overall_<band>_vs_rating.png` - Overall power vs rating scatter plots
- `scatter_pow_overall_<band>_vs_temp.png` - Overall power vs temperature scatter plots
- `scatter_pow_roi_<roi>_<band>_vs_rating.png` - ROI-specific power vs rating scatter plots
- `scatter_pow_roi_<roi>_<band>_vs_temp.png` - ROI-specific power vs temperature scatter plots

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

### Performance and Runtime Issues

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

#### Raw-to-BIDS Conversion (`raw_to_bids.py`)
- **Event annotation filtering**: Use `--keep_all_annotations` if default "Trig_therm" filtering is too restrictive
- **Multiple event prefixes**: Use multiple `--event_prefix` flags for complex event structures
- **MRI volume trimming**: `--trim_to_first_volume` requires "Volume" annotations in the data
- **Montage template mismatch**: Check that your EEG system matches the `easycap-M1` or `standard_1005` templates
- **Line noise frequency**: Ensure `zapline_fline` matches your local power grid frequency (60 Hz in North America, 50 Hz elsewhere)

#### Behavioral Merging (`merge_behavior_to_events.py`)
- **Per-run CSV matching**: Ensure PsychoPy CSV files follow `run<N>` naming convention
- **Column alignment**: Length mismatches between events and behavioral data are automatically trimmed
- **Event type selection**: Use `--event_prefix` and `--event_type` for targeted merging
- **Run number extraction**: Scripts extract run numbers from BIDS filenames (e.g., `sub-001_task-thermalactive_run-01_events.tsv`)

#### ERP Analysis (`01_foundational_analysis.py`)
- **Epoch cropping**: Use `--crop-tmin` and `--crop-tmax` to focus on specific time windows
- **Pain column detection**: Script tries multiple column names automatically; check your behavioral data
- **Global Field Power (GFP)**: Requires multiple channels for meaningful computation
- **Statistical contrasts**: Skip if insufficient trials in pain/non-pain conditions

#### Time-Frequency Analysis (`02_time_frequency_analysis.py`)
- **TFR memory usage**: Large frequency ranges or many time points can cause memory issues
- **Baseline correction**: Ensure pre-stimulus data exists for baseline correction (default: None to 0.0 s)
- **Plateau window**: Adjust `plateau_tmin`/`plateau_tmax` based on your stimulus timing
- **ROI definitions**: Script uses standard 10-10 electrode names for anatomical regions
- **Temperature stratification**: Requires temperature column in metadata for per-temperature analysis

#### Feature Engineering (`03_feature_engineering.py`)
- **Connectivity analysis**: Requires Schaefer atlas files for ROI-based connectivity
- **Frequency band selection**: Modify `POWER_BANDS` in config for different bands
- **Target column detection**: Script tries multiple column names for behavioral targets
- **Trial alignment**: Ensures epochs, events, and features have matching trial counts
- **Memory for large datasets**: Process subjects individually if memory is limited

#### Behavioral Analysis (`04_behavior_feature_analysis.py`)
- **Correlation computation**: Handles missing data with pairwise deletion
- **Permutation testing**: Can be computationally intensive; reduce `n_perm` for faster results
- **Bootstrap confidence intervals**: Subject-level clustering for proper statistical inference
- **ROI analysis**: Requires sensor montage information for anatomical grouping
- **Partial correlations**: Control for covariates like temperature and trial effects

#### Advanced Decoding (`05_decode_pain_experience.py`)
- **LOSO requirements**: Requires ≥2 subjects for leave-one-subject-out cross-validation
- **ConstantInputWarning**: Check for duplicated subjects with identical data causing leakage
- **ElasticNet convergence**: Warnings are normal; increase `max_iter` only if performance degrades
- **Random Forest parallelism**: RF uses `n_jobs=1` to avoid conflicts; use `--n_jobs` for inner CV only
- **Block permutation importance**: Features constant within subject blocks are automatically skipped
- **Bootstrap CI failures**: BCa intervals may fail with small datasets; automatic fallback to percentile
- **LOESS calibration**: Cross-validation for span selection adds computational time but improves accuracy
- **Riemann analysis robustness**: Uses channel intersection across subjects to handle heterogeneous montages

### Configuration Issues

#### Path Configuration
- **Custom paths**: Ensure all paths in `config.py` are absolute or relative to the project root
- **BIDS root**: Must contain the raw EEG data and will receive processed outputs
- **Derivatives root**: Auto-created if missing; contains all analysis results

#### Frequency Band Configuration
- **Band definitions**: Ensure no overlap between adjacent bands
- **Frequency limits**: Must be within your EEG sampling rate limits
- **Band naming**: Use consistent naming across all configuration sections

#### Model Hyperparameters
- **ElasticNet**: `alpha` controls regularization strength, `l1_ratio` balances L1/L2 penalties
- **Random Forest**: `n_estimators` trades off performance vs computation time
- **Grid search**: Parameter ranges are optimized for typical EEG datasets

### File System Issues

#### Permissions
- **Write permissions**: Ensure the user has write access to BIDS and derivatives directories
- **File locking**: Some operations may fail if files are open in other applications
- **Network drives**: Performance may be slower on network-mounted storage

#### File Format Issues
- **TSV format**: Ensure tab-separated values with proper headers
- **JSON/JSONL corruption**: Use `verify_decoding_outputs.py` to check file integrity
- **Figure formats**: PNG for compact storage, SVG for scalable vector graphics

#### Missing Files
- **Atlas files**: Download Schaefer2018 atlas files for connectivity analysis
- **Montage templates**: Use standard montages or provide custom electrode positions
- **Feature matrices**: Ensure feature engineering completes before downstream analyses

### Statistical and Computational Issues

#### Statistical Power
- **Sample size**: Small datasets may not yield reliable statistical results
- **Multiple comparisons**: FDR correction controls false discovery rate across tests
- **Effect sizes**: Small correlations (r < 0.1) may not be practically meaningful

#### Computational Resources
- **CPU cores**: Use `--n_jobs` to control parallel processing
- **Memory limits**: Large TFR computations may require system memory upgrades
- **Disk space**: Analysis outputs can be substantial; monitor available storage

#### Cross-Validation Stability
- **Fold consistency**: Ensure sufficient data per fold for stable model training
- **Random seeds**: Use consistent seeds for reproducible results
- **Data leakage**: Check for subject-level information leakage in features

### Getting Help

#### Logging and Debugging
- **Verbose logging**: Most scripts support detailed console output for debugging
- **Configuration validation**: Add `config_validation = "warn"` to config.py for parameter checking
- **File path debugging**: Scripts provide detailed path resolution logs when files are missing
- **Statistical diagnostics**: Decoding pipeline includes extensive diagnostic plots for model validation

#### Common Error Messages
- **"No cleaned epochs found"**: Ensure preprocessing completed successfully
- **"Feature matrix missing"**: Run feature engineering before behavioral analysis
- **"MemoryError"**: Reduce batch size or process subjects individually
- **"ValueError: array shapes"**: Check data alignment between epochs, events, and features

#### Support Resources
- **Script help**: Run any script with `--help` for detailed argument information
- **Configuration examples**: See the Configuration section for customization examples
- **Output verification**: Use `verify_decoding_outputs.py` to validate pipeline integrity
- **GitHub issues**: Report bugs with detailed error messages and configuration details


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
