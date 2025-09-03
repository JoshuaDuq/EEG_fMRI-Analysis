import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import mne

ROOT = Path(__file__).resolve().parents[1]


def _load_module(fname, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / fname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_foundational_group(tmp_path):
    fa = _load_module('01_foundational_analysis.py', 'fa')
    fa.DERIV_ROOT = tmp_path
    fa.PLOTS_SUBDIR = 'foundational'
    fa._save_fig = lambda *args, **kwargs: None
    fa.ERP_PICKS = ['Cz']
    info = mne.create_info(['Cz'], sfreq=100, ch_types='eeg')
    data = np.random.randn(1, 100)
    ev = mne.EvokedArray(data, info, tmin=0)
    results = [
        {'pain_evokeds': {'painful': ev, 'non-painful': ev}, 'temp_evokeds': {'low': ev}},
        {'pain_evokeds': {'painful': ev, 'non-painful': ev}, 'temp_evokeds': {'low': ev}},
    ]
    fa.aggregate_group_level(results)
    out_dir = tmp_path / 'group' / 'eeg' / 'plots' / 'foundational'
    assert out_dir.exists()


def test_time_frequency_group(tmp_path):
    tf = _load_module('02_time_frequency_analysis.py', 'tf')
    tf.DERIV_ROOT = tmp_path
    tf._save_fig = lambda *args, **kwargs: None
    tf.BAND_BOUNDS = {'alpha': (8, 12)}
    info = mne.create_info(['Cz'], sfreq=100, ch_types='eeg')
    epochs_data = np.random.randn(1, 1, 100)
    epochs = mne.EpochsArray(epochs_data, info)
    tfr1 = mne.time_frequency.tfr_morlet(epochs, freqs=np.array([10]), n_cycles=2, return_itc=False)
    tfr2 = mne.time_frequency.tfr_morlet(epochs, freqs=np.array([10]), n_cycles=2, return_itc=False)
    tf.aggregate_group_level([tfr1, tfr2], plateau_tmin=0, plateau_tmax=0.5)
    out_dir = tmp_path / 'group' / 'eeg' / 'plots' / '02_time_frequency_analysis'
    assert out_dir.exists()


def test_feature_engineering_group(tmp_path):
    fe = _load_module('03_feature_engineering.py', 'fe')
    fe.DERIV_ROOT = tmp_path
    fe._save_fig = lambda *args, **kwargs: None
    subjects = ['001', '002']
    for sub in subjects:
        fdir = tmp_path / f'sub-{sub}' / 'eeg' / 'features'
        fdir.mkdir(parents=True)
        pd.DataFrame({'a': [1.0], 'b': [2.0]}).to_csv(fdir / 'features_all.tsv', sep='\t', index=False)
        pd.DataFrame({'rating': [5]}).to_csv(fdir / 'target_vas_ratings.tsv', sep='\t', index=False)
    fe.aggregate_group_level(subjects)
    out_file = tmp_path / 'group' / 'eeg' / 'features' / 'features_all.tsv'
    assert out_file.exists()


def test_behavior_feature_group(tmp_path):
    bf = _load_module('04_behavior_feature_analysis.py', 'bf')
    bf.DERIV_ROOT = tmp_path
    bf._save_fig = lambda *args, **kwargs: None
    subjects = ['001', '002']
    for sub in subjects:
        sdir = tmp_path / f'sub-{sub}' / 'eeg' / 'stats'
        sdir.mkdir(parents=True)
        pd.DataFrame({'roi': ['roi1'], 'band': ['alpha'], 'r': [0.1]}).to_csv(
            sdir / 'corr_stats_pow_roi_vs_rating.tsv', sep='\t', index=False
        )
    bf.aggregate_group_level(subjects)
    out_file = tmp_path / 'group' / 'eeg' / 'stats' / 'group_corr_pow_roi_vs_rating.tsv'
    assert out_file.exists()
