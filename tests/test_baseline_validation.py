import numpy as np
import importlib.util
from pathlib import Path
import sys
import pytest

# Ensure project root on path and dynamically load module
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
spec = importlib.util.spec_from_file_location("tfa", ROOT / "02_time_frequency_analysis.py")
tfa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tfa)


def test_baseline_mask_valid():
    times = np.linspace(-5, 5, 101)
    b_start, b_end, mask = tfa._validate_baseline_indices(times, (-5.0, -0.1))
    assert b_end < 0
    assert mask.sum() == np.sum((times >= b_start) & (times < b_end))


def test_baseline_mask_invalid_end():
    times = np.linspace(-5, 5, 101)
    with pytest.raises(ValueError):
        tfa._validate_baseline_indices(times, (-5.0, 0.0))


def test_baseline_mask_insufficient_samples():
    times = np.linspace(-5, -4, 4)  # Only 4 samples in baseline region
    with pytest.raises(ValueError):
        tfa._validate_baseline_indices(times, (-5.0, -0.1))
