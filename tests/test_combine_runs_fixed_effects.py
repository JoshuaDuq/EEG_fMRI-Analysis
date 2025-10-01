import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "fmri_pipeline" / "NPS" / "05_combine_runs_fixed_effects.py"
NPS_DIR = MODULE_PATH.parent

if str(NPS_DIR) not in sys.path:
    sys.path.insert(0, str(NPS_DIR))

_spec = importlib.util.spec_from_file_location("combine_runs_fixed_effects", MODULE_PATH)
combine_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(combine_module)


def test_single_run_uniform_variance_positive(tmp_path):
    data = np.ones((5, 5, 5), dtype=np.float32)
    affine = np.eye(4)
    beta_img = nib.Nifti1Image(data, affine)

    beta_path = tmp_path / "run-01_beta_temp.nii.gz"
    nib.save(beta_img, beta_path)

    combined_beta, combined_var, n_runs_img = combine_module.combine_betas_fixed_effects(
        [beta_path],
        variance_method="uniform",
    )

    combined_beta_data = combined_beta.get_fdata()
    combined_var_data = combined_var.get_fdata()
    n_runs_data = n_runs_img.get_fdata()
    mask = combined_beta_data != 0

    assert np.all(np.isfinite(combined_var_data[mask]))
    assert np.all(combined_var_data[mask] > 0)
    assert np.all(n_runs_data[mask] == 1)
    # Ensure no voxels within the mask were zeroed during processing
    assert np.all(combined_beta_data[mask] == data[mask])
