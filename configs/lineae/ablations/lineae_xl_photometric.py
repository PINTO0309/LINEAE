"""XL geometry-preserving Gazelle photometric-distortion ablation."""

_base_ = ['../lineae_xl.py']

output_dir = 'outputs/ablations/lineae_xl_photometric'
use_photometric_distort = True
photometric_distort_probability = 0.5
