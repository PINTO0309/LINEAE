"""X intermediate DINO block-fusion ablation."""

_base_ = ['../lineae_x.py']

output_dir = 'outputs/ablations/lineae_x_intermediate'
dino_intermediate_layers = [3, 7, 11]
