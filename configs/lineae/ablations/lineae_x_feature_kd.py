"""X projected P3/P4/P5 feature-KD ablation on top of output KD."""

_base_ = ['../distill/lineae_x.py']

output_dir = 'outputs/ablations/lineae_x_feature_kd'
distill_feature_weight = 1.0
distill_feature_loss = 'cosine'
