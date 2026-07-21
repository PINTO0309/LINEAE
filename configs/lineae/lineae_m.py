_base_ = ['./lineae_s.py']

from models.lineae.variants import get_variant as _get_variant  # noqa: E402
_spec = _get_variant('M')

output_dir = 'outputs/lineae_m'
variant = 'M'
training_profile = 'single_gpu_96gb'
multi_scale_train = True
backbone = _spec.backbone
backbone_weights = _spec.checkpoint
backbone_pyramid_channels = _spec.pyramid_channels
in_channels_encoder = [256, 256, 256]
dec_layers = 4
eval_idx = 3
batch_size_train = 8
batch_size_val = 8
recipe_reference_effective_batch_size = 8
gradient_accumulation_steps = 1
scheduler_step_unit = 'optimizer'
epochs = 45
