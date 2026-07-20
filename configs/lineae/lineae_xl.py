_base_ = ['./lineae_s.py']

from models.lineae.variants import get_variant as _get_variant  # noqa: E402
_spec = _get_variant('XL')

output_dir = 'outputs/lineae_xl'
variant = 'XL'
training_profile = 'single_gpu_96gb'
multi_scale_train = True
backbone = _spec.backbone
backbone_weights = _spec.checkpoint
backbone_pyramid_channels = _spec.pyramid_channels
in_channels_encoder = [256, 256, 256]
dec_layers = 6
eval_idx = 5
use_checkpoint = True
batch_size_train = 8
batch_size_val = 8
gradient_accumulation_steps = 1
scheduler_step_unit = 'optimizer'
epochs = 36
