_base_ = ['./lineae_a.py']

from models.lineae.variants import get_variant as _get_variant  # noqa: E402
_spec = _get_variant('F')

output_dir = 'outputs/lineae_f'
variant = 'F'
backbone = _spec.backbone
backbone_weights = _spec.checkpoint
eval_spatial_size = (_spec.input_size, _spec.input_size)
data_aug_scales = [(_spec.input_size, _spec.input_size)]
in_channels_encoder = [256, 512, 512]
epochs = 66
