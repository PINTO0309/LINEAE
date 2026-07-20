_base_ = ['./lineae_f.py']

from models.lineae.variants import get_variant as _get_variant  # noqa: E402
_spec = _get_variant('P')

output_dir = 'outputs/lineae_p'
variant = 'P'
backbone = _spec.backbone
backbone_weights = _spec.checkpoint
eval_spatial_size = (_spec.input_size, _spec.input_size)
data_aug_scales = [(_spec.input_size, _spec.input_size)]
epochs = 60
