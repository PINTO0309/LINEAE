_base_ = ['./lineae_p.py']

from models.lineae.variants import get_variant as _get_variant  # noqa: E402
_spec = _get_variant('N')

output_dir = 'outputs/lineae_n'
variant = 'N'
backbone = _spec.backbone
backbone_weights = _spec.checkpoint
# N has HGNetV2-B0's native stage 4 and does not construct a synthetic P5.
synthetic_p5_schema = None
in_channels_encoder = [256, 512, 1024]
num_queries = 1200
epochs = 72
