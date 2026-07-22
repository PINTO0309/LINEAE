_base_ = ['./lineae_n.py']

from models.lineae.variants import get_variant as _get_variant  # noqa: E402
_spec = _get_variant('T')

output_dir = 'outputs/lineae_t'
variant = 'T'
backbone = _spec.backbone
backbone_weights = _spec.checkpoint
# T uses HGNetV2-B1's native stage 4 and the original LINEA-S detector width.
synthetic_p5_schema = None
in_channels_encoder = [256, 512, 1024]
hidden_dim = 256
feat_channels_decoder = [hidden_dim, hidden_dim, hidden_dim]
dec_layers = 3
eval_idx = 2
num_queries = 1100
epochs = 60
