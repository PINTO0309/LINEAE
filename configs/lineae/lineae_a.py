_base_ = ['./lineae_s.py']

from models.lineae.variants import get_variant as _get_variant  # noqa: E402
_spec = _get_variant('A')

output_dir = 'outputs/lineae_a'
variant = 'A'
training_profile = 'single_gpu_96gb'
multi_scale_train = True
backbone = _spec.backbone
backbone_weights = _spec.checkpoint
backbone_trainable_layers = 0
backbone_pyramid_channels = _spec.pyramid_channels
eval_spatial_size = (_spec.input_size, _spec.input_size)
data_aug_scales = [(_spec.input_size, _spec.input_size)]
image_mean = [0.538, 0.494, 0.453]
image_std = [0.257, 0.263, 0.273]
use_lab = True
freeze_norm = False
progressive_unfreeze = False
initial_freeze_epochs = 0
unfreeze_interval = 0
in_channels_encoder = [256, 256, 256]
hidden_dim = 128
feat_channels_decoder = [128, 128, 128]
dec_layers = 3
eval_idx = 2
batch_size_train = 8
batch_size_val = 8
recipe_reference_effective_batch_size = 8
gradient_accumulation_steps = 1
scheduler_step_unit = 'optimizer'
epochs = 72
model_parameters = [
    {'params': '^backbone\\.core\\.(?!.*(?:norm|bn|bias)).*$', 'lr': 0.0004},
    {'params': '^backbone\\.core\\.(?=.*(?:norm|bn|bias)).*$', 'lr': 0.0004, 'weight_decay': 0.0},
    {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$', 'weight_decay': 0.0},
]
lr = 0.0008
