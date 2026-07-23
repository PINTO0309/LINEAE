"""Accuracy-priority LINEAE 3XL with the official DINOv3 ViT-H+/16 backbone."""

_base_ = ['./lineae_xl.py']

from models.lineae.variants import get_variant as _get_variant  # noqa: E402
_spec = _get_variant('3XL')

output_dir = 'outputs/lineae_3xl'
variant = '3XL'
training_profile = 'single_gpu_96gb_accuracy'
accuracy_head_schema = 'wide_multilevel_residual_v1'
multi_scale_train = True
backbone = _spec.backbone
backbone_weights = _spec.checkpoint
backbone_pyramid_channels = _spec.pyramid_channels
backbone_trainable_layers = 0
dino_intermediate_layers = [7, 15, 23, 31]
dino_intermediate_fusion_schema = 'residual_final_v1'
progressive_unfreeze = False
initial_freeze_epochs = 0
unfreeze_interval = 0
use_checkpoint = True
in_channels_encoder = [640, 640, 640]
encoder_use_indices = [1, 2]
encoder_num_layers = 2
hidden_dim = 640
feat_channels_decoder = [640, 640, 640]
nheads = 20
dim_feedforward = 2560
expansion = 0.5
depth_mult = 1.0
dec_layers = 10
eval_idx = 9
dec_n_points = [8, 4, 2]
reg_max = 32
dropout = 0.1
batch_size_train = 2
batch_size_val = 2
recipe_reference_effective_batch_size = 8
gradient_accumulation_steps = 4
lr = 0.0002
lr_scheduler = 'cosine'
use_warmup = False
use_ema = False
distill_weight = 0.0
distill_feature_weight = 0.0
epochs = 50
