"""Accuracy-priority LINEAE 2XL with the official DINOv3 ViT-L/16 backbone."""

_base_ = ['./lineae_xl.py']

from models.lineae.variants import get_variant as _get_variant  # noqa: E402
_spec = _get_variant('2XL')

output_dir = 'outputs/lineae_2xl'
variant = '2XL'
training_profile = 'single_gpu_96gb_accuracy'
accuracy_head_schema = 'wide_multilevel_residual_v1'
multi_scale_train = True
backbone = _spec.backbone
backbone_weights = _spec.checkpoint
backbone_pyramid_channels = _spec.pyramid_channels
backbone_trainable_layers = 0
dino_intermediate_layers = [5, 11, 17, 23]
dino_intermediate_fusion_schema = 'residual_final_v1'
progressive_unfreeze = False
initial_freeze_epochs = 0
unfreeze_interval = 0
use_checkpoint = True
in_channels_encoder = [512, 512, 512]
encoder_use_indices = [1, 2]
encoder_num_layers = 2
hidden_dim = 512
feat_channels_decoder = [512, 512, 512]
nheads = 16
dim_feedforward = 2048
expansion = 0.5
depth_mult = 1.0
dec_layers = 8
eval_idx = 7
dec_n_points = [8, 4, 2]
reg_max = 32
dropout = 0.1
batch_size_train = 4
batch_size_val = 4
recipe_reference_effective_batch_size = 8
gradient_accumulation_steps = 2
lr = 0.0002
lr_scheduler = 'cosine'
use_warmup = False
use_ema = False
distill_weight = 0.0
distill_feature_weight = 0.0
epochs = 50
