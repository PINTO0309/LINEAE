"""Accuracy-priority LINEAE 2XL with the official DINOv3 ViT-L/16 backbone."""

_base_ = ['./lineae_xl.py']

from models.lineae.variants import get_variant as _get_variant  # noqa: E402
_spec = _get_variant('2XL')

output_dir = 'outputs/lineae_2xl'
variant = '2XL'
training_profile = 'single_gpu_96gb_accuracy'
multi_scale_train = True
backbone = _spec.backbone
backbone_weights = _spec.checkpoint
backbone_pyramid_channels = _spec.pyramid_channels
backbone_trainable_layers = 4
use_checkpoint = True
batch_size_train = 4
batch_size_val = 4
recipe_reference_effective_batch_size = 8
gradient_accumulation_steps = 2
lr = 0.0002
lr_scheduler = 'cosine'
use_warmup = False
use_ema = False
dino_intermediate_layers = []
distill_weight = 0.0
distill_feature_weight = 0.0
epochs = 60
