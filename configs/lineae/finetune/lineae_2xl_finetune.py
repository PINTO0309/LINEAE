"""Balanced 2XL fine-tuning with auto-discovered ensemble datasets."""

_base_ = ['../lineae_2xl.py']

output_dir = 'outputs/lineae_2xl-finetune-seed42'
training_profile = 'single_gpu_96gb_ensemble_finetune'

# Keep the completed 2XL model fully trainable, but reduce both optimizer
# tiers for conservative domain adaptation from --init-checkpoint weights.
backbone_trainable_layers = 0
progressive_unfreeze = False
initial_freeze_epochs = 0
unfreeze_interval = 0
model_parameters = [
    {
        'params': '^backbone\\.core\\.(?!.*(?:norm|bias)).*$',
        'lr': 0.000001,
    },
    {
        'params': '^backbone\\.core\\.(?=.*(?:norm|bias)).*$',
        'lr': 0.000001,
        'weight_decay': 0.0,
    },
    {
        'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$',
        'weight_decay': 0.0,
    },
]
lr = 0.00002
weight_decay = 0.0001
clip_max_norm = 0.1

epochs = 12
lr_scheduler = 'cosine'
scheduler_step_unit = 'optimizer'
use_warmup = True
# 5,497 samples / physical batch 4 / accumulation 2 = 688 optimizer steps.
warmup_iters = 688
min_lr = 0.0000001

batch_size_train = 4
batch_size_val = 4
recipe_reference_effective_batch_size = 8
gradient_accumulation_steps = 2

distill_weight = 0.0
distill_feature_weight = 0.0
use_ema = False
