"""XL frozen-backbone control for teacher-recipe selection."""

_base_ = ['../lineae_xl.py']

output_dir = 'outputs/ablations/lineae_xl_frozen'
backbone_trainable_layers = -1
progressive_unfreeze = False
initial_freeze_epochs = 0
unfreeze_interval = 0
use_checkpoint = False
