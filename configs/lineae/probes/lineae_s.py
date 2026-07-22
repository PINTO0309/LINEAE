"""Bounded batch-1 S pipeline probe; not the full supervised S recipe."""

_base_ = ['../lineae_s.py']

output_dir = 'outputs/lineae_s_probe'
training_profile = 'p0_smoke'
multi_scale_train = False
batch_size_train = 1
batch_size_val = 1
recipe_reference_effective_batch_size = 1
gradient_accumulation_steps = 1
scheduler_step_unit = 'epoch'
epochs = 36
use_checkpoint = False
