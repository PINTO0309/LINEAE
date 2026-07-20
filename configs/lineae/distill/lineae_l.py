_base_ = ['../lineae_l.py']

output_dir = 'outputs/lineae_l_distill'
distill_weight = 1.0
batch_size_train = 8
gradient_accumulation_steps = 1
scheduler_step_unit = 'optimizer'
epochs = 30
