_base_ = ['../lineae_s.py']

output_dir = 'outputs/lineae_s_distill'
distill_weight = 1.0
training_profile = 'single_gpu_96gb'
multi_scale_train = True
batch_size_train = 8
batch_size_val = 8
recipe_reference_effective_batch_size = 8
gradient_accumulation_steps = 1
scheduler_step_unit = 'optimizer'
