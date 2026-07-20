lr = 0.00025
weight_decay = 0.000125
betas = [0.9, 0.999]
# PyTorch fused AdamW is used on CUDA; CPU diagnostics fall back to the same
# AdamW equations without requesting an unavailable fused kernel.
optimizer_fused = True

epochs = 12
lr_drop_list = [11]
clip_max_norm = 0.1
scheduler_step_unit = 'epoch'

save_checkpoint_interval = 10
