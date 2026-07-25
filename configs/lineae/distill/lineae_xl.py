_base_ = ['../lineae_xl.py']

output_dir = 'outputs/lineae_xl_distill'
distill_weight = 1.0
distill_teacher_config = 'configs/lineae/lineae_3xl.py'
distill_teacher_checkpoint = 'ckpts/lineae_3xl_teacher.pth'
batch_size_train = 8
gradient_accumulation_steps = 1
scheduler_step_unit = 'optimizer'
