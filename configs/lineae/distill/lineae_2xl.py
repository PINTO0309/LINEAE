_base_ = ['../lineae_2xl.py']

output_dir = 'outputs/lineae_2xl_distill'
distill_weight = 1.0
distill_teacher_config = 'configs/lineae/lineae_3xl.py'
distill_teacher_checkpoint = 'ckpts/lineae_3xl_teacher.pth'
batch_size_train = 4
gradient_accumulation_steps = 2
scheduler_step_unit = 'optimizer'
