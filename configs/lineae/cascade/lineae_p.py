_base_ = ['../distill/lineae_p.py']

output_dir = 'outputs/lineae_p_cascade_x'
distill_teacher_config = 'configs/lineae/lineae_x.py'
distill_teacher_checkpoint = 'ckpts/lineae_x_teacher.pth'
