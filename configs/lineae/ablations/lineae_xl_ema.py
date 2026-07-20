"""XL EMA ablation; compare against the unchanged XL supervised control."""

_base_ = ['../lineae_xl.py']

output_dir = 'outputs/ablations/lineae_xl_ema'
use_ema = True
ema_decay = 0.9997
ema_epoch = 0
eval_ema = True
