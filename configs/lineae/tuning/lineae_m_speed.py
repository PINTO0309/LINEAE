_base_ = ['../distill/lineae_m.py']

from models.lineae.tuning import get_tuning_candidate as _get_candidate  # noqa: E402
_candidate = _get_candidate('M', 'speed')

output_dir = 'outputs/lineae_m_tune_speed'
training_profile = 'single_gpu_96gb_tuning'
enforce_variant_input = False
eval_spatial_size = (_candidate.input_size, _candidate.input_size)
data_aug_scales = [eval_spatial_size]
num_queries = _candidate.num_queries
num_select = _candidate.num_select
dec_layers = _candidate.decoder_layers
eval_idx = dec_layers - 1
