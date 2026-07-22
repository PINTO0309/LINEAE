_base_ = [
    './include/dataset.py',
    './include/optimizer.py',
    './include/lineae.py',
]

from models.lineae.variants import get_variant as _get_variant  # noqa: E402
_spec = _get_variant('S')

output_dir = 'outputs/lineae_s'
variant = 'S'
training_profile = 'single_gpu_96gb'

# DINOv3 ViT-Tiny backbone for the full supervised S recipe.
backbone = _spec.backbone
backbone_weights = _spec.checkpoint
backbone_trainable_layers = 2
backbone_pyramid_channels = _spec.pyramid_channels
pretrained = True
use_checkpoint = False

image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

feat_strides = [8, 16, 32]
in_channels_encoder = [192, 192, 192]
hidden_dim = 256
dim_feedforward = 512
nheads = 8
use_lmap = False
multi_scale_train = True
batch_size_train = 8
batch_size_val = 8
recipe_reference_effective_batch_size = 8

hybrid_encoder = 'hybrid_encoder_asymmetric_conv'
pe_temperatureH = 20
pe_temperatureW = 20
expansion = 0.34
depth_mult = 0.5

feat_channels_decoder = [hidden_dim, hidden_dim, hidden_dim]
dec_layers = 3
num_queries = 1100
num_select = 300
reg_max = 16
reg_scale = 4
eval_idx = 2

epochs = 45
lr_drop_list = [25]
weight_dict = {'loss_logits': 2, 'loss_line': 5}
use_warmup = False
warmup_iters = 625 * 5
amp_init_scale = 1024.0
lr_scheduler = 'cosine'
min_lr = 0.0000001
progressive_unfreeze = True
initial_freeze_epochs = 5
unfreeze_interval = 2
gradient_accumulation_steps = 1
scheduler_step_unit = 'optimizer'

model_parameters = [
    {
        'params': '^backbone\\.core\\.(?!.*(?:norm|bias)).*$',
        'lr': 0.00001,
    },
    {
        'params': '^backbone\\.core\\.(?=.*(?:norm|bias)).*$',
        'lr': 0.00001,
        'weight_decay': 0.0,
    },
    {
        'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$',
        'weight_decay': 0.0,
    },
]
lr = 0.0002
betas = [0.9, 0.999]
weight_decay = 0.0001
