# model
modelname = 'LINEAE'
eval_spatial_size = (640, 640)
eval_idx = 5 # 6 decoder layers
num_classes = 2

## backbone
pretrained = True
use_checkpoint = False
return_interm_indices = [1, 2, 3]
freeze_norm = True
freeze_stem_only = True
backbone_weights = None
backbone_trainable_layers = 2
backbone_pyramid_channels = None
enforce_variant_input = True
enforce_variant_pyramid = True
# Optional P4-resolution fusion of selected 0-based transformer blocks.
# Keep empty for the baseline; e.g. [3, 7, 11] is an independent ablation.
dino_intermediate_layers = []

# Pretrained DINOv3 models use ImageNet normalization. HGNet variants override
# these values with the original LINEA statistics.
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

## encoder
hybrid_encoder = 'hybrid_encoder_asymmetric_conv'
in_channels_encoder = [512, 1024, 2048]
pe_temperatureH = 20
pe_temperatureW = 20

## encoder
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True

## decoder
num_queries = 1100
query_dim = 4
num_feature_levels = 3
dec_n_points = [4, 1, 1] 
dropout = 0.0
pre_norm = False

# denoise
use_dn = True
dn_number = 300
dn_line_noise_scale = 1.0
dn_label_noise_ratio = 0.5
embed_init_tgt = True
dn_labelbook_size = 2
match_unstable_error = True

# matcher
set_cost_class = 2.0
set_cost_lines = 5.0

# criterion
criterionname = 'LINEACRITERION'
criterion_type = 'default'
endpoint_invariant_lines = True
weight_dict = {'loss_logits': 1, 'loss_line': 5}
losses = ['labels', 'lines'] 
focal_alpha = 0.1

matcher_type = 'HungarianMatcher' # or SimpleMinsumMatcher
nms_iou_threshold = -1

# Validation model selection. The latest full state is always saved separately.
selection_metric = 'sap10'
selection_mode = 'max'

# Output-level line-set distillation.  A zero total weight is an exact no-KD
# path and does not construct or run a teacher.
distill_weight = 0.0
distill_teacher_config = 'configs/lineae/lineae_xl.py'
distill_teacher_checkpoint = 'ckpts/lineae_xl_teacher.pth'
# Qualified v2 teacher artifacts bind this canonical inference config by hash.
# Raw/unqualified checkpoints are rejected by default.
distill_allow_unqualified_teacher = False
# Match Gazelle's cross-variant KD path: run the teacher at the canonical input
# declared by its own config. This matters for A/F and tuning candidates whose
# student resolution differs from XL's 640x640 input.
distill_teacher_resize = True
distill_confidence_threshold = 0.3
distill_top_k = 300
distill_match_cost_class = 2.0
distill_match_cost_line = 5.0
distill_class_weight = 1.0
distill_line_weight = 5.0
# Optional projected P3/P4/P5 KD. It is disabled in every default experiment
# and can be enabled independently after the output-KD control has completed.
distill_feature_weight = 0.0
distill_feature_loss = 'cosine'
distill_teacher_feature_channels = [256, 256, 256]
# Optional exact-input disk cache. The key includes every augmented input byte,
# teacher/config hashes and normalization, so random transforms never alias.
distill_teacher_cache_dir = ''
distill_teacher_cache_read_only = False
distill_warmup_steps = 1000
# Gazelle progressively softens teacher targets from T=1 to T=4. ``-1``
# resolves to the final optimizer-step index of the actual full LINEAE run,
# accounting for dataset length, per-rank batch size, and accumulation.
distill_temperature_start = 1.0
distill_temperature_end = 4.0
distill_temperature_steps = -1

# for ema
use_ema = False
ema_decay = 0.9997
ema_epoch = 0
eval_ema = True
