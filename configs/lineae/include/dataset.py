data_aug_scales = [(640, 640)]
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]


data_aug_scale_overlap = None
# Geometry-preserving Gazelle photometric augmentation. Disabled for matched
# baselines and enabled only as an explicit ablation.
use_photometric_distort = False
photometric_distort_probability = 0.5
batch_size_train = 8
batch_size_val = 8
pin_memory = True
prefetch_factor = 2
# Large detection batches contain many independently sized target tensors.
# file_system avoids retaining one open file descriptor per shared storage.
multiprocessing_sharing_strategy = 'file_system'
