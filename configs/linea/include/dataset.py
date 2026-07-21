data_aug_scales = [(640, 640)]
image_preprocess_schema = 'opencv_rgb_inter_linear_v2'
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]


data_aug_scale_overlap = None
use_photometric_distort = False
photometric_distort_probability = 0.5
batch_size_train = 8
batch_size_val = 64
pin_memory = True
prefetch_factor = 2
