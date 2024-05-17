# Define the settings for each model
model_settings_magicpro_filter = {
    'GLOBAL': {
        'gaussian_blur': (3, 3),
        'clahe_clip_limit': 4.0,
        'clahe_tile_grid_size': (40, 40),
        'contrast_alpha': 1.5,
        'brightness_beta': 30,
        'adaptive_thresh_block_size': 11,
        'adaptive_thresh_C': 30,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 3
    },
    'SOPHACA': {
        'gaussian_blur': (3, 3),
        'clahe_clip_limit': 6.0,
        'clahe_tile_grid_size': (30, 30),
        'contrast_alpha': 1.7,
        'brightness_beta': 40,
        'adaptive_thresh_block_size': 11,
        'adaptive_thresh_C': 25,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 2
    },
    'SPR': {
        'gaussian_blur': (3, 3),
        'clahe_clip_limit': 5.0,
        'clahe_tile_grid_size': (30, 30),
        'contrast_alpha': 1.6,
        'brightness_beta': 35,
        'adaptive_thresh_block_size': 11,
        'adaptive_thresh_C': 28,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 3
    },
    'SOPHADIMS': {
        'gaussian_blur': (3, 3),
        'clahe_clip_limit': 4.0,
        'clahe_tile_grid_size': (40, 40),
        'contrast_alpha': 1.5,
        'brightness_beta': 30,
        'adaptive_thresh_block_size': 11,
        'adaptive_thresh_C': 30,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1
    },
    'GPM': {
        'gaussian_blur': (3, 3),
        'clahe_clip_limit': 5.0,
        'clahe_tile_grid_size': (30, 30),
        'contrast_alpha': 1.6,
        'brightness_beta': 35,
        'adaptive_thresh_block_size': 11,
        'adaptive_thresh_C': 28,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 3
    },
    'RECAMED': {
        'gaussian_blur': (3, 3),
        'clahe_clip_limit': 6.0,
        'clahe_tile_grid_size': (30, 30),
        'contrast_alpha': 1.7,
        'brightness_beta': 40,
        'adaptive_thresh_block_size': 11,
        'adaptive_thresh_C': 25,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 2
    },
    'COOPER': {
        'gaussian_blur': (3, 3),
        'clahe_clip_limit': 4.0,
        'clahe_tile_grid_size': (40, 40),
        'contrast_alpha': 1.5,
        'brightness_beta': 30,
        'adaptive_thresh_block_size': 11,
        'adaptive_thresh_C': 30,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 3
    }
}