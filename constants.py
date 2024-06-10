# Define the settings for each model
model_settings_magicpro_filter = {
    'GLOBAL': {
        'gaussian_blur': (5, 5),
        'clahe_clip_limit': 4.5,
        'clahe_tile_grid_size': (5, 5),
        'contrast_alpha': 1.5,
        'brightness_beta': 10,
        'adaptive_thresh_block_size': 15,
        'adaptive_thresh_C': 50,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1
    },
    'SOPHACA': {
        'gaussian_blur': (5, 5),
        'clahe_clip_limit': 4.5,
        'clahe_tile_grid_size': (5, 5),
        'contrast_alpha': 1.5,
        'brightness_beta': 10,
        'adaptive_thresh_block_size': 15,
        'adaptive_thresh_C': 50,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1
    },
    'SPR': {
        'gaussian_blur': (5, 5),
        'clahe_clip_limit': 2.5,
        'clahe_tile_grid_size': (100, 100),
        'contrast_alpha': 1.1,
        'brightness_beta': 20,
        'adaptive_thresh_block_size': 15,
        'adaptive_thresh_C': 100,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1,
        'saturation_factor': 2.0,  # Increase this value to concentrate colors more
        'h': 10,  # Filter strength for luminance component
        'hForColorComponents': 10,  # Same as h but for color components
        'templateWindowSize': 0,  # Size in pixels of the template patch that is used to compute weights
        'searchWindowSize': 41  # Size in pixels of the window that is used to compute weighted average for given pixel

    },
    'SOPHADIMS': {
        'gaussian_blur': (5, 5),
        'clahe_clip_limit': 4.5,
        'clahe_tile_grid_size': (50, 50),
        'contrast_alpha': 1.4,
        'brightness_beta': 50,
        'adaptive_thresh_block_size': 11,
        'adaptive_thresh_C': 30,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 3
    },
    'GPM': {
        'gaussian_blur': (5, 5),
        'clahe_clip_limit': 1.5,
        'clahe_tile_grid_size': (30, 30),
        'contrast_alpha': 1.3,
        'brightness_beta': 30,
        'adaptive_thresh_block_size': 15,
        'adaptive_thresh_C': 30,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1
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
        'clahe_clip_limit': 3.5,
        'clahe_tile_grid_size': (35, 35),
        'contrast_alpha': 1.4,
        'brightness_beta': 35,
        'adaptive_thresh_block_size': 9,
        'adaptive_thresh_C': 30,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 3
    }
}