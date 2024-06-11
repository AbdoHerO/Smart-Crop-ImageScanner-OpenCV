# Define the settings for each model
model_settings_magicpro_filter = {
    'GLOBAL': {
        'gaussian_blur': (5, 5),
        'clahe_clip_limit': 3.0,
        'clahe_tile_grid_size': (8, 8),
        'contrast_alpha': 1.2,
        'brightness_beta': 10,
        'adaptive_thresh_block_size': 85,
        'adaptive_thresh_C': 150,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1,

        'saturation_factor': 3.0,  # Increase this value to concentrate colors more
        'h': 15,  # Filter strength for luminance component
        'hForColorComponents': 0,  # Same as h but for color components
        'templateWindowSize': 0,  # Size in pixels of the template patch that is used to compute weights
        'searchWindowSize': 45,  # Size in pixels of the window that is used to compute weighted average for given pixel

        'adaptive_thresh_block_size_2': 95,
        'adaptive_thresh_C_2': 100,
        'dilate_kernel_size_2': (1, 1),
        'dilate_iterations_2': 1,

        'black_point': 10,  # Adjust black point
        # 'warmth': 2,  # Adjust warmth
        # 'tint': 2,  # Adjust tint
        # 'brilliance': 80,  # Adjust brilliance
        # 'exposure': 1.0  # Adjust exposure

        'color_palettes': {
            'custom': {
                'hue_shift': 0,
                'saturation_factor': 0.8,
                'brightness_beta': 10,
                'white_intensity': 1.0,  # New parameter to control white intensity
                'black_intensity': 0.4   # New parameter to control black intensity
            }
        }
    },
    'SOPHACA': {
        'gaussian_blur': (5, 5),
        'clahe_clip_limit': 3.0,
        'clahe_tile_grid_size': (8, 8),
        'contrast_alpha': 1.2,
        'brightness_beta': 10,
        'adaptive_thresh_block_size': 85,
        'adaptive_thresh_C': 150,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1,

        'saturation_factor': 3.0,  # Increase this value to concentrate colors more
        'h': 15,  # Filter strength for luminance component
        'hForColorComponents': 0,  # Same as h but for color components
        'templateWindowSize': 0,  # Size in pixels of the template patch that is used to compute weights
        'searchWindowSize': 45,  # Size in pixels of the window that is used to compute weighted average for given pixel

        'adaptive_thresh_block_size_2': 95,
        'adaptive_thresh_C_2': 100,
        'dilate_kernel_size_2': (1, 1),
        'dilate_iterations_2': 1,

        'black_point': 10,  # Adjust black point
        # 'warmth': 2,  # Adjust warmth
        # 'tint': 2,  # Adjust tint
        # 'brilliance': 80,  # Adjust brilliance
        # 'exposure': 1.0  # Adjust exposure

        'color_palettes': {
            'custom': {
                'hue_shift': 0,
                'saturation_factor': 0.8,
                'brightness_beta': 10,
                'white_intensity': 1.0,  # New parameter to control white intensity
                'black_intensity': 0.4   # New parameter to control black intensity
            }
        }
    },
    'SPR': {
        'gaussian_blur': (9, 9),
        'clahe_clip_limit': 2,
        'clahe_tile_grid_size': (20, 20),
        'contrast_alpha': 1.1,
        'brightness_beta': 30,
        'adaptive_thresh_block_size': 85,
        'adaptive_thresh_C': 100,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1,
        'saturation_factor': 3.0,
        'h': 5,
        'hForColorComponents': 0,
        'templateWindowSize': 7,
        'searchWindowSize': 1,
        'adaptive_thresh_block_size_2': 95,
        'adaptive_thresh_C_2': 100,
        'dilate_kernel_size_2': (1, 1),
        'dilate_iterations_2': 1,
        'black_point': 10,
        'color_palettes': {
            'custom': {
                'hue_shift': 0,
                'saturation_factor': 0.4,
                'brightness_beta': 0,
                'white_intensity': 1.0,
                'black_intensity': 0.1,
            },
        },
            },
    'GPM': {
        'gaussian_blur': (3, 3),
        'clahe_clip_limit': 1.5,
        'clahe_tile_grid_size': (20, 20),
        'contrast_alpha': 1.1,
        'brightness_beta': 30,
        'adaptive_thresh_block_size': 85,
        'adaptive_thresh_C': 150,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1,

        'saturation_factor': 3.0,  # Increase this value to concentrate colors more
        'h': 5,  # Filter strength for luminance component
        'hForColorComponents': 0,  # Same as h but for color components
        'templateWindowSize': 0,  # Size in pixels of the template patch that is used to compute weights
        'searchWindowSize': 1,  # Size in pixels of the window that is used to compute weighted average for given pixel

        'adaptive_thresh_block_size_2': 95,
        'adaptive_thresh_C_2': 100,
        'dilate_kernel_size_2': (1, 1),
        'dilate_iterations_2': 1,

        'black_point': 10,
        'color_palettes': {
            'custom': {
                'hue_shift': 0,
                'saturation_factor': 0.4,
                'brightness_beta': 0,
                'white_intensity': 1.0,  # New parameter to control white intensity
                'black_intensity': 0.1   # New parameter to control black intensity
            }
        }
    },
    'SOPHADIMS': {
        'gaussian_blur': (5, 5),
        'clahe_clip_limit': 2.8,
        'clahe_tile_grid_size': (100, 100),
        'contrast_alpha': 1.1,
        'brightness_beta': 10,
        'adaptive_thresh_block_size': 85,
        'adaptive_thresh_C': 100,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1,

        'saturation_factor': 4.0,  # Increase this value to concentrate colors more
        'h': 15,  # Filter strength for luminance component
        'hForColorComponents': 0,  # Same as h but for color components
        'templateWindowSize': 0,  # Size in pixels of the template patch that is used to compute weights
        'searchWindowSize': 45,  # Size in pixels of the window that is used to compute weighted average for given pixel

        'adaptive_thresh_block_size_2': 95,
        'adaptive_thresh_C_2': 100,
        'dilate_kernel_size_2': (1, 1),
        'dilate_iterations_2': 1,

        'black_point': 10,  # Adjust black point
        # 'warmth': 2,  # Adjust warmth
        # 'tint': 2,  # Adjust tint
        # 'brilliance': 80,  # Adjust brilliance
        # 'exposure': 1.0  # Adjust exposure

        'color_palettes': {
            'custom': {
                'hue_shift': 5,
                'saturation_factor': 0.7,
                'brightness_beta': 0,
                'white_intensity': 1.0,  # New parameter to control white intensity
                'black_intensity': 1.0   # New parameter to control black intensity
            }
        }
    },
    'RECAMED': {
        'gaussian_blur': (5, 5),
        'clahe_clip_limit': 2.8,
        'clahe_tile_grid_size': (100, 100),
        'contrast_alpha': 1.1,
        'brightness_beta': 10,
        'adaptive_thresh_block_size': 85,
        'adaptive_thresh_C': 100,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1,

        'saturation_factor': 4.0,  # Increase this value to concentrate colors more
        'h': 15,  # Filter strength for luminance component
        'hForColorComponents': 0,  # Same as h but for color components
        'templateWindowSize': 0,  # Size in pixels of the template patch that is used to compute weights
        'searchWindowSize': 45,  # Size in pixels of the window that is used to compute weighted average for given pixel

        'adaptive_thresh_block_size_2': 95,
        'adaptive_thresh_C_2': 100,
        'dilate_kernel_size_2': (1, 1),
        'dilate_iterations_2': 1,

        'black_point': 10,  # Adjust black point
        # 'warmth': 2,  # Adjust warmth
        # 'tint': 2,  # Adjust tint
        # 'brilliance': 80,  # Adjust brilliance
        # 'exposure': 1.0  # Adjust exposure

        'color_palettes': {
            'custom': {
                'hue_shift': 5,
                'saturation_factor': 0.7,
                'brightness_beta': 0,
                'white_intensity': 1.0,  # New parameter to control white intensity
                'black_intensity': 1.0   # New parameter to control black intensity
            }
        }
    },
    'COOPER': {
        'gaussian_blur': (5, 5),
        'clahe_clip_limit': 2.8,
        'clahe_tile_grid_size': (100, 100),
        'contrast_alpha': 1.1,
        'brightness_beta': 10,
        'adaptive_thresh_block_size': 85,
        'adaptive_thresh_C': 100,
        'dilate_kernel_size': (1, 1),
        'dilate_iterations': 1,

        'saturation_factor': 4.0,  # Increase this value to concentrate colors more
        'h': 15,  # Filter strength for luminance component
        'hForColorComponents': 0,  # Same as h but for color components
        'templateWindowSize': 0,  # Size in pixels of the template patch that is used to compute weights
        'searchWindowSize': 45,  # Size in pixels of the window that is used to compute weighted average for given pixel

        'adaptive_thresh_block_size_2': 95,
        'adaptive_thresh_C_2': 100,
        'dilate_kernel_size_2': (1, 1),
        'dilate_iterations_2': 1,

        'black_point': 10,  # Adjust black point
        # 'warmth': 2,  # Adjust warmth
        # 'tint': 2,  # Adjust tint
        # 'brilliance': 80,  # Adjust brilliance
        # 'exposure': 1.0  # Adjust exposure

        'color_palettes': {
            'custom': {
                'hue_shift': 5,
                'saturation_factor': 0.7,
                'brightness_beta': 0,
                'white_intensity': 1.0,  # New parameter to control white intensity
                'black_intensity': 1.0   # New parameter to control black intensity
            }
        }
    }
}

