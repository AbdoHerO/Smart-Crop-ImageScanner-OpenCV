import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import Tk, Scale, HORIZONTAL, Button, Label
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk
import constants

def apply_color_palette(image, palette_settings):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply hue shift
    h = (h.astype(int) + palette_settings.get('hue_shift', 0)) % 180
    h = h.astype(np.uint8)

    # Apply saturation factor
    s = cv2.multiply(s, palette_settings.get('saturation_factor', 1.0))

    # Merge the channels back and convert to BGR
    hsv_colored = cv2.merge([h, s, v])
    image = cv2.cvtColor(hsv_colored, cv2.COLOR_HSV2BGR)

    # Apply brightness adjustment
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=palette_settings.get('brightness_beta', 0))

    # Apply white intensity adjustment
    white_intensity = palette_settings.get('white_intensity', 1.0)
    white_mask = cv2.inRange(image, np.array([200, 200, 200]), np.array([255, 255, 255]))
    if white_intensity != 1.0:
        image[white_mask != 0] = cv2.convertScaleAbs(image[white_mask != 0], alpha=white_intensity, beta=0)

    # Apply black intensity adjustment
    black_intensity = palette_settings.get('black_intensity', 1.0)
    black_mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([50, 50, 50]))
    if black_intensity != 1.0:
        image[black_mask != 0] = cv2.convertScaleAbs(image[black_mask != 0], alpha=black_intensity, beta=0)

    return image


def apply_magic_pro_filter(image_path, output_path, model, palette_name=None):
    settings = constants.model_settings_magicpro_filter[model]

    image = cv2.imread(image_path)
    assert image is not None, "Failed to load image"

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, settings['gaussian_blur'], 0)

    # Convert to LAB color space to work with brightness
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=settings['clahe_clip_limit'], tileGridSize=settings['clahe_tile_grid_size'])
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channels
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Apply noise reduction
    denoised_image = cv2.fastNlMeansDenoisingColored(enhanced_image, None, settings['h'], settings['hForColorComponents'], settings['templateWindowSize'], settings['searchWindowSize'])

    # Increase contrast and brightness
    adjusted = cv2.convertScaleAbs(denoised_image, alpha=settings['contrast_alpha'], beta=settings['brightness_beta'])

    # Apply sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)

    # Convert to grayscale
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                   settings['adaptive_thresh_block_size'], settings['adaptive_thresh_C'])

    # Use morphological operations to thicken the text
    kernel = np.ones(settings['dilate_kernel_size'], np.uint8)
    thickened = cv2.dilate(binary, kernel, iterations=settings['dilate_iterations'])

    # Merge thickened text back with the original color image
    colored_thickened = cv2.bitwise_and(sharpened, sharpened, mask=thickened)

    # Convert to HSV color space
    hsv = cv2.cvtColor(colored_thickened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Increase saturation
    s = cv2.multiply(s, settings['saturation_factor'])

    # Merge the channels back and convert to BGR
    hsv_colored = cv2.merge([h, s, v])
    final_image = cv2.cvtColor(hsv_colored, cv2.COLOR_HSV2BGR)

    """-----------thicken the font-----------"""

    # Apply additional filter to thicken the font further
    gray_final = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    binary_final = cv2.adaptiveThreshold(gray_final, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                         settings['adaptive_thresh_block_size_2'], settings['adaptive_thresh_C_2'])
    kernel_final = np.ones(settings['dilate_kernel_size_2'], np.uint8)
    thickened_final = cv2.dilate(binary_final, kernel_final, iterations=settings['dilate_iterations_2'])

    final_image_colored_thickened = cv2.bitwise_and(final_image, final_image, mask=thickened_final)


    """------------black_point----------"""
    # Convert to LAB color space
    lab = cv2.cvtColor(final_image_colored_thickened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply Black Point adjustment
    black_point = settings['black_point']
    l = cv2.addWeighted(l, 1.0, l, 0, -black_point)

    lab = cv2.merge((l, a, b))
    final_image_colored_thickened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    """ -----------color picker----------- """
    # Apply color palette if specified
    if palette_name and palette_name in settings['color_palettes']:
        palette_settings = settings['color_palettes'][palette_name]
        final_image_colored_thickened = apply_color_palette(final_image_colored_thickened, palette_settings)


    # Save the result
    cv2.imwrite(output_path, final_image_colored_thickened)
    print(f"Filtered image saved to {output_path}")


if __name__ == "__main__":
    palette_name = 'custom'  # Change to the desired palette name

    # input_image_path = 'output/cropped_sophaca_iphn.jpg'
    # output_image_path = 'output/filtered_cropped_sophaca_iphn.jpg'
    # apply_magic_pro_filter(input_image_path, output_image_path, 'SOPHACA', palette_name)

    # input_image_path = 'output/cropped_spr_iphn.jpg'
    # output_image_path = 'output/filtered_cropped_spr_iphn.jpg'
    # apply_magic_pro_filter(input_image_path, output_image_path, 'SPR', palette_name)

    input_image_path = 'output/cropped_model01.jpeg'
    output_image_path = 'output/filtered_cropped_model01.jpeg'
    apply_magic_pro_filter(input_image_path, output_image_path, 'GPM', palette_name)


