import cv2
import numpy as np
from matplotlib import pyplot as plt
from wand.image import Image as WandImage

import constants

def straighten_image_with_imagemagick(image_path):
    with WandImage(filename=image_path) as img:
        img.deskew(0.4 * img.quantum_range)  # Deskew the image
        img_path = 'temp_straightened_image.png'
        img.save(filename=img_path)
        return img_path

def show_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def apply_magic_pro_filter(image_path, output_path, model):
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

    # Save the result
    cv2.imwrite(output_path, final_image)
    print(f"Filtered image saved to {output_path}")

if __name__ == "__main__":
    input_image_path = 'output/cropped_spr_iphn.jpg'
    output_image_path = 'output/filtered_cropped_spr_iphn.jpg'
    apply_magic_pro_filter(input_image_path, output_image_path, 'SPR')
