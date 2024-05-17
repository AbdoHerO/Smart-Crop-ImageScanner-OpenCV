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

    # # Straighten the image using ImageMagick
    # straightened_image_path = straighten_image_with_imagemagick(image_path)
    # straightened_image = cv2.imread(straightened_image_path)
    # assert straightened_image is not None, "Failed to load straightened image"
    #
    # # Show the straightened image
    # show_image(straightened_image, title="Straightened Image")

    """ Clear Magic Pro Filter  """
    # Apply Gaussian Blur to reduce noise
    # blurred = cv2.GaussianBlur(straightened_image, settings['gaussian_blur'], 0)
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

    """ Clear the white fog or beam effect in the image """

    """ Increase contrast and brightness (lighting) """
    # Increase contrast and brightness
    adjusted = cv2.convertScaleAbs(enhanced_image, alpha=settings['contrast_alpha'], beta=settings['brightness_beta'])

    # Apply sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)

    """ Create Mask of the Thickened text """
    # Convert to grayscale
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                   settings['adaptive_thresh_block_size'], settings['adaptive_thresh_C'])

    # Use morphological operations to thicken the text
    kernel = np.ones(settings['dilate_kernel_size'], np.uint8)
    thickened = cv2.dilate(binary, kernel, iterations=settings['dilate_iterations'])

    """ Apply the Mask of the Thickened text """
    # Merge thickened text back with the original color image
    colored_thickened = cv2.bitwise_and(sharpened, sharpened, mask=thickened)

    """ Save the image transformed """
    # Save the result
    cv2.imwrite(output_path, colored_thickened)
    print(f"Filtered image saved to {output_path}")


if __name__ == "__main__":
    # input_image_path = 'output/BL_scanned_1_1.jpg'  # cropped_model05.jpeg
    input_image_path = 'output/cropped_model02.jpeg'  # cropped_model05.jpeg
    output_image_path = 'output/filtered_image.jpg'
    apply_magic_pro_filter(input_image_path, output_image_path, 'COOPER')
