from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2


def apply_magicpro_filter(image_path, contrast=0, brightness=0, details=100):
    # Load image
    image = Image.open(image_path)

    # Convert to numpy array
    image_np = np.array(image)

    # Apply noise reduction using OpenCV
    image_np = cv2.fastNlMeansDenoisingColored(image_np, None, 30, 30, 7, 21)

    # Convert back to PIL image
    image = Image.fromarray(image_np)

    # Apply contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1 + (contrast / 100.0))

    # Apply brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1 + (brightness / 100.0))

    # Apply sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(details / 50.0)  # Scaling to make it more effective

    return image


# Usage example
image_path = 'output/cropped_spr_iphn.jpg'  # cropped_model05.jpeg
output_image_path = 'output/filtered_2.jpg'
filtered_image = apply_magicpro_filter(image_path, contrast=0, brightness=0, details=50)
filtered_image.save(output_image_path)
filtered_image.show()
