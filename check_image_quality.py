import cv2
import numpy as np
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_size_bytes(image_path, min_size=1000000, max_size=10000000):
    file_size = os.path.getsize(image_path)
    logger.info(f"Image size: {file_size} bytes. Expected range: {min_size} - {max_size} bytes")
    return min_size <= file_size <= max_size


def check_resolution(image, min_width=1000, min_height=1000):
    height, width = image.shape[:2]
    logger.info(f"Image resolution: {width}x{height}. Minimum expected: {min_width}x{min_height}")
    return width >= min_width and height >= min_height


def check_contrast(image, min_contrast=40):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.meanStdDev(gray)[1][0][0]
    logger.info(f"Image contrast: {contrast:.2f}. Minimum expected: {min_contrast}")
    return contrast >= min_contrast


def check_lighting(image, min_brightness=100, max_brightness=250):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()
    logger.info(f"Image brightness: {brightness:.2f}. Expected range: {min_brightness} - {max_brightness}")
    return min_brightness <= brightness <= max_brightness


def clean_image(image):
    logger.info("Cleaning image...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    logger.info("Image cleaning completed")
    return binary


def check_orientation(image):
    height, width = image.shape[:2]
    orientation = "landscape" if width > height else "portrait"
    logger.info(f"Image orientation: {orientation}. Expected: landscape")
    return width > height


def preprocess_image(image_path):
    logger.info(f"Starting preprocessing for image: {image_path}")

    if not check_size_bytes(image_path):
        logger.error("Image size check failed")
        return None, "Image size is not within acceptable range"

    image = cv2.imread(image_path)

    if not check_resolution(image):
        logger.error("Image resolution check failed")
        return None, "Image resolution is too low"

    if not check_contrast(image):
        logger.error("Image contrast check failed")
        return None, "Image contrast is too low"

    if not check_lighting(image):
        logger.error("Image lighting check failed")
        return None, "Image lighting is not optimal"

    cleaned_image = clean_image(image)

    if not check_orientation(cleaned_image):
        logger.info("Rotating image to landscape orientation")
        cleaned_image = cv2.rotate(cleaned_image, cv2.ROTATE_90_CLOCKWISE)

    logger.info("Image preprocessing completed successfully")
    return cleaned_image, "Image preprocessed successfully"


# Usage
image_path = 'images/new_gpm.jpeg'
processed_image, message = preprocess_image(image_path)

if processed_image is not None:
    print(message)
    # Continue with OCR processing
else:
    print(f"Error: {message}")