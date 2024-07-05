import cv2
import numpy as np
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_image_quality(image_path):
    result = {
        "status": "success",
        "processed_image": None,
        "messages": []
    }

    # Check file size
    file_size = os.path.getsize(image_path)
    min_size, max_size = 1000000, 10000000
    if not (min_size <= file_size <= max_size):
        result["status"] = "error"
        result["messages"].append(f"Image size ({file_size} bytes) is not within acceptable range ({min_size} - {max_size} bytes)")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        result["status"] = "error"
        result["messages"].append("Failed to read image file")
        return result

    # Check resolution
    height, width = image.shape[:2]
    min_width, min_height = 1000, 1000
    if not (width >= min_width and height >= min_height):
        result["status"] = "error"
        result["messages"].append(f"Image resolution ({width}x{height}) is below minimum required ({min_width}x{min_height})")

    # Check contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.meanStdDev(gray)[1][0][0]
    min_contrast = 40
    if contrast < min_contrast:
        result["status"] = "error"
        result["messages"].append(f"Image contrast ({contrast:.2f}) is below minimum required ({min_contrast})")

    # Check lighting
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()
    min_brightness, max_brightness = 100, 250
    if not (min_brightness <= brightness <= max_brightness):
        result["status"] = "error"
        result["messages"].append(f"Image brightness ({brightness:.2f}) is not within acceptable range ({min_brightness} - {max_brightness})")

    # Clean image
    cleaned_image = clean_image(image)

    # Check orientation
    if not check_orientation(cleaned_image):
        cleaned_image = cv2.rotate(cleaned_image, cv2.ROTATE_90_CLOCKWISE)
        result["messages"].append("Image was rotated to landscape orientation")

    if result["status"] == "success":
        result["processed_image"] = cleaned_image
        result["messages"].append("Image preprocessing completed successfully")
    else:
        result["messages"].insert(0, "Please improve the image quality and try again")

    return result

# Helper functions (unchanged)
def clean_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def check_orientation(image):
    height, width = image.shape[:2]
    return width > height

# Usage
image_path = 'images/BLS_3.jpg'
result = check_image_quality(image_path)

if result["status"] == "success":
    print("Image processing successful")
    processed_image = result["processed_image"]
    # Continue with OCR processing using processed_image
else:
    print("Image processing failed. Please address the following issues:")
    for message in result["messages"]:
        print(f"- {message}")