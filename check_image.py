from PIL import Image, ImageEnhance
import cv2
import numpy as np
from skimage import exposure
from skimage.feature import canny
import os


def check_size_bytes(image_path, min_size_mb=1.5):
    try:
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # Convert bytes to MB
        if file_size < min_size_mb:
            return False, f"Error - Taille de l'image trop grande : {file_size:.2f} MB, taille maximale autorisée : {min_size_mb} MB"
        return True, f"Done - Taille de l'image adéquate : {file_size:.2f} MB"
    except Exception as e:
        return False, f"Error - Problème de taille de l'image : {str(e)}"


def check_resolution(image_path, min_dpi=300):
    try:
        image = Image.open(image_path)
        dpi = image.info.get('dpi', (72, 72))  # Default to 72 DPI if not found
        if dpi[0] == 72 and dpi[1] == 72 and image.info.get('dpi'):
            return True, "Warning - DPI information not found; assuming adequate resolution"
        if dpi[0] < min_dpi or dpi[1] < min_dpi:
            return False, f"Error - Résolution insuffisante : {dpi} DPI, minimum requis : {min_dpi} DPI"
        return True, "Done - Résolution adéquate"
    except Exception as e:
        return False, f"Error - Problème de résolution : {str(e)}"


def check_contrast(image_path, threshold=10):
    try:
        image = Image.open(image_path).convert('L')
        np_image = np.array(image)
        contrast = np.std(np_image)
        if contrast < threshold:
            return False, f"Error - Contraste insuffisant, écart type : {contrast}, seuil requis : {threshold}"
        return True, "Done - Contraste adéquat"
    except Exception as e:
        return False, f"Error - Problème de contraste : {str(e)}"


def check_lighting(image_path, threshold=0.5):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mean, stddev = cv2.meanStdDev(image)
        if stddev[0][0] < threshold * mean[0][0]:
            return False, f"Error - Éclairage non uniforme, écart type : {stddev[0][0]}, seuil requis : {threshold * mean[0][0]}"
        return True, "Done - Éclairage uniforme"
    except Exception as e:
        return False, f"Error - Problème d'éclairage : {str(e)}"


def clean_image(image_path):
    try:
        image = cv2.imread(image_path)
        cleaned_image = cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 21)
        temp_path = 'cleaned_image.png'
        cv2.imwrite(temp_path, cleaned_image)
        return temp_path
    except Exception as e:
        return None, f"Error - Problème de nettoyage de l'image : {str(e)}"


def check_orientation(image_path, angle_threshold=5):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = canny(image)
        edges = (edges * 255).astype(np.uint8)  # Convert boolean array to uint8
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is not None:
            angles = [theta for _, theta in lines[:, 0]]
            avg_angle = np.mean(angles)
            if np.abs(np.degrees(avg_angle)) > angle_threshold:
                return False, f"Error - Orientation incorrecte, l'angle du photo est : {angle_threshold} par a port angle moyen : {np.degrees(avg_angle)} degrés"
        return True, "Done - Orientation correcte"
    except Exception as e:
        return False, f"Error - Problème d'orientation : {str(e)}"


def analyze_image_quality(image_path):
    report = {}

    size_status, size_msg = check_size_bytes(image_path)
    report['size'] = size_msg

    resolution_status, resolution_msg = check_resolution(image_path)
    report['resolution'] = resolution_msg

    contrast_status, contrast_msg = check_contrast(image_path)
    report['contrast'] = contrast_msg

    lighting_status, lighting_msg = check_lighting(image_path)
    report['lighting'] = lighting_msg

    orientation_status, orientation_msg = check_orientation(image_path)
    report['orientation'] = orientation_msg

    return report


def main(image_path):
    report = analyze_image_quality(image_path)

    for criterion, message in report.items():
        print(f"{criterion.capitalize()} : {message}")


# Usage
image_path = 'images/spr_cs__.JPG'
main(image_path)
