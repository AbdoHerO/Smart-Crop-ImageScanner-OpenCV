import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List
import os
import random
import shutil
import tempfile
import pdf2image
from PIL import Image as imageMain


class GraphicsService:
    def openImagePil(self, imagePath: str) -> imageMain.Image:
        return imageMain.open(imagePath)

    def convertPilImageToCvImage(self, pilImage: imageMain.Image):
        return cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)

    def convertCvImagetToPilImage(self, cvImage) -> imageMain.Image:
        return imageMain.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))

    def openImageCv(self, imagePath: str):
        return self.convertPilImageToCvImage(self.openImagePil(imagePath))

    def cvToGrayScale(self, cvImage):
        return cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)

    def cvApplyGaussianBlur(self, cvImage, size: int):
        return cv2.GaussianBlur(cvImage, (size, size), 0)

    def cvExtractContours(self, cvImage):
        contours, hierarchy = cv2.findContours(cvImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours

    def paintOverBorder(self, cvImage, borderX: int, borderY: int, color: Tuple[int, int, int]):
        newImage = cvImage.copy()
        height, width, channels = newImage.shape
        for y in range(height):
            for x in range(width):
                if y <= borderY or height - borderY <= y:
                    newImage[y, x] = color
                if x <= borderX or width - borderX <= x:
                    newImage[y, x] = color
        return newImage

    def rotateImage(self, cvImage, angle: float):
        newImage = cvImage.copy()
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return newImage

    def renderPdfDocumentPageToImageFromPath(self, pdfDocPath: str, pageNumber: int, dpi: int) -> str:
        tempFolder = tempfile.gettempdir()
        pageImagePaths = pdf2image.convert_from_path(pdfDocPath, dpi=dpi, output_folder=tempFolder, fmt='png',
                                                     paths_only=True, thread_count=1, first_page=pageNumber,
                                                     last_page=pageNumber)
        return pageImagePaths[0]


def straighten_image(image):
    gs = GraphicsService()

    gray = gs.cvToGrayScale(image)
    blurred = gs.cvApplyGaussianBlur(gray, 5)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    contours = gs.cvExtractContours(edges)

    if contours:
        largest_contour = contours[0]
        min_area_rect = cv2.minAreaRect(largest_contour)
        angle = min_area_rect[-1]

        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        straightened = gs.rotateImage(image, angle)
    else:
        straightened = image

    return straightened


def show_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


image_path = 'output/cropped_model05.jpeg'  # cropped_model05.jpeg
image = cv2.imread(image_path)
assert image is not None, "Failed to load image"

# Straighten the image
straightened_image = straighten_image(image)

# Show the straightened image
show_image(straightened_image, title="Straightened Image")