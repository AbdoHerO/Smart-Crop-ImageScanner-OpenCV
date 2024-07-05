# Document image orientation correction
# This approach is based on text orientation

# Assumption: Document image contains all text in same orientation

import cv2
import numpy as np
import math

debug = True


def detect_table_contour(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Find lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return 0  # No lines detected, return 0 angle

    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)

    if not horizontal_lines or not vertical_lines:
        return 0  # Not enough lines to form a table, return 0 angle

    # Find the average angle of horizontal lines
    angles = []
    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        angles.append(angle)

    avg_angle = np.mean(angles)
    angle_deg = np.degrees(avg_angle)

    # Draw the detected lines
    for line in horizontal_lines + vertical_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return angle_deg


# Display image
def display(img, frameName="OpenCV Image"):
    if not debug:
        return
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww * (h / w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
    cv2.waitKey(0)


# rotate the image with given theta value
def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def main(filePath):
    img = cv2.imread(filePath)
    original = img.copy()

    # Detect table structure and get angle
    angle = detect_table_contour(img)
    print("Table orientation angle:", angle)

    # Rotate the image
    rotated = rotate(original, angle)  # Note the negative angle here

    # Display results
    display(img, "Original with detected lines")
    display(rotated, "Rotated")


if __name__ == "__main__":
    filePath = 'images/model02.jpeg'
    main(filePath)
