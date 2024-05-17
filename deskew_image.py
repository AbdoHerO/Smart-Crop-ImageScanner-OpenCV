import cv2
import numpy as np
import os
from scipy.spatial import distance as dist


def deskew_image(image_path, coordinates, max_angle=0):
    """
    Deskew the image based on the provided coordinates.
    """
    image = cv2.imread(image_path)
    assert (image is not None)

    # Convert coordinates to integer
    coordinates = np.array(coordinates, dtype=np.int32)

    # Calculate the angle to rotate
    # Assuming coordinates are in the order: top-left, top-right, bottom-right, bottom-left
    (tl, tr, br, bl) = coordinates
    top_edge = dist.euclidean(tl, tr)
    bottom_edge = dist.euclidean(bl, br)
    left_edge = dist.euclidean(tl, bl)
    right_edge = dist.euclidean(tr, br)

    # Find the longer horizontal edge
    if top_edge > bottom_edge:
        rotation_angle = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    else:
        rotation_angle = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))

    # Limit the rotation angle to the specified max_angle
    rotation_angle = np.clip(rotation_angle, -max_angle, max_angle)

    # Rotate the image to deskew it
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    deskewed_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    """ Fill tha part black in image with color white START """

    """ Fill tha part black in image with color white END """

    # Save the deskewed image
    OUTPUT_DIR = 'output'
    basename = os.path.basename(image_path)
    deskewed_basename = f"deskewed_{basename}"
    cv2.imwrite(os.path.join(OUTPUT_DIR, deskewed_basename), deskewed_image)
    print("Deskewed and saved " + deskewed_basename)

    return deskewed_image


coordinates = [[146.4, 0.0], [1370.3999999999999, 31.2], [1507.2, 1147.2], [4.8, 1192.8]]

deskew_image("output/cropped_model05.jpeg", coordinates)
