import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_magic_pro_filter(image_path, output_path):
    image = cv2.imread(image_path)
    assert image is not None, "Failed to load image"

    """ Clear Magic Pro Filter  """

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)  # was (3, 3)

    # Convert to LAB color space to work with brightness
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(40, 40))  # (clipLimit=6.0, tileGridSize=(30, 30))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channels
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    """ Clear the white fog or beam effect in the image """

    # # Apply dehazing/defogging
    # hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    #
    # # Apply CLAHE to the V-channel
    # clahe_v = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    # v_clahe = clahe_v.apply(v)
    #
    # # Merge the CLAHE enhanced V-channel with the H and S channels
    # hsv_clahe = cv2.merge((h, s, v_clahe))
    #
    # # Convert back to BGR color space
    # defogged = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

    """  Increase contrast and brightness (lighting) """
    # Increase contrast and brightness
    alpha = 1.5  # was 1.7 Contrast control (1.0-3.0)
    beta = 30   # Brightness control (0-100)
    # adjusted = cv2.convertScaleAbs(defogged, alpha=alpha, beta=beta)  # was enhanced_image
    adjusted = cv2.convertScaleAbs(enhanced_image, alpha=alpha, beta=beta)  # was enhanced_image

    # Apply sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)



    """ Create Mask of the Thickened text """

    # Convert to grayscale
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 30)

    # Use morphological operations to thicken the text
    kernel = np.ones((1, 1), np.uint8)
    thickened = cv2.dilate(binary, kernel, iterations=3)

    """ Apply the Mask of the Thickened text """

    # Merge thickened text back with the original color image
    colored_thickened = cv2.bitwise_and(sharpened, sharpened, mask=thickened)

    """ Save the image transformed """

    # Save the result
    cv2.imwrite(output_path, colored_thickened)  # was sharpened
    print(f"Filtered image saved to {output_path}")

if __name__ == "__main__":
    # input_image_path = 'output/BL_scanned_1_1.jpg'  # cropped_model05.jpeg
    input_image_path = 'output/cropped_model03.jpeg'  # cropped_model05.jpeg
    output_image_path = 'output/filtered_image.jpg'
    apply_magic_pro_filter(input_image_path, output_image_path)






# def apply_magic_pro_filter(image_path, output_path):
#     image = cv2.imread(image_path)
#     assert image is not None, "Failed to load image"
#
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply Gaussian Blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # Apply adaptive thresholding
#     thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)
#
#     # Enhance edges
#     edges = cv2.Canny(thresholded, 50, 150)    # 50 , 100
#     edges = cv2.dilate(edges, None, iterations=1)
#     edges = cv2.erode(edges, None, iterations=1)
#
#     # Create a mask from the edges
#     mask = cv2.bitwise_not(edges)
#
#     # Apply the mask to the original image to get the filtered result
#     result = cv2.bitwise_and(gray, gray, mask=mask)
#
#     # Increase contrast and brightness
#     alpha = 3  # was 1.5; Contrast control (1.0-3.0)
#     beta = 20   # Brightness control (0-100)
#     adjusted = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
#
#     # Apply sharpening kernel
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     sharpened = cv2.filter2D(adjusted, -1, kernel)
#
#     # Save the result
#     cv2.imwrite(output_path, sharpened)  # was adjusted
#     print(f"Filtered image saved to {output_path}")
#
#     # Show the scanned image
#     # plt.imshow(cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB))
#     # plt.title('Scanned Image')
#     # plt.show()