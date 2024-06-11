import cv2
import numpy as np
from tkinter import Tk, Scale, HORIZONTAL, Button, Label, colorchooser, Toplevel, messagebox, Frame
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk

def apply_magic_pro_filter(image, settings):
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

    """-----------thicken the font-----------"""

    # Apply additional filter to thicken the font further
    gray_final = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    binary_final = cv2.adaptiveThreshold(gray_final, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                         settings['adaptive_thresh_block_size_2'], settings['adaptive_thresh_C_2'])
    kernel_final = np.ones(settings['dilate_kernel_size_2'], np.uint8)
    thickened_final = cv2.dilate(binary_final, kernel_final, iterations=settings['dilate_iterations_2'])

    final_image_colored_thickened = cv2.bitwise_and(final_image, final_image, mask=thickened_final)

    """------------black_point----------"""
    # Convert to LAB color space
    lab = cv2.cvtColor(final_image_colored_thickened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply Black Point adjustment
    black_point = settings['black_point']
    l = cv2.addWeighted(l, 1.0, l, 0, -black_point)

    lab = cv2.merge((l, a, b))
    final_image_colored_thickened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return final_image_colored_thickened

def apply_color_palette(image, palette_settings):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply hue shift
    h = (h.astype(int) + palette_settings.get('hue_shift', 0)) % 180
    h = h.astype(np.uint8)

    # Apply saturation factor
    s = cv2.multiply(s, palette_settings.get('saturation_factor', 1.0))

    # Merge the channels back and convert to BGR
    hsv_colored = cv2.merge([h, s, v])
    image = cv2.cvtColor(hsv_colored, cv2.COLOR_HSV2BGR)

    # Apply brightness adjustment
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=palette_settings.get('brightness_beta', 0))

    # Apply white intensity adjustment
    white_intensity = palette_settings.get('white_intensity', 1.0)
    white_mask = cv2.inRange(image, np.array([200, 200, 200]), np.array([255, 255, 255]))
    if white_intensity != 1.0:
        image[white_mask != 0] = cv2.convertScaleAbs(image[white_mask != 0], alpha=white_intensity, beta=0)

    # Apply black intensity adjustment
    black_intensity = palette_settings.get('black_intensity', 1.0)
    black_mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([50, 50, 50]))
    if black_intensity != 1.0:
        image[black_mask != 0] = cv2.convertScaleAbs(image[black_mask != 0], alpha=black_intensity, beta=0)

    return image

def update_image():
    settings = {
        'gaussian_blur': (blur_slider.get(), blur_slider.get()),
        'clahe_clip_limit': clahe_clip_limit_slider.get(),
        'clahe_tile_grid_size': (clahe_tile_grid_size_slider.get(), clahe_tile_grid_size_slider.get()),
        'contrast_alpha': contrast_alpha_slider.get(),
        'brightness_beta': brightness_beta_slider.get(),
        'adaptive_thresh_block_size': adaptive_thresh_block_size_slider.get(),
        'adaptive_thresh_C': adaptive_thresh_C_slider.get(),
        'dilate_kernel_size': (dilate_kernel_size_slider.get(), dilate_kernel_size_slider.get()),
        'dilate_iterations': dilate_iterations_slider.get(),
        'saturation_factor': saturation_factor_slider.get(),
        'h': h_slider.get(),
        'hForColorComponents': hForColorComponents_slider.get(),
        'templateWindowSize': templateWindowSize_slider.get(),
        'searchWindowSize': searchWindowSize_slider.get(),
        'adaptive_thresh_block_size_2': adaptive_thresh_block_size_2_slider.get(),
        'adaptive_thresh_C_2': adaptive_thresh_C_2_slider.get(),
        'dilate_kernel_size_2': (dilate_kernel_size_2_slider.get(), dilate_kernel_size_2_slider.get()),
        'dilate_iterations_2': dilate_iterations_2_slider.get(),
        'black_point': black_point_slider.get()
    }

    processed_image = apply_magic_pro_filter(original_image, settings)

    # Apply color palette if specified
    if palette_settings:
        processed_image = apply_color_palette(processed_image, palette_settings)

    img = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    image_label.imgtk = imgtk
    image_label.configure(image=imgtk)

def open_image():
    global original_image
    file_path = askopenfilename()
    original_image = cv2.imread(file_path)
    update_image()

def save_image():
    file_path = asksaveasfilename(defaultextension=".jpg")
    processed_image = apply_magic_pro_filter(original_image, {
        'gaussian_blur': (blur_slider.get(), blur_slider.get()),
        'clahe_clip_limit': clahe_clip_limit_slider.get(),
        'clahe_tile_grid_size': (clahe_tile_grid_size_slider.get(), clahe_tile_grid_size_slider.get()),
        'contrast_alpha': contrast_alpha_slider.get(),
        'brightness_beta': brightness_beta_slider.get(),
        'adaptive_thresh_block_size': adaptive_thresh_block_size_slider.get(),
        'adaptive_thresh_C': adaptive_thresh_C_slider.get(),
        'dilate_kernel_size': (dilate_kernel_size_slider.get(), dilate_kernel_size_slider.get()),
        'dilate_iterations': dilate_iterations_slider.get(),
        'saturation_factor': saturation_factor_slider.get(),
        'h': h_slider.get(),
        'hForColorComponents': hForColorComponents_slider.get(),
        'templateWindowSize': templateWindowSize_slider.get(),
        'searchWindowSize': searchWindowSize_slider.get(),
                'adaptive_thresh_block_size_2': adaptive_thresh_block_size_2_slider.get(),
        'adaptive_thresh_C_2': adaptive_thresh_C_2_slider.get(),
        'dilate_kernel_size_2': (dilate_kernel_size_2_slider.get(), dilate_kernel_size_2_slider.get()),
        'dilate_iterations_2': dilate_iterations_2_slider.get(),
        'black_point': black_point_slider.get()
    })

    # Apply color palette if specified
    if palette_settings:
        processed_image = apply_color_palette(processed_image, palette_settings)

    cv2.imwrite(file_path, processed_image)

def choose_white_color():
    global palette_settings
    color_code = colorchooser.askcolor(title="Choose White Color")
    if color_code:
        white_color = color_code[0]
        palette_settings['white_intensity'] = white_color
        white_color_button.config(bg=color_code[1])
        update_image()

def choose_black_color():
    global palette_settings
    color_code = colorchooser.askcolor(title="Choose Black Color")
    if color_code:
        black_color = color_code[0]
        palette_settings['black_intensity'] = black_color
        black_color_button.config(bg=color_code[1])
        update_image()

def open_palette_settings():
    def apply_palette_settings():
        global palette_settings
        palette_settings = {
            'hue_shift': hue_shift_slider.get(),
            'saturation_factor': saturation_factor_slider.get(),
            'brightness_beta': brightness_beta_slider.get(),
            'white_intensity': white_intensity_slider.get(),
            'black_intensity': black_intensity_slider.get()
        }
        update_image()
        palette_window.destroy()

    palette_window = Toplevel(root)
    palette_window.title("Color Palette Settings")

    hue_shift_slider = Scale(palette_window, from_=-180, to=180, orient=HORIZONTAL, label="Hue Shift")
    hue_shift_slider.set(palette_settings.get('hue_shift', 0))
    hue_shift_slider.grid(row=0, column=0, padx=5, pady=5)

    saturation_factor_slider = Scale(palette_window, from_=0.1, to=5, resolution=0.1, orient=HORIZONTAL, label="Saturation Factor")
    saturation_factor_slider.set(palette_settings.get('saturation_factor', 1.0))
    saturation_factor_slider.grid(row=0, column=1, padx=5, pady=5)

    brightness_beta_slider = Scale(palette_window, from_=-100, to=100, orient=HORIZONTAL, label="Brightness Beta")
    brightness_beta_slider.set(palette_settings.get('brightness_beta', 0))
    brightness_beta_slider.grid(row=1, column=0, padx=5, pady=5)

    white_intensity_slider = Scale(palette_window, from_=0.1, to=2, resolution=0.1, orient=HORIZONTAL, label="White Intensity")
    white_intensity_slider.set(palette_settings.get('white_intensity', 1.0))
    white_intensity_slider.grid(row=1, column=1, padx=5, pady=5)

    black_intensity_slider = Scale(palette_window, from_=0.1, to=2, resolution=0.1, orient=HORIZONTAL, label="Black Intensity")
    black_intensity_slider.set(palette_settings.get('black_intensity', 1.0))
    black_intensity_slider.grid(row=2, column=0, padx=5, pady=5)

    Button(palette_window, text="Apply", command=apply_palette_settings).grid(row=2, column=1, padx=5, pady=5)

root = Tk()
root.title("Image Filter Adjuster")

# Create sliders for each parameter
blur_slider = Scale(root, from_=1, to=20, orient=HORIZONTAL, label="Gaussian Blur")
blur_slider.set(3)
blur_slider.pack()

clahe_clip_limit_slider = Scale(root, from_=1, to=10, orient=HORIZONTAL, label="CLAHE Clip Limit")
clahe_clip_limit_slider.set(1.5)
clahe_clip_limit_slider.pack()

clahe_tile_grid_size_slider = Scale(root, from_=1, to=100, orient=HORIZONTAL, label="CLAHE Tile Grid Size")
clahe_tile_grid_size_slider.set(20)
clahe_tile_grid_size_slider.pack()

contrast_alpha_slider = Scale(root, from_=0.1, to=3, resolution=0.1, orient=HORIZONTAL, label="Contrast Alpha")
contrast_alpha_slider.set(1.1)
contrast_alpha_slider.pack()

brightness_beta_slider = Scale(root, from_=-100, to=100, orient=HORIZONTAL, label="Brightness Beta")
brightness_beta_slider.set(30)
brightness_beta_slider.pack()

adaptive_thresh_block_size_slider = Scale(root, from_=3, to=101, orient=HORIZONTAL, label="Adaptive Threshold Block Size")
adaptive_thresh_block_size_slider.set(85)
adaptive_thresh_block_size_slider.pack()

adaptive_thresh_C_slider = Scale(root, from_=-100, to=100, orient=HORIZONTAL, label="Adaptive Threshold C")
adaptive_thresh_C_slider.set(150)
adaptive_thresh_C_slider.pack()

dilate_kernel_size_slider = Scale(root, from_=1, to=10, orient=HORIZONTAL, label="Dilate Kernel Size")
dilate_kernel_size_slider.set(1)
dilate_kernel_size_slider.pack()

dilate_iterations_slider = Scale(root, from_=1, to=10, orient=HORIZONTAL, label="Dilate Iterations")
dilate_iterations_slider.set(1)
dilate_iterations_slider.pack()

saturation_factor_slider = Scale(root, from_=0.1, to=5, resolution=0.1, orient=HORIZONTAL, label="Saturation Factor")
saturation_factor_slider.set(3.0)
saturation_factor_slider.pack()

h_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Denoising Strength (h)")
h_slider.set(5)
h_slider.pack()

hForColorComponents_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Denoising Strength for Color Components (hForColorComponents)")
hForColorComponents_slider.set(0)
hForColorComponents_slider.pack()

templateWindowSize_slider = Scale(root, from_=1, to=10, orient=HORIZONTAL, label="Template Window Size")
templateWindowSize_slider.set(7)
templateWindowSize_slider.pack()

searchWindowSize_slider = Scale(root, from_=1, to=50, orient=HORIZONTAL, label="Search Window Size")
searchWindowSize_slider.set(21)
searchWindowSize_slider.pack()

adaptive_thresh_block_size_2_slider = Scale(root, from_=3, to=101, orient=HORIZONTAL, label="Adaptive Threshold Block Size 2")
adaptive_thresh_block_size_2_slider.set(95)
adaptive_thresh_block_size_2_slider.pack()

adaptive_thresh_C_2_slider = Scale(root, from_=-100, to=100, orient=HORIZONTAL, label="Adaptive Threshold C 2")
adaptive_thresh_C_2_slider.set(100)
adaptive_thresh_C_2_slider.pack()

dilate_kernel_size_2_slider = Scale(root, from_=1, to=10, orient=HORIZONTAL, label="Dilate Kernel Size 2")
dilate_kernel_size_2_slider.set(1)
dilate_kernel_size_2_slider.pack()

dilate_iterations_2_slider = Scale(root, from_=1, to=10, orient=HORIZONTAL, label="Dilate Iterations 2")
dilate_iterations_2_slider.set(1)
dilate_iterations_2_slider.pack()

black_point_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Black Point")
black_point_slider.set(10)
black_point_slider.pack()

# Add color picker buttons for white and black intensities
white_color_button = Button(root, text="Choose White Color", command=choose_white_color)
white_color_button.pack()

black_color_button = Button(root, text="Choose Black Color", command=choose_black_color)
black_color_button.pack()

# Add button to open color palette settings
Button(root, text="Open Color Palette Settings", command=open_palette_settings).pack()

# Add buttons to open and save images
Button(root, text="Open Image", command=open_image).pack()
Button(root, text="Save Image", command=save_image).pack()

# Add a label to display the image
image_label = Label(root)
image_label.pack()

# Initialize color variables
white_color = (255, 255, 255)
black_color = (0, 0, 0)

# Initialize palette settings
palette_settings = {
    'hue_shift': 0,
    'saturation_factor': 0.4,
    'brightness_beta': 0,
    'white_intensity': 1.0,
    'black_intensity': 0.1
}

root.mainloop()

